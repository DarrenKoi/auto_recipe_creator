"""align fail 시 뜨는 'Wait Input' 팝업의 'OK' 버튼 클릭점을 VLM 이 정확히
짚을 수 있는지 확인하는 단독 테스트 스크립트.

배경: 웨이퍼 정렬 중 align fail 이 나면 tool monitor 위에 모달 팝업이 뜬다.
  - 제목(title): 'Wait Input'
  - 본문: "Click [OK] button after setting cross cursor to alignment mark."
  - 버튼: 좌하단에 'OK' 1개, 우하단에 'Retry' / 'Environment' / 'Reject' 3개.
후속 자동화(커서를 정렬 마크에 맞춘 뒤 OK 클릭)는 이 OK 버튼 좌표를 기준으로
하므로, 클릭점이 우하단 버튼들로 새지 않고 좌하단 OK 에 정확히 떨어지는지를
사람이 overlay 로 먼저 눈으로 검증한다.

확인 게이트(핵심): grounding VLM 은 'Wait Input' 팝업이 *없을 때도* 다른 패널의
엉뚱한 버튼을 confidence 0.99 로 자신 있게 짚는다. 그래서 VLM 좌표만 믿지 않고
독립적으로 팝업 존재를 확인한다. 방법 변천:
  - PaddleOCR-VL 로 화면 전체 텍스트를 읽어 거르려 했으나, 문서 파서라 GUI
    스크린샷 전체를 주면 환각 텍스트를 대량 생성해 실패(2026-05-28 확인).
  - 고정 모달이라 템플릿 매칭도 시도했으나, 장비 모델마다 팝업/버튼 모양이 달라
    한 장의 픽셀 템플릿으로는 일반화가 안 됨.
  - 최종: **dialog crop-then-OCR**. VLM 이 (1) 다이얼로그 전체 영역과 (2) OK 버튼
    영역을 짚게 한다. 다이얼로그 *crop* 만 OCR 에 넣어(=PaddleOCR-VL 의 안정적인
    단일 요소 인식 모드, 전체 화면 금지) 본문의 고유 문구 'alignment mark'(또는
    'cross cursor' / 'Wait Input')가 읽히는지 본다. 읽히면 팝업 존재 확정 →
    OK 클릭 좌표는 VLM 의 OK bbox 중심을 쓴다. 안 읽히면 다른 화면을 헛짚은
    것이므로 클릭 금지. 짧은 'OK' 두 글자보다 길고 고유한 본문 문구가 OCR 로
    훨씬 신뢰성 높게 잡히고, 글자를 읽으므로 모델별 모양 차이에 둔감하다.
역할 분담은 workflow_2 규칙과 동일: VLM 은 영역만 짚고, 존재 판정은 OCR(읽은
글자)/사람이 결정한다(production loop 에 바로 넣지 않는 프로브).

입력: ``ALIGN_IMAGES_ROOT/*/*/*/captured_img_from_rcs/<tag>/<tag>_rcs.jpg``
      (RCS_CAPTURE_DIR 환경변수로 임의 폴더를 직접 줄 수도 있다.)
출력: ``debug_images/vlm_wait_input_ok_button/<tag>/`` 에 overlay JPEG + dialog crop + per-image JSON + summary.

실행:
    uv run python poc/workflow_2/vlm_wait_input_ok_button.py
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_1.prompts import build_ocr_assist_prompt
from poc.workflow_1.util import env_int, format_elapsed_ms, make_timestamp_tag
from poc.workflow_1.util.image_utils import encode_image_webp
from poc.workflow_1.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)
from poc.workflow_1.vlm_client import Workflow1VLMClient

load_dotenv()

LOG_NAME = "vlm_wait_input_ok_button"
CAPTURED_RCS_DIRNAME = "captured_img_from_rcs"

# ====================================================================
# 모듈 설정 — CLAUDE.md 규칙상 argparse 미사용, 상수/환경변수로만 조정.
# ====================================================================

# 임의 캡처 폴더 직접 지정(재귀적으로 *_rcs.jpg 수집). 비우면 ALIGN_IMAGES_ROOT 자동 탐색.
RCS_CAPTURE_DIR_OVERRIDE = os.getenv("RCS_CAPTURE_DIR", "").strip()

# 처리할 최대 캡처 장수(VLM 호출 비용 상한). 0 이하면 전체 처리. mtime 최신순으로 자른다.
SAMPLE_LIMIT = env_int("OK_BUTTON_SAMPLE_LIMIT", 0)

DEFAULT_SERVICE = os.getenv("TEST_VLM_SERVICE", "ui-venus").strip() or "ui-venus"
DEFAULT_MODEL = os.getenv("TEST_VLM_MODEL_NAME", UI_VENUS_MODEL_NAME).strip() or UI_VENUS_MODEL_NAME

# 팝업 확인용 OCR — VLM 이 짚은 *다이얼로그 crop* 에만 적용(전체 스크린샷 금지: 환각 유발).
OCR_SERVICE_SLUG = os.getenv("OK_BUTTON_OCR_SERVICE", "paddleocr-vl-1.5").strip() or "paddleocr-vl-1.5"
# 다이얼로그 한 박스의 짧은 텍스트만 읽으면 되므로 토큰을 묶어 폭주(반복 환각)를 막는다.
OCR_MAX_TOKENS = env_int("OK_BUTTON_OCR_MAX_TOKENS", 256)

# 다이얼로그 crop 패딩(bbox 크기 대비) — VLM bbox 가 약간 어긋나도 본문 글자가 잘리지 않게.
DIALOG_CROP_PAD_FRAC = float(os.getenv("OK_DIALOG_CROP_PAD_FRAC", "0.06"))
# crop 업스케일 배수 — 본문 글자가 작을 수 있어 인식기에 픽셀을 더 준다.
DIALOG_CROP_UPSCALE = float(os.getenv("OK_DIALOG_CROP_UPSCALE", "1.6"))

# 다이얼로그 crop 에서 이 중 하나가 읽히면 'Wait Input' 팝업 존재로 인정(정규화: 소문자 영숫자).
# 'alignment mark' 가 화면에서 가장 고유한 서명. 부분 가림 대비로 다른 서명도 함께 본다.
_POPUP_SIGNATURE_TOKENS = ("alignmentmark", "crosscursor", "waitinput")

# overlay 색상 (BGR).
_DIALOG_COLOR = (255, 255, 0)  # cyan — VLM dialog
_VLM_COLOR = (255, 0, 255)     # magenta — VLM OK coarse
_OK_COLOR = (60, 200, 60)      # green — confirmed
_REJECT_COLOR = (60, 60, 230)  # red — confirm 실패


# ------------------------------------------------------------------
# 입력 해석.
# ------------------------------------------------------------------


def _resolve_capture_paths() -> list[Path]:
    """처리할 *_rcs.jpg 캡처 경로들을 mtime 최신순으로 모은다."""
    if RCS_CAPTURE_DIR_OVERRIDE:
        root = Path(RCS_CAPTURE_DIR_OVERRIDE).expanduser()
        if not root.is_dir():
            print(f"[ERROR] RCS_CAPTURE_DIR 가 폴더가 아닙니다: {root}")
            return []
        paths = sorted(root.rglob("*_rcs.jpg"))
        if not paths:  # 네이밍이 다른 경우 일반 jpg 로 폴백.
            paths = sorted(root.rglob("*.jpg"))
        print(f"[INFO] RCS_CAPTURE_DIR 사용: {root} ({len(paths)} 장)")
    else:
        if not ALIGN_IMAGES_ROOT.is_dir():
            print(f"[ERROR] ALIGN_IMAGES_ROOT 가 없습니다: {ALIGN_IMAGES_ROOT}")
            return []
        pattern = f"*/*/*/{CAPTURED_RCS_DIRNAME}/*/*_rcs.jpg"
        paths = list(ALIGN_IMAGES_ROOT.glob(pattern))
        print(f"[INFO] ALIGN_IMAGES_ROOT 자동 탐색: {len(paths)} 장 발견")

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if SAMPLE_LIMIT > 0 and len(paths) > SAMPLE_LIMIT:
        print(f"[INFO] 최신 {SAMPLE_LIMIT} 장으로 제한 (전체 {len(paths)})")
        paths = paths[:SAMPLE_LIMIT]
    return paths


# ------------------------------------------------------------------
# 'Wait Input' 팝업 grounding (VLM 은 다이얼로그/OK 영역만 제안).
# ------------------------------------------------------------------


def _ok_button_system_prompt() -> str:
    """다이얼로그 + OK 버튼 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a Windows CD-SEM Tool application. "
        "Return strict JSON only. "
        "A small modal dialog titled 'Wait Input' may be shown on top of the tool screen "
        "during wafer alignment. Its body text reads roughly: "
        "\"Click [OK] button after setting cross cursor to alignment mark.\" "
        "The dialog has buttons along its bottom edge:\n"
        "  - a single 'OK' button at the BOTTOM-LEFT corner of the dialog;\n"
        "  - a group of 'Retry', 'Environment', 'Reject' buttons at the BOTTOM-RIGHT.\n"
        "All four buttons are INSIDE the dialog, so ok_bbox must lie within dialog_bbox.\n"
        "Report TWO regions:\n"
        "  - dialog_bbox: the ENTIRE 'Wait Input' dialog (title bar, body text, and all buttons);\n"
        "  - ok_bbox: ONLY the 'OK' button (the bottom-left one), tightly.\n"
        "Do NOT return 'Retry', 'Environment', or 'Reject' as the OK button. "
        "Use the dialog title 'Wait Input' and the bottom-LEFT position as your anchors. "
        "If the 'Wait Input' dialog is not clearly visible, say so rather than guessing "
        "on some other panel."
    )


def _ok_button_user_prompt() -> str:
    """다이얼로그 + OK 버튼 탐지 사용자 프롬프트(0-1000 bbox)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "popup_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "dialog_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "ok_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "button_text": "OK",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string explaining how you identified the dialog and OK button"\n'
        "}\n"
        "dialog_bbox must enclose the WHOLE 'Wait Input' dialog (title + body text + all buttons). "
        "ok_bbox must tightly enclose the clickable rectangle of the bottom-left 'OK' button. "
        "If the 'Wait Input' dialog is not clearly visible, set popup_visible=false, "
        "dialog_bbox=null, ok_bbox=null."
    )


def _detect_dialog_and_ok(
    *, image_b64: str, width: int, height: int, client: Workflow1VLMClient
) -> tuple[dict, dict | None, dict | None]:
    """'Wait Input' 다이얼로그/OK 버튼 bbox 를 탐지한다. 반환 (payload, dialog_px|None, ok_px|None)."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_ok_button_system_prompt(),
        user_text=_ok_button_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("popup_visible") is not True:
        return parsed, None, None
    dialog_1000 = normalize_bbox_1000(parsed.get("dialog_bbox"))
    ok_1000 = normalize_bbox_1000(parsed.get("ok_bbox"))
    dialog_px = bbox_1000_to_pixels(dialog_1000, width, height) if dialog_1000 else None
    ok_px = bbox_1000_to_pixels(ok_1000, width, height) if ok_1000 else None
    return parsed, dialog_px, ok_px


# ------------------------------------------------------------------
# OCR 확인 — 다이얼로그 crop 의 고유 본문 문구를 읽어 팝업 존재를 확정.
# ------------------------------------------------------------------


def _normalize_ocr_text(text: str) -> str:
    """OCR 텍스트를 소문자 영숫자만 남겨 붙인다(공백/기호 차이를 흡수)."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _center_inside(inner: dict, outer: dict, pad_frac: float = 0.05) -> bool:
    """inner bbox 중심이 outer bbox(약간 패딩) 안에 있는지.

    OK 버튼은 다이얼로그 안에 있으므로 ok_bbox 중심은 dialog_bbox 안이어야 한다.
    같은 VLM 호출에서 나온 두 박스의 내부 일관성 검사(다이얼로그는 맞다면서 OK 를
    엉뚱한 패널에 찍는 경우를 잡는다). bbox 오차를 감안해 살짝 패딩한다.
    """
    cx = (inner["left"] + inner["right"]) / 2
    cy = (inner["top"] + inner["bottom"]) / 2
    pad_x = (outer["right"] - outer["left"]) * pad_frac
    pad_y = (outer["bottom"] - outer["top"]) * pad_frac
    return (
        outer["left"] - pad_x <= cx <= outer["right"] + pad_x
        and outer["top"] - pad_y <= cy <= outer["bottom"] + pad_y
    )


def _crop_for_ocr(image_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
    """VLM bbox 둘레를 약간 넓혀 자르고 업스케일한 OCR 입력 crop(BGR)."""
    h, w = image_bgr.shape[:2]
    bw = bbox["right"] - bbox["left"]
    bh = bbox["bottom"] - bbox["top"]
    pad_x = int(round(bw * DIALOG_CROP_PAD_FRAC))
    pad_y = int(round(bh * DIALOG_CROP_PAD_FRAC))
    left = max(0, bbox["left"] - pad_x)
    top = max(0, bbox["top"] - pad_y)
    right = min(w, bbox["right"] + pad_x)
    bottom = min(h, bbox["bottom"] + pad_y)
    if right <= left or bottom <= top:
        return None
    crop = image_bgr[top:bottom, left:right]
    if DIALOG_CROP_UPSCALE > 1.0:
        crop = cv2.resize(
            crop, None, fx=DIALOG_CROP_UPSCALE, fy=DIALOG_CROP_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def _confirm_popup_by_ocr(
    *, crop_bgr: np.ndarray, ocr_client: Workflow1VLMClient
) -> tuple[str, str]:
    """다이얼로그 crop 의 글자를 OCR 로 읽어 'Wait Input' 서명 문구 존재를 판정한다.

    반환 (verdict, raw_text). verdict ∈ {'present', 'absent', 'ocr_error'}.
    본문의 고유 문구 'alignment mark'(또는 'cross cursor' / 'Wait Input')가 읽히면 present.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image_b64, _, _ = encode_image_webp(Image.fromarray(rgb), quality=90)
    system_message, user_text = build_ocr_assist_prompt(width=0, height=0)
    try:
        response = ocr_client.chat_with_image_b64(
            image_b64=image_b64,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[ERROR] OCR 확인 호출 실패: {exc}")
        return "ocr_error", str(exc)

    raw = response.text or ""
    norm = _normalize_ocr_text(raw)
    present = any(token in norm for token in _POPUP_SIGNATURE_TOKENS)
    return ("present" if present else "absent"), raw


# ------------------------------------------------------------------
# overlay 그리기.
# ------------------------------------------------------------------


def _draw_rect(img: np.ndarray, bbox: dict, color: tuple, label: str) -> None:
    p1 = (int(bbox["left"]), int(bbox["top"]))
    p2 = (int(bbox["right"]), int(bbox["bottom"]))
    cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
    cv2.putText(
        img, label, (p1[0] + 4, max(16, p1[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
    )


def _save_overlay(
    *,
    image_bgr: np.ndarray,
    dialog_bbox: dict | None,
    ok_bbox: dict | None,
    status: str,
    ocr_text: str,
    output_path: Path,
) -> str:
    """VLM dialog/OK 영역 + OCR 확인 결과(상태/읽은 글자)를 한 이미지에 마킹한다."""
    out = image_bgr.copy()
    banner_color = _OK_COLOR if status == "detected" else _REJECT_COLOR
    if dialog_bbox is not None:
        _draw_rect(out, dialog_bbox, _DIALOG_COLOR, "VLM dialog")
    if ok_bbox is not None:
        ok_color = _OK_COLOR if status == "detected" else _VLM_COLOR
        _draw_rect(out, ok_bbox, ok_color, "OK")
        if status == "detected":
            cx = (ok_bbox["left"] + ok_bbox["right"]) // 2
            cy = (ok_bbox["top"] + ok_bbox["bottom"]) // 2
            cv2.drawMarker(out, (cx, cy), _OK_COLOR, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    snippet = (ocr_text or "").strip().replace("\n", " ")[:48]
    cv2.putText(
        out, f"{status}  ocr='{snippet}'", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2, cv2.LINE_AA
    )
    cv2.imwrite(str(output_path), out)
    return str(output_path)


def run() -> str:
    started = time.time()
    paths = _resolve_capture_paths()
    if not paths:
        print("[ERROR] 처리할 캡처 이미지가 없습니다.")
        return "no_captures"

    tag = make_timestamp_tag()
    out_dir = DEBUG_IMAGE_DIR / LOG_NAME / tag
    overlays_dir = out_dir / "overlays"
    results_dir = out_dir / "results"
    crops_dir = out_dir / "dialog_crops"
    for d in (overlays_dir, results_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE, model_name=DEFAULT_MODEL, log_name=LOG_NAME
    )
    ocr_client = Workflow1VLMClient(service_slug=OCR_SERVICE_SLUG, log_name=LOG_NAME)
    print(
        f"[INFO] OK 버튼 클릭점 프로브 시작: grounding={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"ocr_confirm={OCR_SERVICE_SLUG}, {len(paths)} 장"
    )

    results: list[dict] = []
    detected = 0
    for idx, path in enumerate(paths):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[ERROR] 이미지 디코드 실패: {path.name}")
            continue
        try:
            with Image.open(path) as image:
                image_b64, w, h = encode_image_webp(image.convert("RGB"), quality=90)
        except Exception as exc:
            print(f"[ERROR] WebP 인코딩 실패: {path.name}, error={exc}")
            continue

        # status 종류:
        #   detected       다이얼로그 crop 에서 서명 문구 확인 + OK bbox 있음 → 클릭 허용.
        #   no_popup       VLM 이 다이얼로그 미검출(popup_visible=false 또는 dialog_bbox 없음) → 클릭 금지.
        #   confirm_failed 다이얼로그라 짚었으나 crop 에 서명 문구가 없음 → 다른 화면 헛짚음 → 클릭 금지.
        #   popup_no_ok    팝업은 확인됐으나 VLM 이 OK bbox 를 못 줌 → 클릭 좌표 없음 → 클릭 금지.
        #   ok_outside_dialog 팝업 확인됐으나 OK bbox 가 다이얼로그 밖 → 좌표 모순 → 클릭 금지.
        #   ocr_error      OCR 확인 호출 실패 → 불확실 → 안전하게 클릭 금지.
        #   error          grounding 호출/파싱 실패.
        payload: dict = {}
        dialog_bbox: dict | None = None
        ok_bbox: dict | None = None
        status = ""
        error_msg = ""
        ocr_text = ""

        try:
            payload, dialog_bbox, ok_bbox = _detect_dialog_and_ok(
                image_b64=image_b64, width=w, height=h, client=client
            )
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print(f"[ERROR] grounding 호출 실패: {path.name}, error={exc}")

        if not error_msg:
            if dialog_bbox is None:
                status = "no_popup"
                print(f"[INFO] VLM 다이얼로그 미검출 → 클릭 금지 ({path.name})")
            else:
                crop = _crop_for_ocr(image_bgr, dialog_bbox)
                if crop is None:
                    status = "confirm_failed"
                else:
                    cv2.imwrite(str(crops_dir / f"{path.stem}_dialog.png"), crop)
                    verdict, ocr_text = _confirm_popup_by_ocr(crop_bgr=crop, ocr_client=ocr_client)
                    if verdict == "ocr_error":
                        status = "ocr_error"
                    elif verdict == "present":
                        if ok_bbox is None:
                            status = "popup_no_ok"
                        elif not _center_inside(ok_bbox, dialog_bbox):
                            status = "ok_outside_dialog"
                            print(
                                f"[INFO] OK bbox 가 다이얼로그 밖 → 좌표 모순 "
                                f"→ 클릭 금지 ({path.name})"
                            )
                        else:
                            status = "detected"
                    else:
                        status = "confirm_failed"
                        print(
                            f"[INFO] 다이얼로그 crop 에 서명 문구 없음 "
                            f"(읽은 글자='{ocr_text.strip()[:30]}') → 클릭 금지 ({path.name})"
                        )

        click_point = None
        overlay_path = ""
        if status == "detected" and ok_bbox is not None:
            detected += 1
            click_point = bbox_center(ok_bbox)
        if dialog_bbox is not None or ok_bbox is not None:
            try:
                overlay_path = _save_overlay(
                    image_bgr=image_bgr,
                    dialog_bbox=dialog_bbox,
                    ok_bbox=ok_bbox,
                    status=status,
                    ocr_text=ocr_text,
                    output_path=overlays_dir / f"{path.stem}_ok.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패: {path.name}, error={exc}")

        result = {
            "image_path": str(path),
            "status": status,
            "error": error_msg,
            "overlay_path": overlay_path,
            "dialog_bbox": dialog_bbox or {},
            "ok_bbox": ok_bbox or {},
            "ocr_text": ocr_text[:300],
            "click_point": click_point or {},
            "button_text": payload.get("button_text"),
            "vlm_confidence": payload.get("confidence"),
            "evidence": payload.get("evidence"),
            "raw_payload": payload,
        }
        save_debug_json(results_dir / f"{path.stem}.json", result)
        results.append(result)
        print(
            f"[INFO] {idx:02d} {path.name} status={status} "
            f"dialog={'Y' if dialog_bbox else 'N'} ok_box={'Y' if ok_bbox else 'N'} "
            f"ocr='{ocr_text.strip()[:20]}' click={click_point}"
        )

    def _count(status: str) -> int:
        return sum(1 for r in results if r["status"] == status)

    no_popup = _count("no_popup")
    confirm_failed = _count("confirm_failed")
    popup_no_ok = _count("popup_no_ok")
    ok_outside_dialog = _count("ok_outside_dialog")
    ocr_errors = _count("ocr_error")
    grounding_errors = _count("error")
    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(results),
        "ok_button_detected": detected,
        "no_popup": no_popup,
        "confirm_failed": confirm_failed,
        "popup_no_ok": popup_no_ok,
        "ok_outside_dialog": ok_outside_dialog,
        "ocr_errors": ocr_errors,
        "grounding_errors": grounding_errors,
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "ocr_service": OCR_SERVICE_SLUG,
        "signature_tokens": list(_POPUP_SIGNATURE_TOKENS),
        "elapsed": format_elapsed_ms(started),
        "output_dir": str(out_dir),
        "note": "throwaway click-point probe; VLM dialog+OK region, dialog crop-OCR signature confirm",
    }
    save_debug_json(out_dir / "summary.json", summary)
    save_debug_text(
        out_dir / "timeline.txt",
        "\n".join(
            f"{Path(r['image_path']).name:<40} "
            f"status={r['status']:<14} "
            f"dialog={'Y' if r['dialog_bbox'] else 'N'} "
            f"ok_box={'Y' if r['ok_bbox'] else 'N'} "
            f"ocr='{(r['ocr_text'] or '').strip()[:24]:<24}' click={r['click_point']}"
            for r in results
        )
        + "\n",
    )

    print(
        f"[INFO] 완료: processed={len(results)}/{len(paths)} "
        f"ok_button_detected={detected} no_popup={no_popup} "
        f"confirm_failed={confirm_failed} popup_no_ok={popup_no_ok} "
        f"ok_outside_dialog={ok_outside_dialog} "
        f"ocr_errors={ocr_errors} grounding_errors={grounding_errors} "
        f"elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if results else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
