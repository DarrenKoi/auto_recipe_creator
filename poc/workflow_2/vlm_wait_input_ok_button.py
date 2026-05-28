"""align fail 시 뜨는 'Wait Input' 팝업의 'OK' 버튼 클릭점을 VLM 이 정확히
짚을 수 있는지 확인하는 단독 테스트 스크립트.

배경: 웨이퍼 정렬 중 align fail 이 나면 tool monitor 위에 모달 팝업이 뜬다.
  - 제목(title): 'Wait Input'
  - 본문: "Click [OK] button after setting cross cursor to alignment mark."
  - 버튼: 좌하단에 'OK' 1개, 우하단에 'Retry' / 'Environment' / 'Reject' 3개.
후속 자동화(커서를 정렬 마크에 맞춘 뒤 OK 클릭)는 이 OK 버튼 좌표를 기준으로
하므로, 클릭점이 우하단 버튼들로 새지 않고 좌하단 OK 에 정확히 떨어지는지를
사람이 overlay 로 먼저 눈으로 검증한다.

존재 게이트(핵심): grounding VLM 은 'Wait Input' 팝업이 *없을 때도* 다른 패널의
엉뚱한 버튼을 confidence 0.99 로 자신 있게 짚는다. PaddleOCR-VL 로 화면 전체 텍스트를
읽어 거르려 했으나, PaddleOCR-VL 은 문서 파서라 GUI 스크린샷 전체를 주면 환각 텍스트를
대량 생성해 게이트로 쓸 수 없었다(2026-05-28 확인). 그래서 OCR 을 빼고, 화면이
정지된다는 점(align fail 시 모니터 고정)과 팝업이 고정 모달이라는 점을 이용해
**고전 CV 템플릿 매칭**으로 확인한다:
  1. VLM(ui-venus) 이 OK 버튼 영역(coarse bbox)을 짚는다.
  2. 그 영역을 미리 캡처해 둔 'OK' 글리프 템플릿과 ``cv2.matchTemplate`` 으로 대조한다.
     점수가 임계값 이상이면 진짜 OK 버튼(detected) → 클릭 허용. 낮으면 다른 패널을
     헛짚은 것(confirm_failed) → 클릭 금지.
역할 분담은 workflow_2 규칙과 동일: VLM 은 'OK 버튼 영역' 만 짚고, 클릭 허용 여부는
CV(템플릿 점수)/사람이 결정한다(production loop 에 바로 넣지 않는 프로브).

템플릿 부트스트랩: 템플릿 파일(``OK_TEMPLATE_PATH``)이 없으면 확인 불가 →
status=no_template(클릭 금지). 대신 VLM 이 짚은 OK crop 을 후보로 저장하니, 진짜
OK 버튼인 프레임의 crop 을 골라 ``OK_TEMPLATE_PATH`` 로 복사하면 그 뒤부터 게이트가 켜진다.

입력: ``ALIGN_IMAGES_ROOT/*/*/*/captured_img_from_rcs/<tag>/<tag>_rcs.jpg``
      (RCS_CAPTURE_DIR 환경변수로 임의 폴더를 직접 줄 수도 있다.)
출력: ``debug_images/vlm_wait_input_ok_button/<tag>/`` 에 overlay JPEG + per-image JSON + summary.

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

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR, WORKFLOW_2_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
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

# OK 글리프 템플릿 — VLM 이 짚은 영역이 진짜 OK 버튼인지 CV 로 확인하는 기준 이미지.
# 없으면 게이트 비활성(클릭 금지) + 후보 crop 저장으로 부트스트랩 유도.
OK_TEMPLATE_PATH = Path(
    os.getenv("OK_TEMPLATE_PATH", str(WORKFLOW_2_DIR / "templates" / "wait_input_ok.png"))
).expanduser()

# 템플릿 매칭(TM_CCOEFF_NORMED) 점수가 이 값 이상이면 OK 버튼으로 인정. 콜드스타트값 — 실데이터 보정.
OK_TEMPLATE_MIN_SCORE = float(os.getenv("OK_TEMPLATE_MIN_SCORE", "0.55"))

# VLM bbox 가 약간 어긋나도 템플릿이 미끄러질 여유를 주는 crop 패딩(bbox 크기 대비).
OK_CONFIRM_PAD_FRAC = float(os.getenv("OK_CONFIRM_PAD_FRAC", "0.6"))

# matchTemplate 은 스케일 불변이 아니라, DPI/크기 차를 흡수하려 여러 배율을 시도한다.
OK_CONFIRM_SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)

# overlay 색상 (BGR).
_VLM_COLOR = (255, 0, 255)    # magenta — VLM coarse
_OK_COLOR = (60, 200, 60)     # green — CV-confirmed
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
# 'Wait Input' 팝업 OK 버튼 grounding (VLM 은 영역만 제안).
# ------------------------------------------------------------------


def _ok_button_system_prompt() -> str:
    """OK 버튼 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a Windows CD-SEM Tool application. "
        "Return strict JSON only. "
        "A small modal dialog titled 'Wait Input' is shown on top of the tool screen "
        "during wafer alignment. Its body text reads roughly: "
        "\"Click [OK] button after setting cross cursor to alignment mark.\" "
        "The dialog has buttons along its bottom edge:\n"
        "  - a single 'OK' button at the BOTTOM-LEFT corner of the dialog;\n"
        "  - a group of 'Retry', 'Environment', 'Reject' buttons at the BOTTOM-RIGHT.\n"
        "Locate ONLY the 'OK' button (the bottom-left one). "
        "Do NOT return 'Retry', 'Environment', or 'Reject'. "
        "Use the dialog title 'Wait Input' and the bottom-LEFT position as your anchors. "
        "If the 'Wait Input' dialog or its OK button is not clearly visible, say so "
        "rather than guessing on some other button."
    )


def _ok_button_user_prompt() -> str:
    """OK 버튼 탐지 사용자 프롬프트(0-1000 bbox)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "popup_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "ok_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "button_text": "OK",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string explaining how you identified the OK button"\n'
        "}\n"
        "ok_bbox must tightly enclose the clickable rectangle of the 'OK' button at the "
        "bottom-left of the 'Wait Input' dialog. "
        "button_text is the text you actually read on that button (expected 'OK'). "
        "If the 'Wait Input' popup / OK button is not clearly visible, set "
        "popup_visible=false, ok_bbox=null."
    )


def _detect_ok_button(
    *, image_b64: str, width: int, height: int, client: Workflow1VLMClient
) -> tuple[dict, dict | None]:
    """'Wait Input' 팝업 OK 버튼 bbox 를 탐지한다. 반환 (payload, bbox_px|None)."""
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=_ok_button_system_prompt(),
        user_text=_ok_button_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("popup_visible") is not True:
        return parsed, None
    bbox_1000 = normalize_bbox_1000(parsed.get("ok_bbox"))
    if bbox_1000 is None:
        return parsed, None
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height)


# ------------------------------------------------------------------
# CV 확인 — VLM 영역이 진짜 'OK' 글리프인지 템플릿 매칭으로 판정.
# ------------------------------------------------------------------


def _load_ok_template() -> np.ndarray | None:
    """OK 글리프 템플릿(grayscale)을 로드한다. 없으면 None."""
    if not OK_TEMPLATE_PATH.is_file():
        return None
    tmpl = cv2.imread(str(OK_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if tmpl is None or tmpl.size == 0:
        print(f"[WARNING] OK 템플릿을 디코드하지 못했습니다: {OK_TEMPLATE_PATH}")
        return None
    return tmpl


def _crop_padded(bbox: dict, w: int, h: int) -> dict:
    """bbox 둘레를 ``OK_CONFIRM_PAD_FRAC`` 만큼 넓힌 crop 사각형(이미지 경계 clamp)."""
    bw = bbox["right"] - bbox["left"]
    bh = bbox["bottom"] - bbox["top"]
    pad_x = int(round(bw * OK_CONFIRM_PAD_FRAC))
    pad_y = int(round(bh * OK_CONFIRM_PAD_FRAC))
    return {
        "left": max(0, bbox["left"] - pad_x),
        "top": max(0, bbox["top"] - pad_y),
        "right": min(w, bbox["right"] + pad_x),
        "bottom": min(h, bbox["bottom"] + pad_y),
    }


def _confirm_ok_glyph(
    *, image_bgr: np.ndarray, ok_bbox: dict, template_gray: np.ndarray
) -> tuple[float, dict | None]:
    """VLM 이 짚은 OK 영역을 OK 글리프 템플릿과 매칭해 (최고점수, 매칭 bbox) 를 돌려준다.

    팝업이 없을 때 VLM 이 다른 패널 버튼을 OK 라고 헛짚어도, 그 영역에는 'OK'
    글리프가 없어 점수가 낮게 나와 걸러진다. matchTemplate 은 스케일 불변이
    아니므로 여러 배율을 시도해 최대 점수를 쓴다(DPI/크기 차 흡수).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    crop_box = _crop_padded(ok_bbox, w, h)
    crop = gray[crop_box["top"]:crop_box["bottom"], crop_box["left"]:crop_box["right"]]
    if crop.size == 0:
        return 0.0, None

    th0, tw0 = template_gray.shape[:2]
    best_score = -1.0
    best_box: dict | None = None
    for scale in OK_CONFIRM_SCALES:
        tw = max(1, int(round(tw0 * scale)))
        th = max(1, int(round(th0 * scale)))
        if th > crop.shape[0] or tw > crop.shape[1]:
            continue
        tmpl = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            mx, my = max_loc
            best_box = {
                "left": crop_box["left"] + mx,
                "top": crop_box["top"] + my,
                "right": crop_box["left"] + mx + tw,
                "bottom": crop_box["top"] + my + th,
            }
    return best_score, best_box


def _save_ok_crop(image_bgr: np.ndarray, ok_bbox: dict, out_path: Path) -> None:
    """VLM 이 짚은 OK 영역 crop 을 저장한다(템플릿 부트스트랩 후보용)."""
    h, w = image_bgr.shape[:2]
    left = max(0, ok_bbox["left"])
    top = max(0, ok_bbox["top"])
    right = min(w, ok_bbox["right"])
    bottom = min(h, ok_bbox["bottom"])
    if right <= left or bottom <= top:
        return
    cv2.imwrite(str(out_path), image_bgr[top:bottom, left:right])


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
    vlm_bbox: dict | None,
    matched_box: dict | None,
    status: str,
    score: float | None,
    output_path: Path,
) -> str:
    """VLM coarse 영역 + CV 확인 결과를 한 이미지에 마킹한다."""
    out = image_bgr.copy()
    if vlm_bbox is not None:
        _draw_rect(out, vlm_bbox, _VLM_COLOR, "VLM coarse")
    if matched_box is not None:
        color = _OK_COLOR if status == "detected" else _REJECT_COLOR
        _draw_rect(out, matched_box, color, f"{status} {score:.2f}" if score is not None else status)
        cx = (matched_box["left"] + matched_box["right"]) // 2
        cy = (matched_box["top"] + matched_box["bottom"]) // 2
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
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
    candidates_dir = out_dir / "ok_template_candidates"
    for d in (overlays_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE, model_name=DEFAULT_MODEL, log_name=LOG_NAME
    )
    template_gray = _load_ok_template()
    if template_gray is None:
        candidates_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[WARNING] OK 템플릿 없음({OK_TEMPLATE_PATH}) → CV 확인 비활성(클릭 금지). "
            f"진짜 OK 버튼 프레임의 후보 crop 을 {candidates_dir} 에서 골라 위 경로로 복사하세요."
        )
    print(
        f"[INFO] OK 버튼 클릭점 프로브 시작: grounding={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"cv_confirm={'on' if template_gray is not None else 'off(no template)'}, {len(paths)} 장"
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
        #   detected       VLM 영역이 OK 글리프 템플릿과 매칭(점수 ≥ 임계) → 클릭 허용.
        #   no_popup       VLM 이 팝업/OK 미검출(popup_visible=false 또는 bbox 없음) → 클릭 금지.
        #   confirm_failed VLM 은 OK 라고 짚었으나 템플릿 매칭 실패 → 다른 패널 헛짚음 → 클릭 금지.
        #   no_template    OK 템플릿 파일이 없어 확인 불가 → 클릭 금지(부트스트랩 crop 저장).
        #   error          grounding 호출/파싱 실패.
        payload: dict = {}
        vlm_bbox: dict | None = None
        matched_box: dict | None = None
        status = ""
        error_msg = ""
        score: float | None = None

        try:
            payload, vlm_bbox = _detect_ok_button(
                image_b64=image_b64, width=w, height=h, client=client
            )
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print(f"[ERROR] grounding 호출 실패: {path.name}, error={exc}")

        if not error_msg:
            if vlm_bbox is None:
                status = "no_popup"
                print(f"[INFO] VLM 팝업/OK 미검출 → 클릭 금지 ({path.name})")
            elif template_gray is None:
                status = "no_template"
                _save_ok_crop(image_bgr, vlm_bbox, candidates_dir / f"{path.stem}_okcrop.png")
            else:
                score, matched_box = _confirm_ok_glyph(
                    image_bgr=image_bgr, ok_bbox=vlm_bbox, template_gray=template_gray
                )
                status = "detected" if score >= OK_TEMPLATE_MIN_SCORE else "confirm_failed"
                if status == "confirm_failed":
                    print(
                        f"[INFO] OK 글리프 매칭 점수 {score:.2f} < {OK_TEMPLATE_MIN_SCORE} "
                        f"→ 클릭 금지 ({path.name})"
                    )

        click_point = None
        overlay_path = ""
        if status == "detected" and matched_box is not None:
            detected += 1
            click_point = bbox_center(matched_box)
        if vlm_bbox is not None:
            try:
                overlay_path = _save_overlay(
                    image_bgr=image_bgr,
                    vlm_bbox=vlm_bbox,
                    matched_box=matched_box,
                    status=status,
                    score=score,
                    output_path=overlays_dir / f"{path.stem}_ok.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패: {path.name}, error={exc}")

        result = {
            "image_path": str(path),
            "status": status,
            "error": error_msg,
            "overlay_path": overlay_path,
            "vlm_bbox": vlm_bbox or {},
            "matched_box": matched_box or {},
            "match_score": score,
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
            f"vlm_box={'Y' if vlm_bbox else 'N'} score={score} "
            f"vlm_conf={payload.get('confidence')} click={click_point}"
        )

    def _count(status: str) -> int:
        return sum(1 for r in results if r["status"] == status)

    no_popup = _count("no_popup")
    confirm_failed = _count("confirm_failed")
    no_template = _count("no_template")
    grounding_errors = _count("error")
    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(results),
        "ok_button_detected": detected,
        "no_popup": no_popup,
        "confirm_failed": confirm_failed,
        "no_template": no_template,
        "grounding_errors": grounding_errors,
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "ok_template_path": str(OK_TEMPLATE_PATH),
        "ok_template_present": template_gray is not None,
        "ok_template_min_score": OK_TEMPLATE_MIN_SCORE,
        "elapsed": format_elapsed_ms(started),
        "output_dir": str(out_dir),
        "note": "throwaway click-point probe; VLM region + CV OK-glyph template confirm",
    }
    save_debug_json(out_dir / "summary.json", summary)
    save_debug_text(
        out_dir / "timeline.txt",
        "\n".join(
            f"{Path(r['image_path']).name:<40} "
            f"status={r['status']:<14} "
            f"vlm_box={'Y' if r['vlm_bbox'] else 'N'} "
            f"score={r['match_score']} click={r['click_point']}"
            for r in results
        )
        + "\n",
    )

    print(
        f"[INFO] 완료: processed={len(results)}/{len(paths)} "
        f"ok_button_detected={detected} no_popup={no_popup} "
        f"confirm_failed={confirm_failed} no_template={no_template} "
        f"grounding_errors={grounding_errors} elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if results else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
