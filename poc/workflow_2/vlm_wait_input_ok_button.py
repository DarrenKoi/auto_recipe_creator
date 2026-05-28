"""align fail 시 뜨는 'Wait Input' 팝업의 'OK' 버튼 클릭점을 VLM 이 정확히
짚을 수 있는지 확인하는 단독 테스트 스크립트.

배경: 웨이퍼 정렬 중 align fail 이 나면 tool monitor 위에 모달 팝업이 뜬다.
  - 제목(title): 'Wait Input'
  - 본문: "Click [OK] button after setting cross cursor to alignment mark."
  - 버튼: 좌하단에 'OK' 1개, 우하단에 'Retry' / 'Environment' / 'Reject' 3개.
후속 자동화(커서를 정렬 마크에 맞춘 뒤 OK 클릭)는 이 OK 버튼 좌표를 기준으로
하므로, 클릭점이 우하단 버튼들로 새지 않고 좌하단 OK 에 정확히 떨어지는지를
사람이 overlay 로 먼저 눈으로 검증한다.

팝업 존재 게이트(핵심): grounding VLM 은 'Wait Input' 팝업이 *없을 때도* 다른 패널의
엉뚱한 버튼을 confidence 0.99 로 자신 있게 짚는다. 그래서 VLM 의 confidence/
popup_visible 값을 신뢰하지 않고, **독립적인 OCR(paddleocr)** 로 팝업 서명 텍스트
(제목 'Wait Input' 또는 본문 'cross cursor'+'alignment mark')가 화면에 실제로
읽히는지 먼저 확인한다. 이 게이트를 통과할 때만 OK 버튼을 grounding/클릭한다.
  - 완전히 없으면(텍스트 미검출) → 클릭 금지(no_popup).
  - 다른 창에 가려 일부만 보여도 설명 텍스트가 읽히면 → 통과(존재로 인정).
역할 분담은 workflow_2 규칙과 동일: VLM 은 'OK 버튼 영역' 만 짚고, 존재 판정과
좌표 신뢰 여부는 OCR/사람이 결정한다(production loop 에 바로 넣지 않는 프로브).

입력: ``ALIGN_IMAGES_ROOT/*/*/*/captured_img_from_rcs/<tag>/<tag>_rcs.jpg``
      (RCS_CAPTURE_DIR 환경변수로 임의 폴더를 직접 줄 수도 있다.)
출력: ``debug_images/vlm_wait_input_ok_button/<tag>/`` 에 overlay JPEG + per-image JSON + summary.

실행:
    uv run python poc/workflow_2/vlm_wait_input_ok_button.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
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

# 팝업 존재 게이트용 OCR — grounding VLM 과 다른 독립 모델(paddleocr)로 텍스트를 읽는다.
OCR_SERVICE_SLUG = os.getenv("OK_BUTTON_OCR_SERVICE", "paddleocr-vl-1.5").strip() or "paddleocr-vl-1.5"
OCR_MAX_TOKENS = 4096


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
# 'Wait Input' 팝업 OK 버튼 grounding.
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


def _normalize_ocr_text(text: str) -> str:
    """OCR 텍스트를 소문자 영숫자만 남겨 붙인다(공백/기호 차이를 흡수)."""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _verify_popup_present(
    *, image_b64: str, ocr_client: Workflow1VLMClient
) -> tuple[str, str]:
    """OCR 로 'Wait Input' 팝업 서명 텍스트가 실제 읽히는지 독립 검증.

    grounding VLM 의 confidence 를 신뢰하지 않고, 픽셀에서 직접 읽은 텍스트로만
    존재를 판정한다. 제목 'Wait Input' 이 보이거나, 본문의 'cross cursor' 와
    'alignment mark' 가 둘 다 읽히면(부분 가림 대비) 존재로 본다.

    반환 (verdict, raw_ocr_text), verdict ∈ {'present', 'absent', 'ocr_error'}.
    """
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
        print(f"[ERROR] OCR 게이트 호출 실패: {exc}")
        return "ocr_error", str(exc)

    raw = response.text or ""
    norm = _normalize_ocr_text(raw)
    has_title = "waitinput" in norm
    has_desc = ("crosscursor" in norm) and ("alignmentmark" in norm)
    return ("present" if (has_title or has_desc) else "absent"), raw


def _save_overlay(*, image_path: Path, ok_bbox: dict, output_path: Path) -> str:
    """원본 캡처 위에 OK 버튼 box + 클릭점(center) 을 마킹한다."""
    with Image.open(image_path) as image:
        elements = {"ok_button": {"bbox": ok_bbox, "center": bbox_center(ok_bbox)}}
        save_marked_bboxes(
            image.convert("RGB"),
            elements=elements,
            colors={"ok_button": "red"},
            out_path=output_path,
        )
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
    for d in (overlays_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = Workflow1VLMClient(
        service_slug=DEFAULT_SERVICE, model_name=DEFAULT_MODEL, log_name=LOG_NAME
    )
    ocr_client = Workflow1VLMClient(service_slug=OCR_SERVICE_SLUG, log_name=LOG_NAME)
    print(
        f"[INFO] OK 버튼 클릭점 프로브 시작: grounding={DEFAULT_SERVICE}/{DEFAULT_MODEL}, "
        f"ocr_gate={OCR_SERVICE_SLUG}, {len(paths)} 장"
    )

    results: list[dict] = []
    detected = 0
    for idx, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                image_b64, w, h = encode_image_webp(image.convert("RGB"), quality=90)
        except Exception as exc:
            print(f"[ERROR] WebP 인코딩 실패: {path.name}, error={exc}")
            continue

        # 1) 독립 OCR 게이트로 팝업 존재부터 확인한다(grounding confidence 불신).
        popup_verdict, ocr_raw = _verify_popup_present(
            image_b64=image_b64, ocr_client=ocr_client
        )

        # status 종류:
        #   detected            팝업 텍스트 확인 + OK 버튼 grounding 성공 → 클릭 허용.
        #   no_popup            팝업 텍스트 미검출 → 클릭 금지(오검출 방지의 핵심).
        #   text_present_no_box 텍스트는 있으나 OK 버튼을 못 잡음 → 클릭 금지.
        #   ocr_error           OCR 호출 실패 → 불확실 → 안전하게 클릭 금지.
        #   error               grounding 호출/파싱 실패.
        payload: dict = {}
        ok_bbox: dict | None = None
        status = ""
        error_msg = ""

        if popup_verdict == "absent":
            status = "no_popup"
            print(f"[INFO] Wait Input 텍스트 미검출 → 클릭 금지 ({path.name})")
        elif popup_verdict == "ocr_error":
            status = "ocr_error"
        else:  # present → grounding 진행.
            try:
                payload, ok_bbox = _detect_ok_button(
                    image_b64=image_b64, width=w, height=h, client=client
                )
                status = "detected" if ok_bbox is not None else "text_present_no_box"
            except Exception as exc:
                status = "error"
                error_msg = str(exc)
                print(f"[ERROR] grounding 호출 실패: {path.name}, error={exc}")

        overlay_path = ""
        click_point = None
        if ok_bbox is not None:
            detected += 1
            click_point = bbox_center(ok_bbox)
            try:
                overlay_path = _save_overlay(
                    image_path=path,
                    ok_bbox=ok_bbox,
                    output_path=overlays_dir / f"{path.stem}_ok.jpg",
                )
            except Exception as exc:
                print(f"[ERROR] overlay 저장 실패: {path.name}, error={exc}")

        result = {
            "image_path": str(path),
            "status": status,
            "popup_verdict": popup_verdict,
            "error": error_msg,
            "overlay_path": overlay_path,
            "ok_bbox": ok_bbox or {},
            "click_point": click_point or {},
            "button_text": payload.get("button_text"),
            "confidence": payload.get("confidence"),
            "evidence": payload.get("evidence"),
            "ocr_text": ocr_raw[:500],
            "raw_payload": payload,
        }
        save_debug_json(results_dir / f"{path.stem}.json", result)
        results.append(result)
        print(
            f"[INFO] {idx:02d} {path.name} popup={popup_verdict} status={status} "
            f"ok_btn={'Y' if ok_bbox else 'N'} conf={payload.get('confidence')} click={click_point}"
        )

    def _count(status: str) -> int:
        return sum(1 for r in results if r["status"] == status)

    no_popup = _count("no_popup")
    text_present_no_box = _count("text_present_no_box")
    ocr_errors = _count("ocr_error")
    grounding_errors = _count("error")
    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(results),
        "ok_button_detected": detected,
        "no_popup": no_popup,
        "text_present_no_box": text_present_no_box,
        "ocr_errors": ocr_errors,
        "grounding_errors": grounding_errors,
        "vlm_service": DEFAULT_SERVICE,
        "vlm_model_name": DEFAULT_MODEL,
        "ocr_service": OCR_SERVICE_SLUG,
        "elapsed": format_elapsed_ms(started),
        "output_dir": str(out_dir),
        "note": "throwaway click-point probe; OCR-gated popup presence + human overlay verify",
    }
    save_debug_json(out_dir / "summary.json", summary)
    save_debug_text(
        out_dir / "timeline.txt",
        "\n".join(
            f"{Path(r['image_path']).name:<40} "
            f"popup={r['popup_verdict']:<9} "
            f"status={r['status']:<18} "
            f"ok_btn={'Y' if r['ok_bbox'] else 'N'} "
            f"conf={r['confidence']} click={r['click_point']}"
            for r in results
        )
        + "\n",
    )

    print(
        f"[INFO] 완료: processed={len(results)}/{len(paths)} "
        f"ok_button_detected={detected} no_popup={no_popup} "
        f"text_present_no_box={text_present_no_box} "
        f"ocr_errors={ocr_errors} grounding_errors={grounding_errors} "
        f"elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if results else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
