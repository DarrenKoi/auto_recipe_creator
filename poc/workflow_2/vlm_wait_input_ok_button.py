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

import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_1.debug_artifacts import save_debug_json, save_debug_text, save_marked_bboxes
from poc.workflow_1.flask_vlm import UI_VENUS_MODEL_NAME
from poc.workflow_1.ocr_spotting import parse_spotting_items
from poc.workflow_1.prompts import build_spotting_prompt
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
# Spotting 태스크로 서명 텍스트의 *좌표*까지 받아, OK 버튼이 그 팝업에 속하는지 공간 검증한다.
OCR_SERVICE_SLUG = os.getenv("OK_BUTTON_OCR_SERVICE", "paddleocr-vl-1.5").strip() or "paddleocr-vl-1.5"
OCR_MAX_TOKENS = 4096

# OK 버튼 공간 게이트 — grounding 이 'OK' 라고 짚은 박스 중심이 팝업 서명 텍스트
# 영역에서 (영역 스케일 × 이 배수)보다 멀면 '다른 패널의 버튼' 으로 보고 클릭을 막는다.
# 거리는 사각형까지의 거리(영역 안이면 0). 콜드스타트값 — 실데이터로 보정.
POPUP_OK_MAX_DIST_FRAC = float(os.getenv("OK_BUTTON_MAX_DIST_FRAC", "1.5"))

# Spotting 텍스트를 팝업 서명으로 인정하는 정규화 토큰(소문자 영숫자).
_POPUP_TITLE_TOKEN = "waitinput"
_POPUP_BODY_TOKENS = ("crosscursor", "alignmentmark")


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


def _union_bbox(bboxes: list[dict]) -> dict | None:
    """여러 bbox 의 합집합(외접) 사각형."""
    if not bboxes:
        return None
    return {
        "left": min(b["left"] for b in bboxes),
        "top": min(b["top"] for b in bboxes),
        "right": max(b["right"] for b in bboxes),
        "bottom": max(b["bottom"] for b in bboxes),
    }


def _point_to_rect_distance(point: dict, rect: dict) -> float:
    """점에서 사각형까지의 최단 거리(사각형 안이면 0)."""
    dx = max(rect["left"] - point["x"], 0, point["x"] - rect["right"])
    dy = max(rect["top"] - point["y"], 0, point["y"] - rect["bottom"])
    return math.hypot(dx, dy)


def _locate_popup_region(
    *, image_b64: str, ocr_client: Workflow1VLMClient
) -> tuple[str, dict | None, str]:
    """Spotting 으로 'Wait Input' 팝업 서명 텍스트의 *위치* 까지 독립 검증한다.

    grounding VLM 의 confidence 를 신뢰하지 않고, 픽셀에서 직접 읽은 텍스트로만
    존재를 판정한다. 제목 'Wait Input' 이 보이거나, 본문의 'cross cursor' 와
    'alignment mark' 가 둘 다 읽히면(부분 가림 대비) 존재로 본다. 매칭된 텍스트
    bbox 들의 합집합을 팝업 영역으로 돌려주어, OK 버튼이 이 팝업에 속하는지
    공간 검증에 쓴다. (OCR 좌표는 grounding 과 같은 webp 입력 기준이라 동일 좌표계.)

    반환 (verdict, popup_region|None, raw_text), verdict ∈ {'present','absent','ocr_error'}.
    """
    system_message, user_text = build_spotting_prompt()
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
        return "ocr_error", None, str(exc)

    raw = response.text or ""
    items = parse_spotting_items(raw)

    has_title = False
    body_hits: set[str] = set()
    matched_boxes: list[dict] = []
    for item in items:
        norm = _normalize_ocr_text(item.get("text", ""))
        hit = False
        if _POPUP_TITLE_TOKEN in norm:
            has_title = True
            hit = True
        for token in _POPUP_BODY_TOKENS:
            if token in norm:
                body_hits.add(token)
                hit = True
        if hit and isinstance(item.get("bbox"), dict):
            matched_boxes.append(item["bbox"])

    present = has_title or (set(_POPUP_BODY_TOKENS) <= body_hits)
    if not present:
        return "absent", None, raw
    return "present", _union_bbox(matched_boxes), raw


def _gate_ok_by_region(
    *, ok_bbox: dict, popup_region: dict | None, name: str
) -> tuple[str, float | None]:
    """grounding 이 짚은 OK 박스가 팝업 텍스트 영역에 속하는지 거리로 판정한다.

    OK 버튼은 'Wait Input' 다이얼로그의 일부이므로 서명 텍스트와 가까워야 한다.
    중심점이 팝업 영역에서 (영역 스케일 × ``POPUP_OK_MAX_DIST_FRAC``)보다 멀면
    다른 패널의 버튼을 헛짚은 것으로 보고 클릭을 막는다.

    반환 (status, distance|None). status ∈ {'detected', 'ok_off_popup'}.
    """
    if popup_region is None:  # 텍스트는 확인됐으나 좌표가 없어 공간 검증 불가 → 통과(로그만).
        print(f"[WARNING] 팝업 영역 좌표 없음 → 공간 검증 생략 ({name})")
        return "detected", None

    center = bbox_center(ok_bbox)
    distance = _point_to_rect_distance(center, popup_region)
    region_w = popup_region["right"] - popup_region["left"]
    region_h = popup_region["bottom"] - popup_region["top"]
    scale = max(region_w, region_h, 1)
    limit = POPUP_OK_MAX_DIST_FRAC * scale
    if distance > limit:
        print(
            f"[INFO] OK 박스가 팝업에서 {distance:.0f}px (한계 {limit:.0f}px) → 클릭 금지 ({name})"
        )
        return "ok_off_popup", distance
    return "detected", distance


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

        # 1) 독립 Spotting 게이트로 팝업 존재 + *위치* 부터 확인한다(grounding confidence 불신).
        popup_verdict, popup_region, ocr_raw = _locate_popup_region(
            image_b64=image_b64, ocr_client=ocr_client
        )

        # status 종류:
        #   detected            팝업 텍스트 확인 + OK 버튼이 팝업 영역 근처 → 클릭 허용.
        #   no_popup            팝업 텍스트 미검출 → 클릭 금지(오검출 방지의 핵심).
        #   text_present_no_box 텍스트는 있으나 OK 버튼을 못 잡음 → 클릭 금지.
        #   ok_off_popup        OK 라고 짚었으나 팝업 영역에서 너무 멈(다른 패널) → 클릭 금지.
        #   ocr_error           OCR 호출 실패 → 불확실 → 안전하게 클릭 금지.
        #   error               grounding 호출/파싱 실패.
        payload: dict = {}
        ok_bbox: dict | None = None
        status = ""
        error_msg = ""
        ok_distance: float | None = None

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
                if ok_bbox is None:
                    status = "text_present_no_box"
                else:
                    # 2) 공간 게이트: OK 박스 중심이 팝업 텍스트 영역에 속해야 클릭 허용.
                    status, ok_distance = _gate_ok_by_region(
                        ok_bbox=ok_bbox, popup_region=popup_region, name=path.name
                    )
                    if status != "detected":
                        ok_bbox = None  # 멀리 잡힌 박스는 버려 클릭/오버레이 금지.
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
            "popup_region": popup_region or {},
            "ok_distance": ok_distance,
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
            f"ok_btn={'Y' if ok_bbox else 'N'} dist={ok_distance} "
            f"conf={payload.get('confidence')} click={click_point}"
        )

    def _count(status: str) -> int:
        return sum(1 for r in results if r["status"] == status)

    no_popup = _count("no_popup")
    text_present_no_box = _count("text_present_no_box")
    ok_off_popup = _count("ok_off_popup")
    ocr_errors = _count("ocr_error")
    grounding_errors = _count("error")
    summary = {
        "tag": tag,
        "capture_count": len(paths),
        "processed": len(results),
        "ok_button_detected": detected,
        "no_popup": no_popup,
        "text_present_no_box": text_present_no_box,
        "ok_off_popup": ok_off_popup,
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
        f"text_present_no_box={text_present_no_box} ok_off_popup={ok_off_popup} "
        f"ocr_errors={ocr_errors} grounding_errors={grounding_errors} "
        f"elapsed={format_elapsed_ms(started)}"
    )
    print(f"[INFO] out_dir={out_dir}")
    return "success" if results else "all_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
