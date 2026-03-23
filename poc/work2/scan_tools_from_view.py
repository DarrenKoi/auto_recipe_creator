"""RCS View 탭에서 green box 안의 Tool ID 를 OCR 로 스캔한다.

동작:
  1. 메인 RCS 창에서 `view_list_tab_rcs.py` 와 동일한 방식으로 View 탭 좌표를 찾는다.
  2. View 탭을 연 뒤 본문 영역에서 green box 후보를 탐지한다.
  3. 각 green box 를 개별 OCR 하여 Tool ID 후보를 수집한다.
  4. wheel scroll 로 아래로 내려가며 추가 Tool ID 를 dedupe 한다.

사용법:
  1. 로그인까지 완료해서 `RCS - ...` 메인 창이 떠 있는 상태로 둔다.
  2. 필요하면 `.env` 에 `SAFE_MODE=false` 또는 `SCAN_VIEW_ACTION_ENABLED=true` 를 설정한다.
  3. uv run python poc/work2/scan_tools_from_view.py
"""

import hashlib
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.work2.logger import log_work2_event
from poc.work2.prompts import build_ocr_assist_prompt
from poc.work2.ui_venus_mai_locator import EXIT_SUCCESS, analyze_window_target
from poc.work2.util import (
    activate_window,
    capture_window,
    crop_image,
    debug_image_path,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
    make_timestamp_tag,
    normalize_lines,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_bboxes,
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text
from poc.work2.view_list_tab_rcs import PREDEFINED_TARGETS
from poc.work2.vlm_client import Work2VLMClient

try:
    import cv2
    import numpy as np

    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("[WARNING] cv2/numpy 미설치 — green box 탐지는 full crop fallback 으로 동작합니다.")

try:
    from pynput.mouse import Button, Controller as MouseController

    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False
    print("[WARNING] pynput.mouse 미설치 — 클릭/스크롤 동작은 로그만 출력됩니다.")

load_dotenv()


DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images" / "scan_tools_from_view"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"

EXIT_MAIN_WINDOW_NOT_FOUND = "main_window_not_found"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_VIEW_TAB_NOT_FOUND = "view_tab_not_found"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"
EXIT_NO_TOOL_IDS_FOUND = "no_tool_ids_found"

PRE_CLICK_SETTLE_SEC = 0.2
POST_VIEW_CLICK_WAIT_SEC = 1.5
POST_SCROLL_WAIT_SEC = 1.2
OCR_MAX_TOKENS = 512


def _env_flag(name: str, default: bool = False) -> bool:
    """bool 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    """int 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


def _env_float(name: str, default: float) -> float:
    """float 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


SAFE_MODE = _env_flag("SAFE_MODE", default=True)
ACTION_ENABLED = _env_flag("SCAN_VIEW_ACTION_ENABLED", default=not SAFE_MODE)
MAX_SCROLL_STEPS = max(0, _env_int("SCAN_VIEW_MAX_SCROLL_STEPS", 12))
MAX_NO_NEW_SCROLLS = max(0, _env_int("SCAN_VIEW_MAX_NO_NEW_SCROLLS", 2))
SCROLL_DY = _env_int("SCAN_VIEW_SCROLL_DY", -8)
MAX_GREEN_BOXES = max(1, _env_int("SCAN_VIEW_MAX_GREEN_BOXES", 16))

CONTENT_REGION_LEFT_RATIO = _env_float("SCAN_VIEW_CONTENT_LEFT_RATIO", 0.08)
CONTENT_REGION_TOP_RATIO = _env_float("SCAN_VIEW_CONTENT_TOP_RATIO", 0.10)
CONTENT_REGION_RIGHT_RATIO = _env_float("SCAN_VIEW_CONTENT_RIGHT_RATIO", 0.98)
CONTENT_REGION_BOTTOM_RATIO = _env_float("SCAN_VIEW_CONTENT_BOTTOM_RATIO", 0.96)

GREEN_MIN_WIDTH = max(20, _env_int("SCAN_VIEW_GREEN_MIN_WIDTH", 90))
GREEN_MIN_HEIGHT = max(12, _env_int("SCAN_VIEW_GREEN_MIN_HEIGHT", 18))
GREEN_MAX_HEIGHT = max(GREEN_MIN_HEIGHT, _env_int("SCAN_VIEW_GREEN_MAX_HEIGHT", 120))
GREEN_MIN_AREA = max(200, _env_int("SCAN_VIEW_GREEN_MIN_AREA", 2200))
GREEN_MIN_ASPECT = _env_float("SCAN_VIEW_GREEN_MIN_ASPECT", 1.8)
GREEN_MIN_COVERAGE = _env_float("SCAN_VIEW_GREEN_MIN_COVERAGE", 0.35)

TOOL_ID_MIN_LEN = max(4, _env_int("SCAN_VIEW_TOOL_ID_MIN_LEN", 6))
TOOL_ID_MAX_LEN = max(TOOL_ID_MIN_LEN, _env_int("SCAN_VIEW_TOOL_ID_MAX_LEN", 16))
TOOL_ID_REGEX = os.getenv(
    "SCAN_VIEW_TOOL_ID_REGEX",
    r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{6,16}$",
).strip() or r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{6,16}$"

IGNORED_TOOL_ID_TOKENS = {
    "VIEW",
    "LIST",
    "RCS",
    "IMAGE",
    "TOOL",
    "TOOLS",
    "SCAN",
    "SEM",
}


def _build_relative_crop_box(
    width: int,
    height: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> dict[str, int]:
    """이미지 크기와 비율로 crop box 를 만든다."""
    left = int(round(width * min(max(left_ratio, 0.0), 1.0)))
    top = int(round(height * min(max(top_ratio, 0.0), 1.0)))
    right = int(round(width * min(max(right_ratio, 0.0), 1.0)))
    bottom = int(round(height * min(max(bottom_ratio, 0.0), 1.0)))

    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    right = min(width, right)
    bottom = min(height, bottom)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def _click_at_screen(screen_point: dict[str, int], target_key: str) -> bool:
    """스크린 좌표에서 마우스 좌클릭을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not ACTION_ENABLED or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] 클릭 생략: target={target_key}, "
            f"screen=({sx}, {sy}), action_enabled={ACTION_ENABLED}, "
            f"pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.click(Button.left, 1)
    print(f"[INFO] 클릭 완료: target={target_key}, screen=({sx}, {sy})")
    return True


def _scroll_at_screen(screen_point: dict[str, int], step_index: int) -> bool:
    """스크린 좌표에서 mouse wheel scroll 을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not ACTION_ENABLED or not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] scroll 생략: step={step_index}, "
            f"screen=({sx}, {sy}), dy={SCROLL_DY}, "
            f"action_enabled={ACTION_ENABLED}, pynput={PYNPUT_MOUSE_AVAILABLE}"
        )
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.scroll(0, SCROLL_DY)
    print(f"[INFO] scroll 완료: step={step_index}, screen=({sx}, {sy}), dy={SCROLL_DY}")
    return True


def _capture_main_window(main_window, window_title: str, backend: str):
    """메인 창을 활성화하고 한 번 캡처한다."""
    if not activate_window(
        main_window,
        debug_label=f"{LOG_NAME} activate backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 활성화 실패: title={window_title!r}")
        return None

    if not foreground_window(
        main_window,
        debug_label=f"{LOG_NAME} screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 foreground 실패: title={window_title!r}")
        return None

    try:
        return capture_window(main_window)
    except Exception as exc:
        print(f"[ERROR] 메인 창 캡처 실패: {exc}")
        return None


def _normalize_tool_text(text: str) -> str:
    """OCR 텍스트 비교를 위해 영숫자만 남기고 대문자로 정규화한다."""
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _is_likely_tool_id(candidate: str) -> bool:
    """정규화된 문자열이 Tool ID 패턴처럼 보이는지 확인한다."""
    if not candidate:
        return False
    if candidate in IGNORED_TOOL_ID_TOKENS:
        return False
    if not (TOOL_ID_MIN_LEN <= len(candidate) <= TOOL_ID_MAX_LEN):
        return False
    if not any(ch.isalpha() for ch in candidate):
        return False
    if not any(ch.isdigit() for ch in candidate):
        return False
    return re.match(TOOL_ID_REGEX, candidate) is not None


def _extract_tool_id_candidates(raw_text: str) -> list[str]:
    """OCR 원문에서 Tool ID 후보를 추출한다."""
    candidates: list[str] = []
    seen: set[str] = set()

    for line in normalize_lines(raw_text):
        parts = re.split(r"[^A-Za-z0-9]+", line)
        variants = [_normalize_tool_text(line), *(_normalize_tool_text(part) for part in parts)]
        for candidate in variants:
            if not _is_likely_tool_id(candidate) or candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def _compute_signature(tool_ids: list[str], boxes: list[dict[str, int]]) -> str:
    """현재 step 의 정적인 레이아웃 signature 를 계산한다."""
    parts: list[str] = []
    for tool_id in tool_ids:
        parts.append(tool_id)
    for box in boxes:
        parts.append(
            f"{box['left']//10}:{box['top']//10}:{box['right']//10}:{box['bottom']//10}"
        )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _sort_boxes_reading_order(boxes: list[dict[str, int]]) -> list[dict[str, int]]:
    """bbox 를 상->하, 좌->우 reading order 로 정렬한다."""
    return sorted(boxes, key=lambda box: (box["top"] // 20, box["left"]))


def _boxes_should_merge(a: dict[str, int], b: dict[str, int]) -> bool:
    """서로 인접하거나 겹치는 green box bbox 인지 판단한다."""
    horizontal_overlap = min(a["right"], b["right"]) - max(a["left"], b["left"])
    vertical_overlap = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    min_width = min(a["right"] - a["left"], b["right"] - b["left"])
    min_height = min(a["bottom"] - a["top"], b["bottom"] - b["top"])

    return (
        horizontal_overlap >= max(12, int(min_width * 0.30))
        and vertical_overlap >= -max(4, int(min_height * 0.25))
    )


def _merge_boxes(boxes: list[dict[str, int]]) -> list[dict[str, int]]:
    """겹치거나 거의 붙은 bbox 를 병합한다."""
    merged: list[dict[str, int]] = []

    for box in _sort_boxes_reading_order(boxes):
        if not merged:
            merged.append(dict(box))
            continue

        last = merged[-1]
        if not _boxes_should_merge(last, box):
            merged.append(dict(box))
            continue

        last["left"] = min(last["left"], box["left"])
        last["top"] = min(last["top"], box["top"])
        last["right"] = max(last["right"], box["right"])
        last["bottom"] = max(last["bottom"], box["bottom"])

    return merged


def _detect_green_boxes(content_image) -> list[dict[str, int]]:
    """content crop 안의 green label box 후보를 찾는다."""
    width, height = content_image.size
    if not CV_AVAILABLE:
        return [{
            "left": 0,
            "top": 0,
            "right": width,
            "bottom": height,
        }]

    rgb = np.array(content_image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([35, 35, 35], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = np.ones((5, 9), dtype=np.uint8)
    kernel_open = np.ones((3, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < GREEN_MIN_WIDTH or h < GREEN_MIN_HEIGHT or h > GREEN_MAX_HEIGHT:
            continue
        if area < GREEN_MIN_AREA:
            continue

        aspect_ratio = w / max(1, h)
        if aspect_ratio < GREEN_MIN_ASPECT:
            continue

        crop_mask = mask[y:y + h, x:x + w]
        coverage = float(cv2.countNonZero(crop_mask)) / float(max(1, w * h))
        if coverage < GREEN_MIN_COVERAGE:
            continue

        pad_x = max(4, int(round(w * 0.02)))
        pad_y = max(3, int(round(h * 0.15)))
        candidates.append(
            {
                "left": max(0, x - pad_x),
                "top": max(0, y - pad_y),
                "right": min(width, x + w + pad_x),
                "bottom": min(height, y + h + pad_y),
            }
        )

    merged = _merge_boxes(candidates)
    merged = _sort_boxes_reading_order(merged)
    if len(merged) > MAX_GREEN_BOXES:
        merged = merged[:MAX_GREEN_BOXES]
    return merged


def _save_green_box_overlay(
    content_image,
    boxes: list[dict[str, int]],
    timestamp_tag: str,
    step_index: int,
) -> Path:
    """green box bbox overlay 를 저장한다."""
    overlay_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_green_boxes_overlay.jpg",
        model_name="green_box_detector",
        timestamp_tag=timestamp_tag,
    )
    elements = {
        f"green_box_{index:02d}": {"bbox": box}
        for index, box in enumerate(boxes, start=1)
    }
    colors = {
        key: "lime"
        for key in elements
    }
    save_marked_bboxes(content_image, elements, colors, overlay_path)
    return overlay_path


def _run_box_ocr(
    client: Work2VLMClient,
    box_image,
    timestamp_tag: str,
    step_index: int,
    box_index: int,
    window_title: str,
    backend: str,
) -> dict:
    """green box crop 하나를 OCR 하고 Tool ID 후보를 추출한다."""
    system_message, user_text = build_ocr_assist_prompt(
        box_image.size[0],
        box_image.size[1],
        context_label="view_green_box",
    )

    capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_box_{box_index:02d}_crop.jpg",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    webp_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_box_{box_index:02d}_input.webp",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    raw_response_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_box_{box_index:02d}_ocr.txt",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_box_{box_index:02d}_result.json",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )

    save_debug_jpeg(box_image, capture_path, log_name=LOG_NAME)
    save_debug_webp(box_image, webp_path, quality=90, log_name=LOG_NAME)

    response = client.chat_with_image_path(
        image_path=webp_path,
        system_message=system_message,
        user_text=user_text,
        image_mime="image/webp",
        temperature=0.0,
        max_tokens=OCR_MAX_TOKENS,
    )

    raw_text = response.text.strip()
    tool_id_candidates = _extract_tool_id_candidates(raw_text)
    selected_tool_id = tool_id_candidates[0] if tool_id_candidates else ""

    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": response.service_slug,
            "model_name": response.model_name,
            "api_url": response.api_url,
            "endpoint": client.endpoint,
            "window_title": window_title,
            "backend": backend,
            "step_index": step_index,
            "box_index": box_index,
            "prompt_text": user_text,
            "raw_text": raw_text,
            "normalized_lines": normalize_lines(raw_text),
            "tool_id_candidates": tool_id_candidates,
            "selected_tool_id": selected_tool_id,
            "token_usage": response.token_usage,
        },
    )

    return {
        "raw_text": raw_text,
        "tool_id_candidates": tool_id_candidates,
        "selected_tool_id": selected_tool_id,
        "capture_path": str(capture_path),
        "webp_path": str(webp_path),
        "raw_response_path": str(raw_response_path),
        "result_json_path": str(result_json_path),
        "token_usage": response.token_usage or {},
    }


def _open_view_tab(main_window, window_title: str, backend: str) -> tuple[str, dict | None]:
    """`view_list_tab_rcs.py` 와 동일한 View 탭 타겟 설정으로 View 를 연다."""
    target = PREDEFINED_TARGETS["view_tab"]
    result = analyze_window_target(
        main_window,
        window_title,
        backend,
        target,
        debug_image_dir=DEBUG_IMAGE_DIR,
        log_name=LOG_NAME,
        component_name=COMPONENT_NAME,
        artifact_prefix="scan_tools_from_view_view_tab",
        result_mode="ui_venus_then_mai_ui_main_tabs",
    )
    if result.exit_code != EXIT_SUCCESS or result.point is None:
        return EXIT_VIEW_TAB_NOT_FOUND, None

    screen_point = image_point_to_screen(main_window, result.point)
    if screen_point is None:
        return EXIT_CAPTURE_FAILED, None

    foreground_window(
        main_window,
        debug_label="scan_tools_from_view_pre_click_view_tab",
    )
    time.sleep(PRE_CLICK_SETTLE_SEC)

    clicked = _click_at_screen(screen_point, "view_tab")
    if clicked:
        time.sleep(POST_VIEW_CLICK_WAIT_SEC)

    return EXIT_SUCCESS, {
        "image_point": result.point,
        "screen_point": screen_point,
        "clicked": clicked,
        "action_enabled": ACTION_ENABLED,
    }


def _collect_visible_tool_ids(
    main_window,
    window_title: str,
    backend: str,
    client: Work2VLMClient,
    timestamp_tag: str,
    step_index: int,
) -> tuple[str, dict | None]:
    """현재 View 화면에서 visible Tool ID 를 수집한다."""
    image = _capture_main_window(main_window, window_title, backend)
    if image is None:
        return EXIT_CAPTURE_FAILED, None

    full_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_full.jpg",
        timestamp_tag=timestamp_tag,
    )
    full_webp_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_full.webp",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, full_capture_path, log_name=LOG_NAME)
    save_debug_webp(image, full_webp_path, quality=90, log_name=LOG_NAME)

    full_w, full_h = image.size
    content_crop_box = _build_relative_crop_box(
        full_w,
        full_h,
        CONTENT_REGION_LEFT_RATIO,
        CONTENT_REGION_TOP_RATIO,
        CONTENT_REGION_RIGHT_RATIO,
        CONTENT_REGION_BOTTOM_RATIO,
    )
    content_image = crop_image(image, content_crop_box)

    content_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_step_{step_index:02d}_content.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(content_image, content_capture_path, log_name=LOG_NAME)

    relative_boxes = _detect_green_boxes(content_image)
    overlay_path = _save_green_box_overlay(content_image, relative_boxes, timestamp_tag, step_index)

    absolute_boxes = [
        {
            "left": content_crop_box["left"] + box["left"],
            "top": content_crop_box["top"] + box["top"],
            "right": content_crop_box["left"] + box["right"],
            "bottom": content_crop_box["top"] + box["bottom"],
        }
        for box in relative_boxes
    ]

    print(
        f"[INFO] step={step_index} green box 후보 {len(absolute_boxes)}개 "
        f"(content_box={content_crop_box})"
    )

    ocr_entries: list[dict] = []
    visible_tool_ids: list[str] = []
    seen_tool_ids: set[str] = set()

    for box_index, box in enumerate(absolute_boxes, start=1):
        box_image = crop_image(image, box)
        try:
            ocr_result = _run_box_ocr(
                client,
                box_image,
                timestamp_tag,
                step_index,
                box_index,
                window_title,
                backend,
            )
        except Exception as exc:
            print(f"[ERROR] green box OCR 요청 실패: step={step_index}, box={box_index}, error={exc}")
            return EXIT_OCR_REQUEST_ERROR, None

        selected_tool_id = ocr_result["selected_tool_id"]
        if selected_tool_id and selected_tool_id not in seen_tool_ids:
            seen_tool_ids.add(selected_tool_id)
            visible_tool_ids.append(selected_tool_id)

        ocr_entries.append(
            {
                "box_index": box_index,
                "box_on_full_image": box,
                "box_on_content_crop": relative_boxes[box_index - 1],
                "selected_tool_id": selected_tool_id,
                "tool_id_candidates": ocr_result["tool_id_candidates"],
                "raw_text": ocr_result["raw_text"],
                "artifacts": {
                    "capture": ocr_result["capture_path"],
                    "input_webp": ocr_result["webp_path"],
                    "ocr_text": ocr_result["raw_response_path"],
                    "result_json": ocr_result["result_json_path"],
                },
                "token_usage": ocr_result["token_usage"],
            }
        )

    scroll_anchor_image_point = {
        "x": content_crop_box["left"] + ((content_crop_box["right"] - content_crop_box["left"]) // 2),
        "y": content_crop_box["top"] + ((content_crop_box["bottom"] - content_crop_box["top"]) // 2),
    }
    scroll_anchor_screen_point = image_point_to_screen(main_window, scroll_anchor_image_point)

    step_payload = {
        "step_index": step_index,
        "content_crop_box": content_crop_box,
        "visible_tool_ids": visible_tool_ids,
        "green_box_count": len(absolute_boxes),
        "green_box_overlay": str(overlay_path),
        "full_capture": str(full_capture_path),
        "content_capture": str(content_capture_path),
        "scroll_anchor_image_point": scroll_anchor_image_point,
        "scroll_anchor_screen_point": scroll_anchor_screen_point,
        "ocr_entries": ocr_entries,
        "signature": _compute_signature(visible_tool_ids, absolute_boxes),
    }
    return EXIT_SUCCESS, step_payload


def main() -> str:
    """View 탭에서 visible Tool ID 를 페이지 단위로 스캔한다."""
    script_started_at = time.time()
    timestamp_tag = make_timestamp_tag(script_started_at)

    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        ocr_service=OCR_SERVICE_SLUG,
        action_enabled=ACTION_ENABLED,
        safe_mode=SAFE_MODE,
        max_scroll_steps=MAX_SCROLL_STEPS,
        scroll_dy=SCROLL_DY,
    )

    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print(
            "[ERROR] 메인 RCS 창을 찾지 못했습니다. "
            "먼저 로그인해서 메인 창을 띄운 뒤 다시 실행하세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="main_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=RCS_MAIN_WINDOW_TITLE_PREFIX,
        )
        return EXIT_MAIN_WINDOW_NOT_FOUND

    view_open_exit_code, view_action = _open_view_tab(main_window, window_title, backend)
    if view_open_exit_code != EXIT_SUCCESS:
        print(f"[ERROR] View 탭 열기 실패: {view_open_exit_code}")
        return view_open_exit_code

    client = Work2VLMClient(
        service_slug=OCR_SERVICE_SLUG,
        timeout_sec=120.0,
        log_name=LOG_NAME,
    )

    effective_max_scroll_steps = MAX_SCROLL_STEPS if ACTION_ENABLED else 0
    all_tool_ids: list[str] = []
    all_tool_id_set: set[str] = set()
    steps: list[dict] = []
    stagnant_steps = 0
    previous_signature = ""
    exit_code = EXIT_SUCCESS

    for step_index in range(effective_max_scroll_steps + 1):
        step_exit_code, step_payload = _collect_visible_tool_ids(
            main_window,
            window_title,
            backend,
            client,
            timestamp_tag,
            step_index,
        )
        if step_exit_code != EXIT_SUCCESS or step_payload is None:
            exit_code = step_exit_code
            break

        new_tool_ids: list[str] = []
        for tool_id in step_payload["visible_tool_ids"]:
            if tool_id in all_tool_id_set:
                continue
            all_tool_id_set.add(tool_id)
            all_tool_ids.append(tool_id)
            new_tool_ids.append(tool_id)

        step_payload["new_tool_ids"] = new_tool_ids
        step_payload["new_tool_count"] = len(new_tool_ids)
        steps.append(step_payload)

        print(
            f"[INFO] step={step_index} visible={step_payload['visible_tool_ids']}, "
            f"new={new_tool_ids}, total_unique={len(all_tool_ids)}"
        )

        if step_payload["signature"] == previous_signature:
            stagnant_steps += 1
            print(f"[INFO] step={step_index} signature 변화 없음: stagnant_steps={stagnant_steps}")
        elif not new_tool_ids:
            stagnant_steps += 1
            print(f"[INFO] step={step_index} 신규 Tool ID 없음: stagnant_steps={stagnant_steps}")
        else:
            stagnant_steps = 0

        previous_signature = step_payload["signature"]

        if step_index >= effective_max_scroll_steps:
            break

        if stagnant_steps > MAX_NO_NEW_SCROLLS:
            print(
                f"[INFO] 스캔 조기 종료: step={step_index}, "
                f"stagnant_steps={stagnant_steps}, limit={MAX_NO_NEW_SCROLLS}"
            )
            break

        scroll_anchor_screen_point = step_payload["scroll_anchor_screen_point"]
        if scroll_anchor_screen_point is None:
            exit_code = EXIT_CAPTURE_FAILED
            print("[ERROR] scroll anchor 스크린 좌표 변환 실패")
            break

        foreground_window(
            main_window,
            debug_label=f"scan_tools_from_view_pre_scroll_step_{step_index}",
        )
        time.sleep(PRE_CLICK_SETTLE_SEC)
        _scroll_at_screen(scroll_anchor_screen_point, step_index)
        time.sleep(POST_SCROLL_WAIT_SEC)

    if exit_code == EXIT_SUCCESS and not all_tool_ids:
        exit_code = EXIT_NO_TOOL_IDS_FOUND

    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "scan_tools_from_view_summary.json",
        timestamp_tag=timestamp_tag,
    )
    summary_payload = {
        "window_title": window_title,
        "backend": backend,
        "ocr_service": OCR_SERVICE_SLUG,
        "action_enabled": ACTION_ENABLED,
        "safe_mode": SAFE_MODE,
        "max_scroll_steps": effective_max_scroll_steps,
        "scroll_dy": SCROLL_DY,
        "view_action": view_action,
        "tool_ids": all_tool_ids,
        "tool_id_count": len(all_tool_ids),
        "steps": steps,
        "result": exit_code,
    }
    save_debug_json(summary_path, summary_payload)

    print(f"[INFO] Tool ID unique count={len(all_tool_ids)}")
    for index, tool_id in enumerate(all_tool_ids, start=1):
        print(f"[INFO] Tool ID {index:02d}: {tool_id}")
    print(f"[INFO] 요약 JSON 저장: {summary_path}")
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}, "
        f"result={exit_code}, action_enabled={ACTION_ENABLED}"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=exit_code,
        window_title=window_title,
        backend=backend,
        tool_id_count=len(all_tool_ids),
        summary_path=str(summary_path),
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return exit_code


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
