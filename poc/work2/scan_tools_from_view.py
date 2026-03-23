"""RCS View 탭에서 target Tool ID 를 찾아 아래 화면을 더블클릭한다.

동작:
  1. 메인 RCS 창에서 `view_list_tab_rcs.py` 와 동일한 방식으로 View 탭 좌표를 찾는다.
  2. View 탭을 연 뒤 본문 영역에서 green box 후보를 탐지한다.
  3. 각 green box 를 개별 OCR 하여 target Tool ID 가 보이는지 확인한다.
  4. target 이 안 보이면 wheel down 으로 찾고, 계속 없으면 wheel up 으로 다시 찾는다.
  5. target Tool ID 를 찾으면 green box 바로 아래 화면 영역을 더블클릭해 tool screen 을 연다.

사용법:
  1. 로그인까지 완료해서 `RCS - ...` 메인 창이 떠 있는 상태로 둔다.
  2. `.env` 에 `SCAN_VIEW_TARGET_TOOL_ID=MCDC04` 같은 값을 설정한다.
  3. 실제 동작이 필요하면 `SAFE_MODE=false` 또는 `SCAN_VIEW_ACTION_ENABLED=true` 를 설정한다.
  4. uv run python poc/work2/scan_tools_from_view.py
"""

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
    build_relative_crop_box,
    capture_window,
    click_at_screen,
    crop_image,
    debug_image_path,
    env_flag,
    env_float,
    env_int,
    foreground_window,
    format_elapsed_ms,
    image_point_to_screen,
    make_timestamp_tag,
    normalize_lines,
    normalize_tool_text,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_bboxes,
    scroll_at_screen,
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
    print("[WARNING] cv2/numpy 미설치 — green box 탐지는 content crop 전체 fallback 으로 동작합니다.")

load_dotenv()


DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images" / "scan_tools_from_view"
LOG_NAME = Path(__file__).stem
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"

EXIT_MAIN_WINDOW_NOT_FOUND = "main_window_not_found"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_VIEW_TAB_NOT_FOUND = "view_tab_not_found"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"
EXIT_TARGET_NOT_FOUND = "target_not_found"

PRE_CLICK_SETTLE_SEC = 0.2
POST_VIEW_CLICK_WAIT_SEC = 1.5
POST_SCROLL_WAIT_SEC = 1.2
POST_DOUBLE_CLICK_WAIT_SEC = 0.5
OCR_MAX_TOKENS = 512

SAFE_MODE = env_flag("SAFE_MODE", default=True)
ACTION_ENABLED = env_flag("SCAN_VIEW_ACTION_ENABLED", default=not SAFE_MODE)
TARGET_TOOL_ID = os.getenv("SCAN_VIEW_TARGET_TOOL_ID", "MCDC04").strip() or "MCDC04"

DOWN_STEPS = max(1, env_int("SCAN_VIEW_DOWN_STEPS", 8))
UP_STEPS = max(1, env_int("SCAN_VIEW_UP_STEPS", 8))
DOWN_SCROLL_DY = env_int("SCAN_VIEW_DOWN_SCROLL_DY", -8)
UP_SCROLL_DY = env_int("SCAN_VIEW_UP_SCROLL_DY", 8)
MAX_GREEN_BOXES = max(1, env_int("SCAN_VIEW_MAX_GREEN_BOXES", 16))

CONTENT_REGION_LEFT_RATIO = env_float("SCAN_VIEW_CONTENT_LEFT_RATIO", 0.08)
CONTENT_REGION_TOP_RATIO = env_float("SCAN_VIEW_CONTENT_TOP_RATIO", 0.10)
CONTENT_REGION_RIGHT_RATIO = env_float("SCAN_VIEW_CONTENT_RIGHT_RATIO", 0.98)
CONTENT_REGION_BOTTOM_RATIO = env_float("SCAN_VIEW_CONTENT_BOTTOM_RATIO", 0.96)

GREEN_MIN_WIDTH = max(20, env_int("SCAN_VIEW_GREEN_MIN_WIDTH", 90))
GREEN_MIN_HEIGHT = max(12, env_int("SCAN_VIEW_GREEN_MIN_HEIGHT", 18))
GREEN_MAX_HEIGHT = max(GREEN_MIN_HEIGHT, env_int("SCAN_VIEW_GREEN_MAX_HEIGHT", 120))
GREEN_MIN_AREA = max(200, env_int("SCAN_VIEW_GREEN_MIN_AREA", 2200))
GREEN_MIN_ASPECT = env_float("SCAN_VIEW_GREEN_MIN_ASPECT", 1.8)
GREEN_MIN_COVERAGE = env_float("SCAN_VIEW_GREEN_MIN_COVERAGE", 0.35)

CLICK_Y_OFFSET_RATIO = env_float("SCAN_VIEW_CLICK_Y_OFFSET_RATIO", 1.6)
CLICK_Y_OFFSET_MIN = max(12, env_int("SCAN_VIEW_CLICK_Y_OFFSET_MIN", 28))
CLICK_Y_OFFSET_MAX = max(CLICK_Y_OFFSET_MIN, env_int("SCAN_VIEW_CLICK_Y_OFFSET_MAX", 120))


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
    phase: str,
    step_index: int,
    match_box: dict[str, int] | None = None,
    click_point: dict[str, int] | None = None,
) -> Path:
    """green box bbox overlay 를 저장한다."""
    overlay_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_{phase}_{step_index:02d}_green_boxes_overlay.jpg",
        model_name="green_box_detector",
        timestamp_tag=timestamp_tag,
    )

    elements: dict[str, dict] = {}
    colors: dict[str, str] = {}
    for index, box in enumerate(boxes, start=1):
        key = f"green_box_{index:02d}"
        elements[key] = {"bbox": box}
        colors[key] = "lime"

    if match_box is not None:
        elements["target_tool_box"] = {"bbox": match_box}
        colors["target_tool_box"] = "gold"

    if click_point is not None:
        elements["open_click_point"] = {
            "bbox": {
                "left": max(0, click_point["x"] - 8),
                "top": max(0, click_point["y"] - 8),
                "right": click_point["x"] + 9,
                "bottom": click_point["y"] + 9,
            },
            "center": click_point,
        }
        colors["open_click_point"] = "deepskyblue"

    save_marked_bboxes(content_image, elements, colors, overlay_path)
    return overlay_path


def _extract_normalized_tokens(raw_text: str, cached_lines: list[str] | None = None) -> list[str]:
    """OCR 원문에서 영숫자 토큰 후보를 정규화해서 반환한다."""
    tokens: list[str] = []
    seen: set[str] = set()

    lines = cached_lines if cached_lines is not None else normalize_lines(raw_text)
    for line in lines:
        parts = re.split(r"[^A-Za-z0-9]+", line)
        variants = [normalize_tool_text(line), *(normalize_tool_text(part) for part in parts)]
        for candidate in variants:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            tokens.append(candidate)

    return tokens


def _build_ocr_artifacts(ocr_result: dict) -> dict:
    """OCR 결과에서 artifact 경로 dict 를 만든다."""
    return {
        "capture": ocr_result["capture_path"],
        "input_webp": ocr_result["webp_path"],
        "ocr_text": ocr_result["raw_response_path"],
        "result_json": ocr_result["result_json_path"],
    }


def _compute_click_point_below_box(
    box: dict[str, int],
    content_crop_box: dict[str, int],
) -> dict[str, int]:
    """Tool ID green box 바로 아래 화면을 더블클릭할 좌표를 계산한다."""
    box_height = max(1, box["bottom"] - box["top"])
    offset = int(round(box_height * CLICK_Y_OFFSET_RATIO))
    offset = max(CLICK_Y_OFFSET_MIN, min(CLICK_Y_OFFSET_MAX, offset))

    click_x = box["left"] + ((box["right"] - box["left"]) // 2)
    click_y = min(content_crop_box["bottom"] - 4, box["bottom"] + offset)
    click_y = max(box["bottom"] + 1, click_y)

    return {"x": click_x, "y": click_y}


def _run_box_ocr(
    client: Work2VLMClient,
    box_image,
    timestamp_tag: str,
    phase: str,
    step_index: int,
    box_index: int,
    window_title: str,
    backend: str,
) -> dict:
    """green box crop 하나를 OCR 하고 정규화 토큰을 추출한다."""
    system_message, user_text = build_ocr_assist_prompt(
        box_image.size[0],
        box_image.size[1],
        context_label="view_green_box",
    )

    prefix = f"scan_tools_from_view_{phase}_{step_index:02d}_box_{box_index:02d}"
    capture_path = debug_image_path(
        DEBUG_IMAGE_DIR, f"{prefix}_crop.jpg",
        model_name=client.model_name, timestamp_tag=timestamp_tag,
    )
    webp_path = debug_image_path(
        DEBUG_IMAGE_DIR, f"{prefix}_input.webp",
        model_name=client.model_name, timestamp_tag=timestamp_tag,
    )
    raw_response_path = debug_image_path(
        DEBUG_IMAGE_DIR, f"{prefix}_ocr.txt",
        model_name=client.model_name, timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        DEBUG_IMAGE_DIR, f"{prefix}_result.json",
        model_name=client.model_name, timestamp_tag=timestamp_tag,
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
    normalized_lines_list = normalize_lines(raw_text)
    normalized_tokens = _extract_normalized_tokens(raw_text, cached_lines=normalized_lines_list)

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
            "phase": phase,
            "step_index": step_index,
            "box_index": box_index,
            "prompt_text": user_text,
            "raw_text": raw_text,
            "normalized_lines": normalized_lines_list,
            "normalized_tokens": normalized_tokens,
            "token_usage": response.token_usage,
        },
    )

    return {
        "raw_text": raw_text,
        "normalized_tokens": normalized_tokens,
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
        component_name=LOG_NAME,
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

    click_at_screen(screen_point, "view_tab", click_count=1, action_enabled=ACTION_ENABLED)
    time.sleep(POST_VIEW_CLICK_WAIT_SEC)

    return EXIT_SUCCESS, {
        "image_point": result.point,
        "screen_point": screen_point,
        "action_enabled": ACTION_ENABLED,
    }


def _scan_current_screen_for_target(
    main_window,
    window_title: str,
    backend: str,
    client: Work2VLMClient,
    timestamp_tag: str,
    target_tool_id: str,
    phase: str,
    step_index: int,
) -> tuple[str, dict | None]:
    """현재 View 화면에서 target Tool ID 를 찾는다."""
    image = _capture_main_window(main_window, window_title, backend)
    if image is None:
        return EXIT_CAPTURE_FAILED, None

    full_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        f"scan_tools_from_view_{phase}_{step_index:02d}_full.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, full_capture_path, log_name=LOG_NAME)

    full_w, full_h = image.size
    content_crop_box = build_relative_crop_box(
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
        f"scan_tools_from_view_{phase}_{step_index:02d}_content.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(content_image, content_capture_path, log_name=LOG_NAME)

    relative_boxes = _detect_green_boxes(content_image)
    absolute_boxes = [
        {
            "left": content_crop_box["left"] + box["left"],
            "top": content_crop_box["top"] + box["top"],
            "right": content_crop_box["left"] + box["right"],
            "bottom": content_crop_box["top"] + box["bottom"],
        }
        for box in relative_boxes
    ]

    target_normalized = normalize_tool_text(target_tool_id)
    match_entry: dict | None = None
    scan_entries: list[dict] = []

    for box_index, box in enumerate(absolute_boxes, start=1):
        box_image = crop_image(image, box)
        try:
            ocr_result = _run_box_ocr(
                client,
                box_image,
                timestamp_tag,
                phase,
                step_index,
                box_index,
                window_title,
                backend,
            )
        except Exception as exc:
            print(
                f"[ERROR] green box OCR 요청 실패: phase={phase}, step={step_index}, "
                f"box={box_index}, error={exc}"
            )
            return EXIT_OCR_REQUEST_ERROR, None

        matched = target_normalized in ocr_result["normalized_tokens"]
        artifacts = _build_ocr_artifacts(ocr_result)
        click_point = None
        if matched:
            click_point = _compute_click_point_below_box(box, content_crop_box)
            match_entry = {
                "box_index": box_index,
                "target_tool_id": target_tool_id,
                "tool_box_on_full_image": box,
                "tool_box_on_content_crop": relative_boxes[box_index - 1],
                "click_point_on_full_image": click_point,
                "click_point_on_content_crop": {
                    "x": click_point["x"] - content_crop_box["left"],
                    "y": click_point["y"] - content_crop_box["top"],
                },
                "ocr_raw_text": ocr_result["raw_text"],
                "ocr_normalized_tokens": ocr_result["normalized_tokens"],
                "artifacts": artifacts,
            }

        scan_entries.append(
            {
                "box_index": box_index,
                "box_on_full_image": box,
                "box_on_content_crop": relative_boxes[box_index - 1],
                "matched_target": matched,
                "ocr_raw_text": ocr_result["raw_text"],
                "ocr_normalized_tokens": ocr_result["normalized_tokens"],
                "artifacts": artifacts,
                "token_usage": ocr_result["token_usage"],
            }
        )

        if matched:
            print(
                f"[INFO] target Tool ID 발견: target={target_tool_id}, "
                f"phase={phase}, step={step_index}, box={box_index}"
            )
            break

    overlay_path = _save_green_box_overlay(
        content_image,
        relative_boxes,
        timestamp_tag,
        phase,
        step_index,
        match_box=match_entry["tool_box_on_content_crop"] if match_entry else None,
        click_point=match_entry["click_point_on_content_crop"] if match_entry else None,
    )

    scroll_anchor_image_point = {
        "x": content_crop_box["left"] + ((content_crop_box["right"] - content_crop_box["left"]) // 2),
        "y": content_crop_box["top"] + ((content_crop_box["bottom"] - content_crop_box["top"]) // 2),
    }
    scroll_anchor_screen_point = image_point_to_screen(main_window, scroll_anchor_image_point)

    payload = {
        "phase": phase,
        "step_index": step_index,
        "target_tool_id": target_tool_id,
        "content_crop_box": content_crop_box,
        "green_box_count": len(absolute_boxes),
        "full_capture": str(full_capture_path),
        "content_capture": str(content_capture_path),
        "green_box_overlay": str(overlay_path),
        "scroll_anchor_image_point": scroll_anchor_image_point,
        "scroll_anchor_screen_point": scroll_anchor_screen_point,
        "matched": match_entry is not None,
        "match_entry": match_entry,
        "scan_entries": scan_entries,
    }
    return EXIT_SUCCESS, payload


def _run_scroll_pass(
    main_window,
    window_title: str,
    backend: str,
    client: Work2VLMClient,
    timestamp_tag: str,
    target_tool_id: str,
    scroll_anchor_screen_point: dict[str, int],
    *,
    phase: str,
    num_steps: int,
    scroll_dy: int,
    search_steps: list[dict],
) -> tuple[str, dict | None]:
    """한 방향(down 또는 up) scroll pass 를 실행하고 target 을 찾으면 반환한다."""
    # 첫 scroll 전에 content 영역을 클릭하여 RCS 위젯에 scroll focus 를 잡는다.
    foreground_window(
        main_window,
        debug_label=f"scan_tools_from_view_focus_click_{phase}",
    )
    time.sleep(PRE_CLICK_SETTLE_SEC)
    click_at_screen(
        scroll_anchor_screen_point,
        f"focus_content_{phase}",
        click_count=1,
        action_enabled=ACTION_ENABLED,
    )
    time.sleep(PRE_CLICK_SETTLE_SEC)

    for step_index in range(1, num_steps + 1):
        foreground_window(
            main_window,
            debug_label=f"scan_tools_from_view_pre_scroll_{phase}_{step_index}",
        )
        time.sleep(PRE_CLICK_SETTLE_SEC)
        scroll_at_screen(
            scroll_anchor_screen_point, scroll_dy, phase, step_index,
            action_enabled=ACTION_ENABLED,
        )
        time.sleep(POST_SCROLL_WAIT_SEC)

        step_exit_code, step_payload = _scan_current_screen_for_target(
            main_window,
            window_title,
            backend,
            client,
            timestamp_tag,
            target_tool_id,
            phase=phase,
            step_index=step_index,
        )
        if step_exit_code != EXIT_SUCCESS or step_payload is None:
            return step_exit_code, None

        search_steps.append(step_payload)
        if step_payload["matched"]:
            return EXIT_SUCCESS, step_payload

    return EXIT_TARGET_NOT_FOUND, None


def _search_target_with_scroll(
    main_window,
    window_title: str,
    backend: str,
    client: Work2VLMClient,
    timestamp_tag: str,
    target_tool_id: str,
) -> tuple[str, dict]:
    """현재 화면 -> down pass -> up pass 순으로 target Tool ID 를 찾는다."""
    search_steps: list[dict] = []
    fallback_match: dict | None = None

    initial_exit_code, initial_payload = _scan_current_screen_for_target(
        main_window,
        window_title,
        backend,
        client,
        timestamp_tag,
        target_tool_id,
        phase="initial",
        step_index=0,
    )
    if initial_exit_code != EXIT_SUCCESS or initial_payload is None:
        return initial_exit_code, {"search_steps": search_steps}

    search_steps.append(initial_payload)
    if initial_payload["matched"]:
        fallback_match = initial_payload
        print("[INFO] 현재 화면에서 target 이 보였지만 down/up search 후에 클릭 후보로 유지합니다.")

    scroll_anchor_screen_point = initial_payload["scroll_anchor_screen_point"]
    if scroll_anchor_screen_point is None:
        return EXIT_CAPTURE_FAILED, {"search_steps": search_steps}

    for phase, num_steps, scroll_dy in [
        ("down", DOWN_STEPS, DOWN_SCROLL_DY),
        ("up", UP_STEPS, UP_SCROLL_DY),
    ]:
        pass_exit_code, found_step = _run_scroll_pass(
            main_window, window_title, backend, client,
            timestamp_tag, target_tool_id, scroll_anchor_screen_point,
            phase=phase, num_steps=num_steps, scroll_dy=scroll_dy,
            search_steps=search_steps,
        )
        if pass_exit_code == EXIT_SUCCESS and found_step is not None:
            return EXIT_SUCCESS, {
                "search_steps": search_steps,
                "found_step": found_step,
            }
        if pass_exit_code not in {EXIT_SUCCESS, EXIT_TARGET_NOT_FOUND}:
            return pass_exit_code, {"search_steps": search_steps}

    if fallback_match is not None:
        return EXIT_SUCCESS, {
            "search_steps": search_steps,
            "found_step": fallback_match,
            "used_initial_fallback": True,
        }

    return EXIT_TARGET_NOT_FOUND, {"search_steps": search_steps}


def main() -> str:
    """View 탭에서 target Tool ID 를 찾아 아래 화면을 더블클릭한다."""
    script_started_at = time.time()
    timestamp_tag = make_timestamp_tag(script_started_at)

    log_work2_event(
        component=LOG_NAME,
        message="script_started",
        log_name=LOG_NAME,
        target_tool_id=TARGET_TOOL_ID,
        ocr_service=OCR_SERVICE_SLUG,
        action_enabled=ACTION_ENABLED,
        safe_mode=SAFE_MODE,
        down_steps=DOWN_STEPS,
        up_steps=UP_STEPS,
    )

    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print(
            "[ERROR] 메인 RCS 창을 찾지 못했습니다. "
            "먼저 로그인해서 메인 창을 띄운 뒤 다시 실행하세요."
        )
        log_work2_event(
            component=LOG_NAME,
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

    search_exit_code, search_result = _search_target_with_scroll(
        main_window,
        window_title,
        backend,
        client,
        timestamp_tag,
        TARGET_TOOL_ID,
    )

    found_step = search_result.get("found_step")
    exit_code = search_exit_code
    double_clicked = False
    double_click_screen_point = None

    if search_exit_code == EXIT_SUCCESS and found_step is not None:
        match_entry = found_step.get("match_entry") or {}
        click_point_on_full_image = match_entry.get("click_point_on_full_image")
        if isinstance(click_point_on_full_image, dict):
            double_click_screen_point = image_point_to_screen(main_window, click_point_on_full_image)
        if double_click_screen_point is None:
            exit_code = EXIT_CAPTURE_FAILED
            print("[ERROR] double-click 스크린 좌표 변환 실패")
        else:
            foreground_window(
                main_window,
                debug_label=f"scan_tools_from_view_pre_open_{TARGET_TOOL_ID}",
            )
            time.sleep(PRE_CLICK_SETTLE_SEC)
            double_clicked = click_at_screen(
                double_click_screen_point,
                f"open_{TARGET_TOOL_ID}",
                click_count=2,
                action_enabled=ACTION_ENABLED,
            )
            time.sleep(POST_DOUBLE_CLICK_WAIT_SEC)
    elif search_exit_code == EXIT_TARGET_NOT_FOUND:
        print(f"[WARNING] target Tool ID 를 찾지 못했습니다: {TARGET_TOOL_ID}")

    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "scan_tools_from_view_summary.json",
        timestamp_tag=timestamp_tag,
    )
    summary_payload = {
        "window_title": window_title,
        "backend": backend,
        "target_tool_id": TARGET_TOOL_ID,
        "ocr_service": OCR_SERVICE_SLUG,
        "action_enabled": ACTION_ENABLED,
        "safe_mode": SAFE_MODE,
        "down_steps": DOWN_STEPS,
        "up_steps": UP_STEPS,
        "down_scroll_dy": DOWN_SCROLL_DY,
        "up_scroll_dy": UP_SCROLL_DY,
        "view_action": view_action,
        "search_result": search_result,
        "double_click_screen_point": double_click_screen_point,
        "double_clicked": double_clicked,
        "result": exit_code,
    }
    save_debug_json(summary_path, summary_payload)

    if found_step is not None:
        print(
            f"[INFO] target 발견: {TARGET_TOOL_ID}, "
            f"phase={found_step['phase']}, step={found_step['step_index']}"
        )
        if double_click_screen_point is not None:
            print(
                f"[INFO] target open double-click screen=({double_click_screen_point['x']}, "
                f"{double_click_screen_point['y']})"
            )
    print(f"[INFO] 요약 JSON 저장: {summary_path}")
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}, "
        f"target_tool_id={TARGET_TOOL_ID}, result={exit_code}, "
        f"action_enabled={ACTION_ENABLED}"
    )
    log_work2_event(
        component=LOG_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=exit_code,
        target_tool_id=TARGET_TOOL_ID,
        window_title=window_title,
        backend=backend,
        double_clicked=double_clicked,
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
