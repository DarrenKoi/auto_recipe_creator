"""RCS 로그인 창 UI-Venus Rev2 분석 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
UI-Venus 에게 "화면에서 실제로 보이는 것"만 기준으로
주요 UI 요소를 자유 형식으로 반환하게 한다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_venus_rev2.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event
from poc.work2.util import (
    activate_window,
    capture_window,
    debug_image_path,
    encode_image_webp,
    extract_json,
    format_elapsed_ms,
    foreground_window,
    make_timestamp_tag,
    parse_coords,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_image,
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text
from poc.work2.vlm_client import Work2VLMClient

load_dotenv()

WINDOW_TITLE_PREFIX = "Remote Control System"
PRIMARY_SERVICE_SLUG = "ui-venus"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_VLM_REQUEST_ERROR = "vlm_request_error"
EXIT_VLM_PARSE_ERROR = "vlm_parse_error"
EXIT_CAPTURE_FAILED = "capture_failed"

ROLE_COLORS = {
    "title_text": "tomato",
    "label": "gold",
    "input": "deepskyblue",
    "password_input": "limegreen",
    "button": "orange",
    "combobox": "salmon",
    "icon": "violet",
    "other": "white",
}
FALLBACK_COLORS = (
    "tomato",
    "gold",
    "deepskyblue",
    "limegreen",
    "orange",
    "violet",
    "cyan",
    "magenta",
    "chartreuse",
    "salmon",
)

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0

try:
    DEFAULT_VISIBLE_ELEMENT_LIMIT = int(
        os.getenv("RCS_LOGIN_VISIBLE_ELEMENT_LIMIT", "12")
    )
except ValueError:
    DEFAULT_VISIBLE_ELEMENT_LIMIT = 12


def _find_login_window():
    """기존 Rev2 로그인 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find

    return _find()


def _clean_text(value: object) -> str:
    """문자열 필드를 한 줄 텍스트로 정리한다."""
    text = str(value or "").strip()
    return " ".join(text.split())


def _slugify_label(text: str) -> str:
    """overlay key 용 ASCII label 을 생성한다."""
    lowered = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return lowered.strip("_") or "element"


def _normalize_role(value: object) -> str:
    """모델이 준 role 값을 내부 분류로 정리한다."""
    text = _clean_text(value).lower()
    aliases = {
        "title": "title_text",
        "title_text": "title_text",
        "titlebar_text": "title_text",
        "window_title": "title_text",
        "label": "label",
        "text_label": "label",
        "static_text": "label",
        "input": "input",
        "text_input": "input",
        "editable": "input",
        "field": "input",
        "password": "password_input",
        "password_input": "password_input",
        "button": "button",
        "combobox": "combobox",
        "combo_box": "combobox",
        "dropdown": "combobox",
        "drop_down": "combobox",
        "icon": "icon",
    }
    return aliases.get(text, "other")


def _resolve_raw_point(item: dict) -> tuple[object, object]:
    """응답 item 에서 좌표 후보를 꺼낸다."""
    point = item.get("point")
    if isinstance(point, dict):
        return point.get("x", item.get("x")), point.get("y", item.get("y"))
    return item.get("x"), item.get("y")


def _build_visible_only_prompt(
    width: int,
    height: int,
    max_items: int,
) -> tuple[str, str]:
    """고정 target key 없이 visible-first grounding 프롬프트를 구성한다."""
    system_message = (
        "GROUNDING task for a desktop GUI screenshot. "
        "Observe only what is actually visible in the screenshot. "
        "Do not assume a predefined login schema or hidden elements. "
        f"The screenshot is {width}x{height} pixels. "
        "Use coord_system='relative_1000' where x and y are integers from 0 to 1000. "
        "Respond ONLY with valid JSON."
    )

    user_text = f"""
This screenshot contains a Windows application login dialog.

First inspect the screenshot itself. Return only the major UI elements that are clearly visible and can be grounded reliably.
Do not assume specific fields, labels, buttons, or shortcut controls just because they are common in login windows.
If you cannot clearly see an element, omit it.

Return at most {max_items} items in this JSON shape:
{{
  "coord_system": "relative_1000",
  "elements": [
    {{
      "name": "short stable name based on what is visibly shown",
      "role": "title_text/label/input/password_input/button/combobox/icon/other",
      "visible_text": "exact visible text if readable, otherwise empty string",
      "x": 0,
      "y": 0
    }}
  ]
}}

Rules:
- Use only evidence from the screenshot.
- Do not enumerate expected login components from prior knowledge.
- Prefer important title-bar items, form labels, form controls, and dialog buttons that are truly visible.
- For text items, ground the visible text itself.
- For interactive controls, ground the point a user would actually click.
- visible_text must be copied from the screenshot when readable; otherwise use an empty string.
- name should be short and practical, but it must still come from what is visible in the screenshot.
- x and y must be integers from 0 to 1000.
- Return JSON only, with no markdown and no explanation.
""".strip()
    return system_message, user_text


def _normalize_visible_elements(
    parsed_json: dict,
    *,
    img_w: int,
    img_h: int,
) -> tuple[list[dict], dict[str, dict[str, int]], dict[str, str]]:
    """자유 형식 elements 배열을 overlay 가능한 픽셀 좌표로 정리한다."""
    raw_elements = parsed_json.get("elements")
    if not isinstance(raw_elements, list):
        raise ValueError("JSON 응답에 elements 리스트가 없습니다.")

    coord_payload = {
        "coord_system": parsed_json.get("coord_system") or parsed_json.get("coordinate_system")
    }
    key_order: list[str] = []
    metadata_by_key: dict[str, dict] = {}
    seen_keys: set[str] = set()

    for idx, item in enumerate(raw_elements, start=1):
        if not isinstance(item, dict):
            continue

        raw_x, raw_y = _resolve_raw_point(item)
        if raw_x is None or raw_y is None:
            continue

        visible_text = _clean_text(item.get("visible_text"))
        role = _normalize_role(item.get("role"))
        base_name = _clean_text(item.get("name")) or visible_text or role or f"element_{idx:02d}"
        key_root = _slugify_label(base_name)[:32]
        overlay_key = f"{idx:02d}_{key_root}"
        while overlay_key in seen_keys:
            overlay_key = f"{overlay_key}_dup"
        seen_keys.add(overlay_key)

        coord_payload[overlay_key] = {"x": raw_x, "y": raw_y}
        key_order.append(overlay_key)
        metadata_by_key[overlay_key] = {
            "overlay_key": overlay_key,
            "name": base_name,
            "role": role,
            "visible_text": visible_text,
            "raw_x": raw_x,
            "raw_y": raw_y,
        }

    parsed_coords = parse_coords(coord_payload, key_order, img_w, img_h)

    normalized_elements: list[dict] = []
    overlay_points: dict[str, dict[str, int]] = {}
    overlay_colors: dict[str, str] = {}

    for index, key in enumerate(key_order):
        pt = parsed_coords.get(key)
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue

        meta = metadata_by_key[key]
        normalized_item = {
            "overlay_key": key,
            "name": meta["name"],
            "role": meta["role"],
            "visible_text": meta["visible_text"],
            "raw_x": meta["raw_x"],
            "raw_y": meta["raw_y"],
            "x": int(pt["x"]),
            "y": int(pt["y"]),
        }
        normalized_elements.append(normalized_item)
        overlay_points[key] = {"x": normalized_item["x"], "y": normalized_item["y"]}
        overlay_colors[key] = ROLE_COLORS.get(
            meta["role"],
            FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        )

    return normalized_elements, overlay_points, overlay_colors


def _analyze_visible_login_elements(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 캡처하고 visible-first prompt 로 UI 요소를 분석한다."""
    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)

    if not activate_window(
        login_window,
        debug_label=f"login_window recapture backend={backend} title={window_title!r}",
    ):
        print(
            f"[ERROR] 로그인 창 재활성화 실패: title={window_title!r}, backend={backend}"
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="login_window_reactivate_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    if not foreground_window(
        login_window,
        debug_label=f"login_window screenshot backend={backend} title={window_title!r}",
    ):
        print(
            f"[ERROR] 로그인 창 foreground 활성화 실패: "
            f"title={window_title!r}, backend={backend}"
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="login_window_foreground_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="capture_failed",
            level="error",
            log_name=LOG_NAME,
            window_title=window_title,
            backend=backend,
            error=exc,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_CAPTURE_FAILED

    client = Work2VLMClient(service_slug=PRIMARY_SERVICE_SLUG, log_name=LOG_NAME)
    raw_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_capture.jpg",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    vlm_input_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_vlm_input.webp",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    raw_response_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_visible_response.txt",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    parsed_json_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_visible_response.json",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    overlay_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_visible_overlay.jpg",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )

    save_debug_jpeg(image, raw_capture_path, log_name=LOG_NAME)
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)
    image_b64, width, height = encode_image_webp(image)
    system_message, user_text = _build_visible_only_prompt(
        width=width,
        height=height,
        max_items=DEFAULT_VISIBLE_ELEMENT_LIMIT,
    )

    print(
        f"[INFO] 로그인 창 visible-first 분석 시작: backend={backend}, "
        f"title={window_title!r}, service_slug={PRIMARY_SERVICE_SLUG}"
    )

    try:
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            image_mime="image/webp",
            system_message=system_message,
            user_text=user_text,
            temperature=VLM_TEMPERATURE,
        )
    except Exception as exc:
        print(f"[ERROR] VLM 요청 실패: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="vlm_request_failed",
            level="error",
            log_name=LOG_NAME,
            backend=backend,
            window_title=window_title,
            service=PRIMARY_SERVICE_SLUG,
            model=client.model_name,
            error=exc,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_VLM_REQUEST_ERROR

    save_debug_text(raw_response_path, response.text)
    print(f"[INFO] VLM 응답 수신: tokens={response.token_usage or {}}")
    print(f"[INFO] 원문 응답:\n{response.text}\n")

    try:
        parsed_json = extract_json(response.text)
    except Exception as exc:
        print(f"[ERROR] VLM JSON 파싱 실패: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="vlm_json_parse_failed",
            level="error",
            log_name=LOG_NAME,
            backend=backend,
            window_title=window_title,
            service=PRIMARY_SERVICE_SLUG,
            model=response.model_name or client.model_name,
            error=exc,
            raw_response_path=raw_response_path,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_VLM_PARSE_ERROR

    print(f"[INFO] 파싱된 JSON:\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}\n")

    try:
        normalized_elements, overlay_points, overlay_colors = _normalize_visible_elements(
            parsed_json,
            img_w=width,
            img_h=height,
        )
    except Exception as exc:
        print(f"[ERROR] visible element 정규화 실패: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="visible_elements_normalize_failed",
            level="error",
            log_name=LOG_NAME,
            backend=backend,
            window_title=window_title,
            service=PRIMARY_SERVICE_SLUG,
            model=response.model_name or client.model_name,
            error=exc,
            raw_response_path=raw_response_path,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_VLM_PARSE_ERROR

    normalized_payload = {
        "coord_system": parsed_json.get("coord_system") or parsed_json.get("coordinate_system"),
        "element_count": len(normalized_elements),
        "elements": normalized_elements,
    }
    save_debug_json(parsed_json_path, normalized_payload)
    save_marked_image(image, overlay_points, overlay_colors, overlay_path)

    print(
        "[INFO] visible-first 분석 결과 "
        f"detected={len(normalized_elements)}, elapsed={format_elapsed_ms(started_at)}"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="analysis_finished",
        log_name=LOG_NAME,
        backend=backend,
        window_title=window_title,
        service=PRIMARY_SERVICE_SLUG,
        model=response.model_name or client.model_name,
        detected=len(normalized_elements),
        raw_capture_path=raw_capture_path,
        vlm_input_path=vlm_input_path,
        raw_response_path=raw_response_path,
        parsed_json_path=parsed_json_path,
        overlay_path=overlay_path,
        elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS if normalized_elements else EXIT_VLM_NO_DETECTION


def main() -> str:
    """이미 열려 있는 로그인 창을 읽고 visible-first debug image 를 생성한다."""
    script_started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        service=PRIMARY_SERVICE_SLUG,
    )

    login_window, window_title, backend = _find_login_window()
    if login_window is None:
        print(
            "[ERROR] 이미 떠 있는 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        log_work2_event(
            component=COMPONENT_NAME,
            message="login_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=WINDOW_TITLE_PREFIX,
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    result = _analyze_visible_login_elements(login_window, window_title, backend)
    print(f"[INFO] {LOG_NAME} end-to-end 소요: {format_elapsed_ms(script_started_at)}")
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=result,
        window_title=window_title,
        backend=backend,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return result


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
