"""RCS 로그인 창 UI-Venus Rev2 분석 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
UI-Venus 1.5 공식 grounding 형식으로 단일 요소씩 좌표를 추출한다.

공식 프롬프트:
  "Output the center point of the position corresponding to the following
   instruction: {instruction}. The output should just be the coordinates
   of a point, in the format [x,y]."

응답: [x, y] (0-1000 정규화 좌표) 또는 [-1, -1] (보이지 않는 요소)

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_venus_rev2.py
"""

import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event
from poc.work2.prompts.prompt_login_rcs_ui_venus import (
    build_ui_venus_single_element_prompt_by_key,
)
from poc.work2.util import (
    activate_window,
    capture_window,
    debug_image_path,
    encode_image_webp,
    foreground_window,
    format_elapsed_ms,
    make_timestamp_tag,
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
EXIT_CAPTURE_FAILED = "capture_failed"

# 이번 rev2 에서 찾을 요소 목록 — 지금은 userid_input 만
TARGET_KEYS = ("userid_input",)

ELEMENT_COLORS = {
    "window_title_text": "tomato",
    "close_button": "white",
    "server_label": "gold",
    "server_input": "salmon",
    "userid_label": "gold",
    "userid_input": "deepskyblue",
    "password_label": "gold",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "orange",
    "shortcut_button": "violet",
}
FALLBACK_COLOR = "cyan"

# [x, y] 좌표 파싱 정규식
_COORD_PATTERN = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def _find_login_window():
    """기존 로그인 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find

    return _find()


def _parse_point_response(text: str) -> tuple[float, float] | None:
    """VLM 응답에서 [x, y] 좌표를 추출한다.

    Returns:
        (x, y) 0-1000 정규화 좌표. 파싱 실패 또는 [-1, -1] 이면 None.
    """
    match = _COORD_PATTERN.search(text)
    if not match:
        return None

    x = float(match.group(1))
    y = float(match.group(2))

    # [-1, -1] 은 요소를 찾지 못했다는 공식 refusal 응답
    if x < 0 or y < 0:
        return None

    return x, y


def _to_pixel(value: float, axis_size: int) -> int:
    """0-1000 정규화 좌표를 픽셀 좌표로 변환한다."""
    pixel = value / 1000.0 * (axis_size - 1)
    return max(0, min(int(round(pixel)), axis_size - 1))


def _ground_single_element(
    client: Work2VLMClient,
    image_b64: str,
    element_key: str,
    img_w: int,
    img_h: int,
) -> dict | None:
    """단일 요소에 대해 공식 UI-Venus grounding 을 수행한다.

    Returns:
        {"key": str, "x": int, "y": int, "raw_x": float, "raw_y": float,
         "response_text": str} 또는 None.
    """
    system_message, user_text = build_ui_venus_single_element_prompt_by_key(element_key)

    response = client.chat_with_image_b64(
        image_b64=image_b64,
        image_mime="image/webp",
        system_message=system_message,
        user_text=user_text,
        temperature=0.0,
    )

    print(f"[INFO] [{element_key}] VLM 응답: {response.text!r}")
    print(f"[INFO] [{element_key}] tokens={response.token_usage or {}}")

    point = _parse_point_response(response.text)
    if point is None:
        print(f"[INFO] [{element_key}] 좌표 추출 실패 또는 refusal [-1,-1]")
        return None

    raw_x, raw_y = point
    px_x = _to_pixel(raw_x, img_w)
    px_y = _to_pixel(raw_y, img_h)
    print(f"[INFO] [{element_key}] raw=({raw_x}, {raw_y}) -> px=({px_x}, {px_y})")

    return {
        "key": element_key,
        "x": px_x,
        "y": px_y,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "response_text": response.text,
    }


def _analyze_login_elements(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 캡처하고 공식 단일 요소 grounding 으로 분석한다."""
    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)

    # 창 활성화
    if not activate_window(
        login_window,
        debug_label=f"login_window backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 활성화 실패: title={window_title!r}")
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    if not foreground_window(
        login_window,
        debug_label=f"login_window screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 foreground 실패: title={window_title!r}")
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    # 캡처
    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        return EXIT_CAPTURE_FAILED

    client = Work2VLMClient(service_slug=PRIMARY_SERVICE_SLUG, log_name=LOG_NAME)
    image_b64, width, height = encode_image_webp(image)

    # 디버그 이미지 저장
    capture_path = debug_image_path(
        DEBUG_IMAGE_DIR, "login_rcs_capture.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    vlm_input_path = debug_image_path(
        DEBUG_IMAGE_DIR, "login_rcs_vlm_input.webp",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    save_debug_jpeg(image, capture_path, log_name=LOG_NAME)
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)

    print(
        f"[INFO] 공식 UI-Venus 단일 요소 grounding 시작: "
        f"image={width}x{height}, targets={TARGET_KEYS}"
    )

    # 각 요소별 단일 VLM 호출
    results: list[dict] = []
    for element_key in TARGET_KEYS:
        try:
            result = _ground_single_element(
                client, image_b64, element_key, width, height,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:
            print(f"[ERROR] [{element_key}] VLM 요청 실패: {exc}")
            log_work2_event(
                component=COMPONENT_NAME,
                message="vlm_single_element_failed",
                level="error",
                log_name=LOG_NAME,
                element_key=element_key,
                error=exc,
            )

    # 결과 저장
    result_payload = {
        "coord_system": "official_ui_venus_1000",
        "image_width": width,
        "image_height": height,
        "target_count": len(TARGET_KEYS),
        "detected_count": len(results),
        "elements": results,
    }

    result_json_path = debug_image_path(
        DEBUG_IMAGE_DIR, "login_rcs_grounding_result.json",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    save_debug_json(result_json_path, result_payload)
    print(f"[INFO] 결과 JSON 저장: {result_json_path}")

    # 오버레이 이미지 생성
    if results:
        overlay_points = {}
        overlay_colors = {}
        for r in results:
            overlay_points[r["key"]] = {"x": r["x"], "y": r["y"]}
            overlay_colors[r["key"]] = ELEMENT_COLORS.get(r["key"], FALLBACK_COLOR)

        overlay_path = debug_image_path(
            DEBUG_IMAGE_DIR, "login_rcs_grounding_overlay.jpg",
            model_name=client.model_name, timestamp_tag=debug_stamp,
        )
        save_marked_image(image, overlay_points, overlay_colors, overlay_path)
        print(f"[INFO] 오버레이 이미지 저장: {overlay_path}")

    elapsed = format_elapsed_ms(started_at)
    print(
        f"[INFO] grounding 완료: detected={len(results)}/{len(TARGET_KEYS)}, "
        f"elapsed={elapsed}"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="analysis_finished",
        log_name=LOG_NAME,
        backend=backend,
        window_title=window_title,
        service=PRIMARY_SERVICE_SLUG,
        model=client.model_name,
        target_count=len(TARGET_KEYS),
        detected_count=len(results),
        elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS if results else EXIT_VLM_NO_DETECTION


def main() -> str:
    """이미 열려 있는 로그인 창을 읽고 공식 UI-Venus grounding 을 수행한다."""
    script_started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        service=PRIMARY_SERVICE_SLUG,
        target_keys=list(TARGET_KEYS),
    )

    login_window, window_title, backend = _find_login_window()
    if login_window is None:
        print(
            "[ERROR] 이미 떠 있는 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    result = _analyze_login_elements(login_window, window_title, backend)
    print(f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}")
    return result


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
