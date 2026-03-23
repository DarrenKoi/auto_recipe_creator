"""RCS 로그인 창 UI-TARS Rev2 grounding 체크 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
UI-TARS 공식 단일 action 형식으로 요소별 grounding 성능을 점검한다.

UI-TARS 는 bbox 가 아니라 click action point 계약이 중심이므로,
이 스크립트는 모델 좌표와 원본 픽셀 좌표를 함께 저장해
text / input / button 에서 얼마나 안정적으로 찍는지 확인하는 목적이다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_tars_rev2.py
"""

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_rcs_ui_tars import (
    UI_TARS_FREQUENCY_PENALTY,
    convert_ui_tars_coords,
    resolve_ui_tars_stream_override,
    smart_resize,
)
from poc.work2.logger import log_work2_event
from poc.work2.prompts.prompt_login_rcs_ui_tars import build_single_element_prompt
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
PRIMARY_SERVICE_SLUG = "ui-tars"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_CAPTURE_FAILED = "capture_failed"

TARGET_KEYS = (
    "userid_label",
    "userid_input",
    "password_label",
    "password_input",
    "login_button",
    "cancel_button",
)

MARKER_COLORS = {
    "userid_label": "dodgerblue",
    "userid_input": "deepskyblue",
    "password_label": "chartreuse",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
}

UI_TARS_MAX_TOKENS = 200

_COORD_VALUE = (
    r"['\"]?"
    r"(?:<point>)?"
    r"\(?\s*"
    r"(?P<x>\d+)\s*[,\s]+\s*(?P<y>\d+)"
    r"\s*\)?"
    r"(?:</point>)?"
    r"['\"]?"
)
_POINT_ACTION_PATTERN = re.compile(
    rf"(?:Action\s*:\s*)?click\s*\(\s*point\s*=\s*{_COORD_VALUE}\s*\)",
    re.IGNORECASE,
)
_START_BOX_ACTION_PATTERN = re.compile(
    rf"(?:Action\s*:\s*)?click\s*\(\s*start_box\s*=\s*{_COORD_VALUE}\s*\)",
    re.IGNORECASE,
)


def _find_login_window():
    """기존 로그인 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find

    return _find()


def _parse_single_action_response(text: str) -> tuple[int, int, str] | None:
    """UI-TARS 단일 요소 응답에서 모델 좌표를 추출한다."""
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue

        point_match = _POINT_ACTION_PATTERN.search(line)
        if point_match is not None:
            return (
                int(point_match.group("x")),
                int(point_match.group("y")),
                "point",
            )

        start_box_match = _START_BOX_ACTION_PATTERN.search(line)
        if start_box_match is not None:
            return (
                int(start_box_match.group("x")),
                int(start_box_match.group("y")),
                "start_box",
            )

    return None


def _ground_single_element(
    client: Work2VLMClient,
    image_b64: str,
    element_key: str,
    img_w: int,
    img_h: int,
    stream: bool | None,
) -> dict | None:
    """단일 요소에 대해 UI-TARS 공식 grounding 요청을 수행한다."""
    system_message, user_text = build_single_element_prompt(element_key)

    response = client.chat_with_image_b64(
        image_b64=image_b64,
        image_mime="image/webp",
        system_message=system_message,
        user_text=user_text,
        temperature=0.0,
        max_tokens=UI_TARS_MAX_TOKENS,
        frequency_penalty=UI_TARS_FREQUENCY_PENALTY,
        stream=stream,
    )

    print(f"[INFO] [{element_key}] UI-TARS 응답: {response.text!r}")
    print(f"[INFO] [{element_key}] tokens={response.token_usage or {}}")

    parsed = _parse_single_action_response(response.text)
    if parsed is None:
        print(f"[INFO] [{element_key}] action 좌표 추출 실패")
        return None

    model_x, model_y, action_style = parsed
    pixel_x, pixel_y = convert_ui_tars_coords(model_x, model_y, img_w, img_h)
    resized_h, resized_w = smart_resize(img_h, img_w)
    print(
        f"[INFO] [{element_key}] {action_style}=({model_x}, {model_y}) "
        f"smart_resize={resized_w}x{resized_h} -> px=({pixel_x}, {pixel_y})"
    )

    return {
        "key": element_key,
        "action_style": action_style,
        "model_x": model_x,
        "model_y": model_y,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "x": pixel_x,
        "y": pixel_y,
        "response_text": response.text,
    }


def _analyze_login_elements(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 캡처하고 UI-TARS 단일 요소 grounding 을 점검한다."""
    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)
    stream_override = resolve_ui_tars_stream_override()

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

    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        return EXIT_CAPTURE_FAILED

    client = Work2VLMClient(service_slug=PRIMARY_SERVICE_SLUG, log_name=LOG_NAME)
    image_b64, width, height = encode_image_webp(image)

    capture_path = debug_image_path(
        DEBUG_IMAGE_DIR, "login_rcs_capture.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    vlm_input_path = debug_image_path(
        DEBUG_IMAGE_DIR, "login_rcs_vlm_input.webp",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    responses_dir = debug_image_path(
        DEBUG_IMAGE_DIR, "per_element_responses",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    save_debug_jpeg(image, capture_path, log_name=LOG_NAME)
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)

    print(
        f"[INFO] UI-TARS rev2 grounding 시작: "
        f"image={width}x{height}, targets={TARGET_KEYS}, "
        f"stream={client.prefer_stream if stream_override is None else stream_override}"
    )

    results: list[dict] = []
    overlay_points: dict[str, dict[str, int]] = {}

    for element_key in TARGET_KEYS:
        try:
            result = _ground_single_element(
                client,
                image_b64,
                element_key,
                width,
                height,
                stream_override,
            )
            if result is None:
                continue

            results.append(result)
            overlay_points[element_key] = {"x": result["x"], "y": result["y"]}

            response_path = Path(str(responses_dir)) / f"{element_key}.txt"
            save_debug_text(response_path, result["response_text"])
        except Exception as exc:
            print(f"[ERROR] [{element_key}] UI-TARS 요청 실패: {exc}")
            log_work2_event(
                component=COMPONENT_NAME,
                message="vlm_single_element_failed",
                level="error",
                log_name=LOG_NAME,
                element_key=element_key,
                error=exc,
            )

    resized_h, resized_w = smart_resize(height, width)
    result_payload = {
        "coord_system": "pixel",
        "model_coord_system": "ui_tars_smart_resize_absolute",
        "mode": "ui_tars_per_element_experiment",
        "image_width": width,
        "image_height": height,
        "resized_width": resized_w,
        "resized_height": resized_h,
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

    if overlay_points:
        overlay_path = debug_image_path(
            DEBUG_IMAGE_DIR, "login_rcs_grounding_overlay.jpg",
            model_name=client.model_name, timestamp_tag=debug_stamp,
        )
        save_marked_image(image, overlay_points, MARKER_COLORS, overlay_path)
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
    """이미 열려 있는 로그인 창을 읽고 UI-TARS debug image 를 생성한다."""
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

    result = _analyze_login_elements(login_window, window_title, backend)
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
