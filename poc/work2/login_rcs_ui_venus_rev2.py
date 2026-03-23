"""RCS 로그인 창 UI-Venus Rev2 분석 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
UI-Venus 기반으로 단일 요소씩 bbox grounding 실험을 수행한다.

실험 프롬프트:
  요소별 자연어 instruction 을 넣고 bbox JSON 을 요청한다.
  이 스크립트는 텍스트/버튼/입력창에 대해
  모델이 박스를 얼마나 안정적으로 주는지 확인하는 목적이다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_venus_rev2.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event
from poc.work2.prompts.prompt_login_rcs_ui_venus import (
    build_ui_venus_single_element_bbox_prompt_by_key,
)
from poc.work2.util import (
    activate_window,
    bbox_1000_to_pixels,
    bbox_center,
    capture_window,
    debug_image_path,
    encode_image_webp,
    foreground_window,
    format_elapsed_ms,
    make_timestamp_tag,
    normalize_bbox_1000,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_bboxes,
)
from poc.work2.util.debug_image_utils import save_debug_json
from poc.work2.util.json_utils import extract_json
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


def _find_login_window():
    """기존 로그인 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find

    return _find()


def _ground_single_element_bbox(
    client: Work2VLMClient,
    image_b64: str,
    element_key: str,
    img_w: int,
    img_h: int,
) -> dict | None:
    """단일 요소에 대해 bbox grounding 실험을 수행한다.

    Returns:
        bbox/center/raw_response 정보가 담긴 dict 또는 None.
    """
    system_message, user_text = build_ui_venus_single_element_bbox_prompt_by_key(element_key)

    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=system_message,
        user_text=user_text,
        temperature=0.0,
    )

    print(f"[INFO] [{element_key}] VLM 응답: {response.text!r}")
    print(f"[INFO] [{element_key}] tokens={response.token_usage or {}}")

    try:
        parsed = extract_json(response.text)
    except Exception:
        print(f"[INFO] [{element_key}] JSON 파싱 실패")
        return None

    bbox_1000 = normalize_bbox_1000(parsed.get("bbox"))
    if bbox_1000 is None:
        print(f"[INFO] [{element_key}] bbox 추출 실패 또는 미검출")
        return None

    bbox_pixels = bbox_1000_to_pixels(bbox_1000, img_w, img_h)
    center = bbox_center(bbox_pixels)
    print(
        f"[INFO] [{element_key}] bbox1000={bbox_1000} "
        f"-> px={bbox_pixels}, center=({center['x']}, {center['y']})"
    )

    return {
        "key": element_key,
        "bbox_1000": bbox_1000,
        "bbox_pixels": bbox_pixels,
        "center_x": center["x"],
        "center_y": center["y"],
        "response_text": response.text,
    }


def _analyze_login_elements(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 캡처하고 단일 요소 bbox grounding 으로 분석한다."""
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
        f"[INFO] UI-Venus bbox grounding 실험 시작: "
        f"image={width}x{height}, targets={TARGET_KEYS}"
    )

    # 각 요소별 단일 VLM 호출
    results: list[dict] = []
    for element_key in TARGET_KEYS:
        try:
            result = _ground_single_element_bbox(
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
        "coord_system": "relative_1000",
        "mode": "ui_venus_bbox_experiment",
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
        overlay_items = {
            r["key"]: {
                "bbox": r["bbox_pixels"],
                "center": {"x": r["center_x"], "y": r["center_y"]},
            }
            for r in results
        }

        overlay_path = debug_image_path(
            DEBUG_IMAGE_DIR, "login_rcs_grounding_overlay.jpg",
            model_name=client.model_name, timestamp_tag=debug_stamp,
        )
        save_marked_bboxes(image, overlay_items, MARKER_COLORS, overlay_path)
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
