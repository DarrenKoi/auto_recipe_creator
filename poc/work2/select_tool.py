"""RCS List 탭에서 특정 Tool ID 를 찾아 더블클릭한다.

동작:
  1. 메인 RCS 창을 활성화하고 현재 화면을 캡처한다.
  2. 좌측 Tool List 영역을 crop 한다.
  3. PaddleOCR 로 대상 Tool ID 가 현재 list view 안에 보이는지 확인한다.
  4. ui-venus + mai-ui 로 대상 Tool row 의 더블클릭 좌표를 찾는다.
  5. Tool ID 를 더블클릭해 Tool 화면을 연다.

사용법:
  1. List 탭까지 열어 둔다.
  2. uv run python poc/work2/select_tool.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_rcs_common import RCS_MAIN_WINDOW_TITLE_PREFIX, wait_for_rcs_main_window
from poc.work2.logger import log_work2_event
from poc.work2.prompts import build_ocr_assist_prompt
from poc.work2.ui_venus_mai_locator import (
    EXIT_SUCCESS,
    TargetConfig,
    analyze_window_target,
)
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
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text
from poc.work2.vlm_client import Work2VLMClient

try:
    from pynput.mouse import Button, Controller as MouseController

    PYNPUT_MOUSE_AVAILABLE = True
except ImportError:
    PYNPUT_MOUSE_AVAILABLE = False
    print("[WARNING] pynput.mouse 미설치 — 클릭 동작은 로그만 출력됩니다.")

load_dotenv()


TARGET_TOOL_ID = os.getenv("SELECT_TOOL_TARGET_ID", "6MCD2201").strip() or "6MCD2201"
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images" / "select_tool"
LOG_NAME = Path(__file__).stem

EXIT_MAIN_WINDOW_NOT_FOUND = "main_window_not_found"
EXIT_WINDOW_ACTIVATE_FAILED = "window_activate_failed"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_TOOL_ID_NOT_VISIBLE = "tool_id_not_visible"
EXIT_TOOL_ROW_NOT_FOUND = "tool_row_not_found"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"

PRE_CLICK_SETTLE_SEC = 0.2
POST_DOUBLE_CLICK_SETTLE_SEC = 0.5
OCR_MAX_TOKENS = 2048

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


# List 탭의 Tool 목록이 좌측에 고정 배치된다는 전제를 사용한다.
LIST_REGION_LEFT_RATIO = _env_float("SELECT_TOOL_LIST_LEFT_RATIO", 0.00)
LIST_REGION_TOP_RATIO = _env_float("SELECT_TOOL_LIST_TOP_RATIO", 0.10)
LIST_REGION_RIGHT_RATIO = _env_float("SELECT_TOOL_LIST_RIGHT_RATIO", 0.42)
LIST_REGION_BOTTOM_RATIO = _env_float("SELECT_TOOL_LIST_BOTTOM_RATIO", 0.98)


def _tool_row_target(tool_id: str) -> TargetConfig:
    """지정 Tool ID row 를 찾기 위한 타겟 설정을 반환한다."""
    return TargetConfig(
        key="tool_row",
        description=(
            f"the tool row in the left-side RCS tool list whose visible tool ID text is exactly "
            f"'{tool_id}'. A small colored status square is immediately to the left of the ID text. "
            f"Return a safe point on that same row where a user would double-click to open the tool."
        ),
        left_pad_ratio=0.7,
        right_pad_ratio=1.8,
        vertical_pad_ratio=1.0,
        min_crop_width=360,
        min_crop_height=120,
    )


def _normalize_tool_text(text: str) -> str:
    """OCR 텍스트 비교를 위해 영숫자만 남기고 대문자로 정규화한다."""
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


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


def _double_click_at_screen(screen_point: dict[str, int], target_key: str) -> bool:
    """스크린 좌표에서 좌클릭 더블클릭을 수행한다."""
    sx, sy = screen_point["x"], screen_point["y"]

    if not PYNPUT_MOUSE_AVAILABLE:
        print(
            f"[INFO] [DRY-RUN] 더블클릭 생략 (pynput 없음): "
            f"target={target_key}, screen=({sx}, {sy})"
        )
        return True

    mouse = MouseController()
    mouse.position = (sx, sy)
    time.sleep(0.01)
    mouse.click(Button.left, 2)
    print(f"[INFO] 더블클릭 완료: target={target_key}, screen=({sx}, {sy})")
    return True


def _capture_main_window(main_window, window_title: str, backend: str):
    """메인 창을 활성화하고 한 번 캡처한다."""
    if not activate_window(
        main_window,
        debug_label=f"select_tool activate backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 활성화 실패: title={window_title!r}")
        return None

    if not foreground_window(
        main_window,
        debug_label=f"select_tool screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 메인 창 foreground 실패: title={window_title!r}")
        return None

    try:
        return capture_window(main_window)
    except Exception as exc:
        print(f"[ERROR] 메인 창 캡처 실패: {exc}")
        return None


def _run_list_ocr(
    list_image,
    tool_id: str,
    timestamp_tag: str,
    window_title: str,
    backend: str,
) -> dict:
    """좌측 Tool List crop 을 OCR 로 읽고 대상 Tool ID 존재 여부를 확인한다."""
    client = Work2VLMClient(
        service_slug=OCR_SERVICE_SLUG,
        timeout_sec=120.0,
        log_name=LOG_NAME,
    )
    system_message, user_text = build_ocr_assist_prompt(
        list_image.size[0],
        list_image.size[1],
        context_label="tool_list",
        focus_words=[tool_id],
    )

    list_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_list_crop.jpg",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    list_webp_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_list_input.webp",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    raw_response_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_list_ocr_response.txt",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )
    result_json_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "tool_list_ocr_result.json",
        model_name=client.model_name,
        timestamp_tag=timestamp_tag,
    )

    save_debug_jpeg(list_image, list_capture_path, log_name=LOG_NAME)
    save_debug_webp(list_image, list_webp_path, quality=90, log_name=LOG_NAME)

    try:
        response = client.chat_with_image_path(
            image_path=list_webp_path,
            system_message=system_message,
            user_text=user_text,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[ERROR] Tool List OCR 요청 실패: {exc}")
        raise

    raw_text = response.text.strip()
    normalized_lines = normalize_lines(raw_text, max_items=300)
    normalized_target = _normalize_tool_text(tool_id)
    matched_lines = [
        line for line in normalized_lines
        if normalized_target and normalized_target in _normalize_tool_text(line)
    ]
    target_visible = bool(matched_lines)

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
            "target_tool_id": tool_id,
            "prompt_text": user_text,
            "raw_text": raw_text,
            "normalized_lines": normalized_lines,
            "matched_lines": matched_lines,
            "target_visible": target_visible,
            "token_usage": response.token_usage,
        },
    )

    print(f"[INFO] Tool List OCR line_count={len(normalized_lines)}, target_visible={target_visible}")
    for index, line in enumerate(matched_lines[:10], start=1):
        print(f"[INFO] OCR matched line {index}: {line}")

    return {
        "matched_lines": matched_lines,
        "target_visible": target_visible,
    }


def main() -> str:
    """현재 List 탭에서 지정 Tool ID 를 찾아 더블클릭한다."""
    script_started_at = time.time()
    timestamp_tag = make_timestamp_tag(script_started_at)

    log_work2_event(
        component=LOG_NAME,
        message="script_started",
        log_name=LOG_NAME,
        target_tool_id=TARGET_TOOL_ID,
        ocr_service=OCR_SERVICE_SLUG,
    )

    main_window, window_title, backend = wait_for_rcs_main_window()
    if main_window is None:
        print(
            "[ERROR] 메인 RCS 창을 찾지 못했습니다. "
            "먼저 로그인 후 List 탭까지 연 뒤 다시 실행하세요."
        )
        log_work2_event(
            component=LOG_NAME,
            message="main_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=RCS_MAIN_WINDOW_TITLE_PREFIX,
        )
        return EXIT_MAIN_WINDOW_NOT_FOUND

    image = _capture_main_window(main_window, window_title, backend)
    if image is None:
        log_work2_event(
            component=LOG_NAME,
            message="window_activate_or_capture_failed",
            level="error",
            log_name=LOG_NAME,
            window_title=window_title,
            backend=backend,
        )
        return EXIT_WINDOW_ACTIVATE_FAILED

    full_capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "main_window_capture.jpg",
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, full_capture_path, log_name=LOG_NAME)

    full_w, full_h = image.size
    list_crop_box = _build_relative_crop_box(
        full_w,
        full_h,
        LIST_REGION_LEFT_RATIO,
        LIST_REGION_TOP_RATIO,
        LIST_REGION_RIGHT_RATIO,
        LIST_REGION_BOTTOM_RATIO,
    )
    list_image = crop_image(image, list_crop_box)
    print(
        f"[INFO] Tool List crop: box={list_crop_box}, "
        f"size={list_image.size[0]}x{list_image.size[1]}"
    )

    try:
        ocr_result = _run_list_ocr(
            list_image,
            TARGET_TOOL_ID,
            timestamp_tag,
            window_title,
            backend,
        )
    except Exception as exc:
        log_work2_event(
            component=LOG_NAME,
            message="ocr_request_failed",
            level="error",
            log_name=LOG_NAME,
            target_tool_id=TARGET_TOOL_ID,
            error=exc,
        )
        return EXIT_OCR_REQUEST_ERROR

    if not ocr_result["target_visible"]:
        print(
            f"[WARNING] 현재 Tool List crop 에서 대상 Tool ID 가 보이지 않습니다: "
            f"{TARGET_TOOL_ID}"
        )
        summary_path = debug_image_path(
            DEBUG_IMAGE_DIR,
            "select_tool_summary.json",
            timestamp_tag=timestamp_tag,
        )
        save_debug_json(
            summary_path,
            {
                "window_title": window_title,
                "backend": backend,
                "target_tool_id": TARGET_TOOL_ID,
                "list_crop_box": list_crop_box,
                "ocr_target_visible": False,
                "ocr_matched_lines": ocr_result["matched_lines"],
            },
        )
        return EXIT_TOOL_ID_NOT_VISIBLE

    tool_result = analyze_window_target(
        main_window,
        window_title,
        backend,
        _tool_row_target(TARGET_TOOL_ID),
        debug_image_dir=DEBUG_IMAGE_DIR,
        log_name=LOG_NAME,
        component_name=LOG_NAME,
        artifact_prefix=f"select_tool_{TARGET_TOOL_ID.lower()}",
        result_mode="ui_venus_then_mai_ui_tool_list",
        image=list_image,
    )
    if tool_result.exit_code != EXIT_SUCCESS or tool_result.point is None:
        print(
            f"[ERROR] Tool row 좌표 탐지 실패: target_tool_id={TARGET_TOOL_ID}, "
            f"exit_code={tool_result.exit_code}"
        )
        return EXIT_TOOL_ROW_NOT_FOUND

    full_image_point = {
        "x": list_crop_box["left"] + tool_result.point["x"],
        "y": list_crop_box["top"] + tool_result.point["y"],
    }
    screen_point = image_point_to_screen(main_window, full_image_point)
    if screen_point is None:
        return EXIT_CAPTURE_FAILED

    foreground_window(
        main_window,
        debug_label=f"pre_double_click_{TARGET_TOOL_ID}",
    )
    time.sleep(PRE_CLICK_SETTLE_SEC)

    print(
        f"[INFO] Tool 더블클릭 실행: target_tool_id={TARGET_TOOL_ID}, "
        f"list_image_point=({tool_result.point['x']}, {tool_result.point['y']}), "
        f"full_image_point=({full_image_point['x']}, {full_image_point['y']}), "
        f"screen=({screen_point['x']}, {screen_point['y']})"
    )
    clicked = _double_click_at_screen(screen_point, TARGET_TOOL_ID)
    time.sleep(POST_DOUBLE_CLICK_SETTLE_SEC)

    summary_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "select_tool_summary.json",
        timestamp_tag=timestamp_tag,
    )
    save_debug_json(
        summary_path,
        {
            "window_title": window_title,
            "backend": backend,
            "target_tool_id": TARGET_TOOL_ID,
            "list_crop_box": list_crop_box,
            "ocr_target_visible": ocr_result["target_visible"],
            "ocr_matched_lines": ocr_result["matched_lines"],
            "tool_point_on_list_crop": tool_result.point,
            "tool_point_on_full_image": full_image_point,
            "tool_point_on_screen": screen_point,
            "double_clicked": clicked,
        },
    )

    exit_code = EXIT_SUCCESS if clicked else EXIT_TOOL_ROW_NOT_FOUND
    print(
        f"[INFO] {LOG_NAME} 총 소요: {format_elapsed_ms(script_started_at)}, "
        f"target_tool_id={TARGET_TOOL_ID}, result={exit_code}"
    )
    print(f"[INFO] 요약 JSON 저장: {summary_path}")
    log_work2_event(
        component=LOG_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=exit_code,
        window_title=window_title,
        backend=backend,
        target_tool_id=TARGET_TOOL_ID,
        list_crop_box=list_crop_box,
        screen_point=screen_point,
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
