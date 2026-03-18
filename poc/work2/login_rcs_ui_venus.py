"""RCS 로그인 창 UI-Venus 전용 분석 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
UI-Venus 단일 모델로 grounding 좌표를 읽어 debug image 를 생성한다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_venus.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.login_benchmark import (
    benchmark_has_success,
    print_benchmark_summary,
    run_login_benchmark,
)
from poc.work2.logger import log_work2_event
from poc.work2.util import (
    activate_window,
    capture_window,
    format_elapsed_ms,
    foreground_window,
    make_timestamp_tag,
)

load_dotenv()

WINDOW_TITLE_PREFIX = "Remote Control System"
PRIMARY_SERVICE_SLUG = "ui-venus"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
LOGIN_TARGET_KEYS = [
    "window_title_text",
    "close_button",
    "server_label",
    "server_input",
    "userid_label",
    "userid_input",
    "password_label",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
]
ELEMENT_COLORS = {
    "window_title_text": "tomato",
    "close_button": "violet",
    "server_label": "gold",
    "server_input": "salmon",
    "userid_label": "dodgerblue",
    "userid_input": "deepskyblue",
    "password_label": "chartreuse",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}
EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_VLM_REQUEST_ERROR = "vlm_request_error"
EXIT_CAPTURE_FAILED = "capture_failed"

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0


def _find_login_window():
    """login_rcs_Rev2 의 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find

    return _find()


def _locate_login_controls(login_window, window_title: str, backend: str) -> str:
    """로그인 창 스크린샷을 UI-Venus 단일 모델로 분석하고 overlay 를 저장한다."""
    locate_started_at = time.time()
    debug_stamp = make_timestamp_tag(locate_started_at)

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
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
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
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
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
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
        )
        return EXIT_CAPTURE_FAILED

    print(
        f"[INFO] 로그인 창 분석 시작: backend={backend}, title={window_title!r}, "
        f"service_slug={PRIMARY_SERVICE_SLUG}"
    )

    try:
        results = run_login_benchmark(
            image=image,
            service_slugs=(PRIMARY_SERVICE_SLUG,),
            debug_image_dir=DEBUG_IMAGE_DIR,
            debug_stamp=debug_stamp,
            target_keys=LOGIN_TARGET_KEYS,
            element_colors=ELEMENT_COLORS,
            temperature=VLM_TEMPERATURE,
            base_log_name=LOG_NAME,
            context_fields={
                "backend": backend,
                "window_title": window_title,
            },
        )
    except ValueError as exc:
        print(f"[ERROR] 로그인 분석 설정 오류: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="benchmark_configuration_invalid",
            level="error",
            log_name=LOG_NAME,
            backend=backend,
            window_title=window_title,
            error=exc,
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
        )
        return EXIT_VLM_REQUEST_ERROR

    print_benchmark_summary(results)
    best_detected = max((item.detected_count for item in results), default=0)
    print(f"[INFO] 로그인 이미지 분석 전체 소요: {format_elapsed_ms(locate_started_at)}")
    log_work2_event(
        component=COMPONENT_NAME,
        message="analysis_finished",
        log_name=LOG_NAME,
        backend=backend,
        window_title=window_title,
        service_slugs=",".join(item.service_slug for item in results),
        best_detected=best_detected,
        target_count=len(LOGIN_TARGET_KEYS),
        elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS if benchmark_has_success(results) else EXIT_VLM_NO_DETECTION


def main() -> str:
    """이미 열려 있는 로그인 창을 읽고 UI-Venus debug image 를 생성한다."""
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

    result = _locate_login_controls(login_window, window_title, backend)
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
