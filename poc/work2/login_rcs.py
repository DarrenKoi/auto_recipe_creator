"""RCS 로그인 창 읽기 전용 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
VLM 으로 입력창/버튼 좌표를 읽어 debug image 를 생성한다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event
from poc.work2.prompts import build_rcs_login_locator_prompt
from poc.work2.util import (
    activate_window,
    capture_window,
    debug_image_path,
    encode_image_webp,
    extract_json,
    find_window_by_pid_and_title_prefix,
    find_window_by_title_prefix,
    format_elapsed_ms,
    parse_coords,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_image,
)
from poc.work2.vlm_client import Work2VLMClient

load_dotenv()

WINDOW_TITLE_PREFIX = "Remote Control System"
LOGIN_SERVICE_SLUG = "ui-venus"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
OPEN_RCS_STATE_PATH = Path(__file__).parent / "logs" / "open_rcs_state.json"
OPEN_RCS_SCRIPT_PATH = Path(__file__).parent / "open_rcs.py"
LOG_NAME = Path(__file__).stem
INPUT_BUTTON_TARGETS = [
    "server_input",
    "userid_input",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
]
ELEMENT_COLORS = {
    "server_input": "salmon",
    "userid_input": "deepskyblue",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}
DESKTOP_SCAN_BACKENDS = ("uia", "win32")

EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_VLM_REQUEST_ERROR = "vlm_request_error"
EXIT_VLM_PARSE_ERROR = "vlm_parse_error"
EXIT_CAPTURE_FAILED = "capture_failed"

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0


def _load_open_rcs_pid() -> int | None:
    """open_rcs 가 남긴 상태 파일에서 PID 를 읽는다."""
    if not OPEN_RCS_STATE_PATH.exists():
        print(f"[INFO] open_rcs 상태 파일 없음: {OPEN_RCS_STATE_PATH}")
        return None

    try:
        data = json.loads(OPEN_RCS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[INFO] open_rcs 상태 파일 파싱 실패: {exc}")
        return None

    pid = data.get("pid")
    if isinstance(pid, int) and pid > 0:
        print(
            "[INFO] open_rcs 상태 파일 사용 "
            f"pid={pid}, status={data.get('status')}, path={OPEN_RCS_STATE_PATH}"
        )
        return pid

    print(f"[INFO] open_rcs 상태 파일 PID 없음: {OPEN_RCS_STATE_PATH}")
    return None


def _run_open_rcs_fallback() -> None:
    """open_rcs.py 를 실행해 PID 상태 파일 생성을 재시도한다."""
    command = [sys.executable, str(OPEN_RCS_SCRIPT_PATH)]
    print(f"[INFO] open_rcs fallback 실행: {command!r}")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"[INFO] open_rcs fallback 실행 실패: {exc}")
        return

    print(
        "[INFO] open_rcs fallback 종료 "
        f"returncode={result.returncode}"
    )
    if result.stdout.strip():
        print(f"[INFO] open_rcs stdout:\n{result.stdout.strip()}\n")
    if result.stderr.strip():
        print(f"[INFO] open_rcs stderr:\n{result.stderr.strip()}\n")


def _find_login_window() -> tuple[object | None, str, str]:
    """PID 힌트 우선, 실패 시 title scan 으로 로그인 창을 찾는다."""
    launch_pid = _load_open_rcs_pid()
    if launch_pid is None:
        _run_open_rcs_fallback()
        launch_pid = _load_open_rcs_pid()

    if launch_pid is not None:
        login_window, window_title, backend = find_window_by_pid_and_title_prefix(
            launch_pid,
            WINDOW_TITLE_PREFIX,
            DESKTOP_SCAN_BACKENDS,
        )
        if login_window is not None:
            return login_window, window_title, backend

    login_window, window_title, backend = find_window_by_title_prefix(
        WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=True,
    )
    if login_window is not None:
        return login_window, window_title, backend

    return find_window_by_title_prefix(
        WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=False,
    )


def _locate_login_controls(login_window, window_title: str, backend: str) -> str:
    """로그인 창 스크린샷을 VLM 으로 분석하고 overlay 를 저장한다."""
    locate_started_at = time.time()
    if not activate_window(
        login_window,
        debug_label=f"login_window recapture backend={backend} title={window_title!r}",
    ):
        print(
            f"[ERROR] 로그인 창 재활성화 실패: title={window_title!r}, backend={backend}"
        )
        log_work2_event(
            component="login_rcs",
            message="login_window_reactivate_failed",
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
            component="login_rcs",
            message="capture_failed",
            level="error",
            log_name=LOG_NAME,
            window_title=window_title,
            backend=backend,
            error=exc,
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
        )
        return EXIT_CAPTURE_FAILED
    client = Work2VLMClient(service_slug=LOGIN_SERVICE_SLUG, log_name=LOG_NAME)

    raw_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_capture.jpg",
        model_name=client.model_name,
    )
    save_debug_jpeg(image, raw_path, log_name=LOG_NAME)

    vlm_input_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_vlm_input.webp",
        model_name=client.model_name,
    )
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)

    image_b64, width, height = encode_image_webp(image)
    system_message, user_text = build_rcs_login_locator_prompt(
        width=width,
        height=height,
        target_keys=INPUT_BUTTON_TARGETS,
    )

    print(
        f"[INFO] 로그인 창 분석 시작: backend={backend}, title={window_title!r}, "
        f"size={width}x{height}"
    )
    print(
        f"[INFO] VLM 요청: service={client.service_slug}, "
        f"model={client.model_name}, endpoint={client.endpoint}"
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
            component="login_rcs",
            message="vlm_request_failed",
            level="error",
            log_name=LOG_NAME,
            service=client.service_slug,
            error=exc,
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
        )
        return EXIT_VLM_REQUEST_ERROR

    print(f"[INFO] VLM 응답 수신: tokens={response.token_usage or {}}")
    print(f"[INFO] 원문 응답:\n{response.text}\n")

    try:
        data = extract_json(response.text)
    except Exception as exc:
        print(f"[ERROR] VLM 응답 JSON 파싱 실패: {exc}")
        log_work2_event(
            component="login_rcs",
            message="vlm_json_parse_failed",
            level="error",
            log_name=LOG_NAME,
            service=client.service_slug,
            raw_text=response.text[:500],
            error=exc,
            elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
        )
        return EXIT_VLM_PARSE_ERROR

    print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
    parsed = parse_coords(data, INPUT_BUTTON_TARGETS, width, height)

    detected = sum(
        1 for key in INPUT_BUTTON_TARGETS if key in parsed and isinstance(parsed[key], dict)
    )
    print(f"[INFO] 검출 결과: {detected}/{len(INPUT_BUTTON_TARGETS)}")

    overlay_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_overlay.jpg",
        model_name=response.model_name or client.model_name,
    )
    save_marked_image(image, parsed, ELEMENT_COLORS, overlay_path)
    print(
        "[INFO] 디버그 이미지: "
        f"capture={raw_path}, vlm_input={vlm_input_path}, overlay={overlay_path}"
    )
    print(f"[INFO] 로그인 이미지 분석 전체 소요: {format_elapsed_ms(locate_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="analysis_finished",
        log_name=LOG_NAME,
        service=client.service_slug,
        model=response.model_name or client.model_name,
        backend=backend,
        window_title=window_title,
        capture_path=raw_path,
        vlm_input_path=vlm_input_path,
        overlay_path=overlay_path,
        detected=detected,
        target_count=len(INPUT_BUTTON_TARGETS),
        elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS if detected > 0 else EXIT_VLM_NO_DETECTION


def main() -> str:
    """이미 열려 있는 로그인 창을 읽고 debug image 를 생성한다."""
    script_started_at = time.time()
    log_work2_event(
        component="login_rcs",
        message="script_started",
        log_name=LOG_NAME,
        desktop_backends=",".join(DESKTOP_SCAN_BACKENDS),
    )

    login_window, window_title, backend = _find_login_window()
    if login_window is None:
        print(
            "[ERROR] 이미 떠 있는 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        log_work2_event(
            component="login_rcs",
            message="login_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=WINDOW_TITLE_PREFIX,
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    if not activate_window(
        login_window,
        debug_label=f"login_window backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 활성화 실패: title={window_title!r}, backend={backend}")
        log_work2_event(
            component="login_rcs",
            message="login_window_activate_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
        )
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    result = _locate_login_controls(login_window, window_title, backend)
    print(f"[INFO] login_rcs end-to-end 소요: {format_elapsed_ms(script_started_at)}")
    log_work2_event(
        component="login_rcs",
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
