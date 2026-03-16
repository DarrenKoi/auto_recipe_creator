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

import psutil

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
    foreground_window,
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
LOGIN_WINDOW_MAX_WIDTH = int(os.getenv("RCS_LOGIN_WINDOW_MAX_WIDTH", "900"))
LOGIN_WINDOW_MAX_HEIGHT = int(os.getenv("RCS_LOGIN_WINDOW_MAX_HEIGHT", "700"))
LOGIN_WINDOW_MAX_AREA = int(os.getenv("RCS_LOGIN_WINDOW_MAX_AREA", "500000"))
EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_LOGIN_WINDOW_SIZE_MISMATCH = "login_window_size_mismatch"
EXIT_VLM_NO_DETECTION = "vlm_no_detection"
EXIT_VLM_REQUEST_ERROR = "vlm_request_error"
EXIT_VLM_PARSE_ERROR = "vlm_parse_error"
EXIT_CAPTURE_FAILED = "capture_failed"

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0


def _read_window_size(window) -> tuple[int, int, int] | None:
    """창 크기를 width, height, area로 반환한다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[INFO] 로그인 창 크기 조회 실패: {exc}")
        return None

    width = max(0, int(rect.right - rect.left))
    height = max(0, int(rect.bottom - rect.top))
    return width, height, width * height


def _is_probable_login_dialog(window, window_title: str) -> bool:
    """선택된 창이 로그인 대화상자 크기 범위인지 점검한다."""
    size_info = _read_window_size(window)
    if size_info is None:
        return False

    width, height, area = size_info
    print(
        "[INFO] 로그인 창 크기 점검 "
        f"title={window_title!r}, size={width}x{height}, area={area}, "
        f"limits={LOGIN_WINDOW_MAX_WIDTH}x{LOGIN_WINDOW_MAX_HEIGHT}, "
        f"max_area={LOGIN_WINDOW_MAX_AREA}"
    )
    return (
        width > 0
        and height > 0
        and width <= LOGIN_WINDOW_MAX_WIDTH
        and height <= LOGIN_WINDOW_MAX_HEIGHT
        and area <= LOGIN_WINDOW_MAX_AREA
    )


def _login_window_filter(window, window_title: str) -> bool:
    """로그인 창 후보 필터. 작은 로그인 대화상자만 통과시킨다."""
    size_info = _read_window_size(window)
    if size_info is None:
        return False

    width, height, area = size_info
    is_match = (
        width > 0
        and height > 0
        and width <= LOGIN_WINDOW_MAX_WIDTH
        and height <= LOGIN_WINDOW_MAX_HEIGHT
        and area <= LOGIN_WINDOW_MAX_AREA
    )
    print(
        "[INFO] 로그인 창 후보 점검 "
        f"title={window_title!r}, size={width}x{height}, area={area}, match={is_match}"
    )
    return is_match


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


def _load_open_rcs_exe_path() -> str:
    """open_rcs 상태 파일에서 기대하는 실행 파일 경로를 읽는다."""
    if not OPEN_RCS_STATE_PATH.exists():
        return ""

    try:
        data = json.loads(OPEN_RCS_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ""

    exe_path = data.get("exe_path")
    if isinstance(exe_path, str):
        return exe_path.strip()
    return ""


def _normalize_path_text(path_text: str | None) -> str:
    """경로 비교를 위해 소문자 기준 문자열로 정규화한다."""
    if not path_text:
        return ""
    return str(path_text).replace("\\", "/").lower()


def _is_pid_alive(pid: int, expected_exe_path: str = "") -> bool:
    """PID 가 살아 있고, 필요하면 기대한 RCS 실행 파일과도 일치하는지 확인한다."""
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False

        expected_path = _normalize_path_text(expected_exe_path)
        if not expected_path:
            return True

        try:
            running_exe = proc.exe()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            running_exe = ""

        normalized_running_exe = _normalize_path_text(running_exe)
        if normalized_running_exe:
            is_match = normalized_running_exe == expected_path
            print(
                "[INFO] PID 실행 파일 점검 "
                f"pid={pid}, expected={expected_exe_path!r}, "
                f"running={running_exe!r}, match={is_match}"
            )
            return is_match

        expected_name = Path(expected_exe_path).name.strip().lower()
        try:
            running_name = (proc.name() or "").strip().lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            running_name = ""

        if running_name:
            is_match = running_name == expected_name
            print(
                "[INFO] PID 실행 파일명 점검 "
                f"pid={pid}, expected_name={expected_name!r}, "
                f"running_name={running_name!r}, match={is_match}"
            )
            return is_match

        print(f"[INFO] PID 실행 파일 점검 불가: pid={pid}, expected={expected_exe_path!r}")
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _run_open_rcs_fallback() -> None:
    """open_rcs.py 를 실행해 PID 상태 파일 생성을 재시도한다."""
    command = [sys.executable, str(OPEN_RCS_SCRIPT_PATH)]
    print(f"[INFO] open_rcs fallback 실행: {command!r}")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        print(f"[INFO] open_rcs fallback 실행 실패: {exc}")
        return

    print(
        "[INFO] open_rcs fallback 종료 "
        f"returncode={result.returncode}"
    )
    stdout_text = (result.stdout or "").strip()
    stderr_text = (result.stderr or "").strip()
    if stdout_text:
        print(f"[INFO] open_rcs stdout:\n{stdout_text}\n")
    if stderr_text:
        print(f"[INFO] open_rcs stderr:\n{stderr_text}\n")


def _ensure_rcs_running() -> int | None:
    """RCS 프로세스가 실행 중인지 확인하고, 없으면 open_rcs fallback 을 실행한다.

    Returns:
        살아있는 RCS PID, 또는 확보 실패 시 None.
    """
    launch_pid = _load_open_rcs_pid()
    expected_exe_path = _load_open_rcs_exe_path()

    # 상태 파일 자체가 없거나 PID 가 기록되어 있지 않음
    if launch_pid is None:
        print("[INFO] PID 없음 → open_rcs fallback 실행")
        _run_open_rcs_fallback()
        launch_pid = _load_open_rcs_pid()
        expected_exe_path = _load_open_rcs_exe_path()
    # 상태 파일에 PID 는 있지만 프로세스가 죽어 있음
    elif not _is_pid_alive(launch_pid, expected_exe_path):
        print(
            f"[INFO] PID {launch_pid} 프로세스 검증 실패 → open_rcs fallback 실행"
        )
        _run_open_rcs_fallback()
        launch_pid = _load_open_rcs_pid()
        expected_exe_path = _load_open_rcs_exe_path()

    # fallback 후에도 PID 확보 여부 + 생존 여부 최종 확인
    if launch_pid is None:
        print("[ERROR] open_rcs fallback 후에도 PID 확보 실패")
        return None

    alive = _is_pid_alive(launch_pid, expected_exe_path)
    print(
        f"[INFO] RCS PID 최종 확인: pid={launch_pid}, alive={alive}, "
        f"expected_exe_path={expected_exe_path!r}"
    )
    if not alive:
        print(f"[ERROR] PID {launch_pid} 가 여전히 실행 중이지 않음 — 창 탐색 불가")
        return None

    return launch_pid


def _find_login_window() -> tuple[object | None, str, str]:
    """RCS 프로세스 생존 확인 후 로그인 대화상자를 탐색한다."""
    launch_pid = _ensure_rcs_running()
    if launch_pid is None:
        return None, "", ""

    print(f"[INFO] 로그인 창 탐색 시작: PID 우선 scan (rcs pid={launch_pid})")
    login_window, window_title, backend = find_window_by_pid_and_title_prefix(
        launch_pid,
        WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        window_filter=_login_window_filter,
    )
    if login_window is not None:
        print(f"[INFO] 로그인 창 발견 (PID 우선) → 포커스 활성화: title={window_title!r}")
        activate_window(
            login_window,
            debug_label=f"login_window_found_pid_first backend={backend} title={window_title!r}",
        )
        return login_window, window_title, backend

    print(f"[INFO] 로그인 창 탐색 계속: desktop all scan (rcs pid={launch_pid})")
    login_window, window_title, backend = find_window_by_title_prefix(
        WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=False,
        window_filter=_login_window_filter,
    )
    if login_window is not None:
        print(f"[INFO] 로그인 창 발견 (desktop scan) → 포커스 활성화: title={window_title!r}")
        activate_window(
            login_window,
            debug_label=f"login_window_found_desktop backend={backend} title={window_title!r}",
        )
        return login_window, window_title, backend

    return None, "", ""


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

    if not foreground_window(
        login_window,
        debug_label=f"login_window screenshot backend={backend} title={window_title!r}",
    ):
        print(
            f"[ERROR] 로그인 창 foreground 활성화 실패: "
            f"title={window_title!r}, backend={backend}"
        )
        log_work2_event(
            component="login_rcs",
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

    if not _is_probable_login_dialog(login_window, window_title):
        print(
            "[ERROR] 선택된 창이 로그인 대화상자 크기 범위를 벗어났습니다. "
            "로그인 후 메인 창일 가능성이 큽니다."
        )
        log_work2_event(
            component="login_rcs",
            message="login_window_size_mismatch",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            max_width=LOGIN_WINDOW_MAX_WIDTH,
            max_height=LOGIN_WINDOW_MAX_HEIGHT,
            max_area=LOGIN_WINDOW_MAX_AREA,
        )
        return EXIT_LOGIN_WINDOW_SIZE_MISMATCH

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
