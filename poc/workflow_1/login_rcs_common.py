"""RCS 로그인 창 탐색 공용 헬퍼."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from poc.workflow_1 import LOG_DIR
from poc.workflow_1 import util as workflow_1_util


WINDOW_TITLE_PREFIX = "Remote Control System"
RCS_MAIN_WINDOW_TITLE_PREFIX = "RCS -"
RCS_UPDATER_WINDOW_TITLE_PREFIX = "RCS Updater"
REMOTE_MONITORING_WINDOW_TITLE_PREFIX = "Remote Monitoring System -"
OPEN_RCS_STATE_PATH = LOG_DIR / "open_rcs_state.json"
OPEN_RCS_SCRIPT_PATH = Path(__file__).parent / "open_rcs.py"
DESKTOP_SCAN_BACKENDS = ("uia", "win32")
LOGIN_WINDOW_MAX_WIDTH = int(os.getenv("RCS_LOGIN_WINDOW_MAX_WIDTH", "900"))
LOGIN_WINDOW_MAX_HEIGHT = int(os.getenv("RCS_LOGIN_WINDOW_MAX_HEIGHT", "700"))
activate_window = getattr(workflow_1_util, "activate_window", None)
find_window_by_pid_and_title_prefix = getattr(
    workflow_1_util,
    "find_window_by_pid_and_title_prefix",
    None,
)
find_window_by_title_prefix = getattr(workflow_1_util, "find_window_by_title_prefix", None)
WINDOW_UTILS_AVAILABLE = all(
    callable(func)
    for func in (
        activate_window,
        find_window_by_pid_and_title_prefix,
        find_window_by_title_prefix,
    )
)


def _read_window_size(window) -> tuple[int, int] | None:
    """창 크기를 width, height로 반환한다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[INFO] 로그인 창 크기 조회 실패: {exc}")
        return None

    width = max(0, int(rect.right - rect.left))
    height = max(0, int(rect.bottom - rect.top))
    return width, height


def _login_window_filter(window, window_title: str) -> bool:
    """로그인 창 후보 필터. 작은 로그인 대화상자만 통과시킨다."""
    size_info = _read_window_size(window)
    if size_info is None:
        return False

    width, height = size_info
    area = width * height
    is_match = (
        width > 0
        and height > 0
        and width <= LOGIN_WINDOW_MAX_WIDTH
        and height <= LOGIN_WINDOW_MAX_HEIGHT
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


def _is_pid_alive(pid: int, expected_exe_path: str = "", *, log_detail: bool = False) -> bool:
    """PID 가 살아 있고, 필요하면 기대한 RCS 실행 파일과도 일치하는지 확인한다."""
    if not PSUTIL_AVAILABLE:
        if log_detail:
            print(f"[INFO] psutil unavailable: pid liveness check skipped (pid={pid})")
        return False

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
            if log_detail:
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
            if log_detail:
                print(
                    "[INFO] PID 실행 파일명 점검 "
                    f"pid={pid}, expected_name={expected_name!r}, "
                    f"running_name={running_name!r}, match={is_match}"
                )
            return is_match

        if log_detail:
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

    print(f"[INFO] open_rcs fallback 종료 returncode={result.returncode}")
    stdout_text = (result.stdout or "").strip()
    stderr_text = (result.stderr or "").strip()
    if stdout_text:
        print(f"[INFO] open_rcs stdout:\n{stdout_text}\n")
    if stderr_text:
        print(f"[INFO] open_rcs stderr:\n{stderr_text}\n")


def _ensure_rcs_running() -> int | None:
    """RCS 프로세스가 실행 중인지 확인하고, 없으면 open_rcs fallback 을 실행한다."""
    launch_pid = _load_open_rcs_pid()
    expected_exe_path = _load_open_rcs_exe_path()

    if launch_pid is None:
        print("[INFO] PID 없음 -> open_rcs fallback 실행")
        _run_open_rcs_fallback()
        launch_pid = _load_open_rcs_pid()
        expected_exe_path = _load_open_rcs_exe_path()
    elif not _is_pid_alive(launch_pid, expected_exe_path):
        print(f"[INFO] PID {launch_pid} 프로세스 검증 실패 -> open_rcs fallback 실행")
        _run_open_rcs_fallback()
        launch_pid = _load_open_rcs_pid()
        expected_exe_path = _load_open_rcs_exe_path()

    if launch_pid is None:
        print("[ERROR] open_rcs fallback 후에도 PID 확보 실패")
        return None

    alive = _is_pid_alive(launch_pid, expected_exe_path)
    print(
        f"[INFO] RCS PID 최종 확인: pid={launch_pid}, alive={alive}, "
        f"expected_exe_path={expected_exe_path!r}"
    )
    if not alive:
        print(f"[ERROR] PID {launch_pid} 가 여전히 실행 중이지 않음 - 창 탐색 불가")
        return None

    return launch_pid


def find_login_window() -> tuple[object | None, str, str]:
    """RCS 프로세스 생존 확인 후 로그인 대화상자를 탐색한다."""
    if not WINDOW_UTILS_AVAILABLE:
        print("[ERROR] window_utils unavailable - 로그인 창 탐색 불가")
        return None, "", ""

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
        print(f"[INFO] 로그인 창 발견 (PID 우선) -> 포커스 활성화: title={window_title!r}")
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
        print(f"[INFO] 로그인 창 발견 (desktop scan) -> 포커스 활성화: title={window_title!r}")
        activate_window(
            login_window,
            debug_label=f"login_window_found_desktop backend={backend} title={window_title!r}",
        )
        return login_window, window_title, backend

    return None, "", ""


def find_rcs_main_window() -> tuple[object | None, str, str]:
    """로그인 후 메인 RCS 창을 탐색한다."""
    if not WINDOW_UTILS_AVAILABLE:
        print("[ERROR] window_utils unavailable - 메인 창 탐색 불가")
        return None, "", ""

    print(f"[INFO] 메인 RCS 창 탐색 시작: title_prefix={RCS_MAIN_WINDOW_TITLE_PREFIX!r}")
    main_window, window_title, backend = find_window_by_title_prefix(
        RCS_MAIN_WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=True,
    )
    if main_window is None:
        return None, "", ""

    print(f"[INFO] 메인 RCS 창 발견 -> 포커스 활성화: title={window_title!r}")
    activate_window(
        main_window,
        debug_label=f"rcs_main_window backend={backend} title={window_title!r}",
    )
    return main_window, window_title, backend


def find_rcs_updater_window() -> tuple[object | None, str, str]:
    """로그인 후 RCS Updater 창을 탐색한다."""
    if not WINDOW_UTILS_AVAILABLE:
        print("[ERROR] window_utils unavailable - updater 창 탐색 불가")
        return None, "", ""

    print(f"[INFO] RCS Updater 창 탐색 시작: title_prefix={RCS_UPDATER_WINDOW_TITLE_PREFIX!r}")
    updater_window, window_title, backend = find_window_by_title_prefix(
        RCS_UPDATER_WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=True,
    )
    if updater_window is None:
        return None, "", ""

    print(f"[INFO] RCS Updater 창 발견 -> 포커스 활성화: title={window_title!r}")
    activate_window(
        updater_window,
        debug_label=f"rcs_updater_window backend={backend} title={window_title!r}",
    )
    return updater_window, window_title, backend


def _tool_window_filter(window, window_title: str, tool_name: str = "") -> bool:
    """원격 모니터링 툴 창 후보 필터."""
    del window
    normalized_title = (window_title or "").strip().lower()
    normalized_tool_name = (tool_name or "").strip().lower()
    if not normalized_title.startswith(REMOTE_MONITORING_WINDOW_TITLE_PREFIX.lower()):
        return False
    if normalized_tool_name and normalized_tool_name not in normalized_title:
        return False
    return True


def find_remote_monitoring_window(tool_name: str = "") -> tuple[object | None, str, str]:
    """툴 더블클릭 후 뜨는 Remote Monitoring System 창을 탐색한다."""
    if not WINDOW_UTILS_AVAILABLE:
        print("[ERROR] window_utils unavailable - Remote Monitoring System 창 탐색 불가")
        return None, "", ""

    print(
        "[INFO] Remote Monitoring System 창 탐색 시작: "
        f"title_prefix={REMOTE_MONITORING_WINDOW_TITLE_PREFIX!r}, tool_name={tool_name!r}"
    )
    tool_window, window_title, backend = find_window_by_title_prefix(
        REMOTE_MONITORING_WINDOW_TITLE_PREFIX,
        DESKTOP_SCAN_BACKENDS,
        visible_only=True,
        window_filter=lambda window, title: _tool_window_filter(window, title, tool_name),
    )
    if tool_window is None:
        return None, "", ""

    print(f"[INFO] Remote Monitoring System 창 발견 -> 포커스 활성화: title={window_title!r}")
    activate_window(
        tool_window,
        debug_label=f"remote_monitoring_window backend={backend} title={window_title!r}",
    )
    return tool_window, window_title, backend


def wait_for_remote_monitoring_window(
    tool_name: str,
    timeout_sec: float = 6.0,
    poll_interval_sec: float = 0.5,
    max_attempts: int | None = None,
) -> tuple[object | None, str, str]:
    """툴 창이 나타날 때까지 폴링한다.

    `max_attempts` 가 주어지면 그 횟수만큼만 탐색하고 중단한다. RCS 가 다른
    사용자에게 점유돼 'select'(공유/종료 선택) 팝업이 떠 tool 창이 안 열리는 경우
    무한정 폴링 스팸을 막기 위한 안전 상한이다(timeout 보다 먼저 도달하면 중단).
    """
    print(
        f"[INFO] Remote Monitoring System 창 대기 시작: "
        f"title_prefix={REMOTE_MONITORING_WINDOW_TITLE_PREFIX!r}, "
        f"tool_name={tool_name!r}, timeout={timeout_sec}s, max_attempts={max_attempts}"
    )
    deadline = time.time() + timeout_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        tool_window, window_title, backend = find_remote_monitoring_window(tool_name)
        if tool_window is not None:
            print(
                f"[INFO] Remote Monitoring System 창 발견 (attempt={attempt}): "
                f"title={window_title!r}, backend={backend}"
            )
            return tool_window, window_title, backend

        if max_attempts is not None and attempt >= max_attempts:
            print(
                f"[WARNING] Remote Monitoring System 창 미발견 — {max_attempts}회 시도 후 중단 "
                f"(tool_name={tool_name!r}). RCS 가 다른 사용자에게 점유됐을 수 있음(select 팝업)."
            )
            return None, "", ""

        remaining_sec = deadline - time.time()
        if remaining_sec <= 0:
            break
        time.sleep(min(poll_interval_sec, remaining_sec))

    print(
        f"[WARNING] Remote Monitoring System 창 타임아웃: "
        f"{timeout_sec}s 내 미발견 (tool_name={tool_name!r}, attempts={attempt})"
    )
    return None, "", ""


def wait_for_rcs_main_window(
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 2.0,
) -> tuple[object | None, str, str]:
    """메인 RCS 창이 나타날 때까지 폴링한다."""
    print(
        f"[INFO] 메인 RCS 창 대기 시작: "
        f"title_prefix={RCS_MAIN_WINDOW_TITLE_PREFIX!r}, timeout={timeout_sec}s"
    )
    deadline = time.time() + timeout_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        main_window, window_title, backend = find_rcs_main_window()
        if main_window is not None:
            print(
                f"[INFO] 메인 RCS 창 발견 (attempt={attempt}): "
                f"title={window_title!r}, backend={backend}"
            )
            return main_window, window_title, backend

        time.sleep(poll_interval_sec)

    print(
        f"[WARNING] 메인 RCS 창 타임아웃: "
        f"{timeout_sec}s 내 미발견 (attempts={attempt})"
    )
    return None, "", ""


def wait_for_rcs_updater_window(
    timeout_sec: float = 15.0,
    poll_interval_sec: float = 2.0,
) -> tuple[object | None, str, str]:
    """RCS Updater 창이 나타날 때까지 폴링한다."""
    print(
        f"[INFO] RCS Updater 창 대기 시작: "
        f"title_prefix={RCS_UPDATER_WINDOW_TITLE_PREFIX!r}, timeout={timeout_sec}s"
    )
    deadline = time.time() + timeout_sec
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        updater_window, window_title, backend = find_rcs_updater_window()
        if updater_window is not None:
            print(
                f"[INFO] RCS Updater 창 발견 (attempt={attempt}): "
                f"title={window_title!r}, backend={backend}"
            )
            return updater_window, window_title, backend

        time.sleep(poll_interval_sec)

    print(
        f"[WARNING] RCS Updater 창 타임아웃: "
        f"{timeout_sec}s 내 미발견 (attempts={attempt})"
    )
    return None, "", ""


__all__ = [
    "DESKTOP_SCAN_BACKENDS",
    "RCS_MAIN_WINDOW_TITLE_PREFIX",
    "REMOTE_MONITORING_WINDOW_TITLE_PREFIX",
    "RCS_UPDATER_WINDOW_TITLE_PREFIX",
    "WINDOW_TITLE_PREFIX",
    "find_login_window",
    "find_rcs_main_window",
    "find_remote_monitoring_window",
    "find_rcs_updater_window",
    "wait_for_rcs_main_window",
    "wait_for_remote_monitoring_window",
    "wait_for_rcs_updater_window",
]
