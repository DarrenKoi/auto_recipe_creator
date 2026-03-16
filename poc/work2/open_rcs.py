"""RCS 프로그램 실행 전용 스크립트.

`RcsMainHD.exe` 만 빠르게 실행한다.
로그인 창 탐색, pywinauto 연결, rcs_utils 의존성은 사용하지 않는다.

사용법:
  uv run python poc/work2/open_rcs.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event

try:
    from pywinauto import Desktop

    PYWINAUTO_AVAILABLE = True
except Exception:
    Desktop = None
    PYWINAUTO_AVAILABLE = False

load_dotenv()

RCS_EXE = Path(
    os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe")
)
LOG_NAME = Path(__file__).stem
WINDOW_TITLE_PREFIX = "Remote Control System"
MAIN_WINDOW_TITLE_REGEX = (
    os.environ.get("RCS_MAIN_WINDOW_REGEX", r"\brcs\b.*\[server\s*:[^\]]+\]").strip()
    or r"\brcs\b.*\[server\s*:[^\]]+\]"
)
DESKTOP_SCAN_BACKENDS = ("win32", "uia")
OPEN_ANOTHER_LOGIN_WINDOW = 0

EXIT_SUCCESS = "success"
EXIT_EXE_NOT_FOUND = "exe_not_found"
EXIT_LAUNCH_FAILED = "launch_failed"
EXIT_ALREADY_OPEN = "already_open"


def format_elapsed_ms(start_time: float) -> str:
    """start_time 이후 경과 시간을 사람이 읽기 쉬운 문자열로 반환한다."""
    elapsed_ms = (time.time() - start_time) * 1000
    if elapsed_ms < 1000:
        return f"{elapsed_ms:.0f}ms"
    return f"{elapsed_ms / 1000:.2f}s"


def info(message: str) -> None:
    """open_rcs 전용 정보 로그를 출력한다."""
    print(f"[INFO][open_rcs] {message}")


def error(message: str) -> None:
    """open_rcs 전용 에러 로그를 출력한다."""
    print(f"[ERROR][open_rcs] {message}")


def _is_login_window_title(title: str) -> bool:
    """RCS 로그인 창 제목인지 판별한다."""
    normalized = title.strip()
    return normalized.startswith(WINDOW_TITLE_PREFIX)


def _is_main_window_title(title: str) -> bool:
    """RCS 메인 창 제목인지 판별한다."""
    try:
        return re.search(MAIN_WINDOW_TITLE_REGEX, title, flags=re.IGNORECASE) is not None
    except re.error:
        lowered = title.lower()
        return "rcs" in lowered and "[server" in lowered


def scan_existing_rcs_windows() -> list[tuple[str, str]]:
    """현재 떠 있는 RCS 관련 창 제목을 수집한다."""
    if not PYWINAUTO_AVAILABLE:
        info("pywinauto unavailable; window title scan skipped")
        return []

    matches: list[tuple[str, str]] = []
    for backend in DESKTOP_SCAN_BACKENDS:
        scan_started_at = time.time()
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True,
                visible_only=True,
            )
        except Exception as exc:
            info(f"backend={backend} window scan failed: {exc}")
            continue

        info(
            f"backend={backend} window_count={len(windows)} scan_elapsed={format_elapsed_ms(scan_started_at)}"
        )
        for win in windows:
            try:
                title = (win.window_text() or "").strip()
            except Exception as exc:
                info(f"backend={backend} window_text failed: {exc}")
                continue

            if not title:
                continue

            info(f"backend={backend} title={title!r}")

            if _is_login_window_title(title):
                matches.append(("login", title))
                continue

            if _is_main_window_title(title):
                matches.append(("main", title))

    deduped_matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        deduped_matches.append(item)
    return deduped_matches


def launch_rcs(exe_path: Path) -> subprocess.Popen:
    """RCS 실행 파일만 빠르게 시작한다."""
    launch_started_at = time.time()
    work_dir = str(exe_path.parent)
    command = [str(exe_path)]
    creationflags = 0

    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    info(f"cwd={work_dir}")
    info(f"command={command!r}")
    info(f"creationflags={creationflags}")

    try:
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise RuntimeError(f"실행 파일 시작 실패: {exc}") from exc

    info(
        f"프로세스 시작 완료: pid={process.pid}, launch_elapsed={format_elapsed_ms(launch_started_at)}"
    )
    return process


def main() -> str:
    """RCS 실행 파일을 시작한다."""
    script_started_at = time.time()
    info(f"script start: exe_path={RCS_EXE}")
    info(f"OPEN_ANOTHER_LOGIN_WINDOW={OPEN_ANOTHER_LOGIN_WINDOW}")
    log_work2_event(
        component="open_rcs",
        message="script_started",
        log_name=LOG_NAME,
        exe_path=RCS_EXE,
    )

    if not RCS_EXE.exists():
        error(f"실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        log_work2_event(
            component="open_rcs",
            message="exe_not_found",
            level="error",
            log_name=LOG_NAME,
            exe_path=RCS_EXE,
        )
        return EXIT_EXE_NOT_FOUND

    info(f"exe_exists=True size={RCS_EXE.stat().st_size}")

    existing_windows = scan_existing_rcs_windows()
    if existing_windows:
        info(f"existing_rcs_windows={existing_windows}")
    else:
        info("existing_rcs_windows=[]")

    if existing_windows and OPEN_ANOTHER_LOGIN_WINDOW == 0:
        info("기존 RCS 창이 이미 열려 있으므로 새 로그인 창을 열지 않습니다.")
        log_work2_event(
            component="open_rcs",
            message="launch_skipped_already_open",
            log_name=LOG_NAME,
            exe_path=RCS_EXE,
            existing_windows=existing_windows,
        )
        return EXIT_ALREADY_OPEN

    try:
        process = launch_rcs(RCS_EXE)
    except RuntimeError as exc:
        error(str(exc))
        log_work2_event(
            component="open_rcs",
            message="launch_failed",
            level="error",
            log_name=LOG_NAME,
            exe_path=RCS_EXE,
            error=exc,
        )
        return EXIT_LAUNCH_FAILED

    info(f"RCS 실행 요청 완료: pid={process.pid}")
    info(f"open_rcs end-to-end elapsed={format_elapsed_ms(script_started_at)}")
    log_work2_event(
        component="open_rcs",
        message="script_finished",
        log_name=LOG_NAME,
        result=EXIT_SUCCESS,
        exe_path=RCS_EXE,
        pid=process.pid,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
