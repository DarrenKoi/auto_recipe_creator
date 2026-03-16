"""RCS 프로그램을 실행하는 스크립트.

RcsMainHD.exe 를 시작하고 로그인 창이 나타날 때까지 대기한다.
VLM 분석 없이 단순 실행만 수행한다.

사용법:
  uv run python poc/work2/open_rcs.py
"""

import os
import struct
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.rcs_utils import (
    find_existing_main_window,
    format_elapsed_ms,
    is_main_window_title,
    launch_application,
    wait_for_window_by_title_prefix,
)
from poc.work2.logger import log_work2_event

load_dotenv()

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "uia"
MAIN_WINDOW_TITLE_REGEX = (
    os.environ.get("RCS_MAIN_WINDOW_REGEX", r"\brcs\b.*\[server\s*:[^\]]+\]").strip()
    or r"\brcs\b.*\[server\s*:[^\]]+\]"
)
_desktop_backends_raw = [
    item.strip().lower()
    for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "win32,uia").split(",")
    if item.strip()
]
_desktop_backends = _desktop_backends_raw + [PYWINAUTO_BACKEND]
DESKTOP_SCAN_BACKENDS = tuple(
    dict.fromkeys(item for item in _desktop_backends if item in {"uia", "win32"})
) or ("uia",)

LAUNCH_TIMEOUT = 10.0
WINDOW_TITLE_PREFIX = "Remote Control System"
LOG_NAME = Path(__file__).stem

EXIT_SUCCESS = "success"
EXIT_EXE_NOT_FOUND = "exe_not_found"
EXIT_ALREADY_LOGGED_IN = "already_logged_in"
EXIT_LAUNCH_FAILED = "launch_failed"


def _python_bitness() -> int:
    """현재 Python 인터프리터의 비트 수를 반환한다."""
    return 64 if sys.maxsize > 2**32 else 32


def _exe_bitness(exe_path: Path) -> int | None:
    """PE 헤더를 읽어 실행 파일 비트 수를 판별한다."""
    try:
        with exe_path.open("rb") as fp:
            if fp.read(2) != b"MZ":
                return None
            fp.seek(0x3C)
            e_lfanew = struct.unpack("<I", fp.read(4))[0]
            fp.seek(e_lfanew + 4)
            machine = struct.unpack("<H", fp.read(2))[0]
    except OSError:
        return None

    if machine == 0x8664:
        return 64
    if machine == 0x14C:
        return 32
    return None


def _resolve_backend(exe_path: Path) -> str:
    """혼합 비트 환경에서 32/64비트 호환성이 높은 백엔드를 선택한다."""
    backend = PYWINAUTO_BACKEND
    if backend == "uia":
        return "uia"
    exe_bits = _exe_bitness(exe_path)
    py_bits = _python_bitness()

    if exe_bits and exe_bits != py_bits and backend == "win32":
        print(
            f"[INFO] 비트 수 불일치 감지 (Python={py_bits}-bit, RCS EXE={exe_bits}-bit). "
            "win32 대신 uia 백엔드를 사용합니다."
        )
        return "uia"

    return backend


def main() -> str:
    """RCS 프로그램을 실행하고 로그인 창 출현까지 대기한다."""
    script_started_at = time.time()
    log_work2_event(
        component="open_rcs",
        message="script_started",
        log_name=LOG_NAME,
        exe_path=RCS_EXE,
        backend_default=PYWINAUTO_BACKEND,
        desktop_backends=",".join(DESKTOP_SCAN_BACKENDS),
    )

    # 이미 로그인된 메인 창이 있는지 점검
    existing_window, existing_title, _ = find_existing_main_window(
        DESKTOP_SCAN_BACKENDS,
        lambda title: is_main_window_title(title, MAIN_WINDOW_TITLE_REGEX),
    )
    if existing_window is not None:
        print(f"[WARNING] 이미 로그인된 RCS 메인 창이 떠 있습니다: '{existing_title}'")
        return EXIT_ALREADY_LOGGED_IN

    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return EXIT_EXE_NOT_FOUND

    backend = _resolve_backend(RCS_EXE)
    print(f"[INFO] RCS 시작: {RCS_EXE}")
    print(f"[INFO] pywinauto 백엔드: {backend}")

    launch_started_at = time.time()
    try:
        app = launch_application(RCS_EXE, backend, wait_for_idle=False, log_name=LOG_NAME)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        log_work2_event(
            component="open_rcs",
            message="launch_failed",
            level="error",
            log_name=LOG_NAME,
            error=exc,
        )
        return EXIT_LAUNCH_FAILED
    print(
        "[INFO] RCS 프로세스 spawn/connect 소요 "
        f"(wait_for_idle=False): {format_elapsed_ms(launch_started_at)}"
    )

    # 로그인 창 대기
    wait_started_at = time.time()
    try:
        login_window = wait_for_window_by_title_prefix(
            app,
            DESKTOP_SCAN_BACKENDS,
            WINDOW_TITLE_PREFIX,
            LAUNCH_TIMEOUT,
            log_name=LOG_NAME,
        )
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        log_work2_event(
            component="open_rcs",
            message="login_window_wait_failed",
            level="error",
            log_name=LOG_NAME,
            error=exc,
            elapsed_ms=f"{(time.time() - wait_started_at) * 1000:.1f}",
        )
        return EXIT_LAUNCH_FAILED
    print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    print(f"[INFO] 로그인 창 준비 소요: {format_elapsed_ms(wait_started_at)}")
    print(f"[INFO] open_rcs end-to-end 소요: {format_elapsed_ms(script_started_at)}")

    log_work2_event(
        component="open_rcs",
        message="script_finished",
        log_name=LOG_NAME,
        result=EXIT_SUCCESS,
        window_title=login_window.window_text(),
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
