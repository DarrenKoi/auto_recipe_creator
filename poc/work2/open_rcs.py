"""RCS 프로그램 실행 전용 스크립트.

`RcsMainHD.exe` 만 빠르게 실행한다.
로그인 창 탐색, pywinauto 연결, rcs_utils 의존성은 사용하지 않는다.

사용법:
  uv run python poc/work2/open_rcs.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.logger import log_work2_event

load_dotenv()

RCS_EXE = Path(
    os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe")
)
LOG_NAME = Path(__file__).stem

EXIT_SUCCESS = "success"
EXIT_EXE_NOT_FOUND = "exe_not_found"
EXIT_LAUNCH_FAILED = "launch_failed"


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
