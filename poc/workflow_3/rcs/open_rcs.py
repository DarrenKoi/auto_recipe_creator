"""RCS 프로그램 실행 전용 스크립트.

`RcsMainHD.exe` 만 빠르게 실행한다.
로그인 창 탐색이나 후속 자동화 단계는 여기서 처리하지 않는다.

사용법:
  uv run python poc/workflow_3/rcs/open_rcs.py
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from poc.workflow_3 import LOG_DIR
from poc.workflow_3.logger import log_work2_event

if DOTENV_AVAILABLE:
    load_dotenv()

RCS_EXE = Path(
    os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe")
)
LOG_NAME = Path(__file__).stem
OPEN_ANOTHER_RCS_PROCESS = 0
OPEN_RCS_STATE_PATH = LOG_DIR / "open_rcs_state.json"

EXIT_SUCCESS = "success"
EXIT_EXE_NOT_FOUND = "exe_not_found"
EXIT_LAUNCH_FAILED = "launch_failed"
EXIT_ALREADY_OPEN = "already_open"
EXIT_EARLY_CRASH = "early_crash"

EARLY_CRASH_WAIT_SEC = 0.5


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


def write_open_rcs_state(pid: int, status: str) -> None:
    """후속 스크립트가 재연결할 수 있도록 RCS 프로세스 정보를 저장한다."""
    OPEN_RCS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": int(pid),
        "status": status,
        "exe_path": str(RCS_EXE),
        "written_at": time.time(),
    }
    OPEN_RCS_STATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    info(f"state file updated: {OPEN_RCS_STATE_PATH}")


def _normalize_path_text(path_text: str | None) -> str:
    """경로 비교를 위해 소문자 기준 문자열로 정규화한다."""
    if not path_text:
        return ""
    return str(Path(path_text)).replace("\\", "/").lower()


def find_existing_rcs_processes(exe_path: Path) -> list[dict[str, str | int]]:
    """이미 실행 중인 RCS 프로세스를 빠르게 찾는다."""
    if not PSUTIL_AVAILABLE:
        info("process scan skipped: psutil unavailable")
        return []

    exe_name = exe_path.name.lower()
    exe_path_text = _normalize_path_text(str(exe_path))
    matches: list[dict[str, str | int]] = []
    scan_started_at = time.time()
    info("process scan backend=psutil")
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            pid = proc.info.get("pid")
            name = (proc.info.get("name") or "").strip()
            running_exe = (proc.info.get("exe") or "").strip()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

        name_match = name.lower() == exe_name
        exe_match = _normalize_path_text(running_exe) == exe_path_text
        if not name_match and not exe_match:
            continue

        matches.append(
            {
                "pid": int(pid) if pid is not None else -1,
                "name": name,
                "exe": running_exe,
            }
        )

    info(
        f"process scan done: backend=psutil count={len(matches)} elapsed={format_elapsed_ms(scan_started_at)}"
    )
    return matches


def launch_rcs(exe_path: Path) -> subprocess.Popen:
    """RCS 실행 파일만 빠르게 시작한다."""
    launch_started_at = time.time()
    work_dir = str(exe_path.parent)
    command = [str(exe_path)]

    popen_kwargs: dict = {"cwd": work_dir}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

    info(f"cwd={work_dir}")
    info(f"command={command!r}")
    info(f"creationflags={popen_kwargs.get('creationflags', 'N/A (non-nt)')}")

    try:
        process = subprocess.Popen(command, **popen_kwargs)
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
    info(f"OPEN_ANOTHER_RCS_PROCESS={OPEN_ANOTHER_RCS_PROCESS}")
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
    info("exe confirmed")

    existing_processes = find_existing_rcs_processes(RCS_EXE)
    if existing_processes:
        info(f"existing_rcs_processes={existing_processes}")
    else:
        info("existing_rcs_processes=[]")

    if existing_processes and OPEN_ANOTHER_RCS_PROCESS == 0:
        existing_pid = int(existing_processes[0]["pid"])
        info(f"기존 RCS 프로세스가 이미 실행 중이므로 새로 열지 않습니다: pid={existing_pid}")
        write_open_rcs_state(existing_pid, EXIT_ALREADY_OPEN)
        log_work2_event(
            component="open_rcs",
            message="launch_skipped_already_open",
            log_name=LOG_NAME,
            exe_path=RCS_EXE,
            existing_processes=existing_processes,
        )
        log_work2_event(
            component="open_rcs",
            message="script_finished",
            log_name=LOG_NAME,
            result=EXIT_ALREADY_OPEN,
            exe_path=RCS_EXE,
            pid=existing_pid,
            elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
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

    time.sleep(EARLY_CRASH_WAIT_SEC)
    exit_code = process.poll()
    if exit_code is not None:
        error(f"프로세스가 즉시 종료됨: pid={process.pid} exit_code={exit_code}")
        log_work2_event(
            component="open_rcs",
            message="early_crash",
            level="error",
            log_name=LOG_NAME,
            exe_path=RCS_EXE,
            pid=process.pid,
            exit_code=exit_code,
        )
        return EXIT_EARLY_CRASH

    info(f"RCS 실행 요청 완료: pid={process.pid}")
    write_open_rcs_state(process.pid, EXIT_SUCCESS)
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
    if exit_result not in {EXIT_SUCCESS, EXIT_ALREADY_OPEN}:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
