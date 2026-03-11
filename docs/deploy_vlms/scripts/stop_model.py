"""실행 중인 vLLM 모델 프로세스를 종료한다.

사용법:
  python stop_model.py                # 실행 중인 모든 vLLM 인스턴스 표시
  python stop_model.py <instance>     # 특정 인스턴스 종료 (포트 기준)
  python stop_model.py all            # 모든 vLLM 인스턴스 종료
  python stop_model.py <family> <size>

예시:
  python stop_model.py ui-venus
  python stop_model.py ui-venus 30b
  python stop_model.py mai-ui
  python stop_model.py all

동작:
  - 인스턴스 이름으로 config/models/<instance>.env의 PORT를 찾는다.
  - 해당 포트를 사용 중인 프로세스 또는 vllm 프로세스를 SIGTERM으로 종료한다.
  - SIGTERM 후 일정 시간 내 종료되지 않으면 SIGKILL을 보낸다.
"""

import os
import signal
import sys
import time
from pathlib import Path


VLLM_CMDLINE_MARKER = "vllm.entrypoints.openai.api_server"
SIGTERM_WAIT_SEC = 10


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def read_env_value(path: Path, key: str) -> str:
    """env 파일에서 특정 키 값을 읽는다."""
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, _, value = line.partition("=")
            if current_key.strip() != key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value
    return ""


def resolve_config_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or str(script_dir.parent)
    config_root = os.environ.get("CONFIG_ROOT", "").strip() or os.path.join(deploy_vlms_root, "config")
    return Path(config_root)


def resolve_port_for_instance(instance: str) -> str:
    """인스턴스 env 파일에서 PORT 값을 가져온다."""
    config_root = resolve_config_root()
    env_path = config_root / "models" / f"{instance}.env"
    if not env_path.is_file():
        fail(f"Instance config not found: {env_path}")
    port = read_env_value(env_path, "PORT")
    if not port:
        fail(f"PORT not defined in {env_path}")
    return port


def resolve_served_model_name(instance: str) -> str:
    config_root = resolve_config_root()
    env_path = config_root / "models" / f"{instance}.env"
    return read_env_value(env_path, "SERVED_MODEL_NAME")


def find_vllm_processes() -> list[dict]:
    """실행 중인 vLLM 프로세스를 /proc에서 찾는다."""
    results = []
    proc_path = Path("/proc")

    if not proc_path.is_dir():
        # /proc가 없는 경우 (macOS 등) ps 명령 사용
        return _find_vllm_processes_via_ps()

    for pid_dir in proc_path.iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        cmdline_path = pid_dir / "cmdline"
        try:
            cmdline_raw = cmdline_path.read_bytes()
        except (PermissionError, FileNotFoundError, OSError):
            continue
        if not cmdline_raw:
            continue
        cmdline_parts = cmdline_raw.decode("utf-8", errors="replace").split("\0")
        cmdline = " ".join(cmdline_parts)
        if VLLM_CMDLINE_MARKER in cmdline:
            port = _extract_port_from_cmdline(cmdline_parts)
            served_name = _extract_served_name_from_cmdline(cmdline_parts)
            results.append({
                "pid": pid,
                "port": port,
                "served_model_name": served_name,
                "cmdline": cmdline.strip(),
            })
    return results


def _find_vllm_processes_via_ps() -> list[dict]:
    """ps 명령으로 vLLM 프로세스를 찾는다 (/proc가 없는 환경용)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    results = []
    for line in result.stdout.splitlines():
        if VLLM_CMDLINE_MARKER not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        cmdline_parts = parts[10:]  # ps aux: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND...
        port = _extract_port_from_cmdline(cmdline_parts)
        served_name = _extract_served_name_from_cmdline(cmdline_parts)
        results.append({
            "pid": pid,
            "port": port,
            "served_model_name": served_name,
            "cmdline": " ".join(cmdline_parts).strip(),
        })
    return results


def _extract_port_from_cmdline(parts: list[str]) -> str:
    for i, part in enumerate(parts):
        if part == "--port" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def _extract_served_name_from_cmdline(parts: list[str]) -> str:
    for i, part in enumerate(parts):
        if part == "--served-model-name" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def kill_process(pid: int, sigterm_wait: float = SIGTERM_WAIT_SEC) -> bool:
    """프로세스를 SIGTERM으로 종료 시도 후, 실패 시 SIGKILL."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        warn(f"PID {pid} already exited")
        return True
    except PermissionError:
        fail(f"Permission denied for PID {pid}. Try running with sudo.")

    log(f"Sending SIGTERM to PID {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log(f"PID {pid} exited before SIGTERM")
        return True
    except PermissionError:
        fail(f"Permission denied for PID {pid}. Try running with sudo.")

    # SIGTERM 후 대기
    deadline = time.monotonic() + sigterm_wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"PID {pid} exited after SIGTERM")
            return True
        time.sleep(0.5)

    # 아직 살아있으면 SIGKILL
    warn(f"PID {pid} did not exit after {sigterm_wait}s, sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        log(f"PID {pid} exited before SIGKILL")
        return True
    except PermissionError:
        fail(f"Permission denied for SIGKILL on PID {pid}.")

    time.sleep(1)
    try:
        os.kill(pid, 0)
        warn(f"PID {pid} still alive after SIGKILL")
        return False
    except ProcessLookupError:
        log(f"PID {pid} exited after SIGKILL")
        return True


def list_running(processes: list[dict]) -> None:
    """실행 중인 vLLM 인스턴스를 표시한다."""
    if not processes:
        log("No running vLLM instances found.")
        return

    log(f"Found {len(processes)} running vLLM instance(s):")
    for proc in processes:
        parts = [f"  PID={proc['pid']}"]
        if proc["port"]:
            parts.append(f"PORT={proc['port']}")
        if proc["served_model_name"]:
            parts.append(f"MODEL={proc['served_model_name']}")
        print(" ".join(parts))


def stop_by_port(processes: list[dict], port: str, instance: str) -> None:
    """특정 포트의 vLLM 프로세스를 종료한다."""
    targets = [p for p in processes if p["port"] == port]
    if not targets:
        # 포트 매칭 실패 시 served_model_name으로도 찾기
        served_name = resolve_served_model_name(instance)
        if served_name:
            targets = [p for p in processes if p["served_model_name"] == served_name]

    if not targets:
        log(f"No running vLLM process found for instance={instance} (port={port})")
        list_running(processes)
        return

    for proc in targets:
        log(
            f"Stopping instance={instance}: "
            f"PID={proc['pid']} PORT={proc['port']} MODEL={proc['served_model_name']}"
        )
        kill_process(proc["pid"])


def stop_all(processes: list[dict]) -> None:
    """모든 vLLM 프로세스를 종료한다."""
    if not processes:
        log("No running vLLM instances to stop.")
        return

    log(f"Stopping all {len(processes)} vLLM instance(s)...")
    for proc in processes:
        parts = [f"PID={proc['pid']}"]
        if proc["port"]:
            parts.append(f"PORT={proc['port']}")
        if proc["served_model_name"]:
            parts.append(f"MODEL={proc['served_model_name']}")
        log(f"Stopping {' '.join(parts)}")
        kill_process(proc["pid"])


def main() -> None:
    processes = find_vllm_processes()

    if len(sys.argv) < 2:
        list_running(processes)
        if processes:
            print()
            print("Usage:")
            print("  python stop_model.py <instance>   # stop specific instance")
            print("  python stop_model.py all           # stop all instances")
        return

    # "all" 처리
    if len(sys.argv) == 2 and normalize_token(sys.argv[1]) == "all":
        stop_all(processes)
        return

    # 인스턴스 이름 해석
    if len(sys.argv) == 2:
        instance = normalize_token(sys.argv[1])
    elif len(sys.argv) == 3:
        instance = f"{normalize_token(sys.argv[1])}-{normalize_token(sys.argv[2])}"
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    port = resolve_port_for_instance(instance)
    log(f"Instance={instance} -> PORT={port}")
    stop_by_port(processes, port, instance)


if __name__ == "__main__":
    main()
