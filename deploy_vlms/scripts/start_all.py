"""등록된 5개 VLM 모델을 모두 시작한다.

GPU 배분 (H200 140GB × 2):
  GPU 0: ui-venus (8B, port 8001) + ui-tars (7B, port 8003)
  GPU 1: mai-ui (8B, port 8002) + paddleocr-vl-1.5 (0.9B, port 8004) + got-ocr (port 8005)

vLLM 모델 4개는 start_model.py를 통해 백그라운드로 실행하고,
GOT-OCR은 serve_got_ocr.py를 별도 프로세스로 실행한다.

사용법:
  python start_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path


STARTUP_POLL_SEC = 2.0
INTER_MODEL_DELAY_SEC = 3.0

VLLM_MODELS = [
    "ui-venus",
    "ui-tars",
    "mai-ui",
    "paddleocr-vl-1.5",
]

GOT_OCR_INSTANCE = "got-ocr-2.0-hf"


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


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


def print_gpu_plan(config_root: Path) -> None:
    """시작 전 GPU 배분 계획을 출력한다."""
    log("=" * 60)
    log("GPU 배분 계획 (H200 140GB × 2)")
    log("=" * 60)

    gpu_map: dict[str, list[str]] = {}

    for instance in VLLM_MODELS:
        env_path = config_root / "models" / f"{instance}.env"
        gpu_id = read_env_value(env_path, "GPU_ID") or "?"
        port = read_env_value(env_path, "PORT") or "?"
        served_name = read_env_value(env_path, "SERVED_MODEL_NAME") or instance
        gpu_mem = read_env_value(env_path, "GPU_MEMORY_UTILIZATION")
        auto_tune = read_env_value(env_path, "AUTO_TUNE_GPU_MEMORY_UTILIZATION")

        mem_info = f"u={gpu_mem}" if gpu_mem else ("auto-tune" if auto_tune else "default")
        entry = f"  {served_name:<25s} port={port:<5s} {mem_info} (vLLM)"
        gpu_map.setdefault(f"GPU {gpu_id}", []).append(entry)

    # GOT-OCR
    got_env_path = config_root / "models" / f"{GOT_OCR_INSTANCE}.env"
    got_gpu_id = read_env_value(got_env_path, "GPU_ID") or "?"
    got_port = read_env_value(got_env_path, "PORT") or "8005"
    entry = f"  {'got-ocr-2.0-hf':<25s} port={got_port:<5s} ~4GiB (transformers)"
    gpu_map.setdefault(f"GPU {got_gpu_id}", []).append(entry)

    for gpu_label in sorted(gpu_map):
        log(f"{gpu_label}:")
        for line in gpu_map[gpu_label]:
            log(line)
    log("=" * 60)


def resolve_runtime_paths(deploy_vlms_root: Path, instance: str) -> tuple[Path, Path]:
    """로그/PID 경로를 반환한다."""
    runtime_root = deploy_vlms_root / "runtime"
    log_dir = runtime_root / "logs"
    pid_dir = runtime_root / "pids"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{instance}.log", pid_dir / f"{instance}.pid"


def start_vllm_model(script_dir: Path, deploy_vlms_root: Path, instance: str) -> bool:
    """vLLM 모델을 백그라운드로 시작한다. 성공 시 True."""
    cmd = [sys.executable, str(script_dir / "serve_vlm.py"), instance]
    log_path, pid_path = resolve_runtime_paths(deploy_vlms_root, instance)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8") as log_file:
        launched_at = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n[INFO] Launch requested at {launched_at} for instance={instance}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(STARTUP_POLL_SEC)
    return_code = proc.poll()
    if return_code is not None:
        pid_path.unlink(missing_ok=True)
        warn(f"{instance} exited during startup (rc={return_code}). Check log: {log_path}")
        return False

    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    log(f"Started {instance}: PID={proc.pid}, LOG={log_path}")
    return True


def start_got_ocr(script_dir: Path, deploy_vlms_root: Path) -> bool:
    """GOT-OCR 서버를 백그라운드로 시작한다. 성공 시 True."""
    instance = GOT_OCR_INSTANCE
    cmd = [sys.executable, str(script_dir / "serve_got_ocr.py")]
    log_path, pid_path = resolve_runtime_paths(deploy_vlms_root, instance)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8") as log_file:
        launched_at = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n[INFO] Launch requested at {launched_at} for instance={instance}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    time.sleep(STARTUP_POLL_SEC)
    return_code = proc.poll()
    if return_code is not None:
        pid_path.unlink(missing_ok=True)
        warn(f"{instance} exited during startup (rc={return_code}). Check log: {log_path}")
        return False

    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    log(f"Started {instance}: PID={proc.pid}, LOG={log_path}")
    return True


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = Path(os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or script_dir.parent)
    config_root = deploy_vlms_root / "config"

    print_gpu_plan(config_root)

    succeeded = []
    failed = []

    # GPU 0 모델 먼저 (ui-venus, ui-tars), 그 다음 GPU 1 모델 (mai-ui, paddleocr-vl, got-ocr)
    for instance in VLLM_MODELS:
        log(f"Starting {instance}...")
        if start_vllm_model(script_dir, deploy_vlms_root, instance):
            succeeded.append(instance)
        else:
            failed.append(instance)
        time.sleep(INTER_MODEL_DELAY_SEC)

    # GOT-OCR (transformers 기반, 별도 실행)
    log(f"Starting {GOT_OCR_INSTANCE}...")
    if start_got_ocr(script_dir, deploy_vlms_root):
        succeeded.append(GOT_OCR_INSTANCE)
    else:
        failed.append(GOT_OCR_INSTANCE)

    # 결과 요약
    log("=" * 60)
    log("시작 결과 요약")
    log("=" * 60)
    if succeeded:
        log(f"성공 ({len(succeeded)}): {', '.join(succeeded)}")
    if failed:
        warn(f"실패 ({len(failed)}): {', '.join(failed)}")
        log("실패한 모델 로그 확인:")
        for instance in failed:
            log_path = deploy_vlms_root / "runtime" / "logs" / f"{instance}.log"
            log(f"  {log_path}")
    log("=" * 60)

    if failed:
        log(f"개별 종료: python {script_dir / 'stop_model.py'} <instance>")
        log(f"전체 종료: python {script_dir / 'stop_model.py'} all")
        sys.exit(1)

    log(f"5개 모델 모두 시작 완료")
    log(f"전체 종료: python {script_dir / 'stop_model.py'} all")


if __name__ == "__main__":
    main()
