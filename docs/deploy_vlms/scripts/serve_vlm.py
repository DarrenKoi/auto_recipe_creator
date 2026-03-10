"""vLLM 서빙 스크립트 (Python 버전).

bash 실행 권한 문제가 있는 환경(사내 클라우드 등)에서
Python으로 동일한 기능을 수행한다.

사용법:
  python serve_vlm.py <instance>

예시:
  python serve_vlm.py ui-venus
  python serve_vlm.py mai-ui
  python serve_vlm.py ui-tars

환경변수 오버라이드:
  DEPLOY_VLMS_ROOT=/project/.../docs/deploy_vlms
  CONFIG_ROOT=${DEPLOY_VLMS_ROOT}/config
  COMMON_ENV=${CONFIG_ROOT}/common.env
  MODEL_ENV=${CONFIG_ROOT}/models/<instance>.env
"""

import os
import sys
import subprocess
from pathlib import Path


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def require_file(path: str) -> None:
    if not Path(path).is_file():
        fail(f"Required file not found: {path} (copy from scripts/*.env.example to config/ if first run)")


def require_dir(path: str) -> None:
    if not Path(path).is_dir():
        fail(f"Required directory not found: {path}")


def load_env_file(path: str) -> None:
    """단순 KEY=VALUE .env 파일을 os.environ에 로드한다."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 따옴표 제거
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ[key] = value


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_required(key: str) -> str:
    value = os.environ.get(key, "")
    if not value:
        fail(f"{key} is required (set in common.env or model .env)")
    return value


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    instance = sys.argv[1]

    # 경로 설정
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = env("DEPLOY_VLMS_ROOT") or str(script_dir.parent)
    config_root = env("CONFIG_ROOT") or os.path.join(deploy_vlms_root, "config")
    common_env = env("COMMON_ENV") or os.path.join(config_root, "common.env")
    model_env = env("MODEL_ENV") or os.path.join(config_root, "models", f"{instance}.env")

    os.environ["DEPLOY_VLMS_ROOT"] = deploy_vlms_root
    os.environ["CONFIG_ROOT"] = config_root

    # env 파일 로드
    require_file(common_env)
    require_file(model_env)
    load_env_file(common_env)
    load_env_file(model_env)

    # 필수 변수
    model_id = env_required("MODEL_ID")
    served_model_name = env_required("SERVED_MODEL_NAME")
    port = env_required("PORT")
    gpu_id = env_required("GPU_ID")

    # 기본값
    host = env("HOST") or "127.0.0.1"
    dtype = env("DTYPE") or "bfloat16"
    gpu_memory_utilization = env("GPU_MEMORY_UTILIZATION") or "0.80"
    max_model_len = env("MAX_MODEL_LEN") or "8192"
    max_num_seqs = env("MAX_NUM_SEQS") or "8"
    tensor_parallel_size = env("TENSOR_PARALLEL_SIZE") or "1"
    trust_remote_code = env("TRUST_REMOTE_CODE") or "1"
    limit_mm_per_prompt = env("LIMIT_MM_PER_PROMPT") or "image=1"
    strict_offline = env("STRICT_OFFLINE") or "1"
    disable_outbound_proxies = env("DISABLE_OUTBOUND_PROXIES") or "1"
    allowed_model_root = env("ALLOWED_MODEL_ROOT") or "/data/models"
    create_vllm_do_not_track_file = env("CREATE_VLLM_DO_NOT_TRACK_FILE") or "1"
    api_key = env("API_KEY")
    chat_template = env("CHAT_TEMPLATE")

    # MODEL_ID 검증: 절대경로 + 디렉토리 존재
    if not os.path.isabs(model_id):
        fail(f"MODEL_ID must be an absolute local path: {model_id}")
    require_dir(model_id)
    model_id_real = str(Path(model_id).resolve())

    # STRICT_OFFLINE: 모델 경로가 허용된 루트 아래인지 검증
    if strict_offline == "1":
        require_dir(allowed_model_root)
        allowed_model_root_real = str(Path(allowed_model_root).resolve())
        if not (model_id_real == allowed_model_root_real or model_id_real.startswith(allowed_model_root_real + "/")):
            fail(f"MODEL_ID must stay under ALLOWED_MODEL_ROOT={allowed_model_root_real}: {model_id_real}")

    # 프록시 비활성화
    if disable_outbound_proxies == "1":
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(var, None)

    # HF 토큰 제거
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(var, None)

    # 오프라인/텔레메트리 설정
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["VLLM_DO_NOT_TRACK"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # HF_HOME 디렉토리 생성
    hf_home = env("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)

    # vllm do_not_track 파일 생성
    if create_vllm_do_not_track_file == "1":
        vllm_config_dir = Path.home() / ".config" / "vllm"
        vllm_config_dir.mkdir(parents=True, exist_ok=True)
        (vllm_config_dir / "do_not_track").touch()

    # vllm serve 명령 구성
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", host,
        "--port", port,
        "--served-model-name", served_model_name,
        "--dtype", dtype,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--max-model-len", max_model_len,
        "--max-num-seqs", max_num_seqs,
        "--tensor-parallel-size", tensor_parallel_size,
    ]

    if trust_remote_code == "1":
        cmd.append("--trust-remote-code")

    if limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])

    if chat_template:
        require_file(chat_template)
        cmd.extend(["--chat-template", chat_template])

    if api_key:
        cmd.extend(["--api-key", api_key])

    # 로그 출력
    log(f"Starting instance={instance}")
    log(f"DEPLOY_VLMS_ROOT={deploy_vlms_root}")
    log(f"CONFIG_ROOT={config_root}")
    log(f"MODEL_ID={model_id_real}")
    log(f"SERVED_MODEL_NAME={served_model_name}")
    log(f"HOST={host} PORT={port} GPU_ID={gpu_id}")
    log(f"STRICT_OFFLINE={strict_offline} DISABLE_OUTBOUND_PROXIES={disable_outbound_proxies}")
    log(f"HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1")
    log(f"VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1")

    # vllm 실행 (exec 대체: 현재 프로세스를 대체)
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
