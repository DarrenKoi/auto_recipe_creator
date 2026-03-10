#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="${CONFIG_ROOT:-/srv/arc-vlms/config}"
COMMON_ENV="${COMMON_ENV:-${CONFIG_ROOT}/common.env}"

usage() {
    cat <<'EOF'
Usage:
  bash serve_vlm.sh <instance>

Example:
  bash serve_vlm.sh ui-venus
  bash serve_vlm.sh mai-ui
  bash serve_vlm.sh ui-tars

Environment overrides:
  CONFIG_ROOT=/srv/arc-vlms/config
  COMMON_ENV=/srv/arc-vlms/config/common.env
  MODEL_ENV=/srv/arc-vlms/config/models/ui-venus.env
EOF
}

log() {
    printf '[INFO] %s\n' "$*"
}

fail() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "Required file not found: $path"
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] || fail "Required directory not found: $path"
}

INSTANCE="${1:-}"
[[ -n "$INSTANCE" ]] || {
    usage
    exit 1
}

MODEL_ENV="${MODEL_ENV:-${CONFIG_ROOT}/models/${INSTANCE}.env}"

require_file "$COMMON_ENV"
require_file "$MODEL_ENV"
command -v vllm >/dev/null 2>&1 || fail "vllm command not found in PATH"

# shellcheck disable=SC1090
source "$COMMON_ENV"
# shellcheck disable=SC1090
source "$MODEL_ENV"

: "${MODEL_ID:?MODEL_ID is required}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
: "${PORT:?PORT is required}"
: "${GPU_ID:?GPU_ID is required}"

HOST="${HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-image=1}"
STRICT_OFFLINE="${STRICT_OFFLINE:-1}"
DISABLE_OUTBOUND_PROXIES="${DISABLE_OUTBOUND_PROXIES:-1}"
ALLOWED_MODEL_ROOT="${ALLOWED_MODEL_ROOT:-/data/models}"
CREATE_VLLM_DO_NOT_TRACK_FILE="${CREATE_VLLM_DO_NOT_TRACK_FILE:-1}"
API_KEY="${API_KEY:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

[[ "$MODEL_ID" = /* ]] || fail "MODEL_ID must be an absolute local path: $MODEL_ID"
require_dir "$MODEL_ID"
MODEL_ID_REAL="$(cd "${MODEL_ID}" && pwd -P)"

if [[ "${STRICT_OFFLINE}" == "1" ]]; then
    require_dir "${ALLOWED_MODEL_ROOT}"
    ALLOWED_MODEL_ROOT_REAL="$(cd "${ALLOWED_MODEL_ROOT}" && pwd -P)"
    if [[ "${MODEL_ID_REAL}" != "${ALLOWED_MODEL_ROOT_REAL}" && "${MODEL_ID_REAL}" != "${ALLOWED_MODEL_ROOT_REAL}/"* ]]; then
        fail "MODEL_ID must stay under ALLOWED_MODEL_ROOT=${ALLOWED_MODEL_ROOT_REAL}: ${MODEL_ID_REAL}"
    fi
fi

if [[ "${DISABLE_OUTBOUND_PROXIES}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export DO_NOT_TRACK=1
export VLLM_DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -n "${HF_HOME:-}" ]]; then
    mkdir -p "${HF_HOME}"
fi

if [[ "${CREATE_VLLM_DO_NOT_TRACK_FILE}" == "1" ]]; then
    mkdir -p "${HOME}/.config/vllm"
    : > "${HOME}/.config/vllm/do_not_track"
fi

CMD=(
    vllm serve "${MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    CMD+=(--trust-remote-code)
fi

if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
    CMD+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
    require_file "${CHAT_TEMPLATE}"
    CMD+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi

log "Starting instance=${INSTANCE}"
log "MODEL_ID=${MODEL_ID_REAL}"
log "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
log "HOST=${HOST} PORT=${PORT} GPU_ID=${GPU_ID}"
log "STRICT_OFFLINE=${STRICT_OFFLINE} DISABLE_OUTBOUND_PROXIES=${DISABLE_OUTBOUND_PROXIES}"
log "HF_HUB_OFFLINE=${HF_HUB_OFFLINE} HF_HUB_DISABLE_TELEMETRY=${HF_HUB_DISABLE_TELEMETRY}"
log "VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK} VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS}"

exec "${CMD[@]}"
