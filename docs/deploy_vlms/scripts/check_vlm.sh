#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-}"
EXPECTED_MODEL="${2:-}"

if [[ -z "${BASE_URL}" || -z "${EXPECTED_MODEL}" ]]; then
    echo "Usage: bash check_vlm.sh <base_url> <expected_model_name>" >&2
    exit 1
fi

URL="${BASE_URL%/}/v1/models"
echo "[INFO] Checking ${URL}"

RAW="$(curl -fsS "${URL}")"
echo "${RAW}"

if [[ "${RAW}" == *"${EXPECTED_MODEL}"* ]]; then
    echo "[INFO] Model alias found: ${EXPECTED_MODEL}"
else
    echo "[ERROR] Expected model alias not found: ${EXPECTED_MODEL}" >&2
    exit 1
fi
