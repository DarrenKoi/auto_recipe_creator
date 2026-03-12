"""GOT-OCR-2.0-hf HTTP 서빙 스크립트.

transformers 기반 모델을 Flask 서버로 감싸서
다른 vLLM 서비스처럼 HTTP 요청으로 사용할 수 있게 한다.

엔드포인트:
  GET  /v1/models  — 헬스 체크 (vLLM 호환)
  POST /v1/ocr     — OCR 실행 (base64 이미지 입력)

사용법:
  python serve_got_ocr.py          # 기본 포트 8005
"""

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any


def load_env_file(path: Path) -> None:
    """env 파일에서 환경변수를 로드한다 (기존 값은 덮어쓰지 않음)."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def env_flag(name: str, default: bool = False) -> bool:
    value = env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


# ── env 로드 ──────────────────────────────────────────────────────────

script_dir = Path(__file__).resolve().parent
deploy_vlms_root = Path(env("DEPLOY_VLMS_ROOT") or script_dir.parent)
config_root = Path(env("CONFIG_ROOT") or (deploy_vlms_root / "config"))

load_env_file(config_root / "common.env")
load_env_file(config_root / "models" / "got-ocr-2.0-hf.env")

MODEL_ID = env("MODEL_ID")
GPU_ID = env("GPU_ID") or "0"
DEVICE = env("DEVICE") or "cuda"
TORCH_DTYPE_STR = env("TORCH_DTYPE") or "bfloat16"
USE_FAST_PROCESSOR = env_flag("USE_FAST_PROCESSOR", default=True)
FORMAT_OUTPUT = env_flag("FORMAT_OUTPUT", default=True)
CROP_TO_PATCHES = env_flag("CROP_TO_PATCHES", default=False)
MIN_PATCHES = int(env("MIN_PATCHES") or "1")
MAX_PATCHES = int(env("MAX_PATCHES") or "12")
MAX_NEW_TOKENS = int(env("MAX_NEW_TOKENS") or "4096")
PORT = int(env("GOT_OCR_SERVE_PORT") or "8005")
SERVED_MODEL_NAME = "got-ocr-2.0-hf"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# ── 의존성 import ─────────────────────────────────────────────────────

try:
    import torch
except ImportError:
    print("[ERROR] torch 필요", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("[ERROR] pillow 필요", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    print("[ERROR] transformers 필요", file=sys.stderr)
    sys.exit(1)

try:
    from flask import Flask, jsonify, request as flask_request
except ImportError:
    print("[ERROR] flask 필요 (pip install flask)", file=sys.stderr)
    sys.exit(1)

# ── dtype 파싱 ────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}
torch_dtype = _DTYPE_MAP.get(TORCH_DTYPE_STR.lower())
if torch_dtype is None:
    print(f"[ERROR] Unsupported TORCH_DTYPE: {TORCH_DTYPE_STR}", file=sys.stderr)
    sys.exit(1)

# ── 모델 로드 ─────────────────────────────────────────────────────────

if not MODEL_ID:
    print("[ERROR] MODEL_ID가 설정되지 않음", file=sys.stderr)
    sys.exit(1)

model_path = Path(MODEL_ID).resolve()
if not model_path.is_dir():
    print(f"[ERROR] MODEL_ID 디렉토리 없음: {model_path}", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Loading model from {model_path} ...")
print(f"[INFO] DEVICE={DEVICE}  DTYPE={TORCH_DTYPE_STR}  GPU_ID={GPU_ID}")

processor = AutoProcessor.from_pretrained(str(model_path), use_fast=USE_FAST_PROCESSOR)
model = AutoModelForImageTextToText.from_pretrained(
    str(model_path),
    dtype=torch_dtype,
    device_map=DEVICE,
)

print(f"[INFO] Model loaded. Serving on port {PORT}")

# ── 추론 함수 ─────────────────────────────────────────────────────────


def run_ocr(image: Image.Image, **kwargs: Any) -> str:
    """이미지에 대해 OCR을 수행하고 텍스트를 반환한다."""
    processor_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "format": kwargs.get("format_output", FORMAT_OUTPUT),
        "crop_to_patches": kwargs.get("crop_to_patches", CROP_TO_PATCHES),
        "min_patches": kwargs.get("min_patches", MIN_PATCHES),
        "max_patches": kwargs.get("max_patches", MAX_PATCHES),
    }
    color = kwargs.get("color", "")
    if color:
        processor_kwargs["color"] = color
    box = kwargs.get("box")
    if box is not None:
        processor_kwargs["box"] = box

    inputs = processor(image, **processor_kwargs).to(DEVICE)
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=kwargs.get("max_new_tokens", MAX_NEW_TOKENS),
    )
    return processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ── Flask 앱 ──────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """vLLM 호환 /v1/models 엔드포인트."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    })


@app.route("/v1/ocr", methods=["POST"])
def ocr():
    """OCR 엔드포인트.

    요청 JSON:
      {
        "image": "<base64 encoded image>",
        "max_new_tokens": 4096,     (선택)
        "format_output": true,      (선택)
        "crop_to_patches": false,   (선택)
        "color": "",                (선택)
        "box": [x1, y1, x2, y2]    (선택)
      }
    """
    data = flask_request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "요청 JSON에 'image' (base64) 필드 필요"}), 400

    try:
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"이미지 디코딩 실패: {exc}"}), 400

    start = time.monotonic()
    try:
        text = run_ocr(
            image,
            max_new_tokens=data.get("max_new_tokens", MAX_NEW_TOKENS),
            format_output=data.get("format_output", FORMAT_OUTPUT),
            crop_to_patches=data.get("crop_to_patches", CROP_TO_PATCHES),
            color=data.get("color", ""),
            box=data.get("box"),
        )
    except Exception as exc:
        return jsonify({"error": f"OCR 실행 실패: {exc}"}), 500

    elapsed_ms = (time.monotonic() - start) * 1000
    print(f"[INFO] OCR done in {elapsed_ms:.0f}ms, {len(text)} chars")

    return jsonify({
        "model": SERVED_MODEL_NAME,
        "text": text,
        "elapsed_ms": round(elapsed_ms, 1),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
