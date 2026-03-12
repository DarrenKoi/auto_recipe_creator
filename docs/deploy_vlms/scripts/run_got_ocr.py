"""GOT-OCR-2.0-hf GPU 실행 스크립트."""

import json
import os
import sys
from pathlib import Path
from typing import Any


def load_env_file(path: Path) -> None:
    if not path.is_file():
        print(f"[ERROR] Missing env file: {path}", file=sys.stderr)
        sys.exit(1)

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


def env_required(name: str) -> str:
    value = env(name)
    if not value:
        print(f"[ERROR] Missing required env: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def env_flag(name: str, default: bool = False) -> bool:
    value = env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_dtype(torch_module: Any, value: str) -> Any:
    normalized = value.strip().lower()
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in mapping:
        print(f"[ERROR] Unsupported TORCH_DTYPE: {value}", file=sys.stderr)
        sys.exit(1)
    return mapping[normalized]


def parse_box(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse BOX as JSON: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    deploy_vlms_root = Path(os.environ.get("DEPLOY_VLMS_ROOT", "").strip() or script_dir.parent)
    config_root = Path(os.environ.get("CONFIG_ROOT", "").strip() or (deploy_vlms_root / "config"))
    common_env = Path(os.environ.get("COMMON_ENV", "").strip() or (config_root / "common.env"))
    model_env = Path(
        os.environ.get("MODEL_ENV", "").strip() or (config_root / "models" / "got-ocr-2.0-hf.env")
    )

    load_env_file(common_env)
    load_env_file(model_env)

    model_id = env_required("MODEL_ID")
    image_path = env_required("IMAGE_PATH")
    gpu_id = env("GPU_ID") or "0"
    device = env("DEVICE") or "cuda"
    result_path = env("RESULT_PATH")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    try:
        import torch
    except ImportError:
        print("[ERROR] torch is required to run GOT-OCR-2.0-hf", file=sys.stderr)
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] pillow is required to run GOT-OCR-2.0-hf", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        print("[ERROR] transformers is required to run GOT-OCR-2.0-hf", file=sys.stderr)
        sys.exit(1)

    if device == "cuda" and not torch.cuda.is_available():
        print("[ERROR] DEVICE=cuda but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(1)
    if device not in {"cuda", "cpu"}:
        print(f"[ERROR] Unsupported DEVICE: {device}", file=sys.stderr)
        sys.exit(1)

    torch_dtype = parse_dtype(torch, env("TORCH_DTYPE") or "bfloat16")
    use_fast_processor = env_flag("USE_FAST_PROCESSOR", default=True)
    format_output = env_flag("FORMAT_OUTPUT", default=True)
    crop_to_patches = env_flag("CROP_TO_PATCHES", default=False)
    min_patches = int(env("MIN_PATCHES") or "1")
    max_patches = int(env("MAX_PATCHES") or "12")
    max_new_tokens = int(env("MAX_NEW_TOKENS") or "4096")
    color = env("COLOR")
    box = parse_box(env("BOX"))

    model_path = Path(model_id).resolve()
    input_image_path = Path(image_path).resolve()

    if not model_path.is_dir():
        print(f"[ERROR] MODEL_ID must be an existing directory: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not input_image_path.is_file():
        print(f"[ERROR] IMAGE_PATH must be an existing file: {input_image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] MODEL_ID={model_path}")
    print(f"[INFO] IMAGE_PATH={input_image_path}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"[INFO] DEVICE={device} TORCH_DTYPE={env('TORCH_DTYPE') or 'bfloat16'}")

    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=use_fast_processor)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch_dtype,
        device_map=device,
    )

    image = Image.open(input_image_path).convert("RGB")

    processor_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "format": format_output,
        "crop_to_patches": crop_to_patches,
        "min_patches": min_patches,
        "max_patches": max_patches,
    }
    if color:
        processor_kwargs["color"] = color
    if box is not None:
        processor_kwargs["box"] = box

    inputs = processor(image, **processor_kwargs).to(device)
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=max_new_tokens,
    )
    text = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print("[INFO] OCR result:")
    print(text)

    if result_path:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[INFO] Saved OCR result to {output_path}")


if __name__ == "__main__":
    main()
