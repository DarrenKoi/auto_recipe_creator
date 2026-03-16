"""디버그 이미지 저장 유틸리티."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poc.work2 import debug_image_path as resolve_debug_image_path
from poc.work2.logger import log_work2_event


def debug_image_path(
    debug_dir: Path,
    filename: str,
    model_name: str | None = None,
) -> Path:
    """모델명 하위 디렉터리를 포함한 디버그 이미지 경로를 반환한다."""
    return resolve_debug_image_path(
        debug_dir,
        filename,
        model_name=model_name,
    )


def save_debug_jpeg(
    image: "Image.Image",
    out_path: Path,
    *,
    log_name: str = "work2",
) -> None:
    """원본 스크린샷을 JPEG 로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="JPEG", quality=95)
    print(f"[INFO] 원본 캡처 저장: {out_path}")
    log_work2_event(
        component="debug_image",
        message="saved_jpeg",
        log_name=log_name,
        path=out_path,
        quality=95,
    )


def save_debug_webp(
    image: "Image.Image",
    out_path: Path,
    *,
    quality: int = 90,
    log_name: str = "work2",
) -> None:
    """VLM 입력과 동일한 WebP 이미지를 디버그 경로에 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="WEBP", quality=quality)
    print(f"[INFO] VLM 입력 WebP 저장: {out_path}")
    log_work2_event(
        component="debug_image",
        message="saved_webp",
        log_name=log_name,
        path=out_path,
        quality=quality,
    )


def save_marked_image(
    image: "Image.Image",
    elements: dict,
    colors: dict,
    out_path: Path,
) -> None:
    """좌표를 원본 스크린샷 위에 십자선과 원으로 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    radius = 12
    img_w, img_h = debug_img.size
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        x = max(0, min(int(pt["x"]), img_w - 1))
        y = max(0, min(int(pt["y"]), img_h - 1))
        color = colors.get(name, "white")
        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=color,
            width=2,
        )
        label = f"{name} ({x},{y})"
        if "input" in name or "button" in name:
            draw.text((x + radius + 3, y + 4), label, fill=color, font=font)
        else:
            draw.text((x + radius + 3, y - 16), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"} and debug_img.mode != "RGB":
        debug_img = debug_img.convert("RGB")
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")
