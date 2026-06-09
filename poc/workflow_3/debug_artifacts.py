"""workflow_3 전용 디버그 artifact 저장 유틸리티.

파일은 그대로 저장하되, 콘솔에는 저장 경로를 매번 출력하지 않는다.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from poc.workflow_3 import debug_image_path as resolve_debug_image_path


def debug_image_path(
    debug_dir: Path,
    filename: str,
    model_name: str | None = None,
    timestamp_tag: str | None = None,
    now: float | None = None,
) -> Path:
    """모델명 하위 디렉터리와 타임스탬프 prefix 를 포함한 경로를 반환한다."""
    return resolve_debug_image_path(
        debug_dir,
        filename,
        model_name=model_name,
        timestamp_tag=timestamp_tag,
        now=now,
    )


def save_debug_jpeg(image: "Image.Image", out_path: Path) -> None:
    """원본 스크린샷을 JPEG 로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="JPEG", quality=95)


def save_debug_webp(
    image: "Image.Image",
    out_path: Path,
    *,
    quality: int = 90,
) -> None:
    """VLM 입력과 동일한 WebP 이미지를 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="WEBP", quality=quality)


def save_debug_text(path: Path, text: str) -> None:
    """디버그 텍스트 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_debug_json(path: Path, payload: dict) -> None:
    """디버그 JSON 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_marked_bboxes(
    image: "Image.Image",
    elements: dict,
    colors: dict,
    out_path: Path,
) -> None:
    """bbox 와 중심점을 원본 스크린샷 위에 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    img_w, img_h = debug_img.size
    for name, item in elements.items():
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox")
        center = item.get("center")
        if not isinstance(bbox, dict):
            continue

        left = max(0, min(int(bbox.get("left", 0)), img_w - 1))
        top = max(0, min(int(bbox.get("top", 0)), img_h - 1))
        right = max(left + 1, min(int(bbox.get("right", img_w)), img_w))
        bottom = max(top + 1, min(int(bbox.get("bottom", img_h)), img_h))
        color = colors.get(name, "white")

        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

        if isinstance(center, dict) and "x" in center and "y" in center:
            x = max(0, min(int(center["x"]), img_w - 1))
            y = max(0, min(int(center["y"]), img_h - 1))
            draw.line([(x - 8, y), (x + 8, y)], fill=color, width=2)
            draw.line([(x, y - 8), (x, y + 8)], fill=color, width=2)
            draw.ellipse(
                [(x - 6, y - 6), (x + 6, y + 6)],
                outline=color,
                width=2,
            )

        label = f"{name} [{left},{top},{right},{bottom}]"
        try:
            txt_left, txt_top, txt_right, txt_bottom = draw.textbbox((0, 0), label, font=font)
            text_w = max(0, int(txt_right - txt_left))
            text_h = max(0, int(txt_bottom - txt_top))
        except Exception:
            text_w = max(0, len(label) * 7)
            text_h = 16

        text_x = max(0, min(left, max(0, img_w - text_w - 4)))
        text_y = top - text_h - 4
        if text_y < 0:
            text_y = min(max(0, bottom + 4), max(0, img_h - text_h))

        bg_right = min(img_w, text_x + text_w + 4)
        bg_bottom = min(img_h, text_y + text_h + 4)
        draw.rectangle(
            [(text_x, text_y), (bg_right, bg_bottom)],
            fill="black",
            outline=color,
            width=1,
        )
        draw.text((text_x + 2, text_y + 2), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"} and debug_img.mode != "RGB":
        debug_img = debug_img.convert("RGB")
    debug_img.save(out_path)
