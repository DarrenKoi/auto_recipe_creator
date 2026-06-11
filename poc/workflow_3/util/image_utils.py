"""창 캡처와 이미지 인코딩 유틸리티."""

import base64
from io import BytesIO

from PIL import Image

try:
    import mss
    import mss.tools

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


def capture_window(window) -> "Image.Image":
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
    if not MSS_AVAILABLE:
        raise ImportError("mss 라이브러리가 필요합니다.")

    rect = window.rectangle()
    region = {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_data))
    return image


def encode_image_webp(
    image: "Image.Image", quality: int = 90
) -> tuple[str, int, int]:
    """PIL Image를 base64 WebP로 인코딩한다."""
    width, height = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    print(
        f"[INFO] 이미지 인코딩: {width}x{height}, WebP q={quality}, "
        f"{len(buffer.getvalue()) / 1024:.1f}KB"
    )
    return b64, width, height


def build_relative_crop_box(
    width: int,
    height: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> dict[str, int]:
    """이미지 크기와 비율로 crop box 를 만든다."""
    left = int(round(width * min(max(left_ratio, 0.0), 1.0)))
    top = int(round(height * min(max(top_ratio, 0.0), 1.0)))
    right = int(round(width * min(max(right_ratio, 0.0), 1.0)))
    bottom = int(round(height * min(max(bottom_ratio, 0.0), 1.0)))

    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    right = min(width, right)
    bottom = min(height, bottom)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def crop_image(image: "Image.Image", crop_box: dict[str, int]) -> "Image.Image":
    """crop box 기준으로 이미지를 잘라낸다."""
    return image.crop(
        (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
    )


def ensure_min_span(start: int, end: int, total: int, minimum: int) -> tuple[int, int]:
    """최소 span 을 보장하도록 [start, end] 구간을 확장한다."""
    span = end - start
    if span >= minimum:
        return start, end

    extra = minimum - span
    grow_before = extra // 2
    grow_after = extra - grow_before
    start = max(0, start - grow_before)
    end = min(total, end + grow_after)

    if end - start >= minimum:
        return start, end

    if start == 0:
        end = min(total, minimum)
    elif end == total:
        start = max(0, total - minimum)
    return start, end


def point_to_tiny_bbox(
    point: dict, img_w: int, img_h: int, radius: int = 10,
) -> dict[str, int]:
    """포인트를 overlay 용 작은 bbox 로 감싼다."""
    return {
        "left": max(0, point["x"] - radius),
        "top": max(0, point["y"] - radius),
        "right": min(img_w, point["x"] + radius + 1),
        "bottom": min(img_h, point["y"] + radius + 1),
    }
