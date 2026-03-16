"""창 캡처와 이미지 인코딩 유틸리티."""

import base64
from io import BytesIO

import mss
import mss.tools
from PIL import Image


def capture_window(window) -> "Image.Image":
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
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
    print(f"[INFO] 창 캡처 완료: {image.size[0]}x{image.size[1]} px")
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
