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


def crop_image(image: "Image.Image", crop_box: dict[str, int]) -> "Image.Image":
    """crop box ({left, top, right, bottom}) 기준으로 이미지를 잘라낸다."""
    return image.crop(
        (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
    )
