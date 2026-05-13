"""mss를 사용한 화면 캡처 헬퍼."""

from PIL import Image

import mss


def capture_primary_monitor() -> Image.Image:
    """Primary 모니터 전체 화면을 PIL Image(RGB)로 캡처한다.

    mss의 monitors[0]은 모든 모니터를 합친 가상 화면이므로
    primary 단일 모니터는 monitors[1]을 사용한다.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return image


def save_jpeg(image: Image.Image, out_path, *, quality: int = 92) -> None:
    """PIL Image를 JPEG로 저장한다. 부모 디렉토리는 자동 생성."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(str(out_path), format="JPEG", quality=quality, optimize=True)
