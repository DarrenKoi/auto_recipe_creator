"""mss를 사용한 화면 캡처 + WebP 저장(1MB 캡) 헬퍼."""

from io import BytesIO
from pathlib import Path

from PIL import Image

import mss


# WebP 인코딩 정책 — VLM 요청 1장당 1MB 한도 보장
QUALITY_LADDER: tuple[int, ...] = (90, 80, 70, 60, 50)
DOWNSCALE_FACTOR: float = 0.9
MIN_DIMENSION: int = 512  # 안전 하한 — 이보다 작아지면 가독성/VLM 인식 모두 위험


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


def _encode_webp(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=6)
    return buffer.getvalue()


def save_webp_capped(
    image: Image.Image,
    out_path: Path,
    *,
    max_bytes: int = 1_000_000,
) -> tuple[int, float, int]:
    """PIL Image를 WebP로 저장하되, 파일 크기가 max_bytes 이하가 되도록 보장한다.

    1단계: QUALITY_LADDER(90→50) 순서로 quality 단계 인하 시도.
    2단계: 그래도 초과하면 가로/세로를 0.9배로 축소 후 quality 사다리 재시도.
    이미지 짧은 변이 MIN_DIMENSION 미만이 되면 RuntimeError.

    Returns: (사용된 quality, 최종 scale, 기록된 byte 수).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")

    original_w, original_h = image.size
    current = image
    scale = 1.0

    while True:
        for quality in QUALITY_LADDER:
            payload = _encode_webp(current, quality)
            if len(payload) <= max_bytes:
                out_path.write_bytes(payload)
                w, h = current.size
                print(
                    f"[INFO] {out_path.name}: {w}x{h} q={quality} "
                    f"scale={scale:.2f} {len(payload) / 1024:.1f}KB"
                )
                return quality, scale, len(payload)

        # 모든 quality 단계로 안 줄어듦 → 축소 후 재시도
        new_w = int(current.size[0] * DOWNSCALE_FACTOR)
        new_h = int(current.size[1] * DOWNSCALE_FACTOR)
        if min(new_w, new_h) < MIN_DIMENSION:
            raise RuntimeError(
                f"WebP {max_bytes}B 캡을 만족할 수 없음: "
                f"원본 {original_w}x{original_h}, 현재 {current.size[0]}x{current.size[1]}"
            )
        current = current.resize((new_w, new_h), Image.LANCZOS)
        scale = new_w / original_w
