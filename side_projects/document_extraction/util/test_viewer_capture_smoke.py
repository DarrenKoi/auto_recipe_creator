"""viewer_capture 스모크 테스트 (순수 frame-diff 판정, Windows/뷰어 불필요).

실행:
    uv run python -m side_projects.document_extraction.util.test_viewer_capture_smoke
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from side_projects.document_extraction.util.viewer_capture import (
    frames_look_identical,
)


def _page(color: str = "white", text_y: int = 100) -> Image.Image:
    img = Image.new("RGB", (1280, 720), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, text_y, 1180, text_y + 40], fill="black")
    return img


def test_identical_frames() -> None:
    a = _page()
    b = _page()
    assert frames_look_identical(a, b) is True
    print("[PASS] test_identical_frames")


def test_different_pages_detected() -> None:
    # 내용 블록 위치가 다른 두 "페이지" -> 다른 화면으로 판정돼야 캡처가 계속된다.
    a = _page(text_y=100)
    b = _page(text_y=400)
    assert frames_look_identical(a, b) is False
    print("[PASS] test_different_pages_detected")


def test_minor_noise_tolerated() -> None:
    # 커서 깜빡임 수준의 국소 변화(수 픽셀)는 같은 화면으로 흡수돼야 한다.
    a = _page()
    b = _page()
    draw = ImageDraw.Draw(b)
    draw.rectangle([640, 360, 643, 375], fill="gray")  # 커서 크기 블록
    assert frames_look_identical(a, b) is True
    print("[PASS] test_minor_noise_tolerated")


def test_size_mismatch_is_not_identical() -> None:
    a = Image.new("RGB", (1280, 720), "white")
    b = Image.new("RGB", (1280, 800), "white")
    assert frames_look_identical(a, b) is False
    print("[PASS] test_size_mismatch_is_not_identical")


def main() -> int:
    test_identical_frames()
    test_different_pages_detected()
    test_minor_noise_tolerated()
    test_size_mismatch_is_not_identical()
    print("\n[INFO] 모든 viewer_capture 스모크 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
