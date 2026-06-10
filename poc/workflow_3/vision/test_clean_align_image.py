"""``clean_align_image`` 의 좌표 변환·마스크·inpaint 테스트 (Mac 실행 가능).

cursor 좌표(Pixel 의 10배 oversample) → 이미지 px 변환, 그리고 white box
테두리 / crosshair 선만 마스크로 잡아 cv2.inpaint 로 지우는지를 확인한다.
white box '안쪽'은 실제 웨이퍼 내용이므로 절대 마스크하지 않는다.

실행:
    uv run python poc/workflow_3/vision/test_clean_align_image.py
"""

import numpy as np

from poc.workflow_3.vision.clean_align_image import (
    build_removal_mask,
    clean_image,
    cursor_to_image,
)
from poc.workflow_3.vision.cond_file import CondInfo


def test_cursor_to_image_divides_by_oversample():
    x, y = cursor_to_image((2097, 2561))
    assert abs(x - 209.7) < 1e-6 and abs(y - 256.1) < 1e-6, (x, y)


def test_mask_marks_box_border_not_interior():
    # box (1600,1600,3520,3520)/10 → (160,160,352,352) on 512.
    mask = build_removal_mask((512, 512), box_ltrb=(1600, 1600, 3520, 3520))
    assert mask[256, 256] == 0, "박스 안쪽은 마스크하면 안 됨"
    assert mask[160, 256] > 0, "박스 상단 테두리는 마스크해야 함"
    assert mask[352, 256] > 0, "박스 하단 테두리는 마스크해야 함"


def test_mask_marks_crosshair_full_lines():
    # crosshair (2097,2561)/10 → (210,256) on 512.
    mask = build_removal_mask((512, 512), crosshair_xy=(2097, 2561))
    assert mask[256, 210] > 0, "교차점"
    assert mask[256, 20] > 0, "가로선은 폭 전체"
    assert mask[480, 210] > 0, "세로선은 높이 전체"
    assert mask[100, 100] == 0, "선에서 먼 곳은 마스크 안 됨"


def test_dilate_widens_mask_to_cover_halo():
    # crosshair (2560,2560)/10 → 세로선 x=256. thickness 코어 밖(259) 픽셀은
    # dilate 없을 땐 미마스크, dilate 주면 마스크돼야(=halo 흡수).
    m0 = build_removal_mask((512, 512), crosshair_xy=(2560, 2560), dilate=0)
    m1 = build_removal_mask((512, 512), crosshair_xy=(2560, 2560), dilate=3)
    assert m1.sum() > m0.sum(), "dilate 가 마스크를 넓혀야 함"
    assert m0[100, 259] == 0 and m1[100, 259] > 0, "코어 밖 halo 영역"


def test_clean_image_changes_only_masked_pixels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 512), dtype=np.uint8)
    cond = CondInfo(scope="OM", pixel=(512, 512), crosshair_xy=(2097, 2561))
    mask = build_removal_mask(img.shape[:2], crosshair_xy=cond.crosshair_xy)
    out = clean_image(img, cond)
    assert out.shape == img.shape
    # 마스크 밖 픽셀은 그대로여야 한다.
    assert np.array_equal(out[mask == 0], img[mask == 0]), "마스크 밖이 바뀌었다"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[INFO] PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[ERROR] FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"[INFO] {len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
