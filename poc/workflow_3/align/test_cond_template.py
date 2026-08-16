"""cond_template primitive 합성 테스트 (Mac, 실데이터 불필요).

실행: uv run pytest poc/workflow_3/align/test_cond_template.py -q
"""

import cv2
import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.cond_file import CondInfo
from poc.workflow_3.align.cond_template import (
    CENTER_AREA_RATIO,
    CROP_INSET_PX,
    OFFSET_SKIP,
    OFFSET_WARN,
    centered_area_crop,
    centered_area_crop_bbox,
    check_cond_box,
    cond_align_offset,
    cond_offset_norm,
    cond_template_crop,
)


def _cond(box_ltrb, crosshair_xy=None):
    return CondInfo(scope="OM", pixel=(512, 512),
                    box_ltrb=box_ltrb, crosshair_xy=crosshair_xy)


def test_centered_box_has_zero_offset():
    # box 중심 (256,256) == image center → offset (0,0). cursor ×10: (2060..3060).
    assert cond_align_offset((2060, 2060, 3060, 3060), (512, 512)) == (0, 0)


def test_offcenter_box_offset_is_image_center_minus_box_center():
    # box 중심 (200,256) → offset (256-200, 0) = (56, 0).
    assert cond_align_offset((1500, 2060, 2500, 3060), (512, 512)) == (56, 0)


def test_offset_norm_uses_image_diagonal():
    onorm = cond_offset_norm((1500, 2060, 2500, 3060), (512, 512))
    assert abs(onorm - 56.0 / float(np.hypot(512, 512))) < 1e-6


def test_check_ok_for_normal_centered_box():
    status, reason, onorm = check_cond_box((2060, 2060, 3060, 3060), (512, 512))
    assert (status, reason) == ("ok", "ok") and onorm == 0.0


def test_check_skip_for_tiny_box():
    status, reason, _ = check_cond_box((2510, 2510, 2610, 2610), (512, 512))
    assert (status, reason) == ("skip", "box:too_small")


def test_check_skip_for_out_of_bounds_box():
    status, reason, _ = check_cond_box((4800, 2060, 5600, 3060), (512, 512))
    assert (status, reason) == ("skip", "box:out_of_bounds")


def test_check_skip_for_far_offcenter_box():
    status, reason, onorm = check_cond_box((150, 150, 650, 650), (512, 512))
    assert (status, reason) == ("skip", "offset:too_far") and onorm > OFFSET_SKIP


def test_cond_template_crop_centered_and_inset():
    # 200px box → 대칭 inset 후 crop = (200-2*inset)변, crop 중심 == box 중심, stroke 제거.
    box_ltrb = (1560, 1560, 3560, 3560)  # px box (156,156)-(356,356) = 200px
    gray = np.full((512, 512), 110, dtype=np.uint8)
    cv2.rectangle(gray, (156, 156), (356, 356), 255, 1)
    crop, (x0, y0, w, h) = cond_template_crop(gray, _cond(box_ltrb))
    assert w == 200 - 2 * CROP_INSET_PX and h == 200 - 2 * CROP_INSET_PX
    assert abs((x0 + w / 2.0) - 256) <= 0.5 and abs((y0 + h / 2.0) - 256) <= 0.5
    assert int(crop.max()) < 200  # inpaint + 대칭 inset 로 밝은 stroke(255) 제거.


def test_centered_area_crop_matches_bbox_helper():
    gray = np.full((512, 512), 90, dtype=np.uint8)
    x, y, cw, ch = centered_area_crop_bbox(gray, CENTER_AREA_RATIO)
    crop = centered_area_crop(gray, CENTER_AREA_RATIO)
    assert crop.shape == (ch, cw)
    assert cw < 512 and ch < 512  # 중심부 축소 crop.


def test_check_skip_for_degenerate_box():
    # bw==bh==0 → degenerate guard. cursor (2560,2560,2560,2560)/10 = box (256,256,0,0).
    status, reason, onorm = check_cond_box((2560, 2560, 2560, 2560), (512, 512))
    assert (status, reason) == ("skip", "box:degenerate") and onorm == 0.0


def test_check_warn_for_small_box():
    # inner = 22 - 2*CROP_INSET_PX = 18, in [MIN_INNER_PX(16), WARN_INNER_PX(24)); centered → onorm 0.
    # cursor (2450,2450,2670,2670)/10 = box px (245,245)-(267,267) = 22px, center (256,256).
    status, reason, _ = check_cond_box((2450, 2450, 2670, 2670), (512, 512))
    assert (status, reason) == ("warn", "box:small")


def test_check_warn_for_moderately_offcenter_box():
    # onorm in (OFFSET_WARN 0.25, OFFSET_SKIP 0.38), inner >= WARN_INNER_PX → offset:far warn.
    # cursor (60,2360,460,2760)/10 = box px (6,236)-(46,276) = 40px, center (26,256).
    # offset = (256-26, 256-256) = (230, 0); onorm = 230 / hypot(512,512) ≈ 0.3176.
    status, reason, onorm = check_cond_box((60, 2360, 460, 2760), (512, 512))
    assert (status, reason) == ("warn", "offset:far")
    assert OFFSET_WARN < onorm <= OFFSET_SKIP


def test_load_template_normalizes_pixel_mismatch(tmp_path):
    # cond.txt 는 512 기준(cursor ×10)인데 이미지는 1024 로 저장된 rcp — 실 producer
    # 포맷(.<파일명>/cond.txt) 그대로 기록해 load_template 경로 전체를 검증한다.
    # 512-기준 box px (100,100)-(300,300) → 1024 에선 (200,200)-(600,600), 400px box.
    # offset = image_center(512,512) - box_center(400,400) = (112,112) (512-기준 (56,56)의 2배).
    from poc.workflow_3.align.templates import load_template

    gray = np.full((1024, 1024), 110, dtype=np.uint8)
    cv2.rectangle(gray, (200, 200), (600, 600), 255, 1)
    img_path = tmp_path / "IMAP0001.png"
    assert cv2.imwrite(str(img_path), gray)
    cond_dir = tmp_path / f".{img_path.name}"
    cond_dir.mkdir()
    (cond_dir / "cond.txt").write_text(
        "Scope OM\n"
        "Pixel 512,512\n"
        "!Cursor_info 0,0,0,0,-1,-1,1000,1000,3000,3000\n",
        encoding="utf-8",
    )

    tpl = load_template(img_path, recipe_id="R", key_type="om", cond_box_crop=True)
    assert tpl.align_offset_xy == (112, 112), tpl.align_offset_xy
    # box 400px - 대칭 inset 2*CROP_INSET_PX → crop 변 396.
    expected = 400 - 2 * CROP_INSET_PX
    assert tpl.raw_image.shape == (expected, expected), tpl.raw_image.shape


def test_build_template_carries_align_offset():
    gray = np.full((64, 64), 120, dtype=np.uint8)
    tpl = build_template(gray, recipe_id="R", version="v0", key_type="om",
                         align_offset_xy=(5, -7))
    assert tpl.align_offset_xy == (5, -7)


def test_build_template_defaults_zero_offset():
    gray = np.full((64, 64), 120, dtype=np.uint8)
    tpl = build_template(gray, recipe_id="R", version="v0", key_type="om")
    assert tpl.align_offset_xy == (0, 0)
