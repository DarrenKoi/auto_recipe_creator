"""cond_template primitive 합성 테스트 (Mac, 실데이터 불필요).

실행: uv run pytest poc/workflow_3/align/test_cond_template.py -q
"""

import cv2
import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.cond_file import CondInfo
from poc.workflow_3.align.cond_template import (
    CENTER_AREA_RATIO,
    cond_align_point,
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


# --- align point vs box 중심 (2026-09-16) -----------------------------------
# 증상: 보정 클릭이 recipe 흰 박스의 *중심* 으로 갔다. align point 는 박스 중심이
# 아니므로, cond 의 crosshair 가 있으면 그것을 align point 로 써야 한다.

def test_align_point_prefers_crosshair():
    """crosshair 가 있으면 align point = crosshair (이미지 중심 아님)."""
    cond = CondInfo(pixel=(512, 512), crosshair_xy=(3000, 1000))   # cursor frame x10.
    (ax, ay), source = cond_align_point(cond, (512, 512))
    assert (ax, ay) == (300.0, 100.0), (ax, ay)
    assert source == "crosshair"


def test_align_point_falls_back_to_image_center():
    """crosshair 없음/cond 없음 -> 이미지 중심(기존 동작)."""
    # shape_hw = (h, w) 규약 -> (512, 400) 은 h=512, w=400 이라 중심은 (200, 256).
    assert cond_align_point(None, (512, 400)) == ((200.0, 256.0), "image_center")
    cond = CondInfo(pixel=(400, 512), crosshair_xy=None)
    assert cond_align_point(cond, (512, 400)) == ((200.0, 256.0), "image_center")


def test_align_offset_from_crosshair_differs_from_center_assumption():
    """offset = align_point - box 중심. cond 를 주면 crosshair 기준으로 갈린다."""
    box = (1000, 1000, 3000, 3000)          # 이미지 px 100..300 -> 중심 (200, 200).
    cond = CondInfo(pixel=(512, 512), box_ltrb=box, crosshair_xy=(2500, 1500))  # (250,150).
    assert cond_align_offset(box, (512, 512)) == (56, 56)            # 중심 가정(256-200).
    assert cond_align_offset(box, (512, 512), cond) == (50, -50)     # crosshair 기준.


def test_offset_norm_keeps_center_calibration():
    """cond_offset_norm/check_cond_box 는 crosshair 와 무관하게 중심 가정을 유지한다.

    OFFSET_WARN/OFFSET_SKIP 임계는 'rcp 이미지가 box 를 중심에 두고 찍혔나' 로
    캘리브레이션된 값이다. align point 정의가 바뀌어도 그 게이트는 안 흔들려야 한다.
    """
    box = (1000, 1000, 3000, 3000)
    before = cond_offset_norm(box, (512, 512))
    status, _reason, onorm = check_cond_box(box, (512, 512))
    assert onorm == before
    assert status in {"ok", "warn"}


def _write_rcp(tmp_path, stem, scope, extra=""):
    """rcp 한 장 + 짝 cond.txt (Scope / 추가 키를 바꿔가며 쓴다)."""
    gray = np.full((512, 512), 110, dtype=np.uint8)
    cv2.rectangle(gray, (100, 100), (300, 300), 255, 1)
    img_path = tmp_path / f"{stem}.png"
    assert cv2.imwrite(str(img_path), gray)
    cond_dir = tmp_path / f".{img_path.name}"
    cond_dir.mkdir()
    (cond_dir / "cond.txt").write_text(
        (f"Scope {scope}\n" if scope else "")   # scope="" = Scope 줄 없는 구형 cond
        + "Pixel 512,512\n"
        "!Cursor_info 0,0,0,0,-1,-1,1000,1000,3000,3000\n" + extra,
        encoding="utf-8",
    )
    return img_path


def test_load_template_warns_when_cond_scope_disagrees_with_filename(tmp_path, capsys):
    # IMAP0001 은 규약상 OM 인데 cond 는 SEM 이라고 말한다 - 알리기만 하고 key_type 은 그대로.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0001", "SEM")
    tpl = load_template(path, recipe_id="R", key_type="om", cond_box_crop=True)
    out = capsys.readouterr().out
    assert "[WARNING]" in out and "Scope=SEM" in out, out
    assert tpl.key_type == "om"  # 경고일 뿐 - 라우팅 키를 여기서 뒤집지 않는다.


def test_load_template_accepts_omdf_as_om(tmp_path, capsys):
    # Scope 는 OM/OMDF/SEM 셋이다 - OMDF 는 OM 계열이라 경고가 아니다.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0001", "OMDF", extra="!OM_Brightness\t9000\n")
    load_template(path, recipe_id="R", key_type="om", cond_box_crop=True)
    assert "Scope=" not in capsys.readouterr().out  # 다른 경고(밝기 없음)와 섞이지 않게.


def test_template_carries_rotation_and_required_mode(tmp_path):
    # cond.txt -> 보정 경로로 값을 나르는 유일한 객체가 template 이다. 여기서 떨어뜨리면
    # 소비처가 cond.txt 를 다시 읽어야 하고, 같은 사실에 reader 가 둘이 된다.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0002", "SEM",
                      extra="Magnification\t5000\nImage_rotation\t45.0\n")
    tpl = load_template(path, recipe_id="R", key_type="sem", cond_box_crop=True)
    assert tpl.source_rotation_deg == 45.0
    assert tpl.required_image_mode == "SEM"
    assert tpl.source_magnification == 5000.0


def test_template_required_mode_is_om_d_for_omdf_scope(tmp_path):
    # Scope=OMDF 하나로 확정된다 - 밝기 키가 없어도 된다.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0001", "OMDF", extra="Magnification\t104\n")
    tpl = load_template(path, recipe_id="R", key_type="om", cond_box_crop=True)
    assert tpl.required_image_mode == "OM-D"


def test_template_warns_when_mode_rests_on_unverified_brightness(tmp_path, capsys):
    # Scope 없는 cond 는 미검증 임계(35000)만으로 모드를 정한다. 이때 교차검증 경고는
    # 구조적으로 못 울린다 - required_mode 가 그 임계로 나온 값이라 늘 자기 자신과 같다.
    # 판정을 쓰되 '추정' 임을 반드시 남겨야 오피스에서 임계를 확정할 수 있다.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0001", "", extra="!OM_Brightness\t12000\n")
    tpl = load_template(path, recipe_id="R", key_type="om", cond_box_crop=True)
    assert tpl.required_image_mode == "OM-D"   # 판정은 낸다.
    out = capsys.readouterr().out
    assert "[WARNING]" in out and "추정" in out and "12000" in out, out


def test_template_warns_when_brightness_contradicts_scope(tmp_path, capsys):
    # Scope 를 따르되 갈린 사실을 알린다 - 이 줄이 임계 35000 을 확정하는 신호다.
    from poc.workflow_3.align.templates import load_template

    path = _write_rcp(tmp_path, "IMAP0001", "OM", extra="!OM_Brightness\t12000\n")
    tpl = load_template(path, recipe_id="R", key_type="om", cond_box_crop=True)
    assert tpl.required_image_mode == "OM"  # Scope 가 이긴다.
    out = capsys.readouterr().out
    assert "[WARNING]" in out and "12000" in out, out
