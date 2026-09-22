# poc/workflow_3/align/test_cond_file.py
"""``cond_file.cond_for_image`` — cursor 좌표를 로드된 이미지 크기에 정규화 (Mac 실행 가능).

cond.txt 의 cursor 좌표는 ``Pixel × 10`` 프레임이다. 로드된 이미지가 cond.Pixel 과
다른 해상도면(리사이즈 저장 등) 고정 /10 변환이 좌표를 어긋나게 하므로, 소비 직전에
``cond_for_image`` 로 loaded/pixel 비율 보정을 한 번 적용한다. 멱등(pixel 필드를
로드 크기로 갱신)이라 여러 레이어에서 겹쳐 불러도 이중 보정이 없다.

실행:
    uv run python poc/workflow_3/align/test_cond_file.py
"""

from poc.workflow_3.align.cond_file import (
    CondInfo,
    cond_for_image,
    msr_modality,
    parse_cond,
)


def test_none_cond_passthrough():
    assert cond_for_image(None, (512, 512)) is None


def test_no_pixel_passthrough():
    cond = CondInfo(scope="OM", crosshair_xy=(2560, 2560))
    assert cond_for_image(cond, (512, 512)) is cond


def test_matching_size_passthrough():
    cond = CondInfo(pixel=(512, 512), crosshair_xy=(2560, 2560))
    assert cond_for_image(cond, (512, 512)) is cond


def test_degenerate_pixel_passthrough():
    cond = CondInfo(pixel=(0, 512), crosshair_xy=(2560, 2560))
    assert cond_for_image(cond, (1024, 1024)) is cond


def test_upscale_2x_scales_cursor_coords():
    # pixel 512 기준 cursor 프레임(×10) 좌표를 1024 로 로드된 이미지에 맞춘다.
    # shape_hw = (h, w). 중심 (2560,2560) → (5120,5120) (÷10 하면 512 = 1024 의 중심).
    cond = CondInfo(
        pixel=(512, 512),
        box_ltrb=(1600, 1600, 3520, 3520),
        crosshair_xy=(2560, 2560),
    )
    out = cond_for_image(cond, (1024, 1024))
    assert out.crosshair_xy == (5120, 5120), out.crosshair_xy
    assert out.box_ltrb == (3200, 3200, 7040, 7040), out.box_ltrb
    assert out.pixel == (1024, 1024), out.pixel
    # 원본은 불변(frozen dataclass 사본).
    assert cond.crosshair_xy == (2560, 2560)


def test_anisotropic_scale_per_axis():
    # 축별 비율이 다르면 x/y 를 따로 보정한다 (w=2x, h=0.5x).
    cond = CondInfo(pixel=(512, 512), crosshair_xy=(2560, 2560))
    out = cond_for_image(cond, (256, 1024))  # (h, w)
    assert out.crosshair_xy == (5120, 1280), out.crosshair_xy
    assert out.pixel == (1024, 256), out.pixel


def test_idempotent():
    cond = CondInfo(pixel=(512, 512), crosshair_xy=(2560, 2560),
                    box_ltrb=(1600, 1600, 3520, 3520))
    once = cond_for_image(cond, (1024, 1024))
    twice = cond_for_image(once, (1024, 1024))
    assert twice is once  # pixel 이 이미 로드 크기 → passthrough.


def test_preserves_scope_and_raw():
    cond = CondInfo(scope="SEM", pixel=(512, 512), crosshair_xy=(2560, 2560),
                    raw={"magnification": ["1000"]})
    out = cond_for_image(cond, (1024, 1024))
    assert out.scope == "SEM" and out.raw == {"magnification": ["1000"]}


def test_image_rotation():
    """Image_rotation 은 float 로 읽고, 0 과 '없음'(None)을 구분한다."""
    assert parse_cond("Image_rotation\t90.00\n").image_rotation == 90.0
    assert parse_cond("Image_rotation\t0\n").image_rotation == 0.0      # 0 != None.
    assert parse_cond("Scope\tOM\n").image_rotation is None              # 키 없음.
    assert parse_cond("Image_rotation\tabc\n").image_rotation is None    # 숫자 아님.
    # 키 철자 흔들림(!Cursor_inf 선례) - 접두/부분 일치로 수용.
    assert parse_cond("!Image_Rotation_Deg\t-45\n").image_rotation == -45.0
    assert parse_cond("Rotation\t12.5\n").image_rotation == 12.5


def test_scope_omdf_is_om_d():
    # Scope 가 등록기의 사실이다 - 이것만으로 OM-D 가 확정된다.
    cond = parse_cond("Scope\tOMDF\nMagnification\t104\n!OM_Brightness\t9000\n")
    assert cond.is_om and cond.is_om_dark
    assert cond.required_image_mode == "OM-D"


def test_scope_om_is_om():
    cond = parse_cond("Scope\tOM\nMagnification\t210\n!OM_Brightness\t51000\n")
    assert cond.is_om and not cond.is_om_dark
    assert cond.required_image_mode == "OM"


def test_scope_beats_brightness_both_directions():
    # 밝기 임계(35000)는 우리 가정이고 Scope 는 적힌 사실이다 - 갈리면 Scope 가 이긴다.
    dark_scope_bright_value = parse_cond("Scope\tOMDF\n!OM_Brightness\t51000\n")
    assert dark_scope_bright_value.required_image_mode == "OM-D"
    light_scope_dark_value = parse_cond("Scope\tOM\n!OM_Brightness\t12000\n")
    assert light_scope_dark_value.required_image_mode == "OM"


def test_brightness_fallback_only_when_scope_missing():
    # Scope 없는 cond 에서만 밝기로 가른다.
    assert parse_cond("!OM_Brightness\t12000\n").required_image_mode == "OM-D"
    assert parse_cond("!OM_Brightness\t51000\n").required_image_mode == "OM"
    # 경계는 OM (판정은 < 만, <= 아님).
    assert parse_cond("!OM_Brightness\t35000\n").required_image_mode == "OM"


def test_no_scope_and_no_brightness_is_unknown():
    # 모르는 것을 OM 이라 부르면 반전 화면을 정상이라 보고 지나간다.
    cond = parse_cond("Magnification\t104\n")
    assert cond.om_brightness is None
    assert cond.required_image_mode is None


def test_sem_is_sem_regardless_of_brightness():
    # SEM 에는 dark 모드가 없다 - 밝기 키가 섞여 들어와도 SEM.
    cond = parse_cond("Scope\tSEM\nMagnification\t5000\n!OM_Brightness\t100\n")
    assert cond.required_image_mode == "SEM"
    assert cond.magnification == 5000.0


def test_msr_modality_prefers_scope_over_key_heuristics():
    # msr cond 에도 Scope 가 있다(2026-09-22 정정). OMDF 는 OM 계열이라 'om' 으로 묶인다.
    assert msr_modality(parse_cond("Scope\tOMDF\n!OM_Brightness\t9000\n")) == "om"
    assert msr_modality(parse_cond("Scope\tSEM\nMagnification\t5000\n")) == "sem"
    # Scope 가 키 휴리스틱과 갈려도 Scope 가 이긴다.
    assert msr_modality(parse_cond("Scope\tSEM\n!OM_Brightness\t100\n")) == "sem"


def test_msr_modality_falls_back_to_keys_without_scope():
    assert msr_modality(parse_cond("!OM_Brightness\t128\nMagnification\t104\n")) == "om"
    assert msr_modality(parse_cond("Accelerating_voltage\t800\n")) == "sem"
    assert msr_modality(parse_cond("Magnification\t300\n")) is None


def test_non_finite_rotation_is_rejected():
    # float("nan") 은 예외 없이 통과한다 - 이 값이 장비 회전을 돌리므로 파서가 막는다.
    assert parse_cond("Image_rotation\tnan\n").image_rotation is None
    assert parse_cond("Image_rotation\tinf\n").image_rotation is None
    assert parse_cond("Image_rotation\t45.0\n").image_rotation == 45.0
    assert parse_cond("Image_rotation\t0\n").image_rotation == 0.0  # 0 != None


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
