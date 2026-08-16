# poc/workflow_3/align/test_cond_file.py
"""``cond_file.cond_for_image`` — cursor 좌표를 로드된 이미지 크기에 정규화 (Mac 실행 가능).

cond.txt 의 cursor 좌표는 ``Pixel × 10`` 프레임이다. 로드된 이미지가 cond.Pixel 과
다른 해상도면(리사이즈 저장 등) 고정 /10 변환이 좌표를 어긋나게 하므로, 소비 직전에
``cond_for_image`` 로 loaded/pixel 비율 보정을 한 번 적용한다. 멱등(pixel 필드를
로드 크기로 갱신)이라 여러 레이어에서 겹쳐 불러도 이중 보정이 없다.

실행:
    uv run python poc/workflow_3/align/test_cond_file.py
"""

from poc.workflow_3.align.cond_file import CondInfo, cond_for_image


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
