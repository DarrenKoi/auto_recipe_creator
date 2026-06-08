"""``clean_metrics`` — inpaint 잔상(ghost) 측정 함수 테스트 (Mac 실행 가능).

정답(주석 없는 깨끗한 원본)이 없으므로, cond 좌표로 '주석이 있던 자리(footprint)'
와 '그 바깥 배경 띠(bg_ring)' 를 만들고, 청소된 이미지에서 둘의 밝기 차이를
잔상 점수로 쓴다. 잘 지워졌으면 footprint ≈ bg_ring → 0 에 가깝다.
"""

import numpy as np

from poc.workflow_2.clean_metrics import build_eval_masks, ghost_residual


def test_residual_near_zero_when_uniform():
    gray = np.full((512, 512), 110, np.uint8)
    fp, bg = build_eval_masks((512, 512), crosshair_xy=(2097, 2561))
    assert ghost_residual(gray, fp, bg) < 1.0


def test_residual_high_when_ghost_remains():
    gray = np.full((512, 512), 110, np.uint8)
    fp, bg = build_eval_masks((512, 512), crosshair_xy=(2097, 2561))
    gray[fp > 0] = 200                       # footprint 에 밝은 잔상을 남긴다
    assert ghost_residual(gray, fp, bg) > 50.0


def test_eval_masks_disjoint_and_nonempty():
    fp, bg = build_eval_masks((512, 512), box_ltrb=(1600, 1600, 3520, 3520))
    assert fp.any() and bg.any()
    assert not np.any((fp > 0) & (bg > 0)), "footprint 와 bg_ring 은 겹치면 안 됨"


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
