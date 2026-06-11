# poc/workflow_3/align/test_consensus_cv.py
"""consensus_cv 포팅 프리미티브 스모크 — bench 와 bit-parity 동작 확인."""
import numpy as np

from poc.workflow_3.align.consensus_cv import (
    _consensus, _edge_density, _lap_var, _matched_crop, coregister_crops,
)


def test_consensus_median_of_uint8():
    a = np.zeros((8, 8), np.uint8)
    b = np.full((8, 8), 100, np.uint8)
    c = np.full((8, 8), 200, np.uint8)
    out = _consensus([a, b, c])
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 100  # median(0,100,200)=100


def test_sharpness_metrics_nonnegative():
    g = (np.random.RandomState(0).rand(32, 32) * 255).astype(np.uint8)
    assert _edge_density(g) >= 0.0
    assert _lap_var(g) >= 0.0
    assert _edge_density(np.zeros((0, 0), np.uint8)) == 0.0  # 빈 입력 가드


def test_matched_crop_resizes_to_template_size():
    frame = np.zeros((100, 120), np.uint8)
    crop = _matched_crop(frame, (60, 50), tw=20, th=16, scale=1.0)
    assert crop is not None and crop.shape == (16, 20)


def test_matched_crop_returns_none_when_too_small():
    frame = np.zeros((100, 120), np.uint8)
    assert _matched_crop(frame, (0, 0), tw=20, th=16, scale=0.01) is None


def test_coregister_passthrough_under_two():
    one = [np.zeros((8, 8), np.uint8)]
    assert coregister_crops(one) is one  # <2 면 그대로


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception:
            failed += 1; print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
