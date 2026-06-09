"""reranker_signal_probe 의 순수 신호 함수 단위테스트 (합성, Mac)."""
import numpy as np

from poc.workflow_2.reranker_signal_probe import (
    _frame_patch, _mi, _ncc, _resize_template,
)


def test_ncc_identical_and_inverted():
    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert abs(_ncc(a, a) - 1.0) < 1e-6
    assert abs(_ncc(a, 255.0 - a) - (-1.0)) < 1e-6


def test_ncc_flat_zero():
    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    flat = np.full((8, 8), 5.0)
    assert _ncc(a, flat) == 0.0


def test_mi_identical_gt_constant():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 256, (32, 32)).astype(np.float64)
    const = np.full((32, 32), 100.0)   # 상수 → MI 0 (독립).
    assert _mi(a, a) > _mi(a, const)
    assert abs(_mi(a, const)) < 1e-9


def test_mi_robust_to_monotonic_remap():
    # MI 는 단조 intensity 재맵핑에 강건 — 동일 구조면 remap 해도 MI 높게 유지.
    rng = np.random.RandomState(1)
    a = rng.randint(0, 200, (40, 40)).astype(np.float64)
    remap = (a * 0.5 + 20.0)            # 단조 선형 재맵.
    indep = rng.randint(0, 200, (40, 40)).astype(np.float64)
    assert _mi(a, remap) > _mi(a, indep)


def test_frame_patch_center_and_oob():
    f = np.arange(100, dtype=np.uint8).reshape(10, 10)
    p = _frame_patch(f, 5, 5, 4, 4)    # 중심(5,5) 4x4 → x0=3,y0=3
    assert p.shape == (4, 4)
    assert _frame_patch(f, 0, 0, 4, 4) is None   # 경계 밖


def test_resize_template():
    raw = np.zeros((10, 20), np.uint8)
    r = _resize_template(raw, 0.5)
    assert r.shape == (5, 10)
