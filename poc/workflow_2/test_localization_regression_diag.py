"""localization_regression_diag 의 순수 분류기 단위테스트 (합성, Mac)."""
from poc.workflow_2.localization_regression_diag import _classify_regression


def test_classify_proposer_miss():
    # 진실이 pool 에 없음(모든 후보 err > tol).
    pool = [{"err": 0.5, "chamfer": 0.9}, {"err": 0.4, "chamfer": 0.7}]
    assert _classify_regression(pool, picked_idx=0, tol=0.2) == "proposer_miss"


def test_classify_orb_flip():
    # chamfer-top(idx1)은 within tol 인데 picked(idx0)≠chamfer_top → ORB(combined)가 뒤집음.
    pool = [{"err": 0.5, "chamfer": 0.6}, {"err": 0.1, "chamfer": 0.9}]
    assert _classify_regression(pool, picked_idx=0, tol=0.2) == "orb_flip"


def test_classify_chamfer_miss():
    # 진실(idx1)은 pool 에 있으나 chamfer-top(idx0)이 within tol 아님 → chamfer 가 진실을 1위로 못 올림.
    pool = [{"err": 0.5, "chamfer": 0.9}, {"err": 0.1, "chamfer": 0.6}]
    assert _classify_regression(pool, picked_idx=0, tol=0.2) == "chamfer_miss"


def test_classify_other_when_consistent():
    # chamfer-top 이 within tol 이고 picked==chamfer_top → 회귀 아님(방어적 'other').
    pool = [{"err": 0.1, "chamfer": 0.9}, {"err": 0.5, "chamfer": 0.6}]
    assert _classify_regression(pool, picked_idx=0, tol=0.2) == "other"
