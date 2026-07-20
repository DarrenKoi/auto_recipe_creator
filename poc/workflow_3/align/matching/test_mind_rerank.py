"""mind_rerank 합성 smoke test (Mac/dev PC, 실데이터 불필요).

핵심 불변:
  - mind 는 score-only: 순위만 바꾸고 좌표는 항상 기존 후보 중 하나.
  - 전 후보 거부(평탄/저점수/crop 밖)면 None → engine 은 기존 selection 폴백.
  - 킬스위치 ALIGN_FAIL_MIND_RERANK=0 이면 engine 동작이 기존 NCC selection 과 동일.

실행: uv run python poc/workflow_3/align/matching/test_mind_rerank.py
"""

import os

import numpy as np

from poc.workflow_3.align.matching.engine import (
    build_template,
    compute_align_key_score_ensemble,
)
from poc.workflow_3.align.matching.mind_rerank import (
    mind_rerank_enabled,
    mind_rerank_order,
    mind_score,
    rrf_fuse_orders,
)


def _scene():
    """frame 안에 truth(=template 원본)와 decoy(다른 구조)를 심은 합성 장면."""
    rng = np.random.RandomState(3)
    frame = (rng.rand(240, 360) * 40).astype(np.uint8)
    ty, tx = 120, 90            # truth: 사각 테두리.
    frame[ty - 25:ty + 25, tx - 25] = 255
    frame[ty - 25:ty + 25, tx + 25] = 255
    frame[ty - 25, tx - 25:tx + 25] = 255
    frame[ty + 25, tx - 25:tx + 25] = 255
    dy, dx = 120, 260           # decoy: 대각 십자(다른 self-similarity 구조).
    for i in range(-25, 25):
        frame[dy + i, dx + i] = 255
        frame[dy + i, dx - i] = 255
    tpl = frame[ty - 35:ty + 35, tx - 35:tx + 35].copy()
    return tpl, frame, (tx, ty), (dx, dy)


def test_mind_prefers_truth_over_decoy():
    tpl, frame, truth, decoy = _scene()
    s_truth, _ = mind_score(tpl, tpl)                     # 자기 자신 ≈ 1.0.
    assert s_truth is not None and s_truth > 0.9
    # baseline 순서가 decoy 우선이어도 mind 재정렬은 truth 를 1순위로 올린다.
    order = mind_rerank_order(tpl, frame, [(decoy, 1.0), (truth, 1.0)])
    assert order is not None
    assert order[0] == 1, f"truth 가 1순위가 아님: {order}"


def test_flat_frame_all_rejected_returns_none():
    tpl, _, _, _ = _scene()
    flat = np.full((240, 360), 128, dtype=np.uint8)
    order = mind_rerank_order(tpl, flat, [((90, 120), 1.0), ((260, 120), 1.0)])
    assert order is None


def test_oob_candidate_kept_in_baseline_order_after_valid():
    tpl, frame, truth, _ = _scene()
    # 후보 0 은 frame 경계 밖(crop 불가) → 유효 후보(1) 뒤로 밀린다.
    order = mind_rerank_order(tpl, frame, [((2, 2), 1.0), (truth, 1.0)])
    assert order == [1, 0]


def test_rrf_fuse_orders_majority_and_tiebreak():
    assert rrf_fuse_orders([[1, 0, 2], [1, 2, 0]], 3)[0] == 1   # 다수 합의.
    assert rrf_fuse_orders([], 3) == [0, 1, 2]                  # 빈 입력 = baseline.
    assert rrf_fuse_orders([[0, 1], [1, 0]], 2)[0] == 0         # 동점 → 낮은 index.


def test_engine_picks_truth_with_and_without_mind():
    """engine e2e: 킬스위치 양쪽 모두 유효 결과 + best_xy 는 truth 근방(합성 장면)."""
    tpl, frame, truth, _ = _scene()
    template = build_template(tpl, recipe_id="synthetic/mind", version="t")
    prev = os.environ.get("ALIGN_FAIL_MIND_RERANK")
    try:
        for flag in ("0", "1"):
            os.environ["ALIGN_FAIL_MIND_RERANK"] = flag
            assert mind_rerank_enabled() == (flag == "1")
            result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
            assert result.best_xy is not None
            # score-only 불변: best_xy 는 후보 목록 안의 좌표여야 한다.
            assert any(tuple(c.xy) == tuple(result.best_xy) for c in result.candidates), \
                f"best_xy {result.best_xy} 가 후보 밖(flag={flag})"
            d = np.hypot(result.best_xy[0] - truth[0], result.best_xy[1] - truth[1])
            assert d <= 12, f"best_xy {result.best_xy} truth {truth} 거리 {d:.1f}px (flag={flag})"
    finally:
        if prev is None:
            os.environ.pop("ALIGN_FAIL_MIND_RERANK", None)
        else:
            os.environ["ALIGN_FAIL_MIND_RERANK"] = prev


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[INFO] [OK]   {name}")
            n_pass += 1
        except AssertionError as exc:
            print(f"[ERROR] [FAIL] {name}: {exc}")
    print(f"[INFO] mind_rerank smoke test: {n_pass}/{len(tests)} passed")
    return n_pass == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run() else 1)
