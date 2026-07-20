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
    ecc_rerank_enabled,
    ecc_rerank_order,
    ecc_score,
    is_sem_template,
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


def test_is_sem_template_gate():
    tpl, _, _, _ = _scene()
    assert is_sem_template(build_template(tpl, recipe_id="r", version="t", key_type="sem"))
    for kt in ("om", "box", "checker", None):
        assert not is_sem_template(build_template(tpl, recipe_id="r", version="t", key_type=kt))


def test_ecc_score_self_high_and_flat_rejects():
    tpl, _, _, _ = _scene()
    cc, reason = ecc_score(tpl, tpl.copy())         # 자기 자신 → 높은 cc.
    assert cc is not None and cc > 0.9 and reason is None
    flat = np.full_like(tpl, 128)
    cc2, reason2 = ecc_score(tpl, flat)             # 평탄 → 거부.
    assert cc2 is None and reason2 is not None


def test_ecc_rerank_prefers_truth_over_decoy():
    tpl, frame, truth, decoy = _scene()
    order = ecc_rerank_order(tpl, frame, [(decoy, 1.0), (truth, 1.0)])
    assert order is not None and order[0] == 1        # truth(index 1) 우선.


def test_engine_om_uses_mind_sem_uses_ecc():
    """engine e2e: OM/SEM 각 경로 + 킬스위치가 실제 동작하고 best_xy 는 truth 근방·후보 내부."""
    tpl, frame, truth, _ = _scene()
    saved = {k: os.environ.get(k) for k in ("ALIGN_FAIL_MIND_RERANK", "ALIGN_FAIL_ECC_RERANK")}
    try:
        # (key_type, mind_flag, ecc_flag) 조합 — 네 경로 모두 유효 결과 + score-only 불변.
        cases = [("om", "1", "1"), ("om", "0", "1"), ("sem", "1", "1"), ("sem", "1", "0")]
        for key_type, mflag, eflag in cases:
            os.environ["ALIGN_FAIL_MIND_RERANK"] = mflag
            os.environ["ALIGN_FAIL_ECC_RERANK"] = eflag
            assert mind_rerank_enabled() == (mflag == "1")
            assert ecc_rerank_enabled() == (eflag == "1")
            template = build_template(tpl, recipe_id="synthetic/x", version="t",
                                      key_type=key_type)
            result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
            assert result.best_xy is not None
            # 순위-only 불변: best_xy 는 후보 목록 안의 좌표.
            assert any(tuple(c.xy) == tuple(result.best_xy) for c in result.candidates), \
                f"best_xy 후보 밖 ({key_type},{mflag},{eflag})"
            d = np.hypot(result.best_xy[0] - truth[0], result.best_xy[1] - truth[1])
            assert d <= 12, f"거리 {d:.1f}px ({key_type},{mflag},{eflag})"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
