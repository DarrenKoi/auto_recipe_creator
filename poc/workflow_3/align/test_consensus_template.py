# poc/workflow_3/align/test_consensus_template.py
"""consensus_template build/gate/select 스모크 — 게이트 폴백 신호 + 라우팅 조립."""
import numpy as np

from poc.workflow_3.align.consensus_template import (
    DEFAULT_CONSENSUS_POLICY, ConsensusPolicy,
    build_consensus_template, select_routing_templates,
)


def _sharp_crops(n, size=64):
    """edge 풍부한 또렷한 동일 패턴 crop n장(blur 가드 통과용)."""
    rng = np.random.RandomState(1)
    base = (rng.rand(size, size) * 255).astype(np.uint8)
    return [base.copy() for _ in range(n)]


def test_insufficient_s_returns_none():
    res = build_consensus_template(_sharp_crops(2), recipe_id="E/c/r", modality="sem",
                                   policy=ConsensusPolicy(min_s=4))
    assert res.template is None and res.reason == "insufficient_s" and res.n_crops == 2


def test_enough_sharp_builds_template():
    res = build_consensus_template(_sharp_crops(5), recipe_id="E/c/r", modality="sem",
                                   policy=ConsensusPolicy(min_s=4))
    assert res.reason == "ok" and res.template is not None
    assert res.template.key_type == "sem"


def test_blurry_consensus_rejected():
    # 서로 크게 어긋난 노이즈 → median 이 흐려 edge/lap 비율 < 임계.
    rng = np.random.RandomState(2)
    crops = [(rng.rand(64, 64) * 255).astype(np.uint8) for _ in range(5)]
    res = build_consensus_template(crops, recipe_id="E/c/r", modality="om",
                                   policy=DEFAULT_CONSENSUS_POLICY)
    assert res.template is None and res.reason == "blurry"


def test_select_prefers_consensus_else_rcp():
    class T:  # AlignKeyTemplate 자리 대역(select 는 객체 정체성만 본다).
        pass
    cons_om, rcp_om, rcp_sem = T(), T(), T()
    out = select_routing_templates({"om": cons_om, "sem": None},
                                   {"om": rcp_om, "sem": rcp_sem})
    assert out["OM"] is cons_om   # consensus 우선
    assert out["SEM"] is rcp_sem  # consensus None → rcp 폴백


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
