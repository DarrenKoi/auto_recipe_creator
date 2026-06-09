"""reranker_rule_ab 의 순수 규칙 선택 로직 단위테스트 (합성, Mac)."""
from poc.workflow_2.reranker_rule_ab import _rule_picks


def test_rule_picks_ncc_flips_chamfer_decoy():
    pool = [
        {"chamfer": 0.9, "ncc": 0.1},   # chamfer-top decoy
        {"chamfer": 0.6, "ncc": 0.8},   # truth, 높은 NCC
    ]
    picks = _rule_picks(pool, chamfer_w=0.5, ncc_w=0.5)
    # ens_ncc: 0.5*0.6+0.5*0.8=0.70 > 0.5*0.9+0.5*0.1=0.50 → idx1
    assert picks["ens_ncc"] == 1
    assert picks["ens_ncc_only"] == 1   # ncc argmax


def test_rule_picks_clamps_negative_ncc():
    pool = [{"chamfer": 0.9, "ncc": -0.5}, {"chamfer": 0.85, "ncc": 0.0}]
    picks = _rule_picks(pool, chamfer_w=0.5, ncc_w=0.5)
    # neg clamp(max(0,ncc)): idx0=0.5*0.9+0=0.45 > idx1=0.5*0.85+0=0.425 → idx0
    assert picks["ens_ncc"] == 0
    # ncc_only 는 raw argmax: 0.0 > -0.5 → idx1
    assert picks["ens_ncc_only"] == 1
