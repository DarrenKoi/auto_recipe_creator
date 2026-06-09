"""reranker_rule_ab 의 순수 규칙 선택 로직 단위테스트 (합성, Mac)."""
from poc.workflow_2.reranker_rule_ab import _calibrate_thresholds, _rule_picks


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


def test_calibrate_thresholds_separable():
    # 완전 분리: hit sel >= 0.7, miss sel <= 0.3. Youden 최대점은 t=0.7(J=1).
    pairs = [(0.7, True), (0.8, True), (0.9, True),
             (0.1, False), (0.2, False), (0.3, False)]
    out = _calibrate_thresholds(pairs, recall_target=0.95)
    assert out["match"] == 0.7
    assert out["youden_j"] == 1.0
    assert out["at_match"]["precision"] == 1.0
    assert out["at_match"]["recall"] == 1.0
    assert out["n_pos"] == 3 and out["n_neg"] == 3
    # adjust(고-recall)는 match 이하.
    assert out["adjust"] <= out["match"]


def test_calibrate_thresholds_overlap():
    # 겹침: Youden 최대 t=0.6(J=0.8), adjust(recall>=0.95→전체 hit 포착)=0.4.
    pairs = [(0.4, True), (0.6, True), (0.7, True), (0.8, True), (0.9, True),
             (0.1, False), (0.2, False), (0.3, False), (0.5, False)]
    out = _calibrate_thresholds(pairs, recall_target=0.95)
    assert out["match"] == 0.6
    assert out["adjust"] == 0.4
    assert out["adjust"] < out["match"]


def test_calibrate_thresholds_degenerate():
    # 한쪽 클래스만 → None (분리 불가).
    assert _calibrate_thresholds([(0.5, True), (0.6, True)]) is None
    assert _calibrate_thresholds([]) is None
