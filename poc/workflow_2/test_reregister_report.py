"""reregister 리포트 순수 헬퍼 + config 브리지 테스트."""
import os
from poc.workflow_2 import golden_eval_config_loader as cfg


def test_seed_env_bridges_reregister_defaults():
    # 기존 값 격리
    for k in ("REREGISTER_BOX_SUGGEST", "REREGISTER_TOPN"):
        os.environ.pop(k, None)
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "1"
    assert os.environ["REREGISTER_TOPN"] == "0"


def test_seed_env_respects_existing_reregister(monkeypatch):
    monkeypatch.setenv("REREGISTER_BOX_SUGGEST", "0")
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "0"  # OS env 우선


from poc.workflow_2 import golden_reregister_report_cond as rr


def test_aggregate_strong_counts_off_target_and_missing():
    # 3 프레임: rank1(ok) / rank3(off) / None=in_topk False(missing) → fail 2/3.
    frames = [
        {"in_topk": True, "topk_rank": 1, "best_cand_dist_norm": 0.05},
        {"in_topk": True, "topk_rank": 3, "best_cand_dist_norm": 0.4},
        {"in_topk": False, "topk_rank": None, "best_cand_dist_norm": 0.9},
    ]
    out = rr._aggregate_strong(frames)
    assert out["n_s"] == 3
    assert abs(out["strong_fail_frac"] - 2 / 3) < 1e-9
    assert out["worst_disp"] == 0.9


def test_aggregate_strong_all_clean():
    frames = [{"in_topk": True, "topk_rank": 1, "best_cand_dist_norm": 0.02}]
    assert rr._aggregate_strong(frames)["strong_fail_frac"] == 0.0


def test_aggregate_medium_uses_max_tail_and_zero_for_missing():
    # peak_ratio None(후보<2)은 0 으로 반영, tail=max.
    frames = [{"peak_ratio": 0.7}, {"peak_ratio": None}, {"peak_ratio": 0.93}]
    out = rr._aggregate_medium(frames)
    assert out["msr_peak_tail"] == 0.93
    assert out["n_s"] == 3


def test_self_ratio_excludes_trivial_peak():
    # cands: 자기-peak(원점 score 1.0) + 근접 sidelobe(제외돼야) + 먼 look-alike 0.6.
    class C:
        def __init__(self, xy, score):
            self.xy, self.score = xy, score
    cands = [C((100, 100), 1.0), C((104, 100), 0.95), C((400, 400), 0.6)]
    # excl 10px → sidelobe(거리4) 제외, 먼 look-alike 생존 → 0.6/1.0.
    assert abs(rr._self_ratio(cands, (100, 100), 10.0) - 0.6) < 1e-9


def test_self_ratio_unique_when_no_survivor():
    class C:
        def __init__(self, xy, score):
            self.xy, self.score = xy, score
    cands = [C((100, 100), 1.0), C((103, 100), 0.9)]  # 둘 다 excl 안 → 생존 0.
    assert rr._self_ratio(cands, (100, 100), 10.0) == 0.0
