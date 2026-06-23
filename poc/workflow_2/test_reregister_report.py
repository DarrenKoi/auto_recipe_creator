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


def test_aggregate_strong_counts_only_in_topk_false():
    # fail = in_topk=False 만. rank3(후보엔 있음)은 리랭커 복구 가능 → fail 아님 → 1/3.
    # worst_disp 는 fail(missing) 프레임에서만 → 0.9 (rank3 의 0.4 는 제외).
    frames = [
        {"in_topk": True, "topk_rank": 1, "best_cand_dist_norm": 0.05},
        {"in_topk": True, "topk_rank": 3, "best_cand_dist_norm": 0.4},
        {"in_topk": False, "topk_rank": None, "best_cand_dist_norm": 0.9},
    ]
    out = rr._aggregate_strong(frames)
    assert out["n_s"] == 3
    assert abs(out["strong_fail_frac"] - 1 / 3) < 1e-9
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


def test_tier_strong_when_free_search_fails():
    tier, sev = rr._evidence_tier("sem", 0.5, 0.99, 0.99)
    assert tier == "STRONG" and sev == 0.5


def test_tier_strong_requires_frac_floor():
    # frac < STRONG_FRAC_FLOOR(0.5) 면 STRONG 아님 (재랭킹 가능한 소수 miss). 다른 신호 낮으면 NONE.
    tier, _ = rr._evidence_tier("sem", 0.33, 0.10, 0.10)
    assert tier == "NONE"
    # 단, 같은 sub-floor 라도 msr 모호하면 MEDIUM 으로 떨어진다(STRONG 만 못 됨).
    assert rr._evidence_tier("sem", 0.33, 0.90, 0.10)[0] == "MEDIUM"


def test_tier_medium_on_msr_tail():
    tier, sev = rr._evidence_tier("sem", 0.0, 0.90, 0.99)
    assert tier == "MEDIUM" and sev == 0.90


def test_tier_advisory_only_for_om():
    tier, _ = rr._evidence_tier("om", 0.0, 0.10, 0.90)
    assert tier == "ADVISORY"


def test_sem_self_never_surfaces():
    # SEM self-match 가 높아도(near-degenerate) MEDIUM/ADVISORY 로 안 뜸 → NONE.
    tier, _ = rr._evidence_tier("sem", 0.0, 0.10, 0.99)
    assert tier == "NONE"


def test_tier_none_below_floors():
    assert rr._evidence_tier("om", 0.0, 0.10, 0.10)[0] == "NONE"


def test_risk_score_orders_tiers():
    assert (rr._risk_score("STRONG", 0.0) > rr._risk_score("MEDIUM", 0.99)
            > rr._risk_score("ADVISORY", 0.99) > rr._risk_score("NONE", 0.99))


def test_rank_rows_desc_with_disp_tiebreak():
    rows = [
        {"recipe": "a", "risk_score": 2.5, "worst_disp": 0.3},
        {"recipe": "b", "risk_score": 2.5, "worst_disp": 0.9},  # 동점 → worst_disp 큰 게 위.
        {"recipe": "c", "risk_score": 1.2, "worst_disp": 0.9},
    ]
    ranked = rr._rank_rows(rows)
    assert [r["recipe"] for r in ranked] == ["b", "a", "c"]


def test_rank_rows_single_and_equal_safe():
    # 1-recipe / 동값 cohort 에서 예외·div 없이 동작(min-max 제거 회귀 가드).
    assert rr._rank_rows([{"recipe": "x", "risk_score": 1.0, "worst_disp": 0.0}])[0]["recipe"] == "x"
    rr._rank_rows([{"recipe": "p", "risk_score": 1.0, "worst_disp": 0.0},
                   {"recipe": "q", "risk_score": 1.0, "worst_disp": 0.0}])  # no raise


def _sample_rows():
    return {
        "om": [
            {"recipe": "L/r1", "tier": "STRONG", "strong_fail_frac": 0.5, "worst_disp": 0.8,
             "msr_peak_tail": 0.99, "self_ratio": 0.9, "advisory_confidence": "ok", "n_s": 6,
             "risk_score": 2.5, "suggestion": "box(10,10,40,40)", "sugg_self": 0.5, "sugg_fidelity": 0.7},
        ],
        "sem": [
            {"recipe": "L/r2", "tier": "MEDIUM", "strong_fail_frac": 0.0, "worst_disp": 0.2,
             "msr_peak_tail": 0.92, "self_ratio": 0.99, "advisory_confidence": "low", "n_s": 3,
             "risk_score": 1.92, "suggestion": "insufficient", "sugg_self": None, "sugg_fidelity": None},
        ],
    }


def test_report_is_ascii_and_has_banner_and_rows():
    text = rr._format_report(_sample_rows())
    text.encode("ascii")  # em-dash 등 비-ASCII 있으면 raise.
    assert rr.SURVIVORSHIP_BANNER in text
    assert "L/r1" in text and "L/r2" in text
    assert "STRONG" in text and "MEDIUM" in text


def test_digest_is_ascii_one_line_per_pipe():
    d = rr._format_digest(_sample_rows())
    d.encode("ascii")
    assert d.startswith("[DIGEST] reregister(S-only):")
    assert "om[" in d and "sem[" in d and "|" in d


def test_banner_has_no_emdash():
    assert "—" not in rr.SURVIVORSHIP_BANNER


def test_run_no_data_returns_warning(monkeypatch, tmp_path):
    # 빈 골든 루트 → no_data 경로(예외 없이 경고 문자열).
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))
    out = rr.run()
    assert "no_data" in out.lower()


# ====================================================================
# Task 5: 박스 제안 순수 헬퍼 테스트.
# ====================================================================
def test_split_frames_insufficient():
    assert rr._split_frames(["a", "b", "c"], split_min_s=4) is None


def test_split_frames_deterministic_halves():
    sel, val = rr._split_frames(["a", "b", "c", "d", "e"], split_min_s=4)
    assert sel == ["a", "c", "e"] and val == ["b", "d"]  # even-idx select, odd-idx validate.


def test_split_frames_default_allows_two():
    # 오피스 실측 n_s=2~3 → 기본 SPLIT_MIN_S=2 라야 제안이 돈다(4 면 전부 insufficient).
    assert rr.SPLIT_MIN_S == 2
    sel, val = rr._split_frames(["a", "b"])     # 기본 floor 사용.
    assert sel == ["a"] and val == ["b"]        # 1/1 split (validate 1장, advisory).
    assert rr._split_frames(["a"]) is None       # 1장은 여전히 insufficient.


def test_iter_candidate_boxes_within_bounds():
    boxes = rr._iter_candidate_boxes(200, 200, (80, 80, 120, 120))
    assert boxes  # 비어있지 않음.
    for (l, t, r, b) in boxes:
        assert 0 <= l < r <= 200 and 0 <= t < b <= 200


def test_select_candidate_gates_on_baseline_fidelity():
    baseline = {"self_ratio": 0.95, "sel_fidelities": [0.6, 0.6]}  # mean 0.6.
    cands = [
        {"box": (0, 0, 10, 10), "self_ratio": 0.3, "sel_fidelities": [0.4, 0.4]},  # fid<baseline → 탈락.
        {"box": (1, 1, 11, 11), "self_ratio": 0.5, "sel_fidelities": [0.7, 0.7]},  # 통과, self 0.5.
        {"box": (2, 2, 12, 12), "self_ratio": 0.4, "sel_fidelities": [0.65, 0.65]},  # 통과, self 0.4(최저).
    ]
    pick = rr._select_candidate(cands, baseline)
    assert pick["box"] == (2, 2, 12, 12)


def test_select_candidate_none_when_all_fail_gate():
    baseline = {"self_ratio": 0.9, "sel_fidelities": [0.8, 0.8]}
    cands = [{"box": (0, 0, 1, 1), "self_ratio": 0.1, "sel_fidelities": [0.5, 0.5]}]
    assert rr._select_candidate(cands, baseline) is None


def test_accept_candidate_requires_both_margins():
    baseline = {"self_ratio": 0.95, "val_fidelities": [0.6, 0.6]}
    good = {"self_ratio": 0.4, "val_fidelities": [0.7, 0.7]}  # fid +0.1, self -0.55 → accept.
    assert rr._accept_candidate(good, baseline) is True
    weak_fid = {"self_ratio": 0.4, "val_fidelities": [0.61, 0.61]}  # fid +0.01 < margin.
    assert rr._accept_candidate(weak_fid, baseline) is False
    weak_self = {"self_ratio": 0.93, "val_fidelities": [0.7, 0.7]}  # self -0.02 < margin.
    assert rr._accept_candidate(weak_self, baseline) is False


def test_box_overlap_ratio():
    # box (0,0,10,10) area100; region (5,5,15,15) intersect (5,5,10,10) area25 → 0.25.
    assert abs(rr._box_overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)) - 0.25) < 1e-9
    assert rr._box_overlap_ratio((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_dodge_guard_rejects_overlap_avoidance_near_margin():
    # 현재 overlap 0.5, 후보 0.0(급감) + val_delta 가 margin 부근(0.05) → reject.
    assert rr._dodge_guard(0.0, 0.5, 0.05) is True
    # 후보가 충분히 이기면(val_delta 큼) overlap 급감이어도 통과(가짜 아님).
    assert rr._dodge_guard(0.0, 0.5, 0.5) is False


# ====================================================================
# Task 7: C2 합성 이미지 테스트 — 실 엔진으로 Mac 에서 실행 가능.
# ====================================================================
import numpy as np


def _periodic_img(w=240, h=240, period=24):
    """주기 격자 배경 이미지(uint8). 테스트 전용."""
    g = np.zeros((h, w), np.uint8)
    g[:, ::period] = 255
    g[::period, :] = 255
    return g


def test_suggestion_finds_unique_patch_over_periodic():
    """주기 배경 + 한 곳에 비주기 고유 마크 -> 검색이 그 영역 박스를 찾음(self_ratio 낮음)."""
    img = _periodic_img()
    img[112:128, 112:128] = 180   # 고유 블록.
    base = (10, 10, 30, 30)       # 주기 영역의 엔지니어 박스(모호).
    found = rr._search_unique_box(img, base)
    assert found is not None, "_search_unique_box 가 None 반환(후보 없음) — 고유 마크를 찾지 못했음."
    assert found["self_ratio"] < 0.9, (
        f"self_ratio={found['self_ratio']:.3f} >= 0.9 — 고유 마크 박스가 선택되지 않았음."
    )


def test_suggestion_all_periodic_returns_none_distinctive():
    """전부 주기 배경 -> 어떤 박스도 충분히 변별 안 됨 -> None 또는 self_ratio >= 0.9."""
    img = _periodic_img()
    base = (10, 10, 34, 34)
    found = rr._search_unique_box(img, base)
    # 전부 주기 → None(후보 없음)이거나 self_ratio 가 높아야(변별 안 됨).
    assert found is None or found["self_ratio"] >= 0.9, (
        f"self_ratio={found['self_ratio']:.3f} < 0.9 — 주기 이미지에서 변별 후보가 잘못 선택됨."
    )
