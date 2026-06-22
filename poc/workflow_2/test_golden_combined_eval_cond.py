# poc/workflow_2/test_golden_combined_eval_cond.py
"""golden_combined_eval_cond 의 순수 리포트 헬퍼 테스트(golden 데이터 불요).

라우팅/층화/종합 로직만 합성 row 로 검증한다 — accuracy 숫자는 office golden 셋에서만.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import poc.workflow_2.golden_combined_eval_cond as gcc


# --- _consensus_mode_counts (history/loo cell 분포) ---

def test_consensus_mode_counts_are_cells_never_negative():
    """per-modality 평가 후 per_recipe 는 recipe당 modality cell 다수 → history/loo 는 *cell* 수.

    회귀: n_history(cell 수)를 eligible_keys(recipe 수)에서 빼서 loo 를 구하던 옛 코드는
    같은 recipe 의 두 modality cell 이 둘 다 history 면 loo=recipe1-cell2=-1 (음수, 실측 loo:-58).
    각 모드를 직접 세면 loo>=0 이고 history+loo=len(per_recipe).
    """
    per_recipe = [
        {"recipe": "A", "modality": "om", "mode": "history"},
        {"recipe": "A", "modality": "sem", "mode": "history"},
        {"recipe": "B", "modality": "sem", "mode": "loo"},
    ]
    out = gcc._consensus_mode_counts(per_recipe)
    assert out == {"history": 2, "loo": 1}


# --- _bin_by_s_count (consensus scaling 층화) ---

def _row(rec, n, cons_r1, cons_topk, rcp_r1=0.0, rcp_topk=0.0, lift=None):
    return {
        "recipe": rec, "n_S_loo": n,
        "cons_rank1_rate": cons_r1, "cons_in_topk_rate": cons_topk,
        "rcp_rank1_rate": rcp_r1, "rcp_in_topk_rate": rcp_topk,
        "lift": (cons_topk - rcp_topk) if lift is None else lift,
    }


def test_bin_by_s_count_routes_to_correct_bins():
    rows = [_row("a", 3, 0.5, 0.6), _row("b", 4, 0.7, 0.8),
            _row("c", 6, 0.9, 1.0), _row("d", 12, 1.0, 1.0)]
    bins = {b["label"]: b for b in gcc._bin_by_s_count(rows)}
    assert bins["S=3"]["n_recipes"] == 1
    assert bins["S=4"]["n_recipes"] == 1
    assert bins["S=5-6"]["n_recipes"] == 1     # n=6 → 5-6 bin
    assert bins["S>=10"]["n_recipes"] == 1     # n=12 → open bin
    assert bins["S=7-9"]["n_recipes"] == 0


def test_bin_by_s_count_frame_weighted_mean():
    # 같은 bin(S=5-6) 두 recipe: n=5(r1=0.4) + n=5(r1=0.8) → 가중평균 0.6.
    rows = [_row("a", 5, 0.4, 0.4), _row("b", 5, 0.8, 0.8)]
    b = {x["label"]: x for x in gcc._bin_by_s_count(rows)}["S=5-6"]
    assert b["n_frames"] == 10
    assert b["cons_rank1_rate"] == 0.6


def test_bin_by_s_count_empty_bin_is_none():
    b = {x["label"]: x for x in gcc._bin_by_s_count([])}["S=4"]
    assert b["n_frames"] == 0 and b["cons_rank1_rate"] is None


def test_bin_by_s_count_prefers_cons_pool_n():
    # n_S_loo(=eval frame 수)와 cons_pool_n(=consensus 풀 크기)이 다르면 풀 크기로 층화한다.
    # history mode: eval frame 2개지만 consensus 풀(과거 S)은 6장 → S=5-6 bin.
    rows = [{**_row("a", 2, 0.9, 1.0), "cons_pool_n": 6, "mode": "history"}]
    bins = {b["label"]: b for b in gcc._bin_by_s_count(rows)}
    assert bins["S=5-6"]["n_recipes"] == 1     # cons_pool_n=6 으로 분류(n_S_loo=2 아님)
    assert bins["S=5-6"]["n_frames"] == 2      # 가중치는 eval frame 수(n_S_loo)
    assert bins["S=3"]["n_recipes"] == 0


# --- _pick_cell (production-relevant 셀 선택) ---

def test_pick_cell_prefers_box_inpaint():
    cells = {"center__inpaint": {"x": 1}, "box__inpaint": {"x": 2}}
    assert gcc._pick_cell(cells) == {"x": 2}


def test_pick_cell_falls_back_to_center():
    assert gcc._pick_cell({"center__inpaint": {"x": 9}}) == {"x": 9}


def test_pick_cell_none_when_empty():
    assert gcc._pick_cell({}) is None
    assert gcc._pick_cell(None) is None


# --- _arm_rates (양 arm 동일 지표 정의) ---

def test_arm_rates_rank1_is_topk_rank_eq_1():
    cells = [
        {"topk_rank": 1, "in_topk": True},
        {"topk_rank": 3, "in_topk": True},
        {"topk_rank": None, "in_topk": False},
        {"topk_rank": 1, "in_topk": True},
    ]
    r = gcc._arm_rates(cells)
    assert r["n"] == 4
    assert r["rank1_rate"] == 0.5      # 2/4 topk_rank==1
    assert r["in_topk_rate"] == 0.75   # 3/4 in_topk


def test_arm_rates_empty():
    r = gcc._arm_rates([])
    assert r == {"n": 0, "rank1_rate": None, "in_topk_rate": None}


# --- _routed_overall (frame-weighted 종합) ---

def test_routed_overall_frame_weighted():
    # consensus 10 frame @ r1=0.9, rcp-only 10 frame @ r1=0.5 → 0.7.
    rcp = {"n": 10, "rank1_rate": 0.5, "in_topk_rate": 0.6}
    out = gcc._routed_overall(10, 0.9, 1.0, rcp)
    assert out["n_frames"] == 20
    assert out["consensus_frames"] == 10 and out["rcp_only_frames"] == 10
    assert out["rank1_rate"] == 0.7
    assert out["in_topk_rate"] == 0.8   # (1.0*10 + 0.6*10)/20


def test_routed_overall_handles_empty_arm():
    out = gcc._routed_overall(0, None, None, {"n": 0, "rank1_rate": None, "in_topk_rate": None})
    assert out["n_frames"] == 0 and out["rank1_rate"] is None


def test_routed_overall_consensus_only():
    # rcp-only 가 비면 consensus 숫자가 그대로 종합이 된다.
    rcp = {"n": 0, "rank1_rate": None, "in_topk_rate": None}
    out = gcc._routed_overall(8, 0.75, 0.9, rcp)
    assert out["rank1_rate"] == 0.75 and out["in_topk_rate"] == 0.9


# --- _digest_line (한 줄 다이제스트 — 사용자가 이 줄만 복사) ---

def _summary(**over):
    s = {
        "lab_mode": "edge_ncc", "consensus_min_s": 3,
        "n_recipes_consensus_eligible": 45, "n_recipes_rcp_only": 120,
        "consensus_arm": {"n_frames": 140, "rank1_rate": 0.78, "in_topk_rate": 0.86,
                          "overall_lift": 0.09},
        "rcp_only_arm": {"n": 210, "rank1_rate": 0.61, "in_topk_rate": 0.70},
        "routed_overall": {"n_frames": 350, "rank1_rate": 0.72, "in_topk_rate": 0.81},
        "consensus_scaling_by_s_count": [
            {"label": "S=3", "n_frames": 80, "cons_rank1_rate": 0.70},
            {"label": "S=4", "n_frames": 60, "cons_rank1_rate": 0.82},
            {"label": "S=5-6", "n_frames": 0, "cons_rank1_rate": None},
        ],
    }
    s.update(over)
    return s


def test_digest_line_is_single_line_with_key_numbers():
    d = gcc._digest_line(_summary())
    assert "\n" not in d                       # 반드시 한 줄.
    assert "lab=edge_ncc minS=3" in d
    assert "routed r1/topk=0.72/0.81 (n=350)" in d
    assert "rcp_only r1/topk=0.61/0.7 (n=210,rec=120)" in d   # 0.70 → float repr 0.7
    assert "S=3:0.7" in d and "S=4:0.82" in d   # scaling 추세.
    assert "S=5-6" not in d                     # frames=0 bin 은 생략.


def test_digest_line_lab_off_default():
    d = gcc._digest_line(_summary(lab_mode=""))
    assert "lab=off" in d


# --- modality 층화 (Step 1: OM vs SEM) ---

def test_consensus_by_modality_frame_weighted_split():
    rows = [
        {"modality": "om", "n_S_loo": 4, "cons_rank1_rate": 0.5, "cons_in_topk_rate": 0.6},
        {"modality": "om", "n_S_loo": 6, "cons_rank1_rate": 0.7, "cons_in_topk_rate": 0.8},
        {"modality": "sem", "n_S_loo": 5, "cons_rank1_rate": 0.9, "cons_in_topk_rate": 1.0},
    ]
    out = gcc._consensus_by_modality(rows)
    assert out["om"]["n_frames"] == 10
    assert out["om"]["rank1_rate"] == 0.62      # (0.5*4 + 0.7*6)/10
    assert out["sem"]["rank1_rate"] == 0.9


def test_arm_rates_by_modality_splits_cells():
    cells = [
        {"mod": "om", "topk_rank": 3, "in_topk": True},
        {"mod": "OM", "topk_rank": 1, "in_topk": True},   # 대소문자 정규화
        {"mod": "sem", "topk_rank": 1, "in_topk": True},
    ]
    out = gcc._arm_rates_by_modality(cells)
    assert out["om"]["n"] == 2 and out["om"]["rank1_rate"] == 0.5
    assert out["sem"]["n"] == 1 and out["sem"]["rank1_rate"] == 1.0


def test_routed_by_modality_combines_arms():
    cons = {"om": {"n_frames": 10, "rank1_rate": 0.8, "in_topk_rate": 0.9}}
    rcp = {"om": {"n": 10, "rank1_rate": 0.4, "in_topk_rate": 0.6},
           "sem": {"n": 5, "rank1_rate": 0.7, "in_topk_rate": 0.7}}
    out = gcc._routed_by_modality(cons, rcp)
    assert out["om"]["rank1_rate"] == 0.6       # (0.8*10 + 0.4*10)/20
    assert out["sem"]["rank1_rate"] == 0.7      # sem: rcp-only 만
    assert out["sem"]["n_frames"] == 5


# --- 실패유형 히스토그램 (Step 2: OM/SEM split 증거) ---

def test_classify_cell_three_types():
    assert gcc._classify_cell({"topk_rank": 1, "in_topk": True}) == "rank1_hit"
    assert gcc._classify_cell({"topk_rank": 3, "in_topk": True}) == "look_alike"
    assert gcc._classify_cell({"topk_rank": None, "in_topk": False}) == "recall_miss"


def test_failure_hist_by_modality_counts_and_periodic():
    cells = [
        {"mod": "om", "topk_rank": 1, "in_topk": True, "periodic": False},
        {"mod": "om", "topk_rank": 2, "in_topk": True, "periodic": True},   # periodic look_alike
        {"mod": "om", "topk_rank": 4, "in_topk": True, "periodic": False},  # non-periodic look_alike
        {"mod": "OM", "topk_rank": None, "in_topk": False, "periodic": True},  # recall_miss (periodic 무관)
    ]
    h = gcc._failure_hist_by_modality(cells)["om"]
    assert h["n"] == 4
    assert h["rank1_hit"]["n"] == 1
    assert h["look_alike"]["n"] == 2
    assert h["recall_miss"]["n"] == 1
    assert h["periodic_look_alike"]["n"] == 1          # topk_rank 2 만(rank4 는 non-periodic)
    assert h["look_alike"]["share"] == 0.5


def test_failure_hist_from_rates_reconstructs_counts():
    # n=10, in_topk=0.8(=8 in pool), rank1=0.5(=5 rank1) → recall_miss=2, look_alike=3, rank1_hit=5.
    rows = [{"modality": "sem", "n_S_loo": 10, "cons_in_topk_rate": 0.8,
             "cons_rank1_rate": 0.5, "periodic": False}]
    h = gcc._failure_hist_from_rates(rows)["sem"]
    assert h["recall_miss"]["n"] == 2
    assert h["look_alike"]["n"] == 3
    assert h["rank1_hit"]["n"] == 5
    assert h["periodic_look_alike"]["n"] == 0


def test_failure_hist_from_rates_periodic_tags_lookalike():
    rows = [{"modality": "om", "n_S_loo": 10, "cons_in_topk_rate": 0.8,
             "cons_rank1_rate": 0.5, "periodic": True}]
    h = gcc._failure_hist_from_rates(rows)["om"]
    assert h["periodic_look_alike"]["n"] == 3          # periodic recipe → 전체 look_alike 가 periodic.


def test_merge_failure_hists_sums_and_reshare():
    a = gcc._with_shares({"rank1_hit": 1, "look_alike": 1, "recall_miss": 0,
                          "periodic_look_alike": 1, "n": 2})
    b = gcc._with_shares({"rank1_hit": 0, "look_alike": 2, "recall_miss": 2,
                          "periodic_look_alike": 0, "n": 4})
    m = gcc._merge_failure_hists(a, b)
    assert m["n"] == 6
    assert m["look_alike"]["n"] == 3
    assert m["recall_miss"]["n"] == 2
    assert m["periodic_look_alike"]["n"] == 1
    assert m["look_alike"]["share"] == 0.5


def test_routed_failure_hist_unions_modalities():
    cons = {"om": gcc._with_shares({"rank1_hit": 5, "look_alike": 0, "recall_miss": 0,
                                    "periodic_look_alike": 0, "n": 5})}
    rcp = {"sem": gcc._with_shares({"rank1_hit": 0, "look_alike": 0, "recall_miss": 3,
                                    "periodic_look_alike": 0, "n": 3})}
    out = gcc._routed_failure_hist(cons, rcp)
    assert out["om"]["n"] == 5 and out["sem"]["n"] == 3
    assert out["sem"]["recall_miss"]["share"] == 1.0


def test_failure_hist_from_rates_counts_sum_to_n_under_rounding():
    # 반올림으로 recall_miss+look_alike 가 n 을 넘던 케이스: 누적 클램프로 합이 정확히 n.
    rows = [{"modality": "om", "n_S_loo": 3, "cons_in_topk_rate": 0.5,
             "cons_rank1_rate": 0.0, "periodic": False}]
    h = gcc._failure_hist_from_rates(rows)["om"]
    total = h["rank1_hit"]["n"] + h["look_alike"]["n"] + h["recall_miss"]["n"]
    assert total == 3                                  # 합이 n 을 정확히 보존
    assert h["rank1_hit"]["share"] + h["look_alike"]["share"] + h["recall_miss"]["share"] <= 1.0


# --- per-modality Youden 분리 (분류 축; L1 증거) ---

def test_youden_threshold_finds_separating_point():
    # hit(pos) 점수 {0.7,0.8,0.9}, miss(neg) {0.1,0.2,0.3} -> 임계 0.7 에서 J=1.0.
    samples = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False), (0.1, False)]
    y = gcc._youden_threshold(samples)
    assert y["thr"] == 0.7
    assert y["J"] == 1.0
    assert y["tpr"] == 1.0 and y["fpr"] == 0.0
    assert y["n_pos"] == 3 and y["n_neg"] == 3


def test_youden_threshold_none_when_one_class_empty():
    y = gcc._youden_threshold([(0.5, True), (0.6, True)])
    assert y == {"thr": None, "J": None, "tpr": None, "fpr": None, "n_pos": 2, "n_neg": 0}


def test_youden_by_modality_splits_and_skips_missing_score():
    cells = [
        {"mod": "om", "score": 0.9, "hit": True},
        {"mod": "om", "score": 0.2, "hit": False},
        {"mod": "sem", "score": 0.8, "hit": True},
        {"mod": "sem", "score": 0.1, "hit": False},
        {"mod": "sem", "score": None, "hit": True},   # score 없으면 skip
    ]
    out = gcc._youden_by_modality(cells)
    assert out["om"]["n_pos"] == 1 and out["om"]["n_neg"] == 1
    assert out["sem"]["n_pos"] == 1 and out["sem"]["n_neg"] == 1   # None score 제외


# --- split verdict (판정 규칙) ---

_CFG = {"SPLIT_MIN_FRAMES": 30, "SPLIT_MIN_RECIPES": 5,
        "SPLIT_RANK1_GAP": 0.10, "SPLIT_RANK1_FLOOR": 0.70, "SPLIT_DOMINANCE": 0.40}


def _hist(rank1_hit=0, look_alike=0, recall_miss=0, periodic_look_alike=0):
    n = rank1_hit + look_alike + recall_miss
    return gcc._with_shares({"rank1_hit": rank1_hit, "look_alike": look_alike,
                             "recall_miss": recall_miss,
                             "periodic_look_alike": periodic_look_alike, "n": n})


def _rate(n_frames, rank1_rate, n_recipes):
    return {"n_frames": n_frames, "rank1_rate": rank1_rate, "n_recipes": n_recipes}


def test_dominant_failure_periodic_lookalike():
    h = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)   # 실패 8 전부 periodic
    assert gcc._dominant_failure(h, 0.40) == "periodic_look_alike"


def test_dominant_failure_recall_miss():
    h = _hist(rank1_hit=2, recall_miss=8)
    assert gcc._dominant_failure(h, 0.40) == "recall_miss"


def test_dominant_failure_none_when_no_failures():
    assert gcc._dominant_failure(_hist(rank1_hit=5), 0.40) is None


def test_dominant_failure_none_when_below_dominance():
    # 실패 9 = periodic 3 + other_look_alike 3(=look_alike6-periodic3) + recall 3 -> 각 share 1/3 < 0.4 -> None.
    h = _hist(rank1_hit=0, look_alike=6, recall_miss=3, periodic_look_alike=3)
    assert gcc._dominant_failure(h, 0.40) is None


def test_split_verdict_split_when_gap_and_divergent():
    om = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)    # OM: periodic 지배
    sem = _hist(rank1_hit=2, recall_miss=8)                         # SEM: recall 지배
    v = gcc._split_verdict(om, sem, _rate(40, 0.60, 6), _rate(40, 0.85, 6), _CFG)  # gap 0.25
    assert v["verdict"] == "SPLIT"
    assert v["dominant_om"] == "periodic_look_alike"
    assert v["dominant_sem"] == "recall_miss"
    assert "L2_om_periodicity" in v["suggested_levers"]
    assert "L3_sem_recall" in v["suggested_levers"]


def test_split_verdict_reversed_pattern_emits_neutral_levers():
    # 설계와 반대(OM=recall 지배, SEM=periodic 지배) — named lever 를 swap 해 붙이면 안 됨.
    # (om, recall_miss)·(sem, periodic_look_alike)는 설계 매핑 밖 → 중립 '{mod}:{bucket}' 라벨.
    om = _hist(rank1_hit=2, recall_miss=8)                          # OM: recall 지배(설계 밖)
    sem = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)   # SEM: periodic 지배(설계 밖)
    v = gcc._split_verdict(om, sem, _rate(40, 0.60, 6), _rate(40, 0.85, 6), _CFG)
    assert v["verdict"] == "SPLIT"
    assert v["dominant_om"] == "recall_miss" and v["dominant_sem"] == "periodic_look_alike"
    assert v["suggested_levers"] == ["om:recall_miss", "sem:periodic_look_alike"]
    # named lever 를 modality 무시하고 붙이는 회귀 차단.
    assert "L2_om_periodicity" not in v["suggested_levers"]
    assert "L3_sem_recall" not in v["suggested_levers"]


def test_lever_for_maps_modality_and_bucket():
    assert gcc._lever_for("om", "periodic_look_alike") == "L2_om_periodicity"
    assert gcc._lever_for("sem", "recall_miss") == "L3_sem_recall"
    assert gcc._lever_for("om", "other_look_alike") == "shared_rerank"   # 공유 lever
    assert gcc._lever_for("sem", "periodic_look_alike") == "sem:periodic_look_alike"  # 설계 밖→중립
    assert gcc._lever_for("om", None) is None


def test_split_verdict_shared_tune_when_gap_no_divergence():
    om = _hist(rank1_hit=2, recall_miss=8)
    sem = _hist(rank1_hit=2, recall_miss=8)                         # 같은 실패유형
    v = gcc._split_verdict(om, sem, _rate(40, 0.55, 6), _rate(40, 0.85, 6), _CFG)
    assert v["verdict"] == "shared_tune"
    assert v["suggested_levers"] == []


def test_split_verdict_no_split_when_no_gap():
    om = _hist(rank1_hit=8, periodic_look_alike=2, look_alike=2)
    sem = _hist(rank1_hit=8, recall_miss=2)
    v = gcc._split_verdict(om, sem, _rate(40, 0.82, 6), _rate(40, 0.85, 6), _CFG)  # gap 0.03, floor ok
    assert v["verdict"] == "no_split"


def test_split_verdict_insufficient_when_thin():
    om = _hist(rank1_hit=2, recall_miss=8)
    sem = _hist(rank1_hit=2, recall_miss=8)
    v = gcc._split_verdict(om, sem, _rate(10, 0.5, 2), _rate(40, 0.85, 6), _CFG)   # OM n_frames<30
    assert v["verdict"] == "insufficient"
    assert "om" in v["insufficient_mods"]


def test_recipes_by_modality_counts_distinct():
    per_recipe = [{"modality": "om", "recipe": "A"}, {"modality": "om", "recipe": "B"}]
    cells = [{"mod": "om", "rec_key": "B"}, {"mod": "sem", "rec_key": "C"}]   # B 중복, C 신규
    out = gcc._recipes_by_modality(per_recipe, cells)
    assert out["om"] == 2 and out["sem"] == 1


def test_split_verdict_split_when_floor_triggers_not_abs_gap():
    # abs 차 0.02 < 0.10 이지만 min(0.63) < 0.70 → floor 로 gap, 분기 있으면 SPLIT.
    om = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)
    sem = _hist(rank1_hit=2, recall_miss=8)
    v = gcc._split_verdict(om, sem, _rate(40, 0.63, 6), _rate(40, 0.65, 6), _CFG)
    assert v["verdict"] == "SPLIT"
    assert v["gap_reason"] == "floor"


def test_split_verdict_shared_tune_when_one_dominant_one_none():
    # OM periodic 지배, SEM 실패 균등(dom=None: 각 bucket < 0.40) → divergent=False → shared_tune.
    # SEM: look_alike=6(periodic=3,other=3), recall=3 → 총 실패 9, 각 bucket 3/9≈0.333 < 0.40.
    om = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)
    sem = _hist(rank1_hit=2, look_alike=6, recall_miss=3, periodic_look_alike=3)
    v = gcc._split_verdict(om, sem, _rate(40, 0.60, 6), _rate(40, 0.85, 6), _CFG)
    assert v["verdict"] == "shared_tune"
    assert v["suggested_levers"] == []


# --- production-threshold 비교 (spec 4.3; L1 증거) ---

def test_confusion_at_threshold():
    # pos {0.9,0.7}, neg {0.6,0.2}; thr=0.65 → tp(0.9,0.7)=2, fp(none>=0.65 in neg? 0.6<0.65)=0, tn=2, fn=0.
    samples = [(0.9, True), (0.7, True), (0.6, False), (0.2, False)]
    c = gcc._confusion_at(samples, 0.65)
    assert c["tp"] == 2 and c["fp"] == 0 and c["tn"] == 2 and c["fn"] == 0
    assert c["acc"] == 1.0


def test_confusion_at_with_misses():
    # thr=0.75 → tp(0.9)=1, fn(0.7)=1; neg both <0.75 → tn=2, fp=0. acc=3/4.
    samples = [(0.9, True), (0.7, True), (0.6, False), (0.2, False)]
    c = gcc._confusion_at(samples, 0.75)
    assert c["tp"] == 1 and c["fn"] == 1 and c["tn"] == 2 and c["fp"] == 0
    assert c["acc"] == 0.75


def test_youden_by_modality_adds_prod_comparison():
    cells = [
        {"mod": "om", "score": 0.9, "hit": True},
        {"mod": "om", "score": 0.2, "hit": False},
    ]
    out = gcc._youden_by_modality(cells, prod_thr=0.6053)
    y = out["om"]
    assert y["prod_thr"] == 0.6053
    assert y["delta_vs_prod"] == round(y["thr"] - 0.6053, 4)
    assert y["confusion_prod"]["tp"] == 1 and y["confusion_prod"]["tn"] == 1


def test_youden_by_modality_no_prod_thr_unchanged():
    cells = [{"mod": "om", "score": 0.9, "hit": True}, {"mod": "om", "score": 0.2, "hit": False}]
    y = gcc._youden_by_modality(cells)["om"]   # prod_thr 없으면 기존 키만
    assert "prod_thr" not in y and "thr" in y


def test_youden_by_modality_reports_both_gates():
    # prod_thr(=actuation/adjust 0.4727)이 1차, prod_match_thr(=balanced 0.6053)이 참고용 2차.
    # 두 임계 모두 혼동행렬을 내야 한다(finding 1: "report both").
    cells = [
        {"mod": "om", "score": 0.9, "hit": True},
        {"mod": "om", "score": 0.5, "hit": True},    # adjust(0.4727) 위, match(0.6053) 아래
        {"mod": "om", "score": 0.2, "hit": False},
    ]
    y = gcc._youden_by_modality(cells, prod_thr=0.4727, prod_match_thr=0.6053)["om"]
    assert y["prod_thr"] == 0.4727 and y["prod_match_thr"] == 0.6053
    # adjust 게이트: 0.9·0.5 가 pos 로 통과(tp=2), 0.2 는 tn=1.
    assert y["confusion_prod"]["tp"] == 2 and y["confusion_prod"]["tn"] == 1
    # match 게이트(더 엄격): 0.9 만 통과(tp=1), 0.5 는 fn=1, 0.2 는 tn=1.
    assert y["confusion_match"]["tp"] == 1 and y["confusion_match"]["fn"] == 1


def test_youden_by_modality_match_gate_optional():
    cells = [{"mod": "om", "score": 0.9, "hit": True}, {"mod": "om", "score": 0.2, "hit": False}]
    y = gcc._youden_by_modality(cells, prod_thr=0.4727)["om"]   # prod_match_thr 생략
    assert "confusion_prod" in y and "confusion_match" not in y


# --- digest 풍부화 (spec 4.5) ---

def test_digest_line_split_verdict_token():
    s = _summary(split_verdict={"verdict": "SPLIT", "gap_reason": "abs_diff",
                                "dominant_om": "periodic_look_alike",
                                "dominant_sem": "recall_miss", "suggested_levers": ["L2_om_periodicity"]})
    d = gcc._digest_line(s)
    assert "\n" not in d
    assert "verdict=SPLIT(om=periodic_look_alike,sem=recall_miss)" in d


def test_digest_line_mod_evidence_token():
    s = _summary(**{"by_modality": {
        "failure_modes": {"routed": {
            "om": {"n": 10, "rank1_hit": {"share": 0.2}, "look_alike": {"share": 0.8},
                   "periodic_look_alike": {"share": 0.8}, "recall_miss": {"share": 0.0}},
        }},
        "youden": {"om": {"thr": 0.71, "J": 0.6}},
    }})
    d = gcc._digest_line(s)
    assert "\n" not in d
    assert "OM:" in d and "la/pm/rm=" in d


def test_split_verdict_insufficient_recipes_only_gate():
    om = _hist(rank1_hit=2, recall_miss=8)
    sem = _hist(rank1_hit=2, recall_miss=8)
    # frames pass(40>=30) 지만 recipes 부족(2<5) — recipes 게이트 단독 트리거.
    v = gcc._split_verdict(om, sem, _rate(40, 0.6, 2), _rate(40, 0.85, 6), _CFG)
    assert v["verdict"] == "insufficient" and "om" in v["insufficient_mods"]
