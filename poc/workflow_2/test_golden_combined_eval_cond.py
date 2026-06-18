# poc/workflow_2/test_golden_combined_eval_cond.py
"""golden_combined_eval_cond 의 순수 리포트 헬퍼 테스트(golden 데이터 불요).

라우팅/층화/종합 로직만 합성 row 로 검증한다 — accuracy 숫자는 office golden 셋에서만.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import poc.workflow_2.golden_combined_eval_cond as gcc


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
