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
