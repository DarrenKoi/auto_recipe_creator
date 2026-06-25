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


def test_seed_env_bridges_e_confirm_defaults(monkeypatch):
    for k in ("REREGISTER_E_CONFIRM", "REREGISTER_S_FLOOR", "REREGISTER_E_FLOOR",
              "REREGISTER_COLLAPSE_MARGIN"):
        monkeypatch.delenv(k, raising=False)
    cfg.seed_env()
    assert os.environ["REREGISTER_E_CONFIRM"] == "1"
    assert os.environ["REREGISTER_S_FLOOR"] == "0.6"
    assert os.environ["REREGISTER_E_FLOOR"] == "0.5"
    assert os.environ["REREGISTER_COLLAPSE_MARGIN"] == "0.15"


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


def test_normalize_consensus_key_triplet_to_doublet():
    # consensus 키는 eqp/class/recipe 트리플렛 → reregister 의 class/recipe 더블렛으로.
    assert rr._normalize_consensus_key("EQP01/CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_doublet_passthrough():
    # 이미 더블렛이면 그대로(정규화 멱등).
    assert rr._normalize_consensus_key("CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_extra_segments_keeps_last_two():
    # 혹시 4단 이상이어도 마지막 두 세그먼트(class/recipe)만.
    assert rr._normalize_consensus_key("F/EQP01/CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_strips_whitespace():
    assert rr._normalize_consensus_key("  EQP01/CLSA/REC1  ") == "CLSA/REC1"


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


def test_box_offset_xy_is_box_center_minus_frame_center():
    # frame 240x240 -> center (120,120). box (154,104,190,140) -> center (172,122).
    off = rr._box_offset_xy((154, 104, 190, 140), 240, 240)
    assert off == (52.0, 2.0)
    # 중심 박스는 offset 0.
    assert rr._box_offset_xy((100, 100, 140, 140), 240, 240) == (0.0, 0.0)


def test_fidelity_gt_tol_norm_default_widened():
    # 0.20 은 참 localization(0.20~0.24)을 1~6px 차로 놓쳤다 -> 기본 0.30 으로 넓힘.
    assert rr._FIDELITY_GT_TOL_NORM == 0.30


def test_cap_recipes_limits_when_positive():
    items = ["a", "b", "c", "d", "e"]
    assert rr._cap_recipes(items, 3) == ["a", "b", "c"]   # 양수 cap -> 앞에서 N개.
    assert rr._cap_recipes(items, 0) == items             # 0 = 전체(무제한).
    assert rr._cap_recipes(items, 99) == items            # cap > len -> 전체.


def test_resolve_fidelity_scales_default_is_tight_band():
    # 기본: 0.6/0.75 distractor escape hatch 를 뺀 1.0 근방 tight band.
    band = rr._resolve_fidelity_scales(None)
    assert 1.0 in band
    assert 0.6 not in band and 0.75 not in band
    assert all(0.8 <= s <= 1.2 for s in band)


def test_resolve_fidelity_scales_env_override_and_malformed():
    # env 로 임의 band 지정(예: COMPARE_SCALES 복원 A arm).
    assert rr._resolve_fidelity_scales("0.6,0.75,0.85,1.0") == (0.6, 0.75, 0.85, 1.0)
    # malformed -> 기본 tight band 로 폴백(크래시 금지).
    assert rr._resolve_fidelity_scales("abc, ,") == rr._resolve_fidelity_scales(None)


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


def _frame_with_offset_mark(mark_xy=(170, 120)):
    """저텍스처 배경 + crosshair(=정중앙) 에서 떨어진 위치에 고유 마크를 둔 프레임.

    mark_xy 는 frame 중심(=align point/crosshair)에서 떨어져 있어 이 마크 박스의
    중심은 align point 와 일치하지 않는다(off-center sub-crop). 마크는 비대칭(오른쪽
    돌출)이라 매칭이 유일 위치에 lock 된다.
    """
    g = np.full((240, 240), 40, np.uint8)
    mx, my = mark_xy
    g[my - 12:my + 12, mx - 12:mx + 12] = 200
    g[my - 12:my + 12, mx - 2:mx + 2] = 40    # 내부 세로 십자.
    g[my - 2:my + 2, mx - 12:mx + 12] = 40    # 내부 가로 십자.
    g[my - 12:my + 4, mx + 12:mx + 18] = 200  # 오른쪽 돌출 -> 비대칭.
    return g


def test_fidelity_nonzero_for_offset_box():
    """align point 에서 떨어진 박스라도 box_offset_xy 를 주면 fidelity > 0.

    박스 중심이 crosshair(gt_xy)와 (+52,+2) 어긋나 있어도, offset 을 적용하면
    기대 위치 = gt_xy + offset*scale 부근에서 매칭을 찾아 fidelity 가 살아난다.
    이 경로(off-center sub-crop 의 fidelity)가 바로 all-zero 경고를 양산하던 곳이다.
    """
    img = _frame_with_offset_mark()
    box = (154, 104, 190, 140)         # 마크(170,120) 둘레 박스. 중심 (172,122).
    patch = img[box[1]:box[3], box[0]:box[2]]
    s_frames = [(img, (120, 120))]     # gt_xy = frame 중심(= align point).
    offset = rr._box_offset_xy(box, 240, 240)   # (52,2) = box_center - frame_center.
    fids = rr._compute_fidelity_from_patch(patch, s_frames, box_offset_xy=offset)
    assert fids and fids[0] > 0.0, (
        f"offset 적용 fidelity={fids} -> 0. box_offset 보정이 동작하지 않음."
    )


def test_fidelity_offset_beats_no_offset_for_offset_box():
    """동일 off-center 박스를 offset 없이(=중심 가정) 평가하면 fidelity 가 떨어진다.

    offset 보정이 fidelity 를 살리는 레버임을 못박는 가드. 보정 없으면 후보가
    gt_xy(=중심) 근처에 없어 0 으로 떨어지고, 보정하면 진짜 위치에서 살아난다.
    """
    img = _frame_with_offset_mark()
    box = (154, 104, 190, 140)
    patch = img[box[1]:box[3], box[0]:box[2]]
    s_frames = [(img, (120, 120))]
    with_offset = rr._compute_fidelity_from_patch(
        patch, s_frames, box_offset_xy=rr._box_offset_xy(box, 240, 240))
    no_offset = rr._compute_fidelity_from_patch(patch, s_frames)   # 기본 (0,0) = 중심 가정.
    assert with_offset[0] > no_offset[0], (
        f"offset {with_offset} 가 no-offset {no_offset} 보다 크지 않음 — 보정 효과 없음."
    )


# ====================================================================
# Task 2: _e_confirm rule + E_CONFIRMED tier weight + thresholds.
# ====================================================================
def test_e_confirm_rule_all_branches():
    """_e_confirm 의 모든 분기: high-S premise + (delta collapse | E-floor) | low-S reject | None 거부."""
    # high-S premise met + delta collapse -> confirm.
    assert rr._e_confirm(0.80, 0.60) is True           # delta 0.20 >= 0.15
    # high-S premise met + E below floor (delta small) -> confirm.
    assert rr._e_confirm(0.62, 0.49) is True            # delta 0.13 < 0.15 but e <= 0.50
    # high-S premise met but E holds up -> no collapse.
    assert rr._e_confirm(0.80, 0.70) is False           # delta 0.10 < 0.15 and e > 0.50
    # low S (no premise) -> never confirm even if E tiny.
    assert rr._e_confirm(0.55, 0.10) is False           # s < 0.60
    # missing reps -> no confirm.
    assert rr._e_confirm(None, 0.10) is False
    assert rr._e_confirm(0.90, None) is False


def test_e_confirmed_tier_is_top_weight():
    """E_CONFIRMED 가 TIER_WEIGHT 에서 최상위 무게이고, risk_score 로도 STRONG 을 앞건다."""
    assert rr.TIER_WEIGHT["E_CONFIRMED"] > rr.TIER_WEIGHT["STRONG"]
    assert rr._risk_score("E_CONFIRMED", 0.2) > rr._risk_score("STRONG", 1.0)


# ====================================================================
# Task 3: _median + _s_rep_score 순수 헬퍼.
# ====================================================================
def test_median_odd_even_and_empty():
    """_median: 홀수면 중앙값, 짝수면 가운데 두 값 평균, 빈 리스트는 None."""
    assert rr._median([0.2, 0.9, 0.5]) == 0.5            # 홀수
    assert rr._median([0.2, 0.8]) == 0.5                  # 짝수 -> 가운데 두 값 평균
    assert rr._median([]) is None


def test_s_rep_score_uses_best_per_frame_median():
    """_s_rep_score: frame_results 각 항목의 best proposer 점수(cand_scores[0])의 median.

    cand_scores 가 비거나 없는 프레임은 skip. 모두 비거나 입력 없으면 None.
    """
    frame_results = [
        {"cand_scores": [0.80, 0.4]},
        {"cand_scores": [0.60, 0.3]},
        {"cand_scores": []},          # 점수 없는 프레임은 skip.
    ]
    assert rr._s_rep_score(frame_results) == 0.70          # median(0.80, 0.60)
    assert rr._s_rep_score([]) is None
    assert rr._s_rep_score([{"cand_scores": []}]) is None


# ====================================================================
# Task 4: _free_search_best_score + _e_rep_score (E proposer).
# ====================================================================
def test_free_search_best_score_localizes_mark():
    """_free_search_best_score: center 템플릿 free-search 최고 proposer 점수.

    자기 자신을 템플릿으로 매칭하면 강한 점수(>0)를 반환해야 한다.
    """
    import numpy as np
    img = _frame_with_offset_mark()
    tpl = rr.build_template(img.copy(), recipe_id="e", version="e", key_type="om")
    score = rr._free_search_best_score(tpl, img)
    assert score is not None and score > 0.0


def test_e_rep_score_median_and_empty():
    """_e_rep_score: E 프레임별 best score(None 제외)의 median.

    동일 프레임 2개의 median은 단일 점수와 같아야 한다.
    E 프레임 없으면(빈 리스트) None을 반환한다.
    """
    img = _frame_with_offset_mark()
    tpl = rr.build_template(img.copy(), recipe_id="e", version="e", key_type="om")
    one = rr._free_search_best_score(tpl, img)
    assert rr._e_rep_score(tpl, [img, img]) == one
    assert rr._e_rep_score(tpl, []) is None


# ====================================================================
# Task 3: Fix-type classifier — _classify_fix
# ====================================================================
def test_classify_fix_ok_when_rcp_distinctive():
    assert rr._classify_fix(0.8, 0.4, distinct_floor=0.7) == "OK"   # rcp >= floor -> OK (cons 무시)


def test_classify_fix_fresh_snapshot_when_region_fine():
    assert rr._classify_fix(0.5, 0.8, distinct_floor=0.7) == "FRESH_SNAPSHOT"


def test_classify_fix_new_region_when_region_ambiguous():
    assert rr._classify_fix(0.4, 0.5, distinct_floor=0.7) == "NEW_REGION"


def test_classify_fix_no_data_when_rcp_none():
    assert rr._classify_fix(None, None, distinct_floor=0.7) == "NO_DATA"
    assert rr._classify_fix(None, 0.9, distinct_floor=0.7) == "NO_DATA"


def test_classify_fix_floor_is_inclusive():
    # rcp_rank1 == floor 이면 OK(>=). cons == floor 이면 FRESH(>=).
    assert rr._classify_fix(0.7, 0.0, distinct_floor=0.7) == "OK"
    assert rr._classify_fix(0.6, 0.7, distinct_floor=0.7) == "FRESH_SNAPSHOT"


# ====================================================================
# Task 5: digest confirmed-count + report s_rep->e_rep column + note
# ====================================================================
def test_digest_includes_confirmed_count():
    import poc.workflow_2.golden_reregister_report_cond as r
    rows_by_mod = {"om": [
        {"recipe": "c/a", "tier": "E_CONFIRMED", "e_confirmed": True, "suggestion": "none"},
        {"recipe": "c/b", "tier": "STRONG", "e_confirmed": False, "suggestion": "box(1,2,3,4)"},
    ], "sem": []}
    d = r._format_digest(rows_by_mod)
    assert "confirmed 1" in d
    assert d == d.encode("ascii", "replace").decode("ascii")   # ASCII only.


def test_report_shows_e_columns_and_note():
    import poc.workflow_2.golden_reregister_report_cond as r
    rows_by_mod = {"om": [
        {"recipe": "c/a", "tier": "E_CONFIRMED", "e_confirmed": True,
         "strong_fail_frac": 0.5, "worst_disp": 0.9, "msr_peak_tail": 0.1,
         "self_ratio": 0.2, "advisory_confidence": "ok", "n_s": 3,
         "suggestion": "none", "sugg_self": None, "sugg_fidelity": None,
         "s_rep": 0.80, "e_rep": 0.55, "n_e": 2},
    ], "sem": []}
    out = r._format_report(rows_by_mod)
    assert "0.800->0.550" in out          # s_rep->e_rep column.
    assert "n_e=2" in out
    assert "E_CONFIRMED rows" in out      # confirmation note line present.


# ====================================================================
# Task 6: run() E-confirm post-pass no-data smoke (regression guard).
# ====================================================================
def test_run_no_data_still_returns_warning_with_e_confirm(monkeypatch, tmp_path):
    """E_CONFIRM=1 이어도 빈 루트에서 no_data 경고를 반환하고 크래시 없음 (post-pass regression guard)."""
    import poc.workflow_2.golden_reregister_report_cond as r
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))   # 빈 루트.
    monkeypatch.setenv("REREGISTER_E_CONFIRM", "1")
    assert r.run() == "[WARNING] no_data"


# ====================================================================
# Task 2 (build 2): consensus rank-1 lookup 순수 헬퍼 + I/O 리더.
# ====================================================================
def test_build_rank1_lookup_basic_and_key_normalization():
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5, "n_S_loo": 6, "cons_pool_n": 8},
        {"recipe": "EQP01/CLSA/REC1", "modality": "om",
         "rcp_rank1_rate": 0.9, "cons_rank1_rate": 0.95, "n_S_loo": 4, "cons_pool_n": 4},
    ]
    lk = rr._build_rank1_lookup(per_recipe)
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.4
    assert lk[("CLSA/REC1", "sem")]["cons_rank1"] == 0.5
    assert lk[("CLSA/REC1", "om")]["rcp_rank1"] == 0.9


def test_build_rank1_lookup_collision_keeps_worst():
    # 두 장비의 같은 class/recipe·modality 가 더블렛으로 충돌 -> 최저 rcp_rank1(보수적) 유지.
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.6, "cons_rank1_rate": 0.6, "n_S_loo": 5, "cons_pool_n": 5},
        {"recipe": "EQP02/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.3, "cons_rank1_rate": 0.4, "n_S_loo": 5, "cons_pool_n": 5},
    ]
    lk = rr._build_rank1_lookup(per_recipe)
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.3


def test_build_rank1_lookup_skips_incomplete_rows():
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem"},  # no rates
        {"recipe": "EQP01/CLSA/REC2", "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5},  # no modality
    ]
    assert rr._build_rank1_lookup(per_recipe) == {}


def test_load_consensus_rank1_empty_on_missing(tmp_path):
    missing = tmp_path / "nope" / "summary.json"
    assert rr._load_consensus_rank1(str(missing)) == {}


def test_load_consensus_rank1_reads_fixture(tmp_path):
    import json
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps({"per_recipe": [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5, "n_S_loo": 6, "cons_pool_n": 8},
    ]}), encoding="utf-8")
    lk = rr._load_consensus_rank1(str(summ))
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.4


# ====================================================================
# Task 4: Worklist priority — _worklist_priority
# ====================================================================
def test_worklist_priority_new_region_above_fresh_above_ok_at_equal_rcp():
    # 같은 rcp_rank1·tier 에서 NEW_REGION > FRESH_SNAPSHOT > OK.
    tw = rr.TIER_WEIGHT["STRONG"]
    p_new = rr._worklist_priority("NEW_REGION", 0.5, tw)
    p_fresh = rr._worklist_priority("FRESH_SNAPSHOT", 0.5, tw)
    p_ok = rr._worklist_priority("OK", 0.5, tw)
    assert p_new > p_fresh > p_ok


def test_worklist_priority_lower_rcp_rank1_ranks_higher():
    tw = rr.TIER_WEIGHT["NONE"]
    assert rr._worklist_priority("NEW_REGION", 0.2, tw) > rr._worklist_priority("NEW_REGION", 0.6, tw)


def test_worklist_priority_no_data_below_equal_tier_backed_flag():
    # NO_DATA(rcp_rank1=None) 은 같은 tier 의 rank-1-backed flag 보다 아래.
    tw = rr.TIER_WEIGHT["MEDIUM"]
    p_nodata = rr._worklist_priority("NO_DATA", None, tw)
    p_ok_backed = rr._worklist_priority("OK", 0.9, tw)   # rank-1-backed, 같은 tier
    assert p_ok_backed > p_nodata


# ====================================================================
# Task 5: Worklist assembly + format + histogram
# ====================================================================
def test_worklist_rows_joins_classifies_and_sorts(_distinct=0.7):
    rows_by_mod = {
        "sem": [
            {"recipe": "CLSA/REC1", "tier": "STRONG", "suggestion": "box=10,10,40,40"},
            {"recipe": "CLSA/REC2", "tier": "NONE", "suggestion": "none"},
        ],
        "om": [
            {"recipe": "CLSA/REC3", "tier": "MEDIUM", "suggestion": "none"},
        ],
    }
    lookup = {
        ("CLSA/REC1", "sem"): {"rcp_rank1": 0.3, "cons_rank1": 0.4, "n_S_loo": 6, "cons_pool_n": 8},
        ("CLSA/REC2", "sem"): {"rcp_rank1": 0.5, "cons_rank1": 0.9, "n_S_loo": 5, "cons_pool_n": 5},
        # REC3 (om) 없음 -> NO_DATA
    }
    wl = rr._worklist_rows(rows_by_mod, lookup, distinct_floor=_distinct)
    by_rec = {(w["recipe"], w["modality"]): w for w in wl}
    assert by_rec[("CLSA/REC1", "sem")]["fix_type"] == "NEW_REGION"
    assert by_rec[("CLSA/REC1", "sem")]["suggested_whitebox"] == "box=10,10,40,40"
    assert by_rec[("CLSA/REC2", "sem")]["fix_type"] == "FRESH_SNAPSHOT"
    assert by_rec[("CLSA/REC3", "om")]["fix_type"] == "NO_DATA"
    # worst-first: NEW_REGION(REC1) 이 FRESH(REC2) 보다 앞.
    order = [(w["recipe"], w["modality"]) for w in wl]
    assert order.index(("CLSA/REC1", "sem")) < order.index(("CLSA/REC2", "sem"))


def test_worklist_rows_fresh_snapshot_has_live_correction_hint():
    rows_by_mod = {"om": [{"recipe": "CLSA/REC1", "tier": "NONE", "suggestion": "none"}]}
    lookup = {("CLSA/REC1", "om"): {"rcp_rank1": 0.5, "cons_rank1": 0.9,
                                    "n_S_loo": 4, "cons_pool_n": 4}}
    wl = rr._worklist_rows(rows_by_mod, lookup, distinct_floor=0.7)
    assert "live-correction" in wl[0]["hint"]


def test_format_worklist_is_ascii_and_excludes_ok_body():
    rows = [
        {"recipe": "CLSA/REC1", "modality": "sem", "rcp_rank1": 0.3, "cons_rank1": 0.4,
         "fix_type": "NEW_REGION", "suggested_whitebox": "box=1,2,3,4", "tier": "STRONG",
         "priority": 2.1, "hint": ""},
        {"recipe": "CLSA/REC9", "modality": "om", "rcp_rank1": 0.95, "cons_rank1": 0.95,
         "fix_type": "OK", "suggested_whitebox": "none", "tier": "NONE",
         "priority": 0.0, "hint": ""},
    ]
    txt = rr._format_worklist(rows)
    assert txt.isascii()
    assert "NEW_REGION" in txt and "CLSA/REC1" in txt
    assert "CLSA/REC9" not in txt          # OK 는 body 제외
    assert "—" not in txt             # em-dash 금지


def test_rank1_histogram_ascii_and_per_modality():
    lookup = {
        ("CLSA/REC1", "sem"): {"rcp_rank1": 0.05, "cons_rank1": 0.4, "n_S_loo": 6, "cons_pool_n": 8},
        ("CLSA/REC2", "sem"): {"rcp_rank1": 0.55, "cons_rank1": 0.5, "n_S_loo": 5, "cons_pool_n": 5},
        ("CLSA/REC3", "om"):  {"rcp_rank1": 0.95, "cons_rank1": 0.9, "n_S_loo": 4, "cons_pool_n": 4},
    }
    h = rr._rank1_histogram(lookup)
    assert h.isascii()
    assert "sem" in h and "om" in h


# ====================================================================
# Task 6: REREGISTER_DISTINCT_FLOOR env 브리지 + _join_coverage_line 형식 테스트.
# ====================================================================
def test_seed_env_bridges_distinct_floor(monkeypatch):
    monkeypatch.delenv("REREGISTER_DISTINCT_FLOOR", raising=False)
    cfg.seed_env()
    assert os.environ["REREGISTER_DISTINCT_FLOOR"] == "0.7"


def test_seed_env_respects_existing_distinct_floor(monkeypatch):
    monkeypatch.setenv("REREGISTER_DISTINCT_FLOOR", "0.55")
    cfg.seed_env()
    assert os.environ["REREGISTER_DISTINCT_FLOOR"] == "0.55"


def test_join_coverage_line_format():
    # M/N 커버리지 문자열은 ASCII 한 줄.
    line = rr._join_coverage_line(3, 5, collisions=1)
    assert line.isascii()
    assert "3/5" in line
