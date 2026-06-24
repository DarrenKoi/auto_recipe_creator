"""Template-bank matcher bench fork 테스트 (합성 데이터, Mac). workflow_3 미수정 fork."""
import os

import numpy as np

import poc.workflow_2.golden_eval_config_loader as cfg


def test_seed_env_bridges_tbank_defaults(monkeypatch):
    for k in ("TBANK_HEATMAP", "TBANK_RRF", "TBANK_PEAK_NMS_FRAC",
              "TBANK_CLUSTER_TOL_FRAC", "TBANK_RRF_K"):
        monkeypatch.delenv(k, raising=False)
    cfg.seed_env()
    assert os.environ["TBANK_HEATMAP"] == "1"
    assert os.environ["TBANK_RRF"] == "1"
    assert os.environ["TBANK_PEAK_NMS_FRAC"] == "0.5"
    assert os.environ["TBANK_CLUSTER_TOL_FRAC"] == "0.1"
    assert os.environ["TBANK_RRF_K"] == "60"


def _mark_crop(seed, size=64):
    """저텍스처 배경 + seed 위치 고유 마크가 있는 합성 gray crop."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size) * 30 + 20).astype(np.uint8)
    cx = cy = size // 2
    img[cy - 6:cy + 6, cx - 1:cx + 1] = 230   # 십자 마크(고유 구조).
    img[cy - 1:cy + 1, cx - 6:cx + 6] = 230
    return img


def test_bank_build_keeps_members_individual():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    assert len(bank) == 4                     # median 으로 합치지 않고 N 개 유지.
    assert all(hasattr(m, "edge_map") for m in bank)
    assert all(m.key_type == "om" for m in bank)


def test_bank_build_respects_min_s():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(2)]
    assert tb.bank_build(crops, recipe_id="r", modality="om", min_s=3) == []


def _frame_with_mark(mark_xy, distractor_xy=None, size=256, seed=0):
    """저텍스처 frame + (mark_xy 에 십자) (+ distractor_xy 에 동일 십자).

    배경을 flat(30) 으로 유지해 chamfer DT 가 십자 엣지에만 집중하도록 한다.
    random noise 배경은 spurious edge 를 만들어 점수면을 평평하게 만들기 때문.
    """
    img = np.full((size, size), 30, dtype=np.uint8)

    def _cross(cx, cy):
        img[cy - 6:cy + 6, cx - 1:cx + 1] = 230
        img[cy - 1:cy + 1, cx - 6:cx + 6] = 230

    _cross(*mark_xy)
    if distractor_xy is not None:
        _cross(*distractor_xy)
    return img


def test_heatmap_positive_one_member_distractor():
    """1/N 멤버만 distractor 에 끌려도 합산 peak 는 참 마크에 안착."""
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    frame = _frame_with_mark((120, 120))             # 참 마크 1곳만.
    res = tb.bank_match_heatmap(bank, frame)
    assert res.xy is not None
    assert abs(res.xy[0] - 120) <= 8 and abs(res.xy[1] - 120) <= 8


def test_heatmap_h0_all_members_same_distractor():
    """모든 멤버가 같은 distractor 에 끌리면 합산 peak 도 distractor (실패모드 검출)."""
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    # 참 마크(60,60) 약하게, distractor(190,190) 강하게(모든 멤버 공통).
    frame = _frame_with_mark((190, 190))
    res = tb.bank_match_heatmap(bank, frame)
    assert res.xy is not None
    assert abs(res.xy[0] - 190) <= 8 and abs(res.xy[1] - 190) <= 8


def test_dedup_within_member_collapses_near_duplicates():
    import poc.workflow_2.template_bank_lab as tb

    class C:
        def __init__(self, xy, score):
            self.xy, self.score, self.scale = xy, score, 1.0
    cands = [C((100, 100), 0.9), C((103, 101), 0.5), C((180, 180), 0.7)]
    kept = tb._dedup_within_member(cands, tol=10)
    assert len(kept) == 2                       # (100,100)+(103,101) 한 표로 병합.
    assert kept[0].xy == (100, 100)             # 클러스터 대표 = 최고점.


def test_rrf_positive_one_member_distractor():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    frame = _frame_with_mark((120, 120))
    res = tb.bank_match_rrf(bank, frame, cluster_tol=10, rrf_k=60)
    assert res.xy is not None
    assert abs(res.xy[0] - 120) <= 10 and abs(res.xy[1] - 120) <= 10
    assert res.member_support is not None and res.member_support[0] >= 1


def test_heatmap_recovers_consistent_peak_individual_members_miss():
    """SUM 누적이 '개별 멤버는 1등으로 안 꼽지만 일관된 약 peak' 를 복원 — heatmap-primary 의
    핵심 근거(RRF 가 통과 못 하는 케이스). map 레벨에서 결정적으로 검증(취약한 실이미지 chamfer 회피)."""
    import poc.workflow_2.template_bank_lab as tb
    H = W = 64
    n_members = 4
    true_xy = (32, 32)
    acc = np.zeros((H, W), dtype=np.float32)
    member_maps = []
    for i in range(n_members):
        m = np.zeros((H, W), dtype=np.float32)
        m[true_xy[1], true_xy[0]] = 0.4                  # 모든 멤버 공통 약 peak.
        m[50, 10 + i * 8] = 1.0                          # 멤버별 고유 강 peak(서로 다른 위치).
        member_maps.append(m)
        acc += m
    # 개별 멤버의 argmax 는 각자 distractor(1.0 > 0.4) — 참 위치를 1등으로 안 꼽는다.
    for m in member_maps:
        assert np.unravel_index(int(np.argmax(m)), m.shape) != (true_xy[1], true_xy[0])
    # 합산: 참 위치 4*0.4=1.6 > 각 distractor 1.0 -> _peaks_center 가 참 위치를 1등으로.
    peaks = tb._peaks_center(acc, nms_radius=3, max_peaks=5, min_score=0.0)
    assert peaks and (peaks[0][1], peaks[0][2]) == true_xy


def test_classify_winner_buckets():
    import poc.workflow_2.template_bank_lab as tb
    gt = (100, 100)
    # correct: within tol.
    assert tb.classify_winner((104, 99), gt, period=40, tol_px=8) == "correct"
    # near_periodic: off by ~one period along x.
    assert tb.classify_winner((140, 100), gt, period=40, tol_px=8) == "near_periodic"
    # far_wrong: residual from nearest period multiple exceeds tol.
    assert tb.classify_winner((189, 100), gt, period=40, tol_px=8) == "far_wrong"
    # one_member_only overrides (rrf arm) when support==1.
    assert tb.classify_winner((104, 99), gt, period=40, tol_px=8,
                              member_support=1) == "one_member_only"


def test_estimate_lattice_period_detects_stripes():
    import poc.workflow_2.template_bank_lab as tb
    from poc.workflow_3.align.matching.engine import build_template
    size, p = 96, 12
    img = np.zeros((size, size), np.uint8)
    img[:, ::p] = 220                                # 주기 p 세로 줄무늬.
    tpl = build_template(img, recipe_id="r", version="r", key_type="om")
    per = tb.estimate_lattice_period(tpl)
    assert per is not None and abs(per - p) <= 3


def test_bootstrap_ci_is_deterministic_and_bracketed():
    import poc.workflow_2.golden_consensus_eval_cond as g
    vals = [0.0] * 50 + [1.0] * 50            # mean 0.5.
    lo, hi = g._bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert g._bootstrap_ci(vals, n_boot=500, seed=1) == (lo, hi)   # seed 고정 결정적.
    import math
    a, b = g._bootstrap_ci([], n_boot=10, seed=1)
    assert math.isnan(a) and math.isnan(b)


def test_aggregate_buckets_counts():
    import poc.workflow_2.golden_consensus_eval_cond as g
    labels = ["correct", "correct", "near_periodic", "far_wrong", "one_member_only"]
    d = g._aggregate_buckets(labels)
    assert d["correct"] == 2 and d["near_periodic"] == 1 and d["total"] == 5


def test_format_bank_digest_ascii_one_line():
    import poc.workflow_2.golden_consensus_eval_cond as g
    stats = {"om": {"heatmap_in_topk": 0.71, "cons_in_topk": 0.66, "near_periodic": 0.05},
             "sem": {"heatmap_in_topk": 0.70, "cons_in_topk": 0.66, "near_periodic": 0.30}}
    d = g._format_bank_digest(stats)
    assert d.startswith("[DIGEST] template-bank")
    assert "om[" in d and "sem[" in d and "\n" not in d
    assert d == d.encode("ascii", "replace").decode("ascii")


def test_consensus_run_no_data_with_tbank(monkeypatch, tmp_path):
    """TBANK on + 빈 루트에서 run() 이 no_data/no_ab 로 정상 반환 (no-data smoke)."""
    import poc.workflow_2.golden_consensus_eval_cond as g
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))
    monkeypatch.setenv("TBANK_HEATMAP", "1")
    out = g.run()
    assert out in ("no_data", "no_ab")
