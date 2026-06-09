"""compute_align_key_score_ensemble + rescore/rerank 단위테스트 (합성 데이터, Mac)."""
import numpy as np

from poc.workflow_3.vision import align_key_matcher as akm
from poc.workflow_3.vision.align_key_matcher import (
    AlignKeyCandidate,
    DEFAULT_POLICY,
    _chamfer_score_map_at_scale,
    _decision_for_score,
    _rescore_positions_to_candidates,
    build_template,
    compute_align_key_score_ensemble,
)


def _synthetic_template_and_frame():
    """프레임 중앙에 박힌 사각형 패턴 + 동일 패턴 템플릿(합성)."""
    rng = np.random.RandomState(0)
    frame = (rng.rand(240, 320) * 40).astype(np.uint8)
    # 뚜렷한 구조: 중앙 사각형 테두리.
    cy, cx = 120, 160
    frame[cy - 30:cy + 30, cx - 30] = 255
    frame[cy - 30:cy + 30, cx + 30] = 255
    frame[cy - 30, cx - 30:cx + 30] = 255
    frame[cy + 30, cx - 30:cx + 30] = 255
    tmpl_img = frame[cy - 40:cy + 40, cx - 40:cx + 40].copy()
    template = build_template(tmpl_img, recipe_id="t", version="v")
    return template, frame, (cx, cy)


def test_rescore_matches_direct_score_map_lookup():
    template, frame, _ = _synthetic_template_and_frame()
    _edges, frame_dt = akm.preprocess_for_matching(frame)
    scale = 1.0
    score_map, (tw, th) = _chamfer_score_map_at_scale(template.edge_map, frame_dt, scale)
    # 맵 유효 영역 안의 임의 center 위치.
    x0, y0 = 50, 40
    cx, cy = x0 + tw // 2, y0 + th // 2
    expected = float(score_map[y0, x0])

    cands = _rescore_positions_to_candidates(template, frame_dt, [((cx, cy), scale)])

    assert len(cands) == 1
    assert isinstance(cands[0], AlignKeyCandidate)
    assert cands[0].xy == (cx, cy)
    assert cands[0].template_size == (tw, th)
    assert abs(cands[0].chamfer_score - expected) < 1e-6
    assert cands[0].score == cands[0].chamfer_score


def test_rescore_out_of_bounds_scores_zero():
    template, frame, _ = _synthetic_template_and_frame()
    _edges, frame_dt = akm.preprocess_for_matching(frame)
    cands = _rescore_positions_to_candidates(template, frame_dt, [((-999, -999), 1.0)])
    assert len(cands) == 1
    assert cands[0].chamfer_score == 0.0


def test_ensemble_returns_valid_result_shape():
    template, frame, (cx, cy) = _synthetic_template_and_frame()
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
    assert isinstance(result, akm.AlignKeyMatchResult)
    assert 0 <= result.best_xy[0] < frame.shape[1]
    assert 0 <= result.best_xy[1] < frame.shape[0]
    assert result.decision in ("match", "adjust", "low")
    assert isinstance(result.candidates, list)


def test_ensemble_no_candidates_returns_reject(monkeypatch):
    template, frame, _ = _synthetic_template_and_frame()

    class _Empty:
        fused = []

    monkeypatch.setattr(akm, "compute_ensemble_candidates", lambda *a, **k: _Empty())
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
    assert result.reject_reason == "no_candidates"
    assert result.candidates == []
    assert result.score == 0.0


def test_ensemble_selection_prefers_ncc_favored(monkeypatch):
    """NCC 가 chamfer-비최상 후보를 우세하게 만들면 best_xy 가 그 후보가 되어야(NCC selection)."""
    template, frame, _ = _synthetic_template_and_frame()

    pos_a = (80, 70)    # chamfer 높음, NCC 낮음 (decoy)
    pos_b = (200, 150)  # chamfer 낮음, NCC 높음 (truth)

    class _Cand:
        def __init__(self, xy):
            self.xy = xy
            self.scale = 1.0

    class _Ens:
        fused = [_Cand(pos_a), _Cand(pos_b)]

    monkeypatch.setattr(akm, "compute_ensemble_candidates", lambda *a, **k: _Ens())

    def _fake_rescore(tpl, fdt, positions):
        out = []
        for (cx, cy), scale in positions:
            ch = 0.9 if (cx, cy) == pos_a else 0.5
            out.append(akm.AlignKeyCandidate(
                score=ch, chamfer_score=ch, xy=(cx, cy), scale=1.0, template_size=(40, 40)))
        return out

    monkeypatch.setattr(akm, "_rescore_positions_to_candidates", _fake_rescore)

    # NCC 가 pos_b(truth)만 높게 — selection 이 chamfer-top(pos_a)을 누르고 pos_b 를 골라야.
    def _fake_ncc(template_raw, frame_img, xy, scale):
        return 0.9 if tuple(xy) == pos_b else 0.0

    monkeypatch.setattr(akm, "_candidate_ncc", _fake_ncc)

    from poc.workflow_3.vision.align_key_matcher import MatchPolicy
    policy = MatchPolicy(rerank_chamfer_w=0.5, rerank_ncc_w=0.5)
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,), policy=policy)
    # sel: pos_a=0.5*0.9+0.5*0=0.45, pos_b=0.5*0.5+0.5*0.9=0.70 → pos_b 승.
    assert result.best_xy == pos_b
    # decision/score 정비: result.score == best_sel(=chamfer+NCC), ORB 제거(orb=0).
    assert abs(result.score - 0.70) < 1e-9
    assert result.orb_inlier_ratio == 0.0
    # 0.70 >= ensemble_match_threshold(0.6053) → match.
    assert result.decision == "match"


def test_decision_for_score_threshold_override():
    """_decision_for_score: threshold override 가 policy 보다 우선, 미지정 시 policy."""
    # override 우선 — policy 기본 임계와 무관하게 전달값으로 판정.
    assert _decision_for_score(0.5, match_threshold=0.4, adjust_threshold=0.3) == "match"
    assert _decision_for_score(0.35, match_threshold=0.4, adjust_threshold=0.3) == "adjust"
    assert _decision_for_score(0.2, match_threshold=0.4, adjust_threshold=0.3) == "low"
    # override 미지정 → policy.match/adjust_threshold (기존 동작 불변).
    p = DEFAULT_POLICY
    assert _decision_for_score(p.match_threshold, p) == "match"
    assert _decision_for_score(p.adjust_threshold, p) == "adjust"
    assert _decision_for_score(p.adjust_threshold - 1e-6, p) == "low"


def test_finalize_match_score_override():
    """_finalize_match: score_override 미지정 → chamfer+ORB(baseline 불변), 지정 → 그 값+전달 threshold."""
    template, frame, _ = _synthetic_template_and_frame()
    cand = AlignKeyCandidate(
        score=0.5, chamfer_score=0.5, xy=(160, 120), scale=1.0, template_size=(40, 40))
    # 기본(override 없음) → policy.chamfer_weight·chamfer + orb_weight·orb (baseline 경로).
    r0 = akm._finalize_match(
        cand, [cand], frame, template, DEFAULT_POLICY, (0, 0),
        chamfer_score=0.5, orb_ratio=0.2)
    expected = DEFAULT_POLICY.chamfer_weight * 0.5 + DEFAULT_POLICY.orb_weight * 0.2
    assert abs(r0.score - expected) < 1e-9
    # override → 그 값 그대로, decision 은 전달 threshold 로(0.71 >= 0.7 → match).
    r1 = akm._finalize_match(
        cand, [cand], frame, template, DEFAULT_POLICY, (0, 0),
        chamfer_score=0.5, orb_ratio=0.2,
        score_override=0.71, decision_thresholds=(0.7, 0.4))
    assert abs(r1.score - 0.71) < 1e-9
    assert r1.decision == "match"


def test_ensemble_guard_uses_topn_scope_not_shadow(monkeypatch):
    """no_candidates guard 는 selection 과 동일 범위(top_n)여야 — shadow 슬롯의 양 chamfer 가
    guard 를 통과시키면서 selection 은 zero-chamfer top_n 에서 고르는 불일치 방지."""
    template, frame, _ = _synthetic_template_and_frame()

    # top_n(8) 개의 zero-chamfer 위치 + 1 개의 양 chamfer shadow(9번째).
    top_positions = [(10 + i, 10 + i) for i in range(8)]
    shadow_pos = (200, 150)

    class _Cand:
        def __init__(self, xy):
            self.xy = xy
            self.scale = 1.0

    class _Ens:
        fused = [_Cand(xy) for xy in top_positions] + [_Cand(shadow_pos)]

    monkeypatch.setattr(akm, "compute_ensemble_candidates", lambda *a, **k: _Ens())

    def _fake_rescore(tpl, fdt, positions):
        out = []
        for (cx, cy), scale in positions:
            ch = 0.5 if (cx, cy) == shadow_pos else 0.0   # 오직 shadow 만 양 chamfer.
            out.append(akm.AlignKeyCandidate(
                score=ch, chamfer_score=ch, xy=(cx, cy), scale=1.0, template_size=(40, 40)))
        return out

    monkeypatch.setattr(akm, "_rescore_positions_to_candidates", _fake_rescore)
    # top_n=8 로 cap 되면 shadow 는 rescore 에 안 들어가고 top-8 전부 zero → no_candidates.
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
    assert result.reject_reason == "no_candidates"


def test_ensemble_rejects_invalid_scales():
    template, frame, _ = _synthetic_template_and_frame()
    try:
        compute_align_key_score_ensemble(template, frame, scales=())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty scales")
