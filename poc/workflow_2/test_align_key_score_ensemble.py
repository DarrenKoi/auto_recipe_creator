"""compute_align_key_score_ensemble + rescore/rerank 단위테스트 (합성 데이터, Mac)."""
import numpy as np

from poc.workflow_2 import align_key_matcher as akm
from poc.workflow_2.align_key_matcher import (
    AlignKeyCandidate,
    _chamfer_score_map_at_scale,
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


def test_ensemble_pool_rerank_prefers_orb_favored(monkeypatch):
    """ORB 가 chamfer-비최상 후보를 우세하게 만들면 best_xy 가 그 후보가 되어야."""
    template, frame, _ = _synthetic_template_and_frame()

    pos_a = (80, 70)
    pos_b = (200, 150)

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

    from poc.workflow_2.align_key_matcher import MatchPolicy
    policy = MatchPolicy(chamfer_weight=0.2, orb_weight=0.8)

    def _fake_crop(frame_img, cx, cy, tw, th, *, pad=1.5):
        return np.full((4, 4), 1 if (cx, cy) == pos_b else 0, dtype=np.uint8), (0, 0)

    def _orb_by_tag(tmpl_img, crop, **k):
        return (1.0, 10, 10) if int(crop[0, 0]) == 1 else (0.0, 0, 0)

    monkeypatch.setattr(akm, "_crop_with_padding", _fake_crop)
    monkeypatch.setattr(akm, "compute_orb_inlier_ratio", _orb_by_tag)

    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,), policy=policy)
    # combined: pos_a=0.2*0.9+0.8*0=0.18, pos_b=0.2*0.5+0.8*1.0=0.90 → pos_b 승.
    assert result.best_xy == pos_b


def test_ensemble_rejects_invalid_scales():
    template, frame, _ = _synthetic_template_and_frame()
    try:
        compute_align_key_score_ensemble(template, frame, scales=())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty scales")
