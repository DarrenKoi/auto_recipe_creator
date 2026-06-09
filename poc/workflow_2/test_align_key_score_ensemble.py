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
