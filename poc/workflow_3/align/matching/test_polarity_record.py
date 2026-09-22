"""극성 반전(OM <-> OM-D)이 남기는 흔적 - `best_ncc` 기록과 sel 상한 (합성 데이터, Mac).

OM-D 는 OM 의 명암 반전이다. chamfer 는 Canny edge 기반이라 반전에 불변이지만 NCC 는 부호가
뒤집혀, selection 의 `max(0, ncc)` 가 그것을 0 으로 누른다. 이 테스트는 그래서 **정답 위에
있어도 sel 이 0.5 를 넘지 못한다**는 것을 합성으로 고정한다(2026-09-22 진단의 실행 가능한 근거).
"""
import numpy as np

from poc.workflow_3.align.matching.engine import (
    STRUCTURE_POLICY,
    compute_align_key_score_ensemble,
)
from poc.workflow_3.align.matching.test_engine_ensemble import _synthetic_template_and_frame


def _match(frame):
    template, _f, _xy = _synthetic_template_and_frame()
    return compute_align_key_score_ensemble(template, frame, policy=STRUCTURE_POLICY)


def test_same_polarity_records_a_positive_ncc():
    _t, frame, _xy = _synthetic_template_and_frame()
    result = _match(frame)
    assert result.best_ncc is not None
    assert result.best_ncc > 0.5


def test_inverted_frame_records_a_negative_ncc():
    _t, frame, _xy = _synthetic_template_and_frame()
    result = _match(255 - frame)
    assert result.best_ncc is not None and result.best_ncc < 0


def test_inverted_frame_cannot_reach_the_match_threshold():
    """반전 상태의 sel 상한 = 0.5·chamfer. match 임계(0.6053)를 구조적으로 못 넘는다."""
    _t, frame, _xy = _synthetic_template_and_frame()
    result = _match(255 - frame)
    assert result.score <= 0.5 * result.chamfer_score + 1e-6
    assert result.score < STRUCTURE_POLICY.ensemble_match_threshold
    assert result.decision != "match"


def test_the_record_does_not_move_the_coordinate():
    """기록 전용이라는 계약 - 같은 프레임에서 좌표/판정이 종전과 같아야 한다."""
    _t, frame, (cx, cy) = _synthetic_template_and_frame()
    result = _match(frame)
    assert abs(result.best_xy[0] - cx) <= 4 and abs(result.best_xy[1] - cy) <= 4
    assert result.decision in ("match", "adjust")
