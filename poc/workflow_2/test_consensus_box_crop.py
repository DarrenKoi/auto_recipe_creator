"""box-crop align_offset 매핑 테스트.

box template 의 match 중심(box center)에 align_offset 을 가산하여 align point 를 계산하는
_gt_in_topk 행동 검증.
"""

from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poc.workflow_2 import align_similarity as alsim
from poc.workflow_3.align.matching.engine import build_template, AlignKeyCandidate


def _dummy_template(r=10):
    """간단한 더미 template."""
    tpl_img = np.full((2 * r, 2 * r), 100, np.uint8)
    return tpl_img


def test_gt_in_topk_applies_align_offset_to_reach_align_point():
    """offset 이 있는 box template: match 중심(box center)에 offset 을 더해야 truth(align point)와 일치.

    mock candidate (128,128)을 만들고, offset=(20,-10) scale=1.0 이면
    align point = (128,128) + (20,-10) = (148,118) 로 truth 에 부딪혀야 함.
    """
    frame = np.full((256, 256), 50, np.uint8)
    tpl = build_template(_dummy_template(r=10), recipe_id="R", version="box", key_type="sem")
    tpl.align_offset_xy = (20, -10)   # align point = box center + (20,-10) = (148,118)
    truth_align_point = (148, 118)

    # mock _propose_topk 을 덮어서 (128,128) 에 후보를 반환
    mock_cand = AlignKeyCandidate(score=0.9, chamfer_score=0.9, xy=(128, 128),
                                   scale=1.0, template_size=(20, 20))
    with mock.patch('poc.workflow_2.align_similarity._propose_topk') as mock_propose:
        mock_propose.return_value = [mock_cand]
        out = alsim._gt_in_topk(frame, truth_align_point, {"sem": tpl},
                                scales=(1.0,), topk=8)

    assert out is not None
    assert out["in_topk"] is True, "offset 적용 시 box-center match 가 align point 로 매핑돼 truth 히트해야 함"


def test_gt_in_topk_zero_offset_unchanged():
    """offset (0,0)(center template): align point == match center — 기존 동작 유지(회귀 가드).

    candidate (128,128) 이 그대로 truth (128,128) 와 매칭돼야 함.
    """
    frame = np.full((256, 256), 50, np.uint8)
    tpl = build_template(_dummy_template(r=10), recipe_id="R", version="center", key_type="sem")
    # align_offset_xy 기본 (0,0). truth = (128,128).
    truth_xy = (128, 128)

    mock_cand = AlignKeyCandidate(score=0.9, chamfer_score=0.9, xy=(128, 128),
                                   scale=1.0, template_size=(20, 20))
    with mock.patch('poc.workflow_2.align_similarity._propose_topk') as mock_propose:
        mock_propose.return_value = [mock_cand]
        out = alsim._gt_in_topk(frame, truth_xy, {"sem": tpl}, scales=(1.0,), topk=8)

    assert out is not None
    assert out["in_topk"] is True
