"""consensus_template.build_consensus_template 의 가드/빌드 동작 검증.

핵심 계약: 이 함수는 "consensus 를 쓸 수 있나"를 판단하는 *게이트*다. 쓸 수 없으면
template=None 을 돌려주고(예외 아님), 호출부는 그 신호로 rcp center template 으로 폴백한다.
따라서 다음 4가지를 못박는다.
  (a) 같은 modality crop ≥ min_s 이고 선명하면 → AlignKeyTemplate 반환(reason=ok).
  (b) crop < min_s → None(reason=insufficient_s).  [캐시 부족/측정 이벤트 적음 = rcp 폴백]
  (c) median 이 흐림(정렬 어긋남) → None(reason=blurry).  [blur 가드 = 런타임 폴백 신호]
  (d) modality 가 template/result 에 보존.

합성 crop 만으로 검증(오피스 데이터 불필요, Mac 에서 실행). cv2 필요 → --extra dev.
"""

import numpy as np
import pytest

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_2.consensus_template import (
    DEFAULT_CONSENSUS_POLICY,
    build_consensus_template,
    select_routing_templates,
)


def _bar_crop(size=64, x=30, width=4):
    """세로 흰 막대 한 줄이 박힌 sharp gray crop. x 로 위치를 옮긴다.

    같은 x 면 median 이 막대를 그대로 보존(선명), 서로 다른 x 면 어느 열에서도 막대가
    과반에 못 들어 median 이 평평해진다(흐림) → blur 가드 테스트용.
    """
    img = np.zeros((size, size), np.uint8)
    img[:, x:x + width] = 255
    return img


def test_three_aligned_sharp_crops_yield_template():
    """정렬된 sharp crop 3장 → template 반환, reason=ok, blur 비율≈1."""
    crops = [_bar_crop(x=30) for _ in range(3)]

    res = build_consensus_template(crops, recipe_id="EQP/CLS/RCP", modality="sem")

    assert res.template is not None
    assert res.reason == "ok"
    assert res.n_crops == 3
    # 정렬된 동일 막대 → median 이 곧 막대 → 개별 대비 선명도 손실 없음.
    assert res.edge_ratio >= DEFAULT_CONSENSUS_POLICY.edge_ratio_min
    assert res.lap_ratio >= DEFAULT_CONSENSUS_POLICY.lap_ratio_min


def test_fewer_than_min_s_returns_none():
    """crop 이 min_s 미만이면 template=None, reason=insufficient_s (rcp 폴백 신호)."""
    crops = [_bar_crop(x=30), _bar_crop(x=30)]   # 2장 < 기본 min_s=3.

    res = build_consensus_template(crops, recipe_id="EQP/CLS/RCP", modality="sem")

    assert res.template is None
    assert res.reason == "insufficient_s"
    assert res.n_crops == 2


def test_empty_crops_returns_none():
    """crop 0장 → None(reason=insufficient_s), 크래시 없음."""
    res = build_consensus_template([], recipe_id="EQP/CLS/RCP", modality="sem")

    assert res.template is None
    assert res.reason == "insufficient_s"
    assert res.n_crops == 0


def test_misaligned_crops_fail_blur_guard():
    """막대 위치가 제각각 → median 이 평평(흐림) → None, reason=blurry, 비율은 보고됨."""
    crops = [_bar_crop(x=10), _bar_crop(x=30), _bar_crop(x=50)]

    res = build_consensus_template(crops, recipe_id="EQP/CLS/RCP", modality="sem")

    assert res.template is None
    assert res.reason == "blurry"
    # 가드가 무엇을 보고 떨궜는지 audit 용으로 비율은 채워져야 한다.
    assert res.edge_ratio is not None
    assert res.lap_ratio is not None
    assert (res.edge_ratio < DEFAULT_CONSENSUS_POLICY.edge_ratio_min
            or res.lap_ratio < DEFAULT_CONSENSUS_POLICY.lap_ratio_min)


def test_modality_preserved_in_result_and_template():
    """modality 가 result 와 AlignKeyTemplate.key_type 에 그대로 보존."""
    crops = [_bar_crop(x=30) for _ in range(3)]

    res = build_consensus_template(crops, recipe_id="EQP/CLS/RCP", modality="om")

    assert res.modality == "om"
    assert res.template is not None
    assert res.template.key_type == "om"
    assert res.template.version == "s_consensus_prod"


def _rcp_tpl(modality):
    """rcp center template 스탠드인(폴백 검증용)."""
    return build_template(_bar_crop(x=30), recipe_id="R",
                          version="rcp_center", key_type=modality)


def test_select_prefers_consensus_falls_back_to_rcp_per_modality():
    """SEM 은 consensus, OM 은 consensus 없음 → rcp 폴백. 키는 OM/SEM 로 정규화."""
    cons_sem = build_consensus_template(
        [_bar_crop(x=30) for _ in range(3)], recipe_id="R", modality="sem").template
    rcp_om = _rcp_tpl("om")
    rcp_sem = _rcp_tpl("sem")

    routing = select_routing_templates(
        consensus_by_mod={"sem": cons_sem, "om": None},
        rcp_by_mod={"sem": rcp_sem, "om": rcp_om},
    )

    assert routing["SEM"].version == "s_consensus_prod"   # consensus 우선.
    assert routing["OM"] is rcp_om                         # consensus None → rcp 폴백.


def test_select_omits_modality_when_no_template_available():
    """consensus·rcp 둘 다 없으면 그 modality 는 라우팅 dict 에서 빠진다."""
    routing = select_routing_templates(
        consensus_by_mod={"sem": None},
        rcp_by_mod={"sem": None, "om": None},
    )

    assert routing == {}
