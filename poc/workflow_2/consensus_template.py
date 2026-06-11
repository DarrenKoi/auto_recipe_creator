"""consensus(최근 S median) template 빌더 — 검증된 레버의 *프로덕션* 진입점.

배경
----
PROPOSER_WALL: 단일 rcp 등록 key 는 공정 드리프트로 stale 해져 matcher 후보(top-N)에
진실(align point)이 자주 빠진다([[project_matcher_flat_chamfer_distinctiveness]]). recipe 의
*최근 성공(S)* crop 들을 crosshair 로 정렬해 median 을 뜨면 현재 외형을 추종해 후보 진입률이
오른다(cond A/B: in_topk 0.434→0.876, rank1 0.318→0.764, 저널 260608_163302).

이 함수의 책임 = **게이트**
----------------------------
consensus 는 rcp 위에 얹는 *옵션*이지 대체가 아니다. "이 consensus 를 믿어도 되나"를 판단해
못 믿으면 ``template=None`` 을 돌려준다(예외 아님). 호출부는 그 신호로 rcp center template 으로
폴백한다(``consensus or rcp``). 즉 캐시 부족·blur 등 어떤 사유든 최악의 경우 = 현재 검증된
rcp 베이스라인(in_topk 0.434) → **회귀 위험 0**.

폴백을 부르는 사유(모두 None 반환):
  - ``insufficient_s`` : 같은 modality crop < min_s(기본 3). 측정 이벤트가 적어 median 이
    의미 없음([[project_consensus_sparse_golden_and_recipe_id_collision]]).
  - ``blurry``         : median 이 흐림(정렬 sub-pixel 어긋남 → ghosting). edge/lap 선명도
    비율이 임계 미만. consensus 의 유일한 구조적 실패 모드(저널 260609_045819).

로직 중복/표류 방지를 위해 median(``_consensus``)·선명도 지표(``_edge_density``/``_lap_var``)는
검증된 ``align_similarity`` 의 것을, template 조립은 ``align_key_matcher.build_template`` 을
그대로 재사용한다(eval A/B 와 동일 코드 경로).

입력 crop 규약
--------------
``crops`` 는 **한 modality**의, **이미 정제(crosshair/box 제거)·crosshair 중심 crop·co-registration
까지 끝난** 동일 크기 gray 배열들이어야 한다(=eval ``_build_cond_by_recipe`` 의 ``s_frames[].crop``
과 동일 전처리). 이 crop 파이프라인은 §2-F 리팩터에서 공유 모듈로 추출 예정.
"""

import statistics
from dataclasses import dataclass

import numpy as np

from poc.workflow_3.align.matching.engine import AlignKeyTemplate, build_template
from poc.workflow_2.align_similarity import _consensus, _edge_density, _lap_var

CONSENSUS_VERSION = "s_consensus_prod"   # 로그/overlay 에서 rcp 와 구분(저널 163302 §4-B #4).


@dataclass(frozen=True)
class ConsensusPolicy:
    """consensus 게이트 임계. 기본값은 저널 260608_163302 §4-A #7 / 260609_045819 근거.

    blur 임계(0.70/0.50)는 golden 데이터 1회 실측으로 확정 예정(저널 260609_045819 §5).
    """

    min_s: int = 3                # 같은 modality 최소 S crop 장수(이 golden set sparse → 3).
    edge_ratio_min: float = 0.70  # consensus edge density / 개별 crop median. 미만이면 흐림.
    lap_ratio_min: float = 0.50   # consensus Laplacian 분산 비율. 미만이면 흐림.


DEFAULT_CONSENSUS_POLICY = ConsensusPolicy()


@dataclass
class ConsensusResult:
    """build_consensus_template 의 결과 + audit 데이터.

    ``template`` 이 None 이면 폴백 신호. ``reason`` 과 선명도 비율로 *왜* 떨궜는지 기록해
    캐시 meta(저널 053556 §2-C)·진단에 남긴다.
    """

    template: AlignKeyTemplate | None   # None = rcp center 로 폴백하라는 신호.
    modality: str
    n_crops: int                         # 게이트에 들어온 같은 modality crop 장수.
    edge_ratio: float | None             # consensus / 개별 crop median (None = 산출 불가).
    lap_ratio: float | None
    reason: str                          # "ok" | "insufficient_s" | "blurry".


def _sharpness_ratio(metric_fn, consensus, crops):
    """consensus 선명도 ÷ 개별 crop 선명도 median. 분모 0(featureless)이면 None.

    분모 = "consensus 를 만든 재료 S 한 장" 이라 apples-to-apples(저널 260609_045819 §2).
    """
    c_val = metric_fn(consensus)
    s_med = statistics.median([metric_fn(c) for c in crops])
    if s_med <= 0:
        return None
    return round(c_val / s_med, 4)


def build_consensus_template(crops, *, recipe_id, modality,
                             policy=DEFAULT_CONSENSUS_POLICY):
    """정제·정렬된 같은 modality S crop 들로 consensus template 을 짓는다(게이트 포함).

    Args:
        crops: 동일 크기 gray crop 리스트(한 modality, 정제+crosshair중심+co-reg 완료).
        recipe_id: AlignKeyTemplate.recipe_id 로 들어갈 식별자(eqp/class/recipe 권장).
        modality: 'om' | 'sem' — template.key_type 및 라우팅 키.
        policy: 게이트 임계(min_s/blur).

    Returns:
        ConsensusResult. ``template`` 이 None 이면 호출부는 rcp center 로 폴백한다.
    """
    n = len(crops)
    if n < policy.min_s:
        # 캐시/측정 이벤트 부족 — median 이 의미 없으니 rcp 폴백.
        return ConsensusResult(None, modality, n, None, None, "insufficient_s")

    consensus = _consensus(crops)
    edge_ratio = _sharpness_ratio(_edge_density, consensus, crops)
    lap_ratio = _sharpness_ratio(_lap_var, consensus, crops)

    # blur 가드: 어느 한 비율이라도 임계 미만(또는 산출 불가)이면 median 이 흐린 것 → 폴백.
    edge_bad = edge_ratio is None or edge_ratio < policy.edge_ratio_min
    lap_bad = lap_ratio is None or lap_ratio < policy.lap_ratio_min
    if edge_bad or lap_bad:
        return ConsensusResult(None, modality, n, edge_ratio, lap_ratio, "blurry")

    template = build_template(consensus, recipe_id=recipe_id,
                              version=CONSENSUS_VERSION, key_type=modality)
    return ConsensusResult(template, modality, n, edge_ratio, lap_ratio, "ok")


# modality(빌더 규약) → live_align_search route_template 의 키 규약.
_MOD_TO_ROUTE_KEY = {"om": "OM", "sem": "SEM"}


def select_routing_templates(consensus_by_mod, rcp_by_mod):
    """live_align_search 에 넘길 라우팅 dict 을 *consensus 우선·rcp 폴백*으로 조립.

    저널 163302 §4-A #6 — 폴백은 route_template 을 고치지 않고 dict 구성 시점에서 한다
    (consensus 가 None 이면 그 modality 만 rcp 로). 둘 다 없으면 그 modality 는 빠진다.

    Args:
        consensus_by_mod: {'om'|'sem': AlignKeyTemplate | None}. None = 게이트 폴백 신호.
        rcp_by_mod:       {'om'|'sem': AlignKeyTemplate | None}. 베이스라인 center template.

    Returns:
        {'OM'|'SEM': AlignKeyTemplate} — live_align_search(templates=...) 에 바로 사용.
    """
    out = {}
    for mod, route_key in _MOD_TO_ROUTE_KEY.items():
        chosen = consensus_by_mod.get(mod) or rcp_by_mod.get(mod)   # consensus 우선, 폴백 rcp.
        if chosen is not None:
            out[route_key] = chosen
    return out
