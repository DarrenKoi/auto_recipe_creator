# poc/workflow_3/align/consensus_template.py
"""consensus(최근 S median) template 빌더 — 검증된 레버의 프로덕션 진입점.

bench(poc/workflow_2/consensus_template.py)에서 bit-parity 포팅. 책임 = **게이트**:
consensus 가 신뢰 불가(부족/blur)면 template=None 을 돌려 호출부가 rcp 로 폴백하게 한다
(consensus or rcp). 즉 어떤 사유든 최악 = 검증된 rcp 베이스라인 → 회귀 위험 0.
검증 근거: cond A/B in_topk 0.434→0.876, rank1 0.318→0.764 (저널 260608_163302).

입력 crop 규약: **한 modality**의, 이미 정제(crosshair 제거)·crosshair 중심 crop·
co-registration 까지 끝난 동일 크기 gray 배열들(= consensus_crops.load_coregistered_crops 출력).
"""

import statistics
from dataclasses import dataclass

from poc.workflow_3.align.matching.engine import AlignKeyTemplate, build_template
from poc.workflow_3.align.consensus_cv import _consensus, _edge_density, _lap_var

CONSENSUS_VERSION = "s_consensus_prod"   # 로그/overlay 에서 rcp 와 구분.


@dataclass(frozen=True)
class ConsensusPolicy:
    """consensus 게이트 임계. blur 임계(0.70/0.50)는 golden 실측 확정값."""

    min_s: int = 3                # 같은 modality 최소 S crop 장수(resolver 가 settings 로 주입).
    edge_ratio_min: float = 0.70  # consensus edge density / 개별 crop median. 미만이면 흐림.
    lap_ratio_min: float = 0.50   # consensus Laplacian 분산 비율. 미만이면 흐림.


DEFAULT_CONSENSUS_POLICY = ConsensusPolicy()


@dataclass
class ConsensusResult:
    """build_consensus_template 결과 + audit. template None = rcp 폴백 신호."""

    template: AlignKeyTemplate | None
    modality: str
    n_crops: int
    edge_ratio: float | None
    lap_ratio: float | None
    reason: str                          # "ok" | "insufficient_s" | "blurry".


def _sharpness_ratio(metric_fn, consensus, crops):
    """consensus 선명도 ÷ 개별 crop 선명도 median. 분모 0(featureless)이면 None."""
    c_val = metric_fn(consensus)
    s_med = statistics.median([metric_fn(c) for c in crops])
    if s_med <= 0:
        return None
    return round(c_val / s_med, 4)


def build_consensus_template(crops, *, recipe_id, modality,
                             policy=DEFAULT_CONSENSUS_POLICY):
    """정제·정렬된 같은 modality S crop 들로 consensus template 을 짓는다(게이트 포함).

    template 이 None 이면 호출부는 rcp center 로 폴백한다.
    """
    n = len(crops)
    if n < policy.min_s:
        return ConsensusResult(None, modality, n, None, None, "insufficient_s")

    consensus = _consensus(crops)
    edge_ratio = _sharpness_ratio(_edge_density, consensus, crops)
    lap_ratio = _sharpness_ratio(_lap_var, consensus, crops)

    edge_bad = edge_ratio is None or edge_ratio < policy.edge_ratio_min
    lap_bad = lap_ratio is None or lap_ratio < policy.lap_ratio_min
    if edge_bad or lap_bad:
        return ConsensusResult(None, modality, n, edge_ratio, lap_ratio, "blurry")

    template = build_template(consensus, recipe_id=recipe_id,
                              version=CONSENSUS_VERSION, key_type=modality)
    return ConsensusResult(template, modality, n, edge_ratio, lap_ratio, "ok")


# modality(빌더 규약) → route_template 의 키 규약.
_MOD_TO_ROUTE_KEY = {"om": "OM", "sem": "SEM"}


def select_routing_templates(consensus_by_mod, rcp_by_mod):
    """route_template 에 넘길 dict 을 consensus 우선·rcp 폴백으로 조립.

    Args:
        consensus_by_mod: {'om'|'sem': AlignKeyTemplate | None}. None = 게이트 폴백 신호.
        rcp_by_mod:       {'om'|'sem': AlignKeyTemplate | None}. 베이스라인 center template.
    Returns:
        {'OM'|'SEM': AlignKeyTemplate} — route_template(templates, mode) 에 바로 사용.
    """
    out = {}
    for mod, route_key in _MOD_TO_ROUTE_KEY.items():
        chosen = consensus_by_mod.get(mod) or rcp_by_mod.get(mod)
        if chosen is not None:
            out[route_key] = chosen
    return out
