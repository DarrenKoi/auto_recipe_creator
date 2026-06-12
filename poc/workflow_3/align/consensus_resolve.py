# poc/workflow_3/align/consensus_resolve.py
"""resolve_templates — 실시간 보정용 라우팅 template(consensus 우선·rcp 폴백) 조립.

correction.correct_align_fail_auto 가 build_templates_from_assets 대신 이걸 호출한다.
어떤 실패(부족/blur/sync timeout/예외)든 해당 modality 는 rcp 로 강등 — 회귀 위험 0.
cache_key 는 반드시 "<class>/<recipe>"(gather 가 쓴 키) — assets.recipe_id(leaf)는 금지.

DAG 주의: align 은 monitor 아래 capability 레이어다. cold-cache sync 에 쓰는
monitor.success_gather.wait_for_gather 는 모듈 최상단에서 import 하지 않고(상향 참조/
순환 위험 회피) 아래 thin wrapper 에서 지연 import 한다.
"""

from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR
from poc.workflow_3.align.templates import build_templates_from_assets
from poc.workflow_3.align.consensus_crops import (
    build_center_tpls_for_sizing, load_coregistered_crops,
)
from poc.workflow_3.align.consensus_gather import _events_dir_for
from poc.workflow_3.align.consensus_template import (
    ConsensusPolicy, build_consensus_template, select_routing_templates,
)


def wait_for_gather(eqp_id, recipe_id, timeout):
    """monitor.success_gather.wait_for_gather 로 위임(지연 import — align→monitor 상향 참조 회피).

    모듈 attribute 라 테스트에서 monkeypatch 가능. 실제 import 는 호출 시점(런타임)이라
    import 그래프상 align 레이어가 monitor 를 정적으로 끌어오지 않는다.
    """
    from poc.workflow_3.monitor.success_gather import wait_for_gather as _impl
    return _impl(eqp_id, recipe_id, timeout)


def resolve_templates(assets, *, eqp_id, consensus_enabled, min_s, max_events,
                      sync_timeout_sec, cond_box_crop,
                      cache_root=ALIGN_CONSENSUS_CACHE_DIR):
    """{'OM'|'SEM': AlignKeyTemplate} 반환. consensus 신뢰 가능 modality 만 consensus, 그외 rcp."""
    rcp_route = build_templates_from_assets(assets, cond_box_crop=cond_box_crop)  # 런타임 라우팅(대문자)
    rcp_by_mod = {}                       # 소문자 키로 select 입력 정규화.
    for route_key, tpl in rcp_route.items():
        rcp_by_mod[route_key.lower()] = tpl
    if not consensus_enabled:
        print("[INFO] consensus 비활성(killswitch) → rcp 라우팅")
        return rcp_route

    cache_key = f"{assets.class_name}/{assets.recipe_name}"   # gather 가 쓴 키와 동일(leaf 금지).
    try:
        center_tpls = build_center_tpls_for_sizing(assets)
    except Exception as exc:
        print(f"[WARNING] consensus center tpl 실패 → rcp: {exc}")
        return rcp_route

    crops_by_mod = _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events)

    # cold-cache 일 때만 1회 bounded sync. cold = crop 0장 *이고* events/ 가 아직 없음
    # (= 최초 fail, 아직 한 번도 안 모음). events/ 가 있는데 crop 0장이면(전부 drop된 깨진
    # 캐시) sync 해도 안 늘어나므로 매 알람 8s 를 낭비하지 않고 rcp 로 간다. min_s 미달(있긴
    # 함)도 sync 대상 아님(FTP 보다 더 못 만듦). short-circuit 으로 crop 있으면 is_dir 미평가.
    cold = not any(crops_by_mod.values()) and not _events_dir_for(eqp_id, cache_key, cache_root).is_dir()
    if cold:
        if wait_for_gather(eqp_id, cache_key, sync_timeout_sec) is True:   # bool 분기
            crops_by_mod = _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events)
        # False(timeout/실패) → 재로드 없이 그대로 진행(아래서 insufficient → rcp).

    policy = ConsensusPolicy(min_s=max(3, min_s))   # floor 3 (LOO 바닥 fm>=3) — caller 가 settings 우회해도 보장.
    cons_by_mod = {}
    for mod, crops in crops_by_mod.items():
        try:
            res = build_consensus_template(crops, recipe_id=cache_key, modality=mod, policy=policy)
        except Exception as exc:
            print(f"[WARNING] consensus build 예외({mod}) → rcp: {exc}")
            continue
        reason = res.reason if res.template is not None else f"rcp:{res.reason}"
        print(f"[INFO] consensus[{mod}] n={res.n_crops} edge={res.edge_ratio} "
              f"lap={res.lap_ratio} -> {'consensus' if res.template is not None else reason}")
        if res.template is not None:
            cons_by_mod[mod] = res.template      # ConsensusResult.template 만(객체 아님)

    return select_routing_templates(cons_by_mod, rcp_by_mod)


def _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events):
    """load_coregistered_crops 의 예외/부재를 빈 dict 으로 흡수(rcp 폴백 보장)."""
    try:
        return load_coregistered_crops(cache_root, eqp_id, cache_key, center_tpls,
                                       max_events=max_events)
    except Exception as exc:
        print(f"[WARNING] consensus crops 로드 예외 → rcp: {exc}")
        return {}
