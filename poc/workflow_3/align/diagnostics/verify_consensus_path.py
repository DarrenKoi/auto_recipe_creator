"""consensus 경로 end-to-end 점검(오피스 전용, read-only) — 다운로드/RCS 없이 검증.

`mark_align_feasibility` 가 consensus_enabled 에서 쓰는 라우팅(consensus history 우선·
rcp 폴백)이 오피스 데이터에서 실제로 동작하는지를, 라이브 알람·RCS GUI 없이 확인한다.
production `consensus_resolve.resolve_templates` 의 핵심 루프(staged S → crop 정렬 →
modality 별 consensus 빌드)를 그대로 호출하되, cold-cache sync(wait_for_gather →
office_success_downloader) 만 뺀다 — 즉 **부작용 0(다운로드/쓰기 없음)** 이다.

각 recipe·modality 에 대해:
  staged S 수(count_staged_events) → co-registered crop 수(load_coregistered_crops)
  → consensus 빌드 결과(build_consensus_template.reason: ok/insufficient_s/blurry)
를 한 줄로 찍고, 끝에 [DIGEST] 한 줄(오피스→Mac 텍스트 피드백용)을 출력한다.

OM/SEM 는 끝까지 분리된다(modality 별 crop·빌드). 'ok' 가 하나라도 나오면 그 recipe·
modality 는 라이브에서 consensus 키로 매칭된다는 뜻이고, 전부 폴백 사유면 rcp 로 강등된다.

용도: 오피스에서 consensus 캐시가 실제로 채워졌고 빌드 게이트를 통과하는지 사전 점검.
실행(인자 없음):
  uv run python poc/workflow_3/align/diagnostics/verify_consensus_path.py

대상 좁히기(선택, env): ALIGN_EQP_ID / ALIGN_CLASS_NAME / ALIGN_RECIPE_NAME 로 한 recipe 만,
ALIGN_VERIFY_MAX(기본 40)로 스캔할 recipe 수 상한.
"""

import os
from collections import Counter

from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR, ALIGN_IMAGES_DIR
from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.align.assets import iter_recipe_dirs, resolve_assets
from poc.workflow_3.align.consensus_crops import (
    build_center_tpls_for_sizing,
    load_coregistered_crops,
)
from poc.workflow_3.align.consensus_gather import count_staged_events
from poc.workflow_3.align.consensus_template import (
    ConsensusPolicy,
    build_consensus_template,
)
from poc.workflow_3.monitor.success_gather import DOWNLOADER_AVAILABLE


def _verify_one(assets, *, min_s, max_events, cache_root):
    """recipe 하나의 modality 별 consensus 빌드 결과를 [(mod, n_s, n_crop, reason), ...] 로.

    resolve_templates 와 동일 입력(cache_key=class/recipe, center tpl sizing,
    co-registration, ConsensusPolicy floor 3)이되 cold sync 없이 read-only.
    """
    cache_key = f"{assets.class_name}/{assets.recipe_name}"   # gather 가 쓴 키와 동일(leaf 금지).
    n_events, n_images = count_staged_events(
        assets.eqp_id, cache_key, cache_root=cache_root
    )

    rows = []
    if n_images <= 0:
        return cache_key, n_events, n_images, rows   # 캐시 빈 recipe — 라이브에선 rcp.

    try:
        center_tpls = build_center_tpls_for_sizing(assets)
    except Exception as exc:
        print(f"[WARNING] center tpl 실패({cache_key}): {exc}")
        return cache_key, n_events, n_images, rows

    crops_by_mod = load_coregistered_crops(
        cache_root, assets.eqp_id, cache_key, center_tpls, max_events=max_events
    )
    policy = ConsensusPolicy(min_s=max(3, min_s))   # floor 3 (LOO 바닥) — resolve_templates 와 동일.
    for mod in ("om", "sem"):
        crops = crops_by_mod.get(mod) or []
        if not crops:
            # crop 0장: center tpl 부재(다른 modality recipe)면 정상, 아니면 전부 drop.
            rows.append((mod, len(crops), 0, "no_crops"))
            continue
        try:
            res = build_consensus_template(
                crops, recipe_id=cache_key, modality=mod, policy=policy
            )
        except Exception as exc:
            rows.append((mod, len(crops), 0, f"error:{exc}"))
            continue
        rows.append((mod, len(crops), res.n_crops, res.reason))
    return cache_key, n_events, n_images, rows


def main():
    settings = load_workflow3_settings()
    max_scan = int(os.environ.get("ALIGN_VERIFY_MAX", "40"))

    print("=" * 72)
    print("[INFO] consensus 경로 end-to-end 점검 (read-only, 다운로드/RCS 없음)")
    print(f"[INFO] align_images 루트(rcp/center tpl): {ALIGN_IMAGES_DIR} "
          f"(존재={'예' if ALIGN_IMAGES_DIR.is_dir() else '아니오'})")
    print(f"[INFO] consensus cache 루트(staged S): {ALIGN_CONSENSUS_CACHE_DIR} "
          f"(존재={'예' if ALIGN_CONSENSUS_CACHE_DIR.is_dir() else '아니오'})")
    print(f"[INFO] success downloader: "
          f"{'사용가능' if DOWNLOADER_AVAILABLE else '없음(캐시는 기존 적재분으로 점검)'}")
    print(f"[INFO] gate: consensus_enabled={settings.consensus_enabled} "
          f"min_s={settings.consensus_min_s}(floor3) max_events={settings.gather_max_events}")
    print("=" * 72)

    # 대상 선정: env 로 한 recipe 지정 시 그것만, 아니면 최근 recipe max_scan 개.
    eqp = os.environ.get("ALIGN_EQP_ID", "").strip()
    cls = os.environ.get("ALIGN_CLASS_NAME", "").strip()
    rcp = os.environ.get("ALIGN_RECIPE_NAME", "").strip()
    if eqp and cls and rcp:
        recipe_dirs = [ALIGN_IMAGES_DIR / eqp / cls / rcp]
        print(f"[INFO] 단일 대상: {eqp}/{cls}/{rcp}")
    else:
        recipe_dirs = iter_recipe_dirs(ALIGN_IMAGES_DIR)[:max_scan]
        print(f"[INFO] 스캔 대상 recipe: {len(recipe_dirs)}개(상한 {max_scan})")

    if not recipe_dirs:
        print("[WARNING] 점검할 recipe 가 없습니다. align_images 루트/경로를 확인하세요.")
        print("[DIGEST] consensus_verify recipes=0 (no align_images recipes)")
        return

    recipes_with_cache = 0
    mod_reason = Counter()              # (mod, reason) -> count.
    recipes_with_any_consensus = 0      # OM/SEM 중 하나라도 'ok'.
    for recipe_dir in recipe_dirs:
        if not recipe_dir.is_dir():
            print(f"[WARNING] 경로 없음, 건너뜀: {recipe_dir}")
            continue
        try:
            assets = resolve_assets(recipe_dir)
        except Exception as exc:
            print(f"[WARNING] assets 해석 실패({recipe_dir.name}): {exc}")
            continue

        cache_key, n_events, n_images, rows = _verify_one(
            assets,
            min_s=settings.consensus_min_s,
            max_events=settings.gather_max_events,
            cache_root=ALIGN_CONSENSUS_CACHE_DIR,
        )
        if n_images <= 0:
            print(f"[ ] {cache_key}: staged S 없음(0 imgs) -> 라이브에선 rcp")
            continue

        recipes_with_cache += 1
        any_ok = any(r[3] == "ok" for r in rows)
        recipes_with_any_consensus += int(any_ok)
        mark = "OK" if any_ok else "  "
        detail = ", ".join(
            f"{mod.upper()}: {n_crop}/{n_s}crop->{reason}"
            for (mod, n_s, n_crop, reason) in rows
        )
        for (mod, _n_s, _n_crop, reason) in rows:
            mod_reason[(mod, reason)] += 1
        print(f"[{mark}] {cache_key} ({n_events}ev/{n_images}img): {detail}")

    print("=" * 72)
    print(f"[INFO] recipe 스캔={len(recipe_dirs)} | 캐시 보유={recipes_with_cache} | "
          f"consensus 빌드 성공(OM|SEM 중 1+)={recipes_with_any_consensus}")
    for (mod, reason), n in sorted(mod_reason.items()):
        print(f"       {mod.upper()} {reason}: {n}")
    # 한 줄 digest(오피스→Mac 텍스트 피드백). reason 분포를 mod별로 압축.
    om_ok = mod_reason[("om", "ok")]
    sem_ok = mod_reason[("sem", "ok")]
    print(
        f"[DIGEST] consensus_verify scanned={len(recipe_dirs)} "
        f"cached={recipes_with_cache} built_any={recipes_with_any_consensus} "
        f"OM_ok={om_ok} SEM_ok={sem_ok} "
        f"downloader={'on' if DOWNLOADER_AVAILABLE else 'off'} "
        f"gate_enabled={settings.consensus_enabled}"
    )


if __name__ == "__main__":
    main()
