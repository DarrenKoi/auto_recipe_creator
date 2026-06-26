# poc/workflow_3/align/consensus_crops.py
"""event-cache(align_consensus_cache) → consensus 재료 crop 어댑터.

bench `_build_cond_by_recipe`(golden_consensus_eval_cond.py)의 cache 판. msr/LOO 레이아웃
대신 align_consensus_cache/<eqp>/<class>/<recipe>/events/<event_id>/S* 를 읽는다.
modality 분류는 bench 와 동일하게 `_resolve_mod = msr_modality(cond) or recipe_mod`(단일-modality
recipe 가 silently drop 되지 않게), crop 은 clean→crosshair중심→center tpl 크기 고정,
modality 별 co-registration. 이름(load_cond/load_gray/clean_image/cursor_to_image/msr_modality)은
테스트에서 monkeypatch 하므로 모듈 전역 import 로 둔다.

프로덕션 의도적 divergence(bench 와 1줄 차이): mod 의 center tpl 이 없을 때 bench 는 다른
modality tpl 을 빌려 sizing 했지만(eval coverage 용), 프로덕션은 잘못된 크기 crop 이 median 을
오염시키지 않도록 그 프레임을 drop 한다(no_template). 결정 2026-06-12(code review).
"""

from collections import Counter, defaultdict

from poc.workflow_3.align.assets import load_gray
from poc.workflow_3.align.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.align.cond_file import load_cond, msr_modality
from poc.workflow_3.align.consensus_cv import _matched_crop, coregister_crops
from poc.workflow_3.align.consensus_gather import _events_dir_for
from poc.workflow_3.align.templates import build_templates_from_assets

# rcp 라우팅 키(대문자) → consensus 빌더/center 키(소문자).
_ROUTE_TO_MOD = {"OM": "om", "SEM": "sem"}


def build_center_tpls_for_sizing(assets):
    """consensus crop sizing 전용 center template — 소문자 om/sem, (tpl, offset).

    box-crop 이 아닌 **center-area crop**(cond_box_crop=False)이라 bench center_tpls 와
    동일 기하(crop 크기 = template.raw_image 크기). 런타임 rcp 라우팅 template(대문자,
    box-crop 가능)과 별개로 만든다. 없으면 그 modality 는 빠진다.
    """
    rcp_center = build_templates_from_assets(assets, cond_box_crop=False)  # {"OM":tpl,"SEM":tpl}
    out = {}
    for route_key, tpl in rcp_center.items():
        mod = _ROUTE_TO_MOD.get(route_key)
        if mod and tpl is not None:
            out[mod] = (tpl, (0, 0))
    return out


def _cond_crosshair_xy(cond):
    """cond.crosshair_xy(cursor frame, ×10) → 이미지 px (x, y). 없으면 None."""
    if cond is None or cond.crosshair_xy is None:
        return None
    gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
    return (int(round(gx)), int(round(gy)))


def _cond_consensus_crop(gray, cond, size_wh):
    """crosshair(=align point) 중심·고정 size 의 정제된(crosshair 제거) crop. 없으면 None."""
    xy = _cond_crosshair_xy(cond)
    if xy is None:
        return None
    cleaned = clean_image(gray, cond)        # crosshair(+box) 제거 후 자른다.
    w, h = size_wh
    return _matched_crop(cleaned, xy, w, h, 1.0)


def _resolve_mod(cond, recipe_mod):
    """msr 프레임 routing modality: msr 키/배율 추론 → recipe rcp modality 폴백."""
    return msr_modality(cond) or recipe_mod


def _precrop_drop_reason(cond, xy, mod, has_tpl):
    """S 프레임이 crop 이전 단계에서 빠지는 사유(없으면 None=채택)."""
    if cond is None:
        return "missing_cond"
    if xy is None:
        return "missing_crosshair"
    if mod is None:
        return "missing_modality"
    if not has_tpl:
        return "no_template"
    return None


def _iter_event_s_images(events_dir, max_events):
    """events/ 의 최신 max_events event 의 S* 이미지 경로를 yield(시각 prefix=정렬=시간)."""
    if not events_dir.is_dir():
        return
    try:
        # 점(.) 으로 시작하는 dir 제외 — gather 의 .events_new/.events_old/.events_staging
        # 잔재가 실수로 events/ 안에 들어와도 가짜 최신 event 로 잡혀 진짜를 밀어내지 않게.
        event_dirs = sorted(
            d for d in events_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    except OSError:
        return
    for ev in event_dirs[-max_events:]:          # 최신 우선 cap.
        for img in sorted(ev.glob("S*")):
            if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                yield img


def load_coregistered_crops(cache_root, eqp_id, cache_key, center_tpls, *, max_events):
    """event-cache 의 S 이미지를 modality 별 co-registered crop 리스트로 만든다.

    Args:
        cache_root: ALIGN_CONSENSUS_CACHE_DIR(또는 테스트 temp).
        eqp_id: 장비 id.
        cache_key: "<class>/<recipe>" (gather 가 쓴 키와 동일 — leaf 금지).
        center_tpls: {'om'|'sem': (center_tpl, offset)} sizing 전용.
        max_events: 최신 event 캡(= settings.gather_max_events).
    Returns:
        {'om'|'sem': [gray_crop, ...]} — 빌더 입력. 비면 빈 dict/빈 리스트.
    """
    # eqp_id 는 시그니처 호환을 위해 받지만 경로에는 안 쓴다(pool 은 eqp 무관).
    events_dir = _events_dir_for(cache_key, cache_root)
    rcp_mods = [m for m, v in center_tpls.items() if v is not None]
    recipe_mod = rcp_mods[0] if len(rcp_mods) == 1 else None

    by_mod = defaultdict(list)
    drop_counts = Counter()
    for p in _iter_event_s_images(events_dir, max_events):
        try:
            cond = load_cond(p)
        except Exception as exc:
            # cond 읽기 *실패*(권한/디코드 등)는 cond 부재(정상 None)와 구분해 따로 집계 —
            # 오피스 Windows 파일락 등 체계적 실패를 missing_cond 와 섞지 않는다.
            drop_counts["cond_error"] += 1
            print(f"[WARNING] cond 읽기 실패 {p.name}: {exc}")
            continue
        mod = _resolve_mod(cond, recipe_mod) if cond is not None else None
        xy = _cond_crosshair_xy(cond)
        # 프로덕션 의도적 divergence(bench `_build_cond_by_recipe` 와 1줄 차이): bench 는
        # mod 의 center tpl 이 없으면 `or next(...)` 로 *다른* modality tpl 을 빌려 sizing 했다
        # (sparse golden eval 의 coverage 유지용). 프로덕션은 잘못된 크기 crop 이 그 modality
        # 의 median 을 오염시키는 것보다 그 프레임을 버리는 게 안전하므로 fallback 을 뺀다 —
        # mod 의 center tpl 이 없으면 _precrop_drop_reason 이 no_template 으로 떨군다.
        tpl_item = center_tpls.get(mod)
        reason = _precrop_drop_reason(cond, xy, mod, tpl_item is not None)
        if reason:
            drop_counts[reason] += 1
            continue
        tpl = tpl_item[0]
        size_wh = (tpl.raw_image.shape[1], tpl.raw_image.shape[0])
        try:
            gray = load_gray(p)
        except Exception:
            drop_counts["load_failed"] += 1
            continue
        crop = _cond_consensus_crop(gray, cond, size_wh)
        if crop is None:
            drop_counts["crop_failed"] += 1
            continue
        by_mod[mod].append(crop)

    # modality 별 co-registration(외형 달라 섞으면 안 됨).
    out = {}
    for mod, crops in by_mod.items():
        out[mod] = list(coregister_crops(crops))
    if drop_counts:
        print(f"[INFO] consensus crops drop: {dict(drop_counts)} (cache_key={cache_key})")
    return out
