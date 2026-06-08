"""consensus 재등록 A/B (cond GT) — 단일 rcp template 대신 recipe 의 S 들로 만든
consensus template 이 proposer recall(gt_in_topk)을 올리나 검증한다.

왜 consensus 인가
----------------
PROPOSER_WALL: 진실(align point)이 matcher 후보(top-N)에 자주 없다([[project_matcher_flat_chamfer_distinctiveness]]).
단일 rcp 등록 key 는 공정 드리프트로 stale 해질 수 있다. recipe 의 *최근 성공(S)* 들을
평균낸 consensus 는 현재 외형을 추종 → 후보에 진실이 들어올 확률↑. 과거 실데이터(정제 전)
에서 in_topk +0.282 (0.436→0.718) 로 검증된 유일한 강한 레버.

정렬(registration) 규칙 — **crosshair 중심, 이미지 중심 절대 금지**
-----------------------------------------------------------------
rcp 는 이미지 중심 = align point(고정 anchor). 그러나 msr S 는 웨이퍼마다 달라 이미지
중심이 align point 가 아니다. 각 S 의 align point 는 **crosshair**(align point 를
신뢰성 있게 표시 — 사용자 확인 2026-06-08)에 있으므로, 모든 S crop 을 crosshair 로
정렬해 median 을 떠야 또렷한 align-key 가 된다.

cond 개선점
----------
과거 align_similarity 는 raw 프레임에서 crop → 중앙에 crosshair 잔존(distractor).
cond.txt 가 정확한 crosshair 좌표를 주므로 **crop 전에 crosshair 를 inpaint** 로
지운다 → 중앙 distractor 없는 더 깨끗한 consensus.

검증/측정은 검증된 `align_similarity._consensus_template_ab`(LOO + blur 가드 +
apples-to-apples baseline)를 그대로 재사용한다(로직 중복/표류 방지). baseline = center
template(offset 0) — consensus 도 crosshair 정렬이라 offset 0, 동일 척도 A/B.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/golden_consensus_eval_cond.py
  golden 루트: 기본 align_images_golden/, env ALIGN_GOLDEN_ROOT 로 override.
출력: stdout + DEBUG_IMAGE_DIR/golden_consensus_eval_cond/<ts>/{summary.json, consensus/*.png}
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_2.align_similarity import _consensus_template_ab, _matched_crop
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import _to_int, load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_1.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_consensus_eval_cond"

# co-registration: integer-crosshair 정렬이 남긴 sub-pixel 잔차를 phase-correlation 으로
# 마저 맞춰 median blur(edge_ratio<0.70 경고)를 줄인다 → consensus 가 또렷해져 membership·
# rank1 둘 다 개선 기대. env CONSENSUS_COREGISTER=0 으로 끄면 A/B 비교 가능.
COREGISTER = os.getenv("CONSENSUS_COREGISTER", "1") != "0"
COREG_ITERS = 2                 # ref median 을 다듬으며 원본 crop 을 재정렬(보간 누적 방지).
COREG_MAX_SHIFT_FRAC = 0.3      # 추정 shift 가 crop 변의 이 비율 초과면 spurious → 정렬 생략.


def _align_to_ref(img, ref):
    """img 를 ref 에 sub-pixel 평행이동 정렬(phase correlation). 과도 shift 면 원본 반환."""
    h, w = ref.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(img.astype(np.float32), ref.astype(np.float32), win)
    if abs(dx) > COREG_MAX_SHIFT_FRAC * w or abs(dy) > COREG_MAX_SHIFT_FRAC * h:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def coregister_crops(crops):
    """crop 들을 공통 reference(다듬어진 median)에 sub-pixel 정렬해 median 을 또렷하게.

    매 iter 마다 ref=median(현재 정렬본) 으로 갱신하되, 정렬은 항상 *원본* 에서 한 번만
    적용해 보간 blur 누적을 막는다. crop 2장 미만이면 그대로 반환.
    """
    if len(crops) < 2:
        return crops
    aligned = list(crops)
    for _ in range(COREG_ITERS):
        ref = np.median(np.stack([a.astype(np.float32) for a in aligned]), 0).astype(np.uint8)
        aligned = [_align_to_ref(c, ref) for c in crops]
    return aligned


def _cond_crosshair_xy(cond):
    """cond.crosshair_xy(cursor frame, ×10) → 이미지 px (x, y). 없으면 None."""
    if cond is None or cond.crosshair_xy is None:
        return None
    gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
    return (int(round(gx)), int(round(gy)))


def _cond_consensus_crop(gray, cond, size_wh):
    """crosshair(=align point) 중심·고정 size 의 *정제된*(crosshair 제거) crop. 없으면 None.

    consensus 재료. 모든 S 를 crosshair 로 정렬해 median 이 또렷해지게 하고, crop 전에
    clean_image 로 crosshair(+box) 를 지워 중앙 distractor 를 없앤다(cond 개선점).
    """
    xy = _cond_crosshair_xy(cond)
    if xy is None:
        return None
    cleaned = clean_image(gray, cond)        # crosshair(+box) 제거 후 자른다.
    w, h = size_wh
    return _matched_crop(cleaned, xy, w, h, 1.0)


def _scope_label(cond):
    """cond.scope → 충실 분류 'om' | 'omdf' | 'sem' | None (진단·가시성용).

    'OM' 부분일치로 'OMDF' 를 삼키지 않도록 OMDF 를 먼저 검사. Scope 없음/미상은 None.
    """
    if cond is None or not cond.scope:
        return None
    s = cond.scope.upper()
    if "SEM" in s:
        return "sem"
    if "OMDF" in s:
        return "omdf"
    if "OM" in s:
        return "om"
    return None


def _modality_of(cond):
    """매칭 routing modality 'om' | 'sem' | None.

    OMDF 는 OM 의 한 종류(OM + darkfield 오버레이)이므로 routing 은 OM 으로 묶는다
    (darkfield 외형 차이로 인한 분리는 추후 과제). SEM 만 별도. 미상은 None(침묵 om 금지).
    """
    lbl = _scope_label(cond)
    if lbl in ("om", "omdf"):
        return "om"
    return lbl   # 'sem' | None


# msr cond 는 Scope 가 없다(사용자 확인 2026-06-08). 대신 키/배율로 modality 를 가른다:
# OM = !OM_Brightness 키 + Magnification<200, SEM = Accelerating_voltage 키 + Magnification>500.
# 키 존재가 1순위(확정), Magnification 보조([[project_align_cond_files_and_coords]]).
MSR_OM_MAG_MAX = 200     # Magnification < 이값 → OM (보조 신호).
MSR_SEM_MAG_MIN = 500    # Magnification > 이값 → SEM (보조 신호).


def _msr_modality(cond):
    """msr cond 의 modality 추론 'om' | 'sem' | None (Scope 없음 → 키/배율).

    msr cond.txt 엔 Scope 가 없다. OM 은 ``!OM_Brightness`` 키 + Magnification<200,
    SEM 은 ``Accelerating_voltage`` 키 + Magnification>500. 키 존재가 확정 신호라 1순위,
    Magnification 은 보조(200~500 사이는 모호 → None). raw 키는 _norm_key 로 '!'·소문자화됨.
    """
    if cond is None:
        return None
    raw = cond.raw or {}
    if "accelerating_voltage" in raw:
        return "sem"
    if "om_brightness" in raw:
        return "om"
    mag_tokens = raw.get("magnification") or []
    mag = _to_int(mag_tokens[0]) if mag_tokens else None
    if mag is not None:
        if mag < MSR_OM_MAG_MAX:
            return "om"
        if mag > MSR_SEM_MAG_MIN:
            return "sem"
    return None


def _resolve_mod(cond, recipe_mod):
    """msr 프레임 routing modality. 우선순위: rcp-style Scope → msr 키/배율 → recipe rcp modality.

    msr cond 엔 Scope 가 없어(_modality_of None) 거의 항상 _msr_modality(키/배율)로 결정된다.
    그래도 미상이면 recipe 의 단일 rcp modality 로 폴백, 그것도 없으면 None(skip).
    이 단계가 과거 missing_modality 대량 누락(dual-rcp recipe + Scope 부재)을 해소한다.
    """
    return _modality_of(cond) or _msr_modality(cond) or recipe_mod


def _precrop_drop_reason(cond, xy, mod, has_tpl):
    """S 프레임이 crop 이전 단계에서 빠지는 사유(없으면 None=채택). coverage 손실 가시화.

    조용히 continue 하던 지점들(코드리뷰 [4])을 사유별로 분류해 집계·경고하기 위함.
    우선순위: 근본 원인부터(cond 부재 → crosshair 부재 → modality 미상 → template 없음).
    """
    if cond is None:
        return "missing_cond"
    if xy is None:
        return "missing_crosshair"
    if mod is None:
        return "missing_modality"
    if not has_tpl:
        return "no_template"
    return None


def _build_cond_by_recipe(assets, center_tpls):
    """한 recipe → `_consensus_template_ab` 입력 항목.

    baseline rcp_tpls = center template(offset 0). s_frames 의 crop 은 cond crosshair
    중심·crosshair 제거된 고정 size(해당 modality center tpl 크기). E 는 가드용 경로만.
    """
    entry = {
        "rcp_tpls": {m: t for m, (t, _off) in center_tpls.items() if t is not None},
        "s_frames": [],
        "e_paths": [],
        "scope_counts": Counter(),   # cond.txt Scope 분포(msr 는 보통 전부 missing — Scope 없음).
        "mod_counts": Counter(),     # *해결된* modality 분포(om/sem/unresolved) — 키/배율 추론 결과.
        "drop_counts": Counter(),    # S 프레임 누락 사유(coverage 손실 가시화, code-review [4]/[5]).
    }
    # recipe 의 rcp modality (단일이면 그것, om/sem 둘 다면 모호 → None). msr scope 부재 시 폴백.
    rcp_mods = [m for m, v in center_tpls.items() if v is not None]
    recipe_mod = rcp_mods[0] if len(rcp_mods) == 1 else None
    for p in iter_msr_images(assets):
        label = _tool_label(p.name)
        if label == "E":
            entry["e_paths"].append(p)
            continue
        if label != "S":
            continue
        cond = load_cond(p)
        entry["scope_counts"][_scope_label(cond) or "missing"] += 1   # 충실 Scope(msr 는 대개 missing).
        mod = _resolve_mod(cond, recipe_mod)                           # scope → msr 키/배율 → recipe 폴백.
        entry["mod_counts"][mod or "unresolved"] += 1                  # 해결된 modality 분포.
        xy = _cond_crosshair_xy(cond)
        # 같은 modality center tpl 로 sizing(없으면 첫 가용 tpl) — omdf 등 drop 방지.
        tpl_item = center_tpls.get(mod) or next(
            (t for t in center_tpls.values() if t is not None), None)
        reason = _precrop_drop_reason(cond, xy, mod, tpl_item is not None)
        if reason:
            entry["drop_counts"][reason] += 1     # 조용히 버리지 않고 사유별 집계.
            continue
        tpl = tpl_item[0]
        size_wh = (tpl.raw_image.shape[1], tpl.raw_image.shape[0])
        try:
            gray = load_gray(p)
        except Exception as exc:
            print(f"[WARNING] msr 로드 실패 {p.name}: {exc}")
            entry["drop_counts"]["load_failed"] += 1
            continue
        crop = _cond_consensus_crop(gray, cond, size_wh)
        if crop is None:                          # OOB/너무 작음 — code-review [5].
            entry["drop_counts"]["crop_failed"] += 1
            continue
        entry["s_frames"].append({"path": p, "xy": xy, "mod": mod, "crop": crop})

    # co-registration: modality 별로(외형이 달라 섞으면 안 됨) crop 들을 sub-pixel 정렬해
    # median blur 를 줄인다. 정렬은 crop 내용만 바꾸고 crosshair GT(xy)·full-frame 은 불변.
    if COREGISTER and entry["s_frames"]:
        by_mod = defaultdict(list)
        for f in entry["s_frames"]:
            by_mod[f["mod"]].append(f)
        for fs in by_mod.values():
            for f, aligned in zip(fs, coregister_crops([f["crop"] for f in fs])):
                f["crop"] = aligned
    return entry


def run() -> str:
    """consensus(cond) A/B 실행 (인자 없음). 반환: success | no_data | no_ab."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터를 찾지 못했습니다: {root} (env ALIGN_GOLDEN_ROOT).")
        return "no_data"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] (consensus cond A/B) recipe {len(recipes)}개 → {out_dir}")
    print(f"[INFO] co-registration: {'ON' if COREGISTER else 'OFF'} "
          f"(env CONSENSUS_COREGISTER=0 으로 끄고 A/B 비교 가능)")

    by_recipe = {}
    scope_total = Counter()
    mod_total = Counter()
    drop_total = Counter()
    for assets in recipes:
        if assets is None:
            continue
        try:
            center_tpls, _box = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        if not any(v is not None for v in center_tpls.values()):
            continue
        entry = _build_cond_by_recipe(assets, center_tpls)
        scope_total.update(entry["scope_counts"])
        mod_total.update(entry["mod_counts"])
        drop_total.update(entry["drop_counts"])
        n_s = len(entry["s_frames"])
        n_drop = sum(entry["drop_counts"].values())
        if n_s:
            by_recipe[assets.recipe_id] = entry
        print(f"[INFO] {assets.recipe_id}: S(crosshair) {n_s}장, E {len(entry['e_paths'])}장"
              + (f"  [누락 {n_drop}: {dict(entry['drop_counts'])}]" if n_drop else ""))

    # msr 는 Scope 가 없어 대개 전부 missing(정상). 실제 쓰는 건 키/배율로 *해결된* modality.
    print(f"\n[INFO] === msr S Scope 분포(원문) === "
          f"om={scope_total.get('om', 0)} omdf={scope_total.get('omdf', 0)} "
          f"sem={scope_total.get('sem', 0)} missing={scope_total.get('missing', 0)} "
          f"(msr 엔 Scope 없음 → missing 정상)")
    print(f"[INFO] === msr S 해결된 modality(키/배율) === "
          f"om={mod_total.get('om', 0)} sem={mod_total.get('sem', 0)} "
          f"unresolved={mod_total.get('unresolved', 0)} "
          f"(OM=!OM_Brightness/Mag<200, SEM=Accelerating_voltage/Mag>500)")

    # S 프레임 누락 집계(coverage 손실 가시화) — 조용한 skip 추방(code-review [4]/[5]).
    n_kept = sum(len(e["s_frames"]) for e in by_recipe.values())
    n_dropped = sum(drop_total.values())
    if n_dropped:
        print(f"[WARNING] === S 프레임 누락 {n_dropped}장 (채택 {n_kept}장) === "
              f"{dict(drop_total.most_common())}")
        print("    missing_cond=sidecar 없음 · missing_crosshair=cond 에 crosshair 없음 · "
              "missing_modality=scope/recipe 둘 다 미상 · no_template=center tpl 없음 · "
              "crop_failed=OOB/너무작음 · load_failed=이미지 로드 실패")
    else:
        print(f"[INFO] S 프레임 누락 없음 (채택 {n_kept}장).")

    res = _consensus_template_ab(by_recipe, out_dir=out_dir)
    if res is None:
        print("[ERROR] consensus A/B 불가 — LOO 가능한(≥AB_MIN_S) recipe 가 없음.")
        return "no_ab"

    res["coregister"] = COREGISTER
    res["scope_distribution"] = dict(scope_total)
    res["modality_distribution"] = dict(mod_total)
    res["drop_distribution"] = dict(drop_total)
    (out_dir / "summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    rcp = res["overall_rcp_in_topk_rate"]
    cons = res["overall_cons_in_topk_rate"]
    lift = res["overall_lift"]
    print("\n" + "=" * 64)
    print("[INFO] === consensus 재등록 A/B (cond, LOO; 이 블록만 읽어주면 됨) ===")
    print(f"  recipes={res['n_recipes']}  S_loo={res['n_S_loo']}  "
          f"(baseline=center tpl, offset0, co-reg={'ON' if COREGISTER else 'OFF'})")
    print(f"  in_topk:  rcp(center)={rcp}  →  consensus={cons}   lift={lift:+}")
    print(f"  rank1:    rcp(center)={res['overall_rcp_rank1_rate']}  →  "
          f"consensus={res['overall_cons_rank1_rate']}   lift={res['rank1_lift']:+}")
    print(f"  blur 가드(낮으면 median 흐림): edge_ratio_to_S="
          f"{res['cons_edge_density_ratio_to_S_median']}  lap_ratio_to_S="
          f"{res['cons_lap_var_ratio_to_S_median']}")
    verdict = ("CONSENSUS 채택 권장(lift≥+0.05)" if lift >= 0.05
               else "효과 미미/음수 — proposer ensemble 또는 VLM-region 로 전환 검토")
    print(f"  >>> 판정: {verdict}")
    print("=" * 64)
    print(f"\n[INFO] 완료: {out_dir}  (consensus 템플릿: {out_dir}/consensus/)")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
