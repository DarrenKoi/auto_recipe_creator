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
from poc.workflow_2.align_similarity import GT_TOL_NORM, _consensus_template_ab, _matched_crop
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import (
    MSR_OM_MAG_MAX, MSR_SEM_MAG_MIN, _to_int, load_cond,
    msr_modality as _msr_modality,
)
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

# LOO consensus 최소 S(같은 modality) 장수. align_similarity.AB_MIN_S 기본 4 지만, 이 golden
# set 은 S 가 희박하다(probe 2026-06-08: 298 recipe 중 ≥4 는 1개뿐, 135개가 정확히 3장).
# 4 면 recipe 1개만 통과해 A/B 가 무의미 → 3 으로 낮춰 ~136 recipe(~411 LOO)를 살린다.
# 단, S=3 recipe 는 LOO consensus 가 2장(others)으로 빌드돼 약하다 — blur 가드/lift 로 판정.
MIN_S_FLOOR = 3   # LOO 바닥: len(others)>=2 가드상 fm>=3 이어야 점이 하나라도 난다 → 2 이하는 무의미.
_MIN_S_ENV = int(os.getenv("CONSENSUS_MIN_S", "3"))


def _floor_min_s(value):
    """min_s 를 바닥 MIN_S_FLOOR(3)로 보정한다.

    align_similarity LOO 는 `len(others)>=2` 를 요구하므로 같은 modality fm>=3 이어야 LOO 점이
    하나라도 난다. 2 이하를 줘도 결과가 동일(=조용한 no-op)하므로 3 으로 올려 knob 을 정직하게.
    """
    return max(MIN_S_FLOOR, value)


CONSENSUS_MIN_S = _floor_min_s(_MIN_S_ENV)


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


# msr modality 추론(_msr_modality)·MAG 상수는 cond_file 공유 모듈로 이동(단일 출처).
# 여기서는 alias import 로 기존 이름(_msr_modality)을 보존한다.


def _resolve_mod(cond, recipe_mod):
    """msr 프레임 routing modality. 우선순위: msr 키/배율 추론 → recipe rcp modality 폴백.

    msr cond 엔 Scope 가 없으므로(사용자 확인) Scope 는 보지 않는다 — _msr_modality(키/배율)로
    결정하고, 미상이면 recipe 의 단일 rcp modality 로 폴백, 그것도 없으면 None(skip).
    이 단계가 과거 missing_modality 대량 누락(dual-rcp recipe + Scope 부재)을 해소한다.
    """
    return _msr_modality(cond) or recipe_mod


def _recipe_key(assets):
    """by_recipe dict 의 *고유* 키. assets.recipe_id 는 recipe_name(leaf)만이라 eqp/class 가
    달라도 leaf 이름이 같으면 dict 에서 덮어써져 데이터가 유실된다(probe: 298 dir → 276 고유).
    eqp/class/recipe 조합으로 고유화한다(라벨로도 읽기 좋음 — '/' 는 파일명에서 '__' 치환됨).
    """
    return f"{assets.eqp_id}/{assets.class_name}/{assets.recipe_id}"


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
        "mod_counts": Counter(),     # *해결된* modality 분포(om/sem/unresolved) — 키/배율 추론 결과.
        "drop_counts": Counter(),    # S 프레임 누락 사유(coverage 손실 가시화, code-review [4]/[5]).
    }
    # recipe 의 rcp modality (단일이면 그것, om/sem 둘 다면 모호 → None). msr modality 미상 시 폴백.
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
        mod = _resolve_mod(cond, recipe_mod)                           # msr 키/배율 → recipe 폴백(Scope 없음).
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

    if _MIN_S_ENV < CONSENSUS_MIN_S:   # 2 이하는 LOO 가 못 나와 무의미 → 바닥 3 으로 보정됨.
        print(f"[WARNING] CONSENSUS_MIN_S={_MIN_S_ENV} 는 무의미(LOO 바닥 fm>=3) → "
              f"{CONSENSUS_MIN_S} 로 보정합니다.")

    by_recipe = {}
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
        mod_total.update(entry["mod_counts"])
        drop_total.update(entry["drop_counts"])
        n_s = len(entry["s_frames"])
        n_drop = sum(entry["drop_counts"].values())
        rec_key = _recipe_key(assets)          # eqp/class/recipe 고유 키 — leaf 충돌 방지.
        if n_s:
            by_recipe[rec_key] = entry
        print(f"[INFO] {rec_key}: S(crosshair) {n_s}장, E {len(entry['e_paths'])}장"
              + (f"  [누락 {n_drop}: {dict(entry['drop_counts'])}]" if n_drop else ""))

    # msr cond 엔 Scope 가 없어 키/배율로 modality 를 가른다(Scope 분포는 더 이상 추적 안 함).
    print(f"\n[INFO] === msr S 해결된 modality(키/배율) === "
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
              "missing_modality=키/배율·recipe 둘 다 미상 · no_template=center tpl 없음 · "
              "crop_failed=OOB/너무작음 · load_failed=이미지 로드 실패")
    else:
        print(f"[INFO] S 프레임 누락 없음 (채택 {n_kept}장).")

    res = _consensus_template_ab(by_recipe, min_s=CONSENSUS_MIN_S, out_dir=out_dir)
    if res is None:
        print(f"[ERROR] consensus A/B 불가 — LOO 가능한(같은 modality ≥{CONSENSUS_MIN_S}) recipe 가 없음.")
        return "no_ab"

    res["coregister"] = COREGISTER      # min_s 는 _consensus_template_ab 가 이미 반환에 넣는다.
    res["modality_distribution"] = dict(mod_total)
    res["drop_distribution"] = dict(drop_total)
    (out_dir / "summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    rcp = res["overall_rcp_in_topk_rate"]
    cons = res["overall_cons_in_topk_rate"]
    lift = res["overall_lift"]
    print("\n" + "=" * 64)
    print("[INFO] === consensus 재등록 A/B (cond, LOO; 이 블록만 읽어주면 됨) ===")
    print(f"  recipes={res['n_recipes']}  S_loo={res['n_S_loo']}  min_s={CONSENSUS_MIN_S}  "
          f"(baseline=center tpl, offset0, co-reg={'ON' if COREGISTER else 'OFF'})")
    if CONSENSUS_MIN_S < 4:
        print(f"  ⚠ min_s={CONSENSUS_MIN_S}: S={CONSENSUS_MIN_S} recipe 의 LOO consensus 는 "
              f"{CONSENSUS_MIN_S - 1}장으로 빌드돼 약함 — lift 가 양수여도 blur 가드 함께 확인 "
              f"(env CONSENSUS_MIN_S=4 로 강한판 비교 가능).")
    print(f"  in_topk:  rcp(center)={rcp}  →  consensus={cons}   lift={lift:+}")
    print(f"  rank1:    rcp(center)={res['overall_rcp_rank1_rate']}  →  "
          f"consensus={res['overall_cons_rank1_rate']}   lift={res['rank1_lift']:+}")
    print(f"  blur 가드(낮으면 median 흐림): edge_ratio_to_S="
          f"{res['cons_edge_density_ratio_to_S_median']}  lap_ratio_to_S="
          f"{res['cons_lap_var_ratio_to_S_median']}")
    # consensus 잔여 miss 분포 — "ensemble 을 consensus 에 얹으면 더 오르나" 의 사전 진단.
    md = res.get("cons_miss_dist_distribution")
    if md and md.get("n"):
        print(f"  consensus miss 거리(truth↔최근접 후보, short-side 상대; tol={GT_TOL_NORM}):")
        print(f"    n_miss={md['n']}  median={md['median']}  p25={md['p25']}  "
              f"p75={md['p75']}  max={md['max']}")
        print(f"    bins={md['bins']}")
        print("    * near 多 → ensemble 후보 recall 로 끌어들일 여지(시도 가치) / "
              "far·veryfar 多 → 구조적 모호성(ensemble 무력, 다른 축 필요)")
    verdict = ("CONSENSUS 채택 권장(lift≥+0.05)" if lift >= 0.05
               else "효과 미미/음수 — proposer ensemble 또는 VLM-region 로 전환 검토")
    print(f"  >>> 판정: {verdict}")
    print("=" * 64)
    print(f"\n[INFO] 완료: {out_dir}  (consensus 템플릿: {out_dir}/consensus/)")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
