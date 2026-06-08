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
from pathlib import Path

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_2.align_similarity import _consensus_template_ab, _matched_crop
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import load_cond
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_1.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_consensus_eval_cond"


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


def _modality_of(cond):
    """cond.scope → center_tpls 키('sem'|'om'). 알 수 없으면 'om'."""
    if cond is not None and cond.is_sem:
        return "sem"
    return "om"


def _build_cond_by_recipe(assets, center_tpls):
    """한 recipe → `_consensus_template_ab` 입력 항목.

    baseline rcp_tpls = center template(offset 0). s_frames 의 crop 은 cond crosshair
    중심·crosshair 제거된 고정 size(해당 modality center tpl 크기). E 는 가드용 경로만.
    """
    entry = {
        "rcp_tpls": {m: t for m, (t, _off) in center_tpls.items() if t is not None},
        "s_frames": [],
        "e_paths": [],
    }
    for p in iter_msr_images(assets):
        label = _tool_label(p.name)
        if label == "E":
            entry["e_paths"].append(p)
            continue
        if label != "S":
            continue
        cond = load_cond(p)
        xy = _cond_crosshair_xy(cond)
        if xy is None:
            continue
        mod = _modality_of(cond)
        tpl_item = center_tpls.get(mod)
        if tpl_item is None:
            continue
        tpl = tpl_item[0]
        size_wh = (tpl.raw_image.shape[1], tpl.raw_image.shape[0])
        try:
            gray = load_gray(p)
        except Exception as exc:
            print(f"[WARNING] msr 로드 실패 {p.name}: {exc}")
            continue
        crop = _cond_consensus_crop(gray, cond, size_wh)
        if crop is None:
            continue
        entry["s_frames"].append({"path": p, "xy": xy, "mod": mod, "crop": crop})
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

    by_recipe = {}
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
        n_s = len(entry["s_frames"])
        if n_s:
            by_recipe[assets.recipe_id] = entry
        print(f"[INFO] {assets.recipe_id}: S(crosshair) {n_s}장, E {len(entry['e_paths'])}장")

    res = _consensus_template_ab(by_recipe, out_dir=out_dir)
    if res is None:
        print("[ERROR] consensus A/B 불가 — LOO 가능한(≥AB_MIN_S) recipe 가 없음.")
        return "no_ab"

    (out_dir / "summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    rcp = res["overall_rcp_in_topk_rate"]
    cons = res["overall_cons_in_topk_rate"]
    lift = res["overall_lift"]
    print("\n" + "=" * 64)
    print("[INFO] === consensus 재등록 A/B (cond, LOO; 이 블록만 읽어주면 됨) ===")
    print(f"  recipes={res['n_recipes']}  S_loo={res['n_S_loo']}  (baseline=center tpl, offset0)")
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
