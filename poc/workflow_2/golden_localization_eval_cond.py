"""golden_localization_eval 의 cond.txt 판(GT 를 검출이 아니라 cond.txt 에서 읽는다).

원본 `golden_localization_eval.py` 는 GT 를 **검출**으로 얻었다:
  - crosshair GT = detect_crosshair(msr)           (검출률·conf 게이트 필요)
  - box template = white-box 검출기(앙상블/legacy)  (검출 실패 시 폴백)
그런데 이제 각 이미지에 cond.txt 가 딸려 와 **정확한** 좌표를 알 수 있다
([[project_align_cond_files_and_coords]]):
  - msr crosshair = cond.crosshair_xy   (elements[4],[5])
  - rcp white box = cond.box_ltrb       (elements[6..9])
좌표는 Pixel 의 10배(cursor frame)이므로 이미지 px = coord/10 (cursor_to_image).

본 파일은 **원본을 건드리지 않고** cond GT 로 같은 2×2(template × frame) 위치추정을
돌린다. cond 에 좌표가 없으면 검출로 **폴백**(crosshair)하거나 box 템플릿을 비운다.
원본의 cond-독립 부품(_localize/_save_overlay/_summarize/_print_summary/_offset_diag/
_collect_recipes)은 그대로 재사용해 로직 중복/표류를 막는다.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/golden_localization_eval_cond.py
  golden 루트: 기본 align_images_golden/, env ALIGN_GOLDEN_ROOT 로 override.
  단일 recipe: env ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME.
  (self-test 없음 — cond.txt 는 실데이터에만 있으므로 데이터 없으면 종료.)
출력: stdout 표 + DEBUG_IMAGE_DIR/golden_localization_eval_cond/<ts>/{rows.jsonl, summary.json}
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

import numpy as np

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import (
    iter_msr_images,
    load_gray,
    resolve_assets,
)
from poc.workflow_2.align_key_matcher import build_template
from poc.workflow_2.align_point_correction import (
    RCP_FALLBACK_CENTER_CROP_AREA_RATIO,
    _RcpTemplateBundle,
    _centered_area_crop_bbox,
    _draw_rcp_overlay,
    _inner_crop_for_box,
    _inpaint_crosshair,
    _tool_label,
)
from poc.workflow_2.align_similarity import CENTER_AREA_RATIO
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import load_cond
from poc.workflow_2.crosshair_detect import detect_crosshair
# 원본의 cond-독립 부품 재사용(중복/표류 방지).
from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_1.util.time_utils import make_timestamp_tag

GOLDEN_ROOT = ALIGN_IMAGES_ROOT.parent / "align_images_golden"
OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_localization_eval_cond"
SAVE_OVERLAYS = gle.SAVE_OVERLAYS
TEMPLATES = gle.TEMPLATES
FRAMES = gle.FRAMES


def _cond_box_to_xywh(box_ltrb):
    """cond.box_ltrb(cursor frame, ×10) → 이미지 px (x, y, w, h)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t)))


def _build_offset_templates_cond(
    assets, *, rcp_overlay_dir=None, box_reasons=None, offset_records=None,
):
    """rcp 에서 center / box template 을 만든다 — box 는 cond.box_ltrb 에서(검출 아님).

    center = 정중앙 면적 crop(offset (0,0)). box = cond 흰 box 안쪽 crop
    (offset = image_center - inner_center). cond 에 box 없으면 box[mod]=None.
    원본 _build_offset_templates 와 동일 규약 — match 중심 + offset = align point.
    """
    center, box = {}, {}
    sources = [("om", assets.recipe_om, "rcp_om", "om"),
               ("sem", assets.recipe_sem, "rcp_sem", "sem")]
    for mod, path, version, key_type in sources:
        if path is None:
            continue
        gray = load_gray(path)
        h, w = gray.shape[:2]
        img_cx, img_cy = w // 2, h // 2

        cx, cy, cw, ch = _centered_area_crop_bbox(gray, CENTER_AREA_RATIO)
        center_crop = gray[cy:cy + ch, cx:cx + cw].copy()
        center[mod] = (
            build_template(center_crop, recipe_id=assets.recipe_id,
                           version=version + "_center", key_type=key_type),
            (0, 0),
        )

        cond = load_cond(path)
        box_ltrb = cond.box_ltrb if cond else None
        if box_ltrb is not None:
            det = _cond_box_to_xywh(box_ltrb)
            inner, inner_bbox = _inner_crop_for_box(gray, det)
            ix, iy, iw, ih = inner_bbox
            offset = (img_cx - (ix + iw // 2), img_cy - (iy + ih // 2))
            box[mod] = (
                build_template(inner, recipe_id=assets.recipe_id,
                               version=version + "_box", key_type=key_type),
                offset,
            )
            if box_reasons is not None:
                box_reasons["ok:cond"] = box_reasons.get("ok:cond", 0) + 1
            if offset_records is not None:
                short = max(1, min(iw, ih))
                onorm = float(np.hypot(offset[0], offset[1]) / short)
                offset_records.append(
                    {"recipe": assets.recipe_id, "mod": mod, "offset_norm": round(onorm, 4)})
            if rcp_overlay_dir is not None:
                bundle = _RcpTemplateBundle(
                    template=box[mod][0], align_offset_xy=offset,
                    detected_box=det, inner_crop=inner_bbox)
                _draw_rcp_overlay(
                    gray, bundle=bundle,
                    out_path=rcp_overlay_dir / f"{assets.recipe_id}_{mod}_rcp.jpg",
                    label=f"{assets.recipe_id}/{mod}")
        else:
            box[mod] = None
            if box_reasons is not None:
                box_reasons["cond:absent"] = box_reasons.get("cond:absent", 0) + 1
            if rcp_overlay_dir is not None:
                fb_bbox = _centered_area_crop_bbox(gray, RCP_FALLBACK_CENTER_CROP_AREA_RATIO)
                bundle = _RcpTemplateBundle(
                    template=center[mod][0], align_offset_xy=(0, 0),
                    detected_box=None, inner_crop=fb_bbox)
                _draw_rcp_overlay(
                    gray, bundle=bundle,
                    out_path=rcp_overlay_dir / f"{assets.recipe_id}_{mod}_rcp.jpg",
                    label=f"{assets.recipe_id}/{mod}",
                    fallback_color=(0, 0, 255), fallback_thickness=2)
    return center, box


def _process_msr_cond(msr_path, center_tpls, box_tpls, *, recipe_id="", overlay_dir=None):
    """golden msr 한 장 → 2×2 위치추정. GT crosshair 는 cond.crosshair_xy(없으면 검출 폴백).

    frame 의 crosshair 제거는 cond 가 있으면 clean_image(cond 구동), 없으면 검출+_inpaint.
    """
    try:
        gray_raw = load_gray(msr_path)
    except Exception as exc:
        print(f"[WARNING] msr 로드 실패 {msr_path.name}: {exc}")
        return None

    label = _tool_label(msr_path.name)
    cond = load_cond(msr_path)
    if cond and cond.crosshair_xy is not None:
        gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
        crosshair_xy = (int(round(gx)), int(round(gy)))
        ch_conf, ch_source = 1.0, "cond"
    else:
        ch = detect_crosshair(gray_raw)
        crosshair_xy = ch.xy
        ch_conf, ch_source = round(ch.confidence, 3), "detect"

    row = {
        "msr": msr_path.name, "label": label,
        "crosshair_xy": list(crosshair_xy) if crosshair_xy is not None else None,
        "crosshair_conf": ch_conf, "crosshair_source": ch_source,
        "cells": {}, "overlay": None,
    }
    if label != "S" or crosshair_xy is None:
        return row

    if cond and cond.crosshair_xy is not None:
        gray_inp = clean_image(gray_raw, cond)        # cond 구동 crosshair 제거(튜닝된 기본값).
    else:
        gray_inp = _inpaint_crosshair(gray_raw, crosshair_xy)
    frames = {"raw": gray_raw, "inpaint": gray_inp}
    tpl_sets = {"center": center_tpls, "box": box_tpls}

    for tname in TEMPLATES:
        tpls = tpl_sets[tname]
        if all(v is None for v in tpls.values()):
            continue
        for fname in FRAMES:
            res = gle._localize(tpls, frames[fname], crosshair_xy)
            if res is not None:
                row["cells"][f"{tname}__{fname}"] = res

    if SAVE_OVERLAYS and overlay_dir is not None and row["cells"]:
        row["overlay"] = gle._save_overlay(
            gray_inp, crosshair_xy, row["cells"],
            recipe=recipe_id or "recipe", msr_name=msr_path.name, out_dir=overlay_dir)
    return row


def run() -> str:
    """cond GT 로 golden 위치추정 검증 (인자 없음). 반환: success | no_data | no_rows."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터를 찾지 못했습니다: {root} "
              f"(env ALIGN_GOLDEN_ROOT 로 경로 지정). cond 판은 self-test 없음.")
        return "no_data"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    overlay_dir = (out_dir / "overlays") if SAVE_OVERLAYS else None
    rcp_overlay_dir = (out_dir / "rcp_templates") if SAVE_OVERLAYS else None
    if rcp_overlay_dir is not None:
        rcp_overlay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] (cond GT) recipe {len(recipes)}개 처리 → {out_dir}")

    all_rows, n_skip_no_msr = [], 0
    box_reasons, offset_records = {}, []
    with rows_path.open("w", encoding="utf-8") as rf:
        for assets in recipes:
            if assets is None:
                continue
            msr_imgs = iter_msr_images(assets)
            if not msr_imgs:
                n_skip_no_msr += 1
                print(f"[INFO] {assets.recipe_id}: from_msr 이미지 없음 → 건너뜀(skip)")
                continue
            try:
                center_tpls, box_tpls = _build_offset_templates_cond(
                    assets, rcp_overlay_dir=rcp_overlay_dir,
                    box_reasons=box_reasons, offset_records=offset_records)
            except Exception as exc:
                print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
                continue
            if not any(v is not None for v in center_tpls.values()):
                print(f"[WARNING] {assets.recipe_id}: center template 없음 — 건너뜀.")
                continue
            box_ok = any(v is not None for v in box_tpls.values())
            if gle.MAX_S_PER_RECIPE is not None:
                msr_imgs = msr_imgs[:gle.MAX_S_PER_RECIPE]
            print(f"[INFO] {assets.recipe_id}: msr {len(msr_imgs)}장  "
                  f"(cond box {'OK' if box_ok else '없음(폴백)'})")
            for p in msr_imgs:
                row = _process_msr_cond(
                    p, center_tpls, box_tpls,
                    recipe_id=assets.recipe_id, overlay_dir=overlay_dir)
                if row is None:
                    continue
                row["recipe"] = assets.recipe_id
                all_rows.append(row)
                rf.write(json.dumps(row, ensure_ascii=False) + "\n")

    if n_skip_no_msr:
        print(f"\n[INFO] from_msr 비어 건너뛴 recipe: {n_skip_no_msr}개.")
    if box_reasons:
        total = sum(box_reasons.values())
        ok = box_reasons.get("ok:cond", 0)
        print(f"\n[INFO] === rcp box (cond) === 있음 {ok}/{total} (rate={ok / total:.3f})")
        for reason, n in sorted(box_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<16} {n:>4}")

    if not all_rows:
        print("[ERROR] 처리된 msr 행이 없습니다.")
        return "no_rows"

    # GT 출처 집계(cond vs 검출 폴백).
    s_rows = [r for r in all_rows if r["label"] == "S"]
    n_cond = sum(1 for r in s_rows if r.get("crosshair_source") == "cond")
    print(f"\n[INFO] === GT 출처 (S {len(s_rows)}장) === cond={n_cond}  "
          f"detect-폴백={len(s_rows) - n_cond}")

    summary = gle._summarize(all_rows)
    summary["align_offset_diag"] = gle._offset_diag(offset_records)
    summary["gt_source"] = {"n_S": len(s_rows), "n_cond": n_cond,
                            "n_detect_fallback": len(s_rows) - n_cond}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    gle._print_summary(summary)

    diag = summary["align_offset_diag"]
    print("\n[INFO] === align-offset 진단 (cond box) ===")
    if diag.get("n", 0) == 0:
        print("  cond box 있는 modality 없음 — 진단 불가.")
    else:
        print(f"  offset_norm: median={diag['median_offset_norm']} "
              f"p90={diag['p90_offset_norm']} (n={diag['n']}) · "
              f"가정민감(>{diag['tol']})={diag['n_assumption_sensitive']}개")
    print(f"\n[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
