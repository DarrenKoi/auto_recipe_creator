"""office 실데이터 ablation A/B — "흰 box 제거 + crosshair 제거가 align 지표를 올리나?"

배경:
  `align_similarity.py` 의 기존 진단은 (a) 흰 box stroke 가 template 에 섞이고,
  (b) msr frame 의 crosshair 를 *제거하지 않은 채* 매칭했다. 둘 다 edge/Chamfer
  matcher 가 끌리는 강한 인공 구조라, 지표(S/E 분리·gt-in-topK)가 낮았다는 가설이다.
  새 방법은 (a) photometric 으로 흰 box 안쪽만 깨끗이 crop, (b) crosshair 를 inpaint 로
  제거한다. 이 스크립트는 그 두 변수를 *분리*해 정말 지표가 오르는지 같은 잣대로 잰다.

설계 — 2×2 factorial (각 셀마다 align_similarity 와 동일 지표):
  template: center(구) | box(신, photometric 깨끗 crop)
  frame   : raw(crosshair 있음, 구) | inpaint(crosshair 제거, 신)
    ┌───────────────┬───────────────┬───────────────┐
    │               │ frame=raw     │ frame=inpaint │
    ├───────────────┼───────────────┼───────────────┤
    │ tpl=center    │ OLD baseline  │ crosshair만   │
    │ tpl=box       │ box만         │ NEW (둘 다)   │
    └───────────────┴───────────────┴───────────────┘
  → OLD(center+raw) 대비 NEW(box+inpaint) 의 지표 상승폭을 보고 교체 여부를 판단한다.
    box-only / crosshair-only 셀은 어느 변수가 효과의 주범인지 분해해 준다.

지표(셀별): free / at_center / at_crosshair 의 align score S/E 분리도(balanced accuracy)
  + gt-in-topK recall(S+crosshair). 모두 `align_similarity` 의 함수를 그대로 재사용 →
  기존에 보던 숫자와 같은 정의.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/align_index_ablation.py
  단일 recipe 만 보려면 env: ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME.
출력: stdout 표 + DEBUG_IMAGE_DIR/align_index_ablation/<ts>/{rows.jsonl, summary.json}
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Windows cp949 콘솔에서 cp949 밖 기호(em-dash 등)로 print 가 죽지 않도록 — 인코딩은
# 유지(한글 정상 표시)하고 미지원 문자만 치환한다.
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import json
import statistics
from pathlib import Path

import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import (
    iter_msr_images,
    iter_recipe_dirs,
    load_gray,
    resolve_assets,
    resolve_assets_auto,
)
from poc.workflow_2.align_point_correction import _inpaint_crosshair, _tool_label
from poc.workflow_2.crosshair_detect import detect_crosshair
# align_similarity 의 지표/매칭 헬퍼를 그대로 재사용 — 정의 중복/표류 방지.
from poc.workflow_2.align_similarity import (
    AT_CROSSHAIR_ROI_FACTOR,
    _build_templates,
    _gt_in_topk,
    _race,
    _separation,
    _window_roi,
)
from poc.workflow_1.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "align_index_ablation"

# 처리 상한 (빠른 시험용). None = 전부.
LIMIT_RECIPES = None
MAX_MSR_PER_RECIPE = None

# 2×2 축.
TEMPLATES = ("center", "box")        # center=구, box=신(photometric).
FRAMES = ("raw", "inpaint")          # raw=구(crosshair 있음), inpaint=신.
INDICES = ("free", "at_center", "at_crosshair")

# 셀 → 의미(verdict 출력용).
_CELL_MEANING = {
    ("center", "raw"): "OLD baseline",
    ("box", "raw"): "box-only (clean tpl)",
    ("center", "inpaint"): "crosshair-only (inpaint)",
    ("box", "inpaint"): "NEW (box + inpaint)",
}


def _indices_for(tpls: dict, frame: np.ndarray, crosshair_xy, center_xy, tw: int, th: int) -> dict:
    """주어진 template 집합·frame 에서 free / at_center / at_crosshair align score 를 잰다.

    align_similarity 와 동일하게 `_race`(OM/SEM 중 고점) + `_window_roi` 를 쓴다.
    at_crosshair 는 crosshair 위치(원본 검출)에 ROI 를 고정 — frame 이 inpaint 면
    그 자리의 *crosshair 없는* wafer 가 align key 인지를 본다(진짜 변별 신호).
    """
    out: dict = {}
    free = _race(tpls, frame)
    out["free"] = free[1] if free else None

    croi = _window_roi(frame.shape, center_xy, tw, th)
    atc = _race(tpls, frame, roi=croi)
    out["at_center"] = atc[1] if atc else None

    if crosshair_xy is not None:
        xroi = _window_roi(frame.shape, crosshair_xy, tw, th, factor=AT_CROSSHAIR_ROI_FACTOR)
        atx = _race(tpls, frame, roi=xroi)
        out["at_crosshair"] = atx[1] if atx else None
    else:
        out["at_crosshair"] = None
    return out


def _process_msr(msr_path: Path, center_tpls: dict, box_tpls: dict) -> dict | None:
    """msr 한 장에 대해 2×2 셀의 align score + gt-in-topK 를 모은다."""
    try:
        gray_raw = load_gray(msr_path)
    except Exception as exc:
        print(f"[WARNING] msr 로드 실패 {msr_path.name}: {exc}")
        return None
    h, w = gray_raw.shape[:2]
    label = _tool_label(msr_path.name)
    center_xy = (w // 2, h // 2)

    # 대표 template 크기(center 우선) — ROI 산정용.
    any_center = next((t for t in center_tpls.values() if t is not None), None)
    if any_center is None:
        return None
    th, tw = any_center.raw_image.shape[:2]

    # crosshair 검출(원본에서 1회) → inpaint frame 생성.
    ch_res = detect_crosshair(gray_raw)
    crosshair_xy = ch_res.xy
    gray_inp = _inpaint_crosshair(gray_raw, crosshair_xy) if crosshair_xy is not None else gray_raw
    frames = {"raw": gray_raw, "inpaint": gray_inp}
    tpl_sets = {"center": center_tpls, "box": box_tpls}

    row: dict = {
        "msr": msr_path.name,
        "label": label,
        "crosshair_xy": list(crosshair_xy) if crosshair_xy else None,
        "crosshair_conf": round(ch_res.confidence, 3),
        "cells": {},
        "gt_topk": {},
    }
    for tname in TEMPLATES:
        tpls = tpl_sets[tname]
        if all(v is None for v in tpls.values()):
            continue  # box template 이 없는 recipe(검출 실패) — 해당 행은 box 셀 비움.
        for fname in FRAMES:
            idx = _indices_for(tpls, frames[fname], crosshair_xy, center_xy, tw, th)
            row["cells"][f"{tname}__{fname}"] = idx
            # gt-in-topK: 정답(crosshair) 위치가 chamfer top-N 후보에 드나 (S+crosshair 만).
            if label == "S" and crosshair_xy is not None:
                gt = _gt_in_topk(frames[fname], crosshair_xy, tpls)
                row["gt_topk"][f"{tname}__{fname}"] = (gt["in_topk"] if gt else None)
    return row


def _summarize(rows: list[dict]) -> dict:
    """셀별 S/E 분리도(align_similarity._separation) + gt-in-topK recall 집계."""
    s_rows = [r for r in rows if r["label"] == "S"]
    e_rows = [r for r in rows if r["label"] == "E"]

    sep: dict = {}
    for tname in TEMPLATES:
        for fname in FRAMES:
            cell = f"{tname}__{fname}"
            sep[cell] = {}
            for idx in INDICES:
                s_vals = [r["cells"].get(cell, {}).get(idx) for r in s_rows]
                e_vals = [r["cells"].get(cell, {}).get(idx) for r in e_rows]
                sep[cell][idx] = _separation(s_vals, e_vals)

    # gt-in-topK recall (S+crosshair 행만).
    gt_summary: dict = {}
    for tname in TEMPLATES:
        for fname in FRAMES:
            cell = f"{tname}__{fname}"
            vals = [r["gt_topk"].get(cell) for r in rows if cell in r.get("gt_topk", {})]
            vals = [v for v in vals if v is not None]
            gt_summary[cell] = {
                "n": len(vals),
                "in_topk_rate": round(sum(1 for v in vals if v) / len(vals), 3) if vals else None,
            }

    return {
        "counts": {"S": len(s_rows), "E": len(e_rows), "total": len(rows)},
        "separation": sep,
        "gt_topk": gt_summary,
    }


def _print_summary(summary: dict) -> None:
    c = summary["counts"]
    print(f"\n[INFO] images: S={c['S']}  E={c['E']}  total={c['total']}")
    print("\n[INFO] 셀 = (template × frame). 각 셀 의미:")
    for (t, f), meaning in _CELL_MEANING.items():
        print(f"    {t:>6} × {f:<8} = {meaning}")

    print("\n[INFO] S/E 분리도 — balanced accuracy (1.0 에 가까울수록 좋은 변별, "
          "med_S 높고 med_E 낮아야 함):")
    print(f"  {'cell':<18} {'index':<13} {'med_S':>8} {'med_E':>8} {'bACC':>6}  n(S/E)")
    for cell, by_idx in summary["separation"].items():
        for idx, s in by_idx.items():
            print(f"  {cell:<18} {idx:<13} {str(s.get('median_s')):>8} "
                  f"{str(s.get('median_e')):>8} {str(s.get('balanced_accuracy','-')):>6}  "
                  f"{s.get('n_s')}/{s.get('n_e')}")

    print("\n[INFO] gt-in-topK recall (정답이 chamfer 후보에 드는 비율, S+crosshair):")
    for cell, g in summary["gt_topk"].items():
        print(f"  {cell:<18} in_topk_rate={g['in_topk_rate']}  (n={g['n']})")

    # verdict — at_crosshair(진짜 변별자) bACC 로 OLD vs NEW 비교, 없으면 free 로 폴백.
    def _bacc(cell, idx):
        return summary["separation"].get(cell, {}).get(idx, {}).get("balanced_accuracy")
    print("\n[INFO] VERDICT (at_crosshair bACC, 없으면 free):")
    for cell in ("center__raw", "box__raw", "center__inpaint", "box__inpaint"):
        v = _bacc(cell, "at_crosshair")
        idx_used = "at_crosshair"
        if v is None:
            v = _bacc(cell, "free")
            idx_used = "free"
        gt = summary["gt_topk"].get(cell, {}).get("in_topk_rate")
        meaning = _CELL_MEANING.get(tuple(cell.split("__")), "")
        print(f"  {cell:<18} {idx_used} bACC={v}  gt_topk={gt}   [{meaning}]")
    old = _bacc("center__raw", "at_crosshair") or _bacc("center__raw", "free")
    new = _bacc("box__inpaint", "at_crosshair") or _bacc("box__inpaint", "free")
    if old is not None and new is not None:
        delta = round(new - old, 3)
        better = "NEW better (교체 근거)" if delta > 0 else ("동률" if delta == 0 else "NEW worse [!]")
        print(f"\n  -> OLD(center+raw)={old}  vs  NEW(box+inpaint)={new}  delta={delta:+}  => {better}")
    print("  * box 셀이 비어 있으면 photometric box 미검출(이 recipe rcp 에선 폴백) - overlay 값 히스토그램 확인.")


def _collect_recipes() -> list:
    """완전 override env 면 단일 recipe, 아니면 align_images 전체 recipe leaf."""
    override = all(os.getenv(k) for k in ("ALIGN_EQP_ID", "ALIGN_CLASS_NAME", "ALIGN_RECIPE_NAME"))
    if override:
        a = resolve_assets_auto()
        return [a] if a is not None else []
    dirs = iter_recipe_dirs()
    if LIMIT_RECIPES is not None:
        dirs = dirs[:LIMIT_RECIPES]
    return [resolve_assets(d) for d in dirs]


def run() -> str:
    recipes = _collect_recipes()
    if not recipes:
        print("[ERROR] align_images 트리에서 recipe 를 찾지 못했습니다 - 오피스(실데이터) 환경에서 실행하세요. "
              "단일 지정: ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME.")
        return "no_data"

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    print(f"[INFO] recipe {len(recipes)}개 처리 → {out_dir}")

    all_rows: list[dict] = []
    with rows_path.open("w", encoding="utf-8") as rf:
        for assets in recipes:
            try:
                center_tpls, box_tpls = _build_templates(assets)
            except Exception as exc:
                print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
                continue
            if not any(v is not None for v in center_tpls.values()):
                print(f"[WARNING] {assets.recipe_id}: center template 없음 — 건너뜀.")
                continue
            box_ok = any(v is not None for v in box_tpls.values())
            msr_imgs = iter_msr_images(assets)
            if MAX_MSR_PER_RECIPE is not None:
                msr_imgs = msr_imgs[:MAX_MSR_PER_RECIPE]
            print(f"[INFO] {assets.recipe_id}: msr {len(msr_imgs)}장  "
                  f"(box template {'OK' if box_ok else '없음(폴백)'})")
            for p in msr_imgs:
                row = _process_msr(p, center_tpls, box_tpls)
                if row is None:
                    continue
                row["recipe"] = assets.recipe_id
                all_rows.append(row)
                rf.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not all_rows:
        print("[ERROR] 처리된 msr 행이 없습니다.")
        return "no_rows"

    summary = _summarize(all_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\n[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    run()
