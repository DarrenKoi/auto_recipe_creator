"""golden(항상 성공) 데이터셋에서 align-point *위치추정* 정확도를 검증한다 (S-only).

왜 이 스크립트가 따로 있나:
  `align_similarity.py` / `align_index_ablation.py` 의 주 판정은 S/E **분리도(bACC)** 였다.
  그런데 (a) bACC 는 "key 있음/없음" 분류이지 생산 과제(좌표 찾기)가 아니고,
  (b) recipe 마다 다른 절대점수를 전역 임계로 자르는 cross-recipe confound 가 있고,
  (c) E 의 crosshair 는 정의상 틀린 위치/없음이라 GT 로 신뢰할 수 없다.
  → "내 매처/전처리가 *정확한가*" 를 재려면 **GT 가 믿을 만한 데이터(=성공 S)** 에서
    **위치추정**(best_xy 가 정답 crosshair 에 떨어지나) 을 직접 재야 한다. 그 데이터가
    의도 수집한 golden set(`align_images_golden/`, S 만)이다. 참조:
    `docs/align_success_dataset_plan.md`, `docs/study/reranker_ab_failure_analysis.md`.

핵심 평가 규약 (사용자 확인 2026-06-04):
  - **흰 box 제거** = rcp template 쪽. 등록 이미지의 흰 unique-area box 안쪽만 깨끗이 crop
    한 ``box`` template 을 쓴다(구 ``center`` 면적 crop 과 A/B).
  - **crosshair 제거** = msr frame 쪽. crosshair 를 **검출한 뒤 그 위치를 GT 로 고정**하고,
    그 자리를 inpaint 로 지운 frame 에서 매칭한다. 즉 정답은 보존하되 매칭은 *crosshair
    없는 wafer* 에서 한다 — 이게 live 생산(십자 없음)에 더 충실하고, 매처가 crosshair 의
    강한 edge 에 "치팅"하지 못하게 한다.
  - 점수화 = inpaint 후에도 best_xy 가 *원래 crosshair 위치* 근처(허용오차 GT_TOL_NORM)에
    떨어지면 hit. 정답지가 이미 있으므로 분류가 아니라 **거리 기반 정오**로 채점.

설계 — 2×2 (template × frame), 각 셀에 동일 위치추정 지표:
    ┌────────────┬──────────────┬──────────────┐
    │            │ frame=raw    │ frame=inpaint│
    ├────────────┼──────────────┼──────────────┤
    │ tpl=center │ OLD baseline │ crosshair만  │
    │ tpl=box    │ box만        │ NEW(둘 다)   │
    └────────────┴──────────────┴──────────────┘
  verdict = NEW(box+inpaint) − OLD(center+raw) 의 rank1_hit / gt_in_topk 상승폭.

지표(셀별, S+crosshair 행만):
  - rank1_hit_rate     : free best_xy 가 crosshair 허용오차 내인 비율 (생산의 "1발 명중").
  - gt_in_topk_rate    : 정답이 chamfer top-N 후보에 드는 비율 (proposer recall).
  - topk_not_rank1_rate: 후보엔 있으나 1등이 아닌 비율 (리랭킹으로 메울 수 있던 갭).
  - median/p90 dist_norm: free best 의 정규화 거리 분포 (정밀도).
GT 위생(전역): S 장수, crosshair 검출률·평균 conf, label '?'/E 수 — GT 신뢰도 게이트.

실행 (오피스, 인자 없음):
    uv run python poc/workflow_2/golden_localization_eval.py
  golden 루트: 기본 `align_images_golden/`, env `ALIGN_GOLDEN_ROOT` 로 override.
  단일 recipe: env ALIGN_EQP_ID + ALIGN_CLASS_NAME + ALIGN_RECIPE_NAME (golden 루트 기준).
  golden 데이터가 없으면 합성 self-test 트리를 만들어 파이프라인만 점검한다.
출력: stdout 표 + DEBUG_IMAGE_DIR/golden_localization_eval/<ts>/{rows.jsonl, summary.json}
"""

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Windows cp949 콘솔에서 미지원 기호로 print 가 죽지 않도록(한글은 유지, 불가 문자만 치환).
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import json
import shutil
import statistics
import tempfile
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import (
    DEBUG_IMAGE_DIR,
    FROM_MSR_DIRNAME,
    FROM_RCP_DIRNAME,
    RCP_OM_STEM,
    RCP_SEM_STEM,
)
from poc.workflow_2 import ALIGN_IMAGES_ROOT
from poc.workflow_2.align_fail_assets import (
    iter_msr_images,
    iter_recipe_dirs,
    load_gray,
    resolve_assets,
    resolve_assets_auto,
)
from poc.workflow_2.align_key_matcher import (
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
)
from poc.workflow_2.align_point_correction import (
    _RcpTemplateBundle,
    _centered_area_crop_bbox,
    _detect_white_box,
    _draw_rcp_overlay,
    _inner_crop_for_box,
    _inpaint_crosshair,
    _tool_label,
)
from poc.workflow_2.crosshair_detect import detect_crosshair
# align_similarity 의 crop 비율·scale band·허용오차 상수만 재사용(정의 중복/표류 방지).
from poc.workflow_2.align_similarity import CENTER_AREA_RATIO, COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_1.util.time_utils import make_timestamp_tag

# golden 루트 — fail 트리(align_images)와 형제 폴더. env 로 override 가능.
GOLDEN_ROOT = ALIGN_IMAGES_ROOT.parent / "align_images_golden"
OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_localization_eval"

# golden 데이터가 없을 때 합성 self-test 로 파이프라인만 점검할지.
RUN_SELFTEST_IF_NO_DATA = True

# 처리 상한(빠른 시험용). None = 전부.
LIMIT_RECIPES = None
MAX_S_PER_RECIPE = None

# S 한 장마다 inpaint frame 위에 GT(crosshair) vs 예측 align point overlay 를 저장할지.
SAVE_OVERLAYS = True

# 2×2 축.
TEMPLATES = ("center", "box")    # center=구(면적 crop), box=신(흰 box 안쪽 photometric crop).
FRAMES = ("raw", "inpaint")      # raw=구(crosshair 있음), inpaint=신(crosshair 제거).

_CELL_MEANING = {
    "center__raw": "OLD baseline",
    "box__raw": "box-only (clean tpl)",
    "center__inpaint": "crosshair-only (inpaint)",
    "box__inpaint": "NEW (box + inpaint)",
}


# ------------------------------------------------------------------
# offset-aware template 빌드.
# ------------------------------------------------------------------
#
# 핵심: 매칭이 잡는 위치는 *template 중심*(center crop 의 중심 / 흰 box 안쪽 crop 의 중심)
# 이지만, recipe 에 기록된 align point 는 *이미지 중심*이다. 따라서 match 중심에
# ``align_offset = image_center - template_center`` 를 더해야 frame 에서의 align point 가
# 된다(생산 경로 align_point_correction._build_rcp_template 와 동일 규약). center crop 은
# crop 중심 = 이미지 중심이라 offset (0,0); box 는 box 가 off-center 면 offset != 0 이다.


def _build_offset_templates(assets, *, rcp_overlay_dir: Path | None = None) -> tuple[dict, dict]:
    """rcp 에서 center / box template + align_offset 을 modality 별로 만든다.

    반환: (center, box). 각 값은 ``(AlignKeyTemplate, (dx, dy))`` 또는 None.
    center = 정중앙 면적 crop(offset (0,0)); box = 흰 unique-area box 안쪽 crop
    (offset = image_center - inner_center). box 미검출 modality 는 None.
    offset 규약은 `_build_rcp_template` 와 동일 — match 중심 + offset = align point.

    rcp_overlay_dir 가 주어지면 modality 마다 rcp 위에 [검출 box(노랑)·inner crop
    template(초록)·align point=이미지중심(파랑)·offset 화살표(시안)] 를 그린 JPEG 을
    저장한다 — "흰 box 안쪽 crop 이 제대로 잡혔나" 를 눈으로 검증하는 용도.
    """
    from poc.workflow_2.align_fail_assets import load_gray

    center: dict = {}
    box: dict = {}
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
            (0, 0),  # center crop 중심 = 이미지 중심 = align point.
        )

        det = _detect_white_box(gray)
        if det is not None:
            inner, inner_bbox = _inner_crop_for_box(gray, det)
            ix, iy, iw, ih = inner_bbox
            offset = (img_cx - (ix + iw // 2), img_cy - (iy + ih // 2))
            box[mod] = (
                build_template(inner, recipe_id=assets.recipe_id,
                               version=version + "_box", key_type=key_type),
                offset,
            )
        else:
            box[mod] = None

        # rcp box-crop 검증 overlay (box 검출 성공/실패 둘 다 — 실패는 fallback crop 으로 표시).
        if rcp_overlay_dir is not None:
            if det is not None:
                bundle = _RcpTemplateBundle(
                    template=box[mod][0], align_offset_xy=offset,
                    detected_box=det, inner_crop=inner_bbox,
                )
            else:
                bundle = _RcpTemplateBundle(
                    template=center[mod][0], align_offset_xy=(0, 0),
                    detected_box=None, inner_crop=(cx, cy, cw, ch),
                )
            _draw_rcp_overlay(
                gray, bundle=bundle,
                out_path=rcp_overlay_dir / f"{assets.recipe_id}_{mod}_rcp.jpg",
                label=f"{assets.recipe_id}/{mod}",
            )
    return center, box


# ------------------------------------------------------------------
# 위치추정 한 셀.
# ------------------------------------------------------------------


def _localize(templates: dict, frame: np.ndarray, crosshair_xy: tuple[int, int]) -> dict | None:
    """offset-aware template 집합으로 frame 을 free 검색 → align point 가 crosshair(GT)에
    떨어지나 채점.

    각 modality 를 1회 매칭(compute_align_key_score)하고, match 중심·top-N 후보 좌표에
    그 modality 의 align_offset 을 더해 *align point* 로 환산한 뒤 crosshair 와 비교한다.
    modality 간에는 생산 free-best 와 동일하게 **최고 score** modality 를 채택(_race 규약).
    rank1_hit/dist 와 topk_rank 를 같은 후보 집합에서 일관되게 뽑는다(과거 _race+_gt_in_topk
    이중 패스 + box offset 누락 버그 제거).

    반환: {mod, score, align_xy, dist_norm, hit, topk_rank, in_topk} 또는 매칭 불가 시 None.
    dist_norm 은 승자 modality template 짧은 변 대비 거리(GT_TOL_NORM 과 동일 척도).
    """
    best = None  # (score, dist_norm, hit, rank, align_xy, mod)
    for mod, item in templates.items():
        if item is None:
            continue
        tpl, (dx, dy) = item
        th, tw = tpl.raw_image.shape[:2]
        short = max(1, min(tw, th))
        try:
            r = compute_align_key_score(
                tpl, frame, scales=COMPARE_SCALES, policy=STRUCTURE_POLICY,
            )
        except Exception as exc:
            print(f"[WARNING] score 실패 ({mod}): {exc}")
            continue

        # match 중심 → align point (offset 가산).
        ap = (r.best_xy[0] + dx, r.best_xy[1] + dy)
        dist_norm = float(np.hypot(ap[0] - crosshair_xy[0], ap[1] - crosshair_xy[1]) / short)

        # top-N 후보(score 내림차순)에도 동일 offset 적용 → align point 가 허용오차 내인 첫 rank.
        rank = None
        for i, c in enumerate(r.candidates, 1):
            cap = (c.xy[0] + dx, c.xy[1] + dy)
            if float(np.hypot(cap[0] - crosshair_xy[0], cap[1] - crosshair_xy[1]) / short) <= GT_TOL_NORM:
                rank = i
                break

        cur = (float(r.score), dist_norm, dist_norm <= GT_TOL_NORM, rank, ap, mod)
        if best is None or cur[0] > best[0]:  # race = 최고 score modality.
            best = cur

    if best is None:
        return None
    score, dist_norm, hit, rank, ap, mod = best
    return {
        "mod": mod,
        "score": round(score, 4),
        "align_xy": [int(ap[0]), int(ap[1])],
        "dist_norm": round(dist_norm, 4),
        "hit": bool(hit),
        "topk_rank": rank,
        "in_topk": rank is not None,
    }


# overlay 색상 (BGR). GT=초록, box 예측=주황, center 예측=시안. hit/miss 는 라벨로 표기.
_OVL_GT = (0, 200, 0)
_OVL_CELL = {"box__inpaint": (0, 170, 255), "center__inpaint": (220, 180, 0)}


def _save_overlay(
    frame_gray: np.ndarray,
    crosshair_xy: tuple[int, int],
    cells: dict,
    *,
    recipe: str,
    msr_name: str,
    out_dir: Path,
) -> str | None:
    """inpaint frame 위에 GT(crosshair) 와 예측 align point 를 그려 JPEG 로 저장.

    숫자(점수)뿐 아니라 *어디를 align point 로 찍었는지* 를 눈으로 확인하려는 용도.
    GT 와 각 예측을 선으로 이어 거리(=정오)를 직관적으로 본다. inpaint frame 기준의
    셀(center__inpaint, box__inpaint)만 그린다(raw 는 표로만).
    """
    canvas = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    cx, cy = int(crosshair_xy[0]), int(crosshair_xy[1])

    cv2.drawMarker(canvas, (cx, cy), _OVL_GT, cv2.MARKER_CROSS, 22, 2)
    cv2.circle(canvas, (cx, cy), 10, _OVL_GT, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{recipe}/{msr_name}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    legend = [("GT (crosshair)", _OVL_GT)]
    for cell in ("box__inpaint", "center__inpaint"):
        res = cells.get(cell)
        if not res:
            continue
        col = _OVL_CELL[cell]
        ax = int(np.clip(res["align_xy"][0], 0, w - 1))
        ay = int(np.clip(res["align_xy"][1], 0, h - 1))
        cv2.line(canvas, (cx, cy), (ax, ay), col, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (ax, ay), col, cv2.MARKER_TILTED_CROSS, 18, 2)
        tag = "box" if cell.startswith("box") else "center"
        legend.append((f"{tag} d={res['dist_norm']:.3f} {'HIT' if res['hit'] else 'MISS'}", col))
    for i, (text, col) in enumerate(legend):
        cv2.putText(canvas, text, (6, 38 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    sub = out_dir / recipe
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{Path(msr_name).stem}_overlay.jpg"
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(path)


def _process_msr(
    msr_path: Path,
    center_tpls: dict,
    box_tpls: dict,
    *,
    recipe_id: str = "",
    overlay_dir: Path | None = None,
) -> dict | None:
    """golden msr 한 장 → 2×2 셀의 위치추정 지표 (+ overlay). (S+crosshair 행만 cells 채움.)

    GT 는 **inpaint 전 raw 에서 검출한 crosshair**. 검출 실패 시 cells 없이 위생 행만
    남긴다(생산과 동일: GT 없으면 위치추정 지표에서 제외 — 합성 fallback 없음).
    overlay_dir 이 주어지면 inpaint frame 위에 GT vs 예측 align point 를 그려 저장한다.
    """
    try:
        gray_raw = load_gray(msr_path)
    except Exception as exc:
        print(f"[WARNING] msr 로드 실패 {msr_path.name}: {exc}")
        return None

    label = _tool_label(msr_path.name)
    ch = detect_crosshair(gray_raw)
    crosshair_xy = ch.xy

    row: dict = {
        "msr": msr_path.name,
        "label": label,
        "crosshair_xy": list(crosshair_xy) if crosshair_xy is not None else None,
        "crosshair_conf": round(ch.confidence, 3),
        "cells": {},
        "overlay": None,
    }
    # golden 은 S 만이어야 한다. S 가 아니거나 crosshair 미검출이면 GT 없음 → 위생만 기록.
    if label != "S" or crosshair_xy is None:
        return row

    gray_inp = _inpaint_crosshair(gray_raw, crosshair_xy)
    frames = {"raw": gray_raw, "inpaint": gray_inp}
    tpl_sets = {"center": center_tpls, "box": box_tpls}

    for tname in TEMPLATES:
        tpls = tpl_sets[tname]
        if all(v is None for v in tpls.values()):
            continue  # 이 recipe rcp 에서 box 미검출 → box 셀 비움.
        for fname in FRAMES:
            res = _localize(tpls, frames[fname], crosshair_xy)
            if res is not None:
                row["cells"][f"{tname}__{fname}"] = res

    if SAVE_OVERLAYS and overlay_dir is not None and row["cells"]:
        row["overlay"] = _save_overlay(
            gray_inp, crosshair_xy, row["cells"],
            recipe=recipe_id or "recipe", msr_name=msr_path.name, out_dir=overlay_dir,
        )
    return row


# ------------------------------------------------------------------
# 집계.
# ------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1)))))
    return round(sorted_vals[i], 4)


def _cell_stats(vals: list[dict]) -> dict:
    """한 셀의 위치추정 표본 → 명중률/recall/거리 분포."""
    n = len(vals)
    if n == 0:
        return {"n": 0}
    hits = sum(1 for v in vals if v["hit"])
    in_topk = sum(1 for v in vals if v["in_topk"])
    topk_not_rank1 = sum(
        1 for v in vals if v["in_topk"] and v["topk_rank"] is not None and v["topk_rank"] > 1
    )
    dists = sorted(v["dist_norm"] for v in vals)
    return {
        "n": n,
        "rank1_hit_rate": round(hits / n, 3),
        "gt_in_topk_rate": round(in_topk / n, 3),
        "topk_not_rank1_rate": round(topk_not_rank1 / n, 3),
        "median_dist_norm": _percentile(dists, 0.5),
        "p90_dist_norm": _percentile(dists, 0.9),
    }


def _summarize(rows: list[dict]) -> dict:
    s_rows = [r for r in rows if r["label"] == "S"]
    n_s = len(s_rows)
    with_ch = [r for r in s_rows if r["crosshair_xy"] is not None]
    confs = [r["crosshair_conf"] for r in with_ch]

    hygiene = {
        "n_total": len(rows),
        "n_S": n_s,
        "n_E": sum(1 for r in rows if r["label"] == "E"),
        "n_question": sum(1 for r in rows if r["label"] == "?"),
        "n_S_with_crosshair": len(with_ch),
        "n_S_no_crosshair": n_s - len(with_ch),
        "crosshair_detect_rate": round(len(with_ch) / n_s, 3) if n_s else None,
        "mean_crosshair_conf": round(statistics.mean(confs), 3) if confs else None,
    }

    cells: dict = {}
    for tname in TEMPLATES:
        for fname in FRAMES:
            cell = f"{tname}__{fname}"
            vals = [r["cells"][cell] for r in s_rows if cell in r.get("cells", {})]
            cells[cell] = _cell_stats(vals)

    return {"hygiene": hygiene, "cells": cells}


def _print_summary(summary: dict) -> None:
    h = summary["hygiene"]
    print("\n[INFO] === GT 위생 (위치추정의 정답 신뢰도 게이트) ===")
    print(f"  images: total={h['n_total']}  S={h['n_S']}  E={h['n_E']}  '?'={h['n_question']}")
    print(f"  crosshair 검출: {h['n_S_with_crosshair']}/{h['n_S']} "
          f"(rate={h['crosshair_detect_rate']}, mean_conf={h['mean_crosshair_conf']})")
    if h["n_E"] or h["n_question"]:
        print("  [!] golden 은 S-only 가정 — E/'?' 가 있으면 라벨 오염 점검 필요.")
    if h["n_S_no_crosshair"]:
        print(f"  [!] crosshair 미검출 S {h['n_S_no_crosshair']}장은 GT 없음 → 위치추정에서 제외됨.")

    print("\n[INFO] 셀 = (template × frame). 의미:")
    for cell, meaning in _CELL_MEANING.items():
        t, f = cell.split("__")
        print(f"    {t:>6} × {f:<8} = {meaning}")

    print("\n[INFO] === 위치추정 지표 (S+crosshair 만; 높을수록 좋음, dist 는 낮을수록) ===")
    print(f"  {'cell':<18} {'n':>4} {'rank1_hit':>10} {'gt_topk':>8} "
          f"{'topk!=1':>8} {'med_dist':>9} {'p90_dist':>9}")
    for cell, s in summary["cells"].items():
        if s.get("n", 0) == 0:
            print(f"  {cell:<18} {0:>4}   (표본 없음 — box 미검출 또는 데이터 없음)")
            continue
        print(f"  {cell:<18} {s['n']:>4} {s['rank1_hit_rate']:>10} {s['gt_in_topk_rate']:>8} "
              f"{s['topk_not_rank1_rate']:>8} {str(s['median_dist_norm']):>9} "
              f"{str(s['p90_dist_norm']):>9}")

    old = summary["cells"].get("center__raw", {})
    new = summary["cells"].get("box__inpaint", {})
    print("\n[INFO] === VERDICT: NEW(box+inpaint) vs OLD(center+raw) ===")
    if old.get("n", 0) == 0 or new.get("n", 0) == 0:
        print("  비교 불가 — 한쪽 셀에 표본이 없음(box 미검출 가능). 위 표/위생 확인.")
        return
    for key, label in (("rank1_hit_rate", "rank1_hit"), ("gt_in_topk_rate", "gt_in_topk")):
        d = round(new[key] - old[key], 3)
        verdict = "NEW better (교체 근거)" if d > 0 else ("동률" if d == 0 else "NEW worse [!]")
        print(f"  {label:<12} OLD={old[key]}  NEW={new[key]}  delta={d:+}  => {verdict}")
    print("  * rank1_hit 가 주 판정(생산 1발 명중). gt_in_topk 는 proposer recall 보조.")


# ------------------------------------------------------------------
# recipe 수집.
# ------------------------------------------------------------------


def _collect_recipes(root: Path) -> list:
    """완전 override env 면 단일 recipe(해당 root 기준), 아니면 root 의 모든 recipe leaf."""
    override = all(os.getenv(k) for k in ("ALIGN_EQP_ID", "ALIGN_CLASS_NAME", "ALIGN_RECIPE_NAME"))
    if override:
        a = resolve_assets_auto(root=root)
        return [a] if a is not None else []
    dirs = iter_recipe_dirs(root)
    if LIMIT_RECIPES is not None:
        dirs = dirs[:LIMIT_RECIPES]
    return [resolve_assets(d) for d in dirs]


# ------------------------------------------------------------------
# 합성 self-test — golden 데이터가 없을 때 임시 트리를 만들어 파이프라인만 점검.
# ------------------------------------------------------------------


def _make_rcp_with_box(pattern: np.ndarray, box_shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    """패턴 + 흰 unique-area box(밝은 사각 outline)로 rcp 이미지를 합성한다.

    `_detect_white_box` 의 게이트에 맞춘다: box 면적 ~19%(<40% 상한), 가장자리 비접촉,
    얇은 outline(hollow), box 만 255 의 photometric 섬이 되도록 패턴을 <200 으로 눌러둔다.
    ``box_shift`` 로 box(=패턴) 를 이미지 중심에서 (sx, sy) 만큼 옮기면 align_offset 이
    (-sx, -sy) 가 되어 offset 보정 경로를 점검할 수 있다(0,0 이면 종전과 동일 중앙).
    """
    sx, sy = box_shift
    # 패턴을 200 미만으로 눌러 유일한 255 island 가 box outline 이 되게 한다.
    p = np.clip(pattern.astype(np.float32) * 0.7 + 20.0, 0, 190).astype(np.uint8)
    ph, pw = p.shape[:2]
    box_pad = 10  # 패턴과 box outline 사이 여백.
    iw, ih = pw + 2 * box_pad, ph + 2 * box_pad  # box 한 변(가로, 세로).
    margin = int(round(max(iw, ih) * 0.65))       # 배경 여백 → box 면적 ≈ (1/2.3)^2 ≈ 0.19.
    # shift 만큼 캔버스를 키워 box 가 가장자리에 닿지 않게 한다.
    canvas_w = iw + 2 * margin + 2 * abs(sx)
    canvas_h = ih + 2 * margin + 2 * abs(sy)
    canvas = np.full((canvas_h, canvas_w), 90, dtype=np.uint8)
    # box 중심 = 이미지 중심 + shift → box top-left.
    box_cx, box_cy = canvas_w // 2 + sx, canvas_h // 2 + sy
    bx0, by0 = box_cx - iw // 2, box_cy - ih // 2
    canvas[by0 + box_pad:by0 + box_pad + ph, bx0 + box_pad:bx0 + box_pad + pw] = p
    cv2.rectangle(canvas, (bx0, by0), (bx0 + iw - 1, by0 + ih - 1), 255, 2)
    return canvas


def _make_msr_frame(pattern: np.ndarray, seed: int, align_shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    """wafer 배경에 패턴을 drift 시켜 박고, *align point* 위치에 full-span 십자를 그린다.

    매칭은 패턴(=box 안쪽)을 그 embed 중심에서 잡고, align point = 중심 + align_offset 이다.
    self-test 가 일관되려면 crosshair(GT)를 패턴 중심이 아니라 (중심 + align_shift) 에 둔다
    (``align_shift`` = rcp 의 align_offset = -box_shift).
    """
    from poc.workflow_2.test_align_key_match import embed_pattern, make_wafer_background

    bg = make_wafer_background()
    frame, (cx, cy), _w, _h = embed_pattern(
        bg, pattern, rotation_deg=2.0, scale=0.95, brightness=-8, contrast=0.85, rng_seed=seed,
    )
    fh, fw = frame.shape[:2]
    gx = int(np.clip(cx + align_shift[0], 1, fw - 2))
    gy = int(np.clip(cy + align_shift[1], 1, fh - 2))
    # detect_crosshair 가 잡도록 full-span(>SPAN_RATIO) 밝은(>SAT ladder 235) 십자.
    cv2.line(frame, (0, gy), (fw - 1, gy), 255, 1)
    cv2.line(frame, (gx, 0), (gx, fh - 1), 255, 1)
    return frame


def _build_selftest_golden(root: Path) -> None:
    """임시 golden 트리(S-only)를 합성한다 — reader/template/crosshair/offset/위치추정 경로 점검.

    RCP_A = 중앙 box(offset 0), RCP_B = off-center box(offset != 0)로 두어, offset 보정이
    빠지면 RCP_B 의 box 셀 rank1_hit 가 떨어지도록 회귀 가드를 만든다.
    """
    from poc.workflow_2.test_align_key_match import make_synthetic_template

    pattern = make_synthetic_template(key_type="box")
    # (eqp, class, recipe, seed, box_shift) — box_shift 가 0 이 아니면 off-center.
    recipes = [
        ("EQPSELF", "CLSA", "RCP_A", 11, (0, 0)),
        ("EQPSELF", "CLSA", "RCP_B", 22, (24, 16)),
    ]
    jpg = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    for eqp, cls, rcp_name, base_seed, box_shift in recipes:
        rcp = _make_rcp_with_box(pattern, box_shift=box_shift)
        align_shift = (-box_shift[0], -box_shift[1])  # align_offset = -box_shift.
        leaf = root / eqp / cls / rcp_name
        rcp_dir = leaf / FROM_RCP_DIRNAME
        msr_dir = leaf / FROM_MSR_DIRNAME
        rcp_dir.mkdir(parents=True, exist_ok=True)
        msr_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rcp_dir / f"{RCP_OM_STEM}.jpg"), rcp, jpg)
        cv2.imwrite(str(rcp_dir / f"{RCP_SEM_STEM}.jpg"), rcp, jpg)
        for k in range(4):
            frame = _make_msr_frame(pattern, base_seed + k, align_shift=align_shift)
            cv2.imwrite(str(msr_dir / f"S01A{k + 1:04d}.jpg"), frame, jpg)


# ------------------------------------------------------------------
# 엔트리.
# ------------------------------------------------------------------


def run() -> str:
    """golden(S-only) 트리에서 위치추정 검증을 끝까지 돌린다 (인자 없음).

    golden 루트(기본 `align_images_golden/`, env `ALIGN_GOLDEN_ROOT`/단일 recipe override)를
    순회하며 2×2 셀의 위치추정 지표 + GT 위생을 집계해 stdout 표와 rows.jsonl/summary.json
    으로 남긴다. 데이터가 없으면 합성 self-test 트리로 파이프라인만 점검한다.
    반환: "success" | "no_data" | "no_rows".
    """
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else GOLDEN_ROOT
    recipes = _collect_recipes(root) if root.is_dir() else []

    tmp_dir = None
    selftest = False
    if not recipes:
        if not RUN_SELFTEST_IF_NO_DATA:
            print(f"[ERROR] golden 데이터를 찾지 못했습니다: {root} "
                  f"(env ALIGN_GOLDEN_ROOT 로 경로 지정). self-test 도 꺼져 있습니다.")
            return "no_data"
        print(f"[WARNING] golden 데이터 없음({root}) → 합성 self-test 로 파이프라인 점검")
        tmp_dir = Path(tempfile.mkdtemp(prefix="golden_selftest_"))
        _build_selftest_golden(tmp_dir)
        recipes = _collect_recipes(tmp_dir)
        selftest = True

    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    overlay_dir = (out_dir / "overlays") if SAVE_OVERLAYS else None
    rcp_overlay_dir = (out_dir / "rcp_templates") if SAVE_OVERLAYS else None
    if rcp_overlay_dir is not None:
        rcp_overlay_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] recipe {len(recipes)}개 처리 → {out_dir}"
          + ("  (SELF-TEST)" if selftest else ""))

    all_rows: list[dict] = []
    n_skip_no_msr = 0  # from_msr 가 비어(이미지가 도구에서 삭제됨) 채점 불가로 건너뛴 recipe 수.
    try:
        with rows_path.open("w", encoding="utf-8") as rf:
            for assets in recipes:
                if assets is None:
                    continue
                # 가드레일: from_msr 폴더가 없거나 비었으면(도구에서 측정 이미지 삭제 등) 정답
                # 프레임 자체가 없어 위치추정 불가 → template 빌드 전에 명시적으로 건너뛴다.
                # iter_recipe_dirs 는 from_rcp 만 보고 recipe 를 모으므로 msr 없는 leaf 도
                # 여기까지 들어온다. 조용히 0행이 되지 않게 skip 을 로그+카운트로 드러낸다.
                msr_imgs = iter_msr_images(assets)
                if not msr_imgs:
                    n_skip_no_msr += 1
                    print(f"[INFO] {assets.recipe_id}: from_msr 이미지 없음(도구에서 삭제 가능) → 건너뜀(skip)")
                    continue
                try:
                    center_tpls, box_tpls = _build_offset_templates(
                        assets, rcp_overlay_dir=rcp_overlay_dir,
                    )
                except Exception as exc:
                    print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
                    continue
                if not any(v is not None for v in center_tpls.values()):
                    print(f"[WARNING] {assets.recipe_id}: center template 없음 — 건너뜀.")
                    continue
                box_ok = any(v is not None for v in box_tpls.values())
                if MAX_S_PER_RECIPE is not None:
                    msr_imgs = msr_imgs[:MAX_S_PER_RECIPE]
                print(f"[INFO] {assets.recipe_id}: msr {len(msr_imgs)}장  "
                      f"(box template {'OK' if box_ok else '없음(폴백)'})")
                for p in msr_imgs:
                    row = _process_msr(
                        p, center_tpls, box_tpls,
                        recipe_id=assets.recipe_id, overlay_dir=overlay_dir,
                    )
                    if row is None:
                        continue
                    row["recipe"] = assets.recipe_id
                    all_rows.append(row)
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if n_skip_no_msr:
        print(f"\n[INFO] from_msr 이미지 없어 건너뛴 recipe: {n_skip_no_msr}개 "
              f"(도구에서 측정 이미지 삭제된 케이스 — 정상 skip).")

    if not all_rows:
        print("[ERROR] 처리된 msr 행이 없습니다."
              + (f" 수집 recipe 전부가 from_msr 비어있음(skip {n_skip_no_msr}개)."
                 if n_skip_no_msr else ""))
        return "no_rows"

    summary = _summarize(all_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(summary)
    if overlay_dir is not None and overlay_dir.is_dir():
        n_ovl = sum(1 for r in all_rows if r.get("overlay"))
        print(f"\n[INFO] overlay {n_ovl}장 저장 (GT vs 예측 align point): {overlay_dir}")
    if rcp_overlay_dir is not None and rcp_overlay_dir.is_dir():
        n_rcp = len(list(rcp_overlay_dir.glob("*_rcp.jpg")))
        print(f"[INFO] rcp template overlay {n_rcp}장 저장 "
              f"(노랑=검출 box, 초록=inner crop, 파랑=align point, 시안 화살표=offset): {rcp_overlay_dir}")
    print(f"\n[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
