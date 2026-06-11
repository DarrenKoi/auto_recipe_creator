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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import ALIGN_IMAGES_ROOT, DEBUG_IMAGE_DIR
from poc.workflow_3.align.assets import (
    iter_msr_images,
    load_gray,
    resolve_assets,
)
from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.diagnostics.align_point_correction import (
    RCP_FALLBACK_CENTER_CROP_AREA_RATIO,
    _RcpTemplateBundle,
    _centered_area_crop_bbox,
    _draw_rcp_overlay,
    _inpaint_crosshair,
    _tool_label,
)
from poc.workflow_2.align_similarity import CENTER_AREA_RATIO
from poc.workflow_3.align.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.align.cond_template import (
    CROP_INSET_PX,
    MIN_INNER_PX,
    OFFSET_SKIP,
    OFFSET_WARN,
    WARN_INNER_PX,
    _cond_box_center,
    _cond_box_to_xywh,
    check_cond_box,
    cond_align_offset,
    cond_offset_norm,
    cond_template_crop,
)
from poc.workflow_3.align.cond_file import CondInfo, load_cond, msr_modality
from poc.workflow_3.align.diagnostics.crosshair_detect import detect_crosshair
# 원본의 cond-독립 부품 재사용(중복/표류 방지).
from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_3.util.time_utils import make_timestamp_tag

GOLDEN_ROOT = ALIGN_IMAGES_ROOT.parent / "align_images_golden"
OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_localization_eval_cond"
SAVE_OVERLAYS = gle.SAVE_OVERLAYS
TEMPLATES = gle.TEMPLATES
FRAMES = gle.FRAMES

# 결합 패널(rcp | msr) 시각 상수.
_PANEL_HEADER_PX = 22          # 각 패널 위 라벨 띠 높이.
_PANEL_SEP_PX = 6              # 좌/우 패널 사이 구분선 너비.
_PANEL_SEP_BGR = (40, 40, 40)  # 구분선 색(어두운 회색).
_PANEL_HEADER_BGR = (30, 30, 30)


def _with_header(canvas, text):
    """패널 위에 라벨 띠(_PANEL_HEADER_PX)를 붙인다 — 너비는 보존, 높이만 늘어남."""
    _, w = canvas.shape[:2]
    band = np.full((_PANEL_HEADER_PX, w, 3), _PANEL_HEADER_BGR, dtype=np.uint8)
    cv2.putText(band, text, (6, _PANEL_HEADER_PX - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([band, canvas])


def _resize_to_height(canvas, target_h):
    """aspect 유지하며 target_h 로 리사이즈(이미 같으면 그대로)."""
    h, w = canvas.shape[:2]
    if h == target_h:
        return canvas
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(canvas, (new_w, target_h), interpolation=interp)


def _combine_2up(rcp_canvas, msr_canvas, *, rcp_label, msr_label):
    """rcp(좌) | msr(우) 를 같은 높이로 맞춰 가로 결합한 BGR canvas 를 돌려준다.

    추출(rcp box template) → 매칭 → align point 찍기를 한 장에서 추적하려는 용도.
    각 패널 위에 라벨 띠를 붙이고, 높이가 다르면 더 큰 쪽에 맞춰 aspect 유지 리사이즈한 뒤
    사이에 구분선을 둔다. ``rcp_canvas`` 가 None(해당 modality box template 부재)이면
    msr 패널만 라벨 붙여 돌려준다(너비 보존).
    """
    msr = _with_header(msr_canvas, msr_label)
    if rcp_canvas is None:
        return msr
    rcp = _with_header(rcp_canvas, rcp_label)
    target_h = max(rcp.shape[0], msr.shape[0])
    rcp = _resize_to_height(rcp, target_h)
    msr = _resize_to_height(msr, target_h)
    sep = np.full((target_h, _PANEL_SEP_PX, 3), _PANEL_SEP_BGR, dtype=np.uint8)
    return cv2.hconcat([rcp, sep, msr])


# --- cond box → template/offset (검출 무관, cond.txt 만으로 결정) -----------------
# 원본 eval 은 box 의 *내용 검출* inner-crop 중심에서 offset 을 뽑아, crop 오류가
# offset(전체 eval 의 load-bearing 값)을 오염시켰다. cond.txt 가 정확한 corner 를
# 주므로 (1) offset = image_center − box_center 를 crop 과 분리해 기하로만 계산하고,
# (2) template 은 stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 해 만든다
# (crop 중심 == box 중심 → offset 과 일관). Codex 검토 반영([[project_align_cond_files_and_coords]]).

# --- Tier 1.1 검증: miss-distance bin 층화 (리포팅 전용; 경계는 1차 결과 후 재조정 가능) ---
# A: GT 가 frame 중심에서 떨어진 정도(frame 짧은변 비율) = 구조적 displacement.
DISP_BINS = ((0.10, "near"), (0.20, "mid"), (0.35, "far"))   # 그 외 → "veryfar".
# B: center-crop arm 이 GT 에서 빗나간 거리(GT_TOL_NORM 배수) = rescue framing.
RESCUE_MULT = ((1.0, "hit"), (2.0, "near"), (4.0, "far"))    # 그 외 → "veryfar".
BIN_FRAME = "inpaint"   # 층화는 clean(inpaint) cell 로 — lever_verdict(box__inpaint)와 일관.

FORCE_ENSEMBLE = True   # Tier 1.1 검증은 항상 production ensemble 매처로.


def _apply_matcher_default():
    """FORCE_ENSEMBLE 이면 ALIGN_USE_ENSEMBLE 기본값을 1 로 채운다(명시적 0 은 존중)."""
    if FORCE_ENSEMBLE:
        os.environ.setdefault("ALIGN_USE_ENSEMBLE", "1")


def _bin_label(value, edges, over_label):
    """value 를 오름차순 (경계, 라벨) edges 로 분류. 어느 경계도 안 넘으면 over_label."""
    for edge, label in edges:
        if value < edge:
            return label
    return over_label


def displacement_bin(gt_xy, frame_hw):
    """GT(정렬점)가 frame 중심에서 얼마나 떨어졌나 → near/mid/far/veryfar.

    norm = |GT - frame_center| / frame 짧은변. '구조적 displacement' 의 직접 척도.
    """
    h, w = frame_hw
    short = max(1, min(int(w), int(h)))
    norm = float(np.hypot(gt_xy[0] - w / 2.0, gt_xy[1] - h / 2.0) / short)
    return _bin_label(norm, DISP_BINS, "veryfar")


def rescue_bin(center_dist_norm):
    """center-crop arm 이 GT 에서 얼마나 빗나갔나 → hit/near/far/veryfar (GT_TOL_NORM 배수)."""
    return _bin_label(float(center_dist_norm) / gle.GT_TOL_NORM, RESCUE_MULT, "veryfar")


def _arm_rates(vals):
    """_localize 결과 표본 → {n, gt_in_topk, rank1}. _cell_stats 의 bin 표용 부분집합."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "gt_in_topk": None, "rank1": None}
    return {
        "n": n,
        "gt_in_topk": round(sum(1 for v in vals if v["in_topk"]) / n, 3),
        "rank1": round(sum(1 for v in vals if v["hit"]) / n, 3),
    }


def _binned_localization_report(all_rows):
    """S row 들을 두 방식으로 층화 → bin × arm(center/box) 의 gt_in_topk/rank1 집계.

    A by_displacement: GT-from-frame-center(구조적 displacement). frame_hw 필요.
    B by_center_miss : center-crop arm 의 dist_norm(rescue; box 가 center 실패를 건지나).
    둘 다 inpaint cell 기준. matcher 재실행 없이 row 후처리만. frame_hw 결손 행은
    A 에서 제외하고 경고 카운트만 올린다(조용한 누락 방지).
    """
    cc, bc = f"center__{BIN_FRAME}", f"box__{BIN_FRAME}"
    disp = {b: {"center": [], "box": []} for b in ("near", "mid", "far", "veryfar")}
    resc = {b: {"center": [], "box": []} for b in ("hit", "near", "far", "veryfar")}
    n_no_frame = 0
    for r in all_rows:
        if r.get("label") != "S":
            continue
        cells = r.get("cells", {})
        center, box = cells.get(cc), cells.get(bc)
        gt, fhw = r.get("crosshair_xy"), r.get("frame_hw")
        # A: displacement (GT + frame 크기).
        if gt is not None and fhw is not None:
            b = displacement_bin(gt, fhw)
            if center is not None:
                disp[b]["center"].append(center)
            if box is not None:
                disp[b]["box"].append(box)
        elif gt is not None and center is not None:
            n_no_frame += 1
        # B: rescue (center cell 의 dist_norm).
        if center is not None:
            b = rescue_bin(center["dist_norm"])
            resc[b]["center"].append(center)
            if box is not None:
                resc[b]["box"].append(box)

    def _roll(binmap, order):
        return {b: {arm: _arm_rates(binmap[b][arm]) for arm in ("center", "box")}
                for b in order}

    return {
        "frame": BIN_FRAME,
        "n_no_frame_hw": n_no_frame,
        "by_displacement": _roll(disp, ("near", "mid", "far", "veryfar")),
        "by_center_miss": _roll(resc, ("hit", "near", "far", "veryfar")),
    }


def _print_binned_report(binned):
    """bin × arm(center/box) 표를 콘솔로 — office digest 가 그대로 베껴쓸 형식."""
    print("\n" + "=" * 64)
    print(f"[INFO] === Tier 1.1 box-crop 게이트 (frame={binned['frame']}) ===")
    if binned.get("n_no_frame_hw"):
        print(f"  [WARNING] frame_hw 없는 S 행 {binned['n_no_frame_hw']}개 → displacement 표서 제외.")
    for title, key, order in (
        ("[A] by structural displacement (GT-from-center)", "by_displacement",
         ("near", "mid", "far", "veryfar")),
        ("[B] by center-arm miss (rescue)", "by_center_miss",
         ("hit", "near", "far", "veryfar")),
    ):
        print(f"\n  {title}")
        print(f"    {'bin':<9} {'center gt_in_topk/rank1':<26} {'box gt_in_topk/rank1':<24} n(c/b)")
        for b in order:
            c, x = binned[key][b]["center"], binned[key][b]["box"]
            print(f"    {b:<9} {str(c['gt_in_topk'])+'/'+str(c['rank1']):<26} "
                  f"{str(x['gt_in_topk'])+'/'+str(x['rank1']):<24} {c['n']}/{x['n']}")
    print("=" * 64)


def _offset_diag_cond(records, *, tol=OFFSET_WARN):
    """align-offset 진단 — *대각선 정규화* 척도 전용(cond_offset_norm 과 동일).

    gle._offset_diag 는 GT_TOL_NORM=0.20 *short-side* 기준이라, 대각선 정규화 값을 그대로
    넣으면 척도가 어긋나 '가정민감' 을 과소계상한다. 그래서 cond 판은 OFFSET_WARN(대각선)
    기준으로 자체 집계한다([[project_align_cond_files_and_coords]]).
    """
    if not records:
        return {"n": 0, "tol": tol}
    norms = sorted(r["offset_norm"] for r in records)
    sensitive = [r for r in records if r["offset_norm"] > tol]
    return {
        "n": len(norms),
        "tol": tol,
        "median_offset_norm": gle._percentile(norms, 0.5),
        "p90_offset_norm": gle._percentile(norms, 0.9),
        "n_assumption_sensitive": len(sensitive),
        "sensitive": [{"recipe": r["recipe"], "mod": r["mod"],
                       "offset_norm": r["offset_norm"]} for r in sensitive],
    }


def _build_offset_templates_cond(
    assets, *, rcp_overlay_dir=None, box_reasons=None, offset_records=None,
    rcp_canvases=None,
):
    """rcp 에서 center / box template 을 만든다 — box 는 cond.box_ltrb 에서(검출 아님).

    center = 정중앙 면적 crop(offset (0,0)).
    box = cond box stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 한 crop.
      - offset 은 crop 과 **분리**: cond_align_offset = image_center − box_center
        (검출 inner-crop 중심이 아니라 cond 기하로만 → off-center 오염 없음).
      - 대칭 inset 이라 crop 중심 == box 중심 → offset 과 정확히 일관.
    check_cond_box 가 skip(너무 작음/경계 밖/offset 과도) 판정한 box 는 None 처리.
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
        status, reason = ("absent", "cond:absent")
        if box_ltrb is not None:
            status, reason, onorm = check_cond_box(box_ltrb, gray.shape)
        if box_ltrb is not None and status != "skip":
            # offset 은 crop 과 분리 — cond box 중심으로만 결정(검출 오염 없음).
            offset = cond_align_offset(box_ltrb, gray.shape)
            inner, inner_bbox = cond_template_crop(gray, cond)
            box[mod] = (
                build_template(inner, recipe_id=assets.recipe_id,
                               version=version + "_box", key_type=key_type),
                offset,
            )
            if box_reasons is not None:
                tag = f"ok:cond:{status}"        # ok:cond:ok / ok:cond:warn
                box_reasons[tag] = box_reasons.get(tag, 0) + 1
            if offset_records is not None:
                offset_records.append(
                    {"recipe": assets.recipe_id, "mod": mod,
                     "offset_norm": round(onorm, 4), "status": status})
            if rcp_overlay_dir is not None or rcp_canvases is not None:
                det = _cond_box_to_xywh(box_ltrb)   # 노란 box = 정확한 cond box.
                bundle = _RcpTemplateBundle(
                    template=box[mod][0], align_offset_xy=offset,
                    detected_box=det, inner_crop=inner_bbox)   # 초록 = inpaint+inset template.
                out_path = (rcp_overlay_dir / f"{assets.recipe_id}_{mod}_rcp.jpg"
                            if rcp_overlay_dir is not None else None)
                canvas = _draw_rcp_overlay(
                    gray, bundle=bundle, out_path=out_path,
                    label=f"{assets.recipe_id}/{mod} [{status}]")
                if rcp_canvases is not None:
                    rcp_canvases[mod] = canvas
        else:
            box[mod] = None
            if box_reasons is not None:
                tag = reason if box_ltrb is not None else "cond:absent"
                box_reasons[tag] = box_reasons.get(tag, 0) + 1
            if rcp_overlay_dir is not None or rcp_canvases is not None:
                fb_bbox = _centered_area_crop_bbox(gray, RCP_FALLBACK_CENTER_CROP_AREA_RATIO)
                bundle = _RcpTemplateBundle(
                    template=center[mod][0], align_offset_xy=(0, 0),
                    detected_box=None, inner_crop=fb_bbox)
                out_path = (rcp_overlay_dir / f"{assets.recipe_id}_{mod}_rcp.jpg"
                            if rcp_overlay_dir is not None else None)
                canvas = _draw_rcp_overlay(
                    gray, bundle=bundle, out_path=out_path,
                    label=f"{assets.recipe_id}/{mod} [{reason}]",
                    fallback_color=(0, 0, 255), fallback_thickness=2)
                if rcp_canvases is not None:
                    rcp_canvases[mod] = canvas
    return center, box


def _route_modality(cond, available_mods):
    """msr frame 을 어느 modality rcp template 으로 매칭할지 결정 ('om'|'sem'|None).

    과거 `_localize` 는 om·sem 둘 다 매칭 후 최고 score 채택(_race)이었는데, chamfer
    점수면이 평평([[project_matcher_flat_chamfer_distinctiveness]])해 **틀린 modality
    whitebox 가 이겨** 지표를 오염시켰다(2026-06-09 오피스 결합 패널에서 발각). 그래서
    msr 의 실제 modality 로 라우팅한다:
      - msr 키/배율 추론(`msr_modality`)이 확정되면 그 modality(단, 해당 rcp 가 있을 때).
      - 미상이면 recipe 가 단일 modality 면 그걸로 폴백.
      - 미상 + dual-rcp, 또는 추론 modality 의 rcp 부재 → None(skip; 틀린-modality 측정 차단).
    ``available_mods`` 는 이 recipe 에서 template 이 있는 modality 집합.
    """
    inferred = msr_modality(cond)
    if inferred is not None:
        return inferred if inferred in available_mods else None
    if len(available_mods) == 1:
        return next(iter(available_mods))
    return None


def _winning_mod(cells):
    """결합 패널에 함께 보일 rcp 의 modality — 주판정(box__inpaint) 우선, 없으면 차순위."""
    for key in ("box__inpaint", "center__inpaint", "box__raw", "center__raw"):
        res = cells.get(key)
        if res:
            return res.get("mod")
    return None


def _process_msr_cond(msr_path, center_tpls, box_tpls, *, recipe_id="",
                      overlay_dir=None, rcp_canvases=None, combined_dir=None):
    """golden msr 한 장 → 2×2 위치추정. GT crosshair 는 cond.crosshair_xy(없으면 검출 폴백).

    frame 의 crosshair 제거는 cond 가 있으면 clean_image(cond 구동), 없으면 검출+_inpaint.
    combined_dir 이 주어지면 [rcp(승자 modality) | msr(crosshair 제거+GT+예측)] 결합 패널을
    combined/<recipe>/<msr>.jpg 로 저장한다(추출→매칭→찍기 한눈 추적).
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
        "modality": None, "modality_skip": None,
        "cells": {}, "overlay": None,
        "frame_hw": [int(gray_raw.shape[0]), int(gray_raw.shape[1])],   # Tier 1.1 displacement bin 용.
    }
    if label != "S" or crosshair_xy is None:
        return row

    # msr modality 로 라우팅 — om·sem race 금지. 미상/dual 모호 → skip(틀린-modality 측정 차단).
    available_mods = {m for m, v in center_tpls.items() if v is not None}
    routed = _route_modality(cond, available_mods)
    row["modality"] = routed
    if routed is None:
        row["modality_skip"] = "missing_modality"
        return row

    if cond and cond.crosshair_xy is not None:
        gray_inp = clean_image(gray_raw, cond)        # cond 구동 crosshair 제거(튜닝된 기본값).
    else:
        gray_inp = _inpaint_crosshair(gray_raw, crosshair_xy)
    frames = {"raw": gray_raw, "inpaint": gray_inp}
    tpl_sets = {"center": center_tpls, "box": box_tpls}

    for tname in TEMPLATES:
        routed_tpl = tpl_sets[tname].get(routed)
        if routed_tpl is None:        # 이 modality 의 box template 부재(center 는 항상 있음).
            continue
        one = {routed: routed_tpl}    # 해결된 modality 1개만 매칭(race 제거).
        for fname in FRAMES:
            res = gle._localize(one, frames[fname], crosshair_xy)
            if res is not None:
                row["cells"][f"{tname}__{fname}"] = res

    if SAVE_OVERLAYS and overlay_dir is not None and row["cells"]:
        row["overlay"] = gle._save_overlay(
            gray_inp, crosshair_xy, row["cells"],
            recipe=recipe_id or "recipe", msr_name=msr_path.name, out_dir=overlay_dir)

    if SAVE_OVERLAYS and combined_dir is not None and row["cells"]:
        msr_canvas = gle._render_overlay_canvas(
            gray_inp, crosshair_xy, row["cells"],
            recipe=recipe_id or "recipe", msr_name=msr_path.name)
        mod = _winning_mod(row["cells"])
        rcp_canvas = rcp_canvases.get(mod) if rcp_canvases else None
        panel = _combine_2up(
            rcp_canvas, msr_canvas,
            rcp_label=f"RCP {(mod or '?').upper()} (yellow=cond box, green=template)",
            msr_label=f"MSR {msr_path.name} (green=GT, orange=box pred)")
        sub = combined_dir / (recipe_id or "recipe")
        sub.mkdir(parents=True, exist_ok=True)
        cpath = sub / f"{msr_path.stem}_combined.jpg"
        cv2.imwrite(str(cpath), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        row["combined"] = str(cpath)
    return row


# --- measure-first 결정 게이트 ---------------------------------------------------
# 2026-06-02 실데이터(흰box/crosshair 제거 *전*) 기준선: proposer recall(gt_in_topk)
# 천장 0.594, rerank(MI/contour) 둘 다 음수 lift 로 폐기됨. 이유: 진실이 후보(top-N)에
# 없으면 재정렬은 무력. cond 정제(inpaint+decoupled offset) *후* 어느 레버가 살아있는지
# 이 게이트가 box__inpaint 셀로 직접 판정한다([[project_matcher_flat_chamfer_distinctiveness]]).
OLD_PROPOSER_CEILING = 0.594   # 정제 전 gt_in_topk 천장(이걸 넘었나 = 정제가 membership 을 올렸나).
PROPOSER_WALL = 0.62           # gt_in_topk 이 이 이하면 진실이 후보에 자주 없음 → proposer 가 벽.
RERANKER_MIN_HEADROOM = 0.08   # topk_not_rank1 이 이 이상이면 재정렬로 건질 여지 있음.


def lever_verdict(cell_stats, *, proposer_ceiling=OLD_PROPOSER_CEILING):
    """box__inpaint 셀 통계 → 다음에 어떤 ensemble 레버를 당길지 판정.

    반환 {verdict, gt_in_topk, topk_not_rank1, rank1, improved_vs_old, recommendation}.
    verdict: no_data | proposer_wall | reranker_alive | near_ceiling.
    - proposer_wall  : gt_in_topk ≤ PROPOSER_WALL → 진실이 후보에 없음. ensemble *proposer*
                       (후보 합집합) + consensus 재등록. 재정렬은 여전히 무력.
    - reranker_alive : gt_in_topk 높고 topk_not_rank1 ≥ 여유 → 진실이 후보에 있는데 rank1 이
                       아님 → ensemble *reranker* 가 비로소 유효.
    - near_ceiling   : gt_in_topk 높지만 격차 작음 → 남은 미스는 proposer 몫, reranker 여지 작음.
    """
    n = cell_stats.get("n", 0)
    if not n:
        return {"verdict": "no_data",
                "recommendation": "box__inpaint 표본 없음 — cond box 있는 recipe 가 필요."}
    gt = cell_stats.get("gt_in_topk_rate", 0.0)
    gap = cell_stats.get("topk_not_rank1_rate", 0.0)
    r1 = cell_stats.get("rank1_hit_rate", 0.0)
    improved = gt > proposer_ceiling
    if gt <= PROPOSER_WALL:
        verdict = "proposer_wall"
        rec = ("진실이 후보에 자주 없음 → ensemble PROPOSER(Chamfer+NCC/region+edge-orient 후보 "
               "합집합) + consensus 재등록 우선. 재정렬(rerank)은 아직 무력.")
    elif gap >= RERANKER_MIN_HEADROOM:
        verdict = "reranker_alive"
        rec = ("진실이 후보에 있고 rank1 격차 있음 → ensemble RERANKER(정제된 입력에서 2차 "
               "점수 융합) 가 비로소 유효. 후보 pool 키운 뒤 재정렬.")
    else:
        verdict = "near_ceiling"
        rec = ("후보 안에선 거의 rank1 → 남은 미스는 proposer membership(gt<1.0) 몫. "
               "reranker 여지 작음, proposer/재등록에 집중.")
    return {"verdict": verdict, "gt_in_topk": gt, "topk_not_rank1": gap, "rank1": r1,
            "improved_vs_old": improved, "recommendation": rec}


def run() -> str:
    """cond GT 로 golden 위치추정 검증 (인자 없음). 반환: success | no_data | no_rows."""
    _apply_matcher_default()   # Tier 1.1: ensemble 기본(ALIGN_USE_ENSEMBLE=0 으로 끌 수 있음).
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
    # 결합본(rcp | msr)만 저장 — 분리 파일(rcp_templates/·overlays/)은 끔.
    combined_dir = (out_dir / "combined") if SAVE_OVERLAYS else None
    # _localize 가 쓰는 매처 모드(env ALIGN_USE_ENSEMBLE) — baseline/ensemble 실행 구분용.
    matcher_mode = gle._matcher_for_eval().__name__
    print(f"[INFO] (cond GT) recipe {len(recipes)}개 처리 → {out_dir}  [matcher={matcher_mode}]")
    if matcher_mode == "compute_align_key_score":
        print("[INFO] baseline 매처. ensemble 개선을 보려면 ALIGN_USE_ENSEMBLE=1 로 재실행.")

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
            rcp_canvases = {}
            try:
                center_tpls, box_tpls = _build_offset_templates_cond(
                    assets, box_reasons=box_reasons, offset_records=offset_records,
                    rcp_canvases=rcp_canvases)
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
                    recipe_id=assets.recipe_id,
                    rcp_canvases=rcp_canvases, combined_dir=combined_dir)
                if row is None:
                    continue
                row["recipe"] = assets.recipe_id
                all_rows.append(row)
                rf.write(json.dumps(row, ensure_ascii=False) + "\n")

    if n_skip_no_msr:
        print(f"\n[INFO] from_msr 비어 건너뛴 recipe: {n_skip_no_msr}개.")
    if box_reasons:
        total = sum(box_reasons.values())
        ok = sum(n for k, n in box_reasons.items() if k.startswith("ok:cond"))
        print(f"\n[INFO] === rcp box (cond) === 사용 {ok}/{total} (rate={ok / total:.3f})  "
              f"(skip/absent 는 box template 제외)")
        for reason, n in sorted(box_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<18} {n:>4}")

    if not all_rows:
        print("[ERROR] 처리된 msr 행이 없습니다.")
        return "no_rows"

    # GT 출처 집계(cond vs 검출 폴백).
    s_rows = [r for r in all_rows if r["label"] == "S"]
    n_cond = sum(1 for r in s_rows if r.get("crosshair_source") == "cond")
    print(f"\n[INFO] === GT 출처 (S {len(s_rows)}장) === cond={n_cond}  "
          f"detect-폴백={len(s_rows) - n_cond}")

    # modality 라우팅 집계(om/sem/skip) — race 제거 후 어디로 라우팅됐나.
    mod_counts = Counter(r.get("modality") or "skip" for r in s_rows)
    n_mskip = sum(1 for r in s_rows if r.get("modality_skip"))
    print(f"[INFO] === modality 라우팅 (S {len(s_rows)}장) === "
          f"om={mod_counts.get('om', 0)}  sem={mod_counts.get('sem', 0)}  "
          f"skip(missing_modality)={n_mskip}")

    summary = gle._summarize(all_rows)
    summary["matcher"] = matcher_mode   # baseline/ensemble 실행 구분(ALIGN_USE_ENSEMBLE).
    summary["align_offset_diag"] = _offset_diag_cond(offset_records)   # 대각선 척도 자체 진단.
    summary["gt_source"] = {"n_S": len(s_rows), "n_cond": n_cond,
                            "n_detect_fallback": len(s_rows) - n_cond}
    summary["modality_routing"] = {"om": mod_counts.get("om", 0),
                                   "sem": mod_counts.get("sem", 0),
                                   "skip": n_mskip}
    lv = lever_verdict(summary["cells"].get("box__inpaint", {"n": 0}))
    summary["lever_verdict"] = lv
    summary["binned"] = _binned_localization_report(all_rows)   # Tier 1.1 bin×arm 게이트.
    # summary.json 은 *출력 전에* 먼저 쓴다 — 프린트 단계에서 죽어도 산출물은 남게.
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    gle._print_summary(summary)

    diag = summary["align_offset_diag"]
    print("\n[INFO] === align-offset 진단 (cond box, 대각선 정규화) ===")
    if diag.get("n", 0) == 0:
        print("  cond box 있는 modality 없음 — 진단 불가.")
    else:
        print(f"  offset_norm: median={diag['median_offset_norm']} "
              f"p90={diag['p90_offset_norm']} (n={diag['n']}) · "
              f"가정민감(>{diag['tol']})={diag['n_assumption_sensitive']}개")

    # === 다음 레버 판정 (이 한 줄만 보고하면 다음 계획 결정 가능) ===
    print("\n" + "=" * 64)
    print("[INFO] === 다음 레버 판정 (box__inpaint 셀; 이 블록만 읽어주면 됨) ===")
    if lv["verdict"] == "no_data":
        print(f"  판정: NO_DATA — {lv['recommendation']}")
    else:
        improved = "↑정제효과 있음" if lv["improved_vs_old"] else "≈정제 전과 비슷"
        print(f"  gt_in_topk={lv['gt_in_topk']}  rank1={lv['rank1']}  "
              f"topk!=1={lv['topk_not_rank1']}  (옛 천장 {OLD_PROPOSER_CEILING}, {improved})")
        print(f"  >>> 판정: {lv['verdict'].upper()}")
        print(f"  >>> 다음: {lv['recommendation']}")
    print("=" * 64)
    _print_binned_report(summary["binned"])   # Tier 1.1: bin×arm 게이트 표(office digest).

    print(f"\n[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
