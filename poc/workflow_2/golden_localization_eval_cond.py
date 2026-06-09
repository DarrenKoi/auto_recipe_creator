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
from poc.workflow_3.vision.align_fail_assets import (
    iter_msr_images,
    load_gray,
    resolve_assets,
)
from poc.workflow_3.vision.align_key_matcher import build_template
from poc.workflow_3.vision.align_point_correction import (
    RCP_FALLBACK_CENTER_CROP_AREA_RATIO,
    _RcpTemplateBundle,
    _centered_area_crop_bbox,
    _draw_rcp_overlay,
    _inpaint_crosshair,
    _tool_label,
)
from poc.workflow_2.align_similarity import CENTER_AREA_RATIO
from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import CondInfo, load_cond, msr_modality
from poc.workflow_3.vision.crosshair_detect import detect_crosshair
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


def _cond_box_to_xywh(box_ltrb):
    """cond.box_ltrb(cursor frame, ×10) → 이미지 px (x, y, w, h)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t)))


# --- cond box → template/offset (검출 무관, cond.txt 만으로 결정) -----------------
# 원본 eval 은 box 의 *내용 검출* inner-crop 중심에서 offset 을 뽑아, crop 오류가
# offset(전체 eval 의 load-bearing 값)을 오염시켰다. cond.txt 가 정확한 corner 를
# 주므로 (1) offset = image_center − box_center 를 crop 과 분리해 기하로만 계산하고,
# (2) template 은 stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 해 만든다
# (crop 중심 == box 중심 → offset 과 일관). Codex 검토 반영([[project_align_cond_files_and_coords]]).
CROP_INSET_PX = 2       # inpaint 후 edge-smear 가 template 에 안 들어오게 하는 대칭 inset.
MIN_INNER_PX = 16       # 대칭 inset 후 box 내부 하한(미만이면 skip — 매칭 신호 불안정).
WARN_INNER_PX = 24      # 작은 box 경고 임계(skip 아님).
OFFSET_WARN = 0.25      # offset_norm(÷대각선) 경고 임계(box 가 중심에서 멂).
OFFSET_SKIP = 0.38      # offset_norm 하드 skip(=box≠center 가정 붕괴, 엔지니어 검토 필요).


def _cond_box_center(box_ltrb):
    """cond.box_ltrb → 이미지 px box 중심 (cx, cy) (정수 반올림 전 float)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (l + r) / 2.0, (t + b) / 2.0


def cond_align_offset(box_ltrb, shape_hw):
    """align point(이미지 중심) − box 중심. cond.txt 만으로 결정 → crop 과 분리(decoupled).

    crop 을 어떻게 잡든 align point 의 기하는 안 변한다. 이 분리가 원본의 (B) 결함
    — 내용검출 inner-crop 의 off-center 가 offset 을 오염시키던 경로 — 를 통째로 없앤다.
    """
    h, w = shape_hw[:2]
    bcx, bcy = _cond_box_center(box_ltrb)
    return (int(round(w / 2.0 - bcx)), int(round(h / 2.0 - bcy)))


def cond_offset_norm(box_ltrb, shape_hw):
    """|offset| 를 이미지 *대각선* 으로 정규화(crop 무관 척도)."""
    h, w = shape_hw[:2]
    dx, dy = cond_align_offset(box_ltrb, shape_hw)
    diag = float(np.hypot(w, h)) or 1.0
    return float(np.hypot(dx, dy) / diag)


def check_cond_box(box_ltrb, shape_hw):
    """cond box 가 template 으로 쓸만한지 가드. 반환 (status, reason, offset_norm).

    status: 'ok' | 'warn' | 'skip'. inner = min(box변) − 2·CROP_INSET_PX(대칭 inset 후).
    skip 우선순위: degenerate → out_of_bounds → too_small → offset_too_far.
    """
    h, w = shape_hw[:2]
    x, y, bw, bh = _cond_box_to_xywh(box_ltrb)
    onorm = cond_offset_norm(box_ltrb, shape_hw)
    if bw <= 0 or bh <= 0:
        return "skip", "box:degenerate", onorm
    if x < 0 or y < 0 or x + bw > w or y + bh > h:
        return "skip", "box:out_of_bounds", onorm
    inner = min(bw, bh) - 2 * CROP_INSET_PX
    if inner < MIN_INNER_PX:
        return "skip", "box:too_small", onorm
    if onorm > OFFSET_SKIP:
        return "skip", "offset:too_far", onorm
    if onorm > OFFSET_WARN:
        return "warn", "offset:far", onorm
    if inner < WARN_INNER_PX:
        return "warn", "box:small", onorm
    return "ok", "ok", onorm


def cond_template_crop(gray, cond, *, inset=CROP_INSET_PX):
    """cond box stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 해 template crop.

    대칭 inset → crop 중심 == box 중심 → cond_align_offset 과 정확히 일관.
    inset 후 너무 작아지면 inset 을 생략(작은 box 보호). 반환 (crop, (x0,y0,w,h)).

    **box stroke 만 지운다.** rcp cond 에 crosshair 가 있어도 그건 box 내부를 가로지르는
    *실제 내용* 이므로 inpaint 하면 매칭 신호가 깎인다 → crosshair_xy=None 으로 마스킹해
    box 테두리만 제거한다(msr 프레임의 crosshair 제거는 별개 — 거기선 distractor 라 지움).
    """
    box_only = CondInfo(scope=cond.scope, pixel=cond.pixel,
                        box_ltrb=cond.box_ltrb, crosshair_xy=None)
    cleaned = clean_image(gray, box_only)        # 튜닝된 1/1/2 로 box stroke 만 제거.
    x, y, bw, bh = _cond_box_to_xywh(cond.box_ltrb)
    h, w = gray.shape[:2]
    x0, y0 = max(0, x + inset), max(0, y + inset)
    x1, y1 = min(w, x + bw - inset), min(h, y + bh - inset)
    if x1 - x0 < MIN_INNER_PX or y1 - y0 < MIN_INNER_PX:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
    return cleaned[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


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
    print(f"\n[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
