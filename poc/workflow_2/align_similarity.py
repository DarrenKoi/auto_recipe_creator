"""rcp align 영역(정중앙) ↔ msr(S/E) 유사도 진단 — calibration / 지표 선택용.

목적:
  - rcp 의 align point 는 *정중앙* 이다(흰 box 는 보조 단서). 이 중앙 영역이 msr 에서
    얼마나 유사하게 나타나는지를 **S/E 라벨별로** 재서, 유사도가 "key 있음/없음" 을
    가르는 신호인지 + 임계값을 정한다. S/E 라벨을 ground truth 로 쓴다.
  - E 의 두 유형(사용자 설명)을 우리 유사도로 재현되는지 검증:
      * E-type1: crosshair 가 *틀린 위치* → 장비가 거기서 낮은 점수 → fail.
                 ⇒ "crosshair 위치 유사도" 가 낮아야 한다.
      * E-type2: crosshair 를 아예 못 잡음 → no crosshair.

세 위치에서 유사도를 잰다 (rcp 중앙 template 기준):
  1. at_crosshair : msr 에서 검출된 crosshair 주변 ROI (v2 검출). S 높음 / E-type1 낮음.
  2. at_center    : msr 정중앙 주변 ROI. 성공 정렬이면 align point 가 중앙에 옴.
  3. free_best    : msr 전체 free 검색 best. "key 가 어디든 있나?" E-type1 에서 높으면 이동 복구 가능.

지표: matcher score(Chamfer+ORB, STRUCTURE_POLICY) 주지표 + MI + NCC(지표 변별력 비교).
비교용으로 rcp 흰 box template 의 free_best 도 같이 계산해 "center vs box 중 무엇이 더 잘
가르나" 를 답한다.

좌표 권한 원칙(VLM 아니라 CV)과 무관한 *순수 진단* — 실데이터(오피스)에서 한 번 돌려
S/E 분리 통계를 텍스트로 회신받아 임계/지표를 정한다([[feedback_no_office_data_to_mac]]).

실행:
    uv run python poc/workflow_2/align_similarity.py   # 실데이터면 분석, 없으면 self-test
출력: stdout 요약 + DEBUG_IMAGE_DIR/align_similarity/<ts>/{rows.jsonl, summary.json}
"""

import json
import statistics
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_key_matcher import (
    DT_TAU_PX,
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
)

# rcp 중앙 align 영역 crop 의 면적 비율 (각 변 = sqrt). align_point_correction 의 fallback 과 동일 계열.
CENTER_AREA_RATIO = 0.15
# 정적 비교 scale band (rcp/msr 거의 같은 배율 가정) — align_point_correction.COMPARE_SCALES 와 동일.
COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)
# at_center / at_crosshair ROI 한 변 = template 변 × 이 배수 (검색 여유).
ROI_FACTOR = 1.8
# 한 recipe 당 처리할 msr 상한 (None=전부). 빠른 시험용.
LIMIT_PER_RECIPE = None

# --- 참조 staleness: *상대* 기준 (절대 점수의 confounder 회피) ---
# rcp-center 가 S-consensus 에 맞는 정도(rcp_vs)를, S 들이 그 consensus 에 맞는 정도(s_internal)와
# 비교한다. ratio = rcp_vs / median(s_internal). S 끼리 잘 뭉치는데 rcp 만 동떨어지면 ratio 가 낮다
# → rcp outlier = stale. 같은 metric 으로 재므로 matcher 약함이 상쇄된다.
# consensus 를 신뢰하려면 recipe 당 S 가 충분(>=MIN_S)하고, S 끼리도 일관(CV<=임계)해야 한다.
# 모두 cold-start — golden 데이터셋(align_success_dataset_plan.md)으로 실측 calibration 예정.
RELATIVE_STALE_RATIO = 0.6      # rcp_vs 가 S-internal median 의 60% 미만이면 stale 후보.
MIN_S_FOR_CONSENSUS = 3         # consensus 산정 최소 S 장수. 미만이면 "판단 불가".
S_INCONSISTENT_CV = 0.5         # S-internal 의 변동계수(std/mean) 가 이보다 크면 S 끼리도 안 뭉침 → CV 판단 불가.

# --- truth-forced sweep (병목 분리): 정답(S-crosshair)에서 wide scale band 로 chamfer 강제 측정 ---
# COMPARE 상한(1.0)을 넘는 1.2/1.4 포함 — best scale 이 >1.0 에 몰리면 C4 scale-band 문제.
SWEEP_SCALES = (0.5, 0.6, 0.7, 0.75, 0.85, 1.0, 1.2, 1.4)
COMPARE_BEST_SCALES = (0.6, 0.75, 0.85, 1.0)   # 생산 경로 band — wide 와 비교용.
# truth_err 이 이 이하라야 "truth-locked"(정답에 실제로 lock). 넘으면 wrong_local_peak.
TRUTH_ERR_NORM_MAX = 0.20      # template 짧은 변 대비 비율.

# --- gt-in-topK (proposer recall): 정답이 chamfer top-N 후보에 들어오나 ---
# 평평한 점수면에서 free_best(=rank1) 만 보면 "truth 가 후보엔 있는데 순위만 밀린 것"인지
# "후보에 아예 없는 것"인지 구분 못 한다. 전자면 MI 리랭킹으로 고칠 수 있고(reranker),
# 후자면 후보 생성기 자체를 바꿔야 한다(proposer). 그 갈림길을 재는 계측.
TOPK_CANDIDATES = 8            # MatchPolicy.top_n 과 동일 계열.
GT_TOL_NORM = TRUTH_ERR_NORM_MAX   # 후보가 정답으로 인정되는 거리(template 짧은 변 대비).


# ====================================================================
# 유사도 지표.
# ====================================================================


def _mi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """두 동일 크기 gray crop 의 mutual information (밝기/대비 drift 에 강건)."""
    a = a.ravel()
    b = b.ravel()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / max(hist.sum(), 1.0)
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    nz = pab > 0
    pa_pb = pa[:, None] * pb[None, :]
    return float((pab[nz] * np.log(pab[nz] / pa_pb[nz])).sum())


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """두 동일 크기 crop 의 정규화 상관 (pixel 동일성 baseline — 보통 drift 에 약함)."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _matched_crop(frame: np.ndarray, center_xy, tw: int, th: int, scale: float) -> np.ndarray | None:
    """best 위치/스케일에서 msr crop 을 떼어 template 크기로 리사이즈 (MI/NCC 입력용)."""
    cw = max(1, int(round(tw * scale)))
    ch = max(1, int(round(th * scale)))
    cx, cy = center_xy
    x0 = max(0, int(cx - cw // 2))
    y0 = max(0, int(cy - ch // 2))
    x1 = min(frame.shape[1], x0 + cw)
    y1 = min(frame.shape[0], y0 + ch)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)


def _window_roi(frame_shape, center_xy, tw: int, th: int, factor: float = ROI_FACTOR):
    """center_xy 주변, template 의 factor 배 크기 ROI (x, y, w, h). 매칭을 국소화한다."""
    h, w = frame_shape[:2]
    rw = min(w, int(tw * factor))
    rh = min(h, int(th * factor))
    cx, cy = center_xy
    x0 = int(max(0, min(w - rw, cx - rw // 2)))
    y0 = int(max(0, min(h - rh, cy - rh // 2)))
    return (x0, y0, rw, rh)


def _score(template, frame, *, roi=None, scales=COMPARE_SCALES):
    """compute_align_key_score 래퍼 — (score, chamfer, orb, best_xy, best_scale)."""
    r = compute_align_key_score(
        template, frame, roi_hint=roi, scales=scales, policy=STRUCTURE_POLICY,
    )
    return r.score, r.chamfer_score, r.orb_inlier_ratio, r.best_xy, r.best_scale


def _race(templates: dict, frame, *, roi=None, scales=COMPARE_SCALES):
    """OM/SEM template 중 점수 높은 쪽 채택. 반환 (modality, score, chamfer, orb, best_xy, best_scale)."""
    best = None
    for mod, tpl in templates.items():
        if tpl is None:
            continue
        try:
            s, ch, orb, xy, sc = _score(tpl, frame, roi=roi, scales=scales)
        except Exception as exc:
            print(f"[WARNING] score 실패 ({mod}): {exc}")
            continue
        if best is None or s > best[1]:
            best = (mod, s, ch, orb, xy, sc)
    return best  # None 가능.


def _edge_density(gray: np.ndarray) -> float:
    """Canny(60,160) edge 픽셀 비율 — matcher 전처리와 동일 임계."""
    if gray is None or gray.size == 0:
        return 0.0
    e = cv2.Canny(gray, 60, 160)
    return float((e > 0).mean())


def _diagnose_truth(
    *, truth_valid, wide_chamfer, wide_scale, scale_gain, tpl_ed, msr_ed,
) -> str:
    """병목 1줄 진단 (Codex cold-start 규칙)."""
    if not truth_valid:
        return "wrong_local_peak"          # truth ROI 안에서 best 가 정답을 벗어남.
    if tpl_ed < 0.01:
        return "template_weak"             # rcp key 자체 edge 빈약 — key 품질 문제.
    if wide_scale > 1.0 and scale_gain >= 0.05:
        return "scale_band_problem"        # >1.0 에서만 회복 → 생산 band(≤1.0) 가 놓침 (C4).
    ratio = (msr_ed / tpl_ed) if tpl_ed > 0 else 0.0
    if msr_ed < 0.5 * tpl_ed and msr_ed < 0.02 and wide_chamfer < 0.50:
        return "edge_problem_msr"          # msr crop 에 edge 가 거의 없음.
    if 0.5 <= ratio <= 2.0 and wide_chamfer < 0.50 and scale_gain < 0.03:
        return "metric_or_reference_problem"  # edge 는 있는데 chamfer 낮음 → metric/Canny 또는 reference drift.
    return "ok"


def _truth_forced(gray, crosshair_xy, center_tpls, xhair_crop):
    """정답(crosshair) 위치에서 wide scale band 로 chamfer 강제 측정 → 병목 분리 dict.

    per-modality 로 truth ROI 안에서 scale 별 chamfer 를 재고, wide-best vs compare-best 를
    비교한다. truth_err 이 크면(국소 wrong peak) truth_valid=False 로 분리한다.
    """
    cxh, cyh = crosshair_xy
    best_overall = None  # (mod, chamfer, scale, xy, orb, per_scale, tw, th)
    for mod, tpl in center_tpls.items():
        if tpl is None:
            continue
        th, tw = tpl.raw_image.shape[:2]
        max_s = max(SWEEP_SCALES)
        slack = max(12, int(0.15 * min(tw, th)))
        # ROI 는 sweep 최대 scale 의 template 을 수용하도록 (compare 의 ROI_FACTOR 대신 max_s 기준).
        rw = min(gray.shape[1], int(round(tw * max_s)) + 2 * slack)
        rh = min(gray.shape[0], int(round(th * max_s)) + 2 * slack)
        H, W = gray.shape[:2]
        rx = int(max(0, min(W - rw, cxh - rw // 2)))
        ry = int(max(0, min(H - rh, cyh - rh // 2)))
        roi = (rx, ry, rw, rh)
        per_scale = {}
        mod_best = None  # (chamfer, scale, xy, orb)
        for s in SWEEP_SCALES:
            try:
                _sc, ch, orb, xy, _bs = _score(tpl, gray, roi=roi, scales=(s,))
            except Exception:
                continue
            per_scale[s] = round(float(ch), 4)
            if mod_best is None or ch > mod_best[0]:
                mod_best = (float(ch), float(s), xy, float(orb))
        if mod_best is None:
            continue
        if best_overall is None or mod_best[0] > best_overall[1]:
            best_overall = (mod, mod_best[0], mod_best[1], mod_best[2], mod_best[3], per_scale, tw, th)

    if best_overall is None:
        return None
    mod, wide_ch, wide_scale, wide_xy, wide_orb, per_scale, tw, th = best_overall
    compare_ch = max((per_scale.get(s, 0.0) for s in COMPARE_BEST_SCALES), default=0.0)
    scale_gain = wide_ch - compare_ch
    err = float(np.hypot(wide_xy[0] - cxh, wide_xy[1] - cyh))
    err_norm = err / max(1, min(tw, th))
    truth_valid = err_norm <= TRUTH_ERR_NORM_MAX
    tpl_ed = _edge_density(center_tpls[mod].raw_image)
    msr_ed = _edge_density(xhair_crop) if xhair_crop is not None else None
    mean_dt_px = float(-DT_TAU_PX * np.log(max(wide_ch, 1e-6)))
    diagnosis = _diagnose_truth(
        truth_valid=truth_valid, wide_chamfer=wide_ch, wide_scale=wide_scale,
        scale_gain=scale_gain, tpl_ed=tpl_ed, msr_ed=(msr_ed if msr_ed is not None else 0.0),
    )
    return {
        "modality": mod,
        "valid": truth_valid,
        "err_px": round(err, 1),
        "err_norm": round(err_norm, 3),
        "wide_chamfer": round(wide_ch, 4),
        "wide_scale": wide_scale,
        "wide_orb": round(wide_orb, 4),
        "compare_chamfer": round(compare_ch, 4),
        "scale_gain": round(scale_gain, 4),
        "mean_dt_px": round(mean_dt_px, 2),
        "tpl_edge_density": round(tpl_ed, 4),
        "msr_edge_density": round(msr_ed, 4) if msr_ed is not None else None,
        "per_scale_chamfer": per_scale,
        "diagnosis": diagnosis,
    }


def _gt_in_topk(gray, crosshair_xy, center_tpls, *, topk=TOPK_CANDIDATES, scales=COMPARE_SCALES):
    """정답(crosshair) 위치가 free-search chamfer top-N 후보 안에 들어오는지 측정.

    proposer(후보 생성)가 truth 를 surface 하는지 → 리랭킹(MI 등)으로 고칠 수 있는지 판정.
      - in_topk=True, rank>1  → truth 가 후보엔 있는데 chamfer 순위만 밀림 ⇒ reranker 로 회복 가능.
      - in_topk=False         → truth 가 후보에 아예 없음 ⇒ proposer 교체 필요(리랭킹 무의미).
    반환 dict: {topk_rank(None=miss), in_topk, n_cand, best_cand_dist_norm} 또는 None.
    """
    from poc.workflow_2.align_key_matcher import (
        compute_chamfer_candidates,
        preprocess_for_matching,
    )
    _edges, frame_dt = preprocess_for_matching(gray)
    cxh, cyh = crosshair_xy
    combined = []  # (chamfer, dist_norm)
    for tpl in center_tpls.values():
        if tpl is None:
            continue
        th, tw = tpl.raw_image.shape[:2]
        short = max(1, min(tw, th))
        try:
            cands = compute_chamfer_candidates(tpl, frame_dt, scales=scales, top_n=topk)
        except Exception:
            continue
        for c in cands:
            d = float(np.hypot(c.xy[0] - cxh, c.xy[1] - cyh)) / short
            combined.append((float(c.chamfer_score), d))
    if not combined:
        return None
    combined.sort(key=lambda t: t[0], reverse=True)   # chamfer 내림차순.
    combined = combined[:topk]                          # modality 교차 global top-N.
    rank = next((i for i, (_s, d) in enumerate(combined, 1) if d <= GT_TOL_NORM), None)
    best_dist = min(d for _s, d in combined)
    return {
        "topk_rank": rank,
        "in_topk": rank is not None,
        "n_cand": len(combined),
        "best_cand_dist_norm": round(best_dist, 3),
    }


# ====================================================================
# 템플릿 빌드 (rcp 중앙 + 흰 box).
# ====================================================================


def _build_templates(assets):
    """recipe rcp 에서 center / box template 을 modality 별로 만든다.

    반환: (center_templates: dict[mod->tpl], box_templates: dict[mod->tpl|None])
    center 는 정중앙 면적 crop(= align 영역), box 는 흰 unique-area box 안쪽 crop.
    """
    from poc.workflow_2.align_fail_assets import load_gray
    from poc.workflow_2.align_point_correction import (
        _centered_area_crop_bbox,
        _detect_white_box,
        _inner_crop_for_box,
    )

    center: dict = {}
    box: dict = {}
    sources = [("om", assets.recipe_om, "rcp_om", "om"),
               ("sem", assets.recipe_sem, "rcp_sem", "sem")]
    for mod, path, version, key_type in sources:
        if path is None:
            continue
        gray = load_gray(path)
        cx, cy, cw, ch = _centered_area_crop_bbox(gray, CENTER_AREA_RATIO)
        center_crop = gray[cy:cy + ch, cx:cx + cw].copy()
        center[mod] = build_template(center_crop, recipe_id=assets.recipe_id,
                                     version=version + "_center", key_type=key_type)
        det = _detect_white_box(gray)
        if det is not None:
            inner, _bbox = _inner_crop_for_box(gray, det)
            box[mod] = build_template(inner, recipe_id=assets.recipe_id,
                                      version=version + "_box", key_type=key_type)
        else:
            box[mod] = None
    return center, box


# ====================================================================
# 한 msr 행.
# ====================================================================


def _process_msr(msr_path, *, center_tpls, box_tpls):
    from poc.workflow_2.align_fail_assets import load_gray
    from poc.workflow_2.align_point_correction import _tool_label
    from poc.workflow_2.crosshair_detect import detect_crosshair

    gray = load_gray(msr_path)
    h, w = gray.shape[:2]
    label = _tool_label(msr_path.name)
    center_xy = (w // 2, h // 2)

    # 대표 template 크기 (center, 첫 modality) — ROI 산정용.
    any_center = next((t for t in center_tpls.values() if t is not None), None)
    if any_center is None:
        return None, None, None
    th, tw = any_center.raw_image.shape[:2]

    # 1) free best (center template race).
    free = _race(center_tpls, gray)
    free_best = free[1] if free else None
    free_xy = free[4] if free else None
    free_scale = free[5] if free else 1.0
    free_mod = free[0] if free else None
    free_dist_center = float(np.hypot(free_xy[0] - center_xy[0], free_xy[1] - center_xy[1])) if free_xy else None

    # 2) at_center (center template, ROI = 중앙 window).
    center_roi = _window_roi(gray.shape, center_xy, tw, th)
    at_center = _race(center_tpls, gray, roi=center_roi)
    at_center_score = at_center[1] if at_center else None

    # 3) at_crosshair (v2 검출 → ROI = crosshair window). S 에서 이 위치가 ground-truth align 영역.
    #    rcp-center vs 여기 = 참조 staleness 측정 → score(기하)+MI(정보)+NCC(pixel) 모두 기록.
    ch_res = detect_crosshair(gray)
    at_crosshair_score = mi_xhair = ncc_xhair = None
    xhair_crop = None   # S-at-crosshair crop (consensus/staleness 용) — recipe 단위로 모은다.
    xhair_mod = None
    if ch_res.xy is not None:
        ch_roi = _window_roi(gray.shape, ch_res.xy, tw, th)
        at_ch = _race(center_tpls, gray, roi=ch_roi)
        if at_ch:
            at_crosshair_score = at_ch[1]
            ch_mod, _s, _c, _o, ch_xy, ch_scale = at_ch
            tpl = center_tpls[ch_mod]
            xcrop = _matched_crop(gray, ch_xy, tpl.raw_image.shape[1], tpl.raw_image.shape[0], ch_scale)
            if xcrop is not None:
                mi_xhair = _mi(tpl.raw_image, xcrop)
                ncc_xhair = _ncc(tpl.raw_image, xcrop)
                xhair_crop = xcrop
                xhair_mod = ch_mod

    # MI / NCC — center template (winner modality) vs free-best crop.
    mi_free = ncc_free = None
    if free_mod is not None and free_xy is not None:
        tpl = center_tpls[free_mod]
        crop = _matched_crop(gray, free_xy, tpl.raw_image.shape[1], tpl.raw_image.shape[0], free_scale)
        if crop is not None:
            mi_free = _mi(tpl.raw_image, crop)
            ncc_free = _ncc(tpl.raw_image, crop)

    # 비교: box template free best.
    box_free = _race(box_tpls, gray) if any(v is not None for v in box_tpls.values()) else None
    free_best_box = box_free[1] if box_free else None

    # 4) truth-forced sweep — S + crosshair 일 때만 (정답 위치 알려짐). 병목(edge/scale/metric) 분리.
    #    + gt-in-topK: 정답이 chamfer top-N 후보에 들어오나 (proposer recall = reranker 가능 여부).
    truth = None
    gt_topk = None
    if label == "S" and ch_res.xy is not None:
        truth = _truth_forced(gray, ch_res.xy, center_tpls, xhair_crop)
        gt_topk = _gt_in_topk(gray, ch_res.xy, center_tpls)

    return {
        "msr": msr_path.name,
        "label": label,
        "modality_used": free_mod,
        "free_best": free_best,
        "free_best_xy": list(free_xy) if free_xy else None,
        "free_best_dist_to_center": free_dist_center,
        "at_center": at_center_score,
        "crosshair_xy": list(ch_res.xy) if ch_res.xy else None,
        "crosshair_conf": ch_res.confidence,
        "crosshair_reason": ch_res.debug.get("reason"),
        "at_crosshair": at_crosshair_score,
        "mi_free": mi_free,
        "ncc_free": ncc_free,
        "mi_xhair": mi_xhair,
        "ncc_xhair": ncc_xhair,
        "free_best_box": free_best_box,
        "truth": truth,
        "gt_topk": gt_topk,
    }, xhair_crop, xhair_mod


# ====================================================================
# 분리도(separability) 통계.
# ====================================================================


def _separation(s_vals, e_vals) -> dict:
    """S(높아야)/E(낮아야) 두 분포의 분리도 — median + best threshold(balanced acc)."""
    s = [v for v in s_vals if isinstance(v, (int, float))]
    e = [v for v in e_vals if isinstance(v, (int, float))]
    out = {"n_s": len(s), "n_e": len(e)}
    if not s or not e:
        out["note"] = "insufficient samples"
        out["median_s"] = statistics.median(s) if s else None
        out["median_e"] = statistics.median(e) if e else None
        return out
    out["median_s"] = round(statistics.median(s), 4)
    out["median_e"] = round(statistics.median(e), 4)
    best_thr, best_bacc = None, -1.0
    for t in sorted(set(s + e)):
        tpr = sum(1 for v in s if v >= t) / len(s)     # S 를 맞춘 비율.
        tnr = sum(1 for v in e if v < t) / len(e)      # E 를 맞춘 비율.
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, t
    out["best_threshold"] = round(best_thr, 4)
    out["balanced_accuracy"] = round(best_bacc, 3)
    return out


def _summarize(rows: list[dict]) -> dict:
    by_label = {"S": [r for r in rows if r["label"] == "S"],
                "E": [r for r in rows if r["label"] == "E"]}
    metrics = ["free_best", "at_center", "at_crosshair", "mi_free", "ncc_free", "free_best_box"]
    sep = {}
    for m in metrics:
        sep[m] = _separation([r[m] for r in by_label["S"]], [r[m] for r in by_label["E"]])

    # E-type 추정: crosshair 있고 at_crosshair 낮은데 free_best 높음 = 이동 복구 가능(type1).
    e_with_ch = [r for r in by_label["E"] if r["crosshair_xy"] is not None and r["at_crosshair"] is not None]
    e_no_ch = [r for r in by_label["E"] if r["crosshair_xy"] is None]
    recoverable = [r for r in e_with_ch
                   if r["free_best"] is not None and r["at_crosshair"] is not None
                   and r["free_best"] - r["at_crosshair"] > 0.1]

    # truth-forced sweep 집계 — 병목(edge/scale/metric) 분리.
    truths = [r["truth"] for r in rows if r.get("truth") is not None]
    truth_summary = None
    if truths:
        valid = [t for t in truths if t["valid"]]
        def _med(key, src):
            vals = [t[key] for t in src if isinstance(t.get(key), (int, float))]
            return round(statistics.median(vals), 4) if vals else None
        scale_hist: dict[str, int] = {}
        for t in valid:
            scale_hist[str(t["wide_scale"])] = scale_hist.get(str(t["wide_scale"]), 0) + 1
        diag_counts: dict[str, int] = {}
        for t in truths:
            diag_counts[t["diagnosis"]] = diag_counts.get(t["diagnosis"], 0) + 1
        truth_summary = {
            "n_truth": len(truths),
            "n_valid": len(valid),
            "n_wrong_local_peak": len(truths) - len(valid),
            "median_wide_chamfer": _med("wide_chamfer", valid),
            "median_compare_chamfer": _med("compare_chamfer", valid),
            "median_scale_gain": _med("scale_gain", valid),
            "median_wide_orb": _med("wide_orb", valid),
            "median_mean_dt_px": _med("mean_dt_px", valid),
            "median_tpl_edge_density": _med("tpl_edge_density", valid),
            "median_msr_edge_density": _med("msr_edge_density", valid),
            "wide_best_scale_hist": scale_hist,
            "diagnosis_counts": diag_counts,
        }

    # gt-in-topK 집계 — proposer recall (truth 가 후보에 드는 비율 + 순위 분포).
    gts = [r["gt_topk"] for r in rows if r.get("gt_topk") is not None]
    topk_summary = None
    if gts:
        in_topk = [g for g in gts if g["in_topk"]]
        rank1 = [g for g in in_topk if g["topk_rank"] == 1]
        rank_hist: dict[str, int] = {}
        for g in in_topk:
            rank_hist[str(g["topk_rank"])] = rank_hist.get(str(g["topk_rank"]), 0) + 1
        miss_dists = [g["best_cand_dist_norm"] for g in gts if not g["in_topk"]]
        topk_summary = {
            "n": len(gts),
            "n_in_topk": len(in_topk),
            "n_rank1": len(rank1),
            "n_miss": len(gts) - len(in_topk),
            "in_topk_rate": round(len(in_topk) / len(gts), 3),
            "rank1_rate": round(len(rank1) / len(gts), 3),
            "rank_hist": rank_hist,
            "median_miss_dist_norm": round(statistics.median(miss_dists), 3) if miss_dists else None,
            "topk": TOPK_CANDIDATES,
        }

    # 참조 staleness(상대 기준)는 crop 이 필요해 analyze 에서 _reference_quality 로 채운다.
    return {
        "counts": {"S": len(by_label["S"]), "E": len(by_label["E"]),
                   "other": len(rows) - len(by_label["S"]) - len(by_label["E"]), "total": len(rows)},
        "separation": sep,
        "E_breakdown": {
            "with_crosshair": len(e_with_ch),
            "no_crosshair": len(e_no_ch),
            "recoverable_by_move(free>>at_crosshair)": len(recoverable),
        },
        "truth_forced": truth_summary,
        "gt_topk": topk_summary,
    }


def _consensus(crops: list) -> np.ndarray:
    """동일 크기 gray crop 들의 median 이미지 = '현재 align 영역의 대표 모습'."""
    stack = np.stack([c.astype(np.float32) for c in crops])
    return np.median(stack, axis=0).astype(np.uint8)


def _reference_quality(crops_by_recipe: dict) -> list:
    """recipe 별 *상대* staleness — rcp-center 가 S-consensus 에 맞는 정도를 S 들끼리의 일관성과 비교.

    같은 metric(MI)으로 재므로 matcher 약함이 상쇄된다.
      - S 끼리 잘 뭉치는데(낮은 CV) rcp 만 동떨어짐(낮은 ratio) → status=stale_replace.
      - S 끼리도 안 뭉침(높은 CV) → consensus 불신 → status=S_inconsistent (CV 로 판단 불가).
      - S 장수 부족 → insufficient_S.
    반환: relative_ratio 오름차순(worst-first) 리스트.
    """
    out = []
    for rec, data in crops_by_recipe.items():
        s_crops = data.get("s_crops", {})
        if not s_crops:
            continue
        mod = max(s_crops, key=lambda m: len(s_crops[m]))   # S 가 가장 많은 modality.
        crops = s_crops[mod]
        R = data.get("rcp", {}).get(mod)
        entry = {"recipe": rec, "modality": mod, "n_S": len(crops)}
        if R is None:
            entry["status"] = "no_rcp"
            out.append(entry)
            continue
        if len(crops) < MIN_S_FOR_CONSENSUS:
            entry["status"] = "insufficient_S"
            out.append(entry)
            continue
        consensus = _consensus(crops)
        s_internal = [_mi(c, consensus) for c in crops]
        s_med = statistics.median(s_internal)
        s_mean = statistics.mean(s_internal)
        s_cv = (statistics.pstdev(s_internal) / s_mean) if s_mean > 0 else 0.0
        rcp_vs = _mi(R, consensus)
        ratio = (rcp_vs / s_med) if s_med > 0 else 0.0
        entry.update({
            "s_internal_median_mi": round(s_med, 4),
            "s_internal_cv": round(s_cv, 3),
            "rcp_vs_consensus_mi": round(rcp_vs, 4),
            "relative_ratio": round(ratio, 3),
        })
        if s_cv > S_INCONSISTENT_CV:
            entry["status"] = "S_inconsistent"      # consensus 불신 → CV 단독 판단 불가.
        elif ratio < RELATIVE_STALE_RATIO:
            entry["status"] = "stale_replace"        # rcp 가 S cluster 의 outlier → 재등록 권장.
        else:
            entry["status"] = "ok"
        out.append(entry)
    out.sort(key=lambda d: d.get("relative_ratio", 9.99))
    return out


def _print_summary(summary: dict) -> None:
    c = summary["counts"]
    print(f"\n[INFO] images: S={c['S']} E={c['E']} other={c['other']} total={c['total']}")
    print("\n[INFO] S/E 분리도 (median_s 높고 median_e 낮을수록, balanced_accuracy 1.0 에 가까울수록 좋은 지표):")
    print(f"  {'metric':<14} {'med_S':>8} {'med_E':>8} {'thr':>8} {'bACC':>6}  n(S/E)")
    for m, s in summary["separation"].items():
        print(f"  {m:<14} {str(s.get('median_s')):>8} {str(s.get('median_e')):>8} "
              f"{str(s.get('best_threshold','-')):>8} {str(s.get('balanced_accuracy','-')):>6}  "
              f"{s.get('n_s')}/{s.get('n_e')}")
    eb = summary["E_breakdown"]
    print(f"\n[INFO] E 유형: with_crosshair={eb['with_crosshair']}  no_crosshair={eb['no_crosshair']}  "
          f"recoverable_by_move={eb['recoverable_by_move(free>>at_crosshair)']}")
    print("  * at_crosshair 의 med_E 가 med_S 보다 확실히 낮으면 → 우리 유사도가 장비 fail(낮은 점수)을 재현 = 지표 자격 검증")
    print("  * free_best 는 S/E 둘 다 높을 수 있음(key 가 E 에도 존재) → at_crosshair 가 진짜 변별자")

    # truth-forced sweep — 정답 위치에서 wide scale 로 chamfer 강제 측정 → 병목 분리.
    tf = summary.get("truth_forced")
    if tf:
        print(f"\n[INFO] TRUTH-FORCED (S+crosshair 정답 위치, wide scale {SWEEP_SCALES}):")
        print(f"  valid/truth = {tf['n_valid']}/{tf['n_truth']}  (wrong_local_peak={tf['n_wrong_local_peak']})")
        print(f"  median chamfer: wide={tf['median_wide_chamfer']}  compare(≤1.0)={tf['median_compare_chamfer']}  "
              f"scale_gain={tf['median_scale_gain']}  orb={tf['median_wide_orb']}  mean_dt_px={tf['median_mean_dt_px']}")
        print(f"  edge density median: tpl={tf['median_tpl_edge_density']}  msr={tf['median_msr_edge_density']}")
        print(f"  wide_best_scale hist: {tf['wide_best_scale_hist']}")
        print(f"  진단 counts: {tf['diagnosis_counts']}")
        print("  * 해석: median wide_chamfer 가 낮음(<0.5) → edge/metric. best_scale 이 1.2/1.4 에 몰림 + scale_gain↑ → C4 scale-band.")
        print("           scale_gain≈0 인데 chamfer 낮고 orb 도 낮음 → reference drift(=재등록). orb 만 높으면 Canny/metric 문제.")

    # gt-in-topK — 정답이 chamfer top-N 후보에 드는 비율 → 리랭킹(MI) vs proposer 교체 갈림길.
    gk = summary.get("gt_topk")
    if gk:
        print(f"\n[INFO] GT-IN-TOPK (S+crosshair, top-{gk['topk']} chamfer 후보의 proposer recall):")
        print(f"  in_topk={gk['n_in_topk']}/{gk['n']} ({gk['in_topk_rate']})  "
              f"rank1={gk['n_rank1']} ({gk['rank1_rate']})  miss={gk['n_miss']}  "
              f"median_miss_dist_norm={gk['median_miss_dist_norm']}")
        print(f"  rank_hist(정답이 든 순위): {gk['rank_hist']}")
        print("  * in_topk 높고 rank1 낮음 → truth 가 후보엔 있는데 순위만 밀림 ⇒ MI 리랭커로 회복 가능.")
        print("  * in_topk 낮음 → truth 가 후보에 아예 없음 ⇒ proposer(후보 생성기) 교체 필요(리랭킹 무의미).")

    # 참조 staleness(상대) — rcp 가 S-consensus 의 outlier 인지. status 로 판단 가능/불가 구분.
    rq = summary.get("reference_quality", [])
    if rq:
        n_stale = sum(1 for d in rq if d.get("status") == "stale_replace")
        n_incon = sum(1 for d in rq if d.get("status") in ("S_inconsistent", "insufficient_S"))
        n_ok = sum(1 for d in rq if d.get("status") == "ok")
        print(f"\n[INFO] 참조 staleness(상대): rcp_vs_consensus / S내부일관성. ratio<{RELATIVE_STALE_RATIO} & S일관(CV<={S_INCONSISTENT_CV}) → 재등록 권장.")
        print(f"[INFO] stale={n_stale}  ok={n_ok}  판단불가(S부족/불일치)={n_incon}  / scored={len(rq)} recipes")
        print(f"  {'recipe':<38} {'nS':>3} {'ratio':>6} {'S_cv':>5} {'rcp_MI':>7} {'S_MI':>6}  status")
        for d in rq[:25]:
            print(f"  {d['recipe'][:38]:<38} {d.get('n_S','-'):>3} "
                  f"{str(d.get('relative_ratio','-')):>6} {str(d.get('s_internal_cv','-')):>5} "
                  f"{str(d.get('rcp_vs_consensus_mi','-')):>7} {str(d.get('s_internal_median_mi','-')):>6}  "
                  f"{d.get('status')}")
        print("  * stale_replace = S끼리 뭉치는데 rcp만 동떨어짐(재등록 권장). S_inconsistent = S끼리도 안 뭉침 → CV 단독 판단 불가(golden 필요).")


# ====================================================================
# 엔트리.
# ====================================================================


def analyze(*, limit_per_recipe=LIMIT_PER_RECIPE) -> str:
    from poc.workflow_2.align_fail_assets import (
        iter_msr_images,
        iter_recipe_dirs,
        resolve_assets,
    )

    leaves = iter_recipe_dirs()
    if not leaves:
        print("[ERROR] align_images recipe 없음.")
        return "no_assets"

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DEBUG_IMAGE_DIR / "align_similarity" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    crops_by_recipe: dict = {}   # {recipe: {"s_crops": {mod: [crop]}, "rcp": {mod: R_image}}}
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fp:
        for i, leaf in enumerate(leaves, 1):
            assets = resolve_assets(leaf)
            if assets.recipe_om is None and assets.recipe_sem is None:
                continue
            try:
                center_tpls, box_tpls = _build_templates(assets)
            except Exception as exc:
                print(f"[WARNING] template 빌드 실패 {leaf}: {exc}")
                continue
            tag = f"{assets.eqp_id}/{assets.class_name}/{assets.recipe_id}"
            crops_by_recipe[tag] = {
                "s_crops": {},
                "rcp": {m: t.raw_image for m, t in center_tpls.items() if t is not None},
            }
            msr_images = iter_msr_images(assets)
            if limit_per_recipe:
                msr_images = msr_images[:limit_per_recipe]
            print(f"[{i}/{len(leaves)}] {tag}  msr={len(msr_images)}")
            for msr_path in msr_images:
                try:
                    row, xhair_crop, xhair_mod = _process_msr(
                        msr_path, center_tpls=center_tpls, box_tpls=box_tpls)
                except Exception as exc:
                    print(f"[WARNING] {msr_path.name}: {type(exc).__name__}: {exc}")
                    continue
                if row is None:
                    continue
                row["recipe"] = tag
                rows.append(row)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                # S-at-crosshair crop 누적 — recipe 단위 consensus/staleness 용.
                if row["label"] == "S" and xhair_crop is not None and xhair_mod is not None:
                    crops_by_recipe[tag]["s_crops"].setdefault(xhair_mod, []).append(xhair_crop)

    summary = _summarize(rows)
    summary["reference_quality"] = _reference_quality(crops_by_recipe)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\n[INFO] 저장: {out_dir}/summary.json , rows.jsonl")
    return "success"


# ====================================================================
# 합성 self-test.
# ====================================================================


def _patterned(seed: int, size=(60, 60)) -> np.ndarray:
    """엣지가 뚜렷한 작은 패턴 (Chamfer 가 변별 가능하도록)."""
    rs = np.random.RandomState(seed)
    img = np.full(size, 40, dtype=np.uint8)
    for _ in range(6):
        p1 = (rs.randint(0, size[1]), rs.randint(0, size[0]))
        p2 = (rs.randint(0, size[1]), rs.randint(0, size[0]))
        cv2.line(img, p1, p2, 230, 2)
    cv2.rectangle(img, (10, 10), (size[1] - 10, size[0] - 10), 255, 2)
    return img


def _frame_with(pattern, *, at, canvas=(300, 400), bg=50) -> np.ndarray:
    rs = np.random.RandomState(7)
    img = np.full(canvas, bg, dtype=np.uint8)
    img = cv2.add(img, rs.randint(0, 20, canvas).astype(np.uint8))
    ph, pw = pattern.shape
    x, y = at
    img[y:y + ph, x:x + pw] = pattern
    return img


def _self_test() -> bool:
    """rcp 중앙 패턴이 S frame 중앙엔 있고 E frame 엔 없을 때, 유사도가 S>E 로 갈리는지."""
    from poc.workflow_2.align_key_matcher import build_template as _bt

    align = _patterned(1)
    other = _patterned(99)
    rcp_center_tpl = _bt(align, recipe_id="T", version="rcp_om_center", key_type="om")

    # S: 중앙에 align 패턴. E: 중앙에 other 패턴.
    cxy = (400 // 2 - 30, 300 // 2 - 30)
    s_frame = _frame_with(align, at=cxy)
    e_frame = _frame_with(other, at=cxy)

    s = _race({"om": rcp_center_tpl}, s_frame)
    e = _race({"om": rcp_center_tpl}, e_frame)
    assert s is not None and e is not None, "score 산출 실패"
    print(f"[INFO] self-test free_best: S={s[1]:.3f}  E={e[1]:.3f}")
    ok = s[1] > e[1]
    if not ok:
        print("[ERROR] S 가 E 보다 높지 않음 — 유사도 변별 실패(합성).")

    # separation 함수 sanity.
    sep = _separation([0.8, 0.75, 0.9], [0.3, 0.4, 0.2])
    assert sep["balanced_accuracy"] == 1.0, f"separation 오류: {sep}"
    assert sep["median_s"] > sep["median_e"]
    print(f"[INFO] self-test separation OK: {sep}")
    print("[INFO] self-test 통과." if ok else "[ERROR] self-test 실패.")
    return ok


def run() -> str:
    try:
        from poc.workflow_2.align_fail_assets import iter_recipe_dirs
        has_data = bool(iter_recipe_dirs())
    except Exception:
        has_data = False
    if has_data:
        return analyze()
    print("[WARNING] align_images 데이터 없음 — 합성 self-test 로 대체합니다.\n")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
