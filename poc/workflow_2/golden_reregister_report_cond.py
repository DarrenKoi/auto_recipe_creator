"""re-registration 우선순위 랭킹 리포트 (Phase 1, S-only risk screening).

골든 셋을 스캔해 recipe·modality 별 재등록 필요도를 3-tier 증거로 산출·랭킹하고, flagged recipe 에
교체 whitebox 후보를 제안한다. 순수 로직(이 파일의 _ 헬퍼)은 I/O 와 분리해 합성 데이터로 단위 테스트한다.
정확도 숫자는 오피스 골든에서만(Mac 은 py_compile + no_data).

spec: poc/workflow_2/docs/specs/2026-06-23-reregister-report-design.md
"""
import os

from poc.workflow_2 import golden_eval_config_loader
golden_eval_config_loader.seed_env()  # gce import 전 env 브리지 (다른 드라이버와 동일).

import numpy as np

# ---- module consts (오피스 보정 대상) ----
MSR_FLOOR = 0.85               # peak_ratio tail 이 이 이상이면 MEDIUM.
SELF_FLOOR = 0.85              # self_ratio 가 이 이상이면 ADVISORY(OM 만).
EXCL_RADIUS_FOOTPRINTS = 1.0   # self-match 제외존 = 이 배수 × max(tw,th).
SUGG_SCALES = (0.8, 1.0, 1.25)
SUGG_STRIDE_RATIO = 0.25
SPLIT_MIN_S = 4
ACCEPT_MARGIN = float(os.getenv("REREGISTER_ACCEPT_MARGIN", "0.05"))
TIER_WEIGHT = {"STRONG": 2.0, "MEDIUM": 1.0, "ADVISORY": 0.3, "NONE": 0.0}

SURVIVORSHIP_BANNER = (
    "S-only latent-risk screening: candidates among historically-successful "
    "recipes, NOT a confirmed fail list. E-frame confirmation = Phase 2."
)


# ====================================================================
# 순수 헬퍼 — 증거 집계 (I/O 없음, 합성 데이터로 테스트).
# ====================================================================
def _aggregate_strong(frame_results):
    """STRONG: 무가드 free-search 가 진짜 점을 못 고른 S 프레임 비율 + worst 변위.

    frame_results: 프레임별 `_gt_in_topk` 반환 dict 리스트(None 프레임은 호출부에서 제외).
    fail = in_topk=False 또는 topk_rank>1.
    """
    n = len(frame_results)
    if n == 0:
        return {"strong_fail_frac": 0.0, "worst_disp": 0.0, "n_s": 0}
    fails = sum(
        1 for f in frame_results
        if (not f.get("in_topk")) or (f.get("topk_rank") or 1) > 1
    )
    worst = max((f.get("best_cand_dist_norm") or 0.0) for f in frame_results)
    return {"strong_fail_frac": fails / n, "worst_disp": float(worst), "n_s": n}


def _aggregate_medium(frame_results):
    """MEDIUM: peak_ratio(top2/top1) worst-case(max). None(후보<2)은 0(모호 아님)."""
    n = len(frame_results)
    tail = max((f.get("peak_ratio") or 0.0) for f in frame_results) if n else 0.0
    return {"msr_peak_tail": float(tail), "n_s": n}


def _self_ratio(cands, best_xy, excl_radius_px):
    """rcp self-match 의 변별도: 자기-peak 제외존 밖 최강 look-alike / 자기-peak.

    cands: score 내림차순 후보(.xy, .score). best_xy = 자기-peak 위치(=cands[0].xy).
    제외존 밖 생존 후보가 없으면 0.0(완전 변별). best score 0 가드.
    """
    if not cands:
        return 0.0
    best_score = float(cands[0].score) or 0.0
    if best_score <= 0:
        return 0.0
    bx, by = best_xy
    for c in cands[1:]:
        if float(np.hypot(c.xy[0] - bx, c.xy[1] - by)) > excl_radius_px:
            return float(c.score) / best_score
    return 0.0


# ====================================================================
# 순수 헬퍼 — tier / risk / 랭킹.
# ====================================================================
def _evidence_tier(modality, strong_fail_frac, msr_peak_tail, self_ratio):
    """가장 강한 증거 1개로 tier 결정(상관 축 이중계수 회피). raw 절대 floor 경계.

    SEM self-match 는 near-degenerate 라 단독 tier 를 만들지 않는다(ADVISORY 는 OM 만).
    반환 (tier, severity) — severity 는 tier 내 정렬 키(raw, 정규화 없음).
    """
    if strong_fail_frac > 0:
        return "STRONG", float(strong_fail_frac)
    if msr_peak_tail >= MSR_FLOOR:
        return "MEDIUM", float(msr_peak_tail)
    if modality == "om" and self_ratio >= SELF_FLOOR:
        return "ADVISORY", float(self_ratio)
    return "NONE", 0.0


def _risk_score(tier, severity):
    """tier 가중 + tier 내 raw severity. cohort 통계 없음(1-recipe/동값 안전)."""
    return TIER_WEIGHT[tier] + float(severity)


def _rank_rows(rows):
    """한 modality row 들을 risk_score desc 정렬, 동점은 worst_disp desc tiebreak."""
    return sorted(rows, key=lambda r: (r["risk_score"], r.get("worst_disp", 0.0)), reverse=True)


# ====================================================================
# 순수 헬퍼 — 포맷(ASCII only).
# ====================================================================
def _fmt_num(x):
    """숫자를 3자리 고정소수점으로, None 이면 '-'."""
    return "-" if x is None else f"{float(x):.3f}"


def _format_report(rows_by_mod):
    """modality 별 worst-first 테이블 텍스트(ASCII). 헤더에 survivorship 배너."""
    lines = ["=== Re-registration priority (S-only screening) ===", SURVIVORSHIP_BANNER, ""]
    cols = ("rank recipe tier strong_fail worst_disp msr_tail self_ratio(conf) "
            "n_s suggestion sugg_self/fid")
    for mod in ("om", "sem"):
        rows = rows_by_mod.get(mod, [])
        lines.append(f"-- {mod.upper()} ({len(rows)} screened) --")
        lines.append(cols)
        for i, r in enumerate(rows, 1):
            lines.append(" ".join([
                str(i), r["recipe"], r["tier"], _fmt_num(r["strong_fail_frac"]),
                _fmt_num(r["worst_disp"]), _fmt_num(r["msr_peak_tail"]),
                f"{_fmt_num(r['self_ratio'])}({r.get('advisory_confidence','ok')})",
                str(r["n_s"]), r.get("suggestion", "none"),
                f"{_fmt_num(r.get('sugg_self'))}/{_fmt_num(r.get('sugg_fidelity'))}",
            ]))
        lines.append("")
    return "\n".join(lines)


def _format_digest(rows_by_mod):
    """1줄 DIGEST(ASCII). modality 별 screened/strong/w_sugg + top recipe 2개."""
    parts = []
    for mod in ("om", "sem"):
        rows = rows_by_mod.get(mod, [])
        strong = sum(1 for r in rows if r["tier"] == "STRONG")
        w_sugg = sum(1 for r in rows if str(r.get("suggestion", "none")).startswith("box"))
        top = ",".join(r["recipe"] for r in rows[:2]) or "-"
        parts.append(f"{mod}[screened {len(rows)}, strong {strong}, w_sugg {w_sugg}, top {top}]")
    return "[DIGEST] reregister(S-only): " + " | ".join(parts)


# ====================================================================
# 순수 헬퍼 — 박스 제안 (C2).
# ====================================================================
def _mean(xs):
    """xs 의 산술평균. 빈 리스트면 0.0."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _split_frames(frame_keys, *, split_min_s=SPLIT_MIN_S):
    """held-out 분할. < split_min_s 면 None(insufficient). even-idx=select, odd-idx=validate."""
    if len(frame_keys) < split_min_s:
        return None
    select = [k for i, k in enumerate(frame_keys) if i % 2 == 0]
    validate = [k for i, k in enumerate(frame_keys) if i % 2 == 1]
    return select, validate


def _iter_candidate_boxes(img_w, img_h, base_box, *, scales=SUGG_SCALES, stride_ratio=SUGG_STRIDE_RATIO):
    """엔지니어 박스 크기 × scales 윈도를 stride 로 슬라이드. 이미지 경계 내 박스만."""
    bl, bt, br, bb = base_box
    bw, bh = br - bl, bb - bt
    short = max(1, min(bw, bh))
    stride = max(1, int(round(stride_ratio * short)))
    out = []
    for s in scales:
        w, h = max(1, int(round(bw * s))), max(1, int(round(bh * s)))
        if w >= img_w or h >= img_h:
            continue
        for t in range(0, img_h - h + 1, stride):
            for l in range(0, img_w - w + 1, stride):
                out.append((l, t, l + w, t + h))
    return out


def _select_candidate(cand_metrics, baseline):
    """select-half: mean fidelity >= baseline mean fidelity 인 후보 중 최저 self_ratio. 없으면 None."""
    base_fid = _mean(baseline["sel_fidelities"])
    passing = [c for c in cand_metrics if _mean(c["sel_fidelities"]) >= base_fid]
    if not passing:
        return None
    return min(passing, key=lambda c: c["self_ratio"])


def _accept_candidate(cand, baseline, *, accept_margin=ACCEPT_MARGIN):
    """validate-half: mean paired fidelity delta >= margin AND self_ratio 개선 >= margin."""
    fid_delta = _mean(cand["val_fidelities"]) - _mean(baseline["val_fidelities"])
    self_gain = baseline["self_ratio"] - cand["self_ratio"]
    return fid_delta >= accept_margin and self_gain >= accept_margin


def _box_overlap_ratio(box, region):
    """box 가 region(=removal mask 사각형)과 겹치는 비율 = 교집합/ box 면적."""
    l, t, r, b = box
    rl, rt, rr_, rb = region
    iw = max(0, min(r, rr_) - max(l, rl))
    ih = max(0, min(b, rb) - max(t, rt))
    area = max(1, (r - l) * (b - t))
    return (iw * ih) / area


def _dodge_guard(cand_overlap, base_overlap, val_delta, *, accept_margin=ACCEPT_MARGIN):
    """True=REJECT. 후보가 overlap 급감(현재 대비)으로만 이득(val_delta 가 margin 부근)이면 가짜 이득."""
    avoids = cand_overlap < base_overlap - 0.2
    marginal = val_delta < 2 * accept_margin
    return avoids and marginal


# ====================================================================
# C2 — _search_unique_box + fidelity + suggest + overlay.
# ====================================================================
def _passes_texture(patch, *, min_edge=0.02, min_lap=5.0):
    """blank/저텍스처 패치 skip. _edge_density/_lap_var 재사용.

    텍스처가 없는 패치(흰색 영역, 빈 배경)는 고유 구조가 없어 안정적 매칭이 불가하므로 건너뛴다.
    """
    from poc.workflow_2.align_similarity import _edge_density, _lap_var
    if patch.size == 0:
        return False
    return _edge_density(patch) >= min_edge and _lap_var(patch) >= min_lap


def _patch_self_ratio(img, box):
    """box 패치를 full 이미지에 매칭한 self_ratio(제외존 밖 look-alike 강도).

    저텍스처 패치면 1.0(최대 모호, 선택 안 됨). 반환 값 낮을수록 더 변별.
    """
    l, t, r, b = box
    patch = img[t:b, l:r]
    if not _passes_texture(patch):
        return 1.0   # 저텍스처 → 변별 무의미(최대 모호 취급, 선택 안 됨).
    tpl = build_template(patch.copy(), recipe_id="sugg", version="sugg", key_type="om")
    res = compute_align_key_score_ensemble(tpl, img, scales=_SELF_MATCH_SCALES, policy=STRUCTURE_POLICY)
    cands = list(res.candidates)
    if not cands:
        return 1.0
    excl = EXCL_RADIUS_FOOTPRINTS * max(r - l, b - t)
    return _self_ratio(cands, cands[0].xy, excl)


def _search_unique_box(img, base_box):
    """rcp 이미지 위 후보 박스 중 self_ratio 최저(가장 변별)를 반환.

    반환: {box, self_ratio} 또는 None(텍스처 있는 후보 없음).
    base_box 를 기준 크기로 슬라이드 윈도를 생성해 각 후보의 self_ratio 를 측정.
    """
    h, w = img.shape[:2]
    best = None
    for box in _iter_candidate_boxes(w, h, base_box):
        sr = _patch_self_ratio(img, box)
        if best is None or sr < best["self_ratio"]:
            best = {"box": box, "self_ratio": sr}
    return best


# GT_FIDELITY_TOL: 프레임 GT 위치 근방 hit tolerance(short-side 정규화 기준 0.2 = GT_TOL_NORM).
_FIDELITY_GT_TOL_NORM = 0.20


def _compute_fidelity_from_patch(patch, s_frames):
    """patch(ndarray gray)를 template 으로 각 S 프레임에 매칭한 fidelity 리스트 반환.

    s_frames: [(gray_clean, gt_xy)] 리스트.
    fidelity = GT crosshair 위치 hit tolerance 내 최고 점수 후보의 score. 없으면 0.0.
    """
    if patch.size == 0 or not s_frames:
        return []
    tpl = build_template(patch.copy(), recipe_id="sugg", version="sugg", key_type="om")
    ph, pw = patch.shape[:2]
    short = max(1, min(pw, ph))
    tol_px = _FIDELITY_GT_TOL_NORM * short
    fidelities = []
    for gray, gt_xy in s_frames:
        try:
            res = compute_align_key_score_ensemble(tpl, gray, scales=_SELF_MATCH_SCALES, policy=STRUCTURE_POLICY)
            cands = list(res.candidates)
        except Exception:
            fidelities.append(0.0)
            continue
        if not cands:
            fidelities.append(0.0)
            continue
        gx, gy = gt_xy
        # GT hit tolerance 내 후보 중 최고 score.
        near = [c for c in cands if float(np.hypot(c.xy[0] - gx, c.xy[1] - gy)) <= tol_px]
        if near:
            fidelities.append(float(max(c.score for c in near)))
        else:
            fidelities.append(0.0)
    return fidelities


def _suggest_for_row(row):
    """flagged row 에 C2 박스 제안을 채운다(row 인플레이스 업데이트, 반환 없음).

    row["suggestion"]: "box(l,t,r,b)" | "no distinctive sub-region" | "insufficient"
    row["sugg_self"]: 제안 박스 self_ratio (채택 시)
    row["sugg_fidelity"]: 제안 박스 validate-half mean fidelity (채택 시)
    REREGISTER_BOX_SUGGEST=0 이거나 tier=NONE 이면 호출 자체를 skip(caller 책임).
    """
    assets = row.get("_assets")
    modality = row.get("modality", "om")
    s_frames = row.get("_s_frames", [])

    # held-out split.
    split = _split_frames(s_frames)
    if split is None:
        row["suggestion"] = "insufficient"
        return
    sel_frames, val_frames = split

    # rcp 이미지 로드.
    try:
        from poc.workflow_3.align.assets import load_gray
        rcp_path = assets.recipe_om if modality == "om" else assets.recipe_sem
        if rcp_path is None:
            row["suggestion"] = "no distinctive sub-region"
            return
        rcp_gray = load_gray(rcp_path)
    except Exception as exc:
        print(f"[WARNING] rcp 로드 실패({row['recipe']}/{modality}): {exc}")
        row["suggestion"] = "no distinctive sub-region"
        return

    # base_box: 현재 엔지니어 whitebox(rcp native px). _box 템플릿의 raw_image 크기로 추정.
    box_tpl = (row.get("_box") or {}).get(modality)
    if box_tpl is None:
        row["suggestion"] = "no distinctive sub-region"
        return
    th, tw = box_tpl.raw_image.shape[:2]
    # rcp 이미지 중앙 기준 whitebox 추정(cond.txt box 좌표가 없는 경우 중앙 crop 으로 근사).
    rh, rw = rcp_gray.shape[:2]
    cx, cy = rw // 2, rh // 2
    base_box = (max(0, cx - tw // 2), max(0, cy - th // 2),
                min(rw, cx + tw // 2 + tw % 2), min(rh, cy + th // 2 + th % 2))

    # baseline: 현재 base_box patch 의 fidelity.
    bl, bt, br, bb = base_box
    base_patch = rcp_gray[bt:bb, bl:br]
    baseline_sel = _compute_fidelity_from_patch(base_patch, sel_frames)
    baseline_val = _compute_fidelity_from_patch(base_patch, val_frames)
    baseline_self = _patch_self_ratio(rcp_gray, base_box)

    # select-half: 후보 박스 검색 + 각 후보 fidelity 계산.
    h, w = rcp_gray.shape[:2]
    cand_metrics = []
    for box in _iter_candidate_boxes(w, h, base_box):
        sr = _patch_self_ratio(rcp_gray, box)
        l, t, r, b = box
        patch = rcp_gray[t:b, l:r]
        if not _passes_texture(patch):
            continue
        sel_fid = _compute_fidelity_from_patch(patch, sel_frames)
        cand_metrics.append({
            "box": box,
            "self_ratio": sr,
            "sel_fidelities": sel_fid,
        })

    if not cand_metrics:
        row["suggestion"] = "no distinctive sub-region"
        return

    baseline_metric = {"self_ratio": baseline_self, "sel_fidelities": baseline_sel}
    chosen = _select_candidate(cand_metrics, baseline_metric)
    if chosen is None:
        row["suggestion"] = "no distinctive sub-region"
        return

    # validate-half: 후보 vs baseline 재측정.
    cl, ct, cr, cb = chosen["box"]
    cand_patch = rcp_gray[ct:cb, cl:cr]
    chosen["val_fidelities"] = _compute_fidelity_from_patch(cand_patch, val_frames)
    baseline_val_metric = {"self_ratio": baseline_self, "val_fidelities": baseline_val}

    val_delta = _mean(chosen["val_fidelities"]) - _mean(baseline_val_metric["val_fidelities"])

    if not _accept_candidate(
        {"self_ratio": chosen["self_ratio"], "val_fidelities": chosen["val_fidelities"]},
        {"self_ratio": baseline_self, "val_fidelities": baseline_val},
    ):
        row["suggestion"] = "no distinctive sub-region"
        return

    # inpaint-dodge 가드: removal mask 는 crosshair 위치에서 유도.
    # GT crosshair 를 S 프레임에서 얻어 removal mask 사각형을 근사.
    cand_overlap, base_overlap = 0.0, 0.0
    if s_frames:
        _, gt_xy = s_frames[0]
        gx, gy = gt_xy
        # crosshair removal region 근사: crosshair 중심 ± short/4 사각형.
        short_s = max(1, min(tw, th))
        half = max(2, short_s // 4)
        removal_rect = (max(0, gx - half), max(0, gy - half), gx + half, gy + half)
        cand_overlap = _box_overlap_ratio(chosen["box"], removal_rect)
        base_overlap = _box_overlap_ratio(base_box, removal_rect)

    if _dodge_guard(cand_overlap, base_overlap, val_delta):
        print(f"[WARNING] dodge_guard reject({row['recipe']}/{modality}): overlap avoidance only")
        row["suggestion"] = "no distinctive sub-region"
        return

    # 채택.
    box = chosen["box"]
    row["suggestion"] = f"box{box}"
    row["sugg_self"] = chosen["self_ratio"]
    row["sugg_fidelity"] = _mean(chosen["val_fidelities"])


def _safe_recipe_label(recipe_str):
    """recipe 문자열을 파일명으로 쓸 수 있게 정제(/ 등 대체)."""
    return recipe_str.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _render_overlay(row):
    """현재 박스(자홍) + 제안 박스(초록)를 rcp 이미지에 그려 JPEG 로 저장.

    채택된 박스(row['suggestion'].startswith('box')) 인 경우에만 의미 있으나,
    현재 박스는 rcp 이미지에 항상 표시해 엔지니어가 비교할 수 있게 한다.
    반환: 저장 경로(Path) 또는 None(저장 실패/skip).
    """
    try:
        from poc.workflow_3.align.assets import load_gray

        assets = row.get("_assets")
        modality = row.get("modality", "om")
        rcp_path = assets.recipe_om if modality == "om" else assets.recipe_sem
        if rcp_path is None:
            return None
        rcp_gray = load_gray(rcp_path)
        overlay = cv2.cvtColor(rcp_gray, cv2.COLOR_GRAY2BGR)

        # 현재 whitebox: box template 크기로 중앙 추정.
        box_tpl = (row.get("_box") or {}).get(modality)
        if box_tpl is not None:
            th, tw = box_tpl.raw_image.shape[:2]
            rh, rw = overlay.shape[:2]
            cx, cy = rw // 2, rh // 2
            base_box = (max(0, cx - tw // 2), max(0, cy - th // 2),
                        min(rw, cx + tw // 2 + tw % 2), min(rh, cy + th // 2 + th % 2))
            bl, bt, br, bb = base_box
            cv2.rectangle(overlay, (bl, bt), (br, bb), (255, 0, 255), 2)   # 자홍(BGR).
            cv2.putText(overlay, "current", (bl, max(0, bt - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        # 제안 박스(초록).
        sugg = row.get("suggestion", "none")
        if sugg.startswith("box"):
            try:
                # "box(l,t,r,b)" -> 튜플 파싱.
                inner = sugg[3:].strip("()")
                sl, st, sr, sb = [int(x.strip()) for x in inner.split(",")]
                cv2.rectangle(overlay, (sl, st), (sr, sb), (0, 255, 0), 2)   # 초록(BGR).
                cv2.putText(overlay, "suggested", (sl, max(0, st - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception as exc:
                print(f"[WARNING] overlay suggestion 파싱 실패: {exc}")

        safe = _safe_recipe_label(row["recipe"])
        out_path = OUTPUT_ROOT / f"{safe}_{modality}_reregister.jpg"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return out_path
    except Exception as exc:
        print(f"[WARNING] overlay 렌더 실패({row.get('recipe','?')}): {exc}")
        return None


# ====================================================================
# C1 통합 — 프레임 로드 + 매칭 패스 + run().
# ====================================================================
import cv2
from pathlib import Path

from poc.workflow_2.align_similarity import _build_templates, _gt_in_topk
from poc.workflow_3.align.matching.engine import (
    build_template,
    compute_align_key_score_ensemble,
    STRUCTURE_POLICY,
    DEFAULT_SCALES,
)
from poc.workflow_3.align.clean_align_image import (
    build_removal_mask,
    clean_image,
    cursor_to_image,
    OVERSAMPLE,
)
from poc.workflow_3.align.cond_file import load_cond, msr_modality
from poc.workflow_3.align.diagnostics.align_point_correction import _tool_label
from poc.workflow_3.align.diagnostics.crosshair_detect import detect_crosshair
from poc.workflow_3.align.assets import iter_msr_images, iter_recipe_dirs, resolve_assets
from poc.workflow_2 import DEBUG_IMAGE_DIR

# 자기-match 에 쓸 scale band — DEFAULT_SCALES (paused/static frame 기준).
_SELF_MATCH_SCALES = DEFAULT_SCALES

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_reregister_report_cond"


def _self_match_ratio(box_tpl, full_gray):
    """rcp box(whitebox) 템플릿을 *full* rcp 이미지에 매칭 → exclusion-zone self_ratio + degenerate 판정.

    spec 5 ADVISORY: 등록 key 이미지 안에서 whitebox 가 다른 영역과 닮았는지(look-alike). 반환
    (self_ratio, confidence). 템플릿이 full 이미지를 거의 채우면 confidence='low'(SEM near-degenerate).
    """
    th, tw = box_tpl.raw_image.shape[:2]
    res = compute_align_key_score_ensemble(box_tpl, full_gray, scales=_SELF_MATCH_SCALES, policy=STRUCTURE_POLICY)
    cands = list(res.candidates)
    if not cands:
        return 0.0, "low"
    excl = EXCL_RADIUS_FOOTPRINTS * max(tw, th)
    ratio = _self_ratio(cands, cands[0].xy, excl)
    fh, fw = full_gray.shape[:2]
    conf = "low" if (tw >= fw * 0.9 and th >= fh * 0.9) else "ok"
    return ratio, conf


def _load_s_frames(assets, modality):
    """recipe 의 from_msr S 프레임을 주어진 modality 로 필터 후 (gray_clean, gt_xy) 리스트로 반환.

    _process_msr_cond 의 프레임 modality 배정 + clean_image 호출 + cond.txt crosshair
    px@5120 -> 프레임 px 환산 패턴을 그대로 따른다.
    GT 없는 프레임(E, crosshair 없음, modality 불일치)은 건너뛴다.
    """
    from poc.workflow_3.align.assets import load_gray

    available_mods = {modality}   # 지정 modality 만 허용(race 금지).
    result = []
    for msr_path in iter_msr_images(assets):
        try:
            gray_raw = load_gray(msr_path)
        except Exception as exc:
            print(f"[WARNING] msr 로드 실패 {msr_path.name}: {exc}")
            continue

        label = _tool_label(msr_path.name)
        if label != "S":
            continue   # E 프레임 제외.

        cond = load_cond(msr_path)

        # crosshair GT: cond 가 있으면 cond.txt cursor 좌표 → 프레임 px 변환, 없으면 검출 폴백.
        if cond and cond.crosshair_xy is not None:
            gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
            crosshair_xy = (int(round(gx)), int(round(gy)))
        else:
            ch = detect_crosshair(gray_raw)
            crosshair_xy = ch.xy

        if crosshair_xy is None:
            continue   # GT 없으면 재료로 쓸 수 없음.

        # msr modality 배정(race 금지) — 요청 modality 와 다르면 skip.
        routed = _route_modality_for_mod(cond, available_mods, modality)
        if routed is None:
            continue

        # crosshair 제거 후 clean gray 산출.
        if cond and cond.crosshair_xy is not None:
            gray_clean = clean_image(gray_raw, cond)
        else:
            # cond 없음 — 검출된 crosshair 위치로 inpaint(golden_localization_eval_cond 폴백과 동일).
            try:
                from poc.workflow_3.align.diagnostics.align_point_correction import _inpaint_crosshair
                gray_clean = _inpaint_crosshair(gray_raw, crosshair_xy)
            except Exception:
                gray_clean = gray_raw   # inpaint 불가시 raw 사용(부정확하나 크래시 방지).

        result.append((gray_clean, crosshair_xy))
    return result


def _route_modality_for_mod(cond, available_mods, target_mod):
    """msr frame 의 cond 에서 추론한 modality 가 target_mod 와 일치하면 target_mod, 아니면 None.

    _route_modality(golden_localization_eval_cond) 와 동일 로직 — dual-recipe/미상 + dual 모호는 skip.
    단, 여기서는 target_mod 하나만 허용(race 금지).
    """
    inferred = msr_modality(cond)
    if inferred is not None:
        return inferred if inferred == target_mod else None
    # modality 미상: available_mods 가 단일(=target_mod 하나)이면 그걸로 폴백.
    if len(available_mods) == 1:
        return target_mod
    # 미상 + dual → skip.
    return None


def _walk_recipes(root):
    """ALIGN_GOLDEN_ROOT 아래의 recipe 목록을 AlignFailAssets 리스트로 반환.

    golden_localization_eval._collect_recipes 패턴: root 가 유효한 디렉터리면
    iter_recipe_dirs 로 leaf 목록을 순회하고 resolve_assets 로 assets 를 만든다.
    빈 루트이거나 디렉터리가 아니면 빈 리스트.
    """
    if not root.is_dir():
        return []
    dirs = iter_recipe_dirs(root)
    if not dirs:
        return []
    assets_list = []
    for d in dirs:
        try:
            assets_list.append(resolve_assets(d))
        except Exception as exc:
            print(f"[WARNING] recipe 해석 실패 {d}: {exc}")
    return assets_list


def _recipe_row(assets, modality):
    """한 recipe·modality 의 C1 증거 row. S 프레임/템플릿 없으면 None.

    S 프레임 로딩(clean + 프레임 GT)·modality 배정은 _process_msr_cond 패턴을 따른다.
    각 S 프레임: rcp center 템플릿으로 _gt_in_topk → STRONG/MEDIUM 재료.
    self_ratio: box 템플릿 self-match(ADVISORY).
    """
    try:
        center_tpls, box_tpls = _build_templates(assets)
    except Exception as exc:
        print(f"[WARNING] 템플릿 빌드 실패 {assets.recipe_id}: {exc}")
        return None

    if center_tpls.get(modality) is None:
        return None

    # S 프레임 (gray_clean, gt_xy) 리스트 — _process_msr_cond 패턴으로 로딩.
    s_frames = _load_s_frames(assets, modality)
    if not s_frames:
        return None

    frame_results = []
    for gray, gt_xy in s_frames:
        r = _gt_in_topk(gray, gt_xy, {modality: center_tpls[modality]})
        if r is not None:
            frame_results.append(r)

    if not frame_results:
        return None

    strong = _aggregate_strong(frame_results)
    medium = _aggregate_medium(frame_results)
    self_ratio_val, conf = 0.0, "ok"
    if box_tpls.get(modality) is not None:
        rcp_path = assets.recipe_om if modality == "om" else assets.recipe_sem
        if rcp_path is not None:
            try:
                from poc.workflow_3.align.assets import load_gray
                full_gray = load_gray(rcp_path)
                self_ratio_val, conf = _self_match_ratio(box_tpls[modality], full_gray)
            except Exception as exc:
                print(f"[WARNING] self_match 실패 {assets.recipe_id}/{modality}: {exc}")

    tier, sev = _evidence_tier(modality, strong["strong_fail_frac"], medium["msr_peak_tail"], self_ratio_val)
    return {
        "recipe": f"{assets.class_name}/{assets.recipe_name}",
        "modality": modality, "tier": tier, "risk_score": _risk_score(tier, sev),
        "strong_fail_frac": strong["strong_fail_frac"], "worst_disp": strong["worst_disp"],
        "msr_peak_tail": medium["msr_peak_tail"], "self_ratio": self_ratio_val,
        "advisory_confidence": conf, "n_s": strong["n_s"],
        "suggestion": "none", "sugg_self": None, "sugg_fidelity": None,
        "_assets": assets, "_center": center_tpls, "_box": box_tpls, "_s_frames": s_frames,
    }


def run():
    """골든 루트 walk → recipe·modality 별 row → 랭킹 → 리포트/DIGEST 파일. 반환 = DIGEST(또는 no_data 경고)."""
    root = Path(os.getenv("ALIGN_GOLDEN_ROOT", "")).expanduser()
    recipes = _walk_recipes(root)
    if not recipes:
        print("[WARNING] no_data: ALIGN_GOLDEN_ROOT empty or unset")
        return "[WARNING] no_data"

    rows_by_mod = {"om": [], "sem": []}
    for assets in recipes:
        for mod in ("om", "sem"):
            row = _recipe_row(assets, mod)
            if row is not None:
                rows_by_mod[mod].append(row)

    # 랭킹 전 C2: flagged row 에 박스 제안을 채운다.
    box_suggest_on = os.getenv("REREGISTER_BOX_SUGGEST", "1") != "0"
    topn = int(os.getenv("REREGISTER_TOPN", "0"))
    if box_suggest_on:
        for mod in rows_by_mod:
            # 랭킹 전 임시 정렬(flagged 먼저 처리; topn 이 0 이면 전체).
            flagged = [r for r in rows_by_mod[mod] if r["tier"] != "NONE"]
            flagged_sorted = sorted(flagged, key=lambda r: r["risk_score"], reverse=True)
            if topn > 0:
                flagged_sorted = flagged_sorted[:topn]
            for row in flagged_sorted:
                try:
                    _suggest_for_row(row)
                except Exception as exc:
                    print(f"[WARNING] C2 제안 실패({row['recipe']}/{mod}): {exc}")
                    row["suggestion"] = "no distinctive sub-region"

    for mod in rows_by_mod:
        rows_by_mod[mod] = _rank_rows(rows_by_mod[mod])

    # overlay 저장(채택 행 한정, topn 존중).
    if box_suggest_on:
        for mod in rows_by_mod:
            flagged_ranked = [r for r in rows_by_mod[mod] if r["tier"] != "NONE"]
            if topn > 0:
                flagged_ranked = flagged_ranked[:topn]
            for row in flagged_ranked:
                if str(row.get("suggestion", "none")).startswith("box"):
                    _render_overlay(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reregister_report.txt").write_text(_format_report(rows_by_mod), encoding="utf-8")
    digest = _format_digest(rows_by_mod)
    (OUTPUT_ROOT / "digest.txt").write_text(digest, encoding="utf-8")
    print(digest)
    return digest


if __name__ == "__main__":
    run()
