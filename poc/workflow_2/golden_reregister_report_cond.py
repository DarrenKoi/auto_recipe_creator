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
# STRONG 멤버십 floor: 오피스 1차 실측(2026-06-23)에서 frac>0 게이트가 SEM 95%/OM 52% 로 포화 →
# tier 변별 상실. GT-absent(in_topk=False, 재랭킹 불가)가 *다수* 프레임에서 나야 STRONG.
STRONG_FRAC_FLOOR = 0.5        # strong_fail_frac 이 이 이상이어야 STRONG (그 미만은 하위 tier).
EXCL_RADIUS_FOOTPRINTS = 1.0   # self-match 제외존 = 이 배수 × max(tw,th).
SUGG_SCALES = (0.8, 1.0, 1.25)
SUGG_STRIDE_RATIO = 0.25
# n_s(modality당 S 장수)는 오피스 실측상 2~3 (측정 1건 = OM 2 / SEM 3). held-out split 가 돌려면 >=2 라야
# 하고 4 면 전 recipe insufficient → 제안 0건. validate-half 가 1장이면 advisory(낮은 신뢰)로 표기.
SPLIT_MIN_S = 2
ACCEPT_MARGIN = float(os.getenv("REREGISTER_ACCEPT_MARGIN", "0.05"))
TIER_WEIGHT = {"STRONG": 2.0, "MEDIUM": 1.0, "ADVISORY": 0.3, "NONE": 0.0}

# fidelity 매칭 scale band. 진단상 작은 box crop 은 주기 SEM 텍스처에서 *최소 scale(0.6)* 로
# 줄여 wrong-phase distractor 에 high-score 매칭되어(예: top1 0.9 @ 300px off) 참 위치를 놓쳤다.
# rcp/msr 는 거의 같은 배율이므로 참 매칭은 ~1.0 근방이다. 0.6/0.75 escape hatch 를 빼고 1.0
# 근방 tight band 만 허용해 distractor 매칭을 줄인다. A/B: REREGISTER_FIDELITY_SCALES 로
# 임의 band(예: "0.6,0.75,0.85,1.0" 로 기존 COMPARE_SCALES 복원) 지정 가능.
_FIDELITY_SCALES_DEFAULT = (0.85, 1.0, 1.15)


def _resolve_fidelity_scales(env_val):
    """REREGISTER_FIDELITY_SCALES 환경값(쉼표구분 float)을 band 튜플로. 비거나 malformed 면 기본 tight band."""
    if not env_val:
        return _FIDELITY_SCALES_DEFAULT
    out = []
    for tok in env_val.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return tuple(out) if out else _FIDELITY_SCALES_DEFAULT


_FIDELITY_SCALES = _resolve_fidelity_scales(os.getenv("REREGISTER_FIDELITY_SCALES"))

# fast A/B 용 recipe cap. box-suggestion sweep 가 무거워(>10분) 전체 대신 앞 N개만 돌려
# tight-band A/B 방향만 빠르게 본다. REREGISTER_MAX_RECIPES=20 권장, 0=전체(기본).
def _cap_recipes(recipes, cap):
    """cap 이 양수면 앞 cap 개로 자른다. 0/음수면 전체."""
    return recipes[:cap] if cap and cap > 0 else recipes

SURVIVORSHIP_BANNER = (
    "S-only latent-risk screening: candidates among historically-successful "
    "recipes, NOT a confirmed fail list. E-frame confirmation = Phase 2."
)


# ====================================================================
# 순수 헬퍼 — 증거 집계 (I/O 없음, 합성 데이터로 테스트).
# ====================================================================
def _aggregate_strong(frame_results):
    """STRONG: proposer 가 GT 를 후보에 아예 못 올린(in_topk=False) S 프레임 비율 + worst 변위.

    frame_results: 프레임별 `_gt_in_topk` 반환 dict 리스트(None 프레임은 호출부에서 제외).
    fail = **in_topk=False 만**. topk_rank>1(후보엔 있고 순위만 밀림)은 production 리랭커가 복구하므로
    재등록 신호가 아니다 — 오피스 1차 실측에서 rank>1 포함이 STRONG 을 포화시켜(SEM 95%) 제외.
    [[project_matcher_flat_chamfer_distinctiveness]]: in_topk=False = proposer 벽 = 비변별(재랭킹 무의미).
    worst_disp: FAIL 프레임에서만 측정(pass 프레임 변위는 tiebreak 왜곡 방지). 없으면 0.0.
    """
    n = len(frame_results)
    if n == 0:
        return {"strong_fail_frac": 0.0, "worst_disp": 0.0, "n_s": 0}
    fail_frames = [f for f in frame_results if not f.get("in_topk")]
    fails = len(fail_frames)
    worst = max((f.get("best_cand_dist_norm") or 0.0) for f in fail_frames) if fail_frames else 0.0
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
    if strong_fail_frac >= STRONG_FRAC_FLOOR:
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
    """held-out 분할. < split_min_s 면 None(insufficient). even-idx=select, odd-idx=validate.

    frame_keys: (gray, gt_xy) 튜플 또는 임의 아이템 리스트 — "키(경로)" 외에도 일반 아이템에 동작.
    """
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

    base_box 를 기준 크기로 슬라이드 윈도를 생성해 각 후보의 self_ratio 를 측정.
    반환: {box, self_ratio} 또는 None(텍스처 있는 후보가 하나도 생성되지 않은 경우).
    전부 주기 배경처럼 텍스처 없는 후보만 있을 때는 None 이 아니라 self_ratio=1.0 항목이
    반환될 수 있다 — 호출부가 self_ratio 임계로 필터링해야 한다.
    """
    h, w = img.shape[:2]
    best = None
    for box in _iter_candidate_boxes(w, h, base_box):
        sr = _patch_self_ratio(img, box)
        if best is None or sr < best["self_ratio"]:
            best = {"box": box, "self_ratio": sr}
    return best


# GT_FIDELITY_TOL: 프레임 GT 위치 근방 hit tolerance(short-side 정규화 기준 0.2 = GT_TOL_NORM).
# fidelity hit tolerance = 이 비율 × patch 단변(px). 오피스 실측상 tight band 에서 참 localization
# 은 short 의 0.20~0.24 에 떨어지고 distractor 는 >=0.42 라 깨끗이 갈린다 — 0.20 은 참 매칭을 1~6px
# 차로 놓쳐 baseline 이 all-zero 가 됐다. 0.30 으로 넓혀 참 매칭만 잡고 distractor 는 계속 기각.
_FIDELITY_GT_TOL_NORM = float(os.getenv("REREGISTER_GT_TOL_NORM", "0.30"))


def _real_box_ltrb(rcp_gray):
    """rcp 그레이 이미지에서 엔지니어 whitebox 의 진짜 위치를 (l, t, r, b) 로 반환.

    `_build_templates` 가 쓰는 것과 동일한 `_detect_white_box`/`_inner_crop_for_box` 검출기를 호출한다.
    `_detect_white_box` 는 (x, y, w, h), `_inner_crop_for_box` 의 inner_bbox 도 (x0, y0, w, h) 이지만
    여기서는 외곽 박스 위치(엔지니어 그린 박스 테두리)가 필요하므로 `_detect_white_box` 의 결과를
    (l, t, r, b) 로 변환해 반환한다.
    검출 실패 시 None.
    """
    try:
        from poc.workflow_3.align.diagnostics.align_point_correction import (
            _detect_white_box,
            _inner_crop_for_box,
        )
        det = _detect_white_box(rcp_gray)
        if det is None:
            return None
        # det = (x, y, w, h); inner_bbox 는 inner crop 좌표라 box 위치와 다름.
        # baseline 은 "엔지니어 현재 whitebox" 외곽 bbox 를 써야 한다(spec §7).
        # _inner_crop_for_box 를 호출해 inner_bbox 를 얻고, 그것을 base_box 로 사용한다.
        # inner_bbox = (x0, y0, w, h) in full-image px.
        _inner, inner_bbox = _inner_crop_for_box(rcp_gray, det)
        x0, y0, iw, ih = inner_bbox
        return (int(x0), int(y0), int(x0 + iw), int(y0 + ih))
    except Exception:
        return None


def _box_offset_xy(box, frame_w, frame_h):
    """(box_center - frame_center) in rcp px. off-center sub-crop 의 fidelity 보정용 offset.

    box: (l, t, r, b). align point(crosshair)=frame 중심 가정과 박스 중심의 차이를 돌려준다.

    주의: 부호가 엔진의 `AlignKeyTemplate.align_offset_xy`(= image_center - box_center,
    engine.py)와 **반대**다. 여기 값은 `_compute_fidelity_from_patch` 의 box_offset_xy
    슬롯에만 쓴다(기대 위치 = gt_xy + offset*scale). 엔진 align_offset 슬롯에 그대로
    넘기면 부호가 이중 반전되어 fidelity 가 전부 0 으로 무너지니 섞어 쓰지 말 것.
    """
    l, t, r, b = box
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    return (cx - frame_w / 2.0, cy - frame_h / 2.0)


def _compute_fidelity_from_patch(patch, s_frames, *, box_offset_xy=(0.0, 0.0), tag=""):
    """patch(ndarray gray)를 template 으로 각 S 프레임에 매칭한 fidelity 리스트 반환.

    s_frames: [(gray_clean, gt_xy)] 리스트.
    box_offset_xy: rcp px 기준 (box_center - rcp_center). off-center sub-crop 보정용.
    tag: 진단 라벨(baseline-* / cand / chosen-val). all-zero 경고가 엔지니어 박스(baseline)
         에서 나는지(진짜 문제) 후보 박스(cand, 변별 안 되면 0 이 정상)에서 나는지 구분용.

    fidelity = 기대 위치 hit tolerance 내 최고 점수 후보의 score. 없으면 0.0.

    매칭 후보의 xy 는 patch 중심이 frame 에 박힌 위치다(엔진은 align_offset 을 xy 에
    적용하지 않는다). 따라서 박스가 align point(=crosshair=gt_xy)에서 떨어져 있으면
    기대 매칭 위치는 gt_xy 가 아니라 gt_xy + box_offset_xy * scale 이다. 이 보정을 빼면
    off-center 박스(특히 OM unique-area)는 후보가 gt_xy 근처에 없어 fidelity 가 전부
    0 으로 떨어진다(과거 all-zero 경고의 진짜 원인). box_offset_xy=(0,0) 이면 중심 박스
    가정으로 기존 동작과 동일하다.

    rcp->msr 비교에는 `_FIDELITY_SCALES`(기본 1.0 근방 tight band) 를 사용한다 — 작은 box
    crop 이 최소 scale(0.6)로 줄어 주기 distractor 에 매칭되는 걸 막는다. A/B 는
    REREGISTER_FIDELITY_SCALES 로 band 를 바꿔 측정한다(예: COMPARE_SCALES 복원).
    모든 프레임 fidelity 가 0.0 이면(baseline tag 만) 경고를 출력한다.

    한계(co-magnification 가정): tol_px 는 patch 단변(rcp px) 기준이고 offset 은 후보 scale 로
    환산한다. rcp/msr 가 거의 같은 배율(scale~1.0)일 때 정확하다. 작은 box crop 은 변별력이
    약해(주기 SEM) 참 위치에 안 붙을 수 있고, 그건 좌표 버그가 아니라 매칭 recall 한계다.
    """
    if patch.size == 0 or not s_frames:
        return []
    tpl = build_template(patch.copy(), recipe_id="sugg", version="sugg", key_type="om")
    ph, pw = patch.shape[:2]
    short = max(1, min(pw, ph))
    tol_px = _FIDELITY_GT_TOL_NORM * short
    ox, oy = box_offset_xy
    fidelities = []
    # all-zero 진단 카운터: 세 경로(exc/empty/offtarget)를 구분해 한 줄로 보고.
    n_exc = n_empty = n_offtarget = 0
    first_exc = ""
    nearest_dbg = None   # (best_dist, tol, best_scale, best_xy, expected_xy) — 기대 위치 최근접 후보.
    top1_dbg = None      # (n_cands, top1_xy, top1_scale, top1_score, top1_dist) — 최고점수 후보.
    for gray, gt_xy in s_frames:
        try:
            res = compute_align_key_score_ensemble(tpl, gray, scales=_FIDELITY_SCALES, policy=STRUCTURE_POLICY)
            cands = list(res.candidates)
        except Exception as exc:
            fidelities.append(0.0)
            n_exc += 1
            if not first_exc:
                first_exc = f"{type(exc).__name__}: {exc}"
            continue
        if not cands:
            fidelities.append(0.0)
            n_empty += 1
            continue
        gx, gy = gt_xy
        # 기대 매칭 위치 = gt_xy + box_offset * scale (후보마다 scale 다름) 의 tolerance 내 최고 score.
        dists = [
            (float(np.hypot(c.xy[0] - (gx + ox * c.scale),
                            c.xy[1] - (gy + oy * c.scale))), c)
            for c in cands
        ]
        near = [c for d, c in dists if d <= tol_px]
        if near:
            fidelities.append(float(max(c.score for c in near)))
        else:
            fidelities.append(0.0)
            n_offtarget += 1
            bd, bc = min(dists, key=lambda dc: dc[0])   # 기대 위치에 가장 가까운 후보.
            if nearest_dbg is None or bd < nearest_dbg[0]:
                nearest_dbg = (bd, tol_px, bc.scale, bc.xy,
                               (round(gx + ox * bc.scale, 1), round(gy + oy * bc.scale, 1)))
            # 최고-score 후보(엔진의 실제 1픽)가 기대 위치와 얼마나 먼지 — 변별 실패 vs 좌표계 진단.
            t1d, t1c = max(dists, key=lambda dc: dc[1].score)
            if top1_dbg is None:
                top1_dbg = (len(cands), t1c.xy, t1c.scale, round(float(t1c.score), 3), round(t1d, 1))
    # all-zero 경고는 baseline(엔지니어 박스)에 한정해 출력한다. cand(후보 박스)는 주기 SEM
    # 텍스처의 비변별 sub-region 이라 distractor 에 wrong-phase 매칭(off-target)되어 0 이 나는
    # 게 정상(=올바른 기각)이므로 경고를 띄우지 않는다. baseline all-zero 는 엔지니어 박스조차
    # msr 에 재정착하지 못한다는 뜻이라 진짜 재등록 신호다 — 이때만 진단 한 줄을 남긴다.
    if fidelities and all(f == 0.0 for f in fidelities) and tag.startswith("baseline"):
        # 0.0 은 세 경로(엔진 예외/후보 없음/기대 위치 밖) 모두에서 나올 수 있어 분해해 보고.
        n = len(fidelities)
        msg = (f"[WARNING] fidelity all-zero[{tag or '?'}]: frames={n} exc={n_exc} empty={n_empty} "
               f"offtarget={n_offtarget} | patch={pw}x{ph} tol={tol_px:.1f} "
               f"offset=({ox:.0f},{oy:.0f})")
        if nearest_dbg is not None:
            bd, tp, bsc, bxy, exp = nearest_dbg
            msg += f" | nearest: xy={bxy} expected={exp} dist={bd:.1f} scale={bsc:.2f}"
        if top1_dbg is not None:
            nc, t1xy, t1sc, t1score, t1d = top1_dbg
            msg += f" | top1(n={nc}): xy={t1xy} scale={t1sc:.2f} score={t1score} dist={t1d}"
        if first_exc:
            msg += f" | exc: {first_exc}"
        print(msg)
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

    # base_box: 현재 엔지니어 whitebox 의 실제 위치(rcp native px).
    # _real_box_ltrb 가 _detect_white_box/_inner_crop_for_box 로 진짜 박스 위치를 복원한다.
    # 실패 시 _box 템플릿 크기를 이미지 중앙에 위치시키는 근사로 폴백하고 경고 플래그를 세운다.
    box_tpl = (row.get("_box") or {}).get(modality)
    if box_tpl is None:
        row["suggestion"] = "no distinctive sub-region"
        return
    base_box = _real_box_ltrb(rcp_gray)
    sugg_pos_approx = False
    if base_box is None:
        # 폴백: box 템플릿 크기로 중앙 근사.
        th, tw = box_tpl.raw_image.shape[:2]
        rh, rw = rcp_gray.shape[:2]
        cx, cy = rw // 2, rh // 2
        base_box = (max(0, cx - tw // 2), max(0, cy - th // 2),
                    min(rw, cx + tw // 2 + tw % 2), min(rh, cy + th // 2 + th % 2))
        sugg_pos_approx = True

    # baseline: 현재 base_box patch 의 fidelity.
    # off-center 박스 보정: 매칭 후보는 patch 중심 위치를 돌려주므로 기대 위치는
    # gt_xy + (box_center - rcp_center) 다. 이 offset 없이 평가하면 align point 에서
    # 떨어진 박스(특히 OM)의 fidelity 가 전부 0 으로 떨어진다.
    h, w = rcp_gray.shape[:2]
    bl, bt, br, bb = base_box
    base_patch = rcp_gray[bt:bb, bl:br]
    base_off = _box_offset_xy(base_box, w, h)
    baseline_sel = _compute_fidelity_from_patch(base_patch, sel_frames, box_offset_xy=base_off, tag="baseline-sel")
    baseline_val = _compute_fidelity_from_patch(base_patch, val_frames, box_offset_xy=base_off, tag="baseline-val")
    baseline_self = _patch_self_ratio(rcp_gray, base_box)

    # select-half: 후보 박스 검색 + 각 후보 fidelity 계산.
    cand_metrics = []
    for box in _iter_candidate_boxes(w, h, base_box):
        sr = _patch_self_ratio(rcp_gray, box)
        l, t, r, b = box
        patch = rcp_gray[t:b, l:r]
        if not _passes_texture(patch):
            continue
        sel_fid = _compute_fidelity_from_patch(patch, sel_frames, box_offset_xy=_box_offset_xy(box, w, h), tag="cand")
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
    chosen["val_fidelities"] = _compute_fidelity_from_patch(
        cand_patch, val_frames, box_offset_xy=_box_offset_xy(chosen["box"], w, h), tag="chosen-val")
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
        # base_box 의 단변(short side)으로 기준 크기를 산정(tw/th 의존 제거).
        bl, bt, br, bb = base_box
        bw_bb, bh_bb = br - bl, bb - bt
        short_s = max(1, min(bw_bb, bh_bb))
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
    sugg_str = f"box{box}"
    if sugg_pos_approx:
        sugg_str += "(approx-pos)"
    if len(val_frames) < 2:
        sugg_str += "(1-frame-val)"   # validate-half 1장 -> 약한 검증, 엔지니어 육안 확인 필수.
    row["suggestion"] = sugg_str
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

        # 현재 whitebox: _real_box_ltrb 로 진짜 위치 복원.
        # _real_box_ltrb 실패 시 box template 크기로 중앙 근사(overlay 표기에만 영향).
        box_tpl = (row.get("_box") or {}).get(modality)
        if box_tpl is not None:
            base_box = _real_box_ltrb(rcp_gray)
            if base_box is None:
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

from poc.workflow_2.align_similarity import _build_templates, _gt_in_topk, COMPARE_SCALES
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
    # A/B 자기-라벨: 활성 fidelity scale band 를 한 줄로 찍어 relay 시 어느 arm 인지 명확히.
    print(f"[INFO] fidelity_scales={_FIDELITY_SCALES} (env REREGISTER_FIDELITY_SCALES to A/B)")
    recipes = _walk_recipes(root)
    cap = int(os.getenv("REREGISTER_MAX_RECIPES", "0") or "0")
    if cap and cap > 0:
        n_all = len(recipes)
        recipes = _cap_recipes(recipes, cap)
        print(f"[INFO] fast-mode: recipes capped {len(recipes)}/{n_all} (env REREGISTER_MAX_RECIPES=0 for full)")
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
    # cp949 콘솔에서 비-ASCII recipe 이름이 포함될 수 있으므로 ASCII safe 변환 후 출력(파일은 utf-8 그대로).
    print(digest.encode("ascii", "replace").decode("ascii"))
    return digest


if __name__ == "__main__":
    run()
