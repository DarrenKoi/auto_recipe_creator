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
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_3.align.matching.engine import (
    DT_TAU_PX,
    STRUCTURE_POLICY,
    build_template,
    compute_align_key_score,
)

# rcp 중앙 align 영역 crop 의 면적 비율 (각 변 = sqrt). align_point_correction 의 fallback 과 동일 계열.
CENTER_AREA_RATIO = 0.15
# 정적 비교 scale band (rcp/msr 거의 같은 배율 가정) — align_point_correction.COMPARE_SCALES 와 동일.
COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)
# at_center ROI 한 변 = template 변 × 이 배수 (검색 여유).
ROI_FACTOR = 1.8
# at_crosshair 는 detector 위치(정답)에 *고정*해 재야 mi_xhair 와 co-located 되고
# recoverable_by_move 가 의미를 갖는다 → 매칭이 거의 못 움직이게 좁은 ROI.
AT_CROSSHAIR_ROI_FACTOR = 1.2
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
# S-consensus 자체 정보량이 낮으면 MI ratio verdict 를 내리지 않는다. 저텍스처에서는 MI 가
# 작은 절대값 주변에서 흔들려 stale/ok 판정이 과감해질 수 있으므로 office 재측정으로 보정한다.
MIN_CONSENSUS_SELF_MI = 0.10

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
# 생산 matcher 와 동일한 후보 수를 측정하도록 policy 에서 파생(literal 복사 금지 → drift 방지).
TOPK_CANDIDATES = STRUCTURE_POLICY.top_n
# 후보가 정답으로 인정되는 거리(template 짧은 변 대비). truth-forced 의 lock 판정과는 *독립* 임계라
# 별도 값으로 둔다(한쪽을 조이면 다른 쪽이 따라 움직이는 것 방지).
GT_TOL_NORM = 0.20

# consensus proposer A/B 토글 — 기본 C1(canny chamfer). env CONSENSUS_USE_ENSEMBLE=1 이면 후보
# 생성기(_gt_in_topk)를 3채널 RRF ensemble(C1 canny + C2 scharr + C3 orient)로 바꿔, 같은 LOO
# harness 에서 in_topk(proposer recall) 이 뛰나 측정한다. 두 경로 모두 후보 xy = template-center·
# frame px (동일 계약) → apples-to-apples. 게다가 ensemble.solo['canny'] 가 곧 C1 이므로 ON==OFF
# 면 C2/C3/RRF 가 무효(= 잔여 miss 가 구조적 천장)라는 직접 증거가 된다. 비용: ensemble 은 채널
# 3배(~1s/frame)라 오피스 A/B 전용. (참조 [[project_ensemble_on_consensus_rejected]])
#
# A/B 스위치 (둘 중 편한 방법):
#   1) 코드로: 아래 _USE_ENSEMBLE_DEFAULT 를 False(C1) / True(ensemble) 로 직접 바꿔 두 번 실행.
#      env 안 건드려도 되고 세션 잔존(persist) 함정도 없다 — 단, 바꾼 줄을 commit 하지 않도록 주의.
#   2) env 로: CONSENSUS_USE_ENSEMBLE=1/0 (설정돼 있으면 코드 기본값을 덮어쓴다). tree 깨끗·스크립트용.
_USE_ENSEMBLE_DEFAULT = False
_use_ens_env = os.getenv("CONSENSUS_USE_ENSEMBLE")
USE_ENSEMBLE_PROPOSER = (_use_ens_env != "0") if _use_ens_env is not None else _USE_ENSEMBLE_DEFAULT

# --- S-consensus 템플릿 A/B (재등록 검증): rcp 대신 S-consensus 로 바꾸면 gt_in_topk 가 뛰나 ---
# rcp 가 stale 하다는 가설을 *기존 S 데이터만으로* 검증한다. S crop 은 crosshair 검출 위치
# (matcher 무관, ground truth)에서 떼므로 비순환. leave-one-out 으로 held-out S 의 full frame 에
# consensus 템플릿을 매칭 → in_topk 가 rcp 대비 뛰면 재등록이 정당화되고 consensus 가 새 rcp 후보.
# E false-positive 가드: consensus free_best chamfer 가 E 에서도 S 만큼 높으면 = 흐릿한 generic
# 템플릿(가짜 회복)이므로 경고. (참조 [[project_matcher_flat_chamfer_distinctiveness]])
# LOO 는 held-out 1장을 빼고 consensus 를 만들어야 하므로 staleness 의 consensus 최소치보다 1 더.
AB_MIN_S = MIN_S_FOR_CONSENSUS + 1   # LOO consensus 에 필요한 최소 S(crosshair) 장수.
AB_E_SAMPLE = 8                # recipe 당 E false-positive 가드 표본 상한(비용 제한).


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


def _lap_var(gray: np.ndarray) -> float:
    """Laplacian 분산 — 선명도(sharpness) 지표. median consensus 가 blur 됐는지 확인용."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


def _propose_topk(tpl, gray, frame_dt, *, scales, topk):
    """consensus localization 후보 top-N (xy = template-center·frame px, score 내림차순).

    CONSENSUS_USE_ENSEMBLE 로 C1(canny chamfer) ↔ 3채널 RRF ensemble 을 전환한다. 두 경로의 후보
    좌표 계약이 동일(template-center, frame px)해 _gt_in_topk 의 in_topk A/B 가 apples-to-apples.
    ensemble 은 raw gray 에서 자체 전처리하므로 frame_dt 가 필요 없다 — frame_dt 는 C1 전용.

    ALIGN_ENSEMBLE_LAB_MODE(또는 명시 채널)가 활성이면 lab ensemble(예: edge_ncc=C4 포함)로
    후보를 낸다 — rcp-only arm 과 동일 리졸버(ensemble_lab.lab_channels_from_env)를 써서 'edge_ncc'
    가 두 arm 에서 같은 채널을 뜻하게 한다. recall_miss(consensus arm 의 진짜 약점)에 C4 proposer 를
    A/B 로 댈 수 있는 자리. ensemble_lab 을 모듈로 참조(call-time attr)해 디스패치를 테스트 가능하게 둔다.
    """
    from poc.workflow_2 import ensemble_lab as _lab
    if _lab.lab_active_from_env():
        ens = _lab.compute_ensemble_candidates(
            tpl.raw_image, gray, channels=_lab.lab_channels_from_env(),
            scales=scales, top_n=topk)
        return list(ens.fused[:topk])
    if USE_ENSEMBLE_PROPOSER:
        from poc.workflow_3.align.matching.ensemble import compute_ensemble_candidates
        ens = compute_ensemble_candidates(tpl.raw_image, gray, scales=scales, top_n=topk)
        return list(ens.fused[:topk])   # RRF 내림차순; in_topk 은 집합 멤버십이라 rerank 무관.
    from poc.workflow_3.align.matching.engine import compute_chamfer_candidates
    return compute_chamfer_candidates(tpl, frame_dt, scales=scales, top_n=topk)


def _gt_in_topk(gray, crosshair_xy, center_tpls, *, topk=TOPK_CANDIDATES, scales=COMPARE_SCALES):
    """정답(crosshair) 위치가 free-search 후보 top-N 안에 들어오는지 측정 (proposer recall).

    후보 생성기는 CONSENSUS_USE_ENSEMBLE 토글(C1 canny vs 3채널 RRF ensemble) — _propose_topk 참조.

    proposer(후보 생성)가 truth 를 surface 하는지 → 리랭킹(MI 등)으로 고칠 수 있는지 판정.
      - in_topk=True, rank>1  → truth 가 후보엔 있는데 chamfer 순위만 밀림 ⇒ reranker 로 회복 가능.
      - in_topk=False         → truth 가 후보에 아예 없음 ⇒ proposer 교체 필요(리랭킹 무의미).

    modality 는 *race* (생산 matcher 와 동일): 각 modality 의 top-N 안에서 truth 순위를 보고
    modality 끼리 best(최저 rank)를 채택한다. 후보를 pool 해 global truncate 하면 한 modality 의
    잡음 peak 이 다른 modality 의 truth 후보를 밀어내 recall 을 과소평가하므로 금지.

    반환 dict: {topk_rank, in_topk, n_cand, best_cand_dist_norm, modality, cand_xys} 또는 None.
    cand_xys 는 채택된 modality 의 top-N 후보 좌표(score 내림차순) — 결합 패널 시각화용.
    (rerank[MI·contour] 검증은 끝남 — 둘 다 폐기, `docs/study/reranker_ab_failure_analysis.md`.)
    """
    from poc.workflow_3.align.matching.engine import preprocess_for_matching
    from poc.workflow_2.ensemble_lab import peak_isolation_ratio
    # C1 경로만 frame distance-transform 이 필요 — ensemble 은 raw gray 로 자체 전처리(중복 회피).
    frame_dt = None if USE_ENSEMBLE_PROPOSER else preprocess_for_matching(gray)[1]
    cxh, cyh = crosshair_xy
    best = None  # (rank|None, n_cand, best_dist, mod, cand_xys, peak_ratio)
    any_cand = False
    for mod, tpl in center_tpls.items():
        if tpl is None:
            continue
        th, tw = tpl.raw_image.shape[:2]
        short = max(1, min(tw, th))
        try:
            cands = _propose_topk(tpl, gray, frame_dt, scales=scales, topk=topk)
        except Exception:
            continue
        if not cands:
            continue
        any_cand = True
        dists = [float(np.hypot(c.xy[0] - cxh, c.xy[1] - cyh)) / short for c in cands]
        rank = next((i for i, d in enumerate(dists, 1) if d <= GT_TOL_NORM), None)
        cur = (rank, len(cands), min(dists), mod,
               [[int(c.xy[0]), int(c.xy[1])] for c in cands],
               peak_isolation_ratio(cands), list(cands))   # (B) 모호도 + 변형 ablation 용 cands.
        # race: in_topk(rank!=None) 우선 → 낮은 rank → 가까운 best_dist.
        if best is None:
            best = cur
        else:
            b_rank, _bn, b_dist, _bm, _bc, _bp, _bcands = best
            better = (
                (rank is not None and (b_rank is None or rank < b_rank))
                or (rank is None and b_rank is None and cur[2] < b_dist)
            )
            if better:
                best = cur
    if not any_cand:
        return None
    rank, n_cand, best_dist, mod, cand_xys, peak_ratio, adopted_cands = best
    # (B) 변형 ablation 재료: 채택 modality 후보의 chamfer 점수 + NCC 재점수(같은 후보 순서).
    cand_scores = [round(float(c.score), 5) for c in adopted_cands]
    cand_ncc = None
    tpl_adopted = center_tpls.get(mod)
    if tpl_adopted is not None:
        th2, tw2 = tpl_adopted.raw_image.shape[:2]
        cand_ncc = []
        for c in adopted_cands:
            crop = _matched_crop(gray, c.xy, tw2, th2, getattr(c, "scale", 1.0) or 1.0)
            cand_ncc.append(round(float(_ncc(tpl_adopted.raw_image, crop)), 5)
                            if crop is not None else None)
    return {
        "topk_rank": rank,
        "in_topk": rank is not None,
        "n_cand": n_cand,
        "best_cand_dist_norm": round(best_dist, 3),
        "modality": mod,
        "cand_xys": cand_xys,
        "peak_ratio": round(peak_ratio, 4),   # (B) match-time 모호도: top2/top1, 높을수록 miss-like.
        "cand_scores": cand_scores,           # 변형 ablation: proposer 점수(내림차순).
        "cand_ncc": cand_ncc,                 # 변형 ablation: 후보별 NCC 재점수(C4 form 검증).
    }


# ====================================================================
# 템플릿 빌드 (rcp 중앙 + 흰 box).
# ====================================================================


def _build_templates(assets):
    """recipe rcp 에서 center / box template 을 modality 별로 만든다.

    반환: (center_templates: dict[mod->tpl], box_templates: dict[mod->tpl|None])
    center 는 정중앙 면적 crop(= align 영역), box 는 흰 unique-area box 안쪽 crop.
    """
    from poc.workflow_3.align.assets import load_gray
    from poc.workflow_3.align.diagnostics.align_point_correction import (
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
    from poc.workflow_3.align.assets import load_gray
    from poc.workflow_3.align.diagnostics.align_point_correction import _tool_label
    from poc.workflow_3.align.diagnostics.crosshair_detect import detect_crosshair

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
        # 좁은 ROI 로 detector 위치에 사실상 고정 → at_crosshair 가 ROI 내 다른 peak 으로
        # 새지 않아 mi_xhair 와 같은 위치를 재고 recoverable_by_move 가 의미를 갖는다.
        ch_roi = _window_roi(gray.shape, ch_res.xy, tw, th, factor=AT_CROSSHAIR_ROI_FACTOR)
        at_ch = _race(center_tpls, gray, roi=ch_roi)
        if at_ch:
            at_crosshair_score = at_ch[1]
            ch_mod, _s, _c, _o, _ch_xy, _ch_scale = at_ch
            tpl = center_tpls[ch_mod]
            # S 의 crosshair center 는 ground truth 이므로 consensus crop 은 matcher 위치가 아니라
            # detector 위치에서 고정 scale=1.0 으로 자른다. modality 만 ROI race winner 를 따른다.
            xcrop = _matched_crop(gray, ch_res.xy, tpl.raw_image.shape[1], tpl.raw_image.shape[0], 1.0)
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
        "xhair_crop_source": "detector_xy_scale1" if xhair_crop is not None else None,
        "xhair_crop_modality": xhair_mod,
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
      - S-consensus 정보량 자체가 낮음 → status=low_texture_inconclusive.
      - S 끼리도 안 뭉침(높은 CV) → consensus 불신 → status=S_inconsistent (CV 로 판단 불가).
      - S 장수 부족 → insufficient_S.
    반환: relative_ratio 오름차순(worst-first) 리스트.
    """
    out = []
    for rec, data in crops_by_recipe.items():
        # s_frames(단일 원천)에서 modality 별 crop 으로 group — s_crops 중복 저장 제거.
        s_crops: dict = {}
        for f in data.get("s_frames", []):
            s_crops.setdefault(f["mod"], []).append(f["crop"])
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
        # 우선순위: S 끼리 안 뭉치면(high CV) consensus 자체를 못 믿으므로 texture 보다 먼저 판정.
        if s_cv > S_INCONSISTENT_CV:
            entry["status"] = "S_inconsistent"      # consensus 불신 → CV 단독 판단 불가(golden 필요).
        elif s_med < MIN_CONSENSUS_SELF_MI:
            entry["status"] = "low_texture_inconclusive"   # MI 자체가 낮아 ratio 판정 보류.
        elif ratio < RELATIVE_STALE_RATIO:
            entry["status"] = "stale_replace"        # rcp 가 S cluster 의 outlier → 재등록 권장.
        else:
            entry["status"] = "ok"
        out.append(entry)
    out.sort(key=lambda d: d.get("relative_ratio", 9.99))
    return out


def _valid_ratio_in_topk_pairs(rows: list[dict]) -> list[tuple]:
    """(relative_ratio, in_topk, recipe) 추출 — tertile/상관 공통 전처리(정의 중복 방지)."""
    pairs = []
    for r in rows:
        ratio = r.get("reference_relative_ratio")
        gt = r.get("gt_topk") or {}
        if isinstance(ratio, (int, float)) and "in_topk" in gt:
            pairs.append((float(ratio), bool(gt["in_topk"]), r.get("recipe")))
    return pairs


def _ratio_tertile(rows: list[dict]) -> dict | None:
    """gt_topK 행을 relative_ratio tertile 로 나눠 row-weighted recall 을 본다."""
    scored = _valid_ratio_in_topk_pairs(rows)
    if len(scored) < 3:
        return None
    scored.sort(key=lambda t: t[0])
    buckets = {"low": [], "mid": [], "high": []}
    n = len(scored)
    for i, item in enumerate(scored):
        if i < n / 3:
            buckets["low"].append(item)
        elif i < 2 * n / 3:
            buckets["mid"].append(item)
        else:
            buckets["high"].append(item)
    out = {}
    for name, items in buckets.items():
        n_items = len(items)
        n_in = sum(1 for _ratio, in_topk, _recipe in items if in_topk)
        ratios = [ratio for ratio, _in_topk, _recipe in items]
        out[name] = {
            "n": n_items,
            "n_recipes": len({recipe for _ratio, _in_topk, recipe in items}),
            "n_in_topk": n_in,
            "in_topk_rate": round(n_in / n_items, 3) if n_items else None,
            "ratio_min": round(min(ratios), 3) if ratios else None,
            "ratio_max": round(max(ratios), 3) if ratios else None,
            "ratio_median": round(statistics.median(ratios), 3) if ratios else None,
        }
    return out


def _point_biserial_in_topk_vs_ratio(rows: list[dict]) -> dict | None:
    """binary in_topk 와 continuous relative_ratio 의 Pearson r (= point-biserial), np.corrcoef."""
    pairs = _valid_ratio_in_topk_pairs(rows)
    if len(pairs) < 3:
        return None
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([1.0 if p[1] else 0.0 for p in pairs], dtype=np.float64)
    if xs.std() <= 0 or ys.std() <= 0:
        return {"n": len(pairs), "r": None, "note": "zero variance"}
    r = float(np.corrcoef(xs, ys)[0, 1])
    return {"n": len(pairs), "r": round(r, 3)}


def _gt_topk_reference_crosstab(rows: list[dict], reference_quality: list[dict]) -> dict | None:
    """reference status/ratio 별 gt-in-topK recall. 원인 분리용 진단 출력."""
    ref_by_recipe = {d.get("recipe"): d for d in reference_quality if d.get("recipe")}
    gt_rows = [r for r in rows if r.get("gt_topk") is not None and r.get("recipe") in ref_by_recipe]
    if not gt_rows:
        return None
    enriched = []
    for r in gt_rows:
        ref = ref_by_recipe[r["recipe"]]
        item = dict(r)
        item["reference_status"] = ref.get("status")
        item["reference_relative_ratio"] = ref.get("relative_ratio")
        enriched.append(item)

    by_status = {}
    for r in enriched:
        status = r.get("reference_status") or "unknown"
        bucket = by_status.setdefault(
            status,
            {"n": 0, "n_recipes": set(), "n_in_topk": 0, "ratios": []},
        )
        bucket["n"] += 1
        bucket["n_recipes"].add(r.get("recipe"))
        if r["gt_topk"]["in_topk"]:
            bucket["n_in_topk"] += 1
        ratio = r.get("reference_relative_ratio")
        if isinstance(ratio, (int, float)):
            bucket["ratios"].append(float(ratio))

    by_status_out = {}
    for status, bucket in sorted(by_status.items()):
        n = bucket["n"]
        ratios = bucket["ratios"]
        by_status_out[status] = {
            "n": n,
            "n_recipes": len(bucket["n_recipes"]),
            "n_in_topk": bucket["n_in_topk"],
            "n_miss": n - bucket["n_in_topk"],
            "in_topk_rate": round(bucket["n_in_topk"] / n, 3) if n else None,
            "median_ratio": round(statistics.median(ratios), 3) if ratios else None,
        }

    return {
        "n": len(enriched),
        "n_recipes": len({r.get("recipe") for r in enriched}),
        "by_status": by_status_out,
        "ratio_tertiles": _ratio_tertile(enriched),
        "point_biserial_in_topk_vs_ratio": _point_biserial_in_topk_vs_ratio(enriched),
        "note": "row-weighted S+crosshair only; inspect n_recipes before interpreting small buckets",
    }


def _miss_dist_distribution(dists, *, tol=GT_TOL_NORM):
    """miss(in_topk=False) 의 '진실↔최근접 후보' 거리 분포 — ensemble 적용 여지 진단.

    miss 는 정의상 best_cand_dist_norm > tol. 그 초과분이 tol 바로 밖(near)에 몰리면 후보가
    가까이는 왔으나 허용오차를 못 넘은 것 → ensemble 이 후보를 더/다르게 내 끌어들일 여지(recall
    압력). far/veryfar 로 퍼지면 어떤 후보도 진실 근처에 없는 것 → 구조적 모호성(ensemble 무력,
    다른 축 필요). 거리는 short-side 상대값(GT_TOL_NORM=tol 와 동일 척도).

    반환: {n, median, p25, p75, max, bins{near/mid/far/veryfar}} (비면 n=0).
    """
    if not dists:
        return {"n": 0, "median": None, "p25": None, "p75": None, "max": None, "bins": None}
    s = sorted(float(d) for d in dists)
    n = len(s)
    # 경계를 3자리로 반올림 — best_cand_dist_norm 도 round(_,3) 이라 부동소수점 경계 오분류 방지.
    e1, e2, e3 = round(1.5 * tol, 3), round(2 * tol, 3), round(3 * tol, 3)

    def _q(frac):
        return round(s[min(n - 1, int(frac * n))], 3)

    bins = {
        f"near[{tol:.2f}-{e1:.2f})": sum(1 for d in s if d < e1),
        f"mid[{e1:.2f}-{e2:.2f})": sum(1 for d in s if e1 <= d < e2),
        f"far[{e2:.2f}-{e3:.2f})": sum(1 for d in s if e2 <= d < e3),
        f"veryfar[>={e3:.2f}]": sum(1 for d in s if d >= e3),
    }
    return {"n": n, "median": round(statistics.median(s), 3),
            "p25": _q(0.25), "p75": _q(0.75), "max": round(s[-1], 3), "bins": bins}


def _iter_recipe_modalities(by_recipe: dict):
    """(rec, data, mod) 를 modality별로 하나씩 흘려보낸다 — recipe 의 *모든* modality 평가.

    이전에는 recipe당 `Counter(mod).most_common(1)` 로 dominant modality 한 종류만 평가했다.
    그런데 align 측정 1회당 OM 2장 / SEM 3장이라([[project_om_sem_positions_per_measurement]])
    dual recipe 는 거의 항상 SEM 이 dominant → OM consensus 가 영영 측정되지 못했다. 각 modality
    를 독립 평가하면 dual recipe 도 OM·SEM consensus row 를 각각 낸다(같은 마크·다른 stage 위치라
    modality 내 풀링은 그대로 타당). s_frames 가 없는 recipe 는 건너뛴다.
    """
    for rec, data in by_recipe.items():
        frames = data.get("s_frames", [])
        if not frames:
            continue
        for mod in sorted({f["mod"] for f in frames}):
            yield rec, data, mod


def _consensus_template_ab(by_recipe: dict, *, min_s=AB_MIN_S, out_dir=None,
                           combined_renderer=None, frame_loader=None) -> dict | None:
    """S-consensus 템플릿 A/B (leave-one-out) — rcp 대신 consensus 로 in_topk 가 뛰나.

    by_recipe[rec] = {"s_frames": [{path,xy,mod,crop}], "e_paths": [Path], "rcp_tpls": {mod: tpl}}.
    재등록 검증 + 프로토타입. rcp baseline 을 *이 함수 안에서* 같은 modality·같은 frame·같은
    _gt_in_topk 로 재계산해 apples-to-apples 로 비교한다(파일명 join / om+sem-vs-단일 비대칭 제거).
    generic 가드: S·E 모두 *동일* all-crops consensus 의 free-best chamfer 로 재 비교 가능하게 한다
    (cons_E >= cons_S 면 흐릿한 generic 템플릿 = 가짜 회복). 검증된 consensus 는 out_dir/consensus/
    에 저장 → 재등록 도구가 같은 산출물을 그대로 쓰게 한다(별도 재계산 divergence 방지).
    combined_renderer: LOO 한 점마다 호출되는 시각화 훅(없으면 무시). ctx dict
      {recipe, mod, path, gray, xy, cons_tpl, rcp_tpl, gc, gr} 를 받는다 — 측정은 여기(단일
      출처)서 하고 그림은 호출측이 그리게 분리(매칭 재실행/표류 방지). 렌더 예외는 삼킨다.
    frame_loader: LOO 매칭 프레임 로더 f(s_frame dict)->gray|None (기본 load_gray(path) raw).
      cond 판이 crosshair 정제 프레임을 주입하는 자리 — consensus 중앙의 inpaint 잔상이
      프레임 GT crosshair 와 가짜 lock 하는 비대칭(consensus 만 유리)을 끊는 A/B 용.
      None 반환/예외 = 그 프레임 skip. cons/rcp/가드 S 모두 같은 프레임으로 측정(공정 비교).
    반환: per-recipe + overall in_topk_rate(rcp vs consensus) + lift + generic 가드.
    """
    from poc.workflow_3.align.assets import load_gray

    cons_dir = None
    if out_dir is not None:
        cons_dir = Path(out_dir) / "consensus"
        cons_dir.mkdir(parents=True, exist_ok=True)

    per_recipe = []
    tot_n = tot_rcp = tot_cons = tot_rcp_r1 = tot_cons_r1 = 0
    s_guard: list[float] = []   # all-crops consensus free-best chamfer on S (참고용).
    e_guard: list[float] = []   # 동일 consensus free-best chamfer on E (참고용).
    # 선명도(blur) 비율 — consensus median 이 개별 S crop / rcp 대비 흐려졌는지 (per-recipe).
    edge_ratio_s: list[float] = []
    lap_ratio_s: list[float] = []
    edge_ratio_rcp: list[float] = []
    lap_ratio_rcp: list[float] = []
    # miss(in_topk=False) 의 '진실↔최근접 후보' 거리 — ensemble 적용 여지 진단(버려지던 값 수집).
    cons_miss_dists: list[float] = []
    rcp_miss_dists: list[float] = []
    # (B) match-time 모호도 검증용 per-point (missed, peak_ratio) — periodicity 와 달리 진짜 점 단위.
    cons_points: list[dict] = []
    for rec, data, mod in _iter_recipe_modalities(by_recipe):
        frames = data["s_frames"]
        fm = [f for f in frames if f["mod"] == mod]
        # consensus 재료: from_msr_history(disjoint 과거 S 풀) 우선, 없으면 LOO(fm).
        history = (data.get("history_crops") or {}).get(mod)
        use_history = history is not None and len(history) >= min_s
        if use_history:
            if not fm:                     # 채점할 from_msr S 가 없으면 skip.
                continue
            crops = list(history)          # consensus 풀 = 과거 S (eval=fm 과 disjoint, LOO 불필요).
        else:
            if len(fm) < min_s:
                continue
            crops = [f["crop"] for f in fm]
        rcp_tpl = data.get("rcp_tpls", {}).get(mod)
        # all-crops consensus 1회 빌드(루프 불변) — 가드 + 저장에 재사용.
        all_consensus = _consensus(crops)
        all_cons_tpl = build_template(all_consensus, recipe_id=rec,
                                      version="s_consensus_all", key_type=mod)
        if cons_dir is not None:
            cv2.imwrite(str(cons_dir / (rec.replace("/", "__") + f"_{mod}.png")), all_consensus)
        # 선명도 비율: all-crops consensus vs 개별 S crop median / rcp raw (blur 확인).
        c_ed, c_lap = _edge_density(all_consensus), _lap_var(all_consensus)
        s_ed_med = statistics.median([_edge_density(c) for c in crops])
        s_lap_med = statistics.median([_lap_var(c) for c in crops])
        if s_ed_med > 0:
            edge_ratio_s.append(c_ed / s_ed_med)
        if s_lap_med > 0:
            lap_ratio_s.append(c_lap / s_lap_med)
        if rcp_tpl is not None:
            r_ed, r_lap = _edge_density(rcp_tpl.raw_image), _lap_var(rcp_tpl.raw_image)
            if r_ed > 0:
                edge_ratio_rcp.append(c_ed / r_ed)
            if r_lap > 0:
                lap_ratio_rcp.append(c_lap / r_lap)
        n = rcp_hit = cons_hit = rcp_r1 = cons_r1 = 0
        for i, f in enumerate(fm):
            if use_history:
                cons_tpl = all_cons_tpl    # disjoint 과거 S consensus — 모든 eval frame 동일(LOO 없음).
            else:
                others = [c for j, c in enumerate(crops) if j != i]
                if len(others) < 2:
                    continue
                cons_tpl = build_template(_consensus(others), recipe_id=rec,
                                          version="s_consensus", key_type=mod)
            try:
                gray = frame_loader(f) if frame_loader is not None else load_gray(f["path"])
            except Exception:
                continue
            if gray is None:
                continue
            gc = _gt_in_topk(gray, tuple(f["xy"]), {mod: cons_tpl})
            if gc is None:
                continue
            n += 1
            cons_points.append({"recipe": rec, "missed": not gc["in_topk"],
                                "peak_ratio": gc.get("peak_ratio"),
                                "cand_scores": gc.get("cand_scores"),
                                "cand_ncc": gc.get("cand_ncc")})   # (B) 점 단위 모호도↔miss + 변형 재료.
            if gc["in_topk"]:
                cons_hit += 1
            elif gc.get("best_cand_dist_norm") is not None:
                cons_miss_dists.append(gc["best_cand_dist_norm"])   # miss 거리(ensemble 여지 진단).
            if gc["topk_rank"] == 1:        # rank-1 = consensus 의 free-best 가 정답에 lock(distinctive).
                cons_r1 += 1
            # rcp baseline — 같은 modality·frame·_gt_in_topk (apples-to-apples).
            gr = None
            if rcp_tpl is not None:
                gr = _gt_in_topk(gray, tuple(f["xy"]), {mod: rcp_tpl})
                if gr is not None:
                    if gr["in_topk"]:
                        rcp_hit += 1
                        if gr["topk_rank"] == 1:
                            rcp_r1 += 1
                    elif gr.get("best_cand_dist_norm") is not None:
                        rcp_miss_dists.append(gr["best_cand_dist_norm"])
            # 결합 패널 훅 — 측정에 쓴 것과 동일한 frame/template/후보로 그리게 컨텍스트 전달.
            # 렌더 실패가 A/B 측정을 깨면 안 되므로 예외는 경고만 남기고 계속.
            if combined_renderer is not None:
                try:
                    combined_renderer({
                        "recipe": rec, "mod": mod, "path": f["path"],
                        "gray": gray, "xy": tuple(f["xy"]),
                        "cons_tpl": cons_tpl, "rcp_tpl": rcp_tpl,
                        "gc": gc, "gr": gr,
                    })
                except Exception as exc:
                    print(f"[WARNING] combined 렌더 실패 {rec}/{f['path'].name}: {exc}")
            # generic 가드 S: held-out frame 에 all-crops consensus free-best (E 와 동일 tpl).
            try:
                _s, ch, _o, _xy, _sc = _score(all_cons_tpl, gray)
                s_guard.append(float(ch))
            except Exception:
                pass
        if n == 0:
            continue
        # generic 가드 E: 동일 all-crops consensus 로 free-best (S 와 비교 가능).
        for ep in data.get("e_paths", [])[:AB_E_SAMPLE]:
            try:
                eg = load_gray(ep)
                _s, ch, _o, _xy, _sc = _score(all_cons_tpl, eg)
                e_guard.append(float(ch))
            except Exception:
                continue
        per_recipe.append({
            "recipe": rec, "modality": mod, "n_S_loo": n,
            # consensus 풀 크기(= "consensus 많을수록" 축): history=과거 S 장수, LOO=fm-1.
            "cons_pool_n": len(crops) if use_history else max(0, len(fm) - 1),
            "mode": "history" if use_history else "loo",
            "rcp_template": rcp_tpl is not None,
            "rcp_in_topk_rate": round(rcp_hit / n, 3),
            "cons_in_topk_rate": round(cons_hit / n, 3),
            "lift": round((cons_hit - rcp_hit) / n, 3),
            "rcp_rank1_rate": round(rcp_r1 / n, 3),
            "cons_rank1_rate": round(cons_r1 / n, 3),
        })
        tot_n += n
        tot_rcp += rcp_hit
        tot_cons += cons_hit
        tot_rcp_r1 += rcp_r1
        tot_cons_r1 += cons_r1

    if tot_n == 0:
        return None
    per_recipe.sort(key=lambda d: d["lift"], reverse=True)

    def _med(vals):
        return round(statistics.median(vals), 3) if vals else None

    cons_r1_rate = tot_cons_r1 / tot_n
    cons_topk_rate = tot_cons / tot_n
    return {
        "n_recipes": len(per_recipe),
        "n_S_loo": tot_n,
        "overall_rcp_in_topk_rate": round(tot_rcp / tot_n, 3),
        "overall_cons_in_topk_rate": round(cons_topk_rate, 3),
        "overall_lift": round((tot_cons - tot_rcp) / tot_n, 3),
        # rank-1(정밀도): free-best 가 정답에 lock. in_topk 높은데 rank1 낮으면 → 정밀도 갭.
        # (reranker[MI·contour]로 메우려 했으나 둘 다 폐기 — reranker_ab_failure_analysis.md.)
        "overall_rcp_rank1_rate": round(tot_rcp_r1 / tot_n, 3),
        "overall_cons_rank1_rate": round(cons_r1_rate, 3),
        "rank1_lift": round((tot_cons_r1 - tot_rcp_r1) / tot_n, 3),
        "cons_topk_not_rank1_rate": round(cons_topk_rate - cons_r1_rate, 3),
        # 선명도(blur) 비율: <0.70(edge) 또는 <0.50(lap) 이면 median blur 위험 → co-registration 고려.
        "cons_edge_density_ratio_to_S_median": _med(edge_ratio_s),
        "cons_lap_var_ratio_to_S_median": _med(lap_ratio_s),
        "cons_edge_density_ratio_to_rcp": _med(edge_ratio_rcp),
        "cons_lap_var_ratio_to_rcp": _med(lap_ratio_rcp),
        # miss(in_topk=False) 거리 분포 — near 多=ensemble recall 여지 / far 多=구조적 모호성(ensemble 무력).
        "cons_miss_dist_distribution": _miss_dist_distribution(cons_miss_dists),
        "rcp_miss_dist_distribution": _miss_dist_distribution(rcp_miss_dists),
        # (B) per-point (missed, peak_ratio) — 드라이버가 match-time 모호도↔miss AUC 검증에 사용.
        "consensus_points": cons_points,
        # 참고용(이 도메인에선 변별 신호 아님 — E 에도 key 가 있을 수 있어 cons_E 높음이 정상일 수 있음).
        "median_cons_free_chamfer_S": round(statistics.median(s_guard), 4) if s_guard else None,
        "median_cons_free_chamfer_E": round(statistics.median(e_guard), 4) if e_guard else None,
        "per_recipe": per_recipe,
        "min_s": min_s,
        "note": ("lift = recall(후보에 truth) 개선; rank1_lift = precision(정답 lock) 개선. "
                 "남은 정밀도 갭(topk_not_rank1)은 reranker 로 못 메움(MI·contour 폐기) → "
                 "VLM-region+CV 로 escalation. cons_S≈cons_E 는 blur 신호 아님 → 선명도 비율로 판정."),
    }


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
        print("  * in_topk 높고 rank1 낮음 = 정밀도 갭. reranker[MI·contour]로 메우려 했으나 둘 다 폐기")
        print("    (reranker_ab_failure_analysis.md) → VLM-region+CV 로 escalation.")
        print("  * in_topk 낮음 → truth 가 후보에 아예 없음 ⇒ proposer(후보 생성기) 교체 필요.")

    # 참조 staleness(상대) — rcp 가 S-consensus 의 outlier 인지. status 로 판단 가능/불가 구분.
    rq = summary.get("reference_quality", [])
    if rq:
        sc = Counter(d.get("status") for d in rq)
        incon_statuses = ("S_inconsistent", "insufficient_S", "low_texture_inconclusive")
        n_incon = sum(sc[s] for s in incon_statuses)
        print(f"\n[INFO] 참조 staleness(상대): rcp_vs_consensus / S내부일관성. "
              f"S_MI>={MIN_CONSENSUS_SELF_MI} & ratio<{RELATIVE_STALE_RATIO} & "
              f"S일관(CV<={S_INCONSISTENT_CV}) → 재등록 권장.")
        print(f"[INFO] stale={sc['stale_replace']}  ok={sc['ok']}  판단불가={n_incon}"
              f"(S부족={sc['insufficient_S']} 불일치={sc['S_inconsistent']} 저텍스처={sc['low_texture_inconclusive']})"
              f"  / scored={len(rq)} recipes")
        print(f"  {'recipe':<38} {'nS':>3} {'ratio':>6} {'S_cv':>5} {'rcp_MI':>7} {'S_MI':>6}  status")
        for d in rq[:25]:
            print(f"  {d['recipe'][:38]:<38} {d.get('n_S','-'):>3} "
                  f"{str(d.get('relative_ratio','-')):>6} {str(d.get('s_internal_cv','-')):>5} "
                  f"{str(d.get('rcp_vs_consensus_mi','-')):>7} {str(d.get('s_internal_median_mi','-')):>6}  "
                  f"{d.get('status')}")
        print("  * stale_replace = S끼리 뭉치는데 rcp만 동떨어짐(재등록 권장). "
              "low_texture_inconclusive = S-consensus MI 자체가 낮아 ratio 판정 보류.")

    xt = summary.get("gt_topk_by_reference")
    if xt:
        print("\n[INFO] GT-IN-TOPK × 참조 staleness/ratio (row-weighted S+crosshair):")
        for status, d in xt["by_status"].items():
            print(f"  status={status:<26} n={d['n']:>4} recipes={d['n_recipes']:>3} "
                  f"in_topk={d['n_in_topk']:>4}/{d['n']} ({d['in_topk_rate']}) "
                  f"miss={d['n_miss']:>4} median_ratio={d['median_ratio']}")
        tertiles = xt.get("ratio_tertiles")
        if tertiles:
            print("  ratio tertiles:")
            for name in ("low", "mid", "high"):
                d = tertiles[name]
                print(f"    {name:<4} ratio=[{d['ratio_min']}, {d['ratio_max']}] "
                      f"median={d['ratio_median']} n={d['n']} recipes={d['n_recipes']} "
                      f"in_topk={d['n_in_topk']}/{d['n']} ({d['in_topk_rate']})")
        corr = xt.get("point_biserial_in_topk_vs_ratio")
        if corr:
            print(f"  point_biserial(in_topk, ratio): n={corr['n']} r={corr['r']} "
                  f"{corr.get('note', '')}")
        print(f"  * {xt['note']}")

    # S-consensus 템플릿 A/B — rcp 대신 consensus 로 in_topk 가 뛰나 (재등록 검증/프로토타입).
    ab = summary.get("consensus_ab")
    if ab:
        print(f"\n[INFO] CONSENSUS A/B (leave-one-out, S>={ab['min_s']} recipe만, rcp 대신 S-consensus):")
        print(f"  recipes={ab['n_recipes']}  S(LOO)={ab['n_S_loo']}")
        print(f"  recall  (in_topk):  rcp={ab['overall_rcp_in_topk_rate']}  "
              f"consensus={ab['overall_cons_in_topk_rate']}  lift={ab['overall_lift']:+}")
        print(f"  precision(rank1) :  rcp={ab['overall_rcp_rank1_rate']}  "
              f"consensus={ab['overall_cons_rank1_rate']}  rank1_lift={ab['rank1_lift']:+}  "
              f"topk_not_rank1={ab['cons_topk_not_rank1_rate']}")
        print(f"  consensus 선명도 비율(blur): vs S개별 edge={ab['cons_edge_density_ratio_to_S_median']} "
              f"lap={ab['cons_lap_var_ratio_to_S_median']}  | vs rcp edge={ab['cons_edge_density_ratio_to_rcp']} "
              f"lap={ab['cons_lap_var_ratio_to_rcp']}")
        print(f"  (참고) cons free_best chamfer median S={ab['median_cons_free_chamfer_S']} "
              f"E={ab['median_cons_free_chamfer_E']}  ← 이 도메인선 변별 신호 아님")
        print("  per-recipe lift 상위:")
        for d in ab["per_recipe"][:12]:
            print(f"    {d['recipe'][:36]:<36} nS={d['n_S_loo']:>2} "
                  f"recall {d['rcp_in_topk_rate']}→{d['cons_in_topk_rate']} (lift {d['lift']:+})  "
                  f"rank1 {d['rcp_rank1_rate']}→{d['cons_rank1_rate']}")
        print(f"  * {ab['note']}")
        print("  * 판정: recall lift≥+0.10 & rank1 안 나빠짐 & 선명도비율 edge≥0.70·lap≥0.50 → 재등록 OK.")
        print("           선명도비율 edge<0.70 또는 lap<0.50 → median blur → co-registration(ECC) 후 재측정.")
        print("  * 정밀도 갭(topk_not_rank1): reranker[MI·contour] 둘 다 폐기(reranker_ab_failure_analysis.md)")
        print("    → VLM-region+CV(검색공간 축소)로 escalation. 재등록 소스는 golden 데이터로(plan §7).")
        print("  * 검증된 consensus 는 <out_dir>/consensus/ 에 저장됨 → 재등록 시 그대로 새 rcp 로 사용.")


# ====================================================================
# 엔트리.
# ====================================================================


def analyze(*, limit_per_recipe=LIMIT_PER_RECIPE) -> str:
    from poc.workflow_3.align.assets import (
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
    # {recipe: {"rcp": {mod: R_image}, "rcp_tpls": {mod: tpl}, "s_frames": [...], "e_paths": [...]}}
    crops_by_recipe: dict = {}
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
                "rcp": {m: t.raw_image for m, t in center_tpls.items() if t is not None},
                "rcp_tpls": {m: t for m, t in center_tpls.items() if t is not None},
                "s_frames": [],   # A/B LOO + staleness 의 단일 원천: {path, xy, mod, crop}.
                "e_paths": [],    # A/B E false-positive 가드용.
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
                # S-at-crosshair crop 누적 — s_frames 단일 원천(staleness 는 여기서 mod 로 group).
                if (row["label"] == "S" and xhair_crop is not None
                        and xhair_mod is not None and row.get("crosshair_xy") is not None):
                    crops_by_recipe[tag]["s_frames"].append({
                        "path": msr_path, "xy": tuple(row["crosshair_xy"]),
                        "mod": xhair_mod, "crop": xhair_crop,
                    })
                elif row["label"] == "E":
                    crops_by_recipe[tag]["e_paths"].append(msr_path)

    summary = _summarize(rows)
    summary["reference_quality"] = _reference_quality(crops_by_recipe)
    summary["gt_topk_by_reference"] = _gt_topk_reference_crosstab(rows, summary["reference_quality"])
    summary["consensus_ab"] = _consensus_template_ab(crops_by_recipe, out_dir=out_dir)
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
    from poc.workflow_3.align.matching.engine import build_template as _bt

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

    # gt-in-topK 경로 sanity — 정답(중앙 패턴 center)에서 후보 생성/순위가 도는지.
    #   패턴은 cxy=(170,120) 에 60x60 으로 박혔으니 중심 = (200, 150).
    gt = _gt_in_topk(s_frame, (170 + 30, 120 + 30), {"om": rcp_center_tpl})
    if gt is not None:
        assert "topk_rank" in gt and "in_topk" in gt, f"gt_topk key 누락: {gt}"
        print(f"[INFO] self-test gt_topk: in_topk={gt['in_topk']} rank={gt['topk_rank']}")
    else:
        print("[INFO] self-test gt_topk: 후보 없음(합성) — 경로만 무오류 확인")

    # separation 함수 sanity.
    sep = _separation([0.8, 0.75, 0.9], [0.3, 0.4, 0.2])
    assert sep["balanced_accuracy"] == 1.0, f"separation 오류: {sep}"
    assert sep["median_s"] > sep["median_e"]
    print(f"[INFO] self-test separation OK: {sep}")
    print("[INFO] self-test 통과." if ok else "[ERROR] self-test 실패.")
    return ok


def run() -> str:
    try:
        from poc.workflow_3.align.assets import iter_recipe_dirs
        has_data = bool(iter_recipe_dirs())
    except Exception:
        has_data = False
    if has_data:
        return analyze()
    print("[WARNING] align_images 데이터 없음 — 합성 self-test 로 대체합니다.\n")
    return "success" if _self_test() else "selftest_failed"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
