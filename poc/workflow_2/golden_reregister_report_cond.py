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
