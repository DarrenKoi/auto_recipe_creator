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
