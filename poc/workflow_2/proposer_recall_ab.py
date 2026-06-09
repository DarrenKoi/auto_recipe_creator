"""proposer recall A/B — baseline(C1 chamfer) vs ensemble 의 후보 recall@{8,16,24}.

proposer recall 만 격리: 후보 xy+align_offset 이 GT(cond crosshair) 허용오차 내인지(=membership)만
본다. final score/decision·reranker 재정렬 금지(proposer/reranker 섞임 방지). modality 라우팅·
box template·cond GT 는 golden_localization_eval_cond 재사용. 설계: docs/specs/2026-06-09-...md.
실행(오피스): uv run python poc/workflow_2/proposer_recall_ab.py
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
import math
from pathlib import Path

import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_2.align_key_matcher import (
    _collect_candidates, _to_grayscale, preprocess_for_matching,
)
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import load_cond
from poc.workflow_2.ensemble_proposer import _Cand, compute_ensemble_candidates
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_1.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "proposer_recall_ab"

RECALL_NS = (8, 16, 24)


def _gt_rank(cands, *, gt_xy, offset, short, tol):
    """후보 리스트(정렬됨)에서 (xy+offset) 이 GT 허용오차(tol·short) 내인 첫 1-base rank. 없으면 None."""
    dx, dy = offset
    lim = tol * short
    for i, c in enumerate(cands, 1):
        ax, ay = c.xy[0] + dx, c.xy[1] + dy
        if math.hypot(ax - gt_xy[0], ay - gt_xy[1]) <= lim:
            return i
    return None


def _recall_at(ranks, n):
    """rank 리스트(None=miss)에서 rank<=n 비율."""
    if not ranks:
        return 0.0
    return round(sum(1 for r in ranks if r is not None and r <= n) / len(ranks), 3)


def _baseline_candidates(template_gray, frame_gray, *, top_k=24):
    """C1-only(canny) 후보 top_k — ensemble 비교 기준."""
    t_edges, _ = preprocess_for_matching(_to_grayscale(template_gray))
    _, f_dt = preprocess_for_matching(_to_grayscale(frame_gray))
    cands = _collect_candidates(t_edges, f_dt, scales=COMPARE_SCALES, top_n=top_k)
    return [_Cand(xy=c.xy, score=c.chamfer_score, scale=c.scale) for c in cands]


def run():
    """baseline vs ensemble proposer recall A/B (인자 없음). 반환 success|no_data."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터 없음: {root}")
        return "no_data"
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ranks, ens_ranks = [], []
    solo_ranks = {"canny": [], "scharr": [], "orient": []}
    for assets in recipes:
        if assets is None:
            continue
        try:
            _center, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        available = {m for m, v in box_tpls.items() if v is not None}
        if not available:
            continue
        for p in iter_msr_images(assets):
            if _tool_label(p.name) != "S":
                continue
            cond = load_cond(p)
            routed = glec._route_modality(cond, available)
            if routed is None or box_tpls.get(routed) is None:
                continue
            tpl, (dx, dy) = box_tpls[routed]
            if not (cond and cond.crosshair_xy is not None):
                continue
            gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
            gt_xy = (int(round(gx)), int(round(gy)))
            try:
                gray_raw = load_gray(p)
            except Exception:
                continue
            frame = clean_image(gray_raw, cond)        # crosshair 제거(box__inpaint 경로와 동일).
            t_gray = tpl.raw_image
            short = max(1, min(t_gray.shape[0], t_gray.shape[1]))
            base = _baseline_candidates(t_gray, frame)
            # ensemble 도 baseline 과 동일 scale 밴드(COMPARE_SCALES) — 차이는 오직 C2·C3 추가.
            ens = compute_ensemble_candidates(t_gray, frame, scales=COMPARE_SCALES)
            base_ranks.append(_gt_rank(base, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))
            ens_ranks.append(_gt_rank(ens.fused, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))
            for ch, lst in ens.solo.items():
                solo_ranks[ch].append(_gt_rank(lst, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))

    n = len(base_ranks)
    if not n:
        print("[ERROR] 처리된 S 프레임 없음.")
        return "no_data"
    summary = {"n": n, "GT_TOL_NORM": GT_TOL_NORM,
               "baseline": {f"recall@{k}": _recall_at(base_ranks, k) for k in RECALL_NS},
               "ensemble": {f"recall@{k}": _recall_at(ens_ranks, k) for k in RECALL_NS},
               "solo": {ch: {f"recall@{k}": _recall_at(r, k) for k in RECALL_NS}
                        for ch, r in solo_ranks.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n[INFO] === proposer recall A/B (S {n}장, tol={GT_TOL_NORM}) ===")
    print(f"  {'variant':<12} {'recall@8':>9} {'recall@16':>10} {'recall@24':>10}")
    for name, d in (("baseline(C1)", summary["baseline"]), ("ensemble", summary["ensemble"])):
        print(f"  {name:<12} {d['recall@8']:>9} {d['recall@16']:>10} {d['recall@24']:>10}")
    for ch, d in summary["solo"].items():
        print(f"  solo:{ch:<7} {d['recall@8']:>9} {d['recall@16']:>10} {d['recall@24']:>10}")
    lift8 = round(summary["ensemble"]["recall@8"] - summary["baseline"]["recall@8"], 3)
    lift24 = round(summary["ensemble"]["recall@24"] - summary["baseline"]["recall@24"], 3)
    print(f"\n  >>> budgeted recall@8 lift = {lift8:+}  | shadow recall@24 lift = {lift24:+}")
    print("      @24↑·@8↑ = ensemble 효과 / @24↑·@8≈ = fusion 부족 / @24≈ = 채널 무효")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
