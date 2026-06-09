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

import math

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
