"""Align key matching public surface."""

from poc.workflow_3.align.matching.engine import (
    BROAD_SCALES,
    DEFAULT_POLICY,
    DEFAULT_SCALES,
    STRUCTURE_POLICY,
    AlignKeyCandidate,
    AlignKeyMatchResult,
    AlignKeyTemplate,
    MatchPolicy,
    build_template,
    compute_align_key_score,
    compute_align_key_score_ensemble,
    compute_chamfer_candidates,
    compute_chamfer_score,
    compute_orb_inlier_ratio,
    preprocess_for_matching,
    save_overlay_jpeg,
)

__all__ = [
    "BROAD_SCALES",
    "DEFAULT_POLICY",
    "DEFAULT_SCALES",
    "STRUCTURE_POLICY",
    "AlignKeyCandidate",
    "AlignKeyMatchResult",
    "AlignKeyTemplate",
    "MatchPolicy",
    "build_template",
    "compute_align_key_score",
    "compute_align_key_score_ensemble",
    "compute_chamfer_candidates",
    "compute_chamfer_score",
    "compute_orb_inlier_ratio",
    "preprocess_for_matching",
    "save_overlay_jpeg",
]
