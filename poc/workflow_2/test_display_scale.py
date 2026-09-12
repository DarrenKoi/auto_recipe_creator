"""golden proposer 가 workflow_3 와 같은 저장 FOV/표시 scale 계약을 쓰는지 검증."""

import numpy as np
import pytest

from poc.workflow_2.align_similarity import _propose_topk
from poc.workflow_3.align.matching.engine import preprocess_for_matching
from poc.workflow_3.align.test_display_scale import _scene
from poc.workflow_2 import golden_localization_eval as gle
from poc.workflow_3.align.matching.engine import AlignKeyCandidate, AlignKeyMatchResult
from pathlib import Path
from types import SimpleNamespace


def test_golden_proposer_display_scale():
    template, frame = _scene(1024)
    _, frame_dt = preprocess_for_matching(frame)
    candidates = _propose_topk(template, frame, frame_dt, scales=(1.0,), topk=8)
    assert candidates
    assert all(c.scale == 2.0 for c in candidates)
    assert any(np.linalg.norm(np.subtract(c.xy, (512, 512))) <= 3 for c in candidates)


def test_golden_localization_scales_offset_once(monkeypatch):
    template, frame = _scene(1024)
    def matcher(tpl, frame, **kwargs):
        scale = max(kwargs["scales"])
        return AlignKeyMatchResult(
            score=0.9, chamfer_score=0.9, orb_inlier_ratio=0, best_xy=(512, 512),
            best_scale=scale, decision="match", debug_overlay=frame,
            candidates=[AlignKeyCandidate(0.9, 0.9, (512, 512), scale, (256, 256))],
        )
    monkeypatch.setattr(gle, "_matcher_for_eval", lambda: matcher)
    result = gle._localize({"sem": (template, (16, -8))}, frame, (544, 496))
    assert result["align_xy"] == [544, 496]
    assert result["dist_norm"] == 0.0


def test_static_diagnostics_scale_roi_and_consensus_crop(monkeypatch):
    from poc.workflow_2 import align_similarity as sim
    from poc.workflow_3.align import assets
    from poc.workflow_3.align.diagnostics import crosshair_detect
    template, frame = _scene(1024)
    monkeypatch.setattr(assets, "load_gray", lambda p: frame)
    monkeypatch.setattr(crosshair_detect, "detect_crosshair", lambda g: SimpleNamespace(
        xy=(512, 512), confidence=1.0, debug={}))
    row, crop, mod = sim._process_msr(Path("S1.jpeg"), center_tpls={"sem": template}, box_tpls={})
    assert row["at_center"] is not None
    assert crop is not None and mod == "sem"
    assert row["ncc_xhair"] > 0.95
    assert row["truth"]["valid"]


@pytest.mark.parametrize("incompatible_first", [True, False])
def test_static_diagnostics_skip_only_incompatible_modality(monkeypatch, incompatible_first):
    from poc.workflow_2 import align_similarity as sim
    from poc.workflow_3.align import assets
    from poc.workflow_3.align.diagnostics import crosshair_detect
    sem, frame = _scene()
    om, _ = _scene()
    om.source_wh = (512, 384)
    templates = {"om": om, "sem": sem} if incompatible_first else {"sem": sem, "om": om}
    monkeypatch.setattr(assets, "load_gray", lambda p: frame)
    monkeypatch.setattr(crosshair_detect, "detect_crosshair", lambda g: SimpleNamespace(
        xy=(256, 256), confidence=1.0, debug={}))
    row, crop, mod = sim._process_msr(Path("S1.jpeg"), center_tpls=templates, box_tpls={})
    assert row["at_center"] is not None
    assert crop is not None and mod == "sem"
    assert row["truth"]["modality"] == "sem" and row["truth"]["valid"]
