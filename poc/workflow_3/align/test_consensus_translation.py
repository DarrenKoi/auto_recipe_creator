"""등록 box가 보이면 화면 위치와 무관하게 같은 align point를 찾아야 한다."""

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from poc.workflow_3.align.assets import AlignFailAssets
from poc.workflow_3.align import grid_search
from poc.workflow_3.align.consensus_crops import _cond_consensus_crop
from poc.workflow_3.align.cond_file import CondInfo
from poc.workflow_3.align.correction import GATE_ACT, key_visibility_gate
from poc.workflow_3.align.consensus_resolve import resolve_templates
from poc.workflow_3.align.matching.engine import (
    DEFAULT_SCALES, STRUCTURE_POLICY, compute_align_key_score_ensemble,
)


def _save_image(path, image, *, crosshair, box=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    sidecar = path.parent / f".{path.name}"
    sidecar.mkdir()
    coords = [0, 0, 0, 0, *crosshair, *(box or (-1, -1, -1, -1))]
    (sidecar / "cond.txt").write_text(
        "Scope SEM\nPixel 320,320\nMagnification 30000\n!Cursor_info "
        + ",".join(str(v) for v in coords), encoding="utf-8")


def _resolve_scene(tmp_path):
    rng = np.random.default_rng(23)
    source = cv2.resize(rng.integers(20, 230, (80, 80), dtype=np.uint8),
                        (320, 320), interpolation=cv2.INTER_NEAREST)
    rcp = tmp_path / "rcp" / "IMAP0002.png"
    # box中心(140,140)、align point(160,160): offset=(20,20)。
    _save_image(rcp, source, crosshair=(1600, 1600), box=(1180, 1180, 1620, 1620))
    cache = tmp_path / "cache"
    for i, (dx, dy) in enumerate(((0, 0), (16, -12), (-20, 20))):
        success = cv2.warpAffine(source, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (320, 320), borderValue=120)
        _save_image(cache / "c/r/events" / str(i) / "S1.png", success,
                    crosshair=((160 + dx) * 10, (160 + dy) * 10))
    assets = AlignFailAssets("E1", "c", "r", tmp_path, None, rcp, None, ())
    template = resolve_templates(
        assets, eqp_id="E1", consensus_enabled=True, min_s=3, max_events=4,
        sync_timeout_sec=0, cond_box_crop=True, cache_root=cache)["SEM"]
    return source, template


@pytest.mark.parametrize("key_xy", [(140, 140), (30, 140), (285, 140), (140, 30), (140, 285)])
def test_consensus_localizes_visible_box_away_from_frame_center(tmp_path, key_xy):
    source, template = _resolve_scene(tmp_path)
    dx, dy = key_xy[0] - 140, key_xy[1] - 140
    frame = cv2.warpAffine(source, np.float32([[1, 0, dx], [0, 1, dy]]),
                           (320, 320), borderValue=120)
    result = compute_align_key_score_ensemble(
        template, frame, scales=DEFAULT_SCALES, policy=STRUCTURE_POLICY)
    align_xy = tuple(result.best_xy[i] + round(template.align_offset_xy[i] * result.best_scale)
                     for i in (0, 1))
    assert result.decision == "match", (result.score, result.best_xy, template.raw_image.shape)
    assert key_visibility_gate(result, reregister_ratio_threshold=0.98) == GATE_ACT
    assert align_xy == pytest.approx((160 + dx, 160 + dy), abs=3)
    assert template.raw_image.shape == (40, 40)
    assert template.align_offset_xy == (20, 20)
    assert template.source_wh == (320, 320)
    assert template.source_magnification == 30000
    controller = SimpleNamespace(
        capture=lambda: frame, read_mode=lambda: "SEM",
        move_to_point=lambda *a: pytest.fail("이미 보이는 key를 지나쳐 이동함"))
    search = grid_search.grid_align_search(
        controller, {"SEM": template},
        grid_search.MagnificationControl(lambda: [30000], lambda target: target),
        reg_mag=30000, shift_fn=None)
    assert search.status == "match"
    assert search.best.fov_xy == pytest.approx((160 + dx, 160 + dy), abs=3)


@pytest.mark.parametrize("xy", [(10, 100), (195, 100), (100, 10), (100, 195)])
def test_consensus_drops_partial_success_crop_instead_of_shifting_or_stretching(xy):
    gray = np.zeros((200, 200), dtype=np.uint8)
    cond = CondInfo(pixel=(200, 200), crosshair_xy=tuple(v * 10 for v in xy))
    assert _cond_consensus_crop(gray, cond, (40, 40)) is None


def test_box_consensus_does_not_match_blank_frame(tmp_path):
    _, template = _resolve_scene(tmp_path)
    result = compute_align_key_score_ensemble(
        template, np.full((320, 320), 120, dtype=np.uint8),
        scales=DEFAULT_SCALES, policy=STRUCTURE_POLICY)
    assert result.decision == "low"
