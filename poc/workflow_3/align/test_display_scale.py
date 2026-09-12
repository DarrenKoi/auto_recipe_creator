"""저장 FOV → live box 표시 배율 회귀. 장비/VLM 없이 실제 matcher 를 실행한다."""

import cv2
import numpy as np
import pytest

from poc.workflow_3.align import grid_search as gs
from poc.workflow_3.align.correction import CorrectionConfig, correct_align_fail
from poc.workflow_3.align.matching.engine import build_template, compute_align_key_score_ensemble
from poc.workflow_3.align.test_correction import _FakeController
from poc.workflow_3.align import templates, consensus_crops
from poc.workflow_3.align.cond_file import CondInfo
from types import SimpleNamespace
import json


def _scene(width=512):
    raw = np.zeros((512, 512), np.uint8)
    rng = np.random.default_rng(42)
    key = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    key = cv2.GaussianBlur(key, (3, 3), 0)
    raw[192:320, 192:320] = key
    template = build_template(key, recipe_id="c/r", version="test", key_type="sem",
                              align_offset_xy=(16, -8), source_wh=(512, 512))
    return template, cv2.resize(raw, (width, width), interpolation=cv2.INTER_LINEAR)


@pytest.mark.parametrize("width", [256, 400, 512, 1024])
def test_grid_crop_keeps_source_fov_ratio(width):
    template, frame = _scene(width)
    normalized = gs.normalize_template(template, width)
    assert normalized.raw_image.shape[1] == round(128 * width / 512)
    result = compute_align_key_score_ensemble(normalized, frame, scales=(1.0,))
    assert result.decision == "match"
    assert np.linalg.norm(np.subtract(result.best_xy, (width / 2, width / 2))) <= 3
    assert normalized.align_offset_xy == (round(16 * width / 512), round(-8 * width / 512))


@pytest.mark.parametrize("width", [256, 512, 1024])
def test_primary_display_scale_and_offset(width):
    template, frame = _scene(width)
    controller = _FakeController(frame, frame)
    result = correct_align_fail(
        controller, {"SEM": template}, dry_run=False,
        config=CorrectionConfig(ok_click_enabled=False, fallback_search_enabled=False),
    )
    assert result.status == "awaiting_engineer_ok"
    expected = (width // 2 + round(16 * width / 512), width // 2 - round(8 * width / 512))
    assert np.linalg.norm(np.subtract(controller.move_calls[-1], expected)) <= 3
    assert result.history[0]["best_scale"] == pytest.approx(width / 512)


def test_grid_missing_source_degrades_before_magnification_action():
    template, frame = _scene()
    template.source_wh = None
    controller = _FakeController(frame, frame)
    calls = []
    mag = gs.MagnificationControl(lambda: calls.append("options") or [10000, 30000],
                                  lambda m: calls.append(m) or m)
    result = gs.grid_align_search(controller, {"SEM": template}, mag, reg_mag=30000)
    assert result.status == "degraded"
    assert result.meta["reason"] == "missing_source_geometry"
    assert not calls and not controller.move_calls


def test_template_loader_preserves_full_fov_not_crop(monkeypatch):
    monkeypatch.setattr(templates, "load_gray", lambda p: np.zeros((512, 512), np.uint8))
    monkeypatch.setattr(templates, "load_cond", lambda p: CondInfo(
        pixel=(512, 512), box_ltrb=(1920, 1920, 3200, 3200), raw={"magnification": ["30000"]}))
    template = templates.load_template("unused", recipe_id="c/r", key_type="sem", cond_box_crop=True)
    assert template.source_wh == (512, 512)
    assert template.source_magnification == 30000
    assert template.raw_image.shape[1] < 512


def test_consensus_sizing_matches_golden_center_crop(monkeypatch):
    monkeypatch.setattr(templates, "load_gray", lambda p: np.zeros((512, 512), np.uint8))
    monkeypatch.setattr(templates, "load_cond", lambda p: None)
    assets = SimpleNamespace(recipe_id="c/r", recipe_sem="unused", recipe_om=None)
    template = consensus_crops.build_center_tpls_for_sizing(assets)["sem"][0]
    assert template.raw_image.shape == (198, 198)
    assert template.source_wh == (512, 512)


def test_grid_zoom_budget_uses_crop_size():
    from poc.workflow_3.align.test_grid_search import _low
    template, frame = _scene()
    controller = _FakeController(frame, frame)
    calls = []
    mag = gs.MagnificationControl(lambda: [5000, 10000, 20000, 30000],
                                  lambda m: calls.append(m) or m)
    gs.grid_align_search(controller, {"SEM": template}, mag, reg_mag=30000,
                         config=gs.GridSearchConfig(pan_budget=0), match_fn=_low)
    assert calls[0] == 20000  # 128px key: 10K에서는 43px, 20K에서는 85px.


@pytest.mark.parametrize("readback", [20000, None])
def test_grid_without_zoom_out_uses_readback_or_stops(readback):
    from poc.workflow_3.align.test_grid_search import _low
    template, frame = _scene()
    template = build_template(template.raw_image[:64, :64], recipe_id="c/r", version="test",
                              source_wh=(512, 512))
    controller = _FakeController(frame, frame)
    scales = []
    calls = []

    def set_mag(value):
        calls.append(value)
        return readback

    def match(*args, **kwargs):
        scales.append(kwargs["scales"])
        return _low(*args, **kwargs)

    result = gs.grid_align_search(
        controller, {"SEM": template},
        gs.MagnificationControl(lambda: [20000, 40000], set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=0), match_fn=match,
    )
    assert calls == [20000]
    if readback is None:
        assert result.status == "degraded"
        assert result.meta["reason"] == "mag_unreadable"
        assert not scales and not controller.move_calls
    else:
        assert result.meta["search_mag"] == readback
        assert scales == [(readback / 30000,)]


def test_grid_actual_crop_confirms_at_registered_scale_with_offset():
    from poc.workflow_3.align.test_grid_search import _WaferCtl, _wafer_with_key
    start, target = (2304, 1728), (2904, 1808)
    wafer, full_templates = _wafer_with_key(target)
    raw = full_templates["SEM"].raw_image
    template = build_template(raw[104:296, 112:368], recipe_id="c/r", version="test",
                              key_type="sem", source_wh=(512, 384), align_offset_xy=(16, -8))
    controller = _WaferCtl(wafer, start, reg_mag=30000)
    result = gs.grid_align_search(
        controller, {"SEM": template},
        gs.MagnificationControl(lambda: [10000, 30000], controller.set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=2, radius_um=10),
    )
    assert result.status == "match", (result.meta, result.history)
    assert result.meta["restore_mag"] == 30000
    # coarse recenter의 오차는 confirm 좌표로 남는다. 최종 좌표는 crop 중심이 아닌 align point.
    projected_target = np.add(controller.pos, np.subtract(result.best.fov_xy, (256, 192)))
    assert np.linalg.norm(np.subtract(projected_target, target)) <= 3


def test_incompatible_s_geometry_or_magnification_dropped(tmp_path, monkeypatch, capsys):
    from poc.workflow_3.align.test_consensus_crops import _events, _tpl
    cc = consensus_crops
    _events(tmp_path, 1)
    template = _tpl(40, 32)
    template.source_wh = (200, 200)
    template.source_magnification = 30000
    monkeypatch.setattr(cc, "msr_modality", lambda cond: "sem")
    monkeypatch.setattr(cc, "load_cond", lambda p: CondInfo(
        pixel=(200, 200), crosshair_xy=(1000, 1000), raw={"magnification": ["10000"]}))
    monkeypatch.setattr(cc, "load_gray", lambda p: np.zeros((200, 200), np.uint8))
    out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", {"sem": (template, (0, 0))}, max_events=4)
    assert out.get("sem", []) == []
    assert "magnification_mismatch" in capsys.readouterr().out
    monkeypatch.setattr(cc, "load_cond", lambda p: CondInfo(
        pixel=(400, 400), crosshair_xy=(2000, 2000), raw={"magnification": ["30000"]}))
    monkeypatch.setattr(cc, "load_gray", lambda p: np.zeros((400, 400), np.uint8))
    out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", {"sem": (template, (0, 0))}, max_events=4)
    assert out.get("sem", []) == []
    assert "source_size_mismatch" in capsys.readouterr().out


def test_primary_invalid_geometry_escalates_without_actions():
    template, _ = _scene()
    frame = np.zeros((480, 640), np.uint8)
    controller = _FakeController(frame, frame)
    result = correct_align_fail(controller, {"SEM": template}, dry_run=False)
    assert result.status == "escalated_invalid_geometry"
    assert result.history[0]["reason"] == "invalid_source_geometry"
    assert not controller.move_calls and not controller.zoom_calls and not controller.screen_clicks


def test_grid_invalid_geometry_does_not_zoom_in_legacy_degrade():
    template, _ = _scene()
    frame = np.zeros((480, 640), np.uint8)
    controller = _FakeController(frame, frame)
    result = gs.search_around(controller, {"SEM": template}, reg_mag=30000,
                              grid_mag=gs.MagnificationControl(lambda: [10000, 30000], lambda m: m))
    assert result.status == "escalated"
    assert result.meta["reason"] == "invalid_source_geometry"
    assert not controller.move_calls and not controller.zoom_calls


def test_feasibility_invalid_geometry_records_no_coordinate(tmp_path, monkeypatch):
    from poc.workflow_3.align.diagnostics import feasibility_check as fc
    template, _ = _scene()
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), np.zeros((480, 640), np.uint8))
    monkeypatch.setattr(fc, "count_staged_events", lambda *a: (0, 0))
    monkeypatch.setattr(fc, "resolve_assets_auto", lambda *a, **k: object())
    monkeypatch.setattr(fc, "build_templates_from_assets", lambda *a, **k: {"SEM": template})
    detection = SimpleNamespace(pm_text="30000", pm_mode="SEM", pm_text_source="test", pm_box_px=None,
                                detected=True, bbox_px={"left": 0, "top": 0, "right": 640, "bottom": 480})
    monkeypatch.setattr(fc, "_maybe_detect_sem_box", lambda *a, **k: detection)
    result = fc.mark_align_feasibility(path, eqp_id="E1", recipe_id="c/r")
    assert result.verdict == "invalid_geometry"
    assert result.align_xy is None and result.match_xy is None
    assert json.loads(result.json_path.read_text())["geometry_errors"]["SEM"]
