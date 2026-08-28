"""grid_search — 절대 배율 zoom-out + FOV 격자 sweep + odometry 단위 테스트 (Mac, 장비 없음).

설계: docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md
`uv run pytest poc/workflow_3/align/test_grid_search.py`
"""

import pytest

from poc.workflow_3.align import grid_search as gs

CG_OPTIONS = [1000, 2000, 5000, 8000, 10000, 20000, 50000, 100000]


# ------------------------------------------------------------------
# §1 zoom-out 단 선택 - 최소 key 픽셀 기준 (고정 scale 0.15 가 아니다)
# ------------------------------------------------------------------


def test_zoom_out_picks_lowest_option_keeping_min_key_px():
    """30K 등록 / fw=512 / 최소 60px -> 5K (key 85px). 2K 는 34px 라 탈락."""
    assert gs.choose_zoom_out_mag(CG_OPTIONS, reg_mag=30000, fw=512, min_key_px=60) == 5000


def test_zoom_out_depends_on_runtime_frame_width():
    """같은 30K 라도 fw=320 이면 5K key 가 53px 로 무너져 8K 로 밀린다(검토 반론 2)."""
    assert gs.choose_zoom_out_mag(CG_OPTIONS, reg_mag=30000, fw=320, min_key_px=60) == 8000


def test_zoom_out_50k_registered_goes_to_8k():
    assert gs.choose_zoom_out_mag(CG_OPTIONS, reg_mag=50000, fw=512, min_key_px=60) == 8000


def test_zoom_out_returns_none_when_no_option_is_lower_than_registered():
    """만족하는 가장 낮은 단이 등록 배율 이상이면 zoom-out 이 아니다 -> None(현재 배율 유지)."""
    assert gs.choose_zoom_out_mag([50000, 100000], reg_mag=50000, fw=512, min_key_px=60) is None
    assert gs.choose_zoom_out_mag([], reg_mag=30000, fw=512, min_key_px=60) is None


# ------------------------------------------------------------------
# §2 격자 계획 - 2R 박스를 덮는 홀수 n×n, spiral 순서, 예산 절단
# ------------------------------------------------------------------


def test_plan_grid_3x3_covers_60um_at_5k():
    """FOV 27µm / R 30 -> ceil(60/27)=3 -> 착지 셀 제외 8 셀, spiral 첫 걸음은 (+1, 0)."""
    cells = gs.plan_grid(fov_um=27.0, radius_um=30.0, budget=10)
    assert len(cells) == 8
    assert cells[0] == (1, 0)
    assert (0, 0) not in cells
    assert max(abs(c) for cell in cells for c in cell) == 1


def test_plan_grid_rounds_even_up_to_odd_and_truncates_to_budget():
    """FOV 16.9 -> ceil(60/16.9)=4 -> 홀수 5 -> 24 셀이지만 예산 10 이면 안쪽 spiral 10 셀만."""
    cells = gs.plan_grid(fov_um=16.9, radius_um=30.0, budget=10)
    assert len(cells) == 10
    # 안쪽 링(8 셀)이 먼저 다 나온 뒤 바깥 링으로 나간다.
    assert all(max(abs(c) for c in cell) == 1 for cell in cells[:8])


def test_plan_grid_is_empty_when_fov_already_covers_radius():
    assert gs.plan_grid(fov_um=70.0, radius_um=30.0, budget=10) == []


# ------------------------------------------------------------------
# §3 odometry - 측정값은 게이트를 지나야 누적된다
# ------------------------------------------------------------------


def test_odometer_uses_measured_shift_inside_tolerance():
    odo = gs.Odometer(fov_px=100, tol_fov=0.15)
    used = odo.record(commanded=(38.0, 0.0), measured=(35.0, 1.0))
    assert used == (35.0, 1.0)
    assert odo.position == (35.0, 1.0)
    assert odo.drift_flags == 0


def test_odometer_falls_back_to_commanded_when_measured_is_off_by_a_period():
    """주기 구조에서 한 주기 어긋난 값은 높은 cc 로 나온다(검토 반론 4) -> 명령값 + flag."""
    odo = gs.Odometer(fov_px=100, tol_fov=0.15)
    used = odo.record(commanded=(38.0, 0.0), measured=(70.0, 0.0))
    assert used == (38.0, 0.0)
    assert odo.drift_flags == 1


def test_odometer_falls_back_when_measurement_is_unavailable():
    odo = gs.Odometer(fov_px=100, tol_fov=0.15)
    odo.record(commanded=(38.0, 0.0), measured=None)
    odo.record(commanded=(0.0, -38.0), measured=None)
    assert odo.position == (38.0, -38.0)
    assert odo.drift_flags == 0


# ------------------------------------------------------------------
# §2-§5 sweep 오케스트레이션 - mock controller 로 stage/배율 호출 순서를 관찰한다
# ------------------------------------------------------------------

import numpy as np

from poc.workflow_3.align.matching.engine import AlignKeyMatchResult, build_template


class _Ctl:
    """제스처를 기록만 하는 컨트롤러. frames 로 capture 결과를 흉내낸다."""

    def __init__(self, fw=512, fh=384, mode="SEM"):
        self.fw, self.fh, self.mode = fw, fh, mode
        self.calls: list[tuple] = []

    def capture(self):
        return np.zeros((self.fh, self.fw), dtype=np.uint8)

    def read_mode(self):
        return self.mode

    def move_to_point(self, x, y):
        self.calls.append(("move", int(x), int(y)))

    def zoom(self, d):
        self.calls.append(("zoom", d))

    def capture_screen(self):
        return self.capture()

    def click_screen(self, x, y):
        self.calls.append(("click", x, y))


class _Mag:
    """드롭다운 주입점: options_fn + set_fn(target)->판독값. 호출을 기록한다."""

    def __init__(self, options, reg_mag, readback=None):
        self.options, self.reg_mag = list(options), reg_mag
        self.set_calls: list[float] = []
        self.list_calls = 0
        self._readback = readback  # None 이면 target 을 그대로 판독값으로.

    def options_fn(self):
        self.list_calls += 1
        return list(self.options)

    def set_fn(self, target):
        self.set_calls.append(target)
        return target if self._readback is None else self._readback(target)

    def control(self):
        return gs.MagnificationControl(self.options_fn, self.set_fn)


def _tpl(w=512, h=384):
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    return {"SEM": build_template(raw, recipe_id="c/r", version="v", key_type="sem")}


def _low(template, frame, **kw):
    fh, fw = frame.shape[:2]
    return AlignKeyMatchResult(
        score=0.0, chamfer_score=0.0, orb_inlier_ratio=0.0, best_xy=(fw // 2, fh // 2),
        best_scale=1.0, decision="low", debug_overlay=frame,
    )


def _moves(ctl):
    return [c for c in ctl.calls if c[0] == "move"]


def test_sweep_zooms_out_visits_every_cell_then_returns_home_and_restores_mag():
    """blank 프레임: 5K 로 내려 3×3 의 8 셀(셀당 3 클릭)을 돌고, 원점 복귀(3 클릭) 후
    등록 배율 최근접 단으로 복귀, status exhausted. sweep 중 추격 0회."""
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = gs.grid_align_search(
        ctl, _tpl(), mag.control(), reg_mag=30000,
        config=gs.GridSearchConfig(radius_um=30.0, min_key_px=60, pan_budget=10),
        match_fn=_low,
    )
    assert out.status == "exhausted"
    assert mag.set_calls[0] == 5000
    assert mag.set_calls[-1] == 20000  # 30K 는 단에 없다 -> 최근접(20K/50K 동률이면 낮은 쪽)
    assert len(_moves(ctl)) == 8 * 3 + 3
    assert out.meta["cells_visited"] == 8
    assert out.meta["search_mag"] == 5000
    assert abs(out.meta["final_position_px"][0]) < 1 and abs(out.meta["final_position_px"][1]) < 1
    assert out.pan_count == 8


def test_sweep_never_clicks_outside_margin():
    ctl = _Ctl(fw=512, fh=384)
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    gs.grid_align_search(
        ctl, _tpl(), mag.control(), reg_mag=30000,
        config=gs.GridSearchConfig(click_margin_ratio=0.12), match_fn=_low,
    )
    for _, x, y in _moves(ctl):
        assert 61 <= x <= 451 and 46 <= y <= 338


def test_degrades_when_zoom_out_readback_fails():
    """드롭다운 선택 후 PM OCR 판독이 None 이면 배율을 모른 채 격자를 계산하지 않는다(계약 1)."""
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000, readback=lambda t: None)
    out = gs.grid_align_search(
        ctl, _tpl(), mag.control(), reg_mag=30000,
        match_fn=_low,
    )
    assert out.status == "degraded"
    assert out.meta["reason"] == "mag_unreadable"
    assert _moves(ctl) == []


def test_om_mode_skips_zoom_out_and_pans_budget_cells():
    """OM 은 key 가 작아 zoom-out 불가(§6) -> 현재 배율에서 예산만큼 spiral."""
    ctl = _Ctl(mode="OM")
    mag = _Mag(CG_OPTIONS, reg_mag=104)
    tpl = _tpl()
    tpl["OM"] = tpl.pop("SEM")
    out = gs.grid_align_search(
        ctl, tpl, mag.control(), reg_mag=104,
        config=gs.GridSearchConfig(pan_budget=4), match_fn=_low,
    )
    assert mag.set_calls == []
    assert out.meta["cells_visited"] == 4
    assert out.status == "exhausted"


# ------------------------------------------------------------------
# §4 추격·confirm - 가상 wafer + 실제 matcher. §3 drift 게이트. abort 래치.
# ------------------------------------------------------------------

import cv2

from poc.workflow_3.util.abort_switch import SWITCH


@pytest.fixture(autouse=True)
def _reset_abort():
    SWITCH.reset()
    yield
    SWITCH.reset()


class _WaferCtl:
    """등록 배율 프레임 = (fw×fh) wafer px. 배율을 내리면 창이 reg/cur 배 넓어진다."""

    def __init__(self, wafer, start, reg_mag, fw=512, fh=384):
        self.wafer, self.pos, self.reg, self.cur = wafer, list(start), reg_mag, reg_mag
        self.fw, self.fh = fw, fh
        self.moves = 0

    def _f(self):
        return self.reg / self.cur

    def capture(self):
        f = self._f()
        w, h = int(self.fw * f), int(self.fh * f)
        x0, y0 = int(self.pos[0] - w / 2), int(self.pos[1] - h / 2)
        crop = np.full((h, w), 100, dtype=np.uint8)
        wx0, wy0 = max(0, x0), max(0, y0)
        wx1, wy1 = min(self.wafer.shape[1], x0 + w), min(self.wafer.shape[0], y0 + h)
        if wx1 > wx0 and wy1 > wy0:
            crop[wy0 - y0:wy1 - y0, wx0 - x0:wx1 - x0] = self.wafer[wy0:wy1, wx0:wx1]
        return cv2.resize(crop, (self.fw, self.fh), interpolation=cv2.INTER_AREA)

    def move_to_point(self, x, y):
        f = self._f()
        self.pos[0] += (x - self.fw / 2) * f
        self.pos[1] += (y - self.fh / 2) * f
        self.moves += 1

    def set_mag(self, target):
        self.cur = float(target)
        return self.cur

    def read_mode(self):
        return "SEM"

    def zoom(self, d):
        pass

    def capture_screen(self):
        return self.capture()

    def click_screen(self, x, y):
        pass


def _wafer_with_key(key_wafer_xy, size=(3456, 4608)):
    from poc.workflow_3.align.matching.test_engine import (
        make_synthetic_template,
        make_wafer_background,
    )

    wafer = make_wafer_background(frame_size=size)
    pat = make_synthetic_template(key_type="box")
    th, tw = pat.shape[:2]
    kx, ky = key_wafer_xy
    wafer[ky - th // 2:ky - th // 2 + th, kx - tw // 2:kx - tw // 2 + tw] = pat
    # 등록 이미지 = 등록 배율에서 key 를 중심에 둔 프레임 크기 crop.
    raw = wafer[ky - 192:ky + 192, kx - 256:kx + 256].copy()
    tpl = build_template(raw, recipe_id="c/r", version="v", key_type="sem")
    return wafer, {"SEM": tpl}


def test_chase_finds_key_one_search_fov_away_and_confirms_at_registered_scale():
    """key 가 착지점에서 탐색 FOV 1칸(오른쪽) 밖: 10K 로 내려 sweep 에서 보고, 되돌린 배율(20K,
    scale 0.67)에서 confirm 해 match. 끝난 자리는 key 위(복귀 안 함)."""
    start = (2304, 1728)
    key = (start[0] + 1536 + 200, start[1] + 150)  # 10K 창 폭 1536 -> 셀 (1,0) 안.
    wafer, tpl = _wafer_with_key(key)
    ctl = _WaferCtl(wafer, start, reg_mag=30000)
    out = gs.grid_align_search(
        ctl, tpl, gs.MagnificationControl(lambda: [10000, 20000, 50000], ctl.set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(radius_um=12.0, min_key_px=60, pan_budget=10),
    )
    assert out.status == "match", (out.status, out.meta, out.history[-3:])
    assert out.meta["search_mag"] == 10000
    assert ctl.cur == 20000  # confirm 배율에 그대로 둔다(상위가 reposition 으로 잇는다).
    # stage 가 key 근처에 있어야 한다(등록 배율 프레임의 절반 이내).
    assert abs(ctl.pos[0] - key[0]) < 256 and abs(ctl.pos[1] - key[1]) < 192


def test_drift_gate_falls_back_to_commanded_and_counts_flags():
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = gs.grid_align_search(
        ctl, _tpl(), mag.control(), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=2), match_fn=_low,
        shift_fn=lambda prev, cur: (999.0, 999.0),
    )
    assert out.meta["drift_flags"] == len(_moves(ctl))
    assert abs(out.meta["final_position_px"][0]) < 1


def test_abort_latch_stops_sweep_without_return_moves():
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    n = {"k": 0}

    def _match(t, f, **kw):
        n["k"] += 1
        if n["k"] == 2:
            SWITCH.request("테스트 해제")
        return _low(t, f)

    out = gs.grid_align_search(
        ctl, _tpl(), mag.control(), reg_mag=30000,
        match_fn=_match,
    )
    assert out.status == "aborted"
    assert len(_moves(ctl)) == 3  # 첫 셀로 가는 3 클릭 뒤 래치 -> 더 안 움직인다.


# ------------------------------------------------------------------
# correction 라우팅 - grid_mag 가 주어지면 fallback 은 grid 로, degraded 면 legacy 로
# ------------------------------------------------------------------

from poc.workflow_3.align.correction import correct_align_fail
from poc.workflow_3.align.live_search import LiveSearchConfig


def test_correct_align_fail_routes_fallback_to_grid_when_mag_control_given():
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = correct_align_fail(
        ctl, _tpl(), dry_run=False,
        grid_mag=mag.control(), grid_reg_mag=30000,
        grid_config=gs.GridSearchConfig(pan_budget=2),
    )
    assert out.path == "fallback" and out.status == "fallback_exhausted"
    assert mag.set_calls[0] == 5000
    assert out.fallback.meta["search_mag"] == 5000
    assert ("zoom", -1) not in ctl.calls  # legacy 휠 zoom-out 은 돌지 않았다.


def test_correct_align_fail_degrades_to_legacy_when_grid_cannot_read_mag():
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000, readback=lambda t: None)
    out = correct_align_fail(
        ctl, _tpl(), dry_run=False,
        grid_mag=mag.control(), grid_reg_mag=30000,
        fallback_config=LiveSearchConfig(pan_budget=2, low_streak_limit=3),
    )
    assert out.path == "fallback"
    assert ("zoom", -1) in ctl.calls  # legacy 경로가 돌았다.
    assert out.fallback.meta["degraded_from"] == "mag_unreadable"


def test_correct_align_fail_without_grid_mag_keeps_legacy_path():
    ctl = _Ctl()
    out = correct_align_fail(
        ctl, _tpl(), dry_run=False,
        fallback_config=LiveSearchConfig(pan_budget=2, low_streak_limit=3),
    )
    assert out.path == "fallback" and ("zoom", -1) in ctl.calls
    assert out.fallback.meta == {}


# ------------------------------------------------------------------
# 등록 배율 - cond.txt Magnification -> correct_align_fail_auto 가 MagnificationControl 조립
# ------------------------------------------------------------------

from poc.workflow_3.align import correction as corr
from poc.workflow_3.align.cond_file import parse_cond


def test_registered_magnification_reads_cond_magnification():
    assert gs.registered_magnification(parse_cond("Magnification 30000\nPixel 512,512")) == 30000.0
    assert gs.registered_magnification(parse_cond("Pixel 512,512")) is None
    assert gs.registered_magnification(None) is None


def _patch_assets(monkeypatch, tmp_path, cond_text):
    sem = tmp_path / "from_rcp" / "IMAP0002.jpg"
    sem.parent.mkdir(parents=True)
    sem.write_bytes(b"")
    if cond_text is not None:
        cdir = sem.parent / f".{sem.name}"
        cdir.mkdir()
        (cdir / "cond.txt").write_text(cond_text, encoding="utf-8")

    class _A:
        eqp_id = "E1"; class_name = "c"; recipe_name = "r"; recipe_dir = tmp_path
        recipe_om = None; recipe_sem = sem

    monkeypatch.setattr(corr, "resolve_assets_auto", lambda **k: _A())
    monkeypatch.setattr(corr, "resolve_templates", lambda assets, **kw: _tpl())


def test_auto_builds_mag_control_from_cond_and_runs_grid(monkeypatch, tmp_path):
    _patch_assets(monkeypatch, tmp_path, "Magnification 30000\nPixel 512,512")
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = corr.correct_align_fail_auto(
        ctl, dry_run=False, eqp_id="E1", recipe_name="c/r",
        grid_mag=mag.control(), grid_config=gs.GridSearchConfig(pan_budget=2),
    )
    assert out.path == "fallback" and mag.set_calls[0] == 5000
    assert out.fallback.meta["search_mag"] == 5000


def test_auto_falls_back_to_legacy_when_cond_has_no_magnification(monkeypatch, tmp_path):
    _patch_assets(monkeypatch, tmp_path, None)
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = corr.correct_align_fail_auto(
        ctl, dry_run=False, eqp_id="E1", recipe_name="c/r",
        grid_mag=mag.control(),
        fallback_config=LiveSearchConfig(pan_budget=2, low_streak_limit=3),
    )
    assert out.path == "fallback" and mag.set_calls == []
    assert ("zoom", -1) in ctl.calls


# ------------------------------------------------------------------
# 운영 설정 - env -> Workflow3Settings.search_* (cycle.py 가 GridSearchConfig 로 옮긴다)
# ------------------------------------------------------------------


def test_settings_expose_grid_search_knobs(monkeypatch):
    from poc.workflow_3.config import load_workflow3_settings

    s = load_workflow3_settings()
    assert s.search_mode == "grid" and s.search_radius_um == 30.0
    assert s.search_min_key_px == 60 and s.search_max_chase == 3 and s.search_odom_tol_fov == 0.15
    monkeypatch.setenv("ALIGN_FAIL_SEARCH_MODE", "legacy")
    monkeypatch.setenv("ALIGN_FAIL_SEARCH_RADIUS_UM", "12")
    monkeypatch.setenv("ALIGN_FAIL_SEARCH_MIN_KEY_PX", "80")
    s = load_workflow3_settings()
    assert s.search_mode == "legacy" and s.search_radius_um == 12.0 and s.search_min_key_px == 80


def test_options_are_read_once_and_only_when_zooming_out():
    """options_fn 은 드롭다운을 여는 비용이라 선택 직전에 한 번만 불린다. OM 은 아예 안 부른다."""
    ctl = _Ctl(mode="OM")
    tpl = _tpl(); tpl["OM"] = tpl.pop("SEM")
    mag = _Mag(CG_OPTIONS, reg_mag=104)
    gs.grid_align_search(ctl, tpl, mag.control(), reg_mag=104,
                         config=gs.GridSearchConfig(pan_budget=1), match_fn=_low)
    assert mag.list_calls == 0
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    out = gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=30000,
                               config=gs.GridSearchConfig(pan_budget=1), match_fn=_low)
    assert mag.list_calls == 1 and out.meta["search_mag"] == 5000


def test_empty_options_degrade_without_touching_the_stage():
    """OCR 이 행을 0개 읽으면 내릴 단도 닫을 행도 없다 -> degraded(no_mag_options), 클릭 0회."""
    ctl = _Ctl()
    mag = _Mag([], reg_mag=30000)
    out = gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=30000, match_fn=_low)
    assert out.status == "degraded" and out.meta["reason"] == "no_mag_options"
    assert mag.set_calls == [] and _moves(ctl) == []


def test_restore_failed_is_flagged_when_mag_readback_fails_on_return():
    """복귀 set_fn 판독 None -> restore_failed=True (스펙 §5). 판독 실패는 첫 zoom-out 뒤부터."""
    ctl = _Ctl()
    calls = {"n": 0}

    def _readback(t):
        calls["n"] += 1
        return t if calls["n"] == 1 else None

    mag = _Mag(CG_OPTIONS, reg_mag=30000, readback=_readback)
    out = gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=30000,
                               config=gs.GridSearchConfig(pan_budget=1), match_fn=_low)
    assert out.status == "exhausted" and out.meta["restore_failed"] is True


def test_chase_stops_when_confirm_mag_cannot_be_read():
    """추격 중 등록 배율로 되돌린 판독이 None 이면 모르는 scale 로 confirm 하지 않는다."""
    ctl = _Ctl()
    calls = {"n": 0}

    def _readback(t):
        calls["n"] += 1
        return t if calls["n"] == 1 else None

    def _adjust(template, frame, **kw):
        fh, fw = frame.shape[:2]
        return AlignKeyMatchResult(score=0.7, chamfer_score=0.7, orb_inlier_ratio=0.0,
                                   best_xy=(fw // 2 + 10, fh // 2), best_scale=1.0,
                                   decision="adjust", debug_overlay=frame)

    mag = _Mag(CG_OPTIONS, reg_mag=30000, readback=_readback)
    out = gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=30000,
                               config=gs.GridSearchConfig(pan_budget=1), match_fn=_adjust)
    assert out.status == "exhausted" and out.meta["reason"] == "mag_unreadable_confirm"
    assert not [h for h in out.history if h.get("phase") == "confirm"]


def test_notify_fn_fires_once_when_grid_search_ends_without_match():
    ctl = _Ctl()
    mag = _Mag(CG_OPTIONS, reg_mag=30000)
    seen = []
    gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=30000,
                         config=gs.GridSearchConfig(pan_budget=1), match_fn=_low,
                         notify_fn=lambda state, hist: seen.append(state.pan_count))
    assert seen == [1]


def test_no_zoom_out_still_closes_the_opened_dropdown():
    """옵션을 읽으면 드롭다운이 열린다. 내릴 단이 없으면 현재 단을 다시 골라 닫는다(배율 불변)."""
    ctl = _Ctl()
    mag = _Mag([50000, 100000], reg_mag=50000)
    out = gs.grid_align_search(ctl, _tpl(), mag.control(), reg_mag=50000,
                               config=gs.GridSearchConfig(pan_budget=1), match_fn=_low)
    assert mag.set_calls == [50000]
    assert out.meta["search_mag"] == 50000 and out.status == "exhausted"


# ------------------------------------------------------------------
# cycle 주입점 - 개발 PC/legacy 모드에서는 (None, None) 로 correction 이 legacy 를 탄다
# ------------------------------------------------------------------


def test_cycle_grid_mag_control_is_absent_without_tool_window_or_in_legacy_mode(tmp_path):
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.cycle import _build_grid_mag_control

    s = load_workflow3_settings()
    assert _build_grid_mag_control({"tool_window": None, "tag": "t"}, s, tmp_path) is None
    s2 = type(s)(**{**s.__dict__, "search_mode": "legacy"})
    assert _build_grid_mag_control({"tool_window": object(), "tag": "t"}, s2, tmp_path) is None
    s3 = type(s)(**{**s.__dict__, "fallback_search_enabled": False})
    assert _build_grid_mag_control({"tool_window": object(), "tag": "t"}, s3, tmp_path) is None
