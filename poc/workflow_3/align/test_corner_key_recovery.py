"""모서리에 걸친 align key 복구 + 탐색 커버리지/게이트/연결 회귀 (Mac, 장비 없음).

근본 원인(2026-09-19 진단): matcher 는 전 단계가 valid-mode 라 template 창이 프레임 안에
**통째로** 들어가야만 점수가 난다. key 가 모서리에 걸치면 창이 못 들어가 low 가 되고, 1-FOV
보폭 격자는 그 검출 footprint(프레임 - template)를 서로 잇지 못해 대각선 자리를 끝내 못 본다.
`uv run pytest poc/workflow_3/align/test_corner_key_recovery.py`
"""

import numpy as np
import pytest

from poc.workflow_3.align import correction as corr
from poc.workflow_3.align import grid_search as gs
from poc.workflow_3.align.correction import (
    GATE_ACT,
    PAUSED_SCALES,
    CorrectionConfig,
    correct_align_fail,
    key_visibility_gate,
)
from poc.workflow_3.align.matching.engine import build_template, frame_scales
from poc.workflow_3.align.matching.test_engine import make_synthetic_template, make_wafer_background
from poc.workflow_3.align.partial_hint import PartialHint, partial_key_hint
from poc.workflow_3.align.test_grid_search import _Ctl, _marker_matcher, _marker_wafer, _tpl, _WaferCtl
from poc.workflow_3.util.abort_switch import SWITCH

FW, FH = 512, 384
KEY = (2304, 1728)


@pytest.fixture(autouse=True)
def _reset_abort():
    SWITCH.reset()
    yield
    SWITCH.reset()


class _ClickCtl(_WaferCtl):
    """OK 클릭을 기록하는 _WaferCtl."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.screen_clicks = []

    def click_screen(self, x, y):
        self.screen_clicks.append((x, y))


def _scene(frac, off, ctl_cls=_ClickCtl):
    """key 하나 + 등록 FOV 중심의 frac 비율 box-crop template. key 는 첫 프레임 중심에서
    (off x FOV) 만큼 떨어져 보인다."""
    wafer = make_wafer_background(frame_size=(3456, 4608))
    pat = make_synthetic_template(size=int(FH * frac * 0.8), key_type="box")
    th, tw = pat.shape[:2]
    wafer[KEY[1] - th // 2:KEY[1] - th // 2 + th, KEY[0] - tw // 2:KEY[0] - tw // 2 + tw] = pat
    bw, bh = int(FW * frac), int(FH * frac)
    raw = wafer[KEY[1] - bh // 2:KEY[1] + bh // 2, KEY[0] - bw // 2:KEY[0] + bw // 2].copy()
    tpl = build_template(raw, recipe_id="c/r", version="v", key_type="sem", source_wh=(FW, FH))
    ctl = ctl_cls(wafer, (KEY[0] - off[0] * FW, KEY[1] - off[1] * FH), reg_mag=30000)
    return ctl, tpl


def _near_key(ctl, tol=8):
    return abs(ctl.pos[0] - KEY[0]) <= tol and abs(ctl.pos[1] - KEY[1]) <= tol


# ------------------------------------------------------------------
# 부분 가시 hint - 프레임 밖으로 걸친 창에서만 나온다. 이동 단서일 뿐 accept 가 아니다.
# ------------------------------------------------------------------


def test_partial_hint_points_at_a_key_cut_by_the_frame_corner():
    ctl, tpl = _scene(0.35, (0.40, 0.40))
    frame = ctl.capture()
    hint = partial_key_hint(tpl, frame, frame_scales(tpl, frame.shape, PAUSED_SCALES))
    assert hint is not None
    true_xy = (FW / 2 + 0.40 * FW, FH / 2 + 0.40 * FH)
    assert np.hypot(hint.align_xy[0] - true_xy[0], hint.align_xy[1] - true_xy[1]) <= 12, hint


def test_partial_hint_is_silent_without_a_key_at_the_border():
    # key 가 통째로 안에 있으면 엔진 몫이고, 아예 없으면 가리킬 곳이 없다.
    for off in ((0.0, 0.0), (3.0, 3.0)):
        ctl, tpl = _scene(0.35, off)
        frame = ctl.capture()
        assert partial_key_hint(tpl, frame, frame_scales(tpl, frame.shape, PAUSED_SCALES)) is None


def test_corner_key_on_the_first_frame_is_corrected_without_a_search():
    """사용자 증상 1: 첫 화면 모서리의 DFT. hint 한두 번으로 중심에 데려와 primary 가 끝낸다."""
    ctl, tpl = _scene(0.35, (0.40, 0.40))
    out = correct_align_fail(ctl, {"SEM": tpl}, dry_run=False, ok_locator=lambda s: (10, 10))
    assert out.status == "corrected" and out.path == "primary", (out.status, out.history)
    assert _near_key(ctl) and ctl.screen_clicks == [(10, 10)]
    assert ctl.moves <= 5


def test_wrong_hint_is_undone_so_the_search_keeps_its_origin(monkeypatch):
    ctl = _Ctl()
    monkeypatch.setattr(corr, "partial_key_hint", lambda *a, **k: PartialHint((400, 300), 0.5, 0.5, 1.0))
    correct_align_fail(ctl, _tpl(), dry_run=False,
                       config=CorrectionConfig(partial_hint_moves=1),
                       fallback_config=corr.LiveSearchConfig(pan_budget=0, initial_zoom_out_steps=0))
    moves = [c for c in ctl.calls if c[0] == "move"]
    assert moves[:2] == [("move", 400, 300), ("move", FW - 400, FH - 300)]


# ------------------------------------------------------------------
# 격자 보폭 = 검출 footprint (프레임 - template). 1 FOV 보폭은 대각선 자리를 못 본다.
# ------------------------------------------------------------------


def test_diagonal_key_is_seen_by_the_first_ring_without_zoom_out():
    ctl, tpl = _scene(0.6, (0.40, 0.40))
    out = gs.grid_align_search(
        ctl, {"SEM": tpl}, gs.MagnificationControl(lambda: [30000], ctl.set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=10), shift_fn=None)
    assert out.status == "match", (out.status, out.meta)
    assert ctl.moves <= 12  # 종전: 35 (전 셀 sweep 뒤 추격)
    assert out.meta["stride_fov"][0] < 0.5


# ------------------------------------------------------------------
# 게이트는 하나 - primary 가 받는 key(adjust + distinctive)는 탐색도 받는다
# ------------------------------------------------------------------


def _gate(result):
    return key_visibility_gate(result) == GATE_ACT


def _marker_search(accept_fn):
    start = (2304, 1728)
    ctl = _WaferCtl(_marker_wafer((start[0] + 300, start[1])), start, reg_mag=30000)
    return gs.grid_align_search(
        ctl, _tpl(), gs.MagnificationControl(lambda: [30000], ctl.set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=2), shift_fn=None,
        match_fn=_marker_matcher(score=0.53, confirm_decision="adjust"), accept_fn=accept_fn)


def test_search_accepts_the_adjust_key_that_primary_accepts():
    assert _marker_search(_gate).status == "match"
    assert _marker_search(None).status == "exhausted"  # 주입 없으면 종전 match-only.


# ------------------------------------------------------------------
# zoom-out 에서 강한 후보는 sweep 을 끝까지 돌지 않고 바로 쫓는다
# ------------------------------------------------------------------


def test_strong_zoomed_out_candidate_is_chased_before_the_sweep_continues():
    ctl, tpl = _scene(0.6, (0.40, 0.40))
    out = gs.grid_align_search(
        ctl, {"SEM": tpl},
        gs.MagnificationControl(lambda: [10000, 20000, 30000], ctl.set_mag), reg_mag=30000,
        config=gs.GridSearchConfig(pan_budget=10), shift_fn=None)
    assert out.status == "match" and out.meta["search_mag"] == 10000
    assert ctl.moves <= 3  # 종전: 35
    assert ctl.cur == 30000


# ------------------------------------------------------------------
# 탐색이 찾으면 끝이 아니다 - 같은 closed-loop reposition + OK 로 잇는다
# ------------------------------------------------------------------


def _search_then_correct(**cfg):
    ctl, tpl = _scene(0.35, (1.2, 0.0))  # 첫 화면에 없다(hint 도 없다) -> 탐색.
    out = correct_align_fail(
        ctl, {"SEM": tpl}, dry_run=False, ok_locator=lambda s: (10, 10),
        config=CorrectionConfig(**cfg),
        grid_mag=gs.MagnificationControl(lambda: [10000, 20000, 30000], ctl.set_mag),
        grid_reg_mag=30000, grid_config=gs.GridSearchConfig(pan_budget=10))
    return ctl, out


def test_search_match_continues_into_reposition_and_ok():
    ctl, out = _search_then_correct()
    assert out.status == "corrected" and out.path == "fallback", (out.status, out.history[-3:])
    assert out.fallback is not None and out.fallback.status == "match"
    assert _near_key(ctl) and ctl.screen_clicks == [(10, 10)]


def test_search_continue_off_keeps_the_manual_handoff():
    ctl, out = _search_then_correct(search_continue_enabled=False)
    assert out.status == "fallback_match" and ctl.screen_clicks == []


# ------------------------------------------------------------------
# 추격이 놓친 뒤 탐색 배율 복귀 - 판독이 '다른 단' 이면 px 단위를 모른다 -> 더 움직이지 않는다
# ------------------------------------------------------------------


def test_missed_chase_stops_when_search_mag_is_not_restored():
    start = (2304, 1728)
    ctl = _WaferCtl(_marker_wafer((start[0] + 300, start[1])), start, reg_mag=30000)
    calls = []

    def _set(target):
        calls.append(target)
        ctl.set_mag(target)
        return 20000.0 if len(calls) == 3 else float(target)  # 3번째 = 10K 복귀인데 20K 로 읽힌다.

    out = gs.grid_align_search(
        ctl, _tpl(), gs.MagnificationControl(lambda: [10000, 20000, 50000], _set), reg_mag=30000,
        config=gs.GridSearchConfig(radius_um=12.0, pan_budget=1), shift_fn=None,
        match_fn=_marker_matcher(score=0.7, confirm_decision="low"))
    assert calls[:3] == [10000, 20000, 10000]
    assert out.status == "exhausted" and out.meta["reason"] == "mag_unreadable_restore"
    assert out.meta["restore_failed"] is True
    # 원점 복귀 이동 없음 - 모르는 배율의 px 로 움직이지 않으니 stage 는 추격 자리에 남는다.
    assert abs(out.meta["final_position_px"][0]) > 50
