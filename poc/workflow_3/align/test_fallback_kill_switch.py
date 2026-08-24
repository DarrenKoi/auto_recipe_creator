"""fallback search kill switch + 긴급 해제 래치의 루프 중단 단위 테스트.

배경(2026-08-24 오피스): 실장비에서 paused 매칭이 presence 게이트를 못 넘어
`live_align_search` 로 위임됐고, 그 사각 spiral pan 이 stage 를 최대 pan_budget(10)
회 끌고 다녔다 - 엔지니어 눈에는 "align key 를 잘 찾아 갔다가 갑자기 반대 방향으로
간다" 로 보인다(confirm 실패 -> broad 복귀 -> `_do_pan`). "보정을 못 함" 보다 "장비를
끌고 다님" 이 더 비싼 상황이 있어 끌 수 있어야 한다.

두 계약을 고정한다:
  1. `fallback_search_enabled=False` 면 **actuation 이 0회** 이고 status 는
     `escalated_key_not_visible`(corrected 아님 -> cube 알림이 나간다).
  2. 긴급 해제 래치가 걸리면 live search 루프가 **즉시** 빠져나온다. mouse_utils 가
     개별 클릭을 막는 것만으로는 루프가 남은 예산을 no-op 으로 끝까지 돈다.

실장비/VLM 없이 Mac 에서 돈다.
`uv run pytest poc/workflow_3/align/test_fallback_kill_switch.py` 로 실행.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poc.workflow_3.align import correction as correction_mod
from poc.workflow_3.align import live_search as live_search_mod
from poc.workflow_3.align.correction import (
    GATE_FALLBACK,
    CorrectionConfig,
    correct_align_fail,
)
from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.util.abort_switch import SWITCH


class _RecordingController:
    """모든 제스처를 기록만 하는 컨트롤러 (장비 없음)."""

    def __init__(self, mode: str = "SEM"):
        self._mode = mode
        self.calls: list[tuple] = []

    def capture(self):
        return np.zeros((240, 320), dtype=np.uint8)

    def capture_screen(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def read_mode(self):
        return self._mode

    def move_to_point(self, x, y):
        self.calls.append(("move", x, y))

    def click_screen(self, x, y):
        self.calls.append(("click", x, y))

    def zoom(self, direction):
        self.calls.append(("zoom", direction))


def _template():
    """진짜 producer(build_template)로 만든다 - 손으로 적은 픽스처는 필드 스큐를 가린다."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    return build_template(raw, recipe_id="c/r", version="v1", key_type="sem")


@pytest.fixture(autouse=True)
def _reset_abort():
    """래치는 전역이라 테스트 간 반드시 되돌린다."""
    SWITCH.reset()
    yield
    SWITCH.reset()


def _force_fallback(monkeypatch):
    """가시성 게이트가 항상 fallback 을 고르게 만든다."""
    monkeypatch.setattr(correction_mod, "key_visibility_gate", lambda *a, **k: GATE_FALLBACK)


def test_kill_switch_off_delegates_to_live_search(monkeypatch):
    """기본값(True)에서는 종전대로 live_align_search 에 위임한다 (동작 불변)."""
    _force_fallback(monkeypatch)
    called = {"n": 0}

    def _fake_search(*a, **k):
        called["n"] += 1
        return live_search_mod.LiveSearchOutcome(
            status="exhausted", final_decision="low", best=None, pan_count=3, history=[]
        )

    monkeypatch.setattr(correction_mod, "live_align_search", _fake_search)
    controller = _RecordingController()

    outcome = correct_align_fail(
        controller, {"SEM": _template()},
        config=CorrectionConfig(fallback_search_enabled=True),
        dry_run=False,
    )

    assert called["n"] == 1
    assert outcome.status == "fallback_exhausted"


def test_kill_switch_on_skips_search_entirely(monkeypatch):
    """off 면 live_align_search 를 부르지 않고 escalate 한다."""
    _force_fallback(monkeypatch)

    def _must_not_run(*a, **k):
        raise AssertionError("fallback 이 꺼졌는데 live_align_search 가 호출됐다")

    monkeypatch.setattr(correction_mod, "live_align_search", _must_not_run)
    controller = _RecordingController()

    outcome = correct_align_fail(
        controller, {"SEM": _template()},
        config=CorrectionConfig(fallback_search_enabled=False),
        dry_run=False,
    )

    assert outcome.status == "escalated_key_not_visible"
    assert outcome.path == "primary"
    assert outcome.best_xy is None


def test_kill_switch_on_performs_zero_actuation(monkeypatch):
    """off 경로는 stage 를 건드리지 않는다 - 이게 이 스위치의 존재 이유다."""
    _force_fallback(monkeypatch)
    monkeypatch.setattr(
        correction_mod, "live_align_search",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("호출되면 안 됨")),
    )
    controller = _RecordingController()

    correct_align_fail(
        controller, {"SEM": _template()},
        config=CorrectionConfig(fallback_search_enabled=False),
        dry_run=False,   # dry_run 이 아니어도 움직이지 않아야 한다.
    )

    assert controller.calls == [], f"actuation 이 발생했다: {controller.calls}"


def test_escalated_status_is_not_corrected(monkeypatch):
    """notify 는 corrected 일 때만 cube 를 생략한다 - 이 status 는 알림이 나가야 한다."""
    _force_fallback(monkeypatch)
    monkeypatch.setattr(correction_mod, "live_align_search", lambda *a, **k: None)
    controller = _RecordingController()

    outcome = correct_align_fail(
        controller, {"SEM": _template()},
        config=CorrectionConfig(fallback_search_enabled=False),
        dry_run=False,
    )

    assert outcome.status != "corrected"


def test_live_search_exits_immediately_when_aborted():
    """래치가 걸린 채 들어오면 pan 예산을 돌지 않고 즉시 나온다."""
    SWITCH.request("테스트 긴급 해제")
    controller = _RecordingController()

    outcome = live_search_mod.live_align_search(
        controller, {"SEM": _template()},
        config=live_search_mod.LiveSearchConfig(pan_budget=10),
    )

    assert outcome.status == "aborted"
    # 시작 zoom-out 은 루프 진입 전이라 허용. move(=pan/recenter)는 0 이어야 한다.
    moves = [c for c in controller.calls if c[0] == "move"]
    assert moves == [], f"래치 후 stage 를 움직였다: {moves}"


def test_live_search_stops_mid_loop_when_latched():
    """루프 도중 래치가 걸리면 그 다음 iteration 에서 멈춘다."""
    controller = _RecordingController()

    original_capture = controller.capture
    state = {"n": 0}

    def _capture_then_abort():
        state["n"] += 1
        if state["n"] == 2:
            SWITCH.request("루프 도중 해제")
        return original_capture()

    controller.capture = _capture_then_abort

    outcome = live_search_mod.live_align_search(
        controller, {"SEM": _template()},
        config=live_search_mod.LiveSearchConfig(pan_budget=10),
    )

    assert outcome.status == "aborted"
    # pan_budget 10 을 다 돌았다면 capture 가 10회 이상 불렸을 것이다.
    assert state["n"] <= 3, f"래치 후에도 루프가 계속됐다 (capture {state['n']}회)"


def test_pan_budget_default_is_halved():
    """spiral 시도 상한 기본값 (2026-08-24 오피스 실측으로 10 -> 5)."""
    assert live_search_mod.LiveSearchConfig().pan_budget == 5


def test_auto_forwards_fallback_config_to_live_search(monkeypatch):
    """correct_align_fail_auto 가 fallback_config 를 실제로 넘긴다.

    이전에는 auto 가 이 인자를 받지도 넘기지도 않아 운영 루프가 pan_budget 을
    바꿀 방법이 없었다(항상 라이브러리 기본값). 조용히 무시되면 "설정했는데 안
    듣는다" 가 되므로 계약으로 고정한다.
    """
    _force_fallback(monkeypatch)
    # assets 는 resolve_templates 로 바로 넘어가므로 필드만 있으면 된다.
    fake_assets = SimpleNamespace(
        eqp_id="MCD513", class_name="c", recipe_name="r", recipe_dir=Path("/tmp/x"),
    )
    monkeypatch.setattr(correction_mod, "resolve_assets_auto", lambda **k: fake_assets)
    monkeypatch.setattr(
        correction_mod, "resolve_templates",
        lambda *a, **k: {"SEM": _template()},
    )
    seen = {}

    def _capture_cfg(controller, templates, *, config, **k):
        seen["pan_budget"] = config.pan_budget
        return live_search_mod.LiveSearchOutcome(
            status="exhausted", final_decision="low", best=None, pan_count=0, history=[]
        )

    monkeypatch.setattr(correction_mod, "live_align_search", _capture_cfg)

    correction_mod.correct_align_fail_auto(
        _RecordingController(),
        config=CorrectionConfig(fallback_search_enabled=True),
        fallback_config=live_search_mod.LiveSearchConfig(pan_budget=2),
        dry_run=False,
    )

    assert seen["pan_budget"] == 2, "주입한 fallback_config 가 무시됐다"
