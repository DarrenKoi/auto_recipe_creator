"""긴급 해제 스위치 단위 테스트 - pynput/실장비 없이 Mac 에서 돈다."""

import pytest

from poc.workflow_3.util import abort_switch as ab


@pytest.fixture(autouse=True)
def _clean():
    ab.SWITCH.reset()
    yield
    ab.SWITCH.reset()


def test_abort_latches_and_only_the_first_call_wins():
    """두 번째 호출은 래치를 다시 걸지 않는다 - 사유가 덮이면 안 된다."""
    assert ab.is_aborted() is False
    assert ab.request_abort("hotkey ctrl+alt+q") is True
    assert ab.is_aborted() is True
    assert ab.request_abort("something else") is False
    assert ab.abort_reason() == "hotkey ctrl+alt+q"


# ---- actuation 게이트: 래치가 걸리면 마우스가 즉시 멈춰야 한다 ----


class _FakeMouse:
    """pynput MouseController 대역 - 좌표 기록만 한다."""

    def __init__(self):
        self.positions = []
        self.clicks = []
        self.scrolls = []
        self._pos = (0, 0)

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, xy):
        self._pos = xy
        self.positions.append(xy)

    def click(self, *a, **k):
        self.clicks.append((a, k))

    def press(self, *a, **k):
        self.clicks.append(("press", a))

    def release(self, *a, **k):
        self.clicks.append(("release", a))

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))


@pytest.fixture
def fake_mouse(monkeypatch):
    from poc.workflow_3.util import mouse_utils as mu

    fake = _FakeMouse()
    monkeypatch.setattr(mu, "PYNPUT_MOUSE_AVAILABLE", True)
    monkeypatch.setattr(mu, "MouseController", lambda: fake)
    monkeypatch.setattr(mu, "_GLIDE_DELAY", 0.0)
    return fake


def test_abort_stops_cursor_move_click_and_scroll(fake_mouse):
    """래치 후에는 세 actuation 이 모두 마우스를 건드리지 않는다."""
    from poc.workflow_3.util import mouse_utils as mu

    ab.request_abort("test")

    mu.move_cursor_to_screen({"x": 500, "y": 400}, "t", action_enabled=True)
    mu.click_at_screen({"x": 500, "y": 400}, "t", action_enabled=True)
    mu.scroll_at_screen({"x": 500, "y": 400}, 1, "zoom", 0, action_enabled=True)

    assert fake_mouse.positions == []
    assert fake_mouse.clicks == []
    assert fake_mouse.scrolls == []


def test_abort_mid_glide_stops_the_move_in_flight(fake_mouse):
    """glide 는 24단계 ~290ms 다. 진행 중 래치가 걸리면 그 자리에서 멈춰야 한다.

    함수 진입 시점에만 검사하면 이미 시작된 이동은 끝까지 가고, live search 처럼
    이동이 연달아 나는 구간에서는 해제가 즉각적으로 느껴지지 않는다.
    """
    from poc.workflow_3.util import mouse_utils as mu

    original_setter = type(fake_mouse).position.fset

    def _abort_after_three(self, xy):
        original_setter(self, xy)
        if len(self.positions) == 3:
            ab.request_abort("mid-glide")

    monkey = type("M", (), {})
    monkeypatch_target = type(fake_mouse)
    monkeypatch_target.position = property(
        type(fake_mouse).position.fget, _abort_after_three
    )
    try:
        mu.move_cursor_to_screen({"x": 900, "y": 700}, "t", action_enabled=True)
    finally:
        monkeypatch_target.position = property(
            type(fake_mouse).position.fget, original_setter
        )

    assert len(fake_mouse.positions) < 24, fake_mouse.positions


# ---- 전역 단축키 리스너 ----


class _FakeListener:
    def __init__(self, mapping):
        self.mapping = mapping
        self.started = False
        self.daemon = None

    def start(self):
        self.started = True


def test_hotkey_listener_latches_the_abort_when_pressed():
    """단축키를 누르면(= 매핑된 콜백 실행) 래치가 걸린다."""
    made = {}

    def factory(mapping):
        made["listener"] = _FakeListener(mapping)
        return made["listener"]

    ok = ab.start_abort_hotkey("<ctrl>+<alt>+q", listener_factory=factory)

    assert ok is True
    listener = made["listener"]
    assert listener.started is True
    assert list(listener.mapping) == ["<ctrl>+<alt>+q"]

    assert ab.is_aborted() is False
    listener.mapping["<ctrl>+<alt>+q"]()          # 사용자가 눌렀다
    assert ab.is_aborted() is True
    assert "<ctrl>+<alt>+q" in ab.abort_reason()


def test_missing_pynput_disables_the_hotkey_without_crashing():
    """개발 PC 에는 pynput 이 없을 수 있다 - 경고만 하고 루프는 계속 떠야 한다."""
    def factory(_mapping):
        raise ImportError("no pynput")

    assert ab.start_abort_hotkey("<ctrl>+<alt>+q", listener_factory=factory) is False
    assert ab.is_aborted() is False
