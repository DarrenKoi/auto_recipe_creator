"""close_window 가 tool 창을 닫기 전에 그 창이 소유한 popup 부터 닫는지 (Win32 없이 가짜 user32)."""

import sys
import types

# Mac 에는 pywinauto 가 없다 - 모듈 import 만 되게 stub (Desktop 은 이 테스트에서 안 쓴다).
sys.modules.setdefault("pywinauto", types.SimpleNamespace(Desktop=object))

from poc.workflow_3.util import window_utils as wu  # noqa: E402


class _FakeFn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class _FakeUser32:
    def __init__(self, owners):
        self.owners = owners  # hwnd -> owner hwnd
        self.closed = []
        self.GetWindow = _FakeFn(lambda hwnd, cmd: self.owners.get(hwnd) if cmd == 4 else None)
        self.PostMessageW = _FakeFn(lambda hwnd, msg, w, l: self.closed.append((hwnd, msg)))


def test_closes_only_popups_owned_by_tool_window(monkeypatch):
    tool = 100
    rows = [wu.WindowRow("Remote Monitoring System - MCD019", tool, 1),
            wu.WindowRow("MemoPrint", 200, 1),        # tool 창이 소유한 modal
            wu.WindowRow("RCS Main", 300, 1)]         # 다른 창 - 건드리지 않는다
    monkeypatch.setattr(wu, "collect_window_rows", lambda **_: rows)
    user32 = _FakeUser32({200: tool, 300: None})

    count = wu._close_owned_popups(user32, tool, debug_label="t", settle_sec=0)

    assert count == 1
    assert user32.closed == [(200, 0x0010)]
