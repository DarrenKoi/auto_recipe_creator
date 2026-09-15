"""라이브 SEM box 클릭이 원격 뷰 성사 조건(체류 + 누름 유지)을 지키는지.

2026-09-15 오피스: recenter 더블클릭이 커서는 정확히 가는데 화면이 안 움직였다.
즉시 press/release 쌍이 원격 입력 샘플링 사이로 빠진 것으로, 시연 경로가 이미
해결한 증상이다. 이 테스트는 그 timing 이 controller 의 두 클릭 경로에서 빠지지
않게 못박는다(Mac 실행, 실장비 불필요).
"""

from poc.workflow_3.sem_monitor import controller as ctl
from poc.workflow_3.sem_monitor.panel_locator import SEMPanelMatch


def _monitor(monkeypatch):
    monkeypatch.setattr(ctl, "foreground_window", None)
    monkeypatch.setattr(ctl, "window_rect_size", None)
    panel = SEMPanelMatch(model_id="t", panel_roi=(100, 200, 400, 300),
                          landmark_xy=(0, 0), confidence=1.0, nm_per_pixel=None)
    monitor = ctl.RCSSEMMonitor(object(), panel, action_enabled=True, settle_sec=0)
    monkeypatch.setattr(monitor, "_frame_point_to_screen", lambda x, y: {"x": x, "y": y})
    return monitor


def test_recenter_and_dialog_clicks_use_remote_timing(monkeypatch):
    calls = []
    monkeypatch.setattr(ctl, "click_at_screen", lambda point, key, count=1, **kw: calls.append((point, key, count, kw)))
    monitor = _monitor(monkeypatch)
    monitor.move_to_point(10, 20)
    monitor.click_screen(5, 6)
    assert calls[0][:3] == ({"x": 110, "y": 220}, "sem_recenter", 2)
    assert calls[1][:3] == ({"x": 5, "y": 6}, "sem_dialog_click", 1)
    for _, _, _, kw in calls:
        assert kw["hold_sec"] == ctl.REMOTE_CLICK_HOLD_SEC > 0
        assert kw["pre_click_settle_sec"] == ctl.REMOTE_PRE_CLICK_SETTLE_SEC >= 0.5
