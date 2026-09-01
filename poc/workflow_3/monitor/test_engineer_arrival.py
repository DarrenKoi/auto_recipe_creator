"""엔지니어 도착 게이트 테스트 - VLM/실장비 없이 Mac 에서 돈다.

덮는 계약 셋:
  ① 접근 요청 팝업은 tool 창 **프레임 안**에 그려지므로 프레임 변화가 있을 때만
     VLM 을 부른다(정지 화면에서 콜 0).
  ② 도착 판정은 status 가 아니라 **라벨 확인**이다 - 관찰 전용 모드에서는 팝업이
     없어도 observed 가 나오기 때문이다.
  ③ 도착 전에는 arrival_wait_sec 를, 도착 후에는 engineer_watch_sec 를 센다.

    uv run pytest poc/workflow_3/monitor/test_engineer_arrival.py
"""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from poc.workflow_3.monitor import cycle
from poc.workflow_3.monitor.access_request import (
    STATUS_CONFIRM_FAILED,
    STATUS_GRANTED,
    STATUS_OBSERVED,
    AccessRequestResult,
)


@dataclass
class _Settings:
    access_change_min_px: int = 200
    access_request_watch_enabled: bool = True


class _Image:
    """to_diff_gray 가 받는 최소 인터페이스(np.asarray 가능한 것)."""

    def __init__(self, value, size=(64, 64)):
        self.arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)

    def __array__(self, dtype=None):
        return self.arr if dtype is None else self.arr.astype(dtype)


# ------------------------------------------------------------------
# ① 변화 게이트 - VLM 을 부를 프레임만 통과시킨다.
# ------------------------------------------------------------------


def _gate_with(images, monkeypatch, settings=None):
    frames = list(images)
    monkeypatch.setattr(cycle, "capture_window", lambda w: frames.pop(0))
    return cycle._make_frame_change_gate("WINDOW", settings or _Settings())


def test_first_frame_always_passes(monkeypatch):
    """비교 대상이 없는 첫 프레임은 통과 - 감시 시작 시 1회는 봐야 한다."""
    gate = _gate_with([_Image(10)], monkeypatch)
    assert gate() == "WINDOW"


def test_static_frame_does_not_call_vlm(monkeypatch):
    """정지 화면은 창을 안 내준다 = VLM 콜 0."""
    gate = _gate_with([_Image(10), _Image(10)], monkeypatch)
    gate()
    assert gate() is None


def test_large_change_passes(monkeypatch):
    """팝업 등장 크기의 변화는 통과한다."""
    gate = _gate_with([_Image(10), _Image(200)], monkeypatch)
    gate()
    assert gate() == "WINDOW"


def test_capture_failure_is_not_arrival(monkeypatch):
    """캡처 실패는 None - 관측 실패를 '팝업 있음' 으로 새게 하지 않는다."""
    gate = _gate_with([None], monkeypatch)
    assert gate() is None


def test_no_tool_window_disables_watcher():
    """볼 프레임이 없으면 감시자를 아예 만들지 않는다."""
    assert cycle._make_access_watcher(_Settings(), "tag", tool_window=None) is None


# ------------------------------------------------------------------
# ② 도착 판정 - 라벨 확인만이 증거다.
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        (AccessRequestResult(status=STATUS_GRANTED, verdict="confirmed"), True),
        (AccessRequestResult(status=STATUS_OBSERVED, verdict="confirmed"), True),
        # 관찰 전용에서 팝업 없이 VLM 이 아무 데나 찍은 경우 - 도착 아님.
        (AccessRequestResult(status=STATUS_OBSERVED, verdict="unreadable"), False),
        (AccessRequestResult(status=STATUS_OBSERVED, verdict="forbidden"), False),
        (AccessRequestResult(status=STATUS_CONFIRM_FAILED, verdict="unreadable"), False),
        (AccessRequestResult(status="not_found"), False),
    ],
)
def test_arrival_requires_confirmed_label(result, expected):
    assert cycle._access_result_is_arrival(result) is expected


# ------------------------------------------------------------------
# ③ watch 예산 - 도착 전/후로 갈린다.
# ------------------------------------------------------------------


class _Recording:
    stop_reason = ""

    def is_alive(self):
        return True


def test_watch_ends_on_arrival_wait_when_nobody_comes(monkeypatch):
    """도착이 없으면 arrival_wait_sec 로 끝난다(watch_sec 를 안 쓴다)."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    calls = []
    started = time.time()
    cycle._engineer_watch(
        _Recording(), watch_sec=100.0,
        access_watcher=lambda: calls.append(1) or AccessRequestResult(status="not_found"),
        access_poll_sec=0.0, arrival_wait_sec=0.3,
    )
    assert time.time() - started < 5.0
    assert calls, "감시자는 도착 대기 구간에도 호출되어야 한다"


def test_arrival_restarts_watch_budget(monkeypatch):
    """도착이 잡히면 그 시점부터 watch_sec 를 새로 센다."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    seen = {"n": 0}

    def _watcher():
        seen["n"] += 1
        if seen["n"] == 1:
            return AccessRequestResult(status=STATUS_OBSERVED, verdict="confirmed")
        return AccessRequestResult(status="not_found")

    started = time.time()
    cycle._engineer_watch(
        _Recording(), watch_sec=0.4,
        access_watcher=_watcher, access_poll_sec=0.0, arrival_wait_sec=0.1,
    )
    # arrival_wait(0.1) 만으로 끝났다면 0.4 를 못 채운다.
    assert time.time() - started >= 0.4
