"""access_request 확인 게이트 테스트 - VLM/실장비 없이 Mac 에서 돈다."""

from dataclasses import dataclass

import pytest

from poc.workflow_3.monitor.access_request import (
    STATUS_BLOCKED_SAFE_MODE,
    STATUS_CONFIRM_FAILED,
    STATUS_ERROR,
    STATUS_GRANTED,
    STATUS_NOT_FOUND,
    STATUS_OBSERVED,
    grant_access_request,
)


@dataclass
class _Settings:
    action_enabled: bool = True
    access_grant_enabled: bool = True
    access_confirm_policy: str = "strict"


class _Spy:
    def __init__(self, tokens, point={"x": 10, "y": 20}, popup="POPUP"):
        self.tokens = tokens
        self.point = point
        self.popup = popup
        self.clicks = []

    def locate(self, image, target):
        return self.point

    def read(self, image, point, key):
        return self.tokens

    def click(self, window, image, point, key):
        self.clicks.append((window, point, key))

    def capture(self, window):
        return "IMAGE"

    def find(self):
        return self.popup

    def run(self, settings):
        return grant_access_request(
            settings,
            locate_fn=self.locate,
            read_tokens_fn=self.read,
            click_fn=self.click,
            capture_fn=self.capture,
            find_popup_fn=self.find,
        )


def test_no_popup_is_not_found():
    spy = _Spy(["Allow"], popup=None)
    assert spy.run(_Settings()).status == STATUS_NOT_FOUND
    assert spy.clicks == []


@pytest.mark.parametrize("label", ["Allow", "Accept", "허용", "승인", "GRANT"])
def test_allow_labels_are_clicked(label):
    spy = _Spy([label])
    assert spy.run(_Settings()).status == STATUS_GRANTED
    assert len(spy.clicks) == 1


@pytest.mark.parametrize("label", ["Deny", "Cancel", "거부", "Terminate", "취소"])
def test_forbidden_labels_are_never_clicked(label):
    """거부/취소/종료는 정책과 무관하게 막힌다 - off 도 오클릭 허용이 아니다."""
    for policy in ("strict", "lenient", "off"):
        spy = _Spy([label])
        result = spy.run(_Settings(access_confirm_policy=policy))
        assert result.status == STATUS_CONFIRM_FAILED, policy
        assert spy.clicks == []


def test_unreadable_blocked_under_strict_allowed_under_lenient():
    spy = _Spy(["????"])
    assert spy.run(_Settings()).status == STATUS_CONFIRM_FAILED
    assert spy.clicks == []

    spy2 = _Spy(["????"])
    assert spy2.run(_Settings(access_confirm_policy="lenient")).status == STATUS_GRANTED
    assert len(spy2.clicks) == 1


def test_no_coordinate_never_clicks():
    """좌표 미검출은 정책과 무관하게 클릭 금지."""
    spy = _Spy(["Allow"], point=None)
    result = spy.run(_Settings(access_confirm_policy="off"))
    assert result.status == STATUS_CONFIRM_FAILED
    assert spy.clicks == []


def test_observe_only_logs_tokens_without_clicking():
    """기본값(관찰 전용)은 확인이 통과해도 클릭하지 않는다 - 문구 학습이 목적."""
    spy = _Spy(["Allow"])
    result = spy.run(_Settings(access_grant_enabled=False))
    assert result.status == STATUS_OBSERVED
    assert result.tokens == ["Allow"]
    assert spy.clicks == []


def test_safe_mode_blocks_click():
    spy = _Spy(["Allow"])
    result = spy.run(_Settings(action_enabled=False))
    assert result.status == STATUS_BLOCKED_SAFE_MODE
    assert spy.clicks == []


def test_exception_is_not_reported_as_success():
    def _boom():
        raise RuntimeError("창 열거 실패")

    result = grant_access_request(
        _Settings(),
        locate_fn=lambda *a: None,
        read_tokens_fn=lambda *a: [],
        click_fn=lambda *a: None,
        capture_fn=lambda w: "IMAGE",
        find_popup_fn=_boom,
    )
    assert result.status == STATUS_ERROR
    assert "창 열거 실패" in result.error
