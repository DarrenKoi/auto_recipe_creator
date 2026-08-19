"""rcs_preflight 단위 테스트 — RCS 실행/로그인/List 탭 준비 판정.

협력자를 전부 주입받으므로 Windows/pywinauto/VLM 없이 Mac 에서 그대로 돈다
(rcs_recovery.py, share_request.py 와 같은 규약).

    uv run pytest poc/workflow_3/monitor/test_rcs_preflight.py
"""

import dataclasses

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import rcs_preflight as pf
from poc.workflow_3.monitor.rcs_recovery import RECOVERED, RecoveryOutcome


def _settings(**overrides):
    return dataclasses.replace(load_workflow3_settings(), **overrides)


class _FakeWindow:
    """창 객체 자리표시자."""


def _found(window):
    """`wait_for_rcs_main_window` 의 실제 계약인 3-tuple."""
    return lambda **kwargs: (window, "RCS - MCD630", "uia")


def _absent(**kwargs):
    return (None, "", "")


class _ListSpy:
    """List 탭 클릭 협력자 stub - 호출 인자와 횟수를 기록한다."""

    def __init__(self, exit_code=pf.LIST_TAB_SUCCESS):
        self.calls = []
        self.exit_code = exit_code

    def __call__(self, window, title, backend):
        self.calls.append((window, title, backend))
        return self.exit_code


# ------------------------------------------------------------------
# 1) 이미 로그인되어 있으면 재실행/재로그인하지 않는다.
# ------------------------------------------------------------------


def test_ready_without_relogin_when_window_present():
    """메인 창이 이미 있으면 launch/login 을 건드리지 않는다.

    preflight 는 모니터 시작마다 도는 자리다. 창이 멀쩡한데도 로그인을 다시 돌면
    엔지니어가 쓰던 세션을 매번 흔든다.
    """
    window = _FakeWindow()
    list_spy = _ListSpy()
    recover_calls = []

    outcome = pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_found(window),
        recover_fn=lambda: recover_calls.append(1),
        open_list_fn=list_spy,
    )

    assert outcome.status == pf.STATUS_READY, outcome
    assert outcome.window is window
    assert outcome.launched is False
    assert recover_calls == [], "창이 있는데 복구가 돌았다"
    assert len(list_spy.calls) == 1


# ------------------------------------------------------------------
# 2) 창이 없으면 복구(실행+로그인)를 태운다.
# ------------------------------------------------------------------


def test_recovers_when_window_absent():
    """메인 창이 없으면 recover_fn 으로 실행+로그인하고 그 창을 이어받는다."""
    window = _FakeWindow()
    list_spy = _ListSpy()

    outcome = pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_absent,
        recover_fn=lambda: RecoveryOutcome(
            status=RECOVERED, window=window, title="RCS - MCD630",
            backend="uia", launched=True,
        ),
        open_list_fn=list_spy,
    )

    assert outcome.status == pf.STATUS_READY, outcome
    assert outcome.window is window
    assert outcome.launched is True
    assert list_spy.calls[0][0] is window


def test_recovery_failure_reported_as_status():
    """복구 실패는 예외가 아니라 status 로 나간다 - 모니터 루프는 계속 돌아야 한다."""
    list_spy = _ListSpy()

    outcome = pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_absent,
        recover_fn=lambda: RecoveryOutcome(
            status="rcs_recovery_no_window", error="로그인 후 30s 안에 미출현",
        ),
        open_list_fn=list_spy,
    )

    assert outcome.status == "rcs_recovery_no_window", outcome
    assert outcome.window is None
    assert list_spy.calls == [], "창을 못 얻었는데 List 탭을 눌렀다"


# ------------------------------------------------------------------
# 3) List 탭 - connect_to_tool 이 열려 있다고 가정하는 전제를 여기서 만든다.
# ------------------------------------------------------------------


def test_opens_list_tab_after_securing_window():
    """창 확보 후 반드시 List 탭을 연다.

    connect_to_tool 은 '현재 List 탭에서' 찾는다고 가정한다. 그런데 복구 로그인은
    target_tool_name="" 이라 workflow_login 의 List 클릭 step 이 아예 안 붙는다
    (그 블록 전체가 tool 이름 유무로 갈린다). 그래서 준비 단계가 직접 열어야 한다.
    """
    window = _FakeWindow()
    list_spy = _ListSpy()

    pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_found(window),
        recover_fn=lambda: RecoveryOutcome(status="unused"),
        open_list_fn=list_spy,
    )

    assert list_spy.calls == [(window, "RCS - MCD630", "uia")]


def test_list_tab_failure_is_reported_but_window_kept():
    """List 탭 클릭 실패는 status 로 알리되 확보한 창은 그대로 넘긴다.

    List 가 이미 열려 있어 클릭이 불필요했을 수도 있고, 다음 알람의 connect 가
    성공할 수도 있다. 준비 실패로 모니터를 못 뜨게 만드는 편이 더 나쁘다.
    """
    window = _FakeWindow()

    outcome = pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_found(window),
        recover_fn=lambda: RecoveryOutcome(status="unused"),
        open_list_fn=_ListSpy(exit_code="tab_not_found"),
    )

    assert outcome.status == pf.STATUS_LIST_TAB_FAILED, outcome
    assert outcome.window is window, "창까지 버리면 안 된다"


def test_list_tab_exception_does_not_raise():
    """협력자 예외도 삼킨다 - preflight 는 모니터 기동을 막지 않는다."""
    window = _FakeWindow()

    def _boom(window, title, backend):
        raise RuntimeError("VLM 없음")

    outcome = pf.ensure_rcs_session_ready(
        _settings(),
        find_window_fn=_found(window),
        recover_fn=lambda: RecoveryOutcome(status="unused"),
        open_list_fn=_boom,
    )

    assert outcome.status == pf.STATUS_LIST_TAB_FAILED
    assert "VLM 없음" in outcome.error


# ------------------------------------------------------------------
# 4) 게이트 - 복구가 꺼져 있으면 preflight 도 실행/로그인하지 않는다.
# ------------------------------------------------------------------


def test_preflight_enabled_by_default():
    """준비는 기본 on — 첫 알람이 RCS 부팅+로그인 비용을 내지 않게 한다."""
    assert load_workflow3_settings().rcs_preflight_enabled is True


def test_monitor_skips_preflight_when_disabled():
    """ALIGN_FAIL_RCS_PREFLIGHT=0 이면 기동 준비를 아예 돌리지 않는다."""
    from poc.workflow_3.monitor import align_fail_monitor as afm

    assert afm._run_rcs_preflight(_settings(rcs_preflight_enabled=False)) is None


def test_monitor_preflight_survives_missing_rcs_modules():
    """개발 PC(pywinauto/VLM 없음)에서도 준비 배선이 모니터를 죽이지 않는다.

    replay dry-run 은 RCS 없이 도는 것이 존재 이유다. 기동 준비 때문에 그 경로가
    깨지면 Mac 검증 수단을 잃는다.
    """
    from poc.workflow_3.monitor import align_fail_monitor as afm

    # Mac 에는 RCS 모듈이 없으므로 import 단계에서 걸러져 None 이 나와야 한다.
    # (Windows 에서 이 테스트가 돌면 실제 준비가 도는 대신 outcome 이 나온다.)
    result = afm._run_rcs_preflight(_settings())
    assert result is None or hasattr(result, "status")


def test_respects_recovery_disabled_gate():
    """ALIGN_FAIL_RCS_RECOVERY=0 이면 창이 없어도 실행/로그인하지 않는다.

    같은 스위치가 두 경로(preflight/알람 시 복구)를 함께 끄지 않으면, 껐다고
    생각한 사람이 모니터 기동만으로 RCS 가 뜨는 것을 보게 된다.
    """
    recover_calls = []

    outcome = pf.ensure_rcs_session_ready(
        _settings(rcs_recovery_enabled=False),
        find_window_fn=_absent,
        recover_fn=lambda: recover_calls.append(1),
        open_list_fn=_ListSpy(),
    )

    assert outcome.status == pf.STATUS_NO_WINDOW, outcome
    assert recover_calls == []
