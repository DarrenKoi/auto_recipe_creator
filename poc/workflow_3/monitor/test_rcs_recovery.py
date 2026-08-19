"""RCS 재실행+재로그인 복구(recover_rcs_session) 단위 테스트.

실장비/Windows 없이 Mac 에서 돈다 - 협력자(프로세스 조회/실행/로그인/창 대기)를 전부
주입받는 설계라 판정 로직만 따로 시험할 수 있다(share_request.py 와 같은 규약).

`uv run pytest poc/workflow_3/monitor/test_rcs_recovery.py` 로 실행.
"""

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.monitor import cycle
from poc.workflow_3.monitor.cycle import build_cycle_steps
from poc.workflow_3.monitor.rcs_recovery import (
    RECOVERED,
    STATUS_LOGIN_ERROR,
    STATUS_WINDOW_NOT_FOUND,
    RecoveryOutcome,
    recover_rcs_session,
)
from poc.workflow_3.rcs.workflow_login import resolve_login_tool_name


def _settings(**overrides) -> Workflow3Settings:
    return Workflow3Settings(**overrides)


def _window():
    """창 핸들 자리표시자 - 동일성만 확인하면 되므로 내용은 필요 없다."""
    return object()


def test_skips_launch_when_rcs_process_already_alive():
    """프로세스가 살아 있으면 절대 다시 띄우지 않는다.

    창이 안 보이는 것과 프로세스가 없는 것은 다르다(스플래시/멈춘 창/최소화).
    구분하지 않으면 RcsMainHD.exe 가 두 개 떠서 단일 세션을 두고 서로 싸운다.
    """
    calls = []
    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [{"pid": 42, "name": "RcsMainHD.exe"}],
        launch_fn=lambda exe: calls.append("launch"),
        login_fn=lambda settings, target_tool_name: None,
        wait_window_fn=lambda timeout_sec: (_window(), "RCS Main", "uia"),
    )
    assert calls == [], calls
    assert outcome.launched is False, outcome


def test_launches_when_no_rcs_process_found():
    """프로세스가 없으면 실행한다 - 복구의 본래 목적."""
    calls = []
    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [],
        launch_fn=lambda exe: calls.append(exe),
        login_fn=lambda settings, target_tool_name: None,
        wait_window_fn=lambda timeout_sec: (_window(), "RCS Main", "uia"),
    )
    assert len(calls) == 1, calls
    assert outcome.launched is True, outcome


def test_login_never_opens_a_tool():
    """복구 로그인은 메인 창까지만 - target_tool_name 은 반드시 빈 문자열.

    run_login_workflow 는 target_tool_name 이 비어 있지 않으면 open_target_tool /
    verify_target_tool_opened step 을 붙여 **그 tool 에 접속한다**. 기본값에 맡기면
    env(ACTION_TARGET_TOOL_NAME)/코드 오버라이드가 지목한 엉뚱한 tool 이 열리고,
    정작 알람의 tool 은 connect_tool 이 열기도 전에 wrong_tool_opened 로 깨진다.
    어느 tool 로 들어갈지는 알람이 정한다.
    """
    seen = {}
    recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [],
        launch_fn=lambda exe: None,
        login_fn=lambda settings, target_tool_name: seen.update(
            tool=target_tool_name
        ),
        wait_window_fn=lambda timeout_sec: (_window(), "RCS Main", "uia"),
    )
    assert seen == {"tool": ""}, seen


def test_returns_main_window_after_login():
    """복구 성공 = 로그인 후 확보한 메인 창을 그대로 넘겨준다.

    호출부가 창을 다시 찾지 않게 (window, title, backend) 를 함께 싣는다 - 이 프로젝트에서
    창 재탐색은 포커스 경합을 한 번 더 만드는 비용이다.
    """
    window = _window()
    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [],
        launch_fn=lambda exe: None,
        login_fn=lambda settings, target_tool_name: None,
        wait_window_fn=lambda timeout_sec: (window, "RCS Main HD", "uia"),
    )
    assert outcome.status == RECOVERED, outcome
    assert outcome.window is window, outcome
    assert outcome.title == "RCS Main HD", outcome
    assert outcome.backend == "uia", outcome


def test_window_never_appears_reports_status_without_raising():
    """로그인은 돌았는데 메인 창이 끝내 안 뜨면 status 로 보고한다(예외 아님).

    복구 실패는 사이클의 정상적인 종료 경로 중 하나다 - 엔지니어 직접 처리로 넘기면
    된다. 예외로 올리면 상위 폴링 루프의 error 경로와 뒤섞여 원인이 흐려진다.
    """
    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [],
        launch_fn=lambda exe: None,
        login_fn=lambda settings, target_tool_name: None,
        wait_window_fn=lambda timeout_sec: (None, "", ""),
    )
    assert outcome.status == STATUS_WINDOW_NOT_FOUND, outcome
    assert outcome.window is None, outcome


def test_login_failure_is_captured_as_status_not_exception():
    """로그인이 던져도 예외를 올리지 않고 원인을 error 에 담아 status 로 보고한다."""
    def _boom(settings, target_tool_name):
        raise RuntimeError("login window never appeared")

    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: [],
        launch_fn=lambda exe: None,
        login_fn=_boom,
        wait_window_fn=lambda timeout_sec: (_window(), "RCS Main", "uia"),
    )
    assert outcome.status == STATUS_LOGIN_ERROR, outcome
    assert outcome.window is None, outcome
    assert "login window never appeared" in outcome.error, outcome
    assert "RuntimeError" in outcome.error, outcome


def test_unknown_process_scan_does_not_launch():
    """프로세스 조회가 '모름'(None)이면 실행하지 않는다.

    `find_existing_rcs_processes` 는 psutil 이 없으면 빈 리스트를 돌려준다 - "안 돌고
    있다" 와 "알 수 없다" 가 같은 값이다. 그 값을 그대로 믿으면 중복 실행 가드가
    조용히 무력화되므로(psutil 은 pyproject 에 없었다), 어댑터가 모름을 None 으로
    구분해 주고 여기서는 실행을 보류한다. RCS 가 실제로 죽어 있었다면 이어지는
    로그인이 실패해 login_error 로 정직하게 보고된다.
    """
    calls = []
    outcome = recover_rcs_session(
        _settings(),
        find_processes_fn=lambda exe: None,
        launch_fn=lambda exe: calls.append("launch"),
        login_fn=lambda settings, target_tool_name: None,
        wait_window_fn=lambda timeout_sec: (_window(), "RCS Main", "uia"),
    )
    assert calls == [], calls
    assert outcome.launched is False, outcome


def test_empty_tool_name_means_no_tool_not_env_lookup(monkeypatch):
    """빈 문자열은 "tool 없음" 이며, env 조회로 흘러가면 안 된다.

    `target_tool_name or load_target_tool_name()` 였을 때 빈 문자열이 falsy 라 그대로
    env/오버라이드 값으로 떨어졌다 - recover_rcs_session 이 ""를 넘겨도 오피스에서는
    ACTION_TARGET_TOOL_NAME 이 지목한 tool 이 열렸다는 뜻이다(가드가 겉보기에만 존재).
    None(미지정)만 env 를 본다.
    """
    monkeypatch.setenv("ACTION_TARGET_TOOL_NAME", "MCD916")
    assert resolve_login_tool_name("") == ""
    assert resolve_login_tool_name(None) == "MCD916"


# ------------------------------------------------------------------
# 배선 - cycle 의 ensure_rcs_ready step 이 복구 결과를 어떻게 옮기는가.
# ------------------------------------------------------------------


def _ensure_step():
    """ensure_rcs_ready step 을 **id 로** 찾는다 - 순서에 기대지 않는다.

    인덱스로 집으면 step 이 하나 앞에 끼는 순간 세 배선 테스트가 조용히 엉뚱한 step 을
    검사하게 된다(통과하면서 아무것도 지키지 않는다).
    """
    for step in build_cycle_steps("MCD916"):
        if step.step_id == "ensure_rcs_ready":
            return step
    raise AssertionError("build_cycle_steps 에 ensure_rcs_ready step 이 없다")


def test_step_puts_recovered_window_into_context(monkeypatch):
    """복구가 확보한 창은 context 로 들어가야 한다 - 다음 step 이 그 창을 쓴다."""
    window = _window()
    monkeypatch.setattr(cycle, "wait_for_rcs_main_window", lambda **kw: (None, "", ""))
    monkeypatch.setattr(
        cycle, "recover_rcs_session",
        lambda settings, **kw: RecoveryOutcome(
            status=RECOVERED, window=window, title="RCS Main", backend="uia"
        ),
        raising=False,
    )
    monkeypatch.setattr(cycle, "activate_window", lambda *a, **kw: None)

    context = {"eqp_id": "MCD916"}
    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), context, _settings(rcs_recovery_enabled=True)
    )
    assert result.status == "success", result
    assert context["rcs_main_window"] is window, context


def test_step_reports_recovery_status_as_failure_class(monkeypatch):
    """복구 실패는 그 status 가 그대로 failure_class 가 된다.

    rcs_unavailable(복구 비활성) 과 구분되어야 오피스 manifest 에서 "복구를 시도했으나
    로그인이 깨졌다" 와 "복구를 아예 안 했다" 가 갈린다.
    """
    monkeypatch.setattr(cycle, "wait_for_rcs_main_window", lambda **kw: (None, "", ""))
    monkeypatch.setattr(
        cycle, "recover_rcs_session",
        lambda settings, **kw: RecoveryOutcome(
            status=STATUS_LOGIN_ERROR, error="RuntimeError: boom"
        ),
        raising=False,
    )

    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), {"eqp_id": "MCD916"}, _settings(rcs_recovery_enabled=True)
    )
    assert result.status == "failed", result
    assert result.failure_class == STATUS_LOGIN_ERROR, result
    assert "boom" in (result.error_message or ""), result


def test_step_does_not_recover_when_disabled(monkeypatch):
    """롤백 스위치가 실제로 복구를 막는다(ALIGN_FAIL_RCS_RECOVERY=0 경로)."""
    called = []
    monkeypatch.setattr(cycle, "wait_for_rcs_main_window", lambda **kw: (None, "", ""))
    monkeypatch.setattr(
        cycle, "recover_rcs_session",
        lambda settings, **kw: called.append("recover"),
        raising=False,
    )

    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), {"eqp_id": "MCD916"}, _settings(rcs_recovery_enabled=False)
    )
    assert called == [], called
    assert result.failure_class == "rcs_unavailable", result


def test_recovery_default_is_on():
    """기본값 on - 셸 env 없이도 복구가 돈다(롤백은 ALIGN_FAIL_RCS_RECOVERY=0)."""
    assert Workflow3Settings().rcs_recovery_enabled is True


# ------------------------------------------------------------------
# 복구 후 List 탭 - connect_to_tool 이 '현재 List 탭' 을 전제하는 자리.
# ------------------------------------------------------------------


def test_list_tab_opened_after_recovery(monkeypatch):
    """복구 로그인 직후에는 반드시 List 탭을 연다.

    복구 로그인은 target_tool_name="" 로 부르는데, workflow_login 에서 click_list_tab /
    verify_list_tab_opened / open_target_tool 이 `if normalized_tool_name:` **한 블록**에
    묶여 있다. 즉 'tool 은 열지 않는다' 는 계약이 List 탭 클릭까지 같이 꺼 버린다.
    그 상태로 connect_to_tool 이 돌면 List 가 아닌 화면에서 tool 행을 찾게 된다.
    """
    window = _window()
    opened = []
    monkeypatch.setattr(cycle, "wait_for_rcs_main_window", lambda **kw: (None, "", ""))
    monkeypatch.setattr(
        cycle, "recover_rcs_session",
        lambda settings, **kw: RecoveryOutcome(
            status=RECOVERED, window=window, title="RCS Main", backend="uia"
        ),
        raising=False,
    )
    monkeypatch.setattr(cycle, "activate_window", lambda *a, **kw: None)
    monkeypatch.setattr(
        cycle, "_open_list_tab",
        lambda w, t, b: opened.append((w, t, b)) or True,
        raising=False,
    )

    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), {"eqp_id": "MCD916"}, _settings(rcs_recovery_enabled=True)
    )
    assert result.status == "success", result
    assert opened == [(window, "RCS Main", "uia")], opened


def test_list_tab_not_touched_when_rcs_already_up(monkeypatch):
    """복구를 안 탄 정상 경로에서는 List 탭을 건드리지 않는다.

    RCS 가 이미 List 를 띄운 채 떠 있는 것이 정상 상태다. 알람마다 탭을 다시 누르면
    VLM 왕복과 클릭이 공짜로 늘고, 로케이터가 어긋나면 멀쩡하던 화면을 망친다.
    """
    window = _window()
    opened = []
    monkeypatch.setattr(
        cycle, "wait_for_rcs_main_window", lambda **kw: (window, "RCS Main", "uia")
    )
    monkeypatch.setattr(cycle, "activate_window", lambda *a, **kw: None)
    monkeypatch.setattr(
        cycle, "_open_list_tab",
        lambda w, t, b: opened.append((w, t, b)) or True,
        raising=False,
    )

    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), {"eqp_id": "MCD916"}, _settings(rcs_recovery_enabled=True)
    )
    assert result.status == "success", result
    assert opened == [], opened


def test_list_tab_failure_does_not_fail_the_step(monkeypatch):
    """List 탭 클릭이 실패해도 step 은 성공으로 둔다.

    창은 확보됐고 List 가 이미 열려 있었을 수도 있다. 여기서 step 을 죽이면 확보한
    세션을 버리고 알람을 통째로 놓친다 - 다음 step(connect)이 실제 판정을 한다.
    """
    window = _window()
    monkeypatch.setattr(cycle, "wait_for_rcs_main_window", lambda **kw: (None, "", ""))
    monkeypatch.setattr(
        cycle, "recover_rcs_session",
        lambda settings, **kw: RecoveryOutcome(
            status=RECOVERED, window=window, title="RCS Main", backend="uia"
        ),
        raising=False,
    )
    monkeypatch.setattr(cycle, "activate_window", lambda *a, **kw: None)
    monkeypatch.setattr(
        cycle, "_open_list_tab",
        lambda w, t, b: (_ for _ in ()).throw(RuntimeError("VLM 없음")),
        raising=False,
    )

    context = {"eqp_id": "MCD916"}
    result = cycle._exec_ensure_rcs_ready(
        _ensure_step(), context, _settings(rcs_recovery_enabled=True)
    )
    assert result.status == "success", result
    assert context["rcs_main_window"] is window, "확보한 창까지 버리면 안 된다"
