"""demonstration_rcs_control 단위 테스트 - 실장비/pywinauto 없이 Mac 에서 돈다.

시연 스크립트라도 판정 로직은 있다: 장비 목록 파싱, "닫기는 무조건 시도", "한 장비의
실패가 나머지를 막지 않는다", "탭 클릭이 실패하면 엉뚱한 화면을 휠로 굴리지 않는다".
이 파일은 그 계약만 검사한다 - 실제 클릭/휠은 오피스에서만 확인 가능하다.

    uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py
"""

import pytest

from poc.workflow_3.monitor.demonstration_rcs_control import (
    STATUS_CONNECT_FAILED,
    STATUS_CONNECTED,
    STATUS_ERROR,
    STATUS_VIEW_OK,
    STATUS_VIEW_TAB_FAILED,
    STATUS_WINDOW_NOT_FOUND,
    DemoRunResult,
    FlowStep,
    InToolFlow,
    ToolVisit,
    browse_view_tab,
    parse_flow_map,
    parse_tool_ids,
    resolve_flow_name,
    run_demonstration,
    run_in_tool_flow,
    visit_tool,
)


# ------------------------------------------------------------------
# 장비 목록 파싱.
# ------------------------------------------------------------------


def test_parse_tool_ids_splits_on_comma_and_whitespace():
    assert parse_tool_ids("MCD019, MCDC22", []) == ["MCD019", "MCDC22"]
    assert parse_tool_ids("MCD019 MCDC22", []) == ["MCD019", "MCDC22"]


def test_parse_tool_ids_drops_blanks_and_preserves_order():
    assert parse_tool_ids(" MCD019 ,, MCDC22 ,", []) == ["MCD019", "MCDC22"]


def test_parse_tool_ids_dedupes_case_insensitively_keeping_first_spelling():
    """같은 장비를 두 번 열면 두 번째 접속이 '이미 열린 창' 을 만나 시연이 깨진다."""
    assert parse_tool_ids("MCD019,mcd019,MCDC22", []) == ["MCD019", "MCDC22"]


def test_parse_tool_ids_falls_back_to_default_when_empty():
    assert parse_tool_ids("", ["MCD019"]) == ["MCD019"]
    assert parse_tool_ids("   ", ["MCD019"]) == ["MCD019"]
    assert parse_tool_ids(None, ["MCD019"]) == ["MCD019"]


# ------------------------------------------------------------------
# 장비 1대 방문 - 접속 -> 체류 -> 닫기.
# ------------------------------------------------------------------


class _CloseSpy:
    """close_fn 대역 - 호출 여부와 인자를 남긴다."""

    def __init__(self, exit_code="success", raises=None):
        self.calls = []
        self.exit_code = exit_code
        self.raises = raises

    def __call__(self, tool_id):
        self.calls.append(tool_id)
        if self.raises is not None:
            raise self.raises
        return self.exit_code


def _visit(tool_id="MCD019", *, connect_fn=None, wait_fn=None, close_fn=None,
           dwell_fn=None, action_fn=None):
    return visit_tool(
        tool_id,
        connect_fn=connect_fn or (lambda t: object()),
        wait_window_fn=wait_fn or (lambda t: (object(), "Remote Monitoring System - MCD019", "uia")),
        close_fn=close_fn or _CloseSpy(),
        dwell_fn=dwell_fn or (lambda sec: None),
        dwell_sec=3.0,
        action_fn=action_fn,
    )


def test_visit_tool_happy_path_reports_connected_and_closes():
    closer = _CloseSpy()
    visit = _visit(close_fn=closer)

    assert visit.status == STATUS_CONNECTED
    assert visit.closed is True
    assert closer.calls == ["MCD019"]


def test_visit_tool_dwells_only_after_the_window_appears():
    """창도 없는데 체류하면 시연 시간만 버린다."""
    dwelled = []
    visit = _visit(
        wait_fn=lambda t: (None, "", ""),
        dwell_fn=lambda sec: dwelled.append(sec),
    )

    assert visit.status == STATUS_WINDOW_NOT_FOUND
    assert dwelled == []


def test_visit_tool_closes_even_when_connect_returns_none():
    """접속 실패로 보고돼도 더블클릭이 먹었을 수 있다 - 창을 남기고 다음 장비로 가면 안 된다."""
    closer = _CloseSpy()
    visit = _visit(connect_fn=lambda t: None, close_fn=closer)

    assert visit.status == STATUS_CONNECT_FAILED
    assert closer.calls == ["MCD019"]


def test_visit_tool_closes_even_when_connect_raises():
    closer = _CloseSpy()
    visit = _visit(connect_fn=_raiser(RuntimeError("boom")), close_fn=closer)

    assert visit.status == STATUS_ERROR
    assert "boom" in visit.error
    assert closer.calls == ["MCD019"]


def test_visit_tool_swallows_close_failure_and_records_it():
    """닫기가 깨져도 다음 장비로는 가야 한다 - 시연 전체가 여기서 멈추면 안 된다."""
    visit = _visit(close_fn=_CloseSpy(raises=RuntimeError("close boom")))

    assert visit.status == STATUS_CONNECTED
    assert visit.closed is False
    assert "close boom" in visit.close_error


def test_visit_tool_marks_not_closed_when_close_reports_failure():
    visit = _visit(close_fn=_CloseSpy(exit_code="close_failed"))

    assert visit.closed is False
    assert visit.close_error == "close_failed"


def test_visit_tool_runs_the_in_tool_flow_before_closing_the_tool():
    """장비 안에서 실제 조작을 보여준 뒤 나온다."""
    order = []
    visit = _visit(
        close_fn=lambda t: order.append("close_tool") or "success",
        action_fn=lambda tool_id, w, t, b: order.append("action") or "optics:ok",
    )

    assert order == ["action", "close_tool"]
    assert visit.action_status == "optics:ok"


def test_visit_tool_passes_the_tool_id_so_each_tool_gets_its_own_flow():
    """MCD019 는 Optics, MCDC22 는 Work Sheet - 흐름 선택은 장비가 정한다."""
    seen = []
    _visit("MCDC22", action_fn=lambda tool_id, w, t, b: seen.append(tool_id) or "ok")

    assert seen == ["MCDC22"]


def test_visit_tool_skips_the_flow_when_the_tool_window_never_appeared():
    """창이 없으면 누를 대상 자체가 없다."""
    called = []
    visit = _visit(
        wait_fn=lambda t: (None, "", ""),
        action_fn=lambda tool_id, w, t, b: called.append("action") or "ok",
    )

    assert called == []
    assert visit.status == STATUS_WINDOW_NOT_FOUND


def test_visit_tool_still_closes_the_tool_when_the_flow_raises():
    closer = _CloseSpy()
    visit = _visit(close_fn=closer, action_fn=_raiser(RuntimeError("flow boom")))

    assert closer.calls == ["MCD019"]
    assert "flow boom" in visit.action_status


def _raiser(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


# ------------------------------------------------------------------
# tool 창 안 조작 흐름 (Optics / Work Sheet) - 여는 버튼 -> 창 확인 -> 차례로 클릭.
#
# 첫 오피스 실행에서 드러난 결함이 이 절의 존재 이유다: Optics 클릭이 먹지 않았는데도
# 시퀀스가 계속 진행돼 화면 어딘가의 **다른 Close** 를 눌렀다. 대화상자는 tool 창 안
# (원격 뷰)에 그려지므로 창 열거로는 확인할 수 없고, 라벨 판독으로만 확인된다.
# ------------------------------------------------------------------


class _Target:
    """TargetConfig 대역 - 흐름 정의에는 key 만 있으면 된다."""

    def __init__(self, key):
        self.key = key
        self.description = key


def _flow(name="optics", opener="opener", steps=(("a", (), False), ("b", (), False))):
    """(key, required, requires_previous) 목록으로 흐름을 만든다."""
    return InToolFlow(
        name=name,
        opener=FlowStep(_Target(opener), required=(("open",),), forbidden=("cancel",)),
        steps=[
            FlowStep(_Target(key), required=required, forbidden=("cancel",),
                     requires_previous=req_prev)
            for key, required, req_prev in steps
        ],
    )


class _Screen:
    """tool 창 화면 대역 - 요소별 좌표와 그 자리에서 읽히는 토큰을 흉내낸다."""

    def __init__(self, tokens_by_key=None, missing=(), click_raises=()):
        self.tokens = tokens_by_key or {}
        self.missing = set(missing)
        self.click_raises = set(click_raises)
        self.clicks = []
        self.events = []

    def locate(self, image, target):
        self.events.append(f"locate:{target.key}")
        if target.key in self.missing:
            return None
        return {"x": 10, "y": 20}

    def read_tokens(self, image, point, key):
        self.events.append(f"read:{key}")
        # 지정이 없으면 그 요소의 기대 라벨이 그대로 읽힌 것으로 본다.
        return self.tokens.get(key, ["__match__"])

    def click(self, window, image, point, key):
        self.events.append(f"click:{key}")
        self.clicks.append(key)
        if key in self.click_raises:
            raise RuntimeError(f"click boom: {key}")


def _run_flow(flow=None, screen=None, *, policy="off", attempts=1):
    """확인 정책 기본값을 off 로 둬 흐름 제어만 검사한다(라벨 판정은 별도 테스트)."""
    flow = flow or _flow()
    screen = screen or _Screen()
    status = run_in_tool_flow(
        object(), "Remote Monitoring System - MCD019", "uia", flow,
        capture_fn=lambda w: object(),
        locate_fn=screen.locate,
        read_tokens_fn=screen.read_tokens,
        click_fn=screen.click,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy=policy,
        attempts=attempts,
    )
    return status, screen


def test_flow_clicks_opener_then_every_step_in_order():
    status, screen = _run_flow()

    assert status == "optics:ok"
    assert screen.clicks == ["opener", "a", "b"]


def test_flow_confirms_each_label_before_clicking_it():
    _, screen = _run_flow()

    assert screen.events[:3] == ["locate:opener", "read:opener", "click:opener"]


def test_flow_does_not_click_when_the_opener_is_not_found():
    status, screen = _run_flow(screen=_Screen(missing={"opener"}))

    assert status == "optics:opener_failed"
    assert screen.clicks == []


def test_flow_never_looks_for_later_steps_when_the_window_is_unconfirmed():
    """첫 오피스 실행의 실제 결함 - 창이 안 떴는데 다음 요소를 찾아 다른 것을 눌렀다.

    첫 step 의 라벨이 읽히는 것이 '창이 떴다' 는 유일한 증거다. 확인되지 않으면 뒤
    요소를 찾아 나서면 안 된다 - 화면 어딘가의 비슷한 라벨을 누르게 된다. 남은 창은
    다음 단계의 tool 창 닫기가 정리한다.
    """
    status, screen = _run_flow(screen=_Screen(missing={"a"}))

    assert status == "optics:window_not_found"
    assert screen.clicks == ["opener"]
    assert "locate:b" not in screen.events


def test_flow_retries_the_opener_when_the_window_never_appears():
    """원격 뷰라 첫 클릭이 삼켜질 수 있다 - 확인이 안 되면 한 번 더 누른다."""
    status, screen = _run_flow(screen=_Screen(missing={"a"}), attempts=2)

    assert status == "optics:window_not_found"
    assert screen.clicks == ["opener", "opener"]


def test_flow_stops_retrying_once_the_window_is_confirmed():
    status, screen = _run_flow(attempts=3)

    assert screen.clicks.count("opener") == 1


def test_flow_continues_to_an_independent_step_after_a_failure():
    """Optics 의 Close 는 Memory 와 무관하게 누를 수 있다 - 창이 떠 있는 건 확인됐다."""
    status, screen = _run_flow(screen=_Screen(click_raises={"a"}))

    assert screen.clicks == ["opener", "a", "b"]
    assert status == "optics:step_failed(a)"


def test_flow_skips_a_dependent_step_when_its_predecessor_failed():
    """Work Sheet 의 Exit 는 File 드롭다운 안에 있다 - File 이 실패하면 Exit 는 없다."""
    flow = _flow(name="worksheet", steps=(("file", (), False), ("exit", (), True)))
    status, screen = _run_flow(flow, _Screen(click_raises={"file"}))

    assert screen.clicks == ["opener", "file"]
    assert "locate:exit" not in screen.events
    assert status == "worksheet:step_failed(file)"


def test_flow_skips_a_dependent_step_when_its_predecessor_was_unconfirmed():
    flow = _flow(name="worksheet", steps=(("file", (), False), ("exit", (), True)))
    status, screen = _run_flow(flow, _Screen(missing={"file"}))

    # file 이 첫 step 이라 '창 미확인' 으로 끝나고, exit 는 아예 찾지 않는다.
    assert status == "worksheet:window_not_found"
    assert "locate:exit" not in screen.events


def test_flow_rejects_a_forbidden_label_even_under_lenient_policy():
    """off/lenient 는 좌표 진단용이지, 엉뚱한 버튼을 눌러도 좋다는 뜻이 아니다."""
    status, screen = _run_flow(
        screen=_Screen(tokens_by_key={"opener": ["Open", "Cancel"]}),
        policy="lenient",
    )

    assert status == "optics:opener_failed"
    assert screen.clicks == []


def test_flow_strict_policy_requires_the_expected_label():
    flow = InToolFlow(
        name="optics",
        opener=FlowStep(_Target("optics_button"), required=(("optics",),), forbidden=()),
        steps=[FlowStep(_Target("memory_tab"), required=(("memory",),), forbidden=())],
    )
    status, screen = _run_flow(
        flow, _Screen(tokens_by_key={"optics_button": ["PM"]}), policy="strict",
    )

    assert status == "optics:opener_failed"
    assert screen.clicks == []


def test_flow_with_no_required_label_still_clicks_but_reports_tokens():
    """라벨을 모르는 요소(Work Sheet 아래 버튼)는 forbidden 만으로 거른다."""
    flow = InToolFlow(
        name="worksheet",
        opener=FlowStep(_Target("ws_button"), required=(), forbidden=("cancel",)),
        steps=[FlowStep(_Target("file"), required=(("file",),), forbidden=())],
    )
    status, screen = _run_flow(
        flow, _Screen(tokens_by_key={"ws_button": ["Sheet1"], "file": ["File"]}),
        policy="strict",
    )

    assert status == "worksheet:ok"
    assert screen.clicks == ["ws_button", "file"]


def test_flow_with_no_required_label_still_rejects_forbidden_tokens():
    flow = InToolFlow(
        name="worksheet",
        opener=FlowStep(_Target("ws_button"), required=(), forbidden=("cancel",)),
        steps=[FlowStep(_Target("file"), required=(("file",),), forbidden=())],
    )
    status, screen = _run_flow(
        flow, _Screen(tokens_by_key={"ws_button": ["Cancel"]}), policy="strict",
    )

    assert status == "worksheet:opener_failed"
    assert screen.clicks == []


def test_flow_survives_a_capture_exception():
    status, _ = _run_flow(screen=_Screen(missing=set()), attempts=1)
    assert status == "optics:ok"

    status = run_in_tool_flow(
        object(), "RMS", "uia", _flow(),
        capture_fn=_raiser(RuntimeError("capture boom")),
        locate_fn=lambda i, t: {"x": 1, "y": 1},
        read_tokens_fn=lambda i, p, k: [],
        click_fn=lambda w, i, p, k: None,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="off",
        attempts=1,
    )

    assert status == "optics:opener_failed"


# ------------------------------------------------------------------
# 장비별 조작 흐름 배정.
# ------------------------------------------------------------------


def test_parse_flow_map_reads_tool_equals_flow_pairs():
    assert parse_flow_map("MCD019=optics,MCDC22=worksheet", {}) == {
        "mcd019": "optics", "mcdc22": "worksheet",
    }


def test_parse_flow_map_falls_back_to_default_when_empty():
    default = {"mcd019": "optics"}
    assert parse_flow_map("", default) == default
    assert parse_flow_map(None, default) == default


def test_parse_flow_map_ignores_malformed_entries():
    """시연 직전 오타로 스크립트가 죽는 것보다, 그 항목만 버리고 도는 편이 낫다."""
    assert parse_flow_map("MCD019=optics,garbage,=x,y=", {}) == {"mcd019": "optics"}


def test_resolve_flow_name_is_case_insensitive():
    assert resolve_flow_name("MCDC22", {"mcdc22": "worksheet"}, "optics") == "worksheet"
    assert resolve_flow_name("mcdc22", {"mcdc22": "worksheet"}, "optics") == "worksheet"


def test_resolve_flow_name_uses_the_default_for_unlisted_tools():
    assert resolve_flow_name("MCD999", {"mcdc22": "worksheet"}, "optics") == "optics"


def test_resolve_flow_name_rejects_an_unknown_flow_name():
    """오타난 흐름 이름이 조용히 '아무것도 안 함' 이 되면 시연에서 원인을 못 찾는다."""
    assert resolve_flow_name("MCD019", {"mcd019": "optcis"}, "optics") == "optics"


# ------------------------------------------------------------------
# View 탭 + 휠 훑기.
# ------------------------------------------------------------------


def test_browse_view_tab_scrolls_down_then_back_up():
    scrolls = []
    status = browse_view_tab(
        object(), "RCS", "uia",
        click_tab_fn=lambda w, t, b: "success",
        scroll_fn=lambda dy, idx: scrolls.append(dy) or True,
        sleep_fn=lambda sec: None,
        notches=2,
    )

    assert status == STATUS_VIEW_OK
    # 아래로 2번 -> 위로 2번. 원래 스크롤 위치로 돌려놔야 다음 장면이 재현 가능하다.
    assert scrolls == [-1, -1, 1, 1]


def test_browse_view_tab_skips_scrolling_when_tab_click_fails():
    """View 로 못 갔는데 휠을 굴리면 엉뚱한 화면(List 등)을 스크롤한다."""
    scrolls = []
    status = browse_view_tab(
        object(), "RCS", "uia",
        click_tab_fn=lambda w, t, b: "tab_not_found",
        scroll_fn=lambda dy, idx: scrolls.append(dy) or True,
        sleep_fn=lambda sec: None,
        notches=2,
    )

    assert status == STATUS_VIEW_TAB_FAILED
    assert scrolls == []


def test_browse_view_tab_survives_tab_click_exception():
    status = browse_view_tab(
        object(), "RCS", "uia",
        click_tab_fn=_raiser(RuntimeError("vlm down")),
        scroll_fn=lambda dy, idx: True,
        sleep_fn=lambda sec: None,
        notches=1,
    )

    assert status == STATUS_VIEW_TAB_FAILED


# ------------------------------------------------------------------
# 전체 시나리오 순서.
# ------------------------------------------------------------------


class _Preflight:
    """PreflightOutcome 대역 - 창 확보 여부만 흉내낸다."""

    def __init__(self, status="ready", window=object()):
        self.status = status
        self.window = window
        self.title = "RCS"
        self.backend = "uia"
        self.launched = False
        self.error = ""

    @property
    def ready(self):
        return self.status == "ready"


def _run(tool_ids=("MCD019", "MCDC22"), *, preflight=None, view_fn=None,
         list_tab_fn=None, visit_fn=None, repeat=1):
    return run_demonstration(
        list(tool_ids),
        preflight_fn=lambda: preflight if preflight is not None else _Preflight(),
        view_fn=view_fn or (lambda w, t, b: STATUS_VIEW_OK),
        list_tab_fn=list_tab_fn or (lambda w, t, b: "success"),
        visit_fn=visit_fn or (lambda tool_id: ToolVisit(tool_id=tool_id, status=STATUS_CONNECTED)),
        sleep_fn=lambda sec: None,
        gap_sec=0.0,
        repeat=repeat,
    )


def test_run_demonstration_visits_every_tool_in_order():
    seen = []
    result = _run(visit_fn=lambda t: seen.append(t) or ToolVisit(tool_id=t, status=STATUS_CONNECTED))

    assert seen == ["MCD019", "MCDC22"]
    assert isinstance(result, DemoRunResult)
    assert result.ok_count == 2


def test_run_demonstration_continues_after_one_tool_raises():
    """한 대가 깨져도 나머지를 보여줘야 시연이 성립한다."""
    seen = []

    def _visit(tool_id):
        seen.append(tool_id)
        if tool_id == "MCD019":
            raise RuntimeError("boom")
        return ToolVisit(tool_id=tool_id, status=STATUS_CONNECTED)

    result = _run(visit_fn=_visit)

    assert seen == ["MCD019", "MCDC22"]
    assert result.ok_count == 1
    assert result.visits[0].status == STATUS_ERROR


def test_run_demonstration_aborts_when_rcs_window_is_not_secured():
    """창이 없으면 접속은 전부 실패한다 - 굳이 순회해 실패를 쌓지 않는다."""
    visited = []
    result = _run(
        preflight=_Preflight(status="rcs_preflight_no_window", window=None),
        visit_fn=lambda t: visited.append(t) or ToolVisit(tool_id=t, status=STATUS_CONNECTED),
    )

    assert visited == []
    assert result.aborted == "rcs_preflight_no_window"


def test_run_demonstration_continues_when_only_list_tab_preflight_failed():
    """List 가 이미 열려 있어 클릭이 불필요했을 수도 있다 - 창이 있으면 계속한다."""
    visited = []
    _run(
        preflight=_Preflight(status="rcs_preflight_list_tab_failed"),
        visit_fn=lambda t: visited.append(t) or ToolVisit(tool_id=t, status=STATUS_CONNECTED),
    )

    assert visited == ["MCD019", "MCDC22"]


def test_run_demonstration_continues_when_view_browsing_fails():
    """View 훑기는 곁가지다 - 실패해도 장비 순회(시연 본체)는 돌아야 한다."""
    visited = []
    result = _run(
        view_fn=lambda w, t, b: STATUS_VIEW_TAB_FAILED,
        visit_fn=lambda t: visited.append(t) or ToolVisit(tool_id=t, status=STATUS_CONNECTED),
    )

    assert visited == ["MCD019", "MCDC22"]
    assert result.view_status == STATUS_VIEW_TAB_FAILED


def test_run_demonstration_clicks_list_tab_before_visiting_tools():
    """connect_to_tool 은 '현재 List 탭' 을 전제한다 - View 를 본 뒤 반드시 되돌아와야 한다."""
    order = []
    _run(
        view_fn=lambda w, t, b: order.append("view") or STATUS_VIEW_OK,
        list_tab_fn=lambda w, t, b: order.append("list") or "success",
        visit_fn=lambda t: order.append(f"visit:{t}") or ToolVisit(tool_id=t, status=STATUS_CONNECTED),
    )

    assert order == ["view", "list", "visit:MCD019", "visit:MCDC22"]


def test_run_demonstration_does_not_touch_rcs_when_tool_list_is_empty():
    """장비를 못 정했는데 RCS 를 띄우면 아무 목적 없이 로그인만 한다."""
    called = []
    result = run_demonstration(
        [],
        preflight_fn=lambda: called.append("preflight") or _Preflight(),
        view_fn=lambda w, t, b: STATUS_VIEW_OK,
        list_tab_fn=lambda w, t, b: "success",
        visit_fn=lambda t: ToolVisit(tool_id=t, status=STATUS_CONNECTED),
        sleep_fn=lambda sec: None,
        gap_sec=0.0,
        repeat=1,
    )

    assert called == []
    assert result.aborted == "no_tools"


def test_run_demonstration_repeats_the_tool_loop():
    seen = []
    _run(visit_fn=lambda t: seen.append(t) or ToolVisit(tool_id=t, status=STATUS_CONNECTED), repeat=2)

    assert seen == ["MCD019", "MCDC22", "MCD019", "MCDC22"]


def test_run_demonstration_stops_cleanly_on_keyboard_interrupt():
    """Ctrl+C 로 끊어도 이미 방문한 기록은 요약에 남아야 한다."""

    def _visit(tool_id):
        if tool_id == "MCDC22":
            raise KeyboardInterrupt
        return ToolVisit(tool_id=tool_id, status=STATUS_CONNECTED)

    result = _run(visit_fn=_visit)

    assert result.aborted == "interrupted"
    assert [v.tool_id for v in result.visits] == ["MCD019"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
