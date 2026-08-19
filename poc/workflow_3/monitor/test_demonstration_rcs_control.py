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
    STATUS_OPTICS_BUTTON_FAILED,
    STATUS_OPTICS_CLOSE_FAILED,
    STATUS_OPTICS_MEMORY_FAILED,
    STATUS_OPTICS_OK,
    STATUS_OPTICS_WINDOW_NOT_FOUND,
    STATUS_VIEW_OK,
    STATUS_VIEW_TAB_FAILED,
    STATUS_WINDOW_NOT_FOUND,
    DemoRunResult,
    ToolVisit,
    browse_view_tab,
    parse_tool_ids,
    run_demonstration,
    run_optics_sequence,
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
           dwell_fn=None, optics_fn=None):
    return visit_tool(
        tool_id,
        connect_fn=connect_fn or (lambda t: object()),
        wait_window_fn=wait_fn or (lambda t: (object(), "Remote Monitoring System - MCD019", "uia")),
        close_fn=close_fn or _CloseSpy(),
        dwell_fn=dwell_fn or (lambda sec: None),
        dwell_sec=3.0,
        optics_fn=optics_fn,
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


def test_visit_tool_runs_the_optics_sequence_before_closing_the_tool():
    """장비 안에서 실제 조작(Optics -> Memory -> Close)을 보여준 뒤 나온다."""
    order = []
    visit = _visit(
        close_fn=lambda t: order.append("close_tool") or "success",
        optics_fn=lambda w, t, b: order.append("optics") or STATUS_OPTICS_OK,
    )

    assert order == ["optics", "close_tool"]
    assert visit.optics_status == STATUS_OPTICS_OK


def test_visit_tool_skips_optics_when_the_tool_window_never_appeared():
    """창이 없으면 Optics 버튼을 누를 대상 자체가 없다."""
    called = []
    visit = _visit(
        wait_fn=lambda t: (None, "", ""),
        optics_fn=lambda w, t, b: called.append("optics") or STATUS_OPTICS_OK,
    )

    assert called == []
    assert visit.status == STATUS_WINDOW_NOT_FOUND


def test_visit_tool_still_closes_the_tool_when_optics_raises():
    closer = _CloseSpy()
    visit = _visit(close_fn=closer, optics_fn=_raiser(RuntimeError("optics boom")))

    assert closer.calls == ["MCD019"]
    assert "optics boom" in visit.optics_status


def _raiser(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


# ------------------------------------------------------------------
# tool 창 안 Optics 시퀀스 - Optics... -> Memory 탭 -> Close.
#
# 첫 오피스 실행에서 드러난 결함이 이 절의 존재 이유다: Optics 클릭이 먹지 않았는데도
# 시퀀스가 계속 진행돼 화면 어딘가의 **다른 Close** 를 눌렀다. 대화상자는 tool 창 안
# (원격 뷰)에 그려지므로 창 열거로는 확인할 수 없고, 라벨 판독으로만 확인된다.
# ------------------------------------------------------------------


class _Screen:
    """tool 창 화면 대역 - 요소별 좌표와 그 자리에서 읽히는 토큰을 흉내낸다."""

    def __init__(self, tokens_by_key=None, missing=()):
        self.tokens = tokens_by_key or {
            "optics_button": ["Optics..."],
            "optics_memory_tab": ["Memory"],
            "optics_close_button": ["Close"],
        }
        self.missing = set(missing)
        self.clicks = []
        self.events = []

    def locate(self, image, target):
        self.events.append(f"locate:{target.key}")
        if target.key in self.missing:
            return None
        return {"x": 10, "y": 20}

    def read_tokens(self, image, point, key):
        self.events.append(f"read:{key}")
        return self.tokens.get(key, [])

    def click(self, window, image, point, key):
        self.events.append(f"click:{key}")
        self.clicks.append(key)


def _optics(screen=None, *, policy="strict", attempts=1):
    screen = screen or _Screen()
    status = run_optics_sequence(
        object(), "Remote Monitoring System - MCD019", "uia",
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


def test_optics_sequence_confirms_each_label_before_clicking_it():
    status, screen = _optics()

    assert status == STATUS_OPTICS_OK
    assert screen.clicks == ["optics_button", "optics_memory_tab", "optics_close_button"]
    # 각 요소는 '좌표 -> 라벨 판독 -> 클릭' 순서를 지킨다.
    assert screen.events[:3] == [
        "locate:optics_button", "read:optics_button", "click:optics_button",
    ]


def test_optics_sequence_does_not_click_when_the_optics_label_is_unconfirmed():
    status, screen = _optics(_Screen(tokens_by_key={"optics_button": ["PM"]}))

    assert status == STATUS_OPTICS_BUTTON_FAILED
    assert screen.clicks == []


def test_optics_sequence_does_not_click_when_the_optics_point_is_not_found():
    status, screen = _optics(_Screen(missing={"optics_button"}))

    assert status == STATUS_OPTICS_BUTTON_FAILED
    assert screen.clicks == []


def test_optics_sequence_never_clicks_close_when_the_dialog_is_unconfirmed():
    """첫 오피스 실행의 실제 결함 - 대화상자가 안 떴는데 다른 Close 를 눌렀다.

    Memory 라벨이 읽히는 것이 '대화상자가 떴다' 는 유일한 증거다. 확인되지 않으면
    Close 를 찾아 나서면 안 된다 - 화면 어딘가의 다른 Close 를 누르게 된다. 남은
    대화상자는 다음 단계의 tool 창 닫기가 정리한다.
    """
    status, screen = _optics(_Screen(missing={"optics_memory_tab"}))

    assert status == STATUS_OPTICS_WINDOW_NOT_FOUND
    assert screen.clicks == ["optics_button"]
    assert "locate:optics_close_button" not in screen.events


def test_optics_sequence_does_not_click_close_when_memory_label_mismatches():
    status, screen = _optics(_Screen(tokens_by_key={
        "optics_button": ["Optics..."], "optics_memory_tab": ["Image"],
    }))

    assert status == STATUS_OPTICS_WINDOW_NOT_FOUND
    assert screen.clicks == ["optics_button"]


def test_optics_sequence_retries_the_optics_click_when_the_dialog_never_appears():
    """원격 뷰라 첫 클릭이 삼켜질 수 있다 - 확인이 안 되면 한 번 더 누른다."""
    status, screen = _optics(_Screen(missing={"optics_memory_tab"}), attempts=2)

    assert status == STATUS_OPTICS_WINDOW_NOT_FOUND
    assert screen.clicks == ["optics_button", "optics_button"]


def test_optics_sequence_stops_retrying_once_the_dialog_is_confirmed():
    status, screen = _optics(attempts=3)

    assert status == STATUS_OPTICS_OK
    assert screen.clicks.count("optics_button") == 1


def test_optics_sequence_does_not_click_an_unconfirmed_close():
    status, screen = _optics(_Screen(tokens_by_key={
        "optics_button": ["Optics..."], "optics_memory_tab": ["Memory"],
        "optics_close_button": ["Cancel"],
    }))

    assert status == STATUS_OPTICS_CLOSE_FAILED
    assert screen.clicks == ["optics_button", "optics_memory_tab"]


def test_optics_sequence_reports_memory_click_failure_but_still_closes():
    """Memory 라벨이 확인됐다면 대화상자는 떠 있다 - 그때는 Close 를 눌러 정리한다."""
    screen = _Screen()
    calls = []

    def _click(window, image, point, key):
        calls.append(key)
        if key == "optics_memory_tab":
            raise RuntimeError("click boom")

    status = run_optics_sequence(
        object(), "Remote Monitoring System - MCD019", "uia",
        capture_fn=lambda w: object(),
        locate_fn=screen.locate,
        read_tokens_fn=screen.read_tokens,
        click_fn=_click,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="strict",
        attempts=1,
    )

    assert status == STATUS_OPTICS_MEMORY_FAILED
    assert calls == ["optics_button", "optics_memory_tab", "optics_close_button"]


def test_optics_sequence_lenient_policy_allows_unreadable_labels():
    status, screen = _optics(
        _Screen(tokens_by_key={"optics_button": [], "optics_memory_tab": [],
                               "optics_close_button": []}),
        policy="lenient",
    )

    assert status == STATUS_OPTICS_OK
    assert screen.clicks == ["optics_button", "optics_memory_tab", "optics_close_button"]


def test_optics_sequence_forbidden_label_is_rejected_even_when_lenient():
    """off/lenient 는 좌표 진단용이지, 엉뚱한 버튼을 눌러도 좋다는 뜻이 아니다."""
    status, screen = _optics(
        _Screen(tokens_by_key={"optics_button": ["Optics", "Cancel"]}),
        policy="lenient",
    )

    assert status == STATUS_OPTICS_BUTTON_FAILED
    assert screen.clicks == []


def test_optics_sequence_survives_a_capture_exception():
    status = run_optics_sequence(
        object(), "RMS", "uia",
        capture_fn=_raiser(RuntimeError("capture boom")),
        locate_fn=lambda i, t: {"x": 1, "y": 1},
        read_tokens_fn=lambda i, p, k: ["Optics"],
        click_fn=lambda w, i, p, k: None,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="strict",
        attempts=1,
    )

    assert status == STATUS_OPTICS_BUTTON_FAILED


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
