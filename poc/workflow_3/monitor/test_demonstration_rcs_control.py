"""demonstration_rcs_control 단위 테스트 - 실장비/pywinauto 없이 Mac 에서 돈다.

시연 스크립트라도 판정 로직은 있다: 장비 목록 파싱, "닫기는 무조건 시도", "한 장비의
실패가 나머지를 막지 않는다", "탭 클릭이 실패하면 엉뚱한 화면을 휠로 굴리지 않는다".
이 파일은 그 계약만 검사한다 - 실제 클릭/휠은 오피스에서만 확인 가능하다.

    uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py
"""

import ast
import pathlib

import pytest

from poc.workflow_3.monitor import demonstration_rcs_control as demo
from poc.workflow_3.monitor.demonstration_rcs_control import (
    STATUS_CONNECT_FAILED,
    STATUS_CONNECTED,
    STATUS_ERROR,
    STATUS_VIEW_OK,
    STATUS_VIEW_TAB_FAILED,
    STATUS_WINDOW_NOT_FOUND,
    DEFAULT_CONFIRM_POLICY,
    DEFAULT_TOOL_FLOWS,
    DemoRunResult,
    FlowStep,
    InToolFlow,
    ToolVisit,
    browse_view_tab,
    build_flows,
    parse_flow_map,
    parse_tool_ids,
    perform_remote_click,
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
    assert parse_flow_map("MCD019=memo_print,MCDC22=worksheet", {}) == {
        "mcd019": "memo_print", "mcdc22": "worksheet",
    }


def test_parse_flow_map_falls_back_to_default_when_empty():
    default = {"mcd019": "optics"}
    assert parse_flow_map("", default) == default
    assert parse_flow_map(None, default) == default


def test_parse_flow_map_ignores_malformed_entries():
    """시연 직전 오타로 스크립트가 죽는 것보다, 그 항목만 버리고 도는 편이 낫다."""
    assert parse_flow_map("MCD019=memo_print,garbage,=x,y=", {}) == {
        "mcd019": "memo_print",
    }


def test_resolve_flow_name_is_case_insensitive():
    assert resolve_flow_name("MCDC22", {"mcdc22": "worksheet"}, "optics") == "worksheet"
    assert resolve_flow_name("mcdc22", {"mcdc22": "worksheet"}, "optics") == "worksheet"


def test_resolve_flow_name_uses_the_default_for_unlisted_tools():
    assert resolve_flow_name("MCD999", {"mcdc22": "worksheet"}, "optics") == "optics"


def test_resolve_flow_name_rejects_an_unknown_flow_name():
    """오타난 흐름 이름이 조용히 '아무것도 안 함' 이 되면 시연에서 원인을 못 찾는다."""
    assert resolve_flow_name("MCD019", {"mcd019": "optcis"}, "optics") == "optics"


# ------------------------------------------------------------------
# 실제 흐름 정의 - 데이터가 계약이므로 여기서 고정한다.
# ------------------------------------------------------------------


def _step_by_key(flow_name, key):
    """흐름에서 key 로 step 을 집는다. 마지막 step 에 기대면 step 이 늘 때 깨진다."""
    for step in build_flows()[flow_name].steps:
        if step.target.key == key:
            return step
    raise AssertionError(f"{flow_name} 에 {key} step 이 없다")


def _flow_keys(flow):
    return [flow.opener.target.key] + [step.target.key for step in flow.steps]


def test_memo_print_flow_opens_utility_then_the_menu_item_then_the_editor():
    flow = build_flows()["memo_print"]

    assert _flow_keys(flow)[:3] == [
        "utility_button", "memo_print_menu_item", "memo_print_editor",
    ]


def test_memo_print_editor_receives_the_exact_two_line_message():
    editor_step = _step_by_key("memo_print", "memo_print_editor")

    assert editor_step.input_text == "Infra. Tech Center!!\nOne Stop Solution"
    assert editor_step.input_text == demo.DEFAULT_MEMO_TEXT


def test_memo_print_editor_depends_on_selecting_the_menu_item():
    editor_step = _step_by_key("memo_print", "memo_print_editor")

    assert editor_step.requires_previous is True


def test_mcd019_uses_memo_print_by_default():
    assert DEFAULT_TOOL_FLOWS["mcd019"] == "memo_print"


def test_optics_flow_stays_selectable_after_memo_print_became_the_default():
    """기본 배정에서 빠졌을 뿐 등록은 유지한다 - 설명문 좌표가 오피스 실측이라 못 되살린다."""
    flows = build_flows()

    assert _flow_keys(flows["optics"]) == [
        "optics_button", "optics_memory_tab", "optics_close_button",
    ]
    assert resolve_flow_name("MCD019", {"mcd019": "optics"}, "memo_print") == "optics"


def test_optics_close_does_not_depend_on_the_memory_tab():
    """Close 는 대화상자 상시 버튼이라 Memory 가 실패해도 눌러 정리한다."""
    close_step = build_flows()["optics"].steps[-1]

    assert close_step.requires_previous is False


def test_worksheet_flow_is_button_then_file_then_exit():
    flow = build_flows()["worksheet"]

    assert _flow_keys(flow) == [
        "worksheet_button", "worksheet_file_menu", "worksheet_file_exit",
    ]


def test_worksheet_exit_depends_on_the_file_menu():
    """Exit 는 File 드롭다운 안에만 있다 - File 이 실패하면 그 자리는 빈 화면이다."""
    exit_step = build_flows()["worksheet"].steps[-1]

    assert exit_step.requires_previous is True


def test_worksheet_button_label_is_confirmed_not_forbidden_only():
    """오피스 확인: 버튼에 'Work Sheet' 라고 쓰여 있다 - 확인 게이트를 온전히 건다."""
    opener = build_flows()["worksheet"].opener

    assert opener.required, "required 가 비면 라벨 확인 없이 클릭한다"
    assert ("work", "sheet") in opener.required


def test_worksheet_button_accepts_ocr_that_joins_the_words():
    """OCR 이 'WorkSheet' 로 붙여 읽어도 두 needle 이 부분 일치해 통과해야 한다."""
    from poc.workflow_3.monitor.share_request import classify_label

    opener = build_flows()["worksheet"].opener
    assert classify_label(["WorkSheet"], opener.required, opener.forbidden) == "confirmed"
    assert classify_label(["Work", "Sheet"], opener.required, opener.forbidden) == "confirmed"


def test_worksheet_button_rejects_a_neighbouring_label():
    from poc.workflow_3.monitor.share_request import classify_label

    opener = build_flows()["worksheet"].opener
    assert classify_label(["Recipe"], opener.required, opener.forbidden) != "confirmed"


def test_flow_types_only_after_clicking_the_configured_editor():
    flow = build_flows()["memo_print"]
    screen = _Screen()

    status = run_in_tool_flow(
        object(), "Remote Monitoring System - MCD019", "uia", flow,
        capture_fn=lambda w: object(),
        locate_fn=screen.locate,
        read_tokens_fn=screen.read_tokens,
        click_fn=screen.click,
        type_fn=lambda text, key: screen.events.append(f"type:{key}:{text}"),
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="off",
        attempts=1,
    )

    assert status == "memo_print:ok"
    assert screen.clicks == [
        "utility_button", "memo_print_menu_item", "memo_print_editor",
        "memo_print_close_button",
    ]
    # 입력은 편집 영역 클릭 **직후**, Close 는 그 뒤다.
    typed_at = screen.events.index(
        "type:memo_print_editor:Infra. Tech Center!!\nOne Stop Solution"
    )
    assert screen.events[typed_at - 1] == "click:memo_print_editor"
    assert screen.events[-1] == "click:memo_print_close_button"


def test_memo_print_title_does_not_block_the_editor_focus():
    flow = build_flows()["memo_print"]
    screen = _Screen(tokens_by_key={
        "utility_button": ["Utility"],
        "memo_print_menu_item": ["Memo", "Print"],
        "memo_print_editor": ["MemoPrint"],
        "memo_print_close_button": ["Close"],
    })

    status = run_in_tool_flow(
        object(), "Remote Monitoring System - MCD019", "uia", flow,
        capture_fn=lambda w: object(),
        locate_fn=screen.locate,
        read_tokens_fn=screen.read_tokens,
        click_fn=screen.click,
        type_fn=lambda text, key: None,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="strict",
        attempts=1,
    )

    assert status == "memo_print:ok"
    assert "memo_print_editor" in screen.clicks


class _KeyboardSpy:
    def __init__(self):
        self.events = []

    def type(self, text):
        self.events.append(("type", text))

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))


def test_type_multiline_text_converts_newline_to_enter():
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        "first\nsecond",
        "memo_print_editor",
        action_enabled=True,
        keyboard=keyboard,
        enter_key="ENTER",
        sleep_fn=lambda sec: None,
        char_delay_sec=0.0,
    )

    assert keyboard.events == [
        *(("type", ch) for ch in "first"),
        ("press", "ENTER"),
        ("release", "ENTER"),
        *(("type", ch) for ch in "second"),
    ]


def test_type_multiline_text_dry_run_never_touches_the_keyboard():
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        demo.DEFAULT_MEMO_TEXT,
        "memo_print_editor",
        action_enabled=False,
        keyboard=keyboard,
        enter_key="ENTER",
        sleep_fn=lambda sec: None,
        char_delay_sec=0.0,
    )

    assert keyboard.events == []


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


# ------------------------------------------------------------------
# 이름 위치 가드 - Mac 에서 실행으로는 잡히지 않는 부류.
#
# 이 모듈의 Windows 배선은 pywinauto 가 없는 Mac 에서 import 자체가 안 되므로,
# "이름을 엉뚱한 모듈에서 가져왔다" 가 조용히 통과한다(실제로 capture_window 를
# image_utils 가 아닌 window_utils 에서 가져와 오피스에서만 터졌다). 그래서 실행
# 대신 **AST 로** 각 from-import 의 이름이 대상 모듈에 실제로 있는지 확인한다.
# ------------------------------------------------------------------

def _module_file(module_name):
    """poc.workflow_3.x.y -> 파일 경로. 패키지면 __init__.py."""
    root = pathlib.Path(__file__).resolve().parents[3]
    base = root / pathlib.Path(*module_name.split("."))
    if base.is_dir():
        return base / "__init__.py"
    return base.with_suffix(".py")


def _top_level_names(tree):
    """모듈이 최상위에서 바인딩하는 이름들(def/class/대입/import 별칭)."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Try):
            # import-guard 패턴(try/except ImportError) 안의 import 도 최상위 이름이다.
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
    return names


def test_every_workflow3_import_name_exists_in_its_module():
    source_path = _module_file("poc.workflow_3.monitor.demonstration_rcs_control")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    missing = []
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("poc.workflow_3"):
            continue
        target = _module_file(node.module)
        if not target.is_file():
            missing.append(f"{node.module} (모듈 파일 없음)")
            continue
        available = _top_level_names(ast.parse(target.read_text(encoding="utf-8")))
        for alias in node.names:
            checked += 1
            if alias.name not in available:
                missing.append(f"{node.module}.{alias.name} (line {node.lineno})")

    assert not missing, "엉뚱한 모듈에서 가져온 이름: " + ", ".join(missing)
    assert checked > 10, f"검사한 import 가 너무 적다({checked}) - 가드가 헛돌고 있다"


# ------------------------------------------------------------------
# 원격 뷰 클릭 순서 - 2026-08-19 오피스 실측.
#
# 커서는 Optics/Work Sheet 위로 정확히 갔는데 **클릭만 안 먹었다**. 이 저장소의 다른
# 원격 뷰 조작(sem_monitor.controller._ensure_actionable)은 제스처마다 tool 창을
# foreground 로 다시 잡는데, 시연 경로가 그 단계를 빠뜨렸다. 포커스 없는 창에서는 첫
# 클릭이 창 활성화에 쓰이고 버튼에는 닿지 않는다.
# ------------------------------------------------------------------


def _click_calls(**overrides):
    calls = []
    kwargs = dict(
        foreground_fn=lambda window: calls.append("foreground") or True,
        move_fn=lambda screen, key: calls.append("move") or True,
        click_fn=lambda screen, key: calls.append("click") or True,
        sleep_fn=lambda sec: calls.append(f"sleep:{sec}"),
    )
    kwargs.update(overrides)
    return calls, kwargs


def test_remote_click_foregrounds_the_window_before_moving_and_clicking():
    calls, kwargs = _click_calls()
    perform_remote_click(
        object(), {"x": 1, "y": 2}, "optics_button", settle_sec=0.6, **kwargs
    )

    assert calls == ["foreground", "move", "sleep:0.6", "click"]


def test_remote_click_refuses_to_click_when_the_window_cannot_be_focused():
    """포커스를 못 잡았는데 누르면 그 클릭이 어디로 가는지 알 수 없다."""
    calls, kwargs = _click_calls(foreground_fn=lambda window: False)

    with pytest.raises(RuntimeError, match="foreground"):
        perform_remote_click(
            object(), {"x": 1, "y": 2}, "optics_button", settle_sec=0.0, **kwargs
        )

    assert "click" not in calls


def test_remote_click_still_clicks_when_no_foreground_helper_is_available():
    """전면화 수단이 없는 환경에서는 막지 않는다 - 그건 게이트가 아니라 부재다."""
    calls, kwargs = _click_calls(foreground_fn=None)
    perform_remote_click(
        object(), {"x": 1, "y": 2}, "optics_button", settle_sec=0.0, **kwargs
    )

    assert calls == ["move", "sleep:0.0", "click"]


def test_default_confirm_policy_is_lenient_but_still_blocks_forbidden_labels():
    """버튼이 안전하다고 확인된 뒤로는, 못 읽어서 멈추는 쪽이 더 큰 손해다.

    다만 lenient 는 '못 읽음' 만 통과시킨다 - 금지 토큰은 어떤 정책에서도 막힌다.
    """
    from poc.workflow_3.monitor.share_request import accepts_label

    assert DEFAULT_CONFIRM_POLICY == "lenient"
    assert accepts_label("unreadable", DEFAULT_CONFIRM_POLICY) is True
    assert accepts_label("forbidden", DEFAULT_CONFIRM_POLICY) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------
# 메모 입력 게이트 - 클릭보다 엄격해야 한다. 클릭은 대개 한 번이면 끝나지만
# 타이핑은 상태를 남기고, 포커스가 어디 있는지 틀리면 엉뚱한 필드에 글자가 들어간다.
# ------------------------------------------------------------------


def test_memo_editor_is_gated_by_a_real_label_so_strict_can_reject_it():
    """`required=()` 는 정책을 아예 건너뛴다(_confirm_point 조기 반환) - 타이핑엔 부적합."""
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    editor = _step_by_key("memo_print", "memo_print_editor")

    assert editor.required, "빈 required 는 strict 에서도 무검증 통과가 된다"
    assert classify_label(["MemoPrint"], editor.required, editor.forbidden) == "confirmed"
    # 읽히지 않으면 기본 정책(lenient)에서는 그대로 진행한다 - 시연이 멈추면 안 된다.
    assert accepts_label(
        classify_label([], editor.required, editor.forbidden), "lenient"
    ) is True
    # 같은 상황에서 strict 는 거부한다 - 캘리브레이션 실행이 의미를 갖는다.
    assert accepts_label(
        classify_label([], editor.required, editor.forbidden), "strict"
    ) is False


def test_memo_editor_rejects_a_recipe_field_read_under_strict():
    """popup 이 안 떴는데 눌린 자리가 다른 필드면 strict 에서 타이핑이 막혀야 한다."""
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    editor = _step_by_key("memo_print", "memo_print_editor")

    verdict = classify_label(["Recipe", "Name"], editor.required, editor.forbidden)
    assert accepts_label(verdict, "strict") is False


def test_type_multiline_text_stops_mid_string_on_abort():
    """긴급 해제(ctrl+alt+q) 는 40글자 입력 도중에도 즉시 먹어야 한다."""
    keyboard = _KeyboardSpy()

    def _aborted():
        # 호출 횟수가 아니라 **실제 입력된 글자 수**를 본다 - 루프 앞 가드가 몇 번
        # 묻는지에 테스트가 매달리지 않게 한다.
        return len(keyboard.events) >= 3  # 3글자 뒤 사용자가 단축키를 눌렀다.

    ok = demo.type_multiline_text(
        "abcdefgh",
        "memo_print_editor",
        action_enabled=True,
        keyboard=keyboard,
        enter_key="ENTER",
        sleep_fn=lambda sec: None,
        char_delay_sec=0.0,
        is_aborted_fn=_aborted,
    )

    assert ok is False
    assert keyboard.events == [("type", ch) for ch in "abc"]


def test_type_multiline_text_never_starts_when_already_aborted():
    keyboard = _KeyboardSpy()

    ok = demo.type_multiline_text(
        demo.DEFAULT_MEMO_TEXT,
        "memo_print_editor",
        action_enabled=True,
        keyboard=keyboard,
        enter_key="ENTER",
        sleep_fn=lambda sec: None,
        char_delay_sec=0.0,
        is_aborted_fn=lambda: True,
    )

    assert ok is False
    assert keyboard.events == []


# ------------------------------------------------------------------
# 가려진 여는 버튼 되살리기 (2026-08-24, 사용자 보고).
#
# Utility 버튼은 tool 모니터 **오른쪽 아래**에 있고, 다른 창이 그 위를 덮어 VLM 이
# 아예 찾지 못하는 일이 있다. 그때 엔지니어가 쓰는 손동작이 **Alt+click** 이다 -
# 누른 자리의 창이 뒤로 밀려 Utility 가 드러난다. 그래서 "여는 버튼을 못 찾으면
# 즉시 포기" 라는 기존 규칙이 여기서만 깨진다: 화면을 바꿀 수단이 실제로 있다.
# ------------------------------------------------------------------


class _HiddenThenRevealed:
    """Alt+click 을 n 번 받은 뒤에야 opener 가 보이는 화면."""

    def __init__(self, reveals_needed=1, reveal_result=True):
        self.reveals_needed = reveals_needed
        self.reveal_result = reveal_result
        self.reveals = 0
        self.events = []
        self.clicks = []

    def locate(self, image, target):
        self.events.append(f"locate:{target.key}")
        if target.key == "opener" and self.reveals < self.reveals_needed:
            return None  # 가려져 있어 좌표가 안 나온다.
        return {"x": 10, "y": 20}

    def read_tokens(self, image, point, key):
        return ["__match__"]

    def click(self, window, image, point, key):
        self.events.append(f"click:{key}")
        self.clicks.append(key)

    def reveal(self, window, image, round_index):
        self.events.append(f"reveal:{round_index}")
        self.reveals += 1
        return self.reveal_result


def _run_with_reveal(screen, *, attempts=1, reveal_attempts=2):
    return run_in_tool_flow(
        object(), "Remote Monitoring System - MCD019", "uia", _flow(),
        capture_fn=lambda w: object(),
        locate_fn=screen.locate,
        read_tokens_fn=screen.read_tokens,
        click_fn=screen.click,
        reveal_fn=screen.reveal,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="off",
        attempts=attempts,
        reveal_attempts=reveal_attempts,
    )


def test_a_hidden_opener_is_revealed_then_clicked():
    screen = _HiddenThenRevealed()

    status = _run_with_reveal(screen)

    assert status == "optics:ok"
    # 가림 해제가 클릭보다 먼저, 그리고 해제 뒤에 다시 찾는다.
    assert screen.events[:4] == ["locate:opener", "reveal:1", "locate:opener", "click:opener"]


def test_reveal_repeats_for_stacked_windows_up_to_its_own_budget():
    """창이 여러 장 겹쳐 있으면 한 번의 Alt+click 으로는 안 드러난다."""
    screen = _HiddenThenRevealed(reveals_needed=2)

    status = _run_with_reveal(screen, attempts=1, reveal_attempts=2)

    assert status == "optics:ok"
    assert [e for e in screen.events if e.startswith("reveal")] == ["reveal:1", "reveal:2"]


def test_reveal_budget_is_separate_from_the_click_retry_budget():
    """가림 해제는 '클릭이 삼켜졌다' 재시도 예산(attempts=1)을 쓰지 않는다."""
    screen = _HiddenThenRevealed(reveals_needed=2)

    assert _run_with_reveal(screen, attempts=1, reveal_attempts=2) == "optics:ok"


def test_reveal_gives_up_with_a_distinct_status_when_the_button_never_appears():
    """'가려서 못 찾음' 과 '라벨이 다름' 은 오피스에서 할 일이 다르다."""
    screen = _HiddenThenRevealed(reveals_needed=99)

    status = _run_with_reveal(screen, reveal_attempts=2)

    assert status == "optics:opener_not_visible"
    assert screen.clicks == []


def test_reveal_stops_when_the_alt_click_itself_fails():
    screen = _HiddenThenRevealed(reveals_needed=99, reveal_result=False)

    status = _run_with_reveal(screen, reveal_attempts=3)

    assert status == "optics:opener_not_visible"
    assert [e for e in screen.events if e.startswith("reveal")] == ["reveal:1"]


def test_reveal_is_not_attempted_when_the_label_was_read_as_something_else():
    """좌표가 나왔다면 그 버튼은 이미 보이는 것이다 - 창을 밀어내도 달라지지 않고,
    엉뚱한 창을 뒤로 보내기만 한다."""
    calls = []
    flow = InToolFlow(
        name="optics",
        opener=FlowStep(_Target("opener"), required=(("open",),), forbidden=("cancel",)),
        steps=[FlowStep(_Target("a"), required=(), forbidden=())],
    )

    status = run_in_tool_flow(
        object(), "RMS", "uia", flow,
        capture_fn=lambda w: object(),
        locate_fn=lambda image, target: {"x": 1, "y": 2},
        read_tokens_fn=lambda image, point, key: ["Cancel"],  # 금지 토큰
        click_fn=lambda w, i, p, k: calls.append(f"click:{k}"),
        reveal_fn=lambda w, i, n: calls.append(f"reveal:{n}") or True,
        sleep_fn=lambda sec: None,
        settle_sec=0.0,
        confirm_policy="lenient",
        attempts=2,
        reveal_attempts=2,
    )

    assert status == "optics:opener_failed"
    assert calls == []


def test_flow_without_a_reveal_helper_keeps_giving_up_immediately():
    """다른 흐름의 동작은 그대로다 - reveal 협력자가 없으면 종전과 같다."""
    status, screen = _run_flow(screen=_Screen(missing={"opener"}))

    assert status == "optics:opener_failed"
    assert screen.clicks == []


# --- Alt 를 쥔 채 누르기 - stuck-modifier 와 전면화 순서가 걸린 자리 ---


def test_alt_is_grabbed_after_the_cursor_has_arrived():
    """"오른쪽 아래로 **마우스를 옮긴 뒤** Alt+click" (사용자 설명, 2026-08-24).

    두 가지가 이 순서를 강제한다:
      * `foreground_window` 가 foreground-lock 우회로 더미 Alt down/up 을 주입하므로
        먼저 Alt 를 잡으면 그 up 이 우리 Alt 를 놓아버린다(평범한 클릭이 된다).
      * Alt 를 쥔 채로 커서를 끌고 가면 그동안의 이동이 전부 Alt 눌린 상태가 된다 -
        원격이 그걸 창 조작 제스처로 읽을 여지를 만들 이유가 없다.
    """
    calls, kwargs = _click_calls()

    perform_remote_click(
        object(), {"x": 1, "y": 2}, "reveal_utility", settle_sec=0.6,
        press_modifier_fn=lambda: calls.append("alt_down"),
        release_modifier_fn=lambda: calls.append("alt_up"),
        modifier_settle_sec=0.3,
        **kwargs,
    )

    assert calls == [
        "foreground", "move", "sleep:0.6", "alt_down", "sleep:0.3", "click", "alt_up",
    ]


def test_alt_gets_its_own_dwell_before_the_click():
    """원격은 입력을 샘플링한다 - Alt down 과 버튼 down 이 같은 틱에 들어가면 수정자
    없는 클릭으로 넘어갈 수 있다(DEMO_RCS_CLICK_HOLD_SEC 이 생긴 이유와 같다)."""
    calls, kwargs = _click_calls()

    perform_remote_click(
        object(), {"x": 1, "y": 2}, "reveal_utility", settle_sec=0.0,
        press_modifier_fn=lambda: calls.append("alt_down"),
        release_modifier_fn=lambda: calls.append("alt_up"),
        modifier_settle_sec=0.25,
        **kwargs,
    )

    assert calls[calls.index("alt_down") + 1] == "sleep:0.25"
    assert calls[calls.index("alt_down") + 2] == "click"


def test_a_plain_click_has_no_extra_modifier_dwell():
    """수정자를 안 쓰는 클릭(탭/버튼)의 타이밍은 그대로다."""
    calls, kwargs = _click_calls()

    perform_remote_click(
        object(), {"x": 1, "y": 2}, "utility_button", settle_sec=0.6, **kwargs
    )

    assert calls == ["foreground", "move", "sleep:0.6", "click"]


def test_alt_is_released_even_when_the_click_raises():
    """눌린 채 남으면 이후 모든 클릭이 Alt+click 으로 변질된다(window_utils 의 경고)."""
    calls, kwargs = _click_calls(
        click_fn=lambda screen, key: (_ for _ in ()).throw(RuntimeError("click boom")),
    )

    with pytest.raises(RuntimeError, match="click boom"):
        perform_remote_click(
            object(), {"x": 1, "y": 2}, "reveal_utility", settle_sec=0.0,
            press_modifier_fn=lambda: calls.append("alt_down"),
            release_modifier_fn=lambda: calls.append("alt_up"),
            **kwargs,
        )

    assert calls[-1] == "alt_up"


def test_alt_is_never_pressed_when_the_window_cannot_be_focused():
    calls, kwargs = _click_calls(foreground_fn=lambda window: False)

    with pytest.raises(RuntimeError, match="foreground"):
        perform_remote_click(
            object(), {"x": 1, "y": 2}, "reveal_utility", settle_sec=0.0,
            press_modifier_fn=lambda: calls.append("alt_down"),
            release_modifier_fn=lambda: calls.append("alt_up"),
            **kwargs,
        )

    assert "alt_down" not in calls


# --- 밀어낼 지점 - Utility 는 오른쪽 아래에 있다 ---


def test_reveal_point_is_in_the_bottom_right_quadrant():
    point = demo.covering_window_point(1000, 800)

    assert point["x"] > 500 and point["y"] > 400


def test_reveal_point_stays_inside_the_frame():
    """비율을 1.0 이상으로 잘못 줘도 창 밖을 누르지 않는다."""
    point = demo.covering_window_point(100, 50, x_ratio=1.4, y_ratio=2.0)

    assert point == {"x": 99, "y": 49}


def test_reveal_point_ratios_are_tunable_from_the_office():
    """Mac 에서 이 화면을 볼 수 없으니, 빗나가면 env 로 옮길 수 있어야 한다."""
    assert demo.covering_window_point(1000, 1000, x_ratio=0.5, y_ratio=0.25) == {
        "x": 500, "y": 250,
    }


def test_alt_hold_hooks_do_nothing_in_safe_mode():
    keyboard = _KeyboardSpy()

    press, release = demo.alt_hold_hooks(
        action_enabled=False, keyboard=keyboard, alt_key="ALT",
    )
    press()
    release()

    assert keyboard.events == []


def test_alt_hold_hooks_press_and_release_the_alt_key():
    keyboard = _KeyboardSpy()

    press, release = demo.alt_hold_hooks(
        action_enabled=True, keyboard=keyboard, alt_key="ALT",
    )
    press()
    release()

    assert keyboard.events == [("press", "ALT"), ("release", "ALT")]


def test_alt_hold_hooks_skip_pressing_when_aborted_but_release_is_still_safe():
    keyboard = _KeyboardSpy()

    press, release = demo.alt_hold_hooks(
        action_enabled=True, keyboard=keyboard, alt_key="ALT",
        is_aborted_fn=lambda: True,
    )
    press()
    release()

    assert keyboard.events == []


# ------------------------------------------------------------------
# 오피스 1회차 실측 (2026-08-24) - 두 가지가 드러났다.
#
# ① 메모에 **Shift 글자만** 빠졌다: "Infra. Tech Center!! / One Stop Solution" 이
#    "nfra. ech enter / ne top olution" 으로 들어갔다. 빠진 것은 정확히
#    I T C !! O S S - 전부 Shift 를 함께 눌러야 나오는 글자다. pynput 의
#    `type()` 은 Shift down/키/Shift up 을 간격 없이 보내는데, 원격은 입력을
#    샘플링하므로 그 조합이 한 틱 사이로 통째로 빠져나간다(클릭이 삼켜진 것과 같은
#    원인이며 `DEMO_RCS_CLICK_HOLD_SEC`/`ALT_SETTLE_SEC` 이 생긴 이유와 같다).
# ② Work Sheet 의 File 클릭이 실패했다 - 아래 절 참조.
# ------------------------------------------------------------------


def test_shift_characters_get_their_own_dwell_so_the_remote_registers_them():
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        "aB", "memo",
        action_enabled=True,
        keyboard=keyboard,
        enter_key="ENTER",
        shift_key="SHIFT",
        shift_mode="shift",
        sleep_fn=lambda sec: keyboard.events.append(("sleep", sec)),
        char_delay_sec=0.0,
        shift_settle_sec=0.12,
    )

    assert keyboard.events == [
        ("type", "a"), ("sleep", 0.0),
        # Shift 를 잡고 -> 한 틱 기다리고 -> 기본 키를 눌렀다 놓고 -> 또 한 틱 -> 놓는다.
        ("press", "SHIFT"), ("sleep", 0.12),
        ("press", "b"), ("release", "b"), ("sleep", 0.12),
        ("release", "SHIFT"), ("sleep", 0.0),
    ]


def test_lowercase_typing_is_unchanged():
    """소문자는 오피스에서 정상 입력됐다 - 건드릴 이유가 없다."""
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        "abc", "memo", action_enabled=True, keyboard=keyboard, enter_key="ENTER",
        shift_key="SHIFT", sleep_fn=lambda sec: None, char_delay_sec=0.0,
    )

    assert keyboard.events == [("type", c) for c in "abc"]


def test_exclamation_mark_is_typed_as_shift_plus_its_base_key():
    """'!!' 도 함께 사라졌다 - Shift+1 이라서다(US 기호 배열)."""
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        "!", "memo", action_enabled=True, keyboard=keyboard, enter_key="ENTER",
        shift_key="SHIFT", sleep_fn=lambda sec: None, char_delay_sec=0.0,
    )

    assert keyboard.events == [
        ("press", "SHIFT"), ("press", "1"), ("release", "1"), ("release", "SHIFT"),
    ]


def test_shift_is_released_even_when_the_key_press_explodes():
    """Shift 가 눌린 채 남으면 이후 입력과 클릭이 전부 변질된다."""

    class _Boom(_KeyboardSpy):
        def press(self, key):
            super().press(key)
            if key == "b":
                raise RuntimeError("press boom")

    keyboard = _Boom()

    with pytest.raises(RuntimeError, match="press boom"):
        demo.type_multiline_text(
            "B", "memo", action_enabled=True, keyboard=keyboard, enter_key="ENTER",
            shift_key="SHIFT", shift_mode="shift",
            sleep_fn=lambda sec: None, char_delay_sec=0.0,
        )

    assert keyboard.events[-1] == ("release", "SHIFT")


def test_enter_is_never_shifted():
    keyboard = _KeyboardSpy()

    demo.type_multiline_text(
        "A\nb", "memo", action_enabled=True, keyboard=keyboard, enter_key="ENTER",
        shift_key="SHIFT", shift_mode="shift",
        sleep_fn=lambda sec: None, char_delay_sec=0.0,
    )

    assert ("press", "ENTER") in keyboard.events
    # Enter 앞에서 Shift 는 이미 놓여 있다.
    enter_at = keyboard.events.index(("press", "ENTER"))
    assert keyboard.events[enter_at - 1] == ("release", "SHIFT")


def test_typing_holds_after_the_last_character_so_the_memo_can_be_read():
    """"글자를 다 넣은 뒤 2초 기다리고 Close" (사용자 지시, 2026-08-24)."""
    sleeps = []

    demo.type_multiline_text(
        "ab", "memo", action_enabled=True,
        keyboard=_KeyboardSpy(), enter_key="ENTER", shift_key="SHIFT",
        sleep_fn=sleeps.append, char_delay_sec=0.0, post_dwell_sec=2.0,
    )

    assert sleeps[-1] == 2.0


def test_shift_plan_maps_upper_and_symbol_characters():
    assert demo.shift_plan("S") == ("s", True)
    assert demo.shift_plan("s") == ("s", False)
    assert demo.shift_plan("!") == ("1", True)
    assert demo.shift_plan(".") == (".", False)
    assert demo.shift_plan("1") == ("1", False)


# --- MemoPrint 를 닫고 나온다 ---


def test_memo_print_flow_ends_by_closing_the_popup():
    flow = build_flows()["memo_print"]

    assert _flow_keys(flow) == [
        "utility_button", "memo_print_menu_item", "memo_print_editor",
        "memo_print_close_button",
    ]


def test_memo_close_is_skipped_when_the_popup_was_never_confirmed():
    """편집 영역 클릭이 popup 존재의 증거다. 그게 없으면 'Close' 를 찾아 나서면
    화면 어딘가의 다른 Close 를 누른다(엔진 계약 ②와 같은 이유)."""
    close_step = build_flows()["memo_print"].steps[-1]

    assert close_step.requires_previous is True
    assert close_step.required == (("close",), ("닫기",))


# --- Work Sheet: File 클릭 실패 (오피스 1회차) ---


def test_menu_siblings_are_not_forbidden_tokens():
    """`classify_label` 은 forbidden 을 required 보다 **먼저** 보고, forbidden 은
    lenient 에서도 막는다. 메뉴 항목의 crop 에는 형제 항목이 반드시 들어오므로
    'edit'/'view' 를 금지어로 두면 File 클릭이 스스로 막힌다 - 오피스 1회차 실패."""
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    file_step = build_flows()["worksheet"].steps[0]
    verdict = classify_label(["File", "Edit", "View"], file_step.required, file_step.forbidden)

    assert verdict == "confirmed"
    assert accepts_label(verdict, "strict") is True


def test_dropdown_items_are_not_forbidden_tokens_for_exit():
    """File 드롭다운에는 Save/Print 가 당연히 함께 있다 - 금지어로 두면 Exit 도 막힌다."""
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    exit_step = build_flows()["worksheet"].steps[-1]
    verdict = classify_label(["Save", "Print", "Exit"], exit_step.required, exit_step.forbidden)

    assert verdict == "confirmed"
    assert accepts_label(verdict, "lenient") is True


def test_file_menu_description_says_it_is_a_small_label_near_the_title():
    """VLM 이 못 찾은 요소다 - 설명문에 '작다' 와 '제목 근처' 가 들어가야 한다."""
    description = build_flows()["worksheet"].steps[0].target.description.lower()

    assert "small" in description
    assert "title" in description


# ------------------------------------------------------------------
# 오피스 2회차 실측 (2026-08-24) - Shift 를 쥐어도 **소문자로 들어온다**.
#
# 1회차는 글자가 아예 사라졌다. 원인은 pynput 의 win32 구현이다
# (`pynput/keyboard/_win32.py:83-92`): `VkKeyScan(char)` 이 "Shift 필요" 라고 답하면
# vk=0 / scan=유니코드 코드포인트 / flags=UNICODE 로 보낸다 - **vk 도 scan code 도
# 없는 이벤트**라 vk/scancode 를 중계하는 RCS 원격이 중계할 것이 없어 사라진다.
# Shift 를 직접 쥐면 기본 키가 진짜 vk 이벤트가 되어 **도착은 한다**(2회차 확인).
# 그런데 소문자로 온다 - 원격이 키를 개별 타건으로 중계하고 **쥐고 있는 수정자를
# 함께 실어 보내지 않는다**는 뜻이다.
#
# 그래서 필요한 것은 '쥐는 수정자' 가 아니라 **상태를 남기는 키**다. Caps Lock 은
# 평범한 vk 타건이고(중계된다) 그 상태는 장비 쪽 OS 가 기억한다.
# ------------------------------------------------------------------


def _type(text, **kw):
    keyboard = kw.pop("keyboard", None) or _KeyboardSpy()
    demo.type_multiline_text(
        text, "memo", action_enabled=True, keyboard=keyboard,
        enter_key="ENTER", shift_key="SHIFT", caps_key="CAPS",
        sleep_fn=kw.pop("sleep_fn", lambda sec: None),
        char_delay_sec=kw.pop("char_delay_sec", 0.0),
        **kw,
    )
    return keyboard


def test_caps_mode_toggles_caps_lock_around_an_uppercase_letter():
    """쥐는 수정자가 아니라 **타건 + 장비가 기억하는 상태**로 대문자를 만든다."""
    keyboard = _type("B", shift_settle_sec=0.12)

    assert keyboard.events == [
        ("press", "CAPS"), ("release", "CAPS"),
        ("press", "b"), ("release", "b"),
        ("press", "CAPS"), ("release", "CAPS"),
    ]


def test_caps_mode_is_the_default():
    """2회차 결과가 이 기본값을 정했다 - Shift 쥐기는 이 원격에서 안 먹는다."""
    keyboard = _type("B")

    assert ("press", "CAPS") in keyboard.events
    assert ("press", "SHIFT") not in keyboard.events


def test_caps_lock_is_turned_back_off_even_when_the_key_press_explodes():
    """켠 채로 끝나면 그 뒤 모든 입력이 대문자가 된다 - 로컬 PC 도 같이 켜진다."""

    class _Boom(_KeyboardSpy):
        def press(self, key):
            super().press(key)
            if key == "b":
                raise RuntimeError("press boom")

    keyboard = _Boom()
    with pytest.raises(RuntimeError, match="press boom"):
        _type("B", keyboard=keyboard)

    assert keyboard.events[-2:] == [("press", "CAPS"), ("release", "CAPS")]


def test_caps_mode_leaves_lowercase_alone():
    keyboard = _type("abc")

    assert keyboard.events == [("type", c) for c in "abc"]


def test_caps_mode_still_uses_shift_for_symbols():
    """Caps Lock 은 글자만 바꾼다 - '!' 는 Shift+1 밖에 방법이 없다."""
    keyboard = _type("!")

    assert keyboard.events == [
        ("press", "SHIFT"), ("press", "1"), ("release", "1"), ("release", "SHIFT"),
    ]


def test_type_mode_reproduces_the_original_pynput_behaviour():
    """1회차와 같은 경로 - A/B 로 원인을 재확인할 때만 쓴다(글자가 사라진다)."""
    keyboard = _type("B", shift_mode="type")

    assert keyboard.events == [("type", "B")]


def test_an_unknown_shift_mode_falls_back_to_the_default_not_to_nothing():
    """시연 직전 오타가 조용히 '입력 안 함' 이 되면 안 된다."""
    keyboard = _type("B", shift_mode="typoo")

    assert ("press", "CAPS") in keyboard.events
