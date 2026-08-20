"""점유 공유 요청 기능의 배선 회귀 - occupancy → outcome → notify → retry.

이 파일이 지키는 불변식은 하나다: **보정이 실제로 반영됐다고 보장할 수 없는 사이클은
어디에서도 성공으로 취급되지 않는다.** 조용한 성공(알림 생략 + active 등록)이 이 기능의
가장 위험한 실패 모드이기 때문이다.

  uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py
"""

from dataclasses import dataclass

from poc.workflow_3.monitor.align_fail_monitor import (
    _RETRY_LATER_FAILURE_CLASSES,
    _RETRY_LATER_OUTCOME_STATUSES,
    _should_retry_later,
)
from poc.workflow_3.monitor.cycle import CycleResult, resolve_correction_outcome_status
from poc.workflow_3.monitor.notify import (
    CORRECTED_UNVERIFIED,
    VIEW_ONLY_OBSERVATION,
    notify_correction_outcome,
)
from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER, UNKNOWN


@dataclass
class _Outcome:
    """CorrectionOutcome 중 notify 가 실제로 읽는 필드만 갖는 대역."""

    status: str
    path: str = "primary"
    key_decision: str = ""
    best_xy: object = None
    ok_screen_xy: object = None
    fallback: object = None
    error: object = None
    second_ratio: object = None


# ------------------------------------------------------------------
# occupancy → outcome status.
# ------------------------------------------------------------------


def test_free_keeps_original_status():
    assert resolve_correction_outcome_status(FREE, "corrected") == "corrected"


def test_occupied_not_attempted_is_view_only():
    """보정을 아예 안 했으면 관전 status 그대로."""
    assert (
        resolve_correction_outcome_status(OCCUPIED_BY_OTHER, "corrected", attempted=False)
        == VIEW_ONLY_OBSERVATION
    )


def test_occupied_but_attempted_downgrades_to_unverified():
    """점유 중 보정(opt-in)은 'corrected' 로 보고하지 않는다 - cube 가 나가야 한다.

    화면 공유는 원래 view-only 라 클릭이 먹었는지 확인할 수 없다. unknown 과 같은
    강등을 받아야 조용한 미보정이 남지 않는다.
    """
    assert (
        resolve_correction_outcome_status(OCCUPIED_BY_OTHER, "corrected", attempted=True)
        == CORRECTED_UNVERIFIED
    )


def test_occupied_attempted_keeps_failure_status():
    """실패/인계 경로의 정보는 덮어쓰지 않는다."""
    assert (
        resolve_correction_outcome_status(
            OCCUPIED_BY_OTHER, "awaiting_engineer_ok", attempted=True
        )
        == "awaiting_engineer_ok"
    )


def test_unknown_downgrades_corrected():
    """반영 여부를 보장 못 하므로 성공으로 보고하지 않는다."""
    assert (
        resolve_correction_outcome_status(UNKNOWN, "corrected") == CORRECTED_UNVERIFIED
    )


def test_unknown_leaves_non_corrected_alone():
    """이미 실패/인계 경로면 그대로 둔다 - 담긴 정보를 덮어쓰지 않는다."""
    assert (
        resolve_correction_outcome_status(UNKNOWN, "awaiting_engineer_ok")
        == "awaiting_engineer_ok"
    )
    assert (
        resolve_correction_outcome_status(UNKNOWN, "escalated_no_ok")
        == "escalated_no_ok"
    )


def test_occupied_is_view_only_regardless_of_input():
    """보정을 건너뛴 점유(기본 경로)는 입력 status 와 무관하게 관전."""
    assert (
        resolve_correction_outcome_status(
            OCCUPIED_BY_OTHER, "corrected", attempted=False
        )
        == VIEW_ONLY_OBSERVATION
    )
    assert (
        resolve_correction_outcome_status(
            OCCUPIED_BY_OTHER, "no_assets", attempted=False
        )
        == VIEW_ONLY_OBSERVATION
    )


# ------------------------------------------------------------------
# notify - 두 새 status 는 cube 를 생략하지 않는다.
# ------------------------------------------------------------------


def _sent_calls(monkeypatch, outcome):
    """notify_correction_outcome 이 cube 발송에 도달했는지 기록한다."""
    calls = []
    monkeypatch.setattr(
        "poc.workflow_3.monitor.notify.RICH_NOTIFY_AVAILABLE", True, raising=False
    )
    monkeypatch.setattr(
        "poc.workflow_3.monitor.notify._send_cube_async",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    notify_correction_outcome("MCD427", "CLS/RCP", outcome, enabled=True)
    return calls


def test_corrected_suppresses_cube(monkeypatch):
    """기존 동작 - 성공은 알리지 않는다."""
    assert _sent_calls(monkeypatch, _Outcome(status="corrected")) == []


def test_view_only_observation_still_notifies(monkeypatch):
    """점유자와 알람 담당자는 다른 사람일 수 있다. 생략하면 아무도 모른다."""
    assert _sent_calls(monkeypatch, _Outcome(status=VIEW_ONLY_OBSERVATION)) != []


def test_corrected_unverified_still_notifies(monkeypatch):
    """보정 반영 여부가 미확인이면 반드시 알린다 - 조용한 성공 금지."""
    assert _sent_calls(monkeypatch, _Outcome(status=CORRECTED_UNVERIFIED)) != []


def test_new_statuses_are_not_equal_to_corrected():
    """기존 분기가 전부 정확 비교라, != 만 지키면 watch/cube 가 자동으로 산다."""
    assert VIEW_ONLY_OBSERVATION != "corrected"
    assert CORRECTED_UNVERIFIED != "corrected"


def test_summary_states_required_action(monkeypatch):
    """엔지니어가 무엇을 확인해야 하는지 요약 맨 앞에 있어야 한다."""
    from poc.workflow_3.monitor.notify import build_outcome_summary

    view_only = build_outcome_summary(_Outcome(status=VIEW_ONLY_OBSERVATION))
    assert "점유" in view_only
    unverified = build_outcome_summary(_Outcome(status=CORRECTED_UNVERIFIED))
    assert "미확인" in unverified


# ------------------------------------------------------------------
# 상위 루프 - active 가 아니라 cooldown 재시도.
# ------------------------------------------------------------------


def _cycle(outcome_status="", run_status="completed", failed_step=""):
    cycle = CycleResult(eqp_id="MCD427", recipe_id="CLS/RCP", tag="t")
    cycle.run_status = run_status
    cycle.failed_step = failed_step
    cycle.outcome_status = outcome_status
    return cycle


def test_view_only_never_registers_success():
    """완주했더라도 active 로 가면 tool 이 풀려도 영영 돌아오지 않는다."""
    assert _should_retry_later(_cycle(outcome_status=VIEW_ONLY_OBSERVATION)) is True


def test_corrected_unverified_never_registers_success():
    assert _should_retry_later(_cycle(outcome_status=CORRECTED_UNVERIFIED)) is True


def test_corrected_registers_success():
    assert _should_retry_later(_cycle(outcome_status="corrected")) is False


def test_awaiting_engineer_ok_registers_success():
    """반자동 정상 경로는 기존대로 active - 알람 해제까지 머문다."""
    assert _should_retry_later(_cycle(outcome_status="awaiting_engineer_ok")) is False


def test_empty_outcome_registers_success():
    """보정 미수행(RECIPE_ID 없음)은 기존 동작을 바꾸지 않는다."""
    assert _should_retry_later(_cycle(outcome_status="")) is False


def test_retry_set_contents():
    assert _RETRY_LATER_OUTCOME_STATUSES == {
        VIEW_ONLY_OBSERVATION,
        CORRECTED_UNVERIFIED,
    }


def test_confirm_failed_is_retry_later():
    """확인 게이트가 막은 것은 우리 인식 실패라 재시도 대상이다."""
    assert "rcs_share_confirm_failed" in _RETRY_LATER_FAILURE_CLASSES


def test_existing_retry_classes_preserved():
    """기존 점유/오클릭 분류를 잃지 않았는지."""
    assert "rcs_occupied" in _RETRY_LATER_FAILURE_CLASSES
    assert "rcs_occupied_select" in _RETRY_LATER_FAILURE_CLASSES
    assert "wrong_tool_opened" in _RETRY_LATER_FAILURE_CLASSES


# ------------------------------------------------------------------
# 재시도 상한 - cube spam / 커서 독점 방지.
# ------------------------------------------------------------------


def _settings():
    """진짜 Workflow3Settings 를 쓰고 알림만 끈다.

    손으로 적은 대역을 쓰면 필드가 하나 늘 때마다 조용히 어긋나고, 그 어긋남이
    테스트를 통과시키거나(가짜 초록) 엉뚱한 예외로 죽는다.
    """
    from dataclasses import replace

    from poc.workflow_3.config import load_workflow3_settings

    return replace(
        load_workflow3_settings(),
        popup_enabled=False,
        rich_notify_enabled=False,
        detection_notify_enabled=False,
        cycle_enabled=True,
        share_max_attempts=2,
    )


def _run_poll(monkeypatch, active, cooldown, attempts, outcome_status):
    """알람 1건을 한 번 처리한다 (사이클은 대역으로 치환)."""
    from poc.workflow_3.monitor import align_fail_monitor as afm

    monkeypatch.setattr(afm, "append_alarm_record", lambda *a, **k: None)
    monkeypatch.setattr(afm, "append_cycle_manifest", lambda *a, **k: None)
    monkeypatch.setattr(afm, "send_detection_notify_async", lambda *a, **k: None)
    monkeypatch.setattr(afm, "gather_success_async", lambda *a, **k: None)
    monkeypatch.setattr(afm, "gather_rcp_msr", lambda *a, **k: None)
    monkeypatch.setattr(
        afm, "run_alarm_cycle",
        lambda eqp_id, recipe_id, settings, tag=None: _cycle(
            outcome_status=outcome_status
        ),
    )
    rows = [{"EQP_ID": "MCD427", "RECIPE_ID": "CLS/RCP", "UTC9": "2026-08-18 10:00:00"}]
    afm.process_fail_rows(rows, active, _settings(), cooldown, attempts)


def test_view_only_goes_to_cooldown_first_time(monkeypatch):
    active, cooldown, attempts = set(), {}, {}
    _run_poll(monkeypatch, active, cooldown, attempts, VIEW_ONLY_OBSERVATION)
    assert "MCD427" not in active
    assert "MCD427" in cooldown
    assert attempts["MCD427"] == 1


def test_view_only_stops_after_max_attempts(monkeypatch):
    """상한 도달 시 active 로 넘겨 cube spam 과 커서 독점을 끊는다."""
    active, cooldown, attempts = set(), {}, {"MCD427": 1}
    _run_poll(monkeypatch, active, cooldown, attempts, VIEW_ONLY_OBSERVATION)
    assert "MCD427" in active
    assert "MCD427" not in cooldown
    assert attempts["MCD427"] == 2


def test_attempts_reset_when_alarm_clears(monkeypatch):
    """알람이 해제되면 카운터를 지운다 - 다음 알람은 처음부터 센다."""
    from poc.workflow_3.monitor import align_fail_monitor as afm

    attempts = {"MCD427": 2}
    afm.process_fail_rows([], set(), _settings(), {}, attempts)
    assert attempts == {}


def test_corrected_still_registers_active(monkeypatch):
    """정상 성공 경로는 그대로 active - 회귀 방지."""
    active, cooldown, attempts = set(), {}, {}
    _run_poll(monkeypatch, active, cooldown, attempts, "corrected")
    assert "MCD427" in active
    assert attempts == {}


# ------------------------------------------------------------------
# 배선 회귀 - 순수 함수가 아니라 step 실행부를 직접 태운다.
# ------------------------------------------------------------------


class _Step:
    step_id = "run_correction"


def test_run_correction_never_calls_corrector_when_occupied(monkeypatch):
    """occupied_by_other 면 correct_align_fail_auto 가 아예 불리면 안 된다.

    순수 함수(resolve_...)만 검증하면 '판정은 맞는데 보정은 그대로 도는' 배선 실수를
    놓친다. view-only 세션에서의 클릭은 장비에 먹지 않으면서 화면만 휘젓는다.
    """
    from poc.workflow_3.monitor import cycle as cyc

    called = []
    monkeypatch.setattr(
        "poc.workflow_3.align.correction.correct_align_fail_auto",
        lambda *a, **k: called.append(1),
        raising=False,
    )
    context = {
        "eqp_id": "MCD427",
        "recipe_id": "CLS/RCP",
        "tag": "t",
        "occupancy": OCCUPIED_BY_OTHER,
    }
    result = cyc._exec_run_correction(_Step(), context, _settings())
    assert called == []
    assert result.status == "success"
    assert context["outcome"].status == VIEW_ONLY_OBSERVATION
    assert context["outcome"].path == "observation"


def test_share_click_converts_image_point_to_screen(monkeypatch):
    """실제 _click 이 image->screen 변환을 거치는지.

    확인 게이트는 이미지 좌표에서 라벨을 읽으므로, 변환을 빠뜨리면 '점 A 를 확인하고
    점 B 를 누르는' 상태가 된다. 오피스 125/150% 배율에서 어긋난 클릭이 하필 강제 종료
    라디오에 떨어질 수 있어, 이 변환은 게이트만큼 중요하다.
    """
    from poc.workflow_3.monitor import cycle as cyc

    popup, image = object(), _FakeImage()
    converted, clicked = [], []

    # 실제 계약대로 (window, title, backend) 3-tuple 을 돌려준다. 창을 꺼내는 지점은
    # occupied_popup.find_select_popup_window 하나뿐이라 그 모듈의 이름을 갈아끼운다
    # (occupied_popup 이 import 시점에 바인딩하므로 util 쪽을 패치하면 안 먹는다).
    monkeypatch.setattr(
        "poc.workflow_3.monitor.occupied_popup.find_window_by_title_prefix",
        lambda prefix, *a, **k: (popup, "Select", "uia"), raising=False,
    )
    monkeypatch.setattr(cyc, "capture_window", lambda w: image)
    monkeypatch.setattr(
        "poc.workflow_3.vlm.ui_venus_mai_locator.analyze_window_target",
        lambda *a, **k: _LocatorResult({"x": 10, "y": 20}), raising=False,
    )
    monkeypatch.setattr(
        "poc.workflow_3.vlm.label_verify.read_text_near_point",
        lambda *a, **k: _OcrRead("Request to share the screen / Request"),
        raising=False,
    )

    def _convert(window, point, image_size=None):
        converted.append((point, image_size))
        return {"x": point["x"] * 2, "y": point["y"] * 2}

    monkeypatch.setattr(cyc, "image_point_to_screen", _convert)
    monkeypatch.setattr(
        cyc, "click_at_screen",
        lambda screen, key, **kw: clicked.append((screen, key)),
    )

    result = cyc._run_share_request(_settings(), "tag")
    assert result.status == "requested"
    # 변환을 거쳤고, 클릭은 변환된 좌표로 나갔다.
    assert converted and converted[0][1] == image.size
    assert clicked == [
        ({"x": 20, "y": 40}, "share_share_screen_radio"),
        ({"x": 20, "y": 40}, "share_request_button"),
    ]


class _FakeImage:
    size = (800, 600)
    width, height = 800, 600


class _LocatorResult:
    def __init__(self, point):
        self.point = point


class _OcrRead:
    def __init__(self, raw_text):
        self.ok = True
        self.raw_text = raw_text
