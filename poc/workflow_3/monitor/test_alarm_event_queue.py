"""여러 tool 에서 align fail 이 겹칠 때의 알람 큐 회귀.

MES 알람 피드는 **이벤트 로그**다(2026-09-17 사용자 확인): 알람이 해제된 뒤에도 row 가
같은 UTC9 로 계속 남는다. 그래서 "row 가 보인다" 는 "알람이 살아 있다" 가 아니며, 한
사이클이 분 단위로 루프를 막는 동안 새로 뜬 알람은 고정 60s 창 밖으로 밀려 사라졌다.

  uv run pytest poc/workflow_3/monitor/test_alarm_event_queue.py
"""

from datetime import datetime, timedelta

import pandas as pd

from poc.workflow_3.monitor.align_fail_monitor import AlarmFeedCursor

T0 = datetime(2026, 9, 17, 10, 0, 0)


def _feed(*events) -> pd.DataFrame:
    """(EQP_ID, 발생 시각) 목록을 office 피드 모양의 DataFrame 으로 만든다."""
    return pd.DataFrame(
        [
            {"EQP_ID": eqp, "ALID": "9006", "UTC9": at.strftime("%Y-%m-%d %H:%M:%S")}
            for eqp, at in events
        ],
        columns=["EQP_ID", "ALID", "UTC9"],
    )


def _eqps(rows) -> list[str]:
    return list(rows["EQP_ID"])


def test_alarm_raised_during_long_cycle_is_taken_next_poll():
    """10분짜리 사이클 중 2분에 뜬 알람은 다음 poll 에 반드시 잡힌다."""
    cursor = AlarmFeedCursor(window_sec=60)
    cursor.take(_feed(), now=T0)

    feed = _feed(("MCD427", T0 + timedelta(minutes=2)))
    taken = cursor.take(feed, now=T0 + timedelta(minutes=10))

    assert _eqps(taken) == ["MCD427"]


def test_taken_alarm_is_not_taken_again_while_feed_keeps_the_row():
    """해제 뒤에도 row 가 남으므로, 한 번 넘긴 알람을 다시 넘기면 같은 알람을 두 번 처리한다."""
    cursor = AlarmFeedCursor(window_sec=60)
    feed = _feed(("MCD427", T0))
    cursor.take(feed, now=T0 + timedelta(seconds=5))

    again = cursor.take(feed, now=T0 + timedelta(seconds=15))

    assert _eqps(again) == []


def test_second_fail_on_same_tool_is_taken():
    """같은 tool 이 다시 fail 하면 UTC9 가 다른 새 알람이다 - 첫 번째에 가려지면 안 된다."""
    cursor = AlarmFeedCursor(window_sec=60)
    cursor.take(_feed(("MCD427", T0)), now=T0 + timedelta(seconds=5))

    feed = _feed(("MCD427", T0), ("MCD427", T0 + timedelta(seconds=40)))
    taken = cursor.take(feed, now=T0 + timedelta(seconds=45))

    assert list(taken["UTC9"]) == ["2026-09-17 10:00:40"]


def test_startup_ignores_history_older_than_window():
    """피드는 과거 알람을 계속 돌려주므로, 시작 직후 첫 poll 이 옛 알람을 재생하면 안 된다."""
    cursor = AlarmFeedCursor(window_sec=60)
    feed = _feed(("MCD019", T0 - timedelta(hours=3)), ("MCD427", T0 - timedelta(seconds=20)))

    taken = cursor.take(feed, now=T0)

    assert _eqps(taken) == ["MCD427"]


# ------------------------------------------------------------------
# 처리 순서 - 먼저 멈춘 tool 을 먼저.
# ------------------------------------------------------------------


def _settings():
    from dataclasses import replace

    from poc.workflow_3.config import load_workflow3_settings

    return replace(
        load_workflow3_settings(),
        popup_enabled=False,
        rich_notify_enabled=False,
        detection_notify_enabled=False,
        cycle_enabled=True,
    )


def test_tools_are_handled_in_alarm_order_not_name_order(monkeypatch):
    """먼저 멈춘 장비가 더 오래 기다렸다 - EQP_ID 알파벳순은 우연일 뿐이다."""
    from poc.workflow_3.monitor import align_fail_monitor as afm
    from poc.workflow_3.monitor.cycle import CycleResult

    for name in ("append_alarm_record", "append_cycle_manifest",
                 "send_detection_notify_async", "gather_success_async", "gather_rcp_msr"):
        monkeypatch.setattr(afm, name, lambda *a, **k: None)
    handled = []

    def _cycle(eqp_id, recipe_id, settings, tag=None, **kwargs):
        handled.append(eqp_id)
        result = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "")
        result.run_status = "completed"
        return result

    monkeypatch.setattr(afm, "run_alarm_cycle", _cycle)
    feed = _feed(("MCD019", T0 + timedelta(seconds=30)), ("MCDC10", T0))

    afm.process_fail_rows(feed, set(), _settings(), {}, {})

    assert handled == ["MCDC10", "MCD019"]


# ------------------------------------------------------------------
# 접속했더니 align fail 이 이미 없을 때 - 클릭 없이 끝내고 엔지니어에게 알린다.
# ------------------------------------------------------------------


class _Outcome:
    """CorrectionOutcome 중 notify 가 읽는 필드만."""

    def __init__(self, status):
        self.status = status
        self.path = "precheck"
        self.key_decision = ""
        self.best_xy = None
        self.fallback = None
        self.error = None
        self.second_ratio = None


def _cube_summary(monkeypatch, status) -> str:
    from poc.workflow_3.monitor import notify

    sent = []
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True, raising=False)
    monkeypatch.setattr(notify, "_send_cube_async",
                        lambda eqp, rcp, summary: sent.append(summary), raising=False)
    notify.notify_correction_outcome("MCD427", "CLS/RCP", _Outcome(status), enabled=True)
    assert len(sent) == 1, f"{status} 는 cube 가 나가야 한다"
    return sent[0]


def test_cleared_alarm_cube_says_no_dialog_and_asks_to_check_stopped_tool(monkeypatch):
    """VLM 이 진짜 다이얼로그를 놓쳤을 수도 있다 - 그때 아무도 모르면 장비가 멈춘 채 남는다."""
    from poc.workflow_3.monitor.notify import ALIGN_FAIL_CLEARED

    summary = _cube_summary(monkeypatch, ALIGN_FAIL_CLEARED)

    head = summary.split(" | ")[0]
    assert "다이얼로그" in head and "클릭 안 함" in head
    assert "멈춰 있으면" in summary


def test_unconfirmed_alarm_cube_says_no_click_and_asks_for_manual_align(monkeypatch):
    from poc.workflow_3.monitor.notify import ALIGN_FAIL_UNCONFIRMED

    summary = _cube_summary(monkeypatch, ALIGN_FAIL_UNCONFIRMED)

    head = summary.split(" | ")[0]
    assert "확인 못 함" in head and "클릭 안 함" in head
    assert "직접" in summary


def test_engineer_watch_skipped_only_when_corrected_or_already_cleared():
    """이미 해결된 tool 을 최대 6분 녹화하며 붙잡으면 뒤 알람이 그만큼 더 멈춰 있다.

    확인 못 한(unconfirmed) tool 은 진짜 align fail 일 수 있어 엔지니어 조작을 녹화한다.
    """
    from poc.workflow_3.monitor.cycle import needs_engineer_watch
    from poc.workflow_3.monitor.notify import ALIGN_FAIL_CLEARED, ALIGN_FAIL_UNCONFIRMED

    assert needs_engineer_watch(None) is True
    assert needs_engineer_watch(_Outcome("escalated_no_ok")) is True
    assert needs_engineer_watch(_Outcome(ALIGN_FAIL_UNCONFIRMED)) is True
    assert needs_engineer_watch(_Outcome("corrected")) is False
    assert needs_engineer_watch(_Outcome(ALIGN_FAIL_CLEARED)) is False
