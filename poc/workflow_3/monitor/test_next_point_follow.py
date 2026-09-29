"""OK 뒤 다음 위치 align fail 추적 - wafer 당 OM/SEM 각 2~3 point.

OK 하나로 알람이 끝나지 않는다: 다음 위치에서 다시 align fail 이 뜰 수 있으므로
corrected 뒤에도 tool 을 닫지 않고 다이얼로그 재등장을 기다렸다가 다시 보정한다.

  uv run pytest poc/workflow_3/monitor/test_next_point_follow.py
"""

from dataclasses import dataclass

from poc.workflow_3.monitor.cycle import follow_next_points


@dataclass
class _Outcome:
    status: str
    path: str = "primary"


class _Clock:
    """sleep 이 시계를 전진시키는 가짜 시간 - 실제로 기다리지 않는다."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, sec):
        self.now += sec


def _follow(first, *, probes, corrections, wait_sec=30.0, max_points=5):
    """probes: 호출마다 돌려줄 다이얼로그 유무. 다 쓰면 계속 False(조용함)."""
    clock = _Clock()
    probes, corrections = list(probes), list(corrections)
    corrected_calls = []

    def _probe():
        return probes.pop(0) if probes else False

    def _correct():
        corrected_calls.append(clock.now)
        return corrections.pop(0)

    final = follow_next_points(
        first, probe=_probe, correct=_correct,
        wait_sec=wait_sec, max_points=max_points, poll_sec=5.0,
        clock=clock, sleep=clock.sleep,
    )
    return final, corrected_calls


def test_reappearing_dialog_is_corrected_again():
    """OK 뒤 다음 위치에서 다이얼로그가 다시 뜨면 한 번 더 보정한다."""
    final, calls = _follow(
        _Outcome("corrected"),
        probes=[False, True],
        corrections=[_Outcome("corrected")],
    )

    assert len(calls) == 1
    assert final.status == "corrected"


def test_outcome_without_ok_click_does_not_wait():
    """OK 를 안 누른 결과(에스컬레이션/반자동)는 기다리지 않는다 - cube 가 늦어진다."""
    for status in ("escalated_ambiguous_key", "awaiting_engineer_ok", "align_fail_cleared"):
        final, calls = _follow(_Outcome(status), probes=[True], corrections=[_Outcome("corrected")])

        assert calls == []
        assert final.status == status


def test_quiet_after_ok_ends_without_extra_correction():
    """다음 위치가 정상 align 이면 다이얼로그가 안 뜬다 - wait_sec 뒤 그대로 끝낸다."""
    final, calls = _follow(_Outcome("corrected"), probes=[], corrections=[])

    assert calls == []
    assert final.status == "corrected"


def test_failed_follow_up_becomes_final_outcome():
    """다음 위치 보정이 실패하면 그 결과가 최종이다 - cube + engineer watch 가 나가야 한다."""
    final, calls = _follow(
        _Outcome("corrected"),
        probes=[True, True],
        corrections=[_Outcome("escalated_ambiguous_key"), _Outcome("corrected")],
    )

    assert len(calls) == 1
    assert final.status == "escalated_ambiguous_key"


def test_follow_up_that_finds_dialog_gone_keeps_earlier_success():
    """probe 는 봤는데 보정 직전 재확인에서 사라졌으면 앞 보정 결과를 유지하고 계속 본다.

    cleared 를 최종으로 두면 '이미 해결됨, 직접 확인' cube 가 헛나간다.
    """
    final, calls = _follow(
        _Outcome("corrected"),
        probes=[True, True],
        corrections=[_Outcome("align_fail_cleared"), _Outcome("corrected")],
    )

    assert len(calls) == 2
    assert final.status == "corrected"


def test_dialog_after_max_points_escalates_instead_of_silent_close():
    """상한을 다 쓰고도 다이얼로그가 또 뜨면 corrected 로 닫지 않는다 - 멈춘 장비가 조용히 남는다."""
    final, calls = _follow(
        _Outcome("corrected"),
        probes=[True, True, True],
        corrections=[_Outcome("corrected"), _Outcome("corrected")],
        max_points=2,
    )

    assert len(calls) == 2
    assert final.status == "escalated_next_point_limit"


def test_max_points_zero_is_the_old_close_after_ok():
    """0 = 롤백: 기다리지 않고 종전처럼 OK 뒤 바로 닫는다."""
    final, calls = _follow(_Outcome("corrected"), probes=[True], corrections=[], max_points=0)

    assert calls == []
    assert final.status == "corrected"


# ------------------------------------------------------------------
# 배선 - run_alarm_cycle 이 알림 전에 다음 위치를 따라간다.
# ------------------------------------------------------------------


def test_cycle_follows_next_point_before_notifying(monkeypatch, tmp_path):
    """OK 뒤 다음 위치 fail 이 보정 실패로 끝나면 cube 는 그 결과로 나간다(corrected 로 닫지 않음)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from poc.workflow_3.align import ok_button
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle
    from poc.workflow_3.monitor import notify as ntf

    notified = []
    monkeypatch.setattr(cycle, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(cycle, "RCS_MODULES_AVAILABLE", True)
    monkeypatch.setattr(ntf, "notify_correction_outcome",
                        lambda eqp, rcp, outcome, **k: notified.append(outcome))
    monkeypatch.setattr(ntf, "send_progress_notify", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "close_alert_window", lambda *a, **k: True)
    monkeypatch.setattr(cycle, "log_work2_event", lambda **k: None)

    controller = SimpleNamespace(capture_screen=lambda: None)

    def _run(steps, context, executor):
        context["controller"] = controller
        context["outcome"] = _Outcome("corrected")
        return SimpleNamespace(status="completed", run_dir="", step_results=[])

    monkeypatch.setattr(cycle, "WorkflowRunner", lambda *a, **k: SimpleNamespace(run=_run))

    probes = [ok_button.DIALOG_PRESENT]
    monkeypatch.setattr(
        ok_button, "probe_align_dialog",
        lambda *a, **k: (probes.pop(0) if probes else ok_button.DIALOG_ABSENT, None),
    )
    followed = []

    def _fake_step(step, context, settings):
        followed.append(step.step_id)
        if step.step_id == "run_correction":
            context["outcome"] = _Outcome("escalated_ambiguous_key")
        return SimpleNamespace(status="success")

    monkeypatch.setattr(cycle, "_STEP_EXECUTORS", {
        **cycle._STEP_EXECUTORS,
        "locate_sem_panel": _fake_step,
        "run_correction": _fake_step,
    })
    settings = replace(
        load_workflow3_settings(), next_point_wait_sec=0.0, next_point_max=5,
        engineer_watch_sec=0.0,
    )

    cycle.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="t1")

    assert followed == ["locate_sem_panel", "run_correction"]
    assert [o.status for o in notified] == ["escalated_ambiguous_key"]
