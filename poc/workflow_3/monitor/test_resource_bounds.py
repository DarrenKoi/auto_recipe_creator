"""멈춘 오피스 호출의 스레드 상한과 tool 세션 정리 실패를 검증한다."""

import threading
from types import SimpleNamespace

import pytest

from poc.workflow_3.config import Workflow3Settings
from poc.workflow_3.monitor import cycle, notify, rcp_msr_gather, success_gather
from poc.workflow_3.monitor.teardown import run_teardown


def test_cube_senders_share_a_bound_and_recover(monkeypatch):
    """감지/결과 발송이 막혀도 같은 상한을 지키고, 완료 후 슬롯을 재사용한다."""
    release = threading.Event()
    threads = []
    real_thread = threading.Thread

    def make_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(notify, "_CUBE_SEND_SLOTS", threading.BoundedSemaphore(2), raising=False)
    monkeypatch.setattr(notify, "_SEND_CUBE_FN", lambda *a, **k: release.wait(5))
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **k: None)
    monkeypatch.setattr(notify.threading, "Thread", make_thread)
    try:
        for _ in range(12):
            notify._send_cube_async("EQP", "C/R", "result")
            notify.send_detection_notify_async("EQP", "C/R")
        assert len(threads) == 2
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
    notify._send_cube_async("EQP", "C/R", "result")
    threads[-1].join(2)
    assert len(threads) == 3
    assert not any(thread.is_alive() for thread in threads)


@pytest.mark.parametrize("module", [success_gather, rcp_msr_gather])
def test_gather_bound_covers_different_recipes_and_recovers(monkeypatch, module):
    """서로 다른 recipe 의 멈춘 작업도 제한하고, 완료 후 새 recipe 를 받는다."""
    release = threading.Event()
    monkeypatch.setattr(module, "_IN_FLIGHT", {})
    monkeypatch.setattr(module, "MAX_IN_FLIGHT_GATHERS", 2, raising=False)
    settings = Workflow3Settings()

    if module is success_gather:
        def blocked(*args, **kwargs):
            release.wait(5)
            return SimpleNamespace(reason="ok", n_events=1, n_images=1)

        monkeypatch.setattr(module, "DOWNLOADER_AVAILABLE", True)
        monkeypatch.setattr(module, "gather_success_images", blocked)
        call = lambda recipe: module.gather_success_async("EQP", recipe, settings)
    else:
        def blocked(*args, **kwargs):
            release.wait(5)
            return 1

        monkeypatch.setattr(module, "RCP_MSR_DOWNLOADER_AVAILABLE", True)
        monkeypatch.setattr(module, "_call_downloader", blocked)
        call = lambda recipe: module.gather_rcp_msr("EQP", recipe, settings, timeout_sec=0.01)

    try:
        for n in range(12):
            call(f"C/R{n}")
        assert len(module._IN_FLIGHT) == 2
    finally:
        release.set()
        for thread in module._IN_FLIGHT.values():
            thread.join(2)
    if module is success_gather:
        result = call("C/new")
        assert result is not None
        result.join(2)
    else:
        result = module.gather_rcp_msr("EQP", "C/new", settings, timeout_sec=2)
        assert result is True
    assert len(module._IN_FLIGHT) == 1


@pytest.mark.parametrize("progress", [False, True])
def test_cube_limit_does_not_report_a_send(monkeypatch, capsys, progress):
    """포화로 생략한 알림은 발송 로그를 찍지 않고 생략 사유를 남긴다."""
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(notify, "_CUBE_SEND_SLOTS", slots)
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)
    if progress:
        notify.send_progress_notify("EQP", "C/R", 30)
    else:
        notify.notify_correction_outcome("EQP", "C/R", None)
    output = capsys.readouterr().out
    # 진행 고지는 버리고(늦으면 틀린 정보), 결과 알림은 보류했다가 다시 보낸다.
    assert ("발송 생략" if progress else "발송 보류") in output
    assert "발송(비차단)" not in output


@pytest.mark.parametrize("check_only", [False, True])
@pytest.mark.parametrize("exit_code", ["close_failed", "tool_window_not_found", "success"])
def test_teardown_reports_only_failed_tool_close(monkeypatch, check_only, exit_code):
    """닫기 실패는 기록하고, 엔지니어가 이미 닫은 창은 오류로 보고하지 않는다."""
    monkeypatch.setattr(cycle, "CLOSE_TOOL_AVAILABLE", True)
    monkeypatch.setattr(cycle, "close_tool", lambda eqp: SimpleNamespace(exit_code=exit_code))
    monkeypatch.setattr(cycle, "close_alert_window", lambda **kw: None)
    context = {"tool_window": object()}
    settings = Workflow3Settings()
    if check_only:
        steps = cycle._check_teardown_steps("EQP", context, settings, input_blocked=False)
    else:
        result = cycle.CycleResult(eqp_id="EQP", recipe_id="C/R", tag="test")
        steps = cycle._teardown_steps("EQP", context, result, settings,
                                      input_blocked=False, recording=None)
    failures = run_teardown(steps)
    assert [name for name, _ in failures] == (["close_tool"] if exit_code == "close_failed" else [])


def _outbox(notify_module):
    return sorted(notify_module._OUTBOX_DIR.glob("*.json"))


def test_outcome_cube_at_limit_stays_in_outbox_until_delivered(monkeypatch):
    """상한에 걸린 결과 알림은 사라지지 않고, 슬롯이 난 뒤 재발송으로 나간다."""
    release = threading.Event()
    delivered = []
    done = threading.Event()

    def send(eqp, recipe, summary=None):
        if summary == "blocker":
            release.wait(5)
            return
        delivered.append((eqp, summary))
        done.set()

    monkeypatch.setattr(notify, "_CUBE_SEND_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(notify, "_SEND_CUBE_FN", send)
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)

    assert notify._send_cube_async("EQP0", "C/R", "blocker") is True
    # 상한을 넘는 건수가 몰려도(메모리 큐였다면 밀려났을 양) 전부 남는다.
    for n in range(70):
        notify.CycleNotifier(f"EQP{n + 1}", "C/R").notify_outcome(None)
    assert delivered == [] and len(_outbox(notify)) == 70
    assert notify.retry_outbox_once() == 0  # 아직 슬롯이 없다 - 파일은 그대로

    release.set()
    # 마지막 슬롯이 풀린 **뒤** 에 깨우는 신호가 없어도 다음 pump 주기가 가져간다.
    for _ in range(200):
        if not _outbox(notify):
            break
        notify.retry_outbox_once()
        done.wait(0.05)
    assert not _outbox(notify)
    assert sorted(eqp for eqp, _ in delivered) == sorted(f"EQP{n + 1}" for n in range(70))


def test_office_send_exception_keeps_the_message_for_retry(monkeypatch):
    """office 함수가 던지면 알림은 outbox 에 남고, 복구 뒤 재발송된다."""
    state = {"fail": True}
    attempts = []
    finished = threading.Event()

    def send(eqp, recipe, summary=None):
        attempts.append(summary)
        try:
            if state["fail"]:
                raise ConnectionError("cube down")
        finally:
            finished.set()

    monkeypatch.setattr(notify, "_SEND_CUBE_FN", send)
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)

    notify.notify_operator_action("EQP", "C/R", "tool 창 닫기 실패")
    assert finished.wait(2)
    for _ in range(100):  # 발송 스레드가 in-flight 표시를 지울 때까지
        if not notify._OUTBOX_IN_FLIGHT:
            break
        finished.wait(0.01)
    assert len(_outbox(notify)) == 1

    state["fail"] = False
    finished.clear()
    assert notify.retry_outbox_once() == 1
    assert finished.wait(2)
    for _ in range(100):
        if not _outbox(notify):
            break
        finished.wait(0.01)
    assert not _outbox(notify) and len(attempts) == 2


def test_outbox_drops_stale_messages_and_marks_late_ones(monkeypatch):
    """하루 넘게 묵은 알림은 보내지 않고, 늦게 나가는 알림은 지연 시간을 밝힌다."""
    import json
    import time

    sent = []
    done = threading.Event()
    monkeypatch.setattr(notify, "_SEND_CUBE_FN",
                        lambda eqp, rcp, summary=None: (sent.append(summary), done.set()))
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)
    notify._OUTBOX_DIR.mkdir(parents=True)
    for name, age in [("old", 25 * 3600), ("late", 600)]:
        (notify._OUTBOX_DIR / f"{name}.json").write_text(json.dumps(
            {"eqp_id": "EQP", "recipe_id": "C/R", "summary": name, "created": time.time() - age}
        ), encoding="utf-8")
    assert notify.retry_outbox_once() == 1
    assert done.wait(2)
    assert sent == ["[지연 10분] late"]
    assert not (notify._OUTBOX_DIR / "old.json").exists()


def test_notification_never_raises_into_the_cycle_when_threads_cannot_start(monkeypatch):
    """스레드를 못 띄워도(자원 고갈) 알림은 파일로 남고 예외가 사이클 finally 로 새지 않는다."""
    def no_thread(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(notify, "_OUTBOX_PUMP_ENABLED", True)
    monkeypatch.setattr(notify, "_OUTBOX_PUMP", None)
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)
    monkeypatch.setattr(notify.threading, "Thread", no_thread)

    assert notify.CycleNotifier("EQP", "C/R").notify_outcome(None) is True
    notify.notify_operator_action("EQP", "C/R", "tool 창 닫기 실패")
    notify.start_cube_outbox()
    assert len(_outbox(notify)) == 2 and notify._OUTBOX_PUMP is None


def test_outbox_survives_a_locked_rename(monkeypatch):
    """rename 이 거부돼도(Windows 백신 잠금) 알림은 읽을 수 있는 파일로 남는다."""
    def locked(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(notify.os, "replace", locked)
    monkeypatch.setattr(notify.time, "sleep", lambda s: None)
    path = notify._outbox_put("EQP", "C/R", "msg")
    assert path is not None and _outbox(notify) == [path]
    assert not list(notify._OUTBOX_DIR.glob("*.tmp"))


def test_outbox_sweeps_old_junk_but_keeps_fresh_files(monkeypatch):
    """하루 넘은 .tmp / 깨진 파일만 치운다 - 방금 쓰는 중인 파일은 건드리지 않는다."""
    import os
    import time

    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)
    notify._OUTBOX_DIR.mkdir(parents=True)
    old = time.time() - 25 * 3600
    for name, stale in [("a.tmp", True), ("bad.json", True), ("b.tmp", False), ("bad2.json", False)]:
        path = notify._OUTBOX_DIR / name
        path.write_text("{", encoding="utf-8")
        if stale:
            os.utime(path, (old, old))
    assert notify.retry_outbox_once() == 0
    assert sorted(p.name for p in notify._OUTBOX_DIR.iterdir()) == ["b.tmp", "bad2.json"]


def test_stuck_tool_window_is_retried_then_reported_to_the_engineer(monkeypatch):
    """닫기 실패는 한 번 더 시도하고, 그래도 남으면 엔지니어에게 알린다(3 사이클 공통 경로)."""
    from poc.workflow_3e import abort_cycle

    sent = []
    monkeypatch.setattr(notify, "RICH_NOTIFY_AVAILABLE", True)
    monkeypatch.setattr(notify, "log_work2_event", lambda **kw: None)
    monkeypatch.setattr(notify, "_send_cube_async",
                        lambda eqp, rcp, summary, **kw: sent.append((eqp, summary, kw)) or True)
    settings = Workflow3Settings()
    for module, build in [
        (cycle, lambda ctx: cycle._check_teardown_steps("EQP", ctx, settings, input_blocked=False)),
        (abort_cycle, lambda ctx: abort_cycle._abort_teardown_steps(
            "EQP", ctx, settings, input_blocked=False)),
    ]:
        calls = []
        monkeypatch.setattr(module, "CLOSE_TOOL_AVAILABLE", True)
        monkeypatch.setattr(module, "close_tool",
                            lambda eqp: calls.append(eqp) or SimpleNamespace(exit_code="close_failed"))
        monkeypatch.setattr(module, "close_alert_window", lambda **kw: None)
        failures = run_teardown(build({"tool_window": object()}))
        assert len(calls) == 2, module.__name__
        assert [name for name, _ in failures] == ["close_tool"], module.__name__
        sent.clear()
        notify.notify_teardown_failures("EQP", "C/R", failures)
        assert len(sent) == 1 and "닫기 실패" in sent[0][1] and sent[0][2] == {"must_deliver": True}

    sent.clear()
    notify.notify_teardown_failures("EQP", "C/R", [("close_alert", "x")])
    assert sent == []


def test_rcp_gather_limit_tells_the_engineer_when_there_is_no_rcp(monkeypatch, tmp_path):
    """멈춘 다운로더로 상한에 닿으면 조용히 False 가 아니라 조치 알림이 나간다."""
    release = threading.Event()
    told = []
    monkeypatch.setattr(rcp_msr_gather, "_IN_FLIGHT", {})
    monkeypatch.setattr(rcp_msr_gather, "MAX_IN_FLIGHT_GATHERS", 1)
    monkeypatch.setattr(rcp_msr_gather, "ALIGN_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(rcp_msr_gather, "RCP_MSR_DOWNLOADER_AVAILABLE", True)
    monkeypatch.setattr(rcp_msr_gather, "_call_downloader", lambda *a, **k: release.wait(5))
    monkeypatch.setattr(rcp_msr_gather, "log_work2_event", lambda **kw: None)
    monkeypatch.setattr(rcp_msr_gather, "notify_operator_action",
                        lambda eqp, rcp, summary, **kw: told.append((eqp, summary)))
    settings = Workflow3Settings()
    try:
        rcp_msr_gather.gather_rcp_msr("EQP", "C/stuck", settings, timeout_sec=0.01)
        assert rcp_msr_gather.gather_rcp_msr("EQP", "C/new", settings, timeout_sec=0.01) is False
        assert len(told) == 1 and "재시작" in told[0][1]

        # 이미 받아 둔 rcp 가 있으면 보정은 가능하다 - 알림 없이 로그만.
        rcp_dir = tmp_path / "EQP" / "C" / "old" / "align_img_from_rcp"
        rcp_dir.mkdir(parents=True)
        (rcp_dir / "IMAP0001.jpg").write_bytes(b"x")
        assert rcp_msr_gather.gather_rcp_msr("EQP", "C/old", settings, timeout_sec=0.01) is False
        assert len(told) == 1
    finally:
        release.set()
        for thread in rcp_msr_gather._IN_FLIGHT.values():
            thread.join(2)
