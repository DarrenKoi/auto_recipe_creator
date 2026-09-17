"""이벤트 폴더 배선 테스트 - 알람 1건의 산출물이 `<eqp>-<tag>/` 한 폴더에 모인다.

step executor 는 대역이지만 녹화/캡처 executor 는 진짜를 쓴다(가짜 화면으로). RCS/VLM
없이 Mac 에서 돈다.

`uv run pytest poc/workflow_3/monitor/test_event_folder_wiring.py`
"""

import dataclasses
import sys
import time
from types import SimpleNamespace

from PIL import Image

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.monitor import cycle, notify, recording
from poc.workflow_3.util import event_dir
from poc.workflow_3.util.event_dir import debug_root, read_event_meta


def _fake_screen(monkeypatch):
    fake_capture = lambda _win=None: Image.new("RGB", (64, 48), "white")  # noqa: E731
    monkeypatch.setattr(cycle, "capture_window", fake_capture)
    monkeypatch.setattr(recording, "capture_window", fake_capture)


def _isolate(monkeypatch, tmp_path):
    """이벤트 루트를 tmp 로 돌리고 RCS/알림/창 닫기를 막는다. 반환: 이벤트 루트."""
    events = tmp_path / "align_fail_events"
    monkeypatch.setattr(cycle, "EVENTS_DIR", events)
    monkeypatch.setattr(cycle, "RCS_MODULES_AVAILABLE", True)
    monkeypatch.setattr(cycle, "CLOSE_TOOL_AVAILABLE", False)
    monkeypatch.setattr(cycle, "close_alert_window", lambda *a, **k: True)
    monkeypatch.setattr(notify, "notify_correction_outcome", lambda *a, **k: None)
    monkeypatch.setattr(notify, "send_progress_notify", lambda *a, **k: None)
    _fake_screen(monkeypatch)
    return events


def _fake_executor(step, context, settings):
    """step 마다 콘솔 한 줄 + debug 이미지 한 장을 남기고 성공 조건 키를 채운다."""
    print(f"[INFO] fake executor: {step.step_id}")
    save_debug_jpeg(
        Image.new("RGB", (8, 8)), debug_root() / "align_fail_cycle" / f"{step.step_id}.jpg"
    )
    context.setdefault("rcs_main_window", object())
    context.setdefault("tool_window", object())
    context.setdefault("controller", object())
    if step.step_id == "run_correction":
        context["outcome"] = SimpleNamespace(
            status="corrected", path="primary", key_decision="present", best_xy=(10, 20)
        )
    return cycle._make_result(step, "success", time.time(), settings)


def test_alarm_cycle_keeps_logs_images_and_recording_in_one_folder(tmp_path, monkeypatch):
    events = _isolate(monkeypatch, tmp_path)
    executors = {key: _fake_executor for key in cycle._STEP_EXECUTORS}
    executors["start_recording"] = cycle._exec_start_recording  # 진짜 녹화
    monkeypatch.setattr(cycle, "_STEP_EXECUTORS", executors)
    settings = dataclasses.replace(
        load_workflow3_settings(),
        recording_poll_sec=0.01,
        record_prelude_enabled=False,
        episode_collect_enabled=False,  # Guard/Verification 은 VLM 을 부른다 - 폴더 규약만 본다.
    )
    stdout_before = sys.stdout

    result = cycle.run_alarm_cycle("EQP1", "CLS/RCP", settings, tag="260917_134600", attempt_seq=1)

    take = events / "EQP1-260917_134600" / "attempt_1"
    console = (take / "console.log").read_text(encoding="utf-8")
    assert "[INFO] fake executor: run_correction" in console
    assert "[INFO] step 시작" in console  # runner 의 출력도 같은 파일로 간다.

    runs = list((take / "runs").iterdir())
    assert [run.name.endswith("_align_fail_cycle_EQP1") for run in runs] == [True]
    assert (runs[0] / "step_run_correction.json").is_file()
    assert result.run_dir == str(runs[0])

    assert (take / "debug_images" / "align_fail_cycle" / "run_correction.jpg").is_file()
    assert (take / "recording" / "recording_manifest.json").is_file()
    assert result.recording_dir == str(take / "recording")

    meta = read_event_meta(take)
    assert (meta["eqp_id"], meta["recipe_id"], meta["outcome_status"]) == (
        "EQP1", "CLS/RCP", "corrected"
    )

    # 사이클이 끝나면 전역 상태(콘솔/활성 take)가 돌아온다.
    assert event_dir.active_take_dir() is None
    assert sys.stdout is stdout_before


def test_check_only_cycle_capture_lands_in_the_event_folder(tmp_path, monkeypatch):
    events = _isolate(monkeypatch, tmp_path)
    executors = {key: _fake_executor for key in cycle._CHECK_STEP_EXECUTORS}
    executors["capture_screen"] = cycle._exec_capture_screen  # 진짜 캡처
    monkeypatch.setattr(cycle, "_CHECK_STEP_EXECUTORS", executors)
    monkeypatch.setattr(cycle, "_CHECK_CAPTURE_SETTLE_SEC", 0)

    result = cycle.run_check_only_cycle("EQP2", "", load_workflow3_settings(), tag="260917_140000")

    take = events / "EQP2-260917_140000"
    assert result.outcome_path == str(take / "260917_140000_rcs.jpg")
    assert (take / "260917_140000_rcs.jpg").is_file()
    assert "[INFO] fake executor: wait_tool_window" in (take / "console.log").read_text(encoding="utf-8")
    assert read_event_meta(take)["outcome_status"] == "captured"


def test_monitor_detection_lines_share_the_take_console_log(tmp_path, monkeypatch):
    """알람 감지/다운로드 줄도 사이클과 같은 console.log 에 남는다 - 사이클 전에 연다."""
    from poc.workflow_3.monitor import align_fail_monitor as afm
    from poc.workflow_3.monitor.recovery_episode import EpisodeTracker
    from poc.workflow_3.monitor.test_failure_cooldown import _cycle_returning

    monkeypatch.setattr(cycle, "EVENTS_DIR", tmp_path)
    for name in ("append_alarm_record", "notify_align_fail_popup", "send_detection_notify_async",
                 "gather_success_async", "append_cycle_manifest"):
        monkeypatch.setattr(afm, name, lambda *a, **k: None)
    monkeypatch.setattr(afm, "gather_rcp_msr", lambda *a, **k: print("[INFO] rcp 다운로드 대역"))
    monkeypatch.setattr(afm, "run_alarm_cycle", _cycle_returning(run_status="completed"))
    row = {"EQP_ID": "EQP1", "ALID": "9006", "RECIPE_ID": "CLS/RCP", "UTC9": "2026-08-30 01:02:03",
           "TIMESTAMP": "2026-08-30 01:02:03", "ALARM_NAME": "Align Fail"}

    afm.process_fail_rows([row], set(), load_workflow3_settings(), {}, {},
                          episodes=EpisodeTracker(events_root=tmp_path))

    event = tmp_path / "EQP1-260830_010203"
    console = (event / "attempt_1" / "console.log").read_text(encoding="utf-8")
    assert "Align Fail 감지: EQP_ID=EQP1" in console
    assert "[INFO] rcp 다운로드 대역" in console
    assert (event / "recovery_episode.json").is_file()
    assert event_dir.active_take_dir() is None
