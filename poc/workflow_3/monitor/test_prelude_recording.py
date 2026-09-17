"""접속 구간 prelude 녹화(시연용) 배선 테스트 - 실장비/화면 캡처 없이 Mac 에서 실행.

검사 대상은 화면 그랩 자체가 아니라 **게이트/저장 위치/인계 계약**이다:
  * 기본 off 여야 한다 (상시 운전에서 화면 전체를 찍지 않는다)
  * 저장은 본 녹화의 하위 `prelude/` - recording_filter 의 비재귀 glob 에 안 걸린다
  * manifest 에 `capture_source=screen` 과 절대 시작 시각(started_epoch)이 남는다
  * 두 번 멈춰도 안전하다 (인계 시점 + teardown 백스톱)

  uv run pytest poc/workflow_3/monitor/test_prelude_recording.py
"""

import json
import time
from dataclasses import replace

from PIL import Image

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import cycle


def _settings(**overrides):
    # Workflow3Settings 는 frozen dataclass 라 replace 로 파생시킨다.
    return replace(load_workflow3_settings(), **overrides)


def _patch_capture(monkeypatch, out_dir):
    """화면 캡처와 저장 경로를 테스트용으로 갈아끼운다."""
    monkeypatch.setattr(
        cycle, "capture_screen", lambda index=1: Image.new("RGB", (64, 48), "white")
    )
    monkeypatch.setattr(cycle, "_recording_dir_for", lambda *_args: out_dir)


def test_prelude_disabled_by_default(tmp_path, monkeypatch):
    """기본값에서는 세션을 아예 만들지 않는다 (상시 운전 보호)."""
    _patch_capture(monkeypatch, tmp_path / "recording")
    settings = _settings()
    assert settings.record_prelude_enabled is False

    context = {"eqp_id": "EQP1", "recipe_id": "C/R", "tag": "T"}
    assert cycle.start_prelude_recording(context, settings) is None
    assert "prelude_recording" not in context
    print("[OK] test_prelude_disabled_by_default")


def test_prelude_writes_screen_frames_to_subdir(tmp_path, monkeypatch):
    """켜면 recording/prelude/ 에 화면 프레임과 manifest 를 남긴다."""
    recording_dir = tmp_path / "recording"
    _patch_capture(monkeypatch, recording_dir)
    settings = _settings(record_prelude_enabled=True, prelude_poll_sec=0.01)

    context = {"eqp_id": "EQP1", "recipe_id": "C/R", "tag": "T"}
    session = cycle.start_prelude_recording(context, settings)
    assert session is not None
    time.sleep(0.15)
    frames = cycle.stop_prelude_recording(context, "tool_window_open")

    prelude_dir = recording_dir / "prelude"
    assert session.out_dir == prelude_dir
    assert frames >= 1
    # 본 녹화 폴더 직하에는 아무것도 없어야 한다 - recording_filter 는 그 자리만
    # 비재귀로 훑고, 그 파이프라인은 tool 창 rect 프레임을 전제한다.
    assert list(recording_dir.glob("*.jpg")) == []
    assert len(list(prelude_dir.glob("*.jpg"))) == frames

    manifest = json.loads(
        (prelude_dir / "recording_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture_source"] == "screen"
    assert isinstance(manifest["started_epoch"], float)
    assert manifest["stop_reason"] == "tool_window_open"
    print("[OK] test_prelude_writes_screen_frames_to_subdir")


def test_prelude_stop_is_idempotent(tmp_path, monkeypatch):
    """인계 시점에 멈춘 뒤 teardown 백스톱이 또 불러도 0 을 돌려준다."""
    _patch_capture(monkeypatch, tmp_path / "recording")
    settings = _settings(record_prelude_enabled=True, prelude_poll_sec=0.01)

    context = {"eqp_id": "EQP1", "recipe_id": "C/R", "tag": "T"}
    cycle.start_prelude_recording(context, settings)
    time.sleep(0.05)
    first = cycle.stop_prelude_recording(context, "tool_window_open")
    second = cycle.stop_prelude_recording(context, "cycle_teardown")

    assert first >= 1
    assert second == 0
    # 요약은 인계 시점에 적힌 값이 남아야 한다 - teardown 은 그걸 result 로 옮길 뿐.
    assert context["prelude_frame_count"] == first
    print("[OK] test_prelude_stop_is_idempotent")


def test_teardown_reports_prelude_when_connect_fails(tmp_path, monkeypatch):
    """접속이 깨져 인계가 없던 경로도 teardown 이 세션을 멈추고 결과에 적는다."""
    _patch_capture(monkeypatch, tmp_path / "recording")
    settings = _settings(record_prelude_enabled=True, prelude_poll_sec=0.01)

    context = {"eqp_id": "EQP1", "recipe_id": "C/R", "tag": "T"}
    cycle.start_prelude_recording(context, settings)
    time.sleep(0.05)
    result = cycle.CycleResult(eqp_id="EQP1", recipe_id="C/R", tag="T")
    steps = dict(
        cycle._teardown_steps(
            "EQP1", context, result, settings, input_blocked=False, recording=None
        )
    )
    steps["prelude_stop"]()

    assert result.prelude_frame_count >= 1
    assert result.prelude_dir.endswith("prelude")
    assert context.get("prelude_recording") is None
    print("[OK] test_teardown_reports_prelude_when_connect_fails")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    class _Monkey:
        """pytest 없이 직접 실행할 때 쓰는 최소 monkeypatch 대역."""

        def __init__(self):
            self._saved = []

        def setattr(self, target, name, value):
            self._saved.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._saved):
                setattr(target, name, old)

    for _name, _func in sorted(dict(globals()).items()):
        if not _name.startswith("test_") or not callable(_func):
            continue
        with tempfile.TemporaryDirectory() as _tmp:
            _monkey = _Monkey()
            try:
                _func(Path(_tmp), _monkey)
            finally:
                _monkey.undo()
    print("[INFO] prelude 녹화 배선 테스트 완료")


def test_persistent_save_failure_stops_the_session(monkeypatch, tmp_path):
    """캡처는 되는데 저장만 계속 깨지면(디스크 가득 참) 실패 창에 걸려 멈춰야 한다.

    캡처 직후 실패 창을 리셋하던 때는 매 poll 이 '첫 실패' 라 5s 컷오프를 영영 못
    넘고 경고를 초당 20줄씩 max_sec 까지 쏟았다(codex 리뷰 2026-09-17).
    """
    from poc.workflow_3.monitor import recording

    def _disk_full(*_a, **_k):
        raise OSError("No space left on device")

    monkeypatch.setattr(recording, "save_debug_jpeg", _disk_full)
    monkeypatch.setattr(recording, "FAILURE_WINDOW_SEC", 0.2)
    session = recording.RecordingSession(
        None, tmp_path, tag="t", poll_sec=0.02, max_sec=5.0,
        capture_fn=lambda: Image.new("RGB", (64, 48), "white"),
    ).start()
    session._thread.join(timeout=3.0)

    assert not session.is_alive()
    assert session.stop_reason == "window_gone"
    session.stop()


def test_prune_keeps_only_the_newest_runs(tmp_path):
    """recording/ 만 최신 N 개 남긴다 - 같은 run 의 JSON, _manual 세션, rcp 는 그대로."""
    import os

    from poc.workflow_3.monitor.recording import prune_recordings

    def _run(rel, mtime):
        rec = tmp_path / rel / "recording"
        (rec / "prelude").mkdir(parents=True)
        (rec / "f.jpg").write_bytes(b"x")
        (rec.parent / "recovery_episode.json").write_text("{}")
        os.utime(rec, (mtime, mtime))
        return rec

    old = _run("EQ1/C/R/captured_img_from_rcs/t1", 100)
    mid = _run("EQ1/_unregistered/t2/attempt_1", 200)
    new = _run("EQ2/C/R/captured_img_from_rcs/t3/attempt_2", 300)
    manual = _run("EQ1/_manual/t0", 1)
    rcp = tmp_path / "EQ1/C/R/align_img_from_rcp/IMAP0001.jpg"
    rcp.parent.mkdir(parents=True)
    rcp.write_bytes(b"x")

    assert prune_recordings(tmp_path, 0) == []  # 0 = 끔.
    removed = prune_recordings(tmp_path, 2)

    assert removed == [old]
    assert not old.exists() and mid.exists() and new.exists()
    assert (old.parent / "recovery_episode.json").exists()
    assert manual.exists() and rcp.exists()
