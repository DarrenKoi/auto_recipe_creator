"""알람 사이클 녹화의 프레임 사이드카 - 수동 녹화와 같은 래퍼를 쓰는지 본다.

관측 지점은 `RecordingSession` 이 실제로 쓴 **폴더와 manifest** 다(spec Testing
Decisions). 녹화기 자체는 바뀌지 않았다는 것도 여기서 고정한다.

`uv run pytest poc/workflow_3/monitor/test_frame_meta_recorder.py`
"""

import dataclasses
import json
import time

from PIL import Image

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import cycle
from poc.workflow_3.monitor.frame_meta import FRAME_META_FILENAME, FrameMetaRecorder

# 수동 녹화가 쓰는 사이드카 레코드 키 - 알람 녹화도 같은 스키마여야 한다.
_META_KEYS = {
    "frame", "t_sec", "window_rect", "foreground_title",
    "occlusion", "cursor_screen_xy", "cursor_in_window",
}


class _FakeRect:
    left, top, right, bottom = 10, 20, 810, 620


class _FakeWindow:
    """pywinauto 창 대역 - rect 만 답한다."""

    def rectangle(self):
        return _FakeRect()


def _record(tmp_path, monkeypatch, *, attempt_seq=1, episode_id="ep-123"):
    """알람 사이클의 녹화 step 을 실제로 돌리고 (session, out_dir) 를 돌려준다."""
    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    # 녹화 캡처는 cycle 이 주입한 람다를 거친다(사이드카 래퍼가 그것을 감싼다).
    monkeypatch.setattr(
        cycle, "capture_window", lambda _win: Image.new("RGB", (64, 48), "white")
    )
    settings = dataclasses.replace(
        load_workflow3_settings(), recording_poll_sec=0.01,
        episode_collect_enabled=True,
    )
    context = {
        "eqp_id": "EQP1", "recipe_id": "CLS/RCP", "tag": "T1",
        "tool_window": _FakeWindow(),
        "attempt_seq": attempt_seq,
        "episode_id": episode_id,
    }
    step = cycle.build_cycle_steps("EQP1")[4]
    cycle._exec_start_recording(step, context, settings)
    time.sleep(0.08)
    session = context["recording"]
    session.stop("test")
    return session


def _manifest(session):
    return json.loads(
        (session.out_dir / "recording_manifest.json").read_text(encoding="utf-8")
    )


def test_alarm_recording_writes_the_same_sidecar_schema(tmp_path, monkeypatch):
    """알람 녹화도 attempt 의 recording 폴더에 frame_meta.jsonl 을 남긴다."""
    session = _record(tmp_path, monkeypatch)

    meta_path = session.out_dir / FRAME_META_FILENAME
    assert meta_path.is_file()
    lines = [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    for record in lines:
        assert set(record) == _META_KEYS, set(record) ^ _META_KEYS
        assert record["window_rect"] == {
            "left": 10, "top": 20, "right": 810, "bottom": 620
        }
    # 커서는 기록만 한다 - Action/의도 claim 을 만드는 필드가 없어야 한다.
    assert not (_META_KEYS & {"click", "action", "intent", "target"})


def test_manifest_gains_episode_fields_without_losing_legacy_keys(tmp_path, monkeypatch):
    """manifest 는 additive 로 늘어난다 - 새 필드를 모르는 기존 소비자가 계속 읽는다."""
    session = _record(tmp_path, monkeypatch, attempt_seq=2, episode_id="ep-abc")
    manifest = _manifest(session)

    assert manifest["episode_id"] == "ep-abc"
    assert manifest["attempt_seq"] == 2
    completeness = manifest["capture_completeness"]
    assert completeness["meta_enabled"] is True
    assert completeness["meta_disabled_reason"] == ""
    # 사이드카는 **샘플마다** 1줄이고 프레임은 변화가 있을 때만 저장된다. 두 수가
    # 어긋나는 것이 정상이며, 그래서 분석 단계가 seq 가 아니라 t_sec 으로 조인한다.
    assert completeness["meta_records"] == manifest["sampled_count"]
    assert completeness["meta_records"] >= completeness["frames"] > 0

    # make_demo_video / demo_log_panel / recording_filter 가 읽는 기존 키.
    for key in ("tag", "started_at", "started_epoch", "capture_source",
                "frame_count", "sampled_count", "stop_reason"):
        assert key in manifest, key


def test_recording_continues_when_the_sidecar_fails(tmp_path, monkeypatch):
    """사이드카가 깨져도 녹화는 계속되고 사유가 completeness 에 남는다."""
    class _BadWindow:
        def rectangle(self):
            raise RuntimeError("rect gone")

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    # 녹화 캡처는 cycle 이 주입한 람다를 거친다(사이드카 래퍼가 그것을 감싼다).
    monkeypatch.setattr(
        cycle, "capture_window", lambda _win: Image.new("RGB", (64, 48), "white")
    )
    settings = dataclasses.replace(
        load_workflow3_settings(), recording_poll_sec=0.01,
        episode_collect_enabled=True,
    )
    context = {
        "eqp_id": "EQP1", "recipe_id": "CLS/RCP", "tag": "T1",
        "tool_window": _BadWindow(), "attempt_seq": 1, "episode_id": "ep-1",
    }
    step = cycle.build_cycle_steps("EQP1")[4]
    cycle._exec_start_recording(step, context, settings)
    time.sleep(0.08)
    session = context["recording"]
    session.stop("test")

    assert len(session.frames) > 0, "사이드카 실패가 녹화를 멈추면 안 된다"
    completeness = _manifest(session)["capture_completeness"]
    assert completeness["meta_records"] == 0
    assert "rect gone" in completeness["meta_disabled_reason"]


def test_sidecar_warns_once_then_stays_disabled(capsys, tmp_path):
    """지속 실패에도 경고는 1회다 - 20fps 루프에서 콘솔이 도배되면 안 된다."""
    class _BadWindow:
        def rectangle(self):
            raise RuntimeError("nope")

    recorder = FrameMetaRecorder(_BadWindow(), tmp_path, our_handles=set())
    capture = recorder.wrap(lambda: "IMAGE")
    for _ in range(5):
        assert capture() == "IMAGE"
    recorder.close()

    warnings = [
        line for line in capsys.readouterr().out.splitlines()
        if "frame_meta" in line and "[WARNING]" in line
    ]
    assert len(warnings) == 1, warnings
    assert recorder.completeness()["meta_records"] == 0


def test_sidecar_is_off_when_episode_collection_is_off(tmp_path, monkeypatch):
    """수집 off 면 사이드카도 manifest 확장도 없다 - 녹화 동작이 종전과 같아야 한다."""
    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(
        cycle, "capture_window", lambda _win: Image.new("RGB", (64, 48), "white")
    )
    settings = dataclasses.replace(
        load_workflow3_settings(), recording_poll_sec=0.01,
        episode_collect_enabled=False,
    )
    assert load_workflow3_settings().episode_collect_enabled is False, "기본값은 off 여야 한다"
    context = {"eqp_id": "EQP1", "recipe_id": "CLS/RCP", "tag": "T1",
               "tool_window": _FakeWindow()}
    step = cycle.build_cycle_steps("EQP1")[4]
    cycle._exec_start_recording(step, context, settings)
    time.sleep(0.05)
    session = context["recording"]
    session.stop("test")

    assert not (session.out_dir / FRAME_META_FILENAME).exists()
    manifest = _manifest(session)
    assert "episode_id" not in manifest
    assert "capture_completeness" not in manifest
