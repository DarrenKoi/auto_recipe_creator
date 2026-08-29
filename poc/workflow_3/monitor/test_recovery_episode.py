"""Recovery Episode 수집 테스트 - process_fail_rows 시임에서 관측한다.

Episode 식별은 알람 row 처리의 부수효과이므로, private helper 가 아니라
`process_fail_rows` 가 남긴 **파일**을 본다(spec Testing Decisions). RCS/office 없이
Mac 에서 돌며, ALIGN_IMAGES_DIR 대신 `tmp_path` 를 tracker 에 주입한다.

`uv run pytest poc/workflow_3/monitor/test_recovery_episode.py`
"""

import json

from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult
from poc.workflow_3.monitor.recovery_episode import EpisodeTracker
from poc.workflow_3.monitor.test_failure_cooldown import (
    _cycle_returning,
    _restore,
    _stub_deps,
)


def _row(eqp_id="EQP1", recipe_id="CLS/RCP", utc9="2026-08-30 01:02:03"):
    """replay CSV 와 같은 모양의 알람 row 하나."""
    return {
        "EQP_ID": eqp_id,
        "ALID": "9006",
        "RECIPE_ID": recipe_id,
        "UTC9": utc9,
        "TIMESTAMP": utc9,
        "ALARM_NAME": "Align Fail",
    }


def _episode_files(root):
    """tmp 루트 아래의 recovery_episode.json 전부(정렬)."""
    return sorted(root.rglob("recovery_episode.json"))


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run(tmp_path, rows, *, tracker=None, cycle_fn=None, settings=None):
    """process_fail_rows 를 한 번 돌리고 tracker 를 돌려준다."""
    state = {}
    _stub_deps(state, afm, cycle_fn or _cycle_returning(run_status="completed"))
    try:
        settings = settings or afm.load_workflow3_settings()
        tracker = tracker or EpisodeTracker(images_root=tmp_path)
        afm.process_fail_rows(rows, set(), settings, {}, {}, episodes=tracker)
    finally:
        _restore(state)
    return tracker


def test_alarm_creates_episode_file_with_opaque_identity(tmp_path):
    """알람 1건이 capture 폴더 루트에 Episode 정본을 남긴다 - identity 는 UUID."""
    _run(tmp_path, [_row()])

    files = _episode_files(tmp_path)
    assert len(files) == 1, files
    episode_path = files[0]
    # Episode root = captured_img_from_rcs/<tag> (recipe 는 class/recipe 로 분해).
    assert episode_path.parent.parent.name == "captured_img_from_rcs"
    assert episode_path.parent.parent.parent.name == "RCP"

    data = _read(episode_path)
    assert data["schema_version"] == "recovery_episode.v1"
    assert data["observation_contract"] == "align_fail_observation.v1"
    assert data["bindings_version"] is None
    assert data["execution_mode"] == "live"
    assert data["state"] == "open"
    assert data["outcome"] == "unknown"
    assert data["recovery_actors"] == []

    # identity 는 경로/타임스탬프에서 재구성되지 않는다.
    episode_id = data["episode_id"]
    assert len(episode_id) == 36 and episode_id.count("-") == 4, episode_id
    assert data["tag"] not in episode_id
    assert data["alarm"]["eqp_id"] not in episode_id
    # fingerprint 는 별도 필드다(장비 + alarm code + recipe + 원 UTC9).
    for part in ("EQP1", "9006", "CLS/RCP", "2026-08-30 01:02:03"):
        assert part in data["fingerprint"], (part, data["fingerprint"])


def test_cooldown_retry_reuses_episode_and_increments_attempt(tmp_path):
    """같은 알람의 cooldown 재시도 = 같은 Episode, attempt_seq 1 -> 2."""
    tracker = EpisodeTracker(images_root=tmp_path)
    settings = afm.load_workflow3_settings()
    rows = [_row()]

    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="completed", run_dir="/logs/workflow_runs/run_abc",
        outcome_status="awaiting_engineer_ok", failed_step="", failure_class="",
    ))
    try:
        active, cooldown = set(), {}
        afm.process_fail_rows(rows, active, settings, cooldown, {}, episodes=tracker)
        # 완주한 사이클은 active 로 굳으므로, 재시도를 만들려면 해제 없이 다시 돈다.
        active.clear()
        afm.process_fail_rows(rows, active, settings, cooldown, {}, episodes=tracker)
    finally:
        _restore(state)

    files = _episode_files(tmp_path)
    assert len(files) == 1, files
    data = _read(files[0])
    assert [a["attempt_seq"] for a in data["attempts"]] == [1, 2]
    assert data["state"] == "open"
    first, second = data["attempts"]
    assert first["run_status"] == "completed"
    assert first["outcome_status"] == "awaiting_engineer_ok"
    # run id 만 남기고 경로는 남기지 않는다(runner journal 은 Episode root 밖이다).
    assert first["run_id"] == "run_abc"
    assert "/" not in first["run_id"] and "\\" not in first["run_id"]
    assert first["complete"] is True and second["complete"] is True
    assert data["complete"] is True and data["incomplete_reasons"] == []
    # attempt 폴더는 Episode-relative 로만 가리킨다.
    assert first["artifacts"]["dir"] == "attempt_1"
    assert second["artifacts"]["dir"] == "attempt_2"
    # event_seq 는 Episode 전체를 관통하는 단조 카운터 하나다.
    seqs = [e["event_seq"] for e in data["events"]]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), seqs
    assert [e["kind"] for e in data["events"]] == [
        "attempt_started", "attempt_finished", "attempt_started", "attempt_finished",
    ]


def test_clearance_closes_episode_and_recurrence_gets_new_identity(tmp_path):
    """알람 해제 = clearance 이벤트 + Episode 닫힘. 재발은 다른 episode_id 다."""
    tracker = EpisodeTracker(images_root=tmp_path)
    settings = afm.load_workflow3_settings()

    state = {}
    _stub_deps(state, afm, _cycle_returning(run_status="completed"))
    try:
        active = set()
        afm.process_fail_rows([_row()], active, settings, {}, {}, episodes=tracker)
        # 알람이 poll 에서 사라졌다.
        afm.process_fail_rows([], active, settings, {}, {}, episodes=tracker)
        # 같은 EQP/recipe 가 다시 실패한다(새 알람이라 UTC9 가 다르다).
        afm.process_fail_rows(
            [_row(utc9="2026-08-30 02:03:04")], active, settings, {}, {}, episodes=tracker
        )
    finally:
        _restore(state)

    files = _episode_files(tmp_path)
    assert len(files) == 2, files
    first, second = (_read(p) for p in files)
    assert first["state"] == "closed" and first["closed_at"]
    assert "alarm_cleared" in [e["kind"] for e in first["events"]]
    assert second["state"] == "open"
    assert first["episode_id"] != second["episode_id"]
    assert first["fingerprint"] != second["fingerprint"]


def test_cycle_exception_preserves_episode_as_incomplete(tmp_path):
    """사이클이 예외로 끝나도 파일은 남고 사유가 적힌다 - 삭제하지 않는다."""
    def _boom(eqp_id, recipe_id, settings, tag=None, **_kwargs):
        raise RuntimeError("boom")

    _run(tmp_path, [_row()], cycle_fn=_boom)

    files = _episode_files(tmp_path)
    assert len(files) == 1, files
    data = _read(files[0])
    attempt = data["attempts"][0]
    assert attempt["complete"] is False
    assert "RuntimeError" in attempt["incomplete_reason"]
    assert data["complete"] is False
    assert data["incomplete_reasons"] == [f"attempt_1:{attempt['incomplete_reason']}"]
    assert "attempt_error" in [e["kind"] for e in data["events"]]


def test_rcs_unavailable_attempt_is_incomplete(tmp_path):
    """GUI 를 아예 못 돌린 run_status 는 완주로 보지 않는다(수집 실패가 보여야 한다)."""
    _run(tmp_path, [_row()], cycle_fn=_cycle_returning(run_status="rcs_unavailable"))

    data = _read(_episode_files(tmp_path)[0])
    assert data["attempts"][0]["complete"] is False
    assert data["incomplete_reasons"] == ["attempt_1:run_status:rcs_unavailable"]


def test_load_rejects_absolute_and_escaping_artifact_paths(tmp_path):
    """저장된 artifact 경로는 Episode-relative 만 - 절대 경로와 `..` 는 로드가 거부한다."""
    import pytest

    from poc.workflow_3.monitor.recovery_episode import load_episode

    _run(tmp_path, [_row()])
    path = _episode_files(tmp_path)[0]

    # 정상 파일은 그대로 읽힌다.
    assert load_episode(path)["episode_id"]

    for bad in ("/etc/passwd", "C:\\Windows\\system32", "../../outside/frame.jpg"):
        data = _read(path)
        data["attempts"][0]["artifacts"]["dir"] = bad
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_episode(path)


# ------------------------------------------------------------------
# 티켓 11 - attempt 별 산출물 폴더.
# ------------------------------------------------------------------


def _start_recording(monkeypatch, tmp_path, *, attempt_seq=None, prelude=False):
    """`_exec_start_recording` 을 실제로 돌려 프레임/manifest 를 남기고 세션을 돌려준다."""
    import dataclasses
    import time

    from PIL import Image

    from poc.workflow_3.monitor import cycle, recording

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(
        recording, "capture_window", lambda _win: Image.new("RGB", (64, 48), "white")
    )
    monkeypatch.setattr(
        cycle, "capture_screen", lambda index=1: Image.new("RGB", (64, 48), "black")
    )
    settings = dataclasses.replace(
        afm.load_workflow3_settings(),
        recording_poll_sec=0.01,
        record_prelude_enabled=prelude,
        prelude_poll_sec=0.01,
    )
    context = {
        "eqp_id": "EQP1", "recipe_id": "CLS/RCP", "tag": "T1",
        "tool_window": object(),
    }
    if attempt_seq is not None:
        context["attempt_seq"] = attempt_seq
    if prelude:
        cycle.start_prelude_recording(context, settings)
        time.sleep(0.05)
    step = cycle.build_cycle_steps("EQP1")[4]
    cycle._exec_start_recording(step, context, settings)
    time.sleep(0.05)
    session = context["recording"]
    session.stop("test")
    cycle.stop_prelude_recording(context, "test")
    return session


def test_retries_write_into_separate_attempt_recording_folders(tmp_path, monkeypatch):
    """재시도 2회가 attempt_1/recording 과 attempt_2/recording 으로 갈린다."""
    first = _start_recording(monkeypatch, tmp_path, attempt_seq=1)
    second = _start_recording(monkeypatch, tmp_path, attempt_seq=2)

    episode_root = tmp_path / "EQP1" / "CLS" / "RCP" / "captured_img_from_rcs" / "T1"
    assert first.out_dir == episode_root / "attempt_1" / "recording"
    assert second.out_dir == episode_root / "attempt_2" / "recording"
    for session in (first, second):
        assert (session.out_dir / "recording_manifest.json").is_file()
        assert list(session.out_dir.glob("*.jpg"))
    # 두 테이크의 프레임이 한 폴더에 섞이지 않는다(구 tag 충돌 결함).
    assert not list((episode_root / "recording").glob("*.jpg"))


def test_prelude_goes_under_the_same_attempt_folder(tmp_path, monkeypatch):
    """prelude 는 attempt_<n>/recording/prelude/ 다 - 본 녹화와 같은 attempt 아래."""
    session = _start_recording(monkeypatch, tmp_path, attempt_seq=3, prelude=True)
    prelude_dir = session.out_dir / "prelude"
    assert prelude_dir.is_dir()
    assert list(prelude_dir.glob("*.jpg"))
    # 본 녹화 폴더 직하에 화면 전체 프레임이 섞이면 recording_filter 가 오염된다.
    assert prelude_dir.parent.parent.name == "attempt_3"


def test_collection_off_keeps_the_legacy_tag_recording_folder(tmp_path, monkeypatch):
    """수집 off(attempt_seq 없음)면 종전 <tag>/recording/ 그대로다."""
    session = _start_recording(monkeypatch, tmp_path, attempt_seq=None)
    episode_root = tmp_path / "EQP1" / "CLS" / "RCP" / "captured_img_from_rcs" / "T1"
    assert session.out_dir == episode_root / "recording"


def test_episode_records_attempt_artifacts_as_episode_relative(tmp_path):
    """Episode 파일의 attempt 항목이 자기 폴더/산출물을 Episode-relative 로 가리킨다."""
    from poc.workflow_3.monitor.recovery_episode import load_episode

    tracker = EpisodeTracker(images_root=tmp_path)
    settings = afm.load_workflow3_settings()
    root = tmp_path / "EQP1" / "CLS" / "RCP" / "captured_img_from_rcs" / "260830_010203"

    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="completed",
        recording_dir=str(root / "attempt_1" / "recording"),
        prelude_dir=str(root / "attempt_1" / "recording" / "prelude"),
    ))
    try:
        afm.process_fail_rows([_row()], set(), settings, {}, {}, episodes=tracker)
    finally:
        _restore(state)

    data = load_episode(_episode_files(tmp_path)[0])
    artifacts = data["attempts"][0]["artifacts"]
    assert artifacts["dir"] == "attempt_1"
    assert artifacts["recording"] == "attempt_1/recording"
    assert artifacts["prelude"] == "attempt_1/recording/prelude"
