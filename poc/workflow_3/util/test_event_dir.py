"""이벤트 폴더 보관 상한 테스트.

`uv run pytest poc/workflow_3/util/test_event_dir.py`
"""

import os

from poc.workflow_3.util.event_dir import prune_events


def _take(root, rel, *, mtime):
    """녹화/debug 이미지/저널/콘솔 로그를 가진 take 하나. 최신 판정은 console.log mtime."""
    take = root / rel
    for sub in ("recording", "debug_images/align_fail_cycle", "runs/260917_130000_cycle"):
        (take / sub).mkdir(parents=True)
    (take / "recording" / "000001_0.jpg").write_bytes(b"jpg")
    (take / "debug_images" / "align_fail_cycle" / "match.jpg").write_bytes(b"jpg")
    (take / "runs" / "260917_130000_cycle" / "step_connect_tool.json").write_text("{}")
    log = take / "console.log"
    log.write_text("2026-09-17 13:00:00 [INFO] x\n", encoding="utf-8")
    os.utime(log, (mtime, mtime))
    return take


def test_older_takes_lose_images_but_keep_text(tmp_path):
    newest = _take(tmp_path, "MCD019-260917_130000", mtime=3000)
    episode = tmp_path / "MCD020-260917_120000"
    second = _take(tmp_path, "MCD020-260917_120000/attempt_2", mtime=2000)
    oldest = _take(tmp_path, "MCD020-260917_120000/attempt_1", mtime=1000)
    (episode / "recovery_episode.json").write_text("{}", encoding="utf-8")

    removed = prune_events(tmp_path, keep_runs=2)

    assert removed == 2
    for kept in (newest, second):
        assert (kept / "recording" / "000001_0.jpg").is_file()
        assert (kept / "debug_images" / "align_fail_cycle" / "match.jpg").is_file()
    assert not (oldest / "recording").exists()
    assert not (oldest / "debug_images").exists()
    # 추적에 필요한 텍스트와 Episode 정본은 남는다.
    assert (oldest / "console.log").is_file()
    assert (oldest / "runs" / "260917_130000_cycle" / "step_connect_tool.json").is_file()
    assert (episode / "recovery_episode.json").is_file()


def test_zero_keeps_everything(tmp_path):
    take = _take(tmp_path, "MCD019-260917_130000", mtime=1000)
    _take(tmp_path, "MCD019-260917_140000", mtime=2000)

    assert prune_events(tmp_path, keep_runs=0) == 0
    assert (take / "recording").is_dir()


def test_missing_root_is_not_an_error(tmp_path):
    assert prune_events(tmp_path / "absent", keep_runs=30) == 0


def test_the_running_take_is_never_pruned_even_if_its_log_looks_oldest(tmp_path):
    """시계가 뒤로 가면 mtime 순서가 뒤집힌다 - 지금 쓰는 take 는 순서와 무관하게 지우지 않는다."""
    from poc.workflow_3.util.event_dir import event_scope

    running = _take(tmp_path, "MCD019-260917_130000", mtime=1000)
    _take(tmp_path, "MCD020-260917_120000", mtime=2000)

    with event_scope(running):
        os.utime(running / "console.log", (1000, 1000))
        prune_events(tmp_path, keep_runs=1)

    assert (running / "recording" / "000001_0.jpg").is_file()
