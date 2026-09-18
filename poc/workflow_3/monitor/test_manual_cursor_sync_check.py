"""manual_cursor_sync_check.judge_sync 판정 테스트 (VLM/실장비 불필요)."""

from poc.workflow_3.monitor.manual_cursor_sync_check import judge_sync

TRUTHS = [(100, 100), (900, 100), (900, 700), (100, 700), (500, 400)]


def _probes(found_fn):
    return [{"truth": {"x": x, "y": y}, "found": found_fn(x, y)} for x, y in TRUTHS]


def _judge(probes):
    return judge_sync(probes, sync_dx=50, sync_dy=15, min_found=3)


def test_synced_when_remote_sits_on_local():
    r = _judge(_probes(lambda x, y: {"x": x + 10, "y": y + 12}))
    assert r["verdict"] == "synced" and r["remote_followed"] is True


def test_drifted_with_constant_offset():
    r = _judge(_probes(lambda x, y: {"x": x + 150, "y": y - 90}))
    assert r["verdict"] == "drifted"
    assert r["offset_px"] == {"dx": 150, "dy": -90}
    assert r["remote_followed"] is True


def test_drifted_when_remote_does_not_follow():
    r = _judge(_probes(lambda x, y: {"x": 300, "y": 300}))
    assert r["verdict"] == "drifted" and r["remote_followed"] is False


def test_unknown_when_cursor_mostly_not_found():
    r = _judge(_probes(lambda x, y: {"x": x, "y": y} if x == 100 else None))
    assert r["verdict"] == "unknown" and r["n_found"] == 2


def test_single_decoy_does_not_flip_synced():
    r = _judge(_probes(lambda x, y: {"x": 20, "y": 20} if (x, y) == (500, 400) else {"x": x, "y": y}))
    assert r["verdict"] == "synced" and r["n_drift"] == 1


def test_office_offset_counts_as_synced():
    # 2026-09-18 오피스: 동기화 상태에서 dx~45, dy~12 (사용자 판정 = sync)
    r = _judge(_probes(lambda x, y: {"x": x + 45, "y": y + 12}))
    assert r["verdict"] == "synced"


def test_dy_beyond_limit_is_drift_even_with_small_dx():
    r = _judge(_probes(lambda x, y: {"x": x + 5, "y": y + 20}))
    assert r["verdict"] == "drifted"
