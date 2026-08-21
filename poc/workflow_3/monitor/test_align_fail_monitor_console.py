"""Align Fail production monitor console noise regression tests.

Polling is intentionally quiet after the current state has already been reported.
Warnings, errors, and state transitions remain visible in their existing tests/paths.
"""

from dataclasses import replace

from poc.workflow_3.monitor import align_fail_monitor as afm


class _AlarmSource:
    """Fixed alarm-source double; office/replay I/O is outside this test boundary."""

    kind = "test"

    def __init__(self, fails):
        self._fails = fails

    def poll(self):
        return None

    def filter_align_fail(self, alarms):
        return self._fails


def _run_two_polls(monkeypatch, fails, *, process_count=None):
    """Run the real loop twice, then stop through its normal Ctrl+C path."""
    settings = replace(
        afm.load_workflow3_settings(),
        detection_window_sec=0,
        keep_awake=False,
    )
    sleeps = 0

    def _sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(afm, "load_alarm_source", lambda _kind: _AlarmSource(fails))
    monkeypatch.setattr(afm, "start_abort_hotkey", lambda _hotkey: None)
    monkeypatch.setattr(afm, "is_aborted", lambda: False)
    monkeypatch.setattr(afm, "_run_rcs_preflight", lambda _settings: None)
    monkeypatch.setattr(afm.time, "sleep", _sleep)
    if process_count is not None:
        monkeypatch.setattr(
            afm,
            "process_fail_rows",
            lambda *_args, **_kwargs: process_count,
        )

    afm.monitor_loop(settings)


def test_idle_state_is_reported_once_without_poll_heartbeat(monkeypatch, capsys):
    """A quiet system reports entering idle, not every underlying database poll."""
    _run_two_polls(monkeypatch, [])

    output = capsys.readouterr().out
    assert output.count("Align Fail 없음") == 1
    assert "알람 조회" not in output


def test_unchanged_active_state_does_not_print_each_poll(monkeypatch, capsys):
    """An already-known active alarm must not bury later warnings in heartbeat lines."""
    fails = [{"EQP_ID": "MCD001", "RECIPE_ID": "CLS/RCP"}]
    _run_two_polls(monkeypatch, fails, process_count=0)

    output = capsys.readouterr().out
    assert "신규 없음" not in output


def test_cooldown_skip_is_quiet(capsys):
    """Cooldown entry already logs its reason and duration; each skipped poll stays quiet."""
    settings = afm.load_workflow3_settings()
    afm.process_fail_rows(
        [{"EQP_ID": "MCD001", "RECIPE_ID": "CLS/RCP"}],
        active_tools=set(),
        settings=settings,
        occupied_cooldown={"MCD001": float("inf")},
    )

    output = capsys.readouterr().out
    assert "cooldown 중" not in output
