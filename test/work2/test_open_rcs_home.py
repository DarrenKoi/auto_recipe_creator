"""open_rcs home mock 테스트."""

import json
from pathlib import Path

from poc.work2 import open_rcs


class _DummyProc:
    def __init__(self, pid: int, name: str, exe: str):
        self.info = {"pid": pid, "name": name, "exe": exe}


class _FakeLaunchedProcess:
    def __init__(self, pid: int, poll_result):
        self.pid = pid
        self._poll_result = poll_result

    def poll(self):
        return self._poll_result


def test_find_existing_rcs_processes_matches_name_and_exe(monkeypatch):
    exe_path = Path("/tmp/RcsMainHD.exe")

    class FakePsutil:
        NoSuchProcess = RuntimeError
        AccessDenied = PermissionError
        ZombieProcess = RuntimeError

        @staticmethod
        def process_iter(_attrs):
            return [
                _DummyProc(101, "RcsMainHD.exe", ""),
                _DummyProc(202, "other.exe", "/tmp/RcsMainHD.exe"),
                _DummyProc(303, "unrelated.exe", "/tmp/unrelated.exe"),
            ]

    monkeypatch.setattr(open_rcs, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(open_rcs, "psutil", FakePsutil)

    matches = open_rcs.find_existing_rcs_processes(exe_path)

    assert matches == [
        {"pid": 101, "name": "RcsMainHD.exe", "exe": ""},
        {
            "pid": 202,
            "name": "other.exe",
            "exe": "/tmp/RcsMainHD.exe",
        },
    ]


def test_main_returns_already_open_and_writes_state_file(monkeypatch, tmp_path):
    exe_path = tmp_path / "RcsMainHD.exe"
    exe_path.write_text("", encoding="utf-8")
    state_path = tmp_path / "open_rcs_state.json"

    monkeypatch.setattr(open_rcs, "RCS_EXE", exe_path)
    monkeypatch.setattr(open_rcs, "OPEN_RCS_STATE_PATH", state_path)
    monkeypatch.setattr(
        open_rcs,
        "find_existing_rcs_processes",
        lambda _exe_path: [{"pid": 42, "name": "RcsMainHD.exe", "exe": str(exe_path)}],
    )
    monkeypatch.setattr(open_rcs, "log_work2_event", lambda **_kwargs: None)
    monkeypatch.setattr(
        open_rcs,
        "launch_rcs",
        lambda _exe_path: (_ for _ in ()).throw(AssertionError("launch_rcs should not run")),
    )

    result = open_rcs.main()

    assert result == open_rcs.EXIT_ALREADY_OPEN
    assert json.loads(state_path.read_text(encoding="utf-8"))["pid"] == 42
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == open_rcs.EXIT_ALREADY_OPEN


def test_main_returns_early_crash_when_new_process_exits_immediately(monkeypatch, tmp_path):
    exe_path = tmp_path / "RcsMainHD.exe"
    exe_path.write_text("", encoding="utf-8")
    state_path = tmp_path / "open_rcs_state.json"

    monkeypatch.setattr(open_rcs, "RCS_EXE", exe_path)
    monkeypatch.setattr(open_rcs, "OPEN_RCS_STATE_PATH", state_path)
    monkeypatch.setattr(open_rcs, "find_existing_rcs_processes", lambda _exe_path: [])
    monkeypatch.setattr(open_rcs, "log_work2_event", lambda **_kwargs: None)
    monkeypatch.setattr(open_rcs.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        open_rcs,
        "launch_rcs",
        lambda _exe_path: _FakeLaunchedProcess(pid=77, poll_result=3),
    )

    result = open_rcs.main()

    assert result == open_rcs.EXIT_EARLY_CRASH
    assert not state_path.exists()
