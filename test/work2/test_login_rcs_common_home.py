"""login_rcs_common home mock 테스트."""

from poc.work2 import login_rcs_common


def test_ensure_rcs_running_uses_existing_alive_pid(monkeypatch):
    monkeypatch.setattr(login_rcs_common, "_load_open_rcs_pid", lambda: 321)
    monkeypatch.setattr(login_rcs_common, "_load_open_rcs_exe_path", lambda: r"C:\RCS\RcsMainHD.exe")
    monkeypatch.setattr(login_rcs_common, "_is_pid_alive", lambda pid, exe: pid == 321)
    monkeypatch.setattr(
        login_rcs_common,
        "_run_open_rcs_fallback",
        lambda: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    assert login_rcs_common._ensure_rcs_running() == 321


def test_ensure_rcs_running_recovers_via_fallback(monkeypatch):
    pid_values = iter([None, 555])

    monkeypatch.setattr(login_rcs_common, "_load_open_rcs_pid", lambda: next(pid_values))
    monkeypatch.setattr(login_rcs_common, "_load_open_rcs_exe_path", lambda: r"C:\RCS\RcsMainHD.exe")
    monkeypatch.setattr(login_rcs_common, "_is_pid_alive", lambda pid, exe: pid == 555)

    fallback_calls: list[str] = []
    monkeypatch.setattr(
        login_rcs_common,
        "_run_open_rcs_fallback",
        lambda: fallback_calls.append("called"),
    )

    assert login_rcs_common._ensure_rcs_running() == 555
    assert fallback_calls == ["called"]


def test_find_login_window_prefers_pid_scan_and_activates_found_window(monkeypatch):
    fake_window = object()
    activation_calls: list[str] = []

    monkeypatch.setattr(login_rcs_common, "WINDOW_UTILS_AVAILABLE", True)
    monkeypatch.setattr(login_rcs_common, "_ensure_rcs_running", lambda: 9001)
    monkeypatch.setattr(
        login_rcs_common,
        "find_window_by_pid_and_title_prefix",
        lambda pid, prefix, backends, window_filter=None: (
            fake_window,
            "Remote Control System",
            "uia",
        ),
    )
    monkeypatch.setattr(
        login_rcs_common,
        "find_window_by_title_prefix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("desktop scan should not run when pid scan succeeds")
        ),
    )
    monkeypatch.setattr(
        login_rcs_common,
        "activate_window",
        lambda window, debug_label="": activation_calls.append(debug_label),
    )

    window, title, backend = login_rcs_common.find_login_window()

    assert window is fake_window
    assert title == "Remote Control System"
    assert backend == "uia"
    assert activation_calls == [
        "login_window_found_pid_first backend=uia title='Remote Control System'"
    ]
