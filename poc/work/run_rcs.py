"""Run RCS and auto-login with repository-defined settings."""

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RCS_EXE = r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"
DEFAULT_SERVER = "Dropbox"
DEFAULT_USERNAME = ""
DEFAULT_PASSWORD = ""
DEFAULT_WINDOW_TITLE = ".*RCS|RcsMainHD|Login|로그인.*"
DEFAULT_LAUNCH_TIMEOUT = 30.0
DEFAULT_POST_LOGIN_WAIT = 6.0
LOGIN_HELPER = Path(__file__).with_name("automate_rcs_login.py")


def _get_setting(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def main() -> int:
    exe_path = Path(_get_setting("RCS_EXE_PATH", DEFAULT_RCS_EXE)).expanduser()
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}")
        return 1

    server = _get_setting("RCS_SERVER", DEFAULT_SERVER)
    username = _get_setting("RCS_USERNAME", DEFAULT_USERNAME)
    password = _get_setting("RCS_PASSWORD", DEFAULT_PASSWORD)

    command = [
        sys.executable,
        str(LOGIN_HELPER),
        "--exe",
        str(exe_path),
        "--server",
        server,
        "--window-title",
        DEFAULT_WINDOW_TITLE,
        "--launch-timeout",
        str(DEFAULT_LAUNCH_TIMEOUT),
        "--post-login-wait",
        str(DEFAULT_POST_LOGIN_WAIT),
    ]

    if username:
        command += ["--username", username]
    if password:
        command += ["--password", password]

    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except OSError as exc:
        print(f"[ERROR] Login automation failed to start: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
