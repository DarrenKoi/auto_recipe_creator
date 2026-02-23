"""Run RCS helper script with environment-driven settings (no CLI args)."""

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RCS_EXE = r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"
LOGIN_HELPER = Path(__file__).with_name("automate_rcs_login.py")


def _get_setting(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def main() -> int:
    exe_path = Path(_get_setting("RCS_EXE_PATH", DEFAULT_RCS_EXE)).expanduser()
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}")
        return 1

    command = [sys.executable, str(LOGIN_HELPER)]
    env = os.environ.copy()
    env["RCS_EXE_PATH"] = str(exe_path)

    try:
        result = subprocess.run(command, check=False, env=env)
        return result.returncode
    except OSError as exc:
        print(f"[ERROR] Login automation failed to start: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
