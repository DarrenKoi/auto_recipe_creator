"""Simple launcher for the Windows RCS executable."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RCS_EXE = r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RCS executable.")
    parser.add_argument(
        "--exe",
        default=os.environ.get("RCS_EXE_PATH", DEFAULT_RCS_EXE),
        help="Path to RcsMainHD.exe",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for process to exit and print return code.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to the executable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe_path = Path(args.exe).expanduser()
    if not exe_path.exists():
        print(f"[ERROR] Executable not found: {exe_path}")
        return 1

    try:
        command = [str(exe_path), *args.extra_args]
        cwd = str(exe_path.parent)
        if args.wait:
            completed = subprocess.run(command, cwd=cwd, check=False)
            print(f"[INFO] Process exited with code: {completed.returncode}")
            return completed.returncode
        else:
            proc = subprocess.Popen(command, cwd=cwd)
            print(f"[INFO] Started RCS: PID={proc.pid}, EXE={exe_path}")
            return 0
    except OSError as exc:
        print(f"[ERROR] Failed to launch executable: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
