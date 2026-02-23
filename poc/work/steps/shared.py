"""Shared helpers for standalone step runners."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1]


def run_script(script_name: str, env_overrides: dict[str, str] | None = None) -> int:
    """Run a poc/work script with optional environment overrides."""
    script_path = WORK_DIR / script_name
    command = [sys.executable, str(script_path)]
    print(f"[STEP] Running: {' '.join(command)}")
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update(env_overrides)
    result = subprocess.run(command, check=False, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {script_name} (exit={result.returncode})")
    return result.returncode


def prepare_login() -> int:
    """Run login automation as a prerequisite."""
    return run_script("run_rcs.py")


def prepare_list_tab() -> int:
    """Switch to List tab as a prerequisite."""
    return run_script("switching_tabs.py", {"RCS_TAB_NAME": "List"})
