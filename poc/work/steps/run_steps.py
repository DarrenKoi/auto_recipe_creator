"""Optional orchestrator that runs multiple standalone steps in order."""

from __future__ import annotations

import os
import sys

try:
    from .shared import run_script
except ImportError:
    from shared import run_script

# Edit these constants if needed.
STEPS = ["login", "switch_tab", "list_tools"]
TOOL_NAME = os.environ.get("RCS_TOOL_NAME", "").strip()
DOUBLE_CLICK = False

STEP_MAP = {
    "login": "login.py",
    "switch_tab": "switch_tab.py",
    "list_tools": "list_tools.py",
    "select_tool": "select_tool.py",
}


def main() -> int:
    for step in STEPS:
        script_name = STEP_MAP.get(step)
        if not script_name:
            print(f"[ERROR] Unknown step: {step}")
            print(f"[INFO] Available steps: {', '.join(STEP_MAP)}")
            return 1

        env_overrides: dict[str, str] = {}
        if step == "select_tool":
            if TOOL_NAME:
                env_overrides["RCS_TOOL_NAME"] = TOOL_NAME
            if DOUBLE_CLICK:
                env_overrides["RCS_SELECT_DOUBLE_CLICK"] = "1"

        code = run_script(f"steps/{script_name}", env_overrides or None)
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    sys.exit(main())
