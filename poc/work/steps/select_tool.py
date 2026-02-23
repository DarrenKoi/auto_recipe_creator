"""Standalone step: select one tool from RCS List tab."""

from __future__ import annotations

import os
import sys

try:
    from .shared import prepare_list_tab, prepare_login, run_script
except ImportError:
    from shared import prepare_list_tab, prepare_login, run_script


# Edit these constants if needed.
RUN_LOGIN_FIRST = False
SWITCH_TO_LIST_FIRST = False
TOOL_NAME = os.environ.get("RCS_TOOL_NAME", "").strip()
DOUBLE_CLICK = False
SHOW_LIST_FIRST = False


def main() -> int:
    if RUN_LOGIN_FIRST:
        code = prepare_login()
        if code != 0:
            return code
    if SWITCH_TO_LIST_FIRST:
        code = prepare_list_tab()
        if code != 0:
            return code

    env_overrides: dict[str, str] = {}
    if TOOL_NAME:
        env_overrides["RCS_TOOL_NAME"] = TOOL_NAME
    if DOUBLE_CLICK:
        env_overrides["RCS_SELECT_DOUBLE_CLICK"] = "1"
    if SHOW_LIST_FIRST:
        env_overrides["RCS_SELECT_LIST_FIRST"] = "1"
    return run_script("select_tool.py", env_overrides or None)


if __name__ == "__main__":
    sys.exit(main())
