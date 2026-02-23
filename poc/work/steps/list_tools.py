"""Standalone step: list available tools from RCS List tab."""

from __future__ import annotations

import sys

try:
    from .shared import prepare_list_tab, prepare_login, run_script
except ImportError:
    from shared import prepare_list_tab, prepare_login, run_script


# Edit these constants if needed.
RUN_LOGIN_FIRST = False
SWITCH_TO_LIST_FIRST = False
ENABLE_DEBUG_DUMP = False


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
    if ENABLE_DEBUG_DUMP:
        env_overrides["RCS_LIST_DEBUG"] = "1"
    return run_script("list_up_tools.py", env_overrides or None)


if __name__ == "__main__":
    sys.exit(main())
