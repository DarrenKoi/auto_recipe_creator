"""Standalone step: switch current RCS tab."""

import sys

from poc.work.steps.shared import prepare_login, run_script


# Edit these constants if needed.
RUN_LOGIN_FIRST = False
TARGET_TAB = "List"


def main() -> int:
    if RUN_LOGIN_FIRST:
        code = prepare_login()
        if code != 0:
            return code
    return run_script("switching_tabs.py", {"RCS_TAB_NAME": TARGET_TAB})


if __name__ == "__main__":
    sys.exit(main())
