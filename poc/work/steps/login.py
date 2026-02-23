"""Standalone step: launch RCS and run login automation."""

import sys

from poc.work.steps.shared import run_script


def main() -> int:
    return run_script("run_rcs.py")


if __name__ == "__main__":
    sys.exit(main())
