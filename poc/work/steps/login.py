"""Standalone step: launch RCS and run login automation."""

from __future__ import annotations

import sys

try:
    from .shared import run_script
except ImportError:
    from shared import run_script


def main() -> int:
    return run_script("run_rcs.py")


if __name__ == "__main__":
    sys.exit(main())
