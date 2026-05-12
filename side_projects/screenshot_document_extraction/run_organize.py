"""Step 3 entry point — 페이지별 raw JSON 을 Markdown + JSON 으로 정리한다.

사용법:
  uv run python side_projects/screenshot_document_extraction/run_organize.py
"""

import _bootstrap

_bootstrap.ensure_paths()

from pipeline.organize import organize_runner  # noqa: E402


if __name__ == "__main__":
    organize_runner.run()
