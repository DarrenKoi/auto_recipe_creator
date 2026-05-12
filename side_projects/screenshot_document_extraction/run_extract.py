"""Step 2 entry point — 캡처된 페이지를 paddleocr-vl + ui-venus 로 추출한다.

사용법:
  uv run python side_projects/screenshot_document_extraction/run_extract.py
"""

import _bootstrap

_bootstrap.ensure_paths()

from pipeline.extract import extract_runner  # noqa: E402


if __name__ == "__main__":
    extract_runner.run()
