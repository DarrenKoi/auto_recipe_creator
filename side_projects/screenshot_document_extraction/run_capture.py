"""Step 1 entry point — `data/inputs/` 파일을 페이지별 JPEG 로 캡처한다.

사용법:
  uv run python side_projects/screenshot_document_extraction/run_capture.py
"""

import _bootstrap

_bootstrap.ensure_paths()

from pipeline.capture import capture_runner  # noqa: E402


if __name__ == "__main__":
    capture_runner.run()
