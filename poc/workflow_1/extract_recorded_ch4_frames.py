"""기록된 CH4 프레임 세트에서 마우스 커서를 추출/탐지하는 entrypoint.

기본 실행:
  uv run python poc/workflow_1/extract_recorded_ch4_frames.py

실제 구현은 `locate_cursor_in_captured_frames.py` 를 재사용한다.
기존 모듈명을 유지해도 되고, 새 워크플로에서는 이 파일명을 우선 사용한다.
"""

from poc.workflow_1.locate_cursor_in_captured_frames import locate_cursors


if __name__ == "__main__":
    raise SystemExit(0 if locate_cursors() == "success" else 1)
