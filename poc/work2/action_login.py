"""RCS 로그인 워크플로 entrypoint.

기존 단일 스크립트 로직은 `workflow_login.py` 로 이동했다.
이 파일은 기존 실행 경로를 유지하면서 새 워크플로 러너를 호출한다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/action_login.py
"""

import sys

from poc.work2.workflow_login import EXIT_SUCCESS, main


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
