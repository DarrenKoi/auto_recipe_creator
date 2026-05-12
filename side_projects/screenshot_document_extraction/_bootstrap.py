"""entry-point 들이 공통으로 사용하는 sys.path 부트스트랩.

이 프로젝트는 두 영역의 모듈을 import 한다.
- `pipeline.*`   : 이 side project 안의 패키지 (이 파일의 부모 디렉터리)
- `poc.work2.*`  : 리포지토리 루트의 패키지

`uv run python side_projects/screenshot_document_extraction/run_*.py` 처럼
어디서 실행해도 두 import 가 모두 동작하도록 sys.path 를 보정한다.
"""

import sys
from pathlib import Path


PROJECT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PROJECT_DIR.parent.parent


def ensure_paths() -> None:
    """sys.path 에 project 디렉터리와 repo 루트를 추가한다."""
    for candidate in (str(PROJECT_DIR), str(REPO_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
