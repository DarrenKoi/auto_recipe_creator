"""
Workflow 2 package — Align Key 탐색 워크플로우.

`poc/workflow_1/` 이 RCS 로그인 + Tool 선택 + Align Fail 알람 감지 + CH4
프레임 캡처까지 책임진다면, 본 패키지는 **Tool 화면에서 시작하여 SEM monitor
를 조작하면서 레시피의 align key 와 같은 패턴이 보이는 stage / FOV 위치를
찾아내는 흐름** 을 담는다.

핵심 모듈:

- `align_key_matcher.py` — classical CV 기반 (Chamfer + ORB) 매칭 엔진.
  ``docs/search_align_key.md`` §7.6 의 인터페이스 구현.
- `test_align_key_match.py` — 합성 데이터 기반 smoke test.
- `search_align_key.py` — 매칭 엔진을 호출하는 search loop 오케스트레이션.

설계 문서: ``docs/search_align_key.md``, ``docs/test_align_key_match.md``.
"""

from pathlib import Path

WORKFLOW_2_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW_2_DIR / "debug_images"
LOG_DIR = WORKFLOW_2_DIR / "logs"
