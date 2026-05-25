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

# Align fail 발생 시 workflow_1 핸들러가 recipe 등록 이미지와 현재 실패 SEM 이미지를
# 내려받는 "designated path". recipe_id 별 서브폴더에 아래 파일들이 들어온다고 가정:
#   recipe_om.*   — recipe 등록 OM align key (주변 layout 포함)
#   recipe_sem.*  — recipe 등록 SEM align key (주변 layout 포함)
#   current_sem.* — fail 시점의 live SEM 모니터 이미지
# 실제 오피스 경로가 다르면 각 스크립트 상단 OVERRIDE 상수로 바꾼다.
ALIGN_FAIL_DOWNLOAD_DIR = WORKFLOW_2_DIR / "align_fail_downloads"

# 내려받은 파일의 표준 stem 이름. 확장자는 자동 탐색(jpg/png/bmp/tif).
RECIPE_OM_STEM = "recipe_om"
RECIPE_SEM_STEM = "recipe_sem"
CURRENT_SEM_STEM = "current_sem"
