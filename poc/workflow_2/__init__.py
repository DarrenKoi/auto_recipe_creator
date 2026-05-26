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

# 오피스 MES 가 align fail 시 생성하는 이미지 루트 (우리 코드가 읽기만 하는 입력).
# 실제 레이아웃:
#   align_images/<eqp_id>/<class_name>/<recipe_name>/
#     ├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe 등록 align key
#     └─ align_img_from_msr/   S*/E*                             # 측정 궤적 (E 접두 = fail step)
# 경로 해석은 `align_fail_assets.py` 가 단일 창구로 담당한다 (최신 자동 + override).
ALIGN_IMAGES_ROOT = WORKFLOW_2_DIR.parent / "workflow_1" / "align_images"

# from_rcp / from_msr 서브폴더명.
FROM_RCP_DIRNAME = "align_img_from_rcp"
FROM_MSR_DIRNAME = "align_img_from_msr"

# from_rcp 안의 표준 stem (IMAP0001=OM, IMAP0002=SEM). 확장자는 자동 탐색.
RCP_OM_STEM = "IMAP0001"
RCP_SEM_STEM = "IMAP0002"
