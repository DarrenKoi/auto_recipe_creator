"""수동 Align 보정 트리거 - RCS List 탭 접속부터 자동으로 한다.

`manual_align_correction.py` 와 같은 사이클이지만 tool 창을 엔지니어가 먼저 열지 않는다.
알람 사이클과 같은 순서로 RCS 메인 창 확보(없으면 실행+로그인) -> List 탭 점유 게이트 ->
tool 더블클릭 -> 보정 -> tool 창 닫기까지 돈다. 알람 폴링만 우회한다.

본체는 `manual_align_correction.main` 을 attach_open_tool=False 로 부른다(포크 금지 - 두
진입점의 사이클/설정 시딩이 갈리면 같은 시험이 파일마다 다르게 돈다). 이 파일은 상단
상수 블록만 따로 갖는다.

  * 그 tool 창이 이미 열려 있으면 들어가지 않고 종료한다(내 세션이 점유로 읽힌다).
  * 긴급 해제(ctrl+alt+q)로 끝나도 teardown 이 tool 창을 닫는다 - 이 진입점이 연 창이다.
  * 실운전 기본값은 manual 과 같다(SAFE_MODE=0). 점검만 하려면 셸 `SAFE_MODE=1`.

사용법 (venv 활성화 후, 저장소 루트에서):
  1) 아래 EQP_ID / RECIPE_ID 상수를 채운다.
  2) python poc/workflow_3/monitor/manual_align_correction_semiauto.py
"""

import sys
from pathlib import Path

# manual_align_correction.py 와 같은 이유로 저장소 루트를 먼저 얹는다(직접 실행 시 poc 미발견).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor.manual_align_correction import main  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 이름/의미/우선순위(셸 env > 상수)는 manual_align_correction.py 와 같다.
# ===========================================================================

EQP_ID = "MCD513"
RECIPE_ID = "RJ1BXXX/RJ1B_ISOLINERPOLY_R1"   # 반드시 '<class>/<recipe>' 형태
CLASS_NAME = ""      # 선택. 알람 로그/팝업 표시용
TAG = ""             # 선택. 산출물 폴더 tag. 비우면 wall-clock 으로 생성
FALLBACK_SEARCH = 1  # 1 = key 가 안 보이면 주변 탐색, 0 = 첫 판정 뒤 멈춤
AMBIGUITY_NCC_MARGIN = 0  # chamfer 2nd비 모호라도 NCC 차이가 이 이상이면 보정. 0 = 끔(오탐 확인, manual_align_correction.py 주석)


if __name__ == "__main__":
    raise SystemExit(main(globals(), attach_open_tool=False))
