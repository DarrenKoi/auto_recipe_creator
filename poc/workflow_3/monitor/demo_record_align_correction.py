"""시연 녹화 (in-tool 편, 장비 지정) - 정해 둔 장비/레시피의 Align Fail 보정을 30fps 영상으로 담는다.

`demo_record_alarm.py` 는 MES 알람이 올 때까지 기다린다 - 어느 장비에 언제 올지 몰라 시연
준비가 어렵다. 이 파일은 `manual_align_correction_semiauto.py` 와 같은 사이클(RCS 확보 ->
List 점유 게이트 -> 더블클릭 -> 보정 -> 닫기)을 **지금 바로** 돌리고, 녹화는 알람 편과 같은
`RecordingHooks` 로 끼운다(포크 금지 - 같은 사이클, 같은 clip 형식).

  * 그 장비에 Align Fail 다이얼로그가 떠 있는 상태에서 실행한다(엔지니어가 레시피로 유도).
  * OK 뒤 다음 위치에서 다시 fail 이 나면 사이클이 tool 을 닫지 않고 이어서 보정한다
    (`cycle.follow_next_points`, 대기 `ALIGN_FAIL_NEXT_POINT_WAIT_SEC`, 최대 `..._MAX`).
  * 결과: `align_images/_demo/alarm_<EQP>_<tag>/` - polish 의 `{"clip": "alarm_"}` 가 그대로 잇는다.
  * 실운전(실클릭)이 기본. 리허설: 셸 `SAFE_MODE=1`.
  * 주 모니터 전체를 녹화한다 - 터미널은 다른 모니터로, tool 창은 주 모니터에 뜨게 둔다.

사용법 (오피스 Windows, 저장소 루트에서):
  1) 아래 EQP_ID / RECIPE_ID 를 채운다.
  2) uv run python poc/workflow_3/monitor/demo_record_align_correction.py
"""

import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor.demo_record_alarm import RecordingHooks  # noqa: E402
from poc.workflow_3.monitor.manual_align_correction import main  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 이름/의미는 manual_align_correction_semiauto.py 와 같다.
# 녹화 knob(FPS/MONITOR_INDEX/PRE_ROLL/POPUP_HOLD/TAIL/자막)은 demo_record_alarm.py 상단.
# ===========================================================================

EQP_ID = "MCD513"
RECIPE_ID = "RJ1BXXX/RJ1B_ISOLINERPOLY_R1"   # 반드시 '<class>/<recipe>' 형태
CLASS_NAME = ""      # 선택. 알람 로그/팝업 표시용
TAG = ""             # 선택. 비우면 wall-clock
FALLBACK_SEARCH = 1  # 1 = key 가 안 보이면 주변 탐색, 0 = 첫 판정 뒤 멈춤
AMBIGUITY_NCC_MARGIN = 0  # manual_align_correction.py 주석 참고. 0 = 끔
START_DELAY_SEC = 5  # 실행 뒤 시작까지 - 이 사이에 마우스/키보드에서 손을 뗀다


class _DemoHooks(RecordingHooks):
    """리허설 여부는 설정 시딩(main 안) 뒤에야 확정된다 - 녹화 시작 시점에 env 로 읽는다."""

    def start(self, eqp_id, info, tag):
        self.rehearsal = (os.environ.get("SAFE_MODE", "0") != "0"
                          or os.environ.get("ALIGN_FAIL_CORRECTION_DRY_RUN", "0") != "0")
        super().start(eqp_id, info, tag)


if __name__ == "__main__":
    # 접속 구간 JPEG 녹화는 이 영상과 같은 장면을 한 번 더 저장할 뿐이라 끈다(셸 env 가 이긴다).
    os.environ.setdefault("ALIGN_FAIL_RECORD_PRELUDE", "0")
    for remain in range(int(START_DELAY_SEC), 0, -1):
        print(f"[INFO] {remain}초 뒤 시작 - 마우스/키보드에서 손을 떼세요")
        time.sleep(1)
    raise SystemExit(main(globals(), attach_open_tool=False, alarm_hooks=_DemoHooks(1)))
