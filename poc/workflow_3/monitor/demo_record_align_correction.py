"""시연 녹화 (in-tool 편, 장비 지정) - 정해 둔 장비/레시피의 Align Fail 보정을 30fps 영상으로 담는다.

`demo_record_alarm.py` 는 MES 알람이 올 때까지 기다린다 - 어느 장비에 언제 올지 몰라 시연
준비가 어렵다. 이 파일은 `manual_align_correction.py` 와 같은 사이클(**이미 열린 tool 창에
붙어** 보정 -> 닫기)을 **실행 즉시** 돌리고, 녹화도 바로 시작한다. 녹화는 알람 편과 같은
`RecordingHooks` 로 끼운다(포크 금지 - 같은 사이클, 같은 clip 형식).

  * tool 창을 먼저 열고(주 모니터), 그 장비에 Align Fail 다이얼로그가 뜬 상태에서 실행한다.
  * 알람 감지 popup 은 띄우지 않는다 - 이미 tool 안에서 fail 을 기다리는 장면이라 'MES 알람
    감지' 는 맞지 않다(셸 `ALIGN_FAIL_POPUP=1` 이 이긴다). 첫 자막도 이 장면에 맞춘다.
  * OK 뒤 다음 위치에서 다시 fail 이 나면 사이클이 tool 을 닫지 않고 이어서 보정한다
    (`cycle.follow_next_points`, 대기 `ALIGN_FAIL_NEXT_POINT_WAIT_SEC`, 최대 `..._MAX`).
  * 결과: `align_images/_demo/alarm_<EQP>_<tag>/` - polish 의 `{"clip": "alarm_"}` 가 그대로 잇는다.
  * 실운전(실클릭)이 기본. 리허설: 셸 `SAFE_MODE=1`.
  * 주 모니터 전체를 녹화한다 - 터미널은 다른 모니터로, tool 창은 주 모니터에 뜨게 둔다.

사용법 (오피스 Windows, 저장소 루트에서):
  1) 아래 EQP_ID / RECIPE_ID 를 채우고, RCS 에서 그 tool 창을 연다.
  2) uv run python poc/workflow_3/monitor/demo_record_align_correction.py
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import demo_record_alarm as alarm_demo  # noqa: E402
from poc.workflow_3.monitor.manual_align_correction import main  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 이름/의미는 manual_align_correction.py 와 같다.
# 녹화 knob(FPS/MONITOR_INDEX/PRE_ROLL/POPUP_HOLD/TAIL/자막)은 demo_record_alarm.py 상단.
# ===========================================================================

EQP_ID = "MCD513"
RECIPE_ID = "RJ1BXXX/RJ1B_ISOLINERPOLY_R1"   # 반드시 '<class>/<recipe>' 형태
CLASS_NAME = ""      # 선택. 알람 로그/팝업 표시용
TAG = ""             # 선택. 비우면 wall-clock
FALLBACK_SEARCH = 1  # 1 = key 가 안 보이면 주변 탐색, 0 = 첫 판정 뒤 멈춤
AMBIGUITY_NCC_MARGIN = 0  # manual_align_correction.py 주석 참고. 0 = 끔
# 단계 자막 - 알람 편 표에서 'alarm'/'connect_tool' 만 이 장면에 맞게 바꾼다(접속은 이미 돼 있다).
STAGE_SUBTITLES = {
    **alarm_demo.STAGE_SUBTITLES,
    "alarm": "tool 화면에 Align Fail이 뜨면 AI Agent가 바로 이어받아 보정합니다.",
    "connect_tool": "",
}


class _DemoHooks(alarm_demo.RecordingHooks):
    """리허설 여부는 설정 시딩(main 안) 뒤에야 확정된다 - 녹화 시작 시점에 env 로 읽는다."""

    def start(self, eqp_id, info, tag):
        self.rehearsal = (os.environ.get("SAFE_MODE", "0") != "0"
                          or os.environ.get("ALIGN_FAIL_CORRECTION_DRY_RUN", "0") != "0")
        self.subtitles = STAGE_SUBTITLES
        super().start(eqp_id, info, tag)


if __name__ == "__main__":
    # 접속 구간 JPEG 녹화는 이 영상과 같은 장면을 한 번 더 저장할 뿐이라 끈다(셸 env 가 이긴다).
    os.environ.setdefault("ALIGN_FAIL_RECORD_PRELUDE", "0")
    os.environ.setdefault("ALIGN_FAIL_POPUP", "0")  # 이미 tool 안 - 'MES 알람 감지' popup 은 안 맞다.
    raise SystemExit(main(globals(), attach_open_tool=True, alarm_hooks=_DemoHooks(1)))
