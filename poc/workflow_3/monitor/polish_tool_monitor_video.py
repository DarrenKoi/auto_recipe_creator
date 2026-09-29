"""시연 영상 마무리 (tool monitor 편) - Align Fail 보정 녹화만 따로 다듬는다.

`demo_record_align_correction.py`(또는 `demo_record_alarm.py`)가 남긴 `_demo/alarm_<EQP>_<tag>/`
를 카드와 함께 mp4 로 만든다. 편마다 따로 확인·재촬영하고 마지막에 합치려고 파일을 나눴다 -
커서/확대/자막/판독 패널 그리기는 전부 `polish_demo_video` 것을 그대로 쓴다(포크 금지).

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_tool_monitor_video.py
  -> align_images/_demo/tool_monitor_<시각>.mp4
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402
from poc.workflow_3.monitor.screen_video import DEMO_ROOT  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다. 항목 형식은 polish_demo_video.py 상단 SEQUENCE 주석과 같다.
# 화질/속도/확대 등 공통 knob(PREVIEW_WIDTH, ZOOM, IDLE_MAX_SEC ...)은 polish_demo_video.py 상단.
# ===========================================================================

# "alarm_" = 가장 최근 녹화. 특정 take 는 폴더 경로를 적는다.
# stage 에 "teardown" 을 더하면 tool 창 닫기까지 담는다.
SEQUENCE = [
    {"card": "Align Fail 자동 보정",
     "body": "멈춘 장비의 Align을 AI Agent가 직접 찾아 보정합니다", "sec": 4.0},
    {"clip": "alarm_", "stage": ["alarm", "correction"]},
]
OUTPUT = ""  # 비우면 align_images/_demo/tool_monitor_<시각>.mp4


if __name__ == "__main__":
    polish.main(SEQUENCE, OUTPUT or str(DEMO_ROOT / f"tool_monitor_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
