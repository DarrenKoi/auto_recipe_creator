"""시연 영상 덧붙이기 (Search Around 편) - 완성본 뒤에 align_fail_events 녹화를 자막과 함께 잇는다.

`polish_demo_video.py` 로 만든 `final_<시각>.mp4`(가장 최근)를 그대로 앞에 두고, 이벤트 폴더의
`recording/` 프레임(jpg)을 테스트 하나씩 이어 붙인다. 프레임 잇기/자막/페이드는 전부
`polish_demo_video` 것을 그대로 쓴다(포크 금지).

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_search_around_video.py
  -> align_images/_demo/full_<시각>.mp4
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
# recording 은 align_fail_events 아래 이벤트 폴더 이름(또는 경로). 정지 구간 압축/최소 길이는
# polish_demo_video.py 상단 RECORDING_* 상수.
# ===========================================================================

SEQUENCE = [
    {"video": "final_"},  # 가장 최근 polish 완성본
    {"recording": "MCD026-260929_085628",
     "subtitle": "Search Around 기능으로 Die Fit Target(DFT)이 없어도 주변을 탐색해서 찾아가도록 구현했습니다.",
     "subtitle_sec": 5.0},
    {"card": "주변 탐색",
     "body": "너무 순식간에 끝나 다른 위치에서 다시 시도해보겠습니다.", "sec": 3.5},
    {"recording": "MCD026-260929_085758"},
]
# 출력 이름은 final_ 로 시작하지 않게 둔다 - 다음 실행의 {"video": "final_"} 가 이 파일을 집지 않도록.
OUTPUT = ""  # 비우면 align_images/_demo/full_<시각>.mp4


if __name__ == "__main__":
    polish.main(SEQUENCE, OUTPUT or str(DEMO_ROOT / f"full_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
