"""시연 영상 마무리 (초요약판) - 여는 카드 -> Search Around 한 회차 -> cube 알림 사진.

요약판(`polish_demo_video_brief.py`)에서 알람 대응 clip 과 Search Around 소개 카드를 뺐다.
여는 카드 뒤에 곧바로 Search Around 마지막 회차를 자막 한 줄과 함께 틀고, cube 알림 사진을
끝으로 옮겼다(본편/요약판은 사진이 Search Around 앞이다). 항목은 본편 SEQUENCE 에서 골라
오고 배속/노출 상한은 요약판의 `quicken` 을 그대로 쓴다(포크 금지).

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_demo_video_super_brief.py
  -> align_images/_demo/super_brief_<시각>.mp4
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402
from poc.workflow_3.monitor import polish_demo_video_brief as brief  # noqa: E402
from poc.workflow_3.monitor import polish_search_around_video as search_around  # noqa: E402
from poc.workflow_3.monitor.screen_video import DEMO_ROOT  # noqa: E402

# 출력 이름은 final_/short_ 로 시작하지 않게 둔다 - {"video": "final_"} 류가 이 파일을 집지 않도록.
OUTPUT = ""  # 비우면 align_images/_demo/super_brief_<시각>.mp4
CARD_SEC = brief.CARD_SEC  # 카드/사진 노출 상한(초). 0 = 본편 길이 그대로
SPEED = brief.SPEED        # Search Around 녹화 배속. 1.0 = 본편 속도 그대로
SEARCH_SUBTITLE = "Search Around 기능 구현 with Agent"  # 녹화 내내 띄우는 하단 자막
IMAGE_SUBTITLE = brief.IMAGE_SUBTITLE  # 사진(cube 알림) 자막. 빈 문자열 = 본편 자막 그대로


def super_brief(main: list, search: list, subtitle: str) -> list:
    """본편의 여는 카드(첫 장 제외) + Search Around 마지막 회차(자막) + 본편의 사진."""
    first_clip = next(i for i, item in enumerate(main) if "clip" in item)
    take = [item for item in search if "recording" in item][-1]
    return (main[1:first_clip] + [{**take, "subtitle": subtitle}]
            + [item for item in main if "image" in item])


if __name__ == "__main__":
    polish.main(brief.quicken(super_brief(polish.SEQUENCE, search_around.SEQUENCE, SEARCH_SUBTITLE),
                              CARD_SEC, SPEED, IMAGE_SUBTITLE),
                OUTPUT or str(DEMO_ROOT / f"super_brief_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
