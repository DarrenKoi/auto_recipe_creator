"""시연 영상 마무리 (요약판) - 여는 카드 -> Align Fail 알람 대응 -> Search Around 한 회차.

RCS 순찰(로그인/장비 방문/MemoPrint)은 통째로 빼고, Search Around 는 소개 카드 뒤에 마지막
회차만 잇는다(첫 회차는 너무 빨리 끝나 뺐다 - 그 사이의 "다시 시도" 카드도 같이 빠진다).
항목은 `polish_demo_video.SEQUENCE` / `polish_search_around_video.SEQUENCE` 에서 골라 오므로
(포크 금지) 본편 문구를 고치면 이 요약판에도 그대로 반영된다. 완성본 mp4 를 거치지 않고
clip 에서 바로 조립한다.

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_demo_video_brief.py
  -> align_images/_demo/brief_<시각>.mp4
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402
from poc.workflow_3.monitor import polish_search_around_video as search_around  # noqa: E402
from poc.workflow_3.monitor.screen_video import DEMO_ROOT  # noqa: E402

# 출력 이름은 final_/short_ 로 시작하지 않게 둔다 - {"video": "final_"} 류가 이 파일을 집지 않도록.
OUTPUT = ""  # 비우면 align_images/_demo/brief_<시각>.mp4


def align_fail_only(main: list, search: list) -> list:
    """본편의 여는 카드 + 알람 대응 편(소개 카드부터 끝까지), Search Around 의 소개 카드 + 마지막 회차."""
    first_clip = next(i for i, item in enumerate(main) if "clip" in item)
    alarm = next(i for i, item in enumerate(main) if item.get("clip", "").startswith("alarm_"))
    if alarm > first_clip and "card" in main[alarm - 1]:
        alarm -= 1  # 알람 편 소개 카드
    first_take = next(i for i, item in enumerate(search) if "recording" in item)
    intro = [item for item in search[:first_take] if "card" in item]
    takes = [item for item in search if "recording" in item]
    return main[:first_clip] + main[alarm:] + intro + takes[-1:]


if __name__ == "__main__":
    polish.main(align_fail_only(polish.SEQUENCE, search_around.SEQUENCE),
                OUTPUT or str(DEMO_ROOT / f"brief_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
