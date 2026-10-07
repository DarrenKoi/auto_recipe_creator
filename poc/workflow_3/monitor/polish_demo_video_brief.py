"""시연 영상 마무리 (요약판) - 여는 카드 -> Align Fail 알람 대응 -> Search Around 한 회차.

여는 카드는 "Align Fail 자동 대응" 한 장뿐이다(본편 첫 장 "엔지니어 개입 없는 AI 기반 자동화" 는 뺐다).

RCS 순찰(로그인/장비 방문/MemoPrint)과 알람 편 소개 카드는 빼고(여는 카드에서 곧바로 알람
clip 으로), Search Around 는 소개 카드 뒤에 마지막
회차만 잇는다(첫 회차는 너무 빨리 끝나 뺐다 - 그 사이의 "다시 시도" 카드도 같이 빠진다).
항목은 `polish_demo_video.SEQUENCE` / `polish_search_around_video.SEQUENCE` 에서 골라 오므로
(포크 금지) 본편 문구를 고치면 이 요약판에도 그대로 반영된다. 완성본 mp4 를 거치지 않고
clip 에서 바로 조립한다. 더 줄이려고 카드/사진은 CARD_SEC 까지만 띄우고 clip/녹화는 SPEED 배속으로 돌린다.

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
CARD_SEC = 4.0  # 카드/사진 노출 상한(초). 0 = 본편 길이 그대로
SPEED = 1.3     # 알람 clip / Search Around 녹화 배속. 1.0 = 본편 속도 그대로
# 사진(cube 알림) 자막 - 본편 3줄은 CARD_SEC 안에 읽기 벅차 2줄로 줄였다. 빈 문자열 = 본편 자막 그대로
IMAGE_SUBTITLE = ("Agent가 처리하지 못하면 곧바로 큐브로 엔지니어에게 인계하고,\n"
                  "학습을 위해 엔지니어의 작업을 녹화합니다.")
# Search Around 소개 카드 - 본편의 두 장(질문 / 알고리즘 설명)을 한 장으로 합쳤다. None = 본편 두 장 그대로
SEARCH_CARD = {"card": "기능 고도화 : Search Around",
               "body": "화면에 Die Fit Target (DFT)가 없으면 Agent가 주변을 탐색해서\n"
                       "찾아내도록 알고리즘을 고도화했습니다.", "sec": 4.0}
# 완성본에서 잘라낼 구간(초, 잘라내기 전 이 요약판 시간축). 요청은 직전 판(첫 카드 4초 포함)의
# 00:20~00:23 - 첫 카드를 뺐으므로 4초 당겨 16~19 다. 다른 값을 고치면 구간이 밀린다. [] = 안 자름
CUT_SEC = [(16.0, 19.0)]


def align_fail_only(main: list, search: list, search_card: dict | None = None) -> list:
    """본편의 여는 카드(첫 장 제외) + 알람 clip 부터 끝까지(소개 카드 없이), Search Around 의 소개 카드 + 마지막 회차."""
    first_clip = next(i for i, item in enumerate(main) if "clip" in item)
    alarm = next(i for i, item in enumerate(main) if item.get("clip", "").startswith("alarm_"))
    first_take = next(i for i, item in enumerate(search) if "recording" in item)
    intro = [search_card] if search_card else [item for item in search[:first_take] if "card" in item]
    takes = [item for item in search if "recording" in item]
    return main[1:first_clip] + main[alarm:] + intro + takes[-1:]


def quicken(sequence: list, max_sec: float, speed: float, image_subtitle: str = "") -> list:
    """카드/사진은 max_sec 로 자르고(0 = 그대로), clip/녹화에는 배속을 붙인다. 사진 자막은 바꿔 쓴다."""
    out = []
    for item in sequence:
        if max_sec and ("card" in item or "image" in item):
            item = {**item, "sec": min(float(item.get("sec", max_sec)), max_sec)}
        elif "clip" in item or "recording" in item:
            item = {**item, "speed": speed}
        if image_subtitle and "image" in item:
            item = {**item, "subtitle": image_subtitle}
        out.append(item)
    return out


if __name__ == "__main__":
    polish.main(quicken(align_fail_only(polish.SEQUENCE, search_around.SEQUENCE, SEARCH_CARD),
                        CARD_SEC, SPEED, IMAGE_SUBTITLE),
                OUTPUT or str(DEMO_ROOT / f"brief_{time.strftime('%y%m%d_%H%M%S')}.mp4"),
                cut_sec=CUT_SEC)
