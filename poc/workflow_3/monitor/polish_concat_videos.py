"""완성된 mp4 두 편을 사이에 소개 카드를 넣어 한 편으로 잇는다.

카드/인코딩은 `polish_demo_video` 것을 그대로 쓴다(포크 금지) - 본편과 같은 카드 모양이 나온다.
프레임을 다시 인코딩하므로 원본의 오디오는 빠지고, 크기가 다르면 letterbox 된다.

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_concat_videos.py
  -> video/SmartAlignment_ARC_PWI_<시각>.mp4
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다.
# ===========================================================================
VIDEO_DIR = _REPO_ROOT / "video"
FIRST = "SmartAlignment_Agent.mp4"
SECOND = "ARC_PWI_자막_배속조정-후반부 2배속.mp4"
CARD = {"card": "PWI Auto Recipe Creation", "body": "", "sec": 4.0}
OUTPUT = ""  # 비우면 video/SmartAlignment_ARC_PWI_<시각>.mp4


def build_sequence(first: Path, second: Path, card: dict) -> list:
    """앞 영상 -> 카드 -> 뒤 영상. 원본이 없으면 조립 전에 멈춘다."""
    for path in (first, second):
        if not path.is_file():
            raise FileNotFoundError(f"이어 붙일 mp4 가 없습니다: {path}")
    return [{"video": str(first)}, card, {"video": str(second)}]


if __name__ == "__main__":
    polish.main(build_sequence(VIDEO_DIR / FIRST, VIDEO_DIR / SECOND, CARD),
                OUTPUT or str(VIDEO_DIR / f"SmartAlignment_ARC_PWI_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
