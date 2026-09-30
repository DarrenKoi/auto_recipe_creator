"""시연 영상 마무리 (단축판) - RCS 순찰에서 첫 장비 방문만 남긴다.

`polish_demo_video.py` 의 SEQUENCE 를 그대로 쓰되, 장비 지정 없는 `visit` clip 을 stages.json
의 첫 visit 장비로 좁힌다. 장비마다 같은 흐름(접속 -> 화면 판독 -> MemoPrint -> 닫기)이 반복되므로
두 번째 장비부터 잘라 재생 시간을 줄인다. 카드/자막/그리기는 전부 `polish_demo_video` 것
(포크 금지) - 본편 SEQUENCE 를 고치면 이 단축판에도 그대로 반영된다.

사용법 (Mac 가능, 오프라인):
  uv run python poc/workflow_3/monitor/polish_demo_video_short.py
  -> align_images/_demo/short_<시각>.mp4
"""

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import polish_demo_video as polish  # noqa: E402
from poc.workflow_3.monitor.screen_video import DEMO_ROOT, STAGES_NAME  # noqa: E402

# 출력 이름은 final_ 로 시작하지 않게 둔다 - polish_search_around_video 의 {"video": "final_"} 가
# 이 파일을 집지 않도록. 단축판 뒤에 Search Around 를 붙이려면 그쪽을 {"video": "short_"} 로 바꾼다.
OUTPUT = ""  # 비우면 align_images/_demo/short_<시각>.mp4


def first_visit_only(sequence: list, stages_of) -> list:
    """장비 지정 없는 visit clip 에 첫 visit 장비를 detail 로 붙인다. stages_of(clip 이름) -> stages."""
    out = []
    for item in sequence:
        if item.get("stage") == "visit" and not item.get("detail"):
            visits = [s for s in stages_of(item["clip"]) if s["stage"] == "visit"]
            if visits:
                first = min(visits, key=lambda s: s["start"])
                print(f"[INFO] {item['clip']} visit: 첫 장비 {first['detail']} 만 남김 "
                      f"(빼는 장비 {', '.join(s['detail'] for s in visits if s is not first) or '없음'})")
                item = {**item, "detail": first["detail"]}
        out.append(item)
    return out


def _stages_of(clip: str) -> list:
    path = polish.resolve_clip_dir(clip) / STAGES_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


if __name__ == "__main__":
    # 녹화가 없는 clip 은 먼저 빼야 stages.json 을 찾다 죽지 않는다(main 이 다시 걸러도 무해).
    sequence = first_visit_only(polish.drop_missing_clips(polish.SEQUENCE), _stages_of)
    polish.main(sequence, OUTPUT or str(DEMO_ROOT / f"short_{time.strftime('%y%m%d_%H%M%S')}.mp4"))
