"""시연 녹화 - `demonstration_rcs_control` 시나리오를 30fps 화면 영상으로 담는다.

조작은 기존 시연 스크립트를 그대로 돌린다(포크 금지 - 원격 입력 성사 조건이 갈린다):
RCS 실행 -> 로그인 -> View 탭 훑기 -> List 탭 -> 장비 접속 -> 창 안 조작 -> 닫기.
여기서는 그 앞뒤로 화면 녹화를 켜고 끄고, 단계마다 시작/끝 시각을 `stages.json` 에,
단계별 자막을 `subtitles.json` 에 남긴다. 영상 마무리(커서/클릭 강조/확대/설명 카드/
조각 잇기)는 `polish_demo_video.py` 가 한다 - 이 파일은 원본만 만든다.

  * 로그인 장면부터 담으려면 RCS 를 **닫고** 실행한다(이미 로그인돼 있으면 로그인 생략).
  * 실행 뒤 START_DELAY_SEC 카운트다운 동안 손을 떼면, 끝날 때까지 사람 입력이 필요 없다
    (스스로 끝난다). 영상 첫 카드가 "모든 키보드/마우스 입력은 Agent" 라고 밝히므로 그 말이
    참이 되게 녹화 중에는 건드리지 않는다.
  * 주 모니터 전체를 녹화한다 - 터미널은 다른 모니터로 치운다.
  * 장비/흐름/속도는 demonstration_rcs_control 상단 상수와 `DEMO_RCS_*` env 그대로다.
  * 리허설(클릭 차단): 셸 `SAFE_MODE=1`.

사용법 (오피스 Windows, 저장소 루트에서):
  uv run python poc/workflow_3/monitor/demo_record_rcs.py
  -> align_images/_demo/rcs_<tag>/ 에 raw.mp4 + events.json + stages.json + subtitles.json
     + notes.json(순찰 판독 결과 - polish 가 "AI 판독" 패널과 강조 박스로 그린다)
"""

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.config import load_workflow3_settings  # noqa: E402
from poc.workflow_3.monitor import demonstration_rcs_control as demo  # noqa: E402
from poc.workflow_3.monitor.screen_video import (  # noqa: E402
    DEMO_ROOT,
    NOTES_NAME,
    STAGES_NAME,
    SUBTITLES_NAME,
    ScreenVideoRecorder,
)
from poc.workflow_3.util.time_utils import make_timestamp_tag  # noqa: E402

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다.
# ===========================================================================

# 단계별 자막. 빈 문자열이면 그 단계는 자막 없음. {tool} 은 장비 ID 로 바뀐다.
# 여기서 못 정한 문구는 나중에 polish_demo_video.py 의 SEQUENCE 에서 덮어쓸 수 있다.
STAGE_SUBTITLES = {
    demo.STAGE_LOGIN: "",
    demo.STAGE_VIEW_TAB: "View Tab을 통해 인라인 구성원들이 24시간 Align Fail을 모니터링하고 있습니다.",
    demo.STAGE_LIST_TAB: "",
    demo.STAGE_VISIT: "",
    demo.STAGE_IN_TOOL: "",
}
FPS = 30
MONITOR_INDEX = None  # None = 주 모니터. 다른 화면이면 mss 번호(1, 2, ...)
TAIL_SEC = 2.0       # 시나리오가 끝난 뒤 더 담는 시간
START_DELAY_SEC = 5  # 실행(Enter) 뒤 녹화 시작까지 - 이 사이에 마우스/키보드에서 손을 뗀다


def stage_subtitles(stages: list, texts: dict) -> list:
    """stages.json 항목에 단계별 문구를 붙여 subtitles.json 항목을 만든다."""
    subtitles = []
    for item in stages:
        text = texts.get(item["stage"], "")
        if text:
            subtitles.append({"start": item["start"], "end": item["end"],
                              "text": text.format(tool=item.get("detail", ""))})
    return subtitles


def main() -> int:
    settings = load_workflow3_settings()
    out_dir = DEMO_ROOT / f"rcs_{make_timestamp_tag()}"
    # 녹화 구간은 전부 Agent 입력이어야 한다(영상 첫 카드가 그렇게 밝힌다).
    for remain in range(int(START_DELAY_SEC), 0, -1):
        print(f"[INFO] {remain}초 뒤 녹화+시연 시작 - 마우스/키보드에서 손을 떼세요")
        time.sleep(1)
    recorder = ScreenVideoRecorder(out_dir, fps=FPS, monitor_index=MONITOR_INDEX).start()
    stages, opened = [], {}

    def on_stage(stage, edge, detail):
        now = round(recorder.elapsed(), 2)
        if edge == "start":
            opened[(stage, detail)] = now
        else:
            stages.append({"stage": stage, "detail": detail,
                           "start": opened.pop((stage, detail), now), "end": now})
        print(f"[INFO] [녹화 {now:7.1f}s] {stage} {edge} {detail}")

    def on_note(title, lines, boxes, since_epoch=None):
        now = round(recorder.elapsed(), 2)
        note = {"t": now, "title": title, "lines": list(lines), "boxes": list(boxes)}
        if since_epoch is not None:  # 박스 기준 화면 시각(polish 가 이 화면과 비교한다)
            note["t_ref"] = round(recorder.video_time(since_epoch), 2)
        notes.append(note)
        # 콘솔은 cp949 - 영상용 화살표는 콘솔에서만 ASCII 로.
        print(f"[INFO] [녹화 {now:7.1f}s] 판독 {title}: "
              + " / ".join(lines).replace("→", "->"))

    notes = []
    result, info = None, {}
    try:
        result = demo.main(settings, stage_fn=on_stage, note_fn=on_note)
        time.sleep(TAIL_SEC)
    finally:
        # 단계/자막을 녹화 정지보다 먼저 쓴다 - 정지 중 오류가 나도 사이드카는 남는다.
        stages.sort(key=lambda item: item["start"])
        subtitles = stage_subtitles(stages, STAGE_SUBTITLES)
        (out_dir / STAGES_NAME).write_text(
            json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / SUBTITLES_NAME).write_text(
            json.dumps(subtitles, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / NOTES_NAME).write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
        info = recorder.stop()
        print("=" * 70)
        for item in stages:
            print(f"[INFO]   {item['start']:7.1f}s ~ {item['end']:7.1f}s  "
                  f"{item['stage']} {item['detail']}")
        print(f"[INFO] 녹화 폴더: {out_dir} (자막 {len(subtitles)}개, 판독 {len(notes)}개)")
        print("[INFO] 다음: polish_demo_video.py 상단 SEQUENCE 에 이 폴더/단계를 적고 실행")
        print("=" * 70)
    return 0 if result is not None and not result.aborted and not info.get("error") else 1


if __name__ == "__main__":
    from poc.workflow_3.workflow_3_config_loader import seed_env

    demo._apply_demo_mode_defaults()  # SAFE_MODE=0 기본(실클릭). 오피스 사본보다 먼저.
    seed_env()
    raise SystemExit(main())
