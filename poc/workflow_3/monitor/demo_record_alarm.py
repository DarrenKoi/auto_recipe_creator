"""시연 녹화 (in-tool 편) - Align Fail 알람 발생부터 보정 끝까지를 30fps 화면 영상으로 담는다.

`align_fail_monitor` 의 감지 루프를 **그대로** 돈다(포크 금지 - 보정 사이클이 갈리면 시연이
실전과 다른 것을 보여준다). 루프가 알람마다 부르는 `AlarmHooks` 로만 녹화를 끼운다:

  start        알람 감지 직후, popup 전 -> 녹화 시작 (PRE_ROLL_SEC 동안 평상 화면)
  popup_shown  감지 popup 을 띄운 직후 -> "알람 감지" 판독 패널, POPUP_HOLD_SEC 동안 보여준다
  end          사이클이 끝난 뒤(예외여도) -> runner journal 로 단계 시각을 복원하고 녹화 정지

알람 1건 = clip 폴더 1개 `_demo/alarm_<EQP>_<tag>/` (raw.mp4 + events.json + stages.json +
subtitles.json + notes.json + evidence_match.jpg). `polish_demo_video.py` 의 SEQUENCE 에
`{"clip": "alarm_"}` 로 RCS 편 뒤에 잇는다.

  * 알람은 실제로 와야 한다(MES 피드). 리허설은 replay 소스로:
    `ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<csv>` (+ 클릭 차단 `SAFE_MODE=1`).
  * 설정 시딩은 align_fail_monitor __main__ 과 같다(실클릭 기본값 + 상단 상수 + 오피스 사본).
    접속 구간 JPEG 녹화(RECORD_PRELUDE)는 이 영상과 겹치므로 기본으로 끈다.
  * 보정이 실패하면 사이클이 엔지니어 수동 작업까지 녹화한다(engineer watch). 그 clip 에는
    "모든 입력은 Agent" 카드를 붙이지 말 것 - polish 에서 end 를 correction 까지로 자른다.
  * 주 모니터 전체를 녹화한다 - 터미널은 다른 모니터로 치운다.

사용법 (오피스 Windows, 저장소 루트에서):
  uv run python poc/workflow_3/monitor/demo_record_alarm.py
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor import align_fail_monitor as afm  # noqa: E402
from poc.workflow_3.monitor.demo_record_rcs import stage_subtitles  # noqa: E402
from poc.workflow_3.monitor.screen_video import (  # noqa: E402
    DEMO_ROOT,
    NOTES_NAME,
    STAGES_NAME,
    SUBTITLES_NAME,
    ScreenVideoRecorder,
)

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다.
# ===========================================================================

STOP_AFTER = 1        # 이 건수만큼 알람을 녹화하면 루프를 끝낸다. 0 = Ctrl+C 까지 계속
FPS = 30
MONITOR_INDEX = None  # None = 주 모니터
PRE_ROLL_SEC = 1.5    # 녹화 시작 -> popup 사이 평상 화면
POPUP_HOLD_SEC = 3.0  # popup 을 띄운 뒤 사이클 진행을 늦춰 관객이 읽을 시간(시연 전용 지연)
TAIL_SEC = 2.0        # 사이클이 끝난 뒤 더 담는 시간
# 단계별 자막. 키 = 아래 journal_stages 의 stage 이름. 빈 문자열이면 자막 없음. {tool} = 장비.
STAGE_SUBTITLES = {
    "alarm": "Align Fail 알람이 발생하면 Agent가 즉시 감지합니다.",
    "connect_tool": "RCS List에서 알람이 난 {tool} 장비를 찾아 접속합니다.",
    "locate_sem_panel": "tool monitor 화면에서 SEM 영상 영역과 배율을 읽습니다.",
    "correction": "등록된 Align Key를 Computer Vision으로 찾아 중심으로 재정렬합니다.",
    "teardown": "작업을 마치고 tool 창을 닫습니다.",
}
EVIDENCE_NAME = "evidence_match.jpg"  # 보정 근거 정지화면(패턴 매칭 overlay) 사본


# ------------------------------------------------------------------
# 사후 복원 (순수 함수 - test_polish_demo_video.py).
# ------------------------------------------------------------------


def journal_stages(run_dir, cycle, to_video, *, alarm_start: float, tool: str) -> list:
    """runner journal(step_*.json) + CycleResult 시각 -> stages.json 항목(영상 초).

    step 시각은 종료 timestamp(초 해상도) - elapsed 라 최대 1초 어긋난다(polish 가 앞뒤
    STAGE_PAD_SEC 여유를 둔다). 보정 구간은 CycleResult 의 epoch 로 정확하다. OK 클릭과 창
    닫기는 step 이 아니다(OK 는 run_correction 안, 닫기는 teardown) - correction/teardown 으로 묶는다.
    """
    from datetime import datetime

    stages = []
    first_step = None
    for path in sorted(Path(run_dir).glob("step_*.json")) if run_dir else []:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            end = datetime.strptime(str(raw["timestamp"]), "%Y-%m-%dT%H:%M:%S").timestamp()
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        start = end - float(raw.get("elapsed_ms") or 0) / 1000.0
        first_step = start if first_step is None else min(first_step, start)
        stages.append({"stage": str(raw.get("step_id", "")), "detail": tool,
                       "start": round(to_video(start), 2), "end": round(to_video(end), 2)})

    def _span(name, start, end):
        if start is not None and end is not None:
            stages.append({"stage": name, "detail": tool,
                           "start": round(to_video(start), 2), "end": round(to_video(end), 2)})

    cycle_start = getattr(cycle, "started_at", None)
    alarm_end = first_step if first_step is not None else cycle_start
    if alarm_end is not None:
        stages.append({"stage": "alarm", "detail": tool, "start": round(alarm_start, 2),
                       "end": round(to_video(alarm_end), 2)})
    _span("correction", getattr(cycle, "correction_started_at", None),
          getattr(cycle, "correction_finished_at", None))
    _span("teardown", getattr(cycle, "correction_finished_at", None),
          getattr(cycle, "finished_at", None))
    _span("cycle", cycle_start, getattr(cycle, "finished_at", None))
    stages.sort(key=lambda item: item["start"])
    return stages


def outcome_lines(status: str, *, rehearsal: bool = False) -> list:
    """보정 결과 status -> 패널 문구(한글). 실패 사유는 cube 알림과 같은 표를 쓴다.

    rehearsal(SAFE_MODE/보정 dry-run)이면 corrected 여도 클릭은 없었다 - 완료라고 쓰지 않는다.
    """
    from poc.workflow_3.monitor.notify import _UNCORRECTED_ACTIONS

    if rehearsal and status in ("corrected", "awaiting_engineer_ok"):
        return ["Align Key 위치 찾음", "리허설 - 재정렬/OK 클릭은 하지 않음"]
    if status == "corrected":
        return ["Align Key 위치로 재정렬", "OK 까지 자동 완료"]
    if status == "awaiting_engineer_ok":
        return ["Align Key 위치로 재정렬", "OK 는 엔지니어 확인 대기"]
    if status in _UNCORRECTED_ACTIONS:
        return [_UNCORRECTED_ACTIONS[status][0], "자동 보정 보류 → 엔지니어에게 알림"]
    return [f"결과 {status or '-'}"]


def find_evidence(take_dir) -> Path | None:
    """보정이 남긴 패턴 매칭 overlay 중 가장 최근 것(탐색 뒤 재매칭이 있으면 그것)."""
    if not take_dir:
        return None
    found = list(Path(take_dir).glob("debug_images/**/paused_match.jpg"))
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


# ------------------------------------------------------------------
# 녹화 hook.
# ------------------------------------------------------------------


class RecordingHooks(afm.AlarmHooks):
    """알람 1건 = clip 1개. 녹화 실패는 삼킨다(afm._call_hook) - 보정이 우선이다."""

    def __init__(self, stop_after: int, *, rehearsal: bool = False):
        self.stop_after = stop_after
        self.rehearsal = rehearsal
        self.recorded = 0
        self.done = False
        self.subtitles = STAGE_SUBTITLES  # 진입점이 장면에 맞게 바꿔 끼운다(장비 지정 편)
        self._reset()

    def _reset(self):
        self.recorder = None
        self.out_dir = None
        self.notes = []
        self.alarm_start = 0.0

    def _note(self, title, lines, *, image=""):
        now = round(self.recorder.elapsed(), 2)
        self.notes.append({"t": now, "title": title, "lines": list(lines), "boxes": [],
                           **({"image": image} if image else {})})
        print(f"[INFO] [녹화 {now:7.1f}s] 판독 {title}: "
              + " / ".join(lines).replace("→", "->"))

    def start(self, eqp_id, info, tag):
        self._reset()
        self.out_dir = DEMO_ROOT / f"alarm_{eqp_id}_{tag}"
        self.recorder = ScreenVideoRecorder(self.out_dir, fps=FPS,
                                            monitor_index=MONITOR_INDEX).start()
        self.alarm_start = self.recorder.elapsed()
        time.sleep(PRE_ROLL_SEC)

    def popup_shown(self, eqp_id):
        if self.recorder is None:
            return
        self._note("Align Fail 알람 감지", [f"장비 {eqp_id}", "MES 알람 → 자동 대응 시작"])
        time.sleep(POPUP_HOLD_SEC)

    def end(self, eqp_id, info, cycle):
        if self.recorder is None:
            return
        try:
            time.sleep(TAIL_SEC)
            recorder = self.recorder
            run_dir = getattr(cycle, "run_dir", "") or ""
            stages = journal_stages(run_dir, cycle, recorder.video_time,
                                    alarm_start=self.alarm_start, tool=eqp_id)
            started = getattr(cycle, "correction_started_at", None)
            if started is not None:
                self.notes.append({
                    "t": round(recorder.video_time(started), 2), "title": "AI 보정 시작",
                    "lines": ["등록 Align Key ↔ 실시간 화면", "Computer Vision 패턴 매칭으로 위치 탐색"],
                    "boxes": []})
            finished = getattr(cycle, "correction_finished_at", None)
            if finished is not None:
                evidence = find_evidence(Path(run_dir).parents[1] if run_dir else None)
                image = ""
                if evidence is not None:
                    shutil.copyfile(evidence, self.out_dir / EVIDENCE_NAME)
                    image = EVIDENCE_NAME
                self.notes.append({
                    "t": round(recorder.video_time(finished), 2), "title": "보정 결과",
                    "lines": outcome_lines(getattr(cycle, "outcome_status", ""),
                                           rehearsal=self.rehearsal),
                    "boxes": [], **({"image": image} if image else {})})
            self.notes.sort(key=lambda n: n["t"])
            # 사이드카를 녹화 정지보다 먼저 쓴다 - 정지 중 오류가 나도 남는다.
            for name, data in ((STAGES_NAME, stages), (NOTES_NAME, self.notes),
                               (SUBTITLES_NAME, stage_subtitles(stages, self.subtitles))):
                (self.out_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        finally:
            info_out = self.recorder.stop()
            self.recorded += 1
            print("=" * 70)
            print(f"[INFO] 알람 녹화 {self.recorded}건째: {self.out_dir} "
                  f"(결과 {getattr(cycle, 'outcome_status', '') or '-'}, "
                  f"오류 {info_out.get('error') or '-'})")
            print('[INFO] 다음: polish_demo_video.py SEQUENCE 에 {"clip": "alarm_"} 를 잇고 실행')
            print("=" * 70)
            self._reset()
            if self.stop_after and self.recorded >= self.stop_after:
                self.done = True


def main() -> None:
    from poc.workflow_3.config import load_workflow3_settings

    settings = load_workflow3_settings()
    hooks = RecordingHooks(STOP_AFTER,
                           rehearsal=settings.safe_mode or settings.correction_dry_run)
    print(f"[INFO] 알람 시연 녹화 대기 - 알람 {STOP_AFTER or '무제한'}건 녹화 후 종료 "
          f"(폴더 {DEMO_ROOT}/alarm_<EQP>_<tag>/)")
    afm.monitor_loop(settings, alarm_hooks=hooks)


if __name__ == "__main__":
    # 접속 구간 JPEG 녹화는 이 영상과 같은 장면을 한 번 더 저장할 뿐이라 끈다(셸 env 가 이긴다).
    os.environ.setdefault("ALIGN_FAIL_RECORD_PRELUDE", "0")
    afm.seed_main_env()
    main()
