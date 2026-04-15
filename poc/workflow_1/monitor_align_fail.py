"""Align Fail 모니터링 + RCS 접속 + 화면 녹화 스크립트.

2분 주기로 CD-SEM 알람을 조회하여 ALID=9006 (Align Fail)이 감지되면:
  1. 해당 EQP_ID 의 Tool 에 RCS 로 접속
  2. 화면 녹화를 시작하여 엔지니어 대응 과정을 기록

사용법:
  uv run python poc/workflow_1/monitor_align_fail.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms
from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.record_screen_ch4 import ScreenRecorder
from poc.workflow_1.util import env_int

LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME

# ── 설정 (환경변수) ────────────────────────────────────────────────
POLL_INTERVAL_SEC = env_int("ALIGN_FAIL_POLL_SEC", 120)
MAX_RECORD_SEC = env_int("ALIGN_FAIL_MAX_RECORD_SEC", 600)
RECORDING_DIR = Path(os.getenv(
    "ALIGN_FAIL_RECORDING_DIR",
    str(Path(__file__).resolve().parent / "recordings"),
))
RECORD_FPS = env_int("ALIGN_FAIL_RECORD_FPS", 5)


def connect_rcs_to_tool(eqp_id: str) -> bool:
    """RCS 를 실행하고 해당 EQP_ID Tool 에 접속한다.

    기존 워크플로를 재사용:
      open_rcs → login → select tool
    """
    from poc.workflow_1.open_rcs import launch_rcs, RCS_EXE
    from poc.workflow_1.workflow_login import run_login_workflow

    # 1) RCS 실행
    print(f"[INFO] RCS 실행 시작 (대상 Tool: {eqp_id})")
    try:
        proc = launch_rcs(RCS_EXE)
        if proc is None:
            print("[ERROR] RCS 실행 실패")
            return False
    except Exception as exc:
        print(f"[ERROR] RCS 실행 중 예외: {exc}")
        return False

    # 2) 로그인 + Tool 선택 (run_login_workflow 가 전체 흐름 처리)
    print(f"[INFO] 로그인 워크플로 시작 (target_tool_name={eqp_id})")
    try:
        run = run_login_workflow(target_tool_name=eqp_id)
        if run.status == "completed":
            print(f"[INFO] RCS Tool 접속 성공: {eqp_id}")
            return True
        print(f"[WARNING] 로그인 워크플로 미완료: status={run.status}")
        return False
    except Exception as exc:
        print(f"[ERROR] 로그인 워크플로 중 예외: {exc}")
        return False


def monitor_loop():
    """메인 모니터링 루프.

    2분 주기로 알람을 조회하고, Align Fail 감지 시
    RCS 접속 + 화면 녹화를 시작한다.
    """
    already_handled: set[str] = set()
    active_recorders: dict[str, ScreenRecorder] = {}

    print(f"[INFO] Align Fail 모니터링 시작 (주기={POLL_INTERVAL_SEC}s)")
    print(f"[INFO] 녹화 저장 경로: {RECORDING_DIR}")

    while True:
        try:
            alarms = get_cdsem_alarms()
            fails = filter_align_fail(alarms)

            if fails.empty:
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} — Align Fail 없음")
            else:
                for row in fails.itertuples(index=False):
                    eqp_id = row.EQP_ID
                    alarm_time = getattr(row, "UTC9", None)

                    if eqp_id in already_handled:
                        print(f"[INFO] {eqp_id} 이미 처리됨 — 건너뜀")
                        continue

                    print(f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
                          f"시각={alarm_time}")

                    log_work2_event(
                        component=COMPONENT_NAME,
                        message="align_fail_detected",
                        log_name=LOG_NAME,
                        eqp_id=eqp_id,
                        alarm_time=alarm_time,
                    )

                    # RCS 접속
                    connected = connect_rcs_to_tool(eqp_id)
                    if not connected:
                        print(f"[ERROR] {eqp_id} RCS 접속 실패 — 다음 주기 재시도")
                        continue

                    # 녹화 시작
                    recorder = ScreenRecorder(
                        output_stem=eqp_id,
                        output_dir=RECORDING_DIR,
                        max_record_sec=MAX_RECORD_SEC,
                        fps=RECORD_FPS,
                        log_name=LOG_NAME,
                        component_name=COMPONENT_NAME,
                    )
                    recording_path = recorder.start()
                    if recording_path is None:
                        print(f"[ERROR] {eqp_id} 화면 녹화 시작 실패")
                        continue
                    active_recorders[eqp_id] = recorder

                    already_handled.add(eqp_id)

        except KeyboardInterrupt:
            print("\n[INFO] 모니터링 중단 (Ctrl+C)")
            break
        except Exception as exc:
            print(f"[ERROR] 모니터링 루프 예외: {exc}")

        try:
            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n[INFO] 모니터링 중단 (Ctrl+C)")
            break

    # 종료 시 활성 녹화 정리
    for eqp_id, recorder in active_recorders.items():
        print(f"[INFO] {eqp_id} 녹화 종료 처리 중...")
        recorder.stop()
    print("[INFO] 모니터링 종료")


if __name__ == "__main__":
    monitor_loop()
