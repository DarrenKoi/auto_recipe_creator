"""Align Fail 모니터링 + RCS 접속 + 화면 녹화 스크립트.

2분 주기로 CD-SEM 알람을 조회하여 ALID=9006 (Align Fail)이 감지되면:
  1. 해당 EQP_ID 의 Tool 에 RCS 로 접속
  2. 화면 녹화를 시작하여 엔지니어 대응 과정을 기록

사용법:
  uv run python poc/workflow_1/monitor_align_fail.py
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python 미설치 — 화면 녹화 불가")

try:
    import mss
    import numpy as np

    MSS_AVAILABLE = True
except ImportError:
    mss = None
    np = None
    MSS_AVAILABLE = False
    print("[WARNING] mss 미설치 — 화면 캡처 불가")

from poc.workflow_1.office_align_fail_alarm import filter_align_fail, get_cdsem_alarms
from poc.workflow_1.logger import log_work2_event
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


class AlignFailRecorder:
    """백그라운드 스레드에서 화면을 녹화한다."""

    def __init__(self, eqp_id: str, output_dir: Path = RECORDING_DIR):
        self._eqp_id = eqp_id
        self._output_dir = output_dir
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_path: Path | None = None

    def start(self) -> Path:
        """녹화를 시작하고 출력 파일 경로를 반환한다."""
        if not CV2_AVAILABLE or not MSS_AVAILABLE:
            print("[ERROR] 녹화에 필요한 패키지(opencv-python, mss)가 없습니다.")
            return Path("")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        self._output_path = self._output_dir / f"{self._eqp_id}_{ts}.avi"

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"recorder-{self._eqp_id}",
            daemon=True,
        )
        self._thread.start()
        print(f"[INFO] 녹화 시작: {self._output_path}")
        return self._output_path

    def stop(self) -> Path | None:
        """녹화를 중지하고 저장된 파일 경로를 반환한다."""
        if self._thread is None:
            return None
        self._stop_event.set()
        self._thread.join(timeout=5)
        print(f"[INFO] 녹화 종료: {self._output_path}")
        return self._output_path

    def _record_loop(self):
        """실제 캡처 루프 (스레드 내부)."""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # 전체 화면
            width = monitor["width"]
            height = monitor["height"]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                str(self._output_path), fourcc, RECORD_FPS, (width, height)
            )
            frame_interval = 1.0 / RECORD_FPS
            start_time = time.time()

            try:
                while not self._stop_event.is_set():
                    if time.time() - start_time > MAX_RECORD_SEC:
                        print(f"[WARNING] 최대 녹화 시간 초과 ({MAX_RECORD_SEC}s)")
                        break
                    frame_start = time.time()
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    writer.write(frame)
                    elapsed = time.time() - frame_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
            finally:
                writer.release()


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
    active_recorders: dict[str, AlignFailRecorder] = {}

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
                    alarm_time = getattr(row, "CREATE_DTTS", getattr(row, "ALARM_TIME", None))

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
                    recorder = AlignFailRecorder(eqp_id)
                    recorder.start()
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
