"""Channel 4 CCTV 선택 이후 전체 화면을 녹화한다.

사용법:
  uv run python poc/workflow_1/record_screen_ch4.py
"""

import os
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

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

from poc.workflow_1.logger import log_work2_event
from poc.workflow_1.util import env_int

load_dotenv()

LOG_NAME = "record_screen_ch4"
COMPONENT_NAME = LOG_NAME
DEFAULT_OUTPUT_STEM = "ch4_cctv"
RECORDING_DIR = Path(
    os.getenv(
        "CH4_RECORDING_DIR",
        str(Path(__file__).resolve().parent / "recordings" / "ch4"),
    ),
)
MAX_RECORD_SEC = env_int("CH4_MAX_RECORD_SEC", 600)
RECORD_FPS = env_int("CH4_RECORD_FPS", 5)


def _sanitize_output_stem(output_stem: str) -> str:
    """파일명에 안전한 출력 prefix 를 만든다."""
    safe_chars: list[str] = []
    for char in (output_stem or "").strip():
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")

    safe_name = "".join(safe_chars).strip("-._")
    return safe_name or DEFAULT_OUTPUT_STEM


class ScreenRecorder:
    """백그라운드 스레드에서 전체 화면을 녹화한다."""

    def __init__(
        self,
        output_stem: str = DEFAULT_OUTPUT_STEM,
        *,
        output_dir: Path = RECORDING_DIR,
        max_record_sec: int = MAX_RECORD_SEC,
        fps: int = RECORD_FPS,
        log_name: str = LOG_NAME,
        component_name: str = COMPONENT_NAME,
    ):
        self._output_stem = _sanitize_output_stem(output_stem)
        self._output_dir = Path(output_dir)
        self._max_record_sec = max(1, int(max_record_sec))
        self._fps = max(1, int(fps))
        self._log_name = log_name
        self._component_name = component_name
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._finished_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_path: Path | None = None
        self._error_message = ""

    @property
    def output_path(self) -> Path | None:
        """녹화 파일 경로를 반환한다."""
        return self._output_path

    @property
    def error_message(self) -> str:
        """녹화 중 발생한 오류 메시지를 반환한다."""
        return self._error_message

    def start(self) -> Path | None:
        """녹화를 시작하고 출력 파일 경로를 반환한다."""
        if not CV2_AVAILABLE or not MSS_AVAILABLE:
            self._error_message = (
                "녹화에 필요한 패키지(opencv-python, mss)가 없습니다."
            )
            print(f"[ERROR] {self._error_message}")
            log_work2_event(
                component=self._component_name,
                message="record_start_failed",
                level="error",
                log_name=self._log_name,
                reason=self._error_message,
            )
            return None

        if self._thread is not None and self._thread.is_alive():
            return self._output_path

        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        self._output_path = self._output_dir / f"{self._output_stem}_{ts}.avi"
        self._error_message = ""
        self._stop_event.clear()
        self._ready_event.clear()
        self._finished_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"recorder-{self._output_stem}",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

        if self._error_message:
            print(f"[ERROR] 녹화 시작 실패: {self._error_message}")
            log_work2_event(
                component=self._component_name,
                message="record_start_failed",
                level="error",
                log_name=self._log_name,
                output_path=str(self._output_path),
                reason=self._error_message,
            )
            return None

        print(f"[INFO] 녹화 시작: {self._output_path}")
        log_work2_event(
            component=self._component_name,
            message="record_started",
            log_name=self._log_name,
            output_path=str(self._output_path),
            fps=self._fps,
            max_record_sec=self._max_record_sec,
        )
        return self._output_path

    def stop(self) -> Path | None:
        """녹화를 중지하고 저장된 파일 경로를 반환한다."""
        if self._thread is None:
            return self._output_path

        self._stop_event.set()
        self._thread.join(timeout=5)
        if self._output_path is not None:
            print(f"[INFO] 녹화 종료: {self._output_path}")
            log_work2_event(
                component=self._component_name,
                message="record_stopped",
                log_name=self._log_name,
                output_path=str(self._output_path),
                error=self._error_message or "",
            )
        return self._output_path

    def wait(self, timeout: float | None = None) -> bool:
        """녹화 스레드가 종료될 때까지 대기한다."""
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _record_loop(self) -> None:
        """실제 캡처 루프."""
        writer = None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                width = int(monitor["width"])
                height = int(monitor["height"])
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(
                    str(self._output_path),
                    fourcc,
                    self._fps,
                    (width, height),
                )
                if not writer.isOpened():
                    self._error_message = "cv2.VideoWriter 초기화 실패"
                    return

                self._ready_event.set()
                frame_interval = 1.0 / self._fps
                started_at = time.time()

                while not self._stop_event.is_set():
                    elapsed_total = time.time() - started_at
                    if elapsed_total > self._max_record_sec:
                        print(
                            f"[WARNING] 최대 녹화 시간 초과 "
                            f"({self._max_record_sec}s)"
                        )
                        break

                    frame_started_at = time.time()
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    writer.write(frame)
                    elapsed = time.time() - frame_started_at
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
        except Exception as exc:
            self._error_message = str(exc)
            print(f"[ERROR] 화면 녹화 중 예외: {exc}")
            log_work2_event(
                component=self._component_name,
                message="record_loop_exception",
                level="error",
                log_name=self._log_name,
                output_path=str(self._output_path or ""),
                error=str(exc),
            )
        finally:
            if writer is not None:
                writer.release()
            self._ready_event.set()
            self._finished_event.set()


def record_screen(
    output_stem: str = DEFAULT_OUTPUT_STEM,
    *,
    output_dir: Path = RECORDING_DIR,
    max_record_sec: int = MAX_RECORD_SEC,
    fps: int = RECORD_FPS,
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> Path | None:
    """현재 전체 화면을 녹화하고 저장 파일 경로를 반환한다."""
    recorder = ScreenRecorder(
        output_stem=output_stem,
        output_dir=output_dir,
        max_record_sec=max_record_sec,
        fps=fps,
        log_name=log_name,
        component_name=component_name,
    )
    output_path = recorder.start()
    if output_path is None:
        return None

    try:
        recorder.wait()
    except KeyboardInterrupt:
        print("\n[INFO] 화면 녹화 중단 요청 (Ctrl+C)")
        output_path = recorder.stop()

    if recorder.error_message:
        return None
    return output_path


def main() -> str:
    """Channel 4 전체화면 녹화를 시작한다."""
    output_path = record_screen()
    if output_path is None:
        print("[ERROR] 화면 녹화 실패")
        return "record_failed"

    print(f"[INFO] 화면 녹화 저장 완료: {output_path}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if main() == "success" else 1)
