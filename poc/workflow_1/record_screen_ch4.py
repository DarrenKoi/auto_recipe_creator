"""Channel 4 CCTV 선택 이후 대상 창만 녹화한다.

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
from poc.workflow_1.util import WINDOW_UTILS_AVAILABLE, env_int, foreground_window

load_dotenv()

LOG_NAME = "record_screen_ch4"
COMPONENT_NAME = LOG_NAME
DEFAULT_OUTPUT_STEM = "ch4_cctv"
DEFAULT_CODEC = os.getenv("CH4_RECORD_CODEC", "mp4v").strip() or "mp4v"
DEFAULT_EXTENSION = os.getenv("CH4_RECORD_EXTENSION", "mp4").strip() or "mp4"
RECORDING_DIR = Path(
    os.getenv(
        "CH4_RECORDING_DIR",
        str(Path(__file__).resolve().parent / "recordings" / "ch4"),
    ),
)
MAX_RECORD_SEC = env_int("CH4_MAX_RECORD_SEC", 600)
RECORD_FPS = env_int("CH4_RECORD_FPS", 5)


def _normalize_extension(extension: str) -> str:
    """파일 확장자를 점 없이 정규화한다."""
    normalized = (extension or "").strip().lower().lstrip(".")
    return normalized or DEFAULT_EXTENSION


def _build_codec_candidates(
    preferred_codec: str,
    preferred_extension: str,
) -> list[tuple[str, str]]:
    """시도할 코덱/확장자 후보 목록을 만든다."""
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(codec: str, extension: str) -> None:
        normalized_codec = (codec or "").strip()
        normalized_extension = _normalize_extension(extension)
        if not normalized_codec:
            return
        item = (normalized_codec, normalized_extension)
        if item in seen:
            return
        seen.add(item)
        candidates.append(item)

    add(preferred_codec, preferred_extension)
    add("mp4v", "mp4")
    add("MJPG", "avi")
    add("XVID", "avi")
    return candidates


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
    """백그라운드 스레드에서 대상 창 또는 전체 화면을 녹화한다."""

    def __init__(
        self,
        output_stem: str = DEFAULT_OUTPUT_STEM,
        *,
        output_dir: Path = RECORDING_DIR,
        max_record_sec: int = MAX_RECORD_SEC,
        fps: int = RECORD_FPS,
        codec: str = DEFAULT_CODEC,
        extension: str = DEFAULT_EXTENSION,
        target_window=None,
        target_window_title: str = "",
        log_name: str = LOG_NAME,
        component_name: str = COMPONENT_NAME,
    ):
        self._output_stem = _sanitize_output_stem(output_stem)
        self._output_dir = Path(output_dir)
        self._max_record_sec = max(1, int(max_record_sec))
        self._fps = max(1, int(fps))
        self._codec = (codec or "").strip() or DEFAULT_CODEC
        self._extension = _normalize_extension(extension)
        self._target_window = target_window
        self._target_window_title = (target_window_title or "").strip()
        self._log_name = log_name
        self._component_name = component_name
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._finished_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_path: Path | None = None
        self._error_message = ""
        self._resolved_codec = ""

    @property
    def output_path(self) -> Path | None:
        """녹화 파일 경로를 반환한다."""
        return self._output_path

    @property
    def error_message(self) -> str:
        """녹화 중 발생한 오류 메시지를 반환한다."""
        return self._error_message

    @property
    def resolved_codec(self) -> str:
        """실제로 사용된 코덱 문자열을 반환한다."""
        return self._resolved_codec

    def _resolve_capture_region(self) -> dict[str, int] | None:
        """대상 창의 현재 화면 좌표를 mss capture region 으로 변환한다."""
        if self._target_window is None:
            return None

        try:
            rect = self._target_window.rectangle()
        except Exception as exc:
            self._error_message = f"대상 창 rectangle 조회 실패: {exc}"
            return None

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 1 or height <= 1:
            self._error_message = (
                f"대상 창 크기가 비정상적입니다: {width}x{height}"
            )
            return None

        return {
            "left": int(rect.left),
            "top": int(rect.top),
            "width": width,
            "height": height,
        }

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
        self._output_path = (
            self._output_dir
            / f"{self._output_stem}_{ts}.{self._extension}"
        )
        self._error_message = ""
        self._resolved_codec = ""
        self._stop_event.clear()
        self._ready_event.clear()
        self._finished_event.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"recorder-{self._output_stem}",
            daemon=True,
        )

        if self._target_window is not None and WINDOW_UTILS_AVAILABLE:
            foreground_window(
                self._target_window,
                debug_label=self._target_window_title or self._output_stem,
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
            codec=self._resolved_codec or self._codec,
            capture_target=self._target_window_title or "full_screen",
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
                codec=self._resolved_codec or self._codec,
                capture_target=self._target_window_title or "full_screen",
                error=self._error_message or "",
            )
        return self._output_path

    def wait(self, timeout: float | None = None) -> bool:
        """녹화 스레드가 종료될 때까지 대기한다."""
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _open_video_writer(self, width: int, height: int):
        """지원 가능한 코덱을 순서대로 시도하여 VideoWriter 를 연다."""
        candidates = _build_codec_candidates(self._codec, self._extension)
        last_error = "cv2.VideoWriter 초기화 실패"

        for codec, extension in candidates:
            candidate_path = (
                self._output_dir
                / f"{self._output_stem}_{datetime.now().strftime('%y%m%d_%H%M%S')}.{extension}"
            )
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(candidate_path),
                fourcc,
                self._fps,
                (width, height),
            )
            if writer.isOpened():
                self._output_path = candidate_path
                self._resolved_codec = codec
                if codec != self._codec or extension != self._extension:
                    print(
                        "[WARNING] 요청한 코덱 사용 불가. "
                        f"fallback codec={codec}, ext={extension}"
                    )
                return writer

            writer.release()
            last_error = (
                f"cv2.VideoWriter 초기화 실패 "
                f"(codec={codec}, ext={extension})"
            )

        self._error_message = last_error
        return None

    def _record_loop(self) -> None:
        """실제 캡처 루프."""
        writer = None
        try:
            with mss.mss() as sct:
                capture_region = self._resolve_capture_region()
                if self._target_window is not None and capture_region is None:
                    return

                if capture_region is None:
                    monitor = sct.monitors[0]
                    width = int(monitor["width"])
                    height = int(monitor["height"])
                    monitor_region = monitor
                else:
                    width = int(capture_region["width"])
                    height = int(capture_region["height"])
                    monitor_region = capture_region

                writer = self._open_video_writer(width, height)
                if writer is None:
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
                    if self._target_window is not None:
                        updated_region = self._resolve_capture_region()
                        if updated_region is None:
                            break
                        monitor_region = updated_region

                    img = sct.grab(monitor_region)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
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
    codec: str = DEFAULT_CODEC,
    extension: str = DEFAULT_EXTENSION,
    target_window=None,
    target_window_title: str = "",
    log_name: str = LOG_NAME,
    component_name: str = COMPONENT_NAME,
) -> Path | None:
    """대상 창 또는 현재 전체 화면을 녹화하고 저장 파일 경로를 반환한다."""
    recorder = ScreenRecorder(
        output_stem=output_stem,
        output_dir=output_dir,
        max_record_sec=max_record_sec,
        fps=fps,
        codec=codec,
        extension=extension,
        target_window=target_window,
        target_window_title=target_window_title,
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
    """DVR player 창 또는 전체 화면 녹화를 시작한다."""
    target_window = None
    target_window_title = ""

    if os.name == "nt":
        try:
            from poc.workflow_1.workflow_select_ch4_cctv import _find_player_window

            target_window, target_window_title, _backend, _process_name = _find_player_window()
        except Exception as exc:
            print(f"[ERROR] DVR player 창 조회 실패: {exc}")
            return "player_window_not_found"

        if target_window is None:
            print("[ERROR] DVR player 창을 찾지 못했습니다.")
            return "player_window_not_found"

    output_path = record_screen(
        target_window=target_window,
        target_window_title=target_window_title,
    )
    if output_path is None:
        print("[ERROR] 화면 녹화 실패")
        return "record_failed"

    print(f"[INFO] 화면 녹화 저장 완료: {output_path}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if main() == "success" else 1)
