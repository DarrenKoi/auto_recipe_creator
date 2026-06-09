"""RCS tool 창 상시 녹화 — 알람별 사이클 동안 주기 캡처(엔지니어 수동 조작 포함).

모든 align fail 에서 녹화한다(성공/실패 무관). 자동 보정이 실패해 엔지니어가
직접 장비를 조작하는 동안에도 같은 세션이 계속 캡처하므로, 이 프레임들이 다음
개선(모방 학습/절차 분석)의 원천 데이터가 된다.

계약:
  * 저장 경로(out_dir)는 호출부가 정한다 — 보통
    align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/
    (RECIPE_ID 없으면 align_images/<eqp>/_unregistered/<tag>/recording/)
  * interval_sec 간격 JPEG, 파일명 <tag>_rcs_<seq:04d>_<elapsed_ms>ms.jpg
  * 자동 중지: 연속 캡처 실패 5회(창 닫힘으로 간주) 또는 max_sec 초과
  * 종료 시 recording_manifest.json (시작/종료/프레임수/중지사유) 기록
"""

import json
import threading
import time
from pathlib import Path

from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.util import capture_window

LOG_COMPONENT = "align_fail_recording"

# 연속 캡처 실패 허용 횟수 — 초과 시 창이 닫힌 것으로 보고 세션을 끝낸다.
MAX_CONSECUTIVE_FAILURES = 5


class RecordingSession:
    """tool 창을 주기 캡처하는 데몬 스레드 세션 (context manager 지원)."""

    def __init__(
        self,
        tool_window,
        out_dir: Path,
        *,
        tag: str,
        interval_sec: float = 2.0,
        max_sec: float = 900.0,
    ):
        self.tool_window = tool_window
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.interval_sec = max(0.2, float(interval_sec))
        self.max_sec = float(max_sec)
        self.frames: list[Path] = []
        self.stop_reason: str = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None

    # ---- 수명 주기 ----

    def start(self) -> "RecordingSession":
        """녹화 스레드를 시작한다 (이미 시작했으면 무시)."""
        if self._thread is not None:
            return self
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"[INFO] 녹화 시작: dir={self.out_dir}, interval={self.interval_sec}s, "
            f"max={self.max_sec}s"
        )
        return self

    def stop(self, reason: str = "stopped") -> list[Path]:
        """녹화를 멈추고 manifest 를 기록한 뒤 프레임 목록을 반환한다."""
        if not self.stop_reason:
            self.stop_reason = reason
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5.0)
        self._write_manifest()
        return self.frames

    def is_alive(self) -> bool:
        """녹화 스레드가 아직 도는 중인지 (창 닫힘 감지에 활용)."""
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "RecordingSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop("context_exit")

    # ---- 내부 ----

    def _run(self) -> None:
        consecutive_failures = 0
        seq = 0
        started = self._started_at or time.time()
        while not self._stop_event.is_set():
            elapsed = time.time() - started
            if self.max_sec > 0 and elapsed >= self.max_sec:
                self.stop_reason = "max_sec"
                break
            try:
                image = capture_window(self.tool_window)
                elapsed_ms = int(elapsed * 1000)
                out_path = self.out_dir / f"{self.tag}_rcs_{seq:04d}_{elapsed_ms:08d}ms.jpg"
                save_debug_jpeg(image, out_path)
                self.frames.append(out_path)
                seq += 1
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print(f"[WARNING] 녹화 캡처 실패(1회차, 창 닫힘?): {exc}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    # 창이 닫힌 것으로 간주 — 엔지니어/close_tool 이 창을 닫은 정상 종료.
                    self.stop_reason = "window_gone"
                    break
            self._stop_event.wait(self.interval_sec)

        if not self.stop_reason:
            self.stop_reason = "stopped"

    def _write_manifest(self) -> None:
        """recording_manifest.json 을 기록한다 (실패는 삼켜 사이클을 죽이지 않음)."""
        manifest = {
            "tag": self.tag,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self._started_at or time.time())
            ),
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "frame_count": len(self.frames),
            "interval_sec": self.interval_sec,
            "stop_reason": self.stop_reason,
        }
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "recording_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[WARNING] recording manifest 기록 실패: {exc}")
        print(
            f"[INFO] 녹화 종료: frames={len(self.frames)}, reason={self.stop_reason}, "
            f"dir={self.out_dir}"
        )
        log_work2_event(
            component=LOG_COMPONENT,
            message="recording_finished",
            frame_count=len(self.frames),
            stop_reason=self.stop_reason,
            out_dir=str(self.out_dir),
        )


__all__ = ["RecordingSession", "MAX_CONSECUTIVE_FAILURES"]
