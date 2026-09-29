"""시연 영상용 화면 녹화기 - 주 모니터를 고정 fps mp4 로 담고 커서/클릭/휠을 기록한다.

`RecordingSession` 은 알람 분석용(화면이 바뀔 때만 JPEG, ~5fps)이라 시연 영상에서는 끊겨
보인다. 여기서는 고정 fps 로 mss 캡처를 ffmpeg(imageio-ffmpeg 번들)에 흘린다. 캡처가 fps 를
못 따라가면 직전 프레임을 복제해 **영상 시각 = 경과 시각** 을 지킨다 - 자막과 클릭 시각이
영상과 어긋나지 않는 것이 부드러움보다 우선이다. 실제 캡처율은 끝에 찍힌다.

mss 캡처에는 마우스 커서가 없다. 그래서 프레임마다 커서 위치를, 전역 훅으로 클릭/휠을
`events.json` 에 남기고 `polish_demo_video.py` 가 커서와 클릭 강조를 직접 그린다.
좌표는 pynput 과 mss 가 같은 프로세스 좌표계(mss 가 DPI aware 를 켠다)이며, 모니터 rect 를
같이 남겨 영상 좌표로 환산한다.

**사람 입력 집계**: 전역 훅은 자동화가 보낸 입력(SendInput = injected)과 사람이 직접 한
입력을 구분한다(pynput>=1.8). 녹화 중 사람 입력 횟수를 `human_input` 에 남기고 끝에 찍는다 -
"녹화 구간에 사람 개입 0회" 의 근거다. 키는 **횟수만** 센다(무엇을 눌렀는지는 남기지 않는다).
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

from poc.workflow_3 import ALIGN_IMAGES_DIR

DEMO_ROOT = ALIGN_IMAGES_DIR / "_demo"   # 시연 녹화 조각이 쌓이는 곳
VIDEO_NAME = "raw.mp4"
EVENTS_NAME = "events.json"
STAGES_NAME = "stages.json"
SUBTITLES_NAME = "subtitles.json"


class ScreenVideoRecorder:
    """start() -> (조작) -> stop(). `elapsed()` 는 영상 시각(초)이다."""

    def __init__(self, out_dir: Path, *, fps: int = 30, monitor_index: int | None = None,
                 crf: int = 12):
        self.out_dir = Path(out_dir)
        self.fps = int(fps)
        self.monitor_index = monitor_index  # None = 주 모니터
        self.crf = int(crf)  # 원본은 거의 무손실로 - 화질은 polish 단계에서 정한다.
        self.monitor: dict = {}
        self.cursor: list = []   # 영상 프레임마다 [x, y] (화면 좌표)
        self.events: list = []   # {"t", "kind": click|scroll, "x", "y", ["dy"], "injected"}
        # 사람이 직접 한 입력 횟수. unknown = injected 판별 불가(pynput<1.8) 입력 수.
        self.human_input = {"move": 0, "click": 0, "scroll": 0, "key": 0, "unknown": 0}
        self.captured = 0
        self.written = 0
        self._t0 = 0.0
        self._stop_at: float | None = None
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list = []

    def elapsed(self) -> float:
        # 벽시계(time.time)는 Windows 시각 동기화로 뒤/앞으로 뛸 수 있다 - 시간축은 단조 시계로.
        return time.perf_counter() - self._t0

    # ---- 입력 훅 ----

    def _count(self, kind: str, injected) -> None:
        if injected is None:
            self.human_input["unknown"] += 1
        elif not injected:
            self.human_input[kind] += 1

    def _on_move(self, x, y, injected=None):
        self._count("move", injected)

    def _on_click(self, x, y, button, pressed, injected=None):
        if pressed:
            self._count("click", injected)
            self.events.append({"t": round(self.elapsed(), 3), "kind": "click",
                                "x": x, "y": y, "injected": injected})

    def _on_scroll(self, x, y, dx, dy, injected=None):
        self._count("scroll", injected)
        self.events.append({"t": round(self.elapsed(), 3), "kind": "scroll",
                            "x": x, "y": y, "dy": dy, "injected": injected})

    def _on_key(self, key, injected=None):
        self._count("key", injected)

    # ---- 수명 주기 ----

    def start(self) -> "ScreenVideoRecorder":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(15):
            self._shutdown()
            raise RuntimeError("화면 녹화 시작 실패: 15초 안에 캡처/인코더가 준비되지 않음")
        if self._error is not None:
            raise RuntimeError(f"화면 녹화 시작 실패: {self._error}") from self._error
        try:
            from pynput import keyboard, mouse

            for listener in (
                mouse.Listener(on_move=self._on_move, on_click=self._on_click,
                               on_scroll=self._on_scroll),
                keyboard.Listener(on_press=self._on_key),
            ):
                listener.start()
                self._listeners.append(listener)  # 시작된 것만 - 종료 때 join 대상
        except Exception:
            self._shutdown()
            raise
        print(f"[INFO] 화면 녹화 시작: {self.out_dir / VIDEO_NAME} ({self.fps}fps, "
              f"모니터 {self.monitor})")
        return self

    @staticmethod
    def pick_monitor(monitors: list, index) -> dict:
        """index 가 None 이면 주 모니터(가상 화면 원점 (0,0) 을 가진 모니터)를 고른다.

        mss 의 monitors[1] 은 '처음 열거된' 모니터일 뿐 주 모니터라는 보장이 없다.
        """
        if index is not None and 0 < index < len(monitors):
            return monitors[index]
        if index is not None:
            print(f"[WARNING] 모니터 index={index} 없음 - 주 모니터로 녹화")
        return next((m for m in monitors[1:] if m["left"] == 0 and m["top"] == 0), monitors[1])

    def _run(self) -> None:
        try:
            import imageio_ffmpeg
            import mss
            from pynput.mouse import Controller

            pointer = Controller()
            with mss.mss() as sct:
                if len(sct.monitors) < 2:
                    raise RuntimeError("캡처할 모니터가 없습니다(화면 캡처 권한 확인)")
                mon = self.pick_monitor(sct.monitors, self.monitor_index)
                self.monitor = {k: int(mon[k]) for k in ("left", "top", "width", "height")}
                w, h = mon["width"] - mon["width"] % 2, mon["height"] - mon["height"] % 2
                writer = imageio_ffmpeg.write_frames(
                    str(self.out_dir / VIDEO_NAME), (w, h), fps=self.fps, codec="libx264",
                    macro_block_size=1, quality=None, ffmpeg_log_level="error",
                    output_params=["-preset", "ultrafast", "-crf", str(self.crf)],
                )
                writer.send(None)
                self._t0 = time.perf_counter()
                self._ready.set()
                prev = None
                try:
                    while not self._stop.is_set():
                        shot = sct.grab(mon)
                        pos = [int(v) for v in pointer.position]
                        # 이 화면의 시각 = 손에 넣은 시각. 그 전 칸은 **직전 화면**으로 채우고
                        # (새 화면을 과거로 당기지 않는다) 이 칸부터 새 화면을 놓는다. 인코딩에
                        # 걸린 시간은 칸을 늘리지 않는다 - 다음 캡처가 직전 화면으로 메운다.
                        slot = int(self.elapsed() * self.fps)
                        bgra = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)
                        frame = np.ascontiguousarray(bgra[:h, :w, 2::-1])
                        self.captured += 1
                        if prev is not None:
                            self._write_until(writer, prev, slot)
                        self._write_until(writer, (frame, pos), slot + 1)
                        prev = (frame, pos)
                        time.sleep(max(0.0, self.written / self.fps - self.elapsed()))
                    if prev is not None:  # 마지막 캡처 ~ 정지 요청 사이도 직전 화면으로.
                        self._write_until(writer, prev, self._stop_slot())
                finally:
                    writer.close()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def _stop_slot(self) -> int:
        return int(self._stop_at * self.fps) + 1 if self._stop_at is not None else 1 << 62

    def _write_until(self, writer, item, slot: int) -> None:
        """영상 프레임 수를 slot 까지 채운다. 정지 요청 시각 뒤로는 넘기지 않는다."""
        frame, pos = item
        slot = min(slot, self._stop_slot())
        while self.written < slot:
            writer.send(frame)
            self.cursor.append(pos)
            self.written += 1

    def _shutdown(self) -> None:
        """녹화 스레드와 입력 훅을 멈춘다. 예외는 삼키지 않고 _error 로 남긴다(사이드카 보존)."""
        if self._stop_at is None and self._t0:
            self._stop_at = self.elapsed()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                self._error = self._error or RuntimeError(
                    "녹화 스레드가 60초 안에 끝나지 않음(mp4 미완성 가능)")
        for listener in self._listeners:
            try:
                listener.stop()
                listener.join(timeout=5)
                if listener.is_alive():
                    raise RuntimeError("입력 훅이 5초 안에 끝나지 않음")
            except Exception as exc:  # pynput join 은 콜백 예외를 다시 던진다.
                self._error = self._error or exc
        self._listeners = []

    def stop(self) -> dict:
        """녹화를 멈추고 events.json 을 쓴 뒤 그 내용을 돌려준다.

        녹화 중 오류(캡처 실패/디스크 부족/인코더 종료/훅 오류)는 삼키지 않는다 - events.json
        은 남기되 `error` 에 담고 [ERROR] 로 찍는다. 호출자는 `error` 가 있으면 실패로 끝낸다.
        """
        self._shutdown()
        duration = self.written / self.fps if self.fps else 0.0
        info = {
            "fps": self.fps,
            "monitor": self.monitor,
            "duration": round(duration, 3),
            "capture_fps": round(self.captured / duration, 1) if duration else 0.0,
            "error": f"{type(self._error).__name__}: {self._error}" if self._error else "",
            "human_input": dict(self.human_input),
            "events": list(self.events),
            "cursor": list(self.cursor),
        }
        (self.out_dir / EVENTS_NAME).write_text(json.dumps(info), encoding="utf-8")
        human = info["human_input"]
        print(f"[INFO] 화면 녹화 종료: {duration:.1f}s, 실제 캡처 {info['capture_fps']}fps "
              f"(목표 {self.fps}), 클릭 {sum(e['kind'] == 'click' for e in self.events)}회")
        print(f"[INFO] 녹화 중 사람 입력: 마우스 이동 {human['move']}, 클릭 {human['click']}, "
              f"휠 {human['scroll']}, 키 {human['key']}"
              + (f" / 판별 불가 {human['unknown']} (pynput 1.8 이상 필요)" if human["unknown"] else ""))
        if info["error"]:
            print(f"[ERROR] 녹화 중 오류 - 영상이 중간에 끊겼을 수 있습니다: {info['error']}")
        elif duration and info["capture_fps"] < self.fps * 0.6:
            print("[WARNING] 캡처가 목표 fps 를 크게 못 따라갔습니다 - 영상이 끊겨 보일 수 있습니다 "
                  "(해상도가 높으면 FPS 를 낮추세요).")
        return info
