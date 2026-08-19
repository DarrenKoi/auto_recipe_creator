"""RCS tool 창 상시 녹화 — 변화 감지 기반 적응 캡처(엔지니어 수동 조작 추적).

모든 align fail 에서 녹화한다(성공/실패 무관). 자동 보정이 실패해 엔지니어가
직접 장비를 조작하는 동안에도 같은 세션이 계속 캡처하므로, 이 프레임들이 다음
개선(모방 학습/절차 분석)의 원천 데이터가 된다.

RCS 는 원격 접속 프로그램이라 장비 측 마우스 커서/움직임이 화면 콘텐츠로서
프레임에 그대로 찍힌다. 따라서 커서 궤적을 따라가는 관건은 캡처 *간격* 인데,
고정 간격은 조작 구간에선 너무 성기고 idle 구간에선 낭비라, DVR CH4 캡처와
``filter_frames_by_change`` 의 선례를 따라 **빠른 샘플링 + 변화 감지 저장** 으로
동작한다:

  * ``poll_sec``(기본 0.05s) 간격으로 캡처해 직전 저장 프레임과 비교한다.
    지표는 1/4 다운샘플 grayscale 에서 **변화 폭이 노이즈 바닥(픽셀 delta>10)을
    넘는 픽셀의 개수** — 평균 차이는 작은 커서 이동에 둔감해서 쓰지 않는다.
  * 변화 픽셀이 ``change_min_px``(기본 2) 이상이면 저장 — 조작 중에는 ~20fps 로
    커서 움직임/메뉴/다이얼로그 전이를 촘촘히 따라간다.
  * 변화가 없으면 ``heartbeat_sec``(기본 5s)마다 1장만 저장 — idle 구간의
    디스크 낭비를 막으면서 "이 구간엔 아무 일도 없었다"는 증거를 남긴다.

계약:
  * 저장 경로(out_dir)는 호출부가 정한다 — 보통
    align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/
    (RECIPE_ID 없으면 align_images/<eqp>/_unregistered/<tag>/recording/)
  * 파일명 <tag>_rcs_<seq:04d>_<elapsed_ms>ms.jpg (elapsed 로 시간축 복원)
  * 자동 중지: 연속 캡처 실패 ``FAILURE_WINDOW_SEC`` 지속(창 닫힘 간주) 또는 max_sec
  * 종료 시 recording_manifest.json (샘플/저장 수, 파라미터, 중지사유) 기록
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

from poc.workflow_3.config import (
    DEFAULT_RECORDING_CHANGE_MIN_PX,
    DEFAULT_RECORDING_HEARTBEAT_SEC,
    DEFAULT_RECORDING_POLL_SEC,
)
from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.util import capture_window

LOG_COMPONENT = "align_fail_recording"

# 캡처 실패가 이 시간 동안 연속되면 창이 닫힌 것으로 보고 세션을 끝낸다.
FAILURE_WINDOW_SEC = 5.0
# 변화 비교용 다운샘플 간격 (양 축 1/4 → 픽셀 1/16, 커서 이동도 충분히 감지됨).
_DIFF_DOWNSAMPLE = 4
# 픽셀 변화로 인정하는 최소 delta — RCS 스트림/JPEG 압축 노이즈 바닥값 위.
_PIXEL_DELTA_MIN = 10.0


def to_diff_gray(image) -> np.ndarray:
    """캡처 이미지를 변화 비교용 저해상 grayscale float 배열로 변환한다.

    **반드시 `[::4, ::4]` 격자 그대로 뽑는다.** 전체 해상도 convert("L") 를 피하려고
    `resize((w//4, h//4), NEAREST)` 로 바꾸고 싶어지지만, 그건 같은 연산이 아니다
    (2026-08-15 리뷰에서 확인):

      * NEAREST resize 는 출력 셀의 **중심**을 되짚어 입력 픽셀 4i+2 를 고른다.
        8x8 램프에서 이 격자는 [0,4,32,36] 이 아니라 [18,22,50,54] 를 뽑는다 -
        면적은 같아도 표본이 완전히 다른 픽셀이다.
      * 변의 길이가 4의 배수가 아니면 모양까지 달라진다(내림 vs 올림). 1367x769
        crop 은 (193,342) 가 아니라 (192,341) 이 된다. engineer-done CV gate 는
        1920x1080 프레임이 아니라 **임의 크기 분자 ROI crop** 에 걸리므로 바로
        이 경우에 해당한다.

    표본이 바뀌면 `frame_changed` 의 changed_px 가 바뀌고, 그건 곧
    `change_min_px`/`engineer_done_change_min_px` 민감도를 조용히 재튜닝하는 것이다.
    prev/current 가 같은 변환을 타므로 깨지지는 않지만, 오피스 검증이 없는 CV 게이트의
    임계를 성능 최적화의 부수효과로 옮기면 안 된다. 여기서 아끼는 건 프레임당 0.3ms 로,
    캡처 경로의 PNG 왕복(234ms)을 걷어낸 지금은 폴링 예산(50ms) 대비 무의미하다.
    """
    array = np.asarray(image.convert("L") if hasattr(image, "convert") else image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array[::_DIFF_DOWNSAMPLE, ::_DIFF_DOWNSAMPLE].astype(np.float32)


def frame_changed(
    prev: np.ndarray | None,
    current: np.ndarray,
    min_changed_px: int,
    *,
    pixel_delta_min: float = _PIXEL_DELTA_MIN,
) -> bool:
    """직전 저장 프레임 대비 '확실히 변한' 다운샘플 픽셀 수가 임계 이상인지.

    평균 절대차는 작은 커서(화면의 수백분의 일)가 움직여도 임계를 못 넘는다.
    개수 기반은 커서 한 칸 이동(이전 위치 복원 + 새 위치 등장 = 수 픽셀, delta 큼)
    도 잡고, 압축 노이즈(넓지만 delta 작음)는 ``pixel_delta_min`` 으로 걸러진다.

    ``pixel_delta_min`` 이 인자인 이유: 이 함수는 녹화 저장 게이트와 engineer-done
    카운터 CV 게이트 양쪽이 쓰는데, 두 판정의 목적이 달라 민감도도 갈릴 수 있다.
    모듈 상수로 두면 한쪽을 튜닝할 때 다른 쪽이 조용히 같이 움직인다.
    """
    if prev is None:
        return True
    if prev.shape != current.shape:
        return True  # 창 리사이즈 등 — 변화로 간주.
    changed_px = int((np.abs(current - prev) > pixel_delta_min).sum())
    return changed_px >= min_changed_px


class RecordingSession:
    """tool 창을 변화 감지로 적응 캡처하는 데몬 스레드 세션 (context manager 지원)."""

    def __init__(
        self,
        tool_window,
        out_dir: Path,
        *,
        tag: str,
        poll_sec: float = DEFAULT_RECORDING_POLL_SEC,
        heartbeat_sec: float = DEFAULT_RECORDING_HEARTBEAT_SEC,
        change_min_px: int = DEFAULT_RECORDING_CHANGE_MIN_PX,
        max_sec: float = 900.0,
        max_frames: int = 0,
        max_disk_mb: float = 0.0,
        jpeg_quality: int = 95,
        capture_fn=None,
        capture_source: str = "tool_window",
    ):
        self.tool_window = tool_window
        self.out_dir = Path(out_dir)
        self.tag = tag
        self.poll_sec = max(0.01, float(poll_sec))
        self.heartbeat_sec = max(self.poll_sec, float(heartbeat_sec))
        self.change_min_px = max(1, int(change_min_px))
        self.max_sec = float(max_sec)
        # 프레임/디스크 백스톱. 0 = 무제한 (max_sec 규약과 같다). 예산은 샘플링 주기를
        # 정하는 곳과 같은 깊이에 있어야 한다 - poll_sec 을 6배로 올리면 같은 시간에
        # 6배의 프레임이 쌓이므로, 예산이 호출부에만 있으면 그 호출부만 보호받는다.
        self.max_frames = max(0, int(max_frames))
        self.max_disk_mb = max(0.0, float(max_disk_mb))
        self._disk_bytes = 0
        self.jpeg_quality = int(jpeg_quality)
        # 테스트 주입점 — 기본은 실제 창 캡처.
        self._capture_fn = capture_fn or (lambda: capture_window(self.tool_window))
        # 무엇을 찍은 프레임인지(창 rect / 화면 전체). manifest 에만 남는 라벨이지만
        # 소비자에겐 중요하다 - recording_filter 의 live SEM box 게이트는 'tool 창
        # rect' 를 전제하므로, 화면 전체 프레임을 같은 파이프라인에 넣으면 안 된다.
        self.capture_source = capture_source
        self.frames: list[Path] = []
        self.sampled_count = 0
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
            f"[INFO] 녹화 시작: dir={self.out_dir}, poll={self.poll_sec}s, "
            f"heartbeat={self.heartbeat_sec}s, change_min_px={self.change_min_px}, "
            f"max={self.max_sec}s"
        )
        return self

    def stop(self, reason: str = "stopped") -> list[Path]:
        """녹화를 멈추고 manifest 를 기록한 뒤 저장된 프레임 목록을 반환한다."""
        if not self.stop_reason:
            self.stop_reason = reason
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_sec + 5.0)
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

    def disk_mb(self) -> float:
        """이 세션이 지금까지 쓴 프레임 용량 합계(MB) - 증분 누적값."""
        return self._disk_bytes / (1024.0 * 1024.0)

    def _budget_stop_reason(self) -> str:
        """프레임/디스크 예산 초과 사유. 여유가 있으면 빈 문자열.

        사유는 하나만 돌려준다 - manifest 만 보고 원인을 구분할 수 있어야 한다.
        """
        if self.max_frames > 0 and len(self.frames) >= self.max_frames:
            return "frame_budget"
        if self.max_disk_mb > 0 and self.disk_mb() >= self.max_disk_mb:
            return "disk_budget"
        return ""

    def _run(self) -> None:
        seq = 0
        started = self._started_at or time.time()
        first_failure_at: float | None = None
        prev_gray: np.ndarray | None = None
        last_saved_at = 0.0

        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - started
            if self.max_sec > 0 and elapsed >= self.max_sec:
                self.stop_reason = "max_sec"
                break
            try:
                image = self._capture_fn()
                first_failure_at = None
                self.sampled_count += 1

                gray = to_diff_gray(image)
                changed = frame_changed(prev_gray, gray, self.change_min_px)
                heartbeat_due = (now - last_saved_at) >= self.heartbeat_sec
                if changed or heartbeat_due:
                    elapsed_ms = int(elapsed * 1000)
                    out_path = self.out_dir / f"{self.tag}_rcs_{seq:04d}_{elapsed_ms:08d}ms.jpg"
                    save_debug_jpeg(image, out_path, quality=self.jpeg_quality)
                    self.frames.append(out_path)
                    seq += 1
                    prev_gray = gray
                    last_saved_at = now
                    # 방금 쓴 파일 크기만 누적한다 - 폴더 전체를 다시 훑으면 저장
                    # 프레임 수에 비례하는 비용이 매번 들어 O(n^2) 이 된다.
                    try:
                        self._disk_bytes += out_path.stat().st_size
                    except OSError:
                        pass
                    budget = self._budget_stop_reason()
                    if budget:
                        self.stop_reason = budget
                        break
            except Exception as exc:
                if first_failure_at is None:
                    first_failure_at = now
                    print(f"[WARNING] 녹화 캡처 실패(창 닫힘?): {exc}")
                elif now - first_failure_at >= FAILURE_WINDOW_SEC:
                    # 창이 닫힌 것으로 간주 — 엔지니어/close_tool 이 창을 닫은 정상 종료.
                    self.stop_reason = "window_gone"
                    break
            self._stop_event.wait(self.poll_sec)

        if not self.stop_reason:
            self.stop_reason = "stopped"

    def _write_manifest(self) -> None:
        """recording_manifest.json 을 기록한다 (실패는 삼켜 사이클을 죽이지 않음)."""
        manifest = {
            "tag": self.tag,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self._started_at or time.time())
            ),
            # 초 단위 문자열과 달리 세션 간 접합에 쓸 수 있는 절대 기준점.
            # prelude 녹화와 본 녹화는 t0 가 다르므로, 두 프레임 목록을 하나의
            # 시간축에 올리려면 이 값의 차이가 필요하다(문자열은 1초 해상도라
            # 20fps 구간에서 최대 20프레임이 어긋난다).
            "started_epoch": round(self._started_at or time.time(), 3),
            "capture_source": self.capture_source,
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "frame_count": len(self.frames),
            "sampled_count": self.sampled_count,
            "poll_sec": self.poll_sec,
            "heartbeat_sec": self.heartbeat_sec,
            "change_min_px": self.change_min_px,
            "jpeg_quality": self.jpeg_quality,
            "max_frames": self.max_frames,
            "max_disk_mb": self.max_disk_mb,
            "disk_mb": round(self.disk_mb(), 2),
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
            f"[INFO] 녹화 종료: saved={len(self.frames)}/{self.sampled_count} sampled, "
            f"reason={self.stop_reason}, dir={self.out_dir}"
        )


__all__ = [
    "RecordingSession",
    "FAILURE_WINDOW_SEC",
    "frame_changed",
    "to_diff_gray",
]
