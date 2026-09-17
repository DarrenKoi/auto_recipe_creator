"""이벤트 폴더 - 알람 1건의 로그/이미지/녹화를 한 폴더에 모은다.

레이아웃 (`EVENTS_DIR` 은 `poc.workflow_3` 에서 정한다):

    <EVENTS_DIR>/<eqp_id>-<tag>/          알람 1건 (Recovery Episode 루트)
        recovery_episode.json
        attempt_<n>/                      사이클 1회 = take. Episode 수집 off 면 루트가 곧 take.
            console.log                   이 사이클 동안의 stdout/stderr 전사 (줄마다 시각)
            work2.log, vlm_calls.log      감사 로그 사본 (상세 레벨과 무관하게 전부)
            runs/<run_id>_<name>/         WorkflowRunner 저널 (복구 로그인 run 포함)
            debug_images/<component>/     VLM/OCR crop, overlay, 보정 debug
            recording/                    tool 창 녹화 (+ prelude/)

take 는 **프로세스 전역 하나**다. 모니터가 단일 프로세스에서 사이클을 직렬로 돌리고,
녹화/수집 스레드의 출력도 같은 take 로 가야 하므로 thread-local 이 아니다. 사이클을
동시에 돌리게 되면 이 전제가 깨진다.

로깅은 사이클을 죽이지 않는다: 폴더를 못 만들면 경고 후 전역 폴더(debug_images/,
logs/workflow_runs/)로 떨어진다.
"""

import json
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

from poc.workflow_3 import DEBUG_IMAGE_DIR, LOG_DIR

CONSOLE_LOG_NAME = "console.log"
# take 의 신원(eqp/recipe/tag)과 사이클 결과. 폴더 이름에 recipe 가 없어서 이 파일이
# 유일한 recipe 출처다(make_demo_video_combined 의 묶음 기준).
EVENT_META_NAME = "event.json"

# 보관 상한을 넘은 take 에서 지우는 폴더. 텍스트(console.log/저널/Episode·Guard JSON)는
# 작고 Episode 파일은 "절대 지우지 않는다"는 규약(recovery_episode)이 있어 남긴다.
HEAVY_DIR_NAMES = ("recording", "debug_images")

_active: Path | None = None


def active_take_dir() -> Path | None:
    """지금 사이클이 쓰는 take 폴더. 사이클 밖이면 None."""
    return _active


def debug_root(fallback=None) -> Path:
    """debug 산출물 루트 - 사이클 안이면 `<take>/debug_images`, 밖이면 전역 debug_images.

    `fallback` 은 사이클 밖일 때만 쓰인다 - 모듈이 자기 루트 상수를 테스트에서 바꿔
    끼우는 경로(`monkeypatch.setattr(module, "DEBUG_IMAGE_DIR", tmp_path)`)를 살려 둔다.
    """
    if _active is not None:
        return _active / "debug_images"
    return Path(fallback) if fallback is not None else DEBUG_IMAGE_DIR


def runs_root() -> Path:
    """runner 저널 루트 - 사이클 안이면 `<take>/runs`, 밖이면 `logs/workflow_runs`."""
    return _active / "runs" if _active is not None else LOG_DIR / "workflow_runs"


class _Tee:
    """콘솔 스트림은 그대로 쓰고, 같은 내용을 파일에 줄마다 시각을 붙여 복사한다.

    파일을 **먼저** 쓴다 - 오피스 콘솔(cp949)이 인코딩 못 하는 글자로 print 가 던져도
    파일에는 남아야 한다. 시각 형식은 `demo_log_panel` 이 읽는 `%Y-%m-%d %H:%M:%S`.
    """

    def __init__(self, stream, log_file, lock, state):
        self._stream = stream
        self._log_file = log_file
        self._lock = lock
        self._state = state  # stdout/stderr 가 한 파일을 나눠 쓰므로 줄 시작 여부도 공유한다.

    def write(self, text):
        with self._lock:
            stamped = []
            for piece in str(text).splitlines(keepends=True):
                if self._state["line_start"]:
                    stamped.append(time.strftime("%Y-%m-%d %H:%M:%S "))
                stamped.append(piece)
                self._state["line_start"] = piece.endswith("\n")
            try:
                self._log_file.write("".join(stamped))
            except (OSError, ValueError):
                pass  # 닫힌 뒤 늦게 도착한 스레드 출력 - 콘솔에는 그대로 나간다.
        return self._stream.write(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def write_event_meta(take_dir, result) -> None:
    """take 의 `event.json` 을 쓴다(CycleResult 같은 dataclass 또는 dict). 실패는 삼킨다."""
    payload = asdict(result) if is_dataclass(result) else dict(result)
    try:
        Path(take_dir).mkdir(parents=True, exist_ok=True)
        (Path(take_dir) / EVENT_META_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        print(f"[WARNING] event.json 기록 실패(사이클 영향 없음): {exc}")


def read_event_meta(take_dir) -> dict:
    """take 의 `event.json` 을 읽는다. 없거나 깨졌으면 빈 dict."""
    try:
        return json.loads((Path(take_dir) / EVENT_META_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def prune_events(events_root, keep_runs: int) -> int:
    """최신 `keep_runs` 개 take 만 온전히 두고, 그보다 오래된 take 의 이미지 폴더를 지운다.

    0 이하면 끈다. 지운 폴더 수를 반환하며 예외를 던지지 않는다. take 는 `console.log`
    를 가진 폴더이고(이벤트 루트 또는 `attempt_<n>/`), 최신 판정은 그 파일의 mtime
    (= 사이클이 마지막으로 출력한 시각)이다. 고정 깊이 glob 이라 녹화 프레임 수천 장은
    훑지 않는다.
    """
    root = Path(events_root)
    if keep_runs <= 0 or not root.is_dir():
        return 0
    logs = [*root.glob(f"*/{CONSOLE_LOG_NAME}"), *root.glob(f"*/attempt_*/{CONSOLE_LOG_NAME}")]
    ranked = []
    for log in logs:
        try:
            ranked.append((log.stat().st_mtime, log.parent))
        except OSError:
            pass
    ranked.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for _mtime, take in ranked[keep_runs:]:
        if take == _active:
            continue  # 시계가 뒤로 가면 mtime 순서가 뒤집힌다 - 지금 쓰는 take 는 순서와 무관하게 지킨다.
        for name in HEAVY_DIR_NAMES:
            heavy = take / name
            if not heavy.is_dir():
                continue
            try:
                shutil.rmtree(heavy)
                removed += 1
            except OSError as exc:
                # 뷰어가 프레임을 열어 둔 경우(Windows 파일 잠금) - 다음 사이클이 다시 시도한다.
                print(f"[WARNING] 오래된 이벤트 이미지 삭제 실패(다음 사이클 재시도): {heavy} ({exc})")
    if removed:
        print(f"[INFO] 이벤트 보관 {keep_runs} run 초과분 이미지 폴더 삭제: {removed}개")
    return removed


@contextmanager
def event_scope(take_dir):
    """이 블록 동안 콘솔 전사/debug 이미지/runner 저널을 `take_dir` 로 보낸다.

    같은 폴더로 다시 들어오면 아무것도 하지 않는다 - 모니터가 감지 시점부터 열어 둔
    scope 안에서 `run_alarm_cycle` 이 같은 take 로 한 번 더 연다.
    """
    global _active
    take_dir = Path(take_dir)
    if _active == take_dir:
        yield take_dir
        return
    try:
        take_dir.mkdir(parents=True, exist_ok=True)
        # 줄 단위 flush - 콘솔 창을 닫아 프로세스가 죽어도 마지막 줄까지 남는다.
        log_file = open(take_dir / CONSOLE_LOG_NAME, "a", encoding="utf-8", buffering=1)
    except OSError as exc:
        print(f"[WARNING] 이벤트 폴더를 열지 못해 전역 로그/디버그 폴더를 씁니다: {take_dir} ({exc})")
        yield None
        return

    previous = (_active, sys.stdout, sys.stderr)
    lock = threading.Lock()
    state = {"line_start": True}
    _active = take_dir
    sys.stdout = _Tee(sys.stdout, log_file, lock, state)
    sys.stderr = _Tee(sys.stderr, log_file, lock, state)
    try:
        yield take_dir
    finally:
        _active, sys.stdout, sys.stderr = previous
        with lock:
            try:
                log_file.close()
            except OSError:
                pass  # 디스크가 차서 마지막 flush 가 실패해도 사이클/모니터 루프로 올리지 않는다.
