"""녹화 프레임(JPEG) 을 시연용 mp4 로 조립한다 - 오프라인, 루프와 무관.

`RecordingSession` 은 알람 사이클과 수동 녹화에서 tool 창을
`{tag}_rcs_{seq:04d}_{elapsed_ms:08d}ms.jpg` 형식으로 적재한다. 파일명에 경과시간이
박혀 있으므로 별도 메타 없이 **실제 시간축을 그대로 복원**할 수 있다.

왜 '순서대로 고정 fps' 가 아니라 리샘플링인가:
  녹화는 화면이 바뀔 때만 저장하는 change-triggered 샘플링이라 프레임 간격이
  불규칙하다. 파일 순서대로 같은 간격으로 붙이면 **빠른 조작 구간은 늘어지고 정지
  구간은 순식간에 지나가** 실제와 정반대인 영상이 나온다. 그래서 출력 프레임 k 를
  소스 시간축으로 역매핑한다(그 시각에 화면에 떠 있던 프레임을 고른다).

정지 구간 압축(`MAX_HOLD_SEC`):
  align fail 은 장비가 멈춘 상태라 프레임이 수십 초씩 정지한다. 그 구간을 실시간으로
  담으면 아무도 안 본다. 그래서 프레임 1장이 붙들 수 있는 시간에 상한을 두고, 잘려나간
  시간은 화면에 `>> +NNs skipped` 로 **정직하게 표시**한다(몰래 자르지 않는다).

편집(필요 없는 구간 잘라내기):
  실행하면 먼저 **타임라인 미리보기**가 찍힌다 - 소스 구간(접속/본 녹화)과 30초
  단위 프레임 밀도(막대가 길수록 화면이 많이 바뀐 = 조작이 있던 구간)다. 그걸 보고
  남길 구간을 정해 다시 돌리면 된다.

    DEMO_VIDEO_START_SEC=40 DEMO_VIDEO_END_SEC=-20   # 앞 40s 버리고, 뒤 20s 도 버림
    DEMO_VIDEO_SEGMENTS="0-30,120-260"               # 중간을 통째로 들어냄

  잘라낸 구간은 숨기지 않는다 - 프레임 시각은 원본 그대로 두고, 사라진 간격은
  화면에 `>> +NNs skipped` 로 표시된다(정지 구간 압축과 같은 규약).

사용법:
  uv run python poc/workflow_3/monitor/make_demo_video.py
  DEMO_VIDEO_INPUT_DIR=<recording 경로> uv run python poc/workflow_3/monitor/make_demo_video.py

  입력을 지정하지 않으면 ALIGN_IMAGES_DIR 아래에서 **가장 최근 recording 폴더**를
  자동으로 고른다(후보 목록도 함께 출력하므로 다른 것을 고르려면 env 로 지정).

주요 env:
  DEMO_VIDEO_INPUT_DIR    입력 recording 폴더 (미지정 시 자동 탐색)
  DEMO_VIDEO_OUTPUT       출력 파일 (기본 <input>/demo.mp4)
  DEMO_VIDEO_SPEED        1.0=실시간, 2.0=2배속 (기본 1.0)
  DEMO_VIDEO_FPS          출력 fps (기본 15)
  DEMO_VIDEO_MAX_HOLD_SEC 한 프레임 최대 유지 시간 = 정지 구간 압축 (기본 2.0, 0=압축 안 함)
  DEMO_VIDEO_START_SEC    이 경과시간 이후만 (기본 0; 음수면 끝에서부터)
  DEMO_VIDEO_END_SEC      이 경과시간 이전까지 (기본 0=끝까지; 음수면 끝에서 N초 자름)
  DEMO_VIDEO_SEGMENTS     남길 구간 목록 "12-45,120-" (START/END 보다 우선)
  DEMO_VIDEO_MAX_WIDTH    가로 상한 (기본 1920, 0=원본)
  DEMO_VIDEO_OVERLAY      타임스탬프 burn-in (기본 1)
  DEMO_VIDEO_LABEL        좌상단 고정 라벨 (ASCII 만, 기본 없음)
  DEMO_VIDEO_TAIL_HOLD_SEC 마지막 프레임 유지 (기본 2.0)
  DEMO_VIDEO_PRELUDE      recording/prelude/ 접속 구간을 앞에 붙일지 (기본 1)

접속 구간(RCS 실행 -> 로그인 -> tool 진입):
  본 녹화는 tool 창 rect 를 찍으므로 **창이 뜨기 전 장면은 프레임 자체가 없다**.
  루프에서 `ALIGN_FAIL_RECORD_PRELUDE=1` 로 켜면 사이클 시작부터 화면 전체를
  `recording/prelude/` 에 따로 녹화하며, 이 스크립트가 그걸 자동으로 앞에 잇는다.
  두 세션은 t0 가 다르므로 manifest 의 `started_epoch` 차이로 시간축을 맞추고,
  화면(16:9)과 tool 창의 종횡비가 다르므로 캔버스에 letterbox 로 합성한다.

로그 패널 (프레임 옆에 '그때 콘솔에 뭐가 찍혔는지' 합성; `demo_log_panel` 참고):
  DEMO_VIDEO_LOG_PANEL       패널 켜기 (기본 0)
  DEMO_VIDEO_LOG_FILE        감사 로그 (기본 logs/work2.log)
  DEMO_VIDEO_CONSOLE_LOG     콘솔 tee 파일 (선택; 줄마다 시각이 있어야 함)
  DEMO_VIDEO_RUN_DIR         실행 저널 폴더 logs/workflow_runs/<run> (선택)
  DEMO_VIDEO_LOG_PANEL_WIDTH 패널 폭 px (기본 520)
  DEMO_VIDEO_LOG_LINES       패널에 보일 최대 줄 수 (기본 14)
  DEMO_VIDEO_FONT            한글 TrueType 경로 (미지정 시 자동 탐색)

  녹화 프레임은 **tool 창 rect** 만 담기므로 터미널이 절대 찍히지 않는다(prelude
  구간은 화면 전체라 예외 - 그 구간은 터미널이 그대로 보인다). 패널은
  `recording_manifest.json` 의 started_at 을 기준점으로 로그를 시간 정렬해 그 공백을
  메운다. 감사 로그는 stdout 전사가 아니므로, 콘솔 원문이 필요하면 촬영 시 시각을
  붙여 tee 한 뒤 DEMO_VIDEO_CONSOLE_LOG 로 넘긴다. PowerShell 예:

    uv run python poc/workflow_3/monitor/align_fail_monitor.py 2>&1 |
      ForEach-Object { "{0:HH:mm:ss} {1}" -f (Get-Date), $_ } |
      Tee-Object -FilePath console.log

주의: burn-in 문자는 cv2.putText 라 **ASCII 만** 렌더링된다(한글은 깨진다).
      로그 패널은 PIL+TrueType 이라 한글이 나온다(두 경로가 다르다).
"""

import json
import os
import re
import sys
import time
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import ALIGN_IMAGES_DIR, LOG_DIR
from poc.workflow_3.monitor.demo_log_panel import (
    LogPanel,
    load_log_entries,
    load_step_entries,
    read_recording_start,
)

# RecordingSession 이 쓰는 파일명 규약. seq 는 4자리 기준이지만 9999 를 넘으면
# 자릿수가 늘어나므로 하한만 걸고 받는다(경과시간은 항상 8자리 고정).
_FRAME_RE = re.compile(r"_(\d{4,})_(\d{8})ms\.jpg$", re.IGNORECASE)

# 자동 탐색 시 콘솔에 보여줄 최근 후보 수.
_CANDIDATE_PREVIEW = 5

# 편집 구간 한 조각: "12-45" / "120-" / "-45" / "0--30"(끝에서 30초 전까지).
# 앞 그룹이 욕심껏 "-45" 를 먹어도 구분자가 없어 실패하면 역추적으로 빈 시작이 된다.
_SEGMENT_RE = re.compile(
    r"^(-?\d+(?:\.\d+)?)?\s*[-:]\s*(-?\d+(?:\.\d+)?)?$"
)


# ==========================================================================
# 실행 인자 = 이 블록의 상수 (CLI 인자를 쓰지 않는 프로젝트 규약)
# --------------------------------------------------------------------------
# 우선순위: 실제 셸 env > 이 상수 > (없음). 상수가 곧 기본값이다.
# "" / None = 미설정(자동 탐색 또는 기능 off). 숫자 0 은 유효한 값이다.
#
# 오프라인 스크립트다 - RCS 도 장비도 필요 없고 Mac/dev PC 에서 돈다.
# 렌더 knob 의 env 이름은 make_demo_video_combined.py 와 **공유한다**(같은 뜻이므로
# 새 이름을 만들지 않는다).
# ==========================================================================

# --- 입력 / 출력 ---
INPUT_DIR = ""               # recording 폴더. 비우면 가장 최근 것을 자동 탐색.
OUTPUT = ""                  # 출력 mp4. 비우면 입력 폴더 옆에 만든다.
LABEL = ""                   # 화면 좌상단에 새길 라벨.

# --- 자를 구간 ---
SEGMENTS = ""                # 남길 구간 "0-30,120-260" (START/END 보다 우선, 음수=끝에서부터).
                             # 실행하면 먼저 타임라인 미리보기를 찍어 자를 지점을 고르게 한다.
START_SEC = 0.0              # SEGMENTS 가 비었을 때만 쓰인다.
END_SEC = 0.0                # 0 = 끝까지.

# --- 렌더 ---
SPEED = 1.0                  # 1.0=실시간, 2.0=2배속.
FPS = 15.0
MAX_HOLD_SEC = 2.0           # 정지 구간을 이 길이로 압축(원본은 화면이 오래 안 바뀐다).
TAIL_HOLD_SEC = 2.0          # 마지막 화면 머무는 시간.
MAX_WIDTH = 1920             # 0 = 원본 크기.
OVERLAY = 1                  # 경과시간 등 burn-in 오버레이.

# --- 접속 구간(prelude) 잇기 ---
PRELUDE = 1                  # recording/prelude/ 가 있으면 앞에 잇는다.
                             # 본 녹화는 tool 창 rect 라 'RCS 실행->로그인->tool 진입'
                             # 구간의 프레임이 원리상 없다. 그 구간은 화면 전체 녹화이며
                             # manifest 의 started_epoch 차이로 시간축을 맞춘다.

# --- 로그 패널 (녹화 옆에 콘솔/감사 로그를 합성) ---
LOG_PANEL = 0                # 켜면 아래 경로들을 시간 정렬해 프레임 옆에 그린다.
LOG_PANEL_WIDTH = 520
LOG_LINES = 14
LOG_FILE = ""                # work2.log 경로. 비우면 자동 탐색.
CONSOLE_LOG = ""             # 콘솔 tee 파일 경로.
RUN_DIR = ""                 # step_*.json 이 있는 run 폴더.


def _env_float(name: str, default: float) -> float:
    """env 를 float 으로 읽는다 (빈값/파싱실패는 기본값)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARNING] {name}={raw!r} 파싱 실패 - 기본값 {default} 사용")
        return default


def _env_int(name: str, default: int) -> int:
    """env 를 int 로 읽는다 (빈값/파싱실패는 기본값)."""
    return int(_env_float(name, float(default)))


def _env_flag(name: str, default: bool) -> bool:
    """env 를 bool 로 읽는다 ('0'/'false'/'no' 만 거짓)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


# ------------------------------------------------------------------
# 입력 탐색.
# ------------------------------------------------------------------


def find_recording_dirs(root: Path) -> list[Path]:
    """root 아래의 recording 폴더를 최근 순으로 찾는다.

    알람 사이클(`captured_img_from_rcs/<tag>/recording`) 과 수동 녹화
    (`_manual/<tag>/recording`) 가 같은 이름을 쓰므로 한 번에 잡힌다.
    """
    if not root.exists():
        return []
    dirs = [p for p in root.rglob("recording") if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def resolve_input_dir() -> Path | None:
    """DEMO_VIDEO_INPUT_DIR 또는 가장 최근 recording 폴더를 고른다."""
    raw = os.environ.get("DEMO_VIDEO_INPUT_DIR", INPUT_DIR).strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_dir():
            print(f"[ERROR] 입력 폴더가 없습니다: {path}")
            return None
        return path

    print(f"[INFO] DEMO_VIDEO_INPUT_DIR 미지정 - {ALIGN_IMAGES_DIR} 에서 자동 탐색")
    candidates = find_recording_dirs(ALIGN_IMAGES_DIR)
    if not candidates:
        print(f"[ERROR] recording 폴더를 찾지 못했습니다: {ALIGN_IMAGES_DIR}")
        return None

    for idx, path in enumerate(candidates[:_CANDIDATE_PREVIEW]):
        frames = len(list(path.glob("*.jpg")))
        mark = "<- 선택" if idx == 0 else ""
        print(f"[INFO]   {frames:5d} frames  {path} {mark}")
    if len(candidates) > _CANDIDATE_PREVIEW:
        print(f"[INFO]   ... 외 {len(candidates) - _CANDIDATE_PREVIEW}개")
    print("[INFO] 다른 것을 쓰려면 DEMO_VIDEO_INPUT_DIR 로 지정하세요.")
    return candidates[0]


def scan_frames(input_dir: Path) -> list[tuple[float, Path]]:
    """recording 폴더의 프레임을 (경과초, 경로) 목록으로 읽는다.

    파일명 규약에 맞지 않는 jpg(디버그 산출물 등)는 조용히 건너뛴다.
    """
    items: list[tuple[float, int, Path]] = []
    for path in input_dir.glob("*.jpg"):
        matched = _FRAME_RE.search(path.name)
        if not matched:
            continue
        items.append((int(matched.group(2)) / 1000.0, int(matched.group(1)), path))
    items.sort(key=lambda item: (item[0], item[1]))
    return [(t_sec, path) for t_sec, _, path in items]


def read_started_epoch(recording_dir: Path) -> float | None:
    """recording_manifest.json 의 절대 시작 시각(epoch)을 읽는다.

    `started_epoch` 이 있으면 그걸 쓴다. 없는 구버전 녹화는 `started_at` 문자열을
    쓰되 **초 단위 해상도**라 20fps 구간에서 최대 20프레임까지 어긋날 수 있다 -
    두 세션을 잇는 데는 충분하지만, 정확도 차이를 아는 채로 쓰라고 구분해 둔다.
    """
    path = recording_dir / "recording_manifest.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[WARNING] {path} 해석 실패: {exc}")
        return None
    epoch = raw.get("started_epoch")
    if isinstance(epoch, (int, float)):
        return float(epoch)
    stamp = str(raw.get("started_at", ""))
    try:
        return time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def merge_prelude(
    input_dir: Path, main_frames: list[tuple[float, Path]]
) -> tuple[list[tuple[float, Path]], Path]:
    """접속 구간 prelude 프레임을 앞에 이어 붙인다.

    prelude(화면 전체 녹화)와 본 녹화(tool 창)는 **세션이 달라 t0 가 다르다**.
    파일명의 elapsed_ms 는 각 세션 기준 상대시간이므로 그대로 이으면 두 구간이
    겹쳐 버린다. 그래서 두 manifest 의 절대 시작 시각 차이만큼 본 녹화를 밀어
    하나의 시간축에 올린다.

    기준점을 못 구하면(구버전 녹화·manifest 유실) 시간을 **지어내지 않는다** -
    prelude 마지막 프레임 뒤 1초에 본 녹화를 붙이고 근사임을 콘솔에 밝힌다.
    반환값의 두 번째는 로그 패널이 쓸 기준 manifest 폴더(가장 이른 세션)다.
    """
    if not _env_flag("DEMO_VIDEO_PRELUDE", bool(PRELUDE)):
        return main_frames, input_dir

    prelude_dir = input_dir / "prelude"
    pre_frames = scan_frames(prelude_dir) if prelude_dir.is_dir() else []
    if not pre_frames:
        return main_frames, input_dir

    pre_epoch = read_started_epoch(prelude_dir)
    main_epoch = read_started_epoch(input_dir)
    offset = None
    if pre_epoch is not None and main_epoch is not None:
        offset = main_epoch - pre_epoch
    if offset is None or offset < 0:
        offset = pre_frames[-1][0] + 1.0
        print(
            "[WARNING] prelude/본 녹화의 절대 시작 시각을 못 구했습니다 - "
            f"근사 접합({offset:.1f}s 뒤에 본 녹화 배치)"
        )
    print(
        f"[INFO] 접속 구간 prelude {len(pre_frames)} 프레임 병합 "
        f"({pre_frames[-1][0] - pre_frames[0][0]:.1f}s), 본 녹화 오프셋 {offset:.1f}s"
    )
    merged = pre_frames + [(t_sec + offset, path) for t_sec, path in main_frames]
    merged.sort(key=lambda item: item[0])
    return merged, prelude_dir


def print_timeline_preview(frames: list[tuple[float, Path]], bucket_sec: float = 30.0):
    """어디에 무엇이 있는지 콘솔에 요약한다 - 자를 구간을 고르라고 있는 출력이다.

    잘라내기 자체는 env 숫자 두 개면 되지만, **몇 초를 적어야 하는지**는 영상을
    만들어 보기 전에는 알 수 없다. 그래서 소스별 구간(접속/본 녹화)과 구간별
    프레임 수(=화면이 얼마나 바뀌었나 = 조작이 있었나)를 먼저 보여준다.
    """
    if not frames:
        return
    span_start, span_end = frames[0][0], frames[-1][0]
    print(f"[INFO] 타임라인 {span_start:.0f}s ~ {span_end:.0f}s ({len(frames)} frames)")

    # 소스(폴더)가 바뀌는 지점 = 접속 -> tool 창 인계 지점.
    current = frames[0][1].parent
    started = span_start
    for t_sec, path in frames:
        if path.parent != current:
            print(f"[INFO]   {started:7.0f}s ~ {t_sec:7.0f}s  [{current.name}]")
            current, started = path.parent, t_sec
    print(f"[INFO]   {started:7.0f}s ~ {span_end:7.0f}s  [{current.name}]")

    # 구간별 프레임 밀도 - 막대가 긴 구간이 조작이 몰린 구간이다. 긴 녹화에서
    # 30초 고정이면 줄이 수십 개 나오므로 20줄 안쪽으로 버킷을 넓힌다.
    span = max(1.0, span_end - span_start)
    step = max(bucket_sec, round(span / 20 / 10) * 10)
    counts: dict[int, int] = {}
    for t_sec, _ in frames:
        bucket = int((t_sec - span_start) // step)
        counts[bucket] = counts.get(bucket, 0) + 1
    peak = max(counts.values()) if counts else 1
    for bucket in sorted(counts):
        head = span_start + bucket * step
        bar = "#" * max(1, int(round(counts[bucket] * 30 / peak)))
        print(f"[INFO]   {head:7.0f}s |{bar:<30}| {counts[bucket]}")


def parse_segments(raw: str, first_t: float, last_t: float) -> list[tuple[float, float]]:
    """"12-45, 120-" 같은 남길 구간 목록을 파싱한다 (구분자는 '-' 또는 ':').

    한쪽이 비면 열린 구간이다("-45" = 처음부터 45s, "120-" = 120s 부터 끝까지).
    음수는 **끝에서부터** 센다("-30" 이 아니라 "0--30" 이 아니라, 끝값에 -30 을 쓰면
    마지막 30초를 버린다) - 영상 길이를 모르는 채로 뒷부분을 자를 때 쓴다.

    해석 실패한 조각은 조용히 버리지 않고 경고한다 - 시연 직전에 오타 하나로
    엉뚱한 구간이 나가는 게 최악이다.
    """
    segments: list[tuple[float, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 손으로 split 하면 "0--30"(끝에서 30초 전까지) 의 음수 부호와 구분자가
        # 섞여 조각 수가 어긋난다. 역추적이 되는 정규식에 맡긴다.
        matched = _SEGMENT_RE.match(chunk)
        if not matched:
            print(f"[WARNING] 구간 형식을 못 읽었습니다(무시): {chunk!r} - 예) 12-45, 120-")
            continue
        head, tail = matched.group(1), matched.group(2)
        start = float(head) if head else first_t
        end = float(tail) if tail else last_t
        if start < 0:
            start += last_t
        if end <= 0:
            end += last_t
        if end <= start:
            print(f"[WARNING] 끝이 시작보다 앞인 구간(무시): {chunk!r} -> {start:.1f}~{end:.1f}")
            continue
        segments.append((start, end))
    return sorted(segments)


def resolve_segments(
    frames: list[tuple[float, Path]], raw: str, start_sec: float, end_sec: float
) -> list[tuple[float, float]]:
    """env 를 남길 구간 목록으로 정리한다 (SEGMENTS 가 START/END 보다 우선).

    START/END 는 구간 하나의 축약형일 뿐이라 같은 경로로 흘려보낸다 - 두 벌의
    필터 규칙이 생기면 "어느 쪽이 이겼는지" 를 콘솔만 보고 알 수 없게 된다.
    """
    first_t, last_t = frames[0][0], frames[-1][0]
    if raw.strip():
        segments = parse_segments(raw, first_t, last_t)
        if segments:
            return segments
        print("[WARNING] DEMO_VIDEO_SEGMENTS 를 하나도 못 읽어 전체를 씁니다")
        return []
    if start_sec <= 0 and end_sec == 0:
        return []
    start = start_sec + last_t if start_sec < 0 else start_sec
    end = end_sec + last_t if end_sec <= 0 else end_sec
    return [(max(first_t, start), end)]


def trim_frames(
    frames: list[tuple[float, Path]], segments: list[tuple[float, float]]
) -> list[tuple[float, Path]]:
    """남길 구간에 드는 프레임만 고른다 (빈 목록이면 전체).

    잘려나간 구간을 **따로 이어붙이지 않는다** - 프레임의 t_sec 은 원본 시각 그대로
    두고, 사라진 간격은 `build_timeline` 이 압축하면서 화면에 `>> +NNs skipped` 로
    표시한다. 그래야 overlay 시각과 로그 패널이 원본과 계속 맞고, 편집한 사실이
    영상에 남는다(이 모듈의 '몰래 자르지 않는다' 원칙).
    """
    if not segments:
        return frames
    kept = [
        (t_sec, path)
        for t_sec, path in frames
        if any(start <= t_sec <= end for start, end in segments)
    ]
    spans = ", ".join(f"{start:.0f}-{end:.0f}s" for start, end in segments)
    print(f"[INFO] 편집: 남길 구간 [{spans}] -> {len(kept)}/{len(frames)} 프레임")
    return kept


# ------------------------------------------------------------------
# 시간축.
# ------------------------------------------------------------------


def build_timeline(
    frames: list[tuple[float, Path]], max_hold_sec: float, tail_hold_sec: float
) -> tuple[list[float], list[float], float, float]:
    """압축된 출력 시간축을 만든다.

    프레임 i 가 실제로 화면에 떠 있던 시간(다음 프레임까지의 간격)을 `max_hold_sec`
    으로 자른 것이 출력에서의 유지 시간이다. 반환값:

      starts      프레임 i 의 출력 시작 시각 (압축 후, 초)
      skipped     프레임 i 에서 잘려나간 시간 (초; 0 이면 손대지 않음)
      total       전체 출력 길이 (초, tail 포함)
      skipped_total 잘려나간 총 시간 (초)
    """
    starts = [0.0]
    skipped = []
    skipped_total = 0.0
    for idx in range(len(frames) - 1):
        gap = max(0.0, frames[idx + 1][0] - frames[idx][0])
        held = min(gap, max_hold_sec) if max_hold_sec > 0 else gap
        skipped.append(gap - held)
        skipped_total += gap - held
        starts.append(starts[-1] + held)
    skipped.append(0.0)  # 마지막 프레임은 다음이 없으므로 자를 것도 없다.
    total = starts[-1] + max(0.0, tail_hold_sec)
    return starts, skipped, total, skipped_total


# ------------------------------------------------------------------
# 이미지.
# ------------------------------------------------------------------


def read_image(path: Path):
    """JPEG 를 읽는다 (Windows 비ASCII 경로 대응).

    `cv2.imread` 는 Windows 에서 비ASCII 경로를 조용히 실패시키고 None 을 돌려준다.
    오피스 경로에 한글이 섞일 수 있으므로 바이트로 읽어 디코드한다.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        print(f"[WARNING] 프레임 읽기 실패({path.name}): {exc}")
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def canvas_size(images: list, max_width: int) -> tuple[int, int]:
    """출력 캔버스 크기를 정한다 (가로 상한 + 짝수 보정).

    프레임 소스가 **둘 이상**일 수 있다: 접속 구간은 화면 전체(16:9), 본 녹화는
    tool 창 rect 로 종횡비가 다르다. 한쪽 크기에 맞춰 늘리면 다른 쪽이 찌그러지므로,
    양쪽을 담을 수 있게 각 축의 최대치를 캔버스로 잡고 프레임은 비율을 지킨 채
    이 안에 넣는다(`fit_into_canvas`).

    mp4 코덱은 홀수 해상도를 거부하는 경우가 있어 짝수로 내림한다.
    """
    width = max(image.shape[1] for image in images)
    height = max(image.shape[0] for image in images)
    if 0 < max_width < width:
        height = int(round(height * max_width / width))
        width = max_width
    return width - (width % 2), height - (height % 2)


def fit_into_canvas(image, size: tuple[int, int]):
    """종횡비를 지킨 채 캔버스 안에 넣고 남는 자리는 검게 채운다 (letterbox).

    단순 `resize(image, size)` 는 종횡비가 다른 프레임을 늘려 버린다 - 시연 영상에서
    장비 화면이 찌그러지면 "이게 실제 화면이다" 라는 영상의 목적 자체가 흔들린다.
    """
    target_w, target_h = size
    height, width = image.shape[:2]
    if (width, height) == size:
        return image
    scale = min(target_w / width, target_h / height)
    new_w = max(1, min(target_w, int(round(width * scale))))
    new_h = max(1, min(target_h, int(round(height * scale))))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def draw_overlay(
    image, t_sec: float, index: int, total_frames: int, skipped_sec: float, label: str
):
    """경과시간/프레임번호/압축표시를 burn-in 한다 (ASCII 전용).

    cv2.putText 는 한글을 렌더링하지 못하므로 의도적으로 ASCII 만 쓴다.
    잘려나간 정지 시간을 함께 표시해 '몰래 편집한 영상' 이 되지 않게 한다.
    """
    height, width = image.shape[:2]
    scale = max(0.5, min(1.2, width / 1600.0))
    thickness = max(1, int(round(scale * 2)))
    text = f"t={t_sec:7.1f}s  f={index + 1}/{total_frames}"
    if skipped_sec >= 1.0:
        text += f"   >> +{skipped_sec:.0f}s skipped"

    baseline_y = height - int(18 * scale)
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(
        image,
        (int(10 * scale), baseline_y - text_h - int(10 * scale)),
        (int(20 * scale) + text_w, baseline_y + int(8 * scale)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image, text, (int(15 * scale), baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

    if label:
        cv2.putText(
            image, label, (int(15 * scale), int(40 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 255), thickness, cv2.LINE_AA,
        )
    return image


def build_log_panel(
    manifest_dir: Path, frames: list[tuple[float, Path]], height: int
) -> LogPanel | None:
    """로그 패널을 만든다 (비활성/기준점 없음/로그 없음이면 None).

    기준점은 `recording_manifest.json` 의 started_at 이다. 그게 없으면 로그를 프레임에
    맞출 방법이 없으므로 **패널을 조용히 붙이지 않고** 사유를 남기고 포기한다 -
    시각이 어긋난 로그를 붙이면 없느니만 못하다.
    """
    if not _env_flag("DEMO_VIDEO_LOG_PANEL", bool(LOG_PANEL)):
        return None

    # 기준점은 **가장 이른 세션**의 manifest 다. prelude 가 붙으면 시간축 0 이
    # 접속 구간 시작으로 옮겨가므로, 본 녹화 manifest 를 기준으로 삼으면 로그가
    # 통째로 밀린다.
    base = read_recording_start(manifest_dir)
    if base is None:
        print("[WARNING] recording_manifest.json 의 started_at 이 없어 로그 패널을 생략합니다")
        return None

    log_paths = []
    audit = os.environ.get("DEMO_VIDEO_LOG_FILE", LOG_FILE).strip()
    log_paths.append(Path(audit).expanduser() if audit else LOG_DIR / "work2.log")
    console = os.environ.get("DEMO_VIDEO_CONSOLE_LOG", CONSOLE_LOG).strip()
    if console:
        log_paths.append(Path(console).expanduser())

    span = frames[-1][0] - frames[0][0]
    entries = load_log_entries(log_paths, base, span)

    run_dir = os.environ.get("DEMO_VIDEO_RUN_DIR", RUN_DIR).strip()
    if run_dir:
        entries.extend(load_step_entries(Path(run_dir).expanduser(), base))
        entries.sort(key=lambda entry: entry.t_sec)

    if not entries:
        print("[WARNING] 녹화 구간에 걸치는 로그 줄이 없어 패널을 생략합니다 "
              "(로그 파일 경로/시각을 확인하세요)")
        return None

    width = max(240, _env_int("DEMO_VIDEO_LOG_PANEL_WIDTH", LOG_PANEL_WIDTH))
    width -= width % 2
    print(f"[INFO] 로그 패널: {len(entries)}줄, 폭 {width}px, 기준 {base:%Y-%m-%d %H:%M:%S}")
    return LogPanel(
        entries, width, height, max_lines=_env_int("DEMO_VIDEO_LOG_LINES", LOG_LINES)
    )


# ------------------------------------------------------------------
# 인코딩.
# ------------------------------------------------------------------


def open_writer(out_path: Path, fps: float, size: tuple[int, int]):
    """VideoWriter 를 연다 (mp4v 실패 시 MJPG/avi 로 폴백).

    오피스 PC 의 OpenCV 빌드에 따라 mp4v 가 없을 수 있다. 영상 자체를 못 만드는 것보다
    avi 라도 나오는 편이 낫고, 어느 쪽으로 떨어졌는지는 콘솔에 남긴다.
    """
    if not out_path.name.isascii():
        print("[WARNING] 출력 경로에 비ASCII 문자가 있습니다 - 인코딩 실패 시 ASCII 경로로 바꿔보세요")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, out_path
    writer.release()

    fallback = out_path.with_suffix(".avi")
    print(f"[WARNING] mp4v 인코더 사용 불가 - MJPG/avi 로 폴백: {fallback.name}")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        return writer, fallback
    writer.release()
    return None, out_path


def render(
    frames: list[tuple[float, Path]],
    out_path: Path,
    manifest_dir: Path,
    *,
    fps: float,
    speed: float,
    max_hold_sec: float,
    tail_hold_sec: float,
    max_width: int,
    overlay: bool,
    label: str,
) -> str:
    """프레임 목록을 영상으로 인코딩한다. 성공 시 'success'."""
    starts, skipped, total, skipped_total = build_timeline(
        frames, max_hold_sec, tail_hold_sec
    )
    source_span = frames[-1][0] - frames[0][0]
    out_duration = total / speed
    print(
        f"[INFO] 원본 {source_span:.1f}s ({len(frames)} frames) -> "
        f"압축 {starts[-1]:.1f}s (정지 {skipped_total:.1f}s 제거) "
        f"+ tail {tail_hold_sec:.1f}s -> 출력 {out_duration:.1f}s @ {speed:g}x"
    )

    # 소스 폴더마다 대표 1장을 읽어 캔버스를 정한다 - prelude(화면 전체)와 본
    # 녹화(tool 창)는 크기·종횡비가 다르므로 첫 프레임만 보면 뒤쪽이 찌그러진다.
    probes = []
    seen_dirs = set()
    for _, path in frames:
        if path.parent in seen_dirs:
            continue
        seen_dirs.add(path.parent)
        image = read_image(path)
        if image is not None:
            probes.append(image)
    if not probes:
        return f"프레임을 하나도 읽지 못했습니다: {frames[0][1]}"
    size = canvas_size(probes, max_width)
    if size[0] <= 0 or size[1] <= 0:
        return f"해상도 계산 실패: {[im.shape for im in probes]}"
    if len(probes) > 1:
        print(f"[INFO] 소스 {len(probes)}종({[im.shape[1::-1] for im in probes]}) - "
              "종횡비 유지 letterbox 로 합성")

    # 패널은 프레임과 같은 높이로 옆에 붙으므로 출력 폭이 늘어난다.
    panel = build_log_panel(manifest_dir, frames, size[1])
    out_size = (size[0] + panel.width, size[1]) if panel else size
    print(f"[INFO] 출력 해상도: {out_size[0]}x{out_size[1]}, fps={fps:g}")

    writer, actual_path = open_writer(out_path, fps, out_size)
    if writer is None:
        return "VideoWriter 를 열지 못했습니다 (코덱 없음)"

    total_out = max(1, int(round(out_duration * fps)))
    cached_index = -1
    cached_image = None
    written = 0
    try:
        for step in range(total_out):
            target_ct = step * speed / fps
            index = min(len(frames) - 1, max(0, bisect_right(starts, target_ct) - 1))
            if index != cached_index:
                image = read_image(frames[index][1])
                if image is None:
                    # 깨진 프레임 1장 때문에 영상 전체를 버리지 않는다 - 직전 화면을 유지.
                    if cached_image is None:
                        continue
                else:
                    cached_image = fit_into_canvas(image, size)
                    cached_index = index
            if cached_image is None:
                continue

            frame = cached_image
            if overlay:
                frame = draw_overlay(
                    cached_image.copy(), frames[index][0], index, len(frames),
                    skipped[index], label,
                )
            if panel is not None:
                frame = np.hstack((frame, panel.render(frames[index][0])))
            writer.write(frame)
            written += 1
            if written % 300 == 0:
                print(f"[INFO]   인코딩 {written}/{total_out} 프레임")
    finally:
        writer.release()

    if written == 0:
        return "인코딩된 프레임이 없습니다 (모든 프레임 읽기 실패)"
    size_mb = actual_path.stat().st_size / (1024 * 1024) if actual_path.exists() else 0.0
    print(f"[INFO] 완료 -> {actual_path} ({written} frames, {size_mb:.1f} MB)")
    return "success"


def main() -> str:
    """recording 폴더를 시연 영상으로 만든다."""
    input_dir = resolve_input_dir()
    if input_dir is None:
        return "입력 폴더를 정하지 못했습니다"

    frames = scan_frames(input_dir)
    if not frames:
        return f"프레임을 찾지 못했습니다 (파일명 규약 불일치?): {input_dir}"
    print(f"[INFO] 입력: {input_dir} ({len(frames)} frames)")

    # 접속 구간(RCS 실행/로그인/tool 진입) 화면 녹화가 있으면 앞에 잇는다.
    frames, manifest_dir = merge_prelude(input_dir, frames)

    print_timeline_preview(frames)

    segments = resolve_segments(
        frames,
        os.environ.get("DEMO_VIDEO_SEGMENTS", SEGMENTS),
        _env_float("DEMO_VIDEO_START_SEC", START_SEC),
        _env_float("DEMO_VIDEO_END_SEC", END_SEC),
    )
    frames = trim_frames(frames, segments)
    if len(frames) < 2:
        return f"프레임이 부족합니다 ({len(frames)}장) - 편집 구간을 확인하세요"

    raw_output = os.environ.get("DEMO_VIDEO_OUTPUT", OUTPUT).strip()
    out_path = Path(raw_output).expanduser() if raw_output else input_dir / "demo.mp4"

    return render(
        frames,
        out_path,
        manifest_dir,
        fps=max(1.0, _env_float("DEMO_VIDEO_FPS", FPS)),
        speed=max(0.1, _env_float("DEMO_VIDEO_SPEED", SPEED)),
        max_hold_sec=max(0.0, _env_float("DEMO_VIDEO_MAX_HOLD_SEC", MAX_HOLD_SEC)),
        tail_hold_sec=max(0.0, _env_float("DEMO_VIDEO_TAIL_HOLD_SEC", TAIL_HOLD_SEC)),
        max_width=max(0, _env_int("DEMO_VIDEO_MAX_WIDTH", MAX_WIDTH)),
        overlay=_env_flag("DEMO_VIDEO_OVERLAY", bool(OVERLAY)),
        label=os.environ.get("DEMO_VIDEO_LABEL", LABEL).strip(),
    )


if __name__ == "__main__":
    result = main()
    if result != "success":
        print(f"[EXIT] {result}")
        sys.exit(1)
    sys.exit(0)
