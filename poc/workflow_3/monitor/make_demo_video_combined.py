"""여러 번의 시연(trial) 녹화를 자막과 함께 하나의 mp4 로 잇는다 - 오프라인.

`manual_align_correction.py` 를 같은 tool/recipe 로 여러 번 돌리면 실행마다 별도
타임스탬프 폴더가 생긴다:

    align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/
      ├─ 260824_101530/recording/   <- 1회차
      ├─ 260824_104512/recording/   <- 2회차
      └─ 260824_112003/recording/   <- 3회차

이 스크립트는 그 폴더들을 시간 순으로 이어 붙이고, 각 구간 앞에 `1st Trial` 타이틀
카드를 넣고 프레임마다 좌상단에 회차 라벨을 새긴다. 보는 사람이 "지금 몇 번째
시도를 보고 있는지" 를 항상 알 수 있게 하는 것이 이 모듈의 목적이다.

`make_demo_video.py` 와 무엇이 다른가:
  단일 시연용 스크립트는 `prelude`(접속 구간 화면 녹화)와 본 녹화를 manifest 의
  절대 시각으로 맞춰 **하나의 시간축**에 올린다. 두 세션이 같은 사이클 안에서 수십
  초 간격이기 때문이다. 회차는 다르다 - 사이에 수십 분~수 시간의 공백이 있어서
  같은 방식으로 합치면 영상의 대부분이 정지 화면이 된다. 그래서 회차는 **각자의
  시간축(t=0 부터)** 을 유지한 채 순서대로 이어 붙인다. 화면의 `t=` 는 언제나
  '그 회차 안에서의 경과시간' 이다.

  프레임 리샘플링·정지구간 압축·letterbox·인코딩은 전부 `make_demo_video` 의 함수를
  그대로 쓴다(포크하지 않는다) - 두 스크립트의 시간축 규약이 갈리면 같은 녹화가
  스크립트에 따라 다르게 보이게 된다.

사용법 - 파일 상단 상수를 고치고 그냥 실행한다:

    # make_demo_video_combined.py 상단
    ROOT = "D:/align_images/MCD513/RJ1BXXX/RJ1B_ISOLINERPOLY_R1/captured_img_from_rcs"
    LABELS = ["Baseline", "After tuning", "3rd try"]

    uv run python poc/workflow_3/monitor/make_demo_video_combined.py

  이 프로젝트는 CLI 인자를 쓰지 않는다. "인자"는 상단 상수이거나 같은 뜻의
  환경변수이고, 우선순위는 **실제 셸 env > 파일 상수 > 코드 기본값** 이다
  (`seed_env_from_constants`). 어느 쪽이 쓰였는지는 시작 시 콘솔에 찍힌다.
  셸에 env 를 붙이는 것은 1회성 오버라이드일 뿐, 평소 경로는 파일 편집이다.

입력 고르기 (셋 중 하나):
  1) INPUT_DIRS 에 recording 폴더를 나열 (리스트로 적으면 그 순서를 그대로 쓴다)
  2) ROOT 아래의 모든 recording 폴더를 자동 수집 (보통 `.../captured_img_from_rcs`)
  3) 둘 다 비우면 ALIGN_IMAGES_DIR 전체를 훑어 recipe 별로 묶고 **가장 최근에
     녹화된 묶음**을 고른다. 다른 후보도 함께 출력하므로 틀렸으면 2)로 지정한다.

  회차 순서는 recording_manifest.json 의 시작 시각(없으면 폴더 tag, 그것도 못 읽으면
  파일 mtime) 기준 오름차순이다 - 폴더 이름 문자열 정렬에 기대지 않는다.

인자 (상수 = env, 자세한 설명은 아래 상수 블록의 주석):
  ROOT              = DEMO_COMBINED_ROOT        모을 폴더
  INPUT_DIRS        = DEMO_COMBINED_INPUT_DIRS  회차를 직접 고를 때
  OUTPUT            = DEMO_COMBINED_OUTPUT      출력 파일
  LABELS            = DEMO_COMBINED_LABELS      회차 라벨
  TITLE_SEC         = DEMO_COMBINED_TITLE_SEC   타이틀 카드 노출 시간 (0=카드 없음)
  SEGMENTS          = DEMO_COMBINED_SEGMENTS    모든 회차에 적용할 남길 구간
  SEGMENTS_BY_TRIAL = DEMO_COMBINED_SEGMENTS_<n> 회차별 구간 (공용보다 우선)
  PREVIEW           = DEMO_COMBINED_PREVIEW     회차별 타임라인 미리보기
  SPEED / FPS / MAX_HOLD_SEC / TAIL_HOLD_SEC / MAX_WIDTH / OVERLAY / PRELUDE
                    = DEMO_VIDEO_*             렌더 knob

  렌더 knob 은 `make_demo_video` 와 **같은 env 이름을 공유한다** - 두 스크립트에서
  같은 뜻이어야 하므로 새 이름을 만들지 않는다.

  TAIL_HOLD_SEC 는 **회차마다** 적용된다 - 다음 타이틀 카드로 넘어가기 전에 마지막
  화면이 잠깐 머무는 편이 읽기 쉽다. TITLE_SEC 은 배속(SPEED)의 영향을 받지 않는다
  (자막을 읽는 시간은 실제 시간이지 영상 속 시간이 아니다).

지원하지 않는 것: 로그 패널(DEMO_VIDEO_LOG_PANEL). 패널은 manifest 의 started_at 을
기준점으로 로그를 정렬하는데 회차마다 기준점이 달라, 한 벌의 패널로는 맞출 수 없다.
로그가 필요한 회차는 `make_demo_video.py` 로 따로 뽑는다.

주의: burn-in 문자는 cv2.putText 라 **ASCII 만** 렌더링된다(한글 라벨은 깨진다).
"""

import os
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.monitor.make_demo_video import (
    _env_flag,
    _env_float,
    _env_int,
    build_timeline,
    canvas_size,
    draw_overlay,
    fit_into_canvas,
    merge_prelude,
    open_writer,
    print_timeline_preview,
    read_image,
    read_started_epoch,
    resolve_segments,
    scan_frames,
    trim_frames,
)

# ===========================================================================
# 실행 인자 - 여기만 고쳐서 쓴다 (셸 env 를 붙이지 않고 실행하려는 경우).
#
# 이 프로젝트는 CLI 인자(argparse/flag)를 쓰지 않는다. 그래서 "인자"는 아래 상수이거나
# 아래 표의 env 이름 둘 중 하나다. 우선순위는 workflow_3_config.py 와 동일하게
#   실제 셸 env  >  아래 상수  >  코드 기본값
# 이며, 어느 쪽이 쓰였는지 시작 시 콘솔에 찍힌다(둘이 어긋난 채 도는 것을 막는다).
#
# 규약: 문자열은 ""(빈 문자열), 숫자/스위치는 None 이 '건드리지 않음'이다. 0 은
# 유효한 값이라(TITLE_SEC=0=카드 없음, MAX_HOLD_SEC=0=압축 안 함) 미설정 표시로
# 쓸 수 없다 - workflow_3_config.py 와 같은 규약.
# ===========================================================================

# [1] 어떤 회차를 모을 것인가 (INPUT_DIRS 가 ROOT 보다 우선, 둘 다 비면 자동 탐색).
ROOT = ""            # 이 폴더 아래 recording 전부. 보통 <...>/captured_img_from_rcs
INPUT_DIRS = []      # 회차를 직접 고를 때. 리스트로 적으면 그 순서를 그대로 쓴다
OUTPUT = ""          # 출력 파일. 비우면 <공통부모>/demo_combined.mp4

# [2] 자막.
LABELS = []          # 회차 라벨 ["Baseline", "After tuning"]. 비면 1st/2nd/3rd Trial
TITLE_SEC = None     # 타이틀 카드 노출 시간 (기본 2.5, 0=카드 없음)

# [3] 편집 (남길 구간). 회차별 지정이 공용보다 우선.
SEGMENTS = ""                # 모든 회차에 적용 "0-30,120-"
SEGMENTS_BY_TRIAL: dict = {} # 회차별 {2: "10-90", 3: "0-45"} (회차 번호는 1-based)
PREVIEW = None               # 회차별 타임라인 미리보기 출력 (기본 off)

# [4] 렌더. make_demo_video 와 같은 뜻이라 env 이름도 DEMO_VIDEO_* 를 공유한다.
SPEED = None         # 1.0=실시간, 2.0=2배속 (기본 1.0)
FPS = None           # 출력 fps (기본 15)
MAX_HOLD_SEC = None  # 한 프레임 최대 유지 = 정지 구간 압축 (기본 2.0, 0=압축 안 함)
TAIL_HOLD_SEC = None # 회차 끝 프레임 유지 (기본 2.0)
MAX_WIDTH = None     # 가로 상한 (기본 1920, 0=원본)
OVERLAY = None       # 프레임 좌하단 t=/f= burn-in (기본 on)
PRELUDE = None       # 회차 안에서 접속 구간 prelude 를 앞에 붙일지 (기본 on)


# (상수명, env 이름) 매핑. 새 인자는 여기 한 줄만 추가하면 된다.
# env 이름은 이 파일과 make_demo_video 가 실제로 읽는 것과 일치해야 한다.
_CONST_TO_ENV = (
    ("ROOT", "DEMO_COMBINED_ROOT"),
    ("INPUT_DIRS", "DEMO_COMBINED_INPUT_DIRS"),
    ("OUTPUT", "DEMO_COMBINED_OUTPUT"),
    ("LABELS", "DEMO_COMBINED_LABELS"),
    ("TITLE_SEC", "DEMO_COMBINED_TITLE_SEC"),
    ("SEGMENTS", "DEMO_COMBINED_SEGMENTS"),
    ("PREVIEW", "DEMO_COMBINED_PREVIEW"),
    ("SPEED", "DEMO_VIDEO_SPEED"),
    ("FPS", "DEMO_VIDEO_FPS"),
    ("MAX_HOLD_SEC", "DEMO_VIDEO_MAX_HOLD_SEC"),
    ("TAIL_HOLD_SEC", "DEMO_VIDEO_TAIL_HOLD_SEC"),
    ("MAX_WIDTH", "DEMO_VIDEO_MAX_WIDTH"),
    ("OVERLAY", "DEMO_VIDEO_OVERLAY"),
    ("PRELUDE", "DEMO_VIDEO_PRELUDE"),
)


def _as_env_value(value) -> str:
    """상수 하나를 env 문자열로 바꾼다 (빈 값이면 "").

    리스트를 받는 이유는 py 파일에서 경로/라벨을 콤마 문자열로 적는 것이 실수하기
    쉽기 때문이다(따옴표 안에서 콤마를 빠뜨리면 두 경로가 한 덩어리가 된다).
    bool 은 "True"/"False" 가 아니라 "1"/"0" 으로 내린다 - 숫자 파서(_env_int)가
    같은 값을 읽을 수도 있는 자리라 문자열 표기를 하나로 맞춘다.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def seed_env_from_constants() -> None:
    """파일 상단 상수를 env 로 setdefault 한다 (셸 env 가 항상 이긴다).

    `workflow_3_config_loader.seed_env()` 와 같은 규약이다 - 한 방향으로만 흐른다:
    상수 -> os.environ -> 기존 reader. 그래서 이 함수를 빼도 env 로 돌던 사용법이
    그대로 살아 있고, 읽는 쪽 코드는 상수의 존재를 몰라도 된다.

    셸 env 때문에 무시된 상수는 반드시 콘솔에 남긴다 - 파일을 고쳤는데 예전 env 가
    남아 있어 다른 영상이 나오는 사고가 이 스크립트에서 제일 흔한 실수다.
    """
    applied: list[str] = []
    ignored: list[str] = []
    for const_name, env_name in _CONST_TO_ENV:
        value = _as_env_value(globals().get(const_name))
        if not value:
            continue
        if os.environ.get(env_name, "").strip():
            ignored.append(f"{const_name}(env {env_name})")
            continue
        os.environ[env_name] = value
        applied.append(f"{const_name}={value}")

    # 회차별 구간은 env 이름이 번호를 달고 있어 표에 못 넣는다 - dict 로 받는다.
    for position, raw in sorted((SEGMENTS_BY_TRIAL or {}).items()):
        value = _as_env_value(raw)
        if not value:
            continue
        env_name = f"DEMO_COMBINED_SEGMENTS_{int(position)}"
        if os.environ.get(env_name, "").strip():
            ignored.append(f"SEGMENTS_BY_TRIAL[{position}](env {env_name})")
            continue
        os.environ[env_name] = value
        applied.append(f"SEGMENTS_BY_TRIAL[{position}]={value}")

    if applied:
        print(f"[INFO] 파일 상수 적용: {', '.join(applied)}")
    if ignored:
        print(f"[INFO] 셸 env 가 이미 있어 파일 상수를 무시: {', '.join(ignored)}")


# 자동 탐색 시 콘솔에 보여줄 후보 묶음 수.
_GROUP_PREVIEW = 5

# 폴더 tag 규약 (`util.time_utils.make_timestamp_tag`). manifest 가 없을 때의 2순위.
_TAG_FORMAT = "%y%m%d_%H%M%S"


# ------------------------------------------------------------------
# 회차 라벨.
# ------------------------------------------------------------------


def ordinal(number: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th' (영어 서수).

    11~13 은 1/2/3 으로 끝나지만 'st/nd/rd' 가 아니다 - 이 예외를 빼먹으면
    11회차가 '11st' 로 나간다.
    """
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def resolve_labels(count: int) -> list[str]:
    """회차 라벨을 정한다 (DEMO_COMBINED_LABELS 우선, 기본은 '1st Trial').

    cv2.putText 는 ASCII 만 그리므로 한글 라벨은 조용히 깨진다 - 미리 경고한다.
    """
    raw = os.environ.get("DEMO_COMBINED_LABELS", "")
    custom = [chunk.strip() for chunk in raw.split(",")] if raw.strip() else []
    if len(custom) > count:
        print(
            f"[WARNING] DEMO_COMBINED_LABELS 가 {len(custom)}개인데 회차는 {count}개입니다 "
            "- 뒤쪽은 무시합니다"
        )

    labels: list[str] = []
    for idx in range(count):
        label = custom[idx] if idx < len(custom) and custom[idx] else f"{ordinal(idx + 1)} Trial"
        if not label.isascii():
            print(f"[WARNING] 라벨에 비ASCII 문자가 있어 화면에서 깨집니다: {label!r}")
        labels.append(label)
    return labels


# ------------------------------------------------------------------
# 입력 탐색.
# ------------------------------------------------------------------


def trial_start_time(recording_dir: Path) -> float:
    """회차의 시작 시각(epoch)을 3단으로 구한다 - 정렬 기준.

    manifest > 폴더 tag > 파일 mtime 순. 문자열 정렬에 기대지 않는 이유는 tag 가
    `%y%m%d_%H%M%S` 라 연도가 바뀌면(25 -> 26) 자릿수 정렬과 시간 정렬이 갈리고,
    사용자가 폴더를 복사·이름변경한 경우에도 순서가 뒤집히기 때문이다.
    """
    epoch = read_started_epoch(recording_dir)
    if epoch is not None:
        return epoch
    try:
        return time.mktime(time.strptime(recording_dir.parent.name, _TAG_FORMAT))
    except (ValueError, OSError):
        pass
    try:
        return recording_dir.stat().st_mtime
    except OSError:
        return 0.0


def find_trial_dirs(root: Path) -> list[Path]:
    """root 아래의 recording 폴더를 시간 순으로 모은다.

    `prelude/` 는 이름이 달라 잡히지 않는다 - 접속 구간은 회차가 아니라 그 회차의
    앞부분이며, `merge_prelude` 가 해당 회차 안에서 이어 붙인다.
    """
    if not root.is_dir():
        return []
    dirs = [path for path in root.rglob("recording") if path.is_dir()]
    dirs.sort(key=trial_start_time)
    return dirs


def group_by_recipe(dirs: list[Path]) -> dict[Path, list[Path]]:
    """recording 폴더를 `captured_img_from_rcs` 단위(= eqp/recipe)로 묶는다.

    경로 규약이 `<...>/captured_img_from_rcs/<tag>/recording` 이라 조부모가 곧
    recipe 묶음이다. 자동 탐색에서 **서로 다른 장비/레시피를 한 영상에 섞지 않기
    위한** 안전장치다.
    """
    groups: dict[Path, list[Path]] = {}
    for path in dirs:
        groups.setdefault(path.parent.parent, []).append(path)
    return groups


def normalize_trial_dir(path: Path) -> Path:
    """tag 폴더를 받으면 그 안의 recording/ 으로 내려간다.

    INPUT_DIRS 에 붙여넣기 좋은 것은 tag 폴더(`captured_img_from_rcs/<tag>`)인데
    프레임은 그 아래 `recording/` 에 있다. 구분을 모르고 tag 를 적으면 모든 회차가
    "프레임이 없어 제외" 되어 결과물이 통째로 비는데, 경고만 봐서는 한 단계 아래를
    가리켜야 한다는 걸 알 수 없다. 프레임이 없고 recording/ 이 있을 때만 내려가므로
    (둘 다 있는 경우는 준 경로를 그대로 존중) 뜻이 갈리지 않는다.
    """
    if scan_frames(path):
        return path
    nested = path / "recording"
    if nested.is_dir() and scan_frames(nested):
        print(f"[INFO] tag 폴더를 받아 recording/ 으로 내려갑니다: {path.name}")
        return nested
    return path


def resolve_trial_dirs() -> list[Path]:
    """env 3단(명시 목록 / ROOT / 자동 탐색)으로 회차 폴더 목록을 정한다."""
    raw = os.environ.get("DEMO_COMBINED_INPUT_DIRS", "").strip()
    if raw:
        dirs: list[Path] = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            path = Path(chunk).expanduser()
            if not path.is_dir():
                print(f"[WARNING] 폴더가 없어 건너뜁니다: {path}")
                continue
            dirs.append(normalize_trial_dir(path))
        # 명시 목록은 **적은 순서를 그대로** 쓴다 - 사용자가 순서를 정한 것이다.
        print(f"[INFO] DEMO_COMBINED_INPUT_DIRS 지정 {len(dirs)}개 (적은 순서 유지)")
        return dirs

    root_raw = os.environ.get("DEMO_COMBINED_ROOT", "").strip()
    if root_raw:
        root = Path(root_raw).expanduser()
        dirs = find_trial_dirs(root)
        print(f"[INFO] DEMO_COMBINED_ROOT={root} 아래 recording {len(dirs)}개")
        return dirs

    print(f"[INFO] 입력 미지정 - {ALIGN_IMAGES_DIR} 에서 자동 탐색")
    groups = group_by_recipe(find_trial_dirs(ALIGN_IMAGES_DIR))
    if not groups:
        print(f"[ERROR] recording 폴더를 찾지 못했습니다: {ALIGN_IMAGES_DIR}")
        return []

    ranked = sorted(
        groups.items(), key=lambda item: trial_start_time(item[1][-1]), reverse=True
    )
    for idx, (parent, dirs) in enumerate(ranked[:_GROUP_PREVIEW]):
        mark = "<- 선택" if idx == 0 else ""
        print(f"[INFO]   {len(dirs):3d} trials  {parent} {mark}")
    if len(ranked) > _GROUP_PREVIEW:
        print(f"[INFO]   ... 외 {len(ranked) - _GROUP_PREVIEW}개 묶음")
    print("[INFO] 다른 묶음을 쓰려면 DEMO_COMBINED_ROOT 로 지정하세요.")
    return ranked[0][1]


# ------------------------------------------------------------------
# 회차.
# ------------------------------------------------------------------


@dataclass
class Trial:
    """영상 한 구간 = 녹화 폴더 하나 (prelude 가 있으면 그것까지 포함)."""

    label: str
    tag: str
    directory: Path
    frames: list[tuple[float, Path]] = field(default_factory=list)
    started_epoch: float = 0.0

    @property
    def span_sec(self) -> float:
        """이 회차 녹화가 담고 있는 실제 시간(초)."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1][0] - self.frames[0][0]

    def subtitle_lines(self, position: int, total: int) -> list[str]:
        """타이틀 카드 아래에 붙는 설명 (ASCII 만)."""
        stamp = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_epoch))
            if self.started_epoch
            else "(start time unknown)"
        )
        return [
            f"trial {position}/{total}   tag {self.tag}",
            stamp,
            f"{len(self.frames)} frames   {self.span_sec:.0f}s recorded",
        ]


def trial_segments(position: int) -> str:
    """이 회차에 적용할 남길-구간 env 를 고른다 (개별 지정이 공용보다 우선)."""
    specific = os.environ.get(f"DEMO_COMBINED_SEGMENTS_{position}", "").strip()
    if specific:
        return specific
    return os.environ.get("DEMO_COMBINED_SEGMENTS", "").strip()


def load_trials(dirs: list[Path]) -> list[Trial]:
    """폴더 목록을 Trial 로 읽는다 (prelude 접합 + 구간 편집 포함).

    번호를 두 단계로 확정한다. 먼저 **프레임이 하나도 없는 폴더를 걸러낸 뒤** 회차
    번호를 매기고, 그 번호로 `DEMO_COMBINED_SEGMENTS_<n>` 을 찾는다. 순서를 뒤집으면
    빈 폴더 하나 때문에 사용자가 지정한 구간이 옆 회차에 붙는다.

    라벨은 **모든 필터가 끝난 뒤** 붙인다 - 편집으로 통째로 사라진 회차가 있으면
    번호가 밀리므로, 그때는 콘솔에 밀렸다는 사실을 남긴다.
    """
    loaded: list[tuple[Path, list[tuple[float, Path]]]] = []
    for directory in dirs:
        frames = scan_frames(directory)
        if not frames:
            print(f"[WARNING] 프레임이 없어 제외합니다: {directory}")
            continue
        # 접속 구간(RCS 실행/로그인/tool 진입)이 녹화돼 있으면 그 회차 앞에 붙인다.
        frames, _ = merge_prelude(directory, frames)
        loaded.append((directory, frames))

    trials: list[Trial] = []
    for position, (directory, frames) in enumerate(loaded, start=1):
        raw_segments = trial_segments(position)
        if raw_segments:
            frames = trim_frames(frames, resolve_segments(frames, raw_segments, 0.0, 0.0))
        if not frames:
            print(f"[WARNING] 편집 후 남은 프레임이 없어 제외합니다: {directory}")
            continue

        # 회차 안에서는 t=0 부터 다시 센다 - 화면의 t= 는 '이 회차의 경과시간'이다.
        origin = frames[0][0]
        trials.append(
            Trial(
                label="",
                tag=directory.parent.name,
                directory=directory,
                frames=[(t_sec - origin, path) for t_sec, path in frames],
                started_epoch=trial_start_time(directory),
            )
        )

    if len(trials) != len(loaded):
        print(f"[WARNING] {len(loaded) - len(trials)}개 회차가 편집으로 사라져 번호를 다시 매깁니다")
    for trial, label in zip(trials, resolve_labels(len(trials))):
        trial.label = label
    return trials


# ------------------------------------------------------------------
# 타이틀 카드.
# ------------------------------------------------------------------


def make_title_card(size: tuple[int, int], title: str, lines: list[str]):
    """회차 시작을 알리는 검은 카드를 그린다 (ASCII 전용).

    글자 크기는 폭에서 파생한다 - 캔버스가 tool 창 크기라 실행마다 달라지므로 고정
    스케일을 쓰면 어떤 해상도에서는 제목이 화면 밖으로 나간다. 제목+부제를 한 덩어리로
    보고 그 **덩어리를 세로 중앙**에 놓는다(제목만 중앙에 두면 부제 무게 때문에
    전체가 아래로 쏠려 보인다).
    """
    width, height = size
    card = np.zeros((height, width, 3), dtype=np.uint8)

    title_scale = max(0.8, min(4.0, width / 520.0))
    title_thick = max(2, int(round(title_scale * 1.8)))
    (title_w, title_h), _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick
    )
    # 그래도 안 들어가면 폭에 맞춰 한 번 더 줄인다.
    if title_w > width * 0.9:
        title_scale *= width * 0.9 / title_w
        title_thick = max(2, int(round(title_scale * 1.8)))
        (title_w, title_h), _ = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick
        )

    sub_scale = max(0.4, title_scale * 0.32)
    sub_thick = max(1, int(round(sub_scale * 2)))
    sub_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, sub_thick)[0]
        for line in lines
    ]
    gap = int(title_h * 0.9)
    block_h = title_h + (gap + sum(int(h * 1.9) for _, h in sub_sizes) if sub_sizes else 0)

    baseline_y = max(title_h, (height - block_h) // 2 + title_h)
    cv2.putText(
        card, title, ((width - title_w) // 2, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 220, 255), title_thick, cv2.LINE_AA,
    )

    y_pos = baseline_y + gap
    for line, (line_w, line_h) in zip(lines, sub_sizes):
        y_pos += int(line_h * 1.9)
        if y_pos > height - 4:
            break
        cv2.putText(
            card, line, ((width - line_w) // 2, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (210, 210, 210), sub_thick, cv2.LINE_AA,
        )
    return card


# ------------------------------------------------------------------
# 인코딩.
# ------------------------------------------------------------------


def probe_canvas(trials: list[Trial], max_width: int) -> tuple[int, int] | None:
    """모든 회차의 소스 폴더에서 1장씩 읽어 공통 캔버스를 정한다.

    회차마다 tool 창 크기가 다를 수 있고(창을 옮기거나 최대화 상태가 다르면) prelude
    는 화면 전체라 종횡비까지 다르다. 첫 회차만 보고 캔버스를 잡으면 뒤 회차가
    찌그러지므로 각 축의 최대치를 캔버스로 잡고 `fit_into_canvas` 로 letterbox 한다.
    """
    probes = []
    seen: set[Path] = set()
    for trial in trials:
        for _, path in trial.frames:
            if path.parent in seen:
                continue
            seen.add(path.parent)
            image = read_image(path)
            if image is not None:
                probes.append(image)
    if not probes:
        return None
    size = canvas_size(probes, max_width)
    if size[0] <= 0 or size[1] <= 0:
        return None
    if len({(im.shape[1], im.shape[0]) for im in probes}) > 1:
        print(
            f"[INFO] 소스 크기 {len(probes)}종 - 종횡비 유지 letterbox 로 합성 "
            f"({sorted({im.shape[1::-1] for im in probes})})"
        )
    return size


def render_trial(
    writer, trial: Trial, size: tuple[int, int], *,
    fps: float, speed: float, max_hold_sec: float, tail_hold_sec: float, overlay: bool,
) -> int:
    """회차 하나를 이미 열린 writer 에 이어 쓴다. 쓴 프레임 수를 반환."""
    starts, skipped, total, skipped_total = build_timeline(
        trial.frames, max_hold_sec, tail_hold_sec
    )
    out_duration = total / speed
    print(
        f"[INFO] [{trial.label}] 원본 {trial.span_sec:.1f}s ({len(trial.frames)} frames) -> "
        f"압축 {starts[-1]:.1f}s (정지 {skipped_total:.1f}s 제거) -> 출력 {out_duration:.1f}s"
    )

    total_out = max(1, int(round(out_duration * fps)))
    cached_index = -1
    cached_image = None
    written = 0
    for step in range(total_out):
        target_ct = step * speed / fps
        index = min(len(trial.frames) - 1, max(0, bisect_right(starts, target_ct) - 1))
        if index != cached_index:
            image = read_image(trial.frames[index][1])
            if image is None:
                # 깨진 프레임 1장 때문에 회차 전체를 버리지 않는다 - 직전 화면 유지.
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
                cached_image.copy(), trial.frames[index][0], index, len(trial.frames),
                skipped[index], trial.label,
            )
        writer.write(frame)
        written += 1
    return written


def render_combined(
    trials: list[Trial],
    out_path: Path,
    *,
    fps: float,
    speed: float,
    max_hold_sec: float,
    tail_hold_sec: float,
    max_width: int,
    overlay: bool,
    title_sec: float,
) -> str:
    """회차 목록을 하나의 영상으로 인코딩한다. 성공 시 'success'."""
    size = probe_canvas(trials, max_width)
    if size is None:
        return "프레임을 하나도 읽지 못했습니다 (경로/파일 손상 확인)"
    print(f"[INFO] 출력 해상도: {size[0]}x{size[1]}, fps={fps:g}, {len(trials)}회차")

    writer, actual_path = open_writer(out_path, fps, size)
    if writer is None:
        return "VideoWriter 를 열지 못했습니다 (코덱 없음)"

    written = 0
    try:
        for position, trial in enumerate(trials, start=1):
            if title_sec > 0:
                # 타이틀은 배속을 적용하지 않는다 - 글을 읽는 시간은 실제 시간이다.
                card = make_title_card(
                    size, trial.label, trial.subtitle_lines(position, len(trials))
                )
                for _ in range(max(1, int(round(title_sec * fps)))):
                    writer.write(card)
                    written += 1
            written += render_trial(
                writer, trial, size,
                fps=fps, speed=speed, max_hold_sec=max_hold_sec,
                tail_hold_sec=tail_hold_sec, overlay=overlay,
            )
            print(f"[INFO]   누적 {written} 프레임 ({written / fps:.1f}s)")
    finally:
        writer.release()

    if written == 0:
        return "인코딩된 프레임이 없습니다 (모든 프레임 읽기 실패)"
    size_mb = actual_path.stat().st_size / (1024 * 1024) if actual_path.exists() else 0.0
    print(
        f"[INFO] 완료 -> {actual_path} "
        f"({written} frames, {written / fps:.1f}s, {size_mb:.1f} MB)"
    )
    return "success"


def resolve_output_path(trials: list[Trial]) -> Path:
    """출력 경로를 정한다 (기본은 회차들의 공통 부모 아래 demo_combined.mp4)."""
    raw = os.environ.get("DEMO_COMBINED_OUTPUT", "").strip()
    if raw:
        return Path(raw).expanduser()
    parents = {trial.directory.parent.parent for trial in trials}
    base = parents.pop() if len(parents) == 1 else Path.cwd()
    return base / "demo_combined.mp4"


def main() -> str:
    """여러 회차 recording 폴더를 자막 붙은 단일 시연 영상으로 만든다."""
    # env 를 읽는 어떤 함수보다 **먼저** 상수를 흘려보낸다.
    seed_env_from_constants()

    dirs = resolve_trial_dirs()
    if not dirs:
        return "회차 폴더를 정하지 못했습니다"

    trials = load_trials(dirs)
    if not trials:
        return "쓸 수 있는 회차가 없습니다 (프레임 없음 / 편집 구간 확인)"

    print(f"[INFO] 회차 {len(trials)}개:")
    for position, trial in enumerate(trials, start=1):
        print(
            f"[INFO]   {position}. {trial.label:<16} {trial.tag}  "
            f"{len(trial.frames):5d} frames  {trial.span_sec:7.1f}s  {trial.directory}"
        )
        if _env_flag("DEMO_COMBINED_PREVIEW", False):
            print_timeline_preview(trial.frames)

    return render_combined(
        trials,
        resolve_output_path(trials),
        fps=max(1.0, _env_float("DEMO_VIDEO_FPS", 15.0)),
        speed=max(0.1, _env_float("DEMO_VIDEO_SPEED", 1.0)),
        max_hold_sec=max(0.0, _env_float("DEMO_VIDEO_MAX_HOLD_SEC", 2.0)),
        tail_hold_sec=max(0.0, _env_float("DEMO_VIDEO_TAIL_HOLD_SEC", 2.0)),
        max_width=max(0, _env_int("DEMO_VIDEO_MAX_WIDTH", 1920)),
        overlay=_env_flag("DEMO_VIDEO_OVERLAY", True),
        title_sec=max(0.0, _env_float("DEMO_COMBINED_TITLE_SEC", 2.5)),
    )


if __name__ == "__main__":
    result = main()
    if result != "success":
        print(f"[EXIT] {result}")
        sys.exit(1)
    sys.exit(0)
