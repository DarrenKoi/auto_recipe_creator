"""시연 영상용 로그 패널 - 프레임 옆에 '그때 콘솔에 뭐가 찍혔는지' 를 합성한다.

왜 필요한가:
  `RecordingSession` 은 **tool 창 rect 의 스크린 그랩**이라 터미널이 프레임에 없다.
  RCS 가 전체화면이면 터미널은 뒤에 가려 있기까지 하다. 그래서 녹화 프레임만으로
  영상을 만들면 "장비 화면만 보이고 시스템이 무슨 판단을 했는지는 안 보이는" 영상이
  된다. 이미 지나간 실알람 녹화는 화면 녹화를 소급해서 할 수도 없다.

어떻게 되살리는가:
  `recording_manifest.json` 의 `started_at`(벽시계) + 프레임 파일명의 `elapsed_ms` 로
  **프레임마다 절대 시각**이 나온다. 로그도 타임스탬프를 갖고 있으므로, 그 시각까지
  찍힌 줄을 골라 패널로 그려 붙이면 시간이 맞는다.

두 가지 소스를 받는다 (섞어 쓸 수 있다):
  * 감사 로그 `logs/work2.log` - **소급 적용 가능**. 이미 녹화된 알람에도 쓸 수 있다.
    다만 stdout 전사가 아니라 감사 이벤트라, 콘솔의 모든 `[INFO]` 줄이 있지는 않다.
  * 콘솔 tee 파일 - 진짜 콘솔 전사. 대신 촬영 전에 걸어둬야 하고 **줄마다 시각이
    붙어 있어야** 한다(PowerShell 예시는 make_demo_video docstring 참고).
  * 실행 저널 `logs/workflow_runs/<run>/step_*.json` - step 경계. 데모에서 가장 읽기
    좋은 줄("STEP connect_tool ok")이라 있으면 함께 넣는다.

한글 렌더링:
  cv2.putText 는 한글을 못 그린다. 로그가 한국어라 **PIL + TrueType** 로 그린다.
  폰트를 못 찾으면 패널을 끄지 않고 경고 후 기본 비트맵 폰트로 떨어진다(한글은 깨지지만
  영상 생성 자체는 실패시키지 않는다).
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# `%Y-%m-%d %H:%M:%S [LEVEL] message` - logger.py 의 Formatter 와 같은 규약.
_FULL_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*(?:\[(\w+)\])?\s*(.*)$"
)
# `HH:MM:SS ...` - 콘솔 tee 에 시각만 붙인 경우.
_TIME_ONLY_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$")
# 콘솔 print 규약의 레벨 표기.
_LEVEL_RE = re.compile(r"\[(INFO|WARNING|ERROR|DIGEST)\]")

# 감사 로그의 기계용 필드 - 패널에서는 앞의 둘만 살리고 나머지는 뒤에 붙인다.
_COMPONENT_RE = re.compile(r"component=(\S+)")
_MESSAGE_RE = re.compile(r"message=(\S+)")

_LEVEL_COLORS = {
    "INFO": (216, 222, 233),
    "WARNING": (250, 196, 90),
    "ERROR": (255, 110, 110),
    "STEP": (120, 230, 200),
}
_DEFAULT_COLOR = (170, 178, 190)

# 한글이 되는 TrueType 후보. 오피스(Windows)가 첫 줄에 걸리도록 둔다.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunsl.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)


@dataclass
class LogEntry:
    """패널에 그릴 로그 한 줄 (녹화 시작 기준 상대 시각)."""

    t_sec: float
    level: str
    text: str


# ------------------------------------------------------------------
# 시각 기준점.
# ------------------------------------------------------------------


def read_recording_start(input_dir: Path) -> datetime | None:
    """recording_manifest.json 의 started_at 을 읽는다 (없으면 None).

    이 값이 프레임 elapsed_ms 를 절대 시각으로 바꾸는 기준점이다. 없으면 로그를
    프레임에 맞출 방법이 없으므로 패널을 켤 수 없다.
    """
    path = input_dir / "recording_manifest.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return datetime.strptime(str(raw.get("started_at", "")), "%Y-%m-%dT%H:%M:%S")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[WARNING] recording_manifest.json 의 started_at 해석 실패: {exc}")
        return None


def _parse_line_time(line: str, base: datetime) -> tuple[datetime, str, str] | None:
    """로그 한 줄에서 (시각, 레벨, 본문) 을 뽑는다. 시각이 없으면 None.

    날짜 없이 시각만 있는 형식(콘솔 tee)은 녹화 시작일의 날짜를 빌려 쓴다.
    """
    matched = _FULL_TS_RE.match(line)
    if matched:
        stamp = matched.group(1).replace("T", " ")
        try:
            when = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return when, (matched.group(2) or "").upper(), matched.group(3).strip()

    matched = _TIME_ONLY_RE.match(line)
    if not matched:
        return None
    try:
        clock = datetime.strptime(matched.group(1), "%H:%M:%S").time()
    except ValueError:
        return None
    body = matched.group(2).strip()
    level_match = _LEVEL_RE.search(body)
    level = level_match.group(1) if level_match else ""
    return datetime.combine(base.date(), clock), level, body


def _condense(text: str) -> str:
    """감사 로그의 `component=/message=` 를 사람이 읽을 앞머리로 줄인다.

    원문은 기계용이라 필드가 길다. 패널은 폭이 좁으므로 무엇이 일어났는지를 앞에
    세우고 나머지는 뒤로 민다(잘리더라도 핵심은 남는다).
    """
    component = _COMPONENT_RE.search(text)
    message = _MESSAGE_RE.search(text)
    if not (component and message):
        return text
    rest = _MESSAGE_RE.sub("", _COMPONENT_RE.sub("", text)).strip()
    head = f"{component.group(1)} / {message.group(1)}"
    return f"{head}  {rest}".strip()


def load_log_entries(
    log_paths: list[Path], base: datetime, span_sec: float, pre_roll_sec: float = 30.0
) -> list[LogEntry]:
    """로그 파일들에서 녹화 구간(+앞 여유)에 걸치는 줄을 상대 시각으로 읽는다.

    `pre_roll_sec` 만큼 앞을 포함하는 이유: 접속(connect_tool)은 녹화 시작 **이전**
    step 이라, 녹화 시각부터 자르면 데모에서 가장 중요한 "왜 이 창이 열렸는가" 가
    통째로 빠진다.
    """
    entries: list[LogEntry] = []
    for path in log_paths:
        if not path.exists():
            print(f"[WARNING] 로그 파일 없음 - 건너뜀: {path}")
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"[WARNING] 로그 읽기 실패({path.name}): {exc}")
            continue

        kept = 0
        for line in lines:
            parsed = _parse_line_time(line.strip(), base)
            if parsed is None:
                continue
            when, level, body = parsed
            t_sec = (when - base).total_seconds()
            if t_sec < -pre_roll_sec or t_sec > span_sec:
                continue
            entries.append(LogEntry(t_sec, level or "INFO", _condense(body)))
            kept += 1
        print(f"[INFO] 로그 {path.name}: {kept}줄 채택 (전체 {len(lines)}줄)")

    entries.sort(key=lambda item: item.t_sec)
    return entries


def load_step_entries(run_dir: Path, base: datetime) -> list[LogEntry]:
    """실행 저널의 step_*.json 을 'STEP <id> <status>' 줄로 만든다.

    데모에서 가장 읽기 좋은 줄이라 감사 로그와 별개로 뽑아 섞는다.
    """
    if not run_dir.is_dir():
        print(f"[WARNING] 실행 저널 폴더 없음 - 건너뜀: {run_dir}")
        return []

    entries: list[LogEntry] = []
    for path in sorted(run_dir.glob("step_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            when = datetime.strptime(str(raw.get("timestamp", "")), "%Y-%m-%dT%H:%M:%S")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        status = str(raw.get("status", ""))
        text = f"STEP {raw.get('step_id', '?')}  [{status}]"
        error = str(raw.get("error_message") or "")
        if error:
            text += f"  {error}"
        level = "ERROR" if status == "failed" else "STEP"
        entries.append(LogEntry((when - base).total_seconds(), level, text))

    print(f"[INFO] 실행 저널 {run_dir.name}: step {len(entries)}건")
    return entries


# ------------------------------------------------------------------
# 렌더링.
# ------------------------------------------------------------------


def resolve_font(size: int) -> tuple[object, bool]:
    """한글이 되는 TrueType 폰트를 찾는다. (font, 한글가능여부) 를 반환.

    못 찾아도 패널을 끄지 않는다 - 경고 후 기본 비트맵으로 떨어뜨려, 폰트 문제가
    영상 생성 실패로 번지지 않게 한다.
    """
    override = os.environ.get("DEMO_VIDEO_FONT", "").strip()
    candidates = ([override] if override else []) + list(_FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size), True
        except (OSError, ValueError):
            continue
    print("[WARNING] 한글 TrueType 폰트를 찾지 못했습니다 - 기본 폰트 사용(한글 깨짐). "
          "DEMO_VIDEO_FONT 로 경로를 지정하세요.")
    return ImageFont.load_default(), False


class LogPanel:
    """로그 목록을 시각에 맞춰 세로 패널 이미지(BGR)로 그린다."""

    def __init__(
        self,
        entries: list[LogEntry],
        width: int,
        height: int,
        *,
        max_lines: int = 14,
        font_size: int = 17,
    ) -> None:
        self.entries = entries
        self.width = width
        self.height = height
        self.max_lines = max(3, max_lines)
        self.font, self.korean_ok = resolve_font(font_size)
        self.line_height = font_size + 8
        self._cache_key = -1
        self._cache: np.ndarray | None = None
        # 시각 목록을 미리 뽑아두면 프레임마다 이진탐색 없이 선형 커서로 훑을 수 있다.
        self._times = [entry.t_sec for entry in entries]

    def _visible_count(self, t_sec: float) -> int:
        """t_sec 까지 찍힌 줄 수 (그 시점의 콘솔 상태)."""
        count = 0
        for stamp in self._times:
            if stamp > t_sec:
                break
            count += 1
        return count

    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, max_width: int) -> list[str]:
        """패널 폭에 맞춰 줄바꿈한다 (한글은 공백이 드물어 글자 단위로도 자른다)."""
        lines: list[str] = []
        current = ""
        for char in text:
            trial = current + char
            if draw.textlength(trial, font=self.font) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
            current = char
            if len(lines) >= 3:  # 한 이벤트가 패널을 다 먹지 않게 한다.
                break
        if current:
            lines.append(current)
        return lines or [""]

    def render(self, t_sec: float) -> np.ndarray:
        """t_sec 시점의 패널을 BGR ndarray 로 만든다 (같은 상태면 캐시 반환)."""
        count = self._visible_count(t_sec)
        if count == self._cache_key and self._cache is not None:
            return self._cache

        image = Image.new("RGB", (self.width, self.height), (18, 20, 24))
        draw = ImageDraw.Draw(image)
        pad = 14
        text_width = self.width - pad * 2

        # 최신 줄이 아래에 오도록 아래에서 위로 쌓는다(터미널과 같은 방향).
        y = self.height - pad
        shown = 0
        for index in range(count - 1, -1, -1):
            if shown >= self.max_lines:
                break
            entry = self.entries[index]
            color = _LEVEL_COLORS.get(entry.level, _DEFAULT_COLOR)
            prefix = f"{entry.t_sec:+7.1f}s "
            wrapped = self._wrap(draw, prefix + entry.text, text_width)
            block_height = self.line_height * len(wrapped)
            if y - block_height < pad:
                break
            y -= block_height
            # 가장 최근 줄은 밝게 - 지금 무슨 일이 벌어지는지가 한눈에 보여야 한다.
            faded = color if index == count - 1 else tuple(int(c * 0.62) for c in color)
            for offset, chunk in enumerate(wrapped):
                draw.text(
                    (pad, y + offset * self.line_height), chunk,
                    font=self.font, fill=faded,
                )
            shown += 1

        draw.line([(0, 0), (0, self.height)], fill=(60, 66, 76), width=2)
        # PIL 은 RGB, VideoWriter 는 BGR 이라 채널을 뒤집어 돌려준다.
        self._cache = np.asarray(image)[:, :, ::-1].copy()
        self._cache_key = count
        return self._cache


__all__ = [
    "LogEntry",
    "LogPanel",
    "load_log_entries",
    "load_step_entries",
    "read_recording_start",
    "resolve_font",
]
