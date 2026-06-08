"""align 이미지에 딸린 ``cond.txt`` 조건 파일을 읽어 좌표를 뽑아내는 파서.

오피스 다운로더가 각 이미지 옆에 숨김 폴더로 cond.txt 를 떨군다:
``<image>.jpeg`` → ``.<image>.jpeg/cond.txt`` (파일명 그대로, 앞에 점).

cond.txt 한 줄 형식: ``key  값,값,...`` (key 와 값 사이는 공백/탭, 값끼리는 콤마).
우리가 쓰는 키 ([[project_align_cond_files_and_coords]]):
  - ``Scope``        : OM / SEM (modality — fail 멈춘 step 의 종류)
  - ``Pixel``        : 이미지 크기 (예: 512,512 / 1024,1024)
  - ``!Cursor_info`` : crosshair / white box 좌표가 한 줄에 들어 있다.
        elements[4],[5]      = crosshair (cx, cy)        — 둘 다 -1 이 아니면 존재
        elements[6],[7],[8],[9] = white box (left, top, right, bottom)
                              — [8],[9] 가 -1 이 아니면 존재
    cursor 좌표는 Pixel 의 10배 oversample 프레임이다(이미지 px = cursor/10).
    실제 이미지 위 좌표 변환은 본 파서가 아니라 그리기/inpaint 단계에서 적용한다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# !Cursor_info 요소 인덱스 (0-base). 좌표는 cursor oversample 프레임 기준 raw 값.
_CROSSHAIR_IDX = (4, 5)
_BOX_IDX = (6, 7, 8, 9)


@dataclass(frozen=True)
class CondInfo:
    """cond.txt 한 장에서 뽑은 조건 (없는 항목은 None)."""

    scope: str | None = None                       # "OM" / "SEM" (원문 토큰)
    pixel: tuple[int, int] | None = None           # (width, height)
    box_ltrb: tuple[int, int, int, int] | None = None   # cursor 프레임 raw 좌표
    crosshair_xy: tuple[int, int] | None = None         # cursor 프레임 raw 좌표
    raw: dict[str, list[str]] = field(default_factory=dict)  # key → 값 토큰 (디버그용)

    @property
    def is_sem(self) -> bool:
        return bool(self.scope) and "SEM" in self.scope.upper()

    @property
    def is_om(self) -> bool:
        return bool(self.scope) and "OM" in self.scope.upper()


def _norm_key(key: str) -> str:
    """비교용 키 정규화: 앞의 '!' 제거 + 소문자."""
    return key.lstrip("!").strip().lower()


def _to_int(token: str) -> int | None:
    """토큰을 int 로. 실패하면 None."""
    try:
        return int(token.strip())
    except (ValueError, AttributeError):
        return None


def _present(tokens: list[str], idx: tuple[int, ...]) -> bool:
    """주어진 인덱스 값들이 모두 존재하고 -1 이 아니면 True."""
    if max(idx) >= len(tokens):
        return False
    return all(_to_int(tokens[i]) not in (None, -1) for i in idx)


def parse_cond(text: str) -> CondInfo:
    """cond.txt 본문 문자열을 CondInfo 로 파싱한다."""
    raw: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # key 는 첫 공백/탭 이전 토큰, 값은 나머지를 콤마로 분해.
        parts = line.split(None, 1)
        if not parts:
            continue
        key = parts[0]
        value_str = parts[1].strip() if len(parts) > 1 else ""
        raw[_norm_key(key)] = [t.strip() for t in value_str.split(",")] if value_str else []

    scope = raw.get("scope", [None])[0]

    pixel = None
    px = raw.get("pixel", [])
    if len(px) >= 2 and _to_int(px[0]) is not None and _to_int(px[1]) is not None:
        pixel = (_to_int(px[0]), _to_int(px[1]))

    # 실데이터 키는 "!Cursor_inf"(끝 o 없음)·"!Cursor_info" 등 흔들린다 → 접두 매칭.
    cur = next((v for k, v in raw.items() if k.startswith("cursor_inf")), [])
    box_ltrb = None
    if _present(cur, _BOX_IDX):
        box_ltrb = tuple(_to_int(cur[i]) for i in _BOX_IDX)
    crosshair_xy = None
    if _present(cur, _CROSSHAIR_IDX):
        crosshair_xy = tuple(_to_int(cur[i]) for i in _CROSSHAIR_IDX)

    return CondInfo(
        scope=scope,
        pixel=pixel,
        box_ltrb=box_ltrb,
        crosshair_xy=crosshair_xy,
        raw=raw,
    )


def cond_path_for(image_path) -> Path:
    """이미지 경로 → 짝이 되는 cond.txt 경로 (.<파일명>/cond.txt)."""
    image_path = Path(image_path)
    return image_path.parent / f".{image_path.name}" / "cond.txt"


def load_cond(image_path) -> CondInfo | None:
    """이미지에 딸린 cond.txt 를 읽어 파싱한다. 없으면 None."""
    path = cond_path_for(image_path)
    if not path.is_file():
        return None
    return parse_cond(path.read_text(encoding="utf-8", errors="replace"))
