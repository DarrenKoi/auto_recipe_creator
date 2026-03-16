"""현재 열려 있는 창 제목 목록을 빠르게 출력하는 테스트 스크립트.

사용법:
  uv run python window_titles.py

선택 환경변수:
  WINDOW_TITLES_VISIBLE_ONLY=true
"""

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_bool_env(env_name: str, default: bool) -> bool:
    """환경변수 bool 값을 읽는다."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class WindowTitlesSettings:
    """창 제목 조회 설정."""

    visible_only: bool = _parse_bool_env("WINDOW_TITLES_VISIBLE_ONLY", True)


@dataclass(frozen=True)
class WindowRow:
    """출력용 창 정보."""

    title: str
    handle: int


def _normalize_window_title(title: str) -> str:
    """창 제목 비교 전 공백/제어 문자를 정리한다."""
    cleaned = (
        title.replace("\x00", " ")
        .replace("\u200b", " ")
        .replace("\ufeff", " ")
        .strip()
    )
    return " ".join(cleaned.split())


def _format_handle(handle: int | None) -> str:
    """handle 값을 사람이 읽기 쉬운 형태로 변환한다."""
    if handle is None:
        return "N/A"
    return hex(handle)


_TITLE_BUF_SIZE = 512
"""고정 버퍼 크기. GetWindowTextLengthW 호출을 생략하기 위해 사용."""


def _read_window_text(user32, hwnd: int, buffer=None) -> str:
    """Win32 API 로 창 제목을 읽는다.

    buffer 를 미리 할당해 전달하면 재할당 비용을 줄인다.
    """
    if buffer is None:
        buffer = ctypes.create_unicode_buffer(_TITLE_BUF_SIZE)

    copied = int(user32.GetWindowTextW(hwnd, buffer, _TITLE_BUF_SIZE))
    if copied <= 0:
        return ""

    return _normalize_window_title(buffer.value)


def _read_foreground_window_handle(user32) -> int | None:
    """현재 foreground 창 handle 을 읽는다."""
    handle = int(user32.GetForegroundWindow())
    return handle or None


def _collect_window_rows(*, visible_only: bool) -> list[WindowRow]:
    """현재 top-level 창 제목 목록을 Win32 API 로 빠르게 수집한다."""
    user32 = ctypes.windll.user32
    rows: list[WindowRow] = []
    seen_handles: set[int] = set()
    buf = ctypes.create_unicode_buffer(_TITLE_BUF_SIZE)

    enum_windows_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    @enum_windows_proc
    def _enum_callback(hwnd, _lparam):
        handle = int(hwnd)
        if handle <= 0 or handle in seen_handles:
            return True

        if visible_only and not user32.IsWindowVisible(hwnd):
            return True

        copied = int(user32.GetWindowTextW(hwnd, buf, _TITLE_BUF_SIZE))
        if copied <= 0:
            return True

        title = _normalize_window_title(buf.value)
        if not title:
            return True

        seen_handles.add(handle)
        rows.append(WindowRow(title=title, handle=handle))
        return True

    result = user32.EnumWindows(_enum_callback, 0)
    if result == 0:
        raise ctypes.WinError()

    return rows


_SW_RESTORE = 9
_CUBEMAIN_KEYWORD = "CubeMain"


def _focus_window(user32, hwnd: int) -> bool:
    """창을 foreground 로 활성화한다.

    최소화 상태면 복원 후 SetForegroundWindow 를 호출한다.
    """
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)

    return bool(user32.SetForegroundWindow(hwnd))


def _find_and_focus_cubemain(user32, rows: list[WindowRow]) -> bool:
    """rows 에서 CubeMain 창을 찾아 foreground 로 올린다."""
    for row in rows:
        if _CUBEMAIN_KEYWORD in row.title:
            print(
                f"[INFO] CubeMain 창 발견: title={row.title!r}, "
                f"handle={_format_handle(row.handle)}"
            )
            ok = _focus_window(user32, row.handle)
            if ok:
                print("[INFO] CubeMain 창을 foreground 로 활성화했습니다.")
            else:
                print("[WARNING] SetForegroundWindow 실패 — 권한 또는 포커스 제한.")
            return ok

    print(f"[INFO] '{_CUBEMAIN_KEYWORD}' 창을 찾지 못했습니다.")
    return False


def _print_report(
    *,
    visible_only: bool,
    rows: list[WindowRow],
    foreground_handle: int | None,
    foreground_title: str,
) -> None:
    """조회 결과를 콘솔에 출력한다."""
    print(
        f"[INFO] visible_only={visible_only}, titled_window_count={len(rows)}"
    )

    if foreground_handle is not None or foreground_title:
        print(
            "[INFO] foreground_window "
            f"handle={_format_handle(foreground_handle)}, title={foreground_title!r}"
        )
    else:
        print("[WARNING] foreground 창 제목을 읽지 못했습니다.")

    if not rows:
        print("[WARNING] 제목이 있는 top-level 창을 찾지 못했습니다.")
        return

    print("[INFO] current window titles:")
    for index, row in enumerate(rows, start=1):
        marker = "*" if foreground_handle == row.handle else " "
        print(
            f"[INFO] [{index:02d}] {marker} {row.title} "
            f"(handle={_format_handle(row.handle)})"
        )


def main() -> int:
    """현재 창 제목 목록을 출력한다."""
    if os.name != "nt":
        print(f"[WARNING] 이 스크립트는 Windows 전용입니다: platform={sys.platform}")
        return 1

    settings = WindowTitlesSettings()
    print(f"[INFO] 창 제목 조회 시작: visible_only={settings.visible_only}")

    try:
        user32 = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(_TITLE_BUF_SIZE)
        foreground_handle = _read_foreground_window_handle(user32)
        foreground_title = ""
        if foreground_handle is not None:
            foreground_title = _read_window_text(user32, foreground_handle, buf)
        rows = _collect_window_rows(visible_only=settings.visible_only)
    except Exception as exc:
        print(f"[ERROR] 창 제목 조회 실패: error={exc}")
        return 1

    _find_and_focus_cubemain(user32, rows)

    _print_report(
        visible_only=settings.visible_only,
        rows=rows,
        foreground_handle=foreground_handle,
        foreground_title=foreground_title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
