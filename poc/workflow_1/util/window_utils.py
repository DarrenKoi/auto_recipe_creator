"""윈도우 창 탐색 유틸리티."""

import ctypes
import os
import re
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

from pywinauto import Desktop

from .time_utils import format_elapsed_ms

_SW_RESTORE = 9
_TITLE_BUF_SIZE = 512


@dataclass(frozen=True)
class WindowRow:
    """top-level 창 스캔 결과."""

    title: str
    handle: int
    process_id: int


def _normalize_window_title(title: str) -> str:
    """창 제목 비교 전 공백/제어 문자를 정리한다."""
    cleaned = (
        title.replace("\x00", " ")
        .replace("\u200b", " ")
        .replace("\ufeff", " ")
        .strip()
    )
    return " ".join(cleaned.split())


def _compile_title_prefix_pattern(title_prefix: str) -> re.Pattern[str] | None:
    """title_prefix 문자열을 regex prefix 패턴으로 변환한다."""
    normalized_prefix = _normalize_window_title(title_prefix)
    if not normalized_prefix:
        return None

    pattern_text = rf"^{re.escape(normalized_prefix)}(?:\b| .*)"
    return re.compile(pattern_text, re.IGNORECASE)


def _format_handle(handle: int | None) -> str:
    """handle 값을 사람이 읽기 쉬운 형태로 변환한다."""
    if handle is None:
        return "N/A"
    return hex(handle)


def _read_window_text(user32, hwnd: int, buffer=None) -> str:
    """Win32 API 로 창 제목을 읽는다."""
    if buffer is None:
        buffer = ctypes.create_unicode_buffer(_TITLE_BUF_SIZE)

    copied = int(user32.GetWindowTextW(hwnd, buffer, _TITLE_BUF_SIZE))
    if copied <= 0:
        return ""

    return _normalize_window_title(buffer.value)


def _read_window_process_id(user32, hwnd: int) -> int:
    """Win32 API 로 창의 process id 를 읽는다."""
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return int(process_id.value)


def read_foreground_window_info() -> tuple[int | None, str]:
    """현재 foreground 창의 handle 과 title 을 반환한다."""
    if os.name != "nt":
        return None, ""

    user32 = ctypes.windll.user32
    foreground_handle = int(user32.GetForegroundWindow()) or None
    if foreground_handle is None:
        return None, ""

    buf = ctypes.create_unicode_buffer(_TITLE_BUF_SIZE)
    return foreground_handle, _read_window_text(user32, foreground_handle, buf)


def collect_window_rows(
    *,
    visible_only: bool = True,
    process_id: int | None = None,
) -> list[WindowRow]:
    """현재 top-level 창 제목 목록을 Win32 API 로 빠르게 수집한다."""
    if os.name != "nt":
        return []

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

        window_process_id = _read_window_process_id(user32, hwnd)
        if process_id is not None and process_id > 0 and window_process_id != process_id:
            return True

        title = _read_window_text(user32, hwnd, buf)
        if not title:
            return True

        seen_handles.add(handle)
        rows.append(
            WindowRow(
                title=title,
                handle=handle,
                process_id=window_process_id,
            )
        )
        return True

    result = user32.EnumWindows(_enum_callback, 0)
    if result == 0:
        raise ctypes.WinError()

    return rows


def _extract_window_handle(window) -> int | None:
    """pywinauto wrapper 에서 Win32 handle 을 최대한 안전하게 꺼낸다."""
    for attr_name in ("handle", "hwnd"):
        try:
            value = getattr(window, attr_name, None)
        except Exception:
            value = None
        if isinstance(value, int) and value > 0:
            return value

    element_info = getattr(window, "element_info", None)
    if element_info is not None:
        try:
            value = getattr(element_info, "handle", None)
        except Exception:
            value = None
        if isinstance(value, int) and value > 0:
            return value

    try:
        wrapper = window.wrapper_object()
    except Exception:
        wrapper = None

    if wrapper is not None and wrapper is not window:
        for attr_name in ("handle", "hwnd"):
            try:
                value = getattr(wrapper, attr_name, None)
            except Exception:
                value = None
            if isinstance(value, int) and value > 0:
                return value

    return None


def get_window_process_id(window) -> int | None:
    """pywinauto wrapper 에서 process id 를 최대한 안전하게 꺼낸다."""
    for attr_name in ("process_id",):
        try:
            value = getattr(window, attr_name, None)
        except Exception:
            value = None

        if callable(value):
            try:
                value = value()
            except Exception:
                value = None

        if isinstance(value, int) and value > 0:
            return value

    element_info = getattr(window, "element_info", None)
    if element_info is not None:
        try:
            value = getattr(element_info, "process_id", None)
        except Exception:
            value = None
        if isinstance(value, int) and value > 0:
            return value

    try:
        wrapper = window.wrapper_object()
    except Exception:
        wrapper = None

    if wrapper is not None and wrapper is not window:
        try:
            value = getattr(wrapper, "process_id", None)
        except Exception:
            value = None

        if callable(value):
            try:
                value = value()
            except Exception:
                value = None

        if isinstance(value, int) and value > 0:
            return value

    return None


def foreground_window(
    window,
    *,
    debug_label: str = "window",
    settle_sec: float = 0.15,
) -> bool:
    """Win32 API 로 창을 foreground 로 올린다."""
    if os.name != "nt":
        return False

    handle = _extract_window_handle(window)
    if handle is None:
        return False

    try:
        user32 = ctypes.windll.user32
    except Exception as exc:
        print(f"[INFO] user32 접근 실패: {debug_label}, error={exc}")
        return False

    try:
        if user32.IsIconic(handle):
            user32.ShowWindow(handle, _SW_RESTORE)
            time.sleep(settle_sec)

        set_foreground_ok = bool(user32.SetForegroundWindow(handle))
        time.sleep(settle_sec)
        foreground_handle = int(user32.GetForegroundWindow()) or None
    except Exception as exc:
        print(f"[INFO] Win32 foreground 실패: {debug_label}, error={exc}")
        return False

    is_foreground = foreground_handle == handle
    if is_foreground:
        return True

    print(f"[INFO] Win32 foreground 미확인: {debug_label}, set_foreground_ok={set_foreground_ok}")
    return False


def activate_window(
    window,
    *,
    debug_label: str = "window",
    settle_sec: float = 0.15,
) -> bool:
    """창을 restore/focus 해서 캡처 가능한 상태로 만든다."""
    try:
        if hasattr(window, "is_minimized") and window.is_minimized():
            window.restore()
            time.sleep(settle_sec)
    except Exception:
        pass

    if foreground_window(window, debug_label=debug_label, settle_sec=settle_sec):
        print(f"[INFO] 창 활성화 완료: {debug_label}, strategy=foreground")
        return True

    try:
        window.set_focus()
        time.sleep(settle_sec)
        print(f"[INFO] 창 활성화 완료: {debug_label}, strategy=set_focus")
        return True
    except Exception:
        pass

    try:
        rect = window.rectangle()
        rel_x = min(100, max(1, rect.right - rect.left - 2))
        rel_y = min(18, max(1, rect.bottom - rect.top - 2))
        window.click_input(coords=(rel_x, rel_y), button="left")
        time.sleep(settle_sec)
        print(f"[INFO] 창 활성화 완료: {debug_label}, strategy=click_input")
        return True
    except Exception:
        print(f"[INFO] 창 활성화 실패: {debug_label}")
        return False


def _wrap_window_handle(handle: int, backend: str, desktops: dict[str, object]):
    """Win32 handle 을 pywinauto wrapper 로 변환한다."""
    desktop = desktops.get(backend)
    if desktop is None:
        desktop = Desktop(backend=backend)
        desktops[backend] = desktop

    window_spec = desktop.window(handle=handle)
    return window_spec.wrapper_object()


def _find_window_from_rows(
    rows: list[WindowRow],
    title_prefix: str,
    backends: tuple[str, ...],
    *,
    search_started_at: float,
    search_label: str,
    window_filter: Callable[[object, str], bool] | None = None,
    extra_log: str = "",
) -> tuple[object | None, str, str]:
    """raw Win32 rows 중 title prefix 에 맞는 창을 wrapper 로 반환한다."""
    title_pattern = _compile_title_prefix_pattern(title_prefix)
    if title_pattern is None:
        print(f"[INFO] {search_label} 생략: 빈 title_prefix")
        return None, "", ""

    matched_rows = [row for row in rows if title_pattern.match(row.title)]
    print(
        f"[INFO] {search_label} "
        f"title_prefix={title_prefix!r}, "
        f"window_count={len(rows)}, matched_count={len(matched_rows)}, "
        f"elapsed={format_elapsed_ms(search_started_at)}{extra_log}"
    )
    if not matched_rows:
        return None, "", ""

    desktops: dict[str, object] = {}
    for row in matched_rows:
        for backend in backends:
            try:
                window = _wrap_window_handle(row.handle, backend, desktops)
            except Exception as exc:
                print(
                    f"[INFO] {search_label} wrapper 생성 실패 "
                    f"backend={backend}, title={row.title!r}, "
                    f"handle={_format_handle(row.handle)}, error={exc}"
                )
                continue

            if window_filter is not None and not window_filter(window, row.title):
                print(
                    f"[INFO] {search_label} 후보 제외 "
                    f"backend={backend}, title={row.title!r}, "
                    f"handle={_format_handle(row.handle)}, pid={row.process_id}"
                )
                continue

            print(
                f"[INFO] {search_label} 발견 "
                f"backend={backend}, title={row.title!r}, "
                f"handle={_format_handle(row.handle)}, pid={row.process_id}, "
                f"total_elapsed={format_elapsed_ms(search_started_at)}"
            )
            return window, row.title, backend

    print(
        f"[INFO] {search_label} wrapper 변환 실패 "
        f"title_prefix={title_prefix!r}, matched_count={len(matched_rows)}, "
        f"elapsed={format_elapsed_ms(search_started_at)}"
    )
    return None, "", ""


def find_window_by_pid_and_title_prefix(
    process_id: int,
    title_prefix: str,
    backends: tuple[str, ...] = ("uia", "win32"),
    *,
    connect_timeout: float = 2.0,
    window_filter: Callable[[object, str], bool] | None = None,
) -> tuple[object | None, str, str]:
    """특정 PID 의 top-level 창 중 title_prefix 와 유사 매칭되는 첫 창을 반환한다."""
    del connect_timeout

    search_started_at = time.time()
    try:
        rows = collect_window_rows(visible_only=False, process_id=process_id)
    except Exception as exc:
        print(
            "[INFO] 로그인 창 PID raw 조회 실패 "
            f"pid={process_id}, error={exc}"
        )
        return None, "", ""

    return _find_window_from_rows(
        rows,
        title_prefix,
        backends,
        search_started_at=search_started_at,
        search_label="로그인 창 PID raw 조회",
        window_filter=window_filter,
        extra_log=f", pid={process_id}",
    )


def find_window_by_title_prefix(
    title_prefix: str,
    backends: tuple[str, ...] = ("uia", "win32"),
    *,
    visible_only: bool = True,
    window_filter: Callable[[object, str], bool] | None = None,
) -> tuple[object | None, str, str]:
    """top-level 창 중 title_prefix 와 유사 매칭되는 첫 창을 반환한다."""
    search_started_at = time.time()
    try:
        rows = collect_window_rows(visible_only=visible_only)
    except Exception as exc:
        print(
            "[INFO] 로그인 창 raw 조회 실패 "
            f"visible_only={visible_only}, error={exc}"
        )
        return None, "", ""

    return _find_window_from_rows(
        rows,
        title_prefix,
        backends,
        search_started_at=search_started_at,
        search_label="로그인 창 raw 조회",
        window_filter=window_filter,
        extra_log=f", visible_only={visible_only}",
    )


def image_point_to_screen(window, image_point: dict) -> dict[str, int] | None:
    """윈도우 이미지 좌표를 스크린 절대 좌표로 변환한다."""
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[ERROR] 창 rectangle 조회 실패: {exc}")
        return None

    return {
        "x": rect.left + image_point["x"],
        "y": rect.top + image_point["y"],
    }
