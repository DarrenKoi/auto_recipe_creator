"""윈도우 창 탐색 유틸리티."""

import ctypes
import os
import re
import time
from typing import Callable

from pywinauto import Desktop
from pywinauto.application import Application

from .time_utils import format_elapsed_ms

_SW_RESTORE = 9


def _normalize_window_title(title: str) -> str:
    """창 제목 비교 전 공백/제어 문자를 정리한다."""
    cleaned = (
        title.replace("\x00", " ")
        .replace("\u200b", " ")
        .replace("\ufeff", " ")
        .strip()
    )
    return " ".join(cleaned.split())


def _collect_window_title_candidates(window) -> list[tuple[str, str]]:
    """window_text 외 후보 필드까지 모아 비교용 제목 목록을 만든다."""
    raw_candidates: list[str] = []

    try:
        title = window.window_text()
    except Exception:
        title = ""
    if isinstance(title, str):
        raw_candidates.append(title)

    try:
        texts = window.texts()
    except Exception:
        texts = []
    for item in texts:
        if isinstance(item, str):
            raw_candidates.append(item)

    element_info = getattr(window, "element_info", None)
    if element_info is not None:
        for attr_name in ("name", "rich_text"):
            try:
                value = getattr(element_info, attr_name, "")
            except Exception:
                value = ""
            if isinstance(value, str):
                raw_candidates.append(value)

    candidates: list[tuple[str, str]] = []
    seen_normalized: set[str] = set()
    for raw_title in raw_candidates:
        normalized = _normalize_window_title(raw_title)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        candidates.append((raw_title, normalized))

    return candidates


def _compile_title_prefix_pattern(title_prefix: str) -> re.Pattern[str] | None:
    """title_prefix 문자열을 regex prefix 패턴으로 변환한다."""
    normalized_prefix = _normalize_window_title(title_prefix)
    if not normalized_prefix:
        return None

    pattern_text = rf"^{re.escape(normalized_prefix)}(?:\b| .*)"
    return re.compile(pattern_text, re.IGNORECASE)


def _match_title_prefix(
    window,
    title_prefix: str,
) -> tuple[bool, str, str, str]:
    """창의 여러 title 후보 중 regex prefix 와 매칭되는 값을 찾는다."""
    candidates = _collect_window_title_candidates(window)
    if not candidates:
        return False, "", "", ""

    title_pattern = _compile_title_prefix_pattern(title_prefix)
    if title_pattern is None:
        raw_title, normalized_title = candidates[0]
        return False, raw_title, normalized_title, ""

    for raw_title, normalized_title in candidates:
        if title_pattern.match(normalized_title):
            return True, raw_title, normalized_title, "regex_prefix"

    raw_title, normalized_title = candidates[0]
    return False, raw_title, normalized_title, ""


def _format_handle(handle: int | None) -> str:
    """handle 값을 사람이 읽기 쉬운 형태로 변환한다."""
    if handle is None:
        return "N/A"
    return hex(handle)


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
        print(f"[INFO] Win32 foreground 생략(non-nt): {debug_label}")
        return False

    handle = _extract_window_handle(window)
    if handle is None:
        print(f"[INFO] 창 handle 조회 실패: {debug_label}")
        return False

    try:
        user32 = ctypes.windll.user32
    except Exception as exc:
        print(f"[INFO] user32 접근 실패: {debug_label}, error={exc}")
        return False

    try:
        if user32.IsIconic(handle):
            print(
                f"[INFO] Win32 restore 시도: {debug_label}, "
                f"handle={_format_handle(handle)}"
            )
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
        print(
            f"[INFO] Win32 foreground 완료: {debug_label}, "
            f"handle={_format_handle(handle)}, "
            f"foreground_handle={_format_handle(foreground_handle)}"
        )
        return True

    print(
        f"[INFO] Win32 foreground 미확인: {debug_label}, "
        f"handle={_format_handle(handle)}, "
        f"foreground_handle={_format_handle(foreground_handle)}, "
        f"set_foreground_ok={set_foreground_ok}"
    )
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
            print(f"[INFO] 창 restore 시도: {debug_label}")
            window.restore()
            time.sleep(settle_sec)
    except Exception as exc:
        print(f"[INFO] 창 restore 실패: {debug_label}, error={exc}")

    if foreground_window(window, debug_label=debug_label, settle_sec=settle_sec):
        return True

    try:
        window.set_focus()
        time.sleep(settle_sec)
        print(f"[INFO] 창 focus 완료: {debug_label}")
        return True
    except Exception as exc:
        print(f"[INFO] 창 focus 실패: {debug_label}, error={exc}")

    try:
        rect = window.rectangle()
        rel_x = min(100, max(1, rect.right - rect.left - 2))
        rel_y = min(18, max(1, rect.bottom - rect.top - 2))
        window.click_input(coords=(rel_x, rel_y), button="left")
        time.sleep(settle_sec)
        print(f"[INFO] 창 click_input focus 폴백 성공: {debug_label}")
        return True
    except Exception as exc:
        print(f"[INFO] 창 click_input focus 폴백 실패: {debug_label}, error={exc}")
        return False


def find_window_by_pid_and_title_prefix(
    process_id: int,
    title_prefix: str,
    backends: tuple[str, ...] = ("uia", "win32"),
    *,
    connect_timeout: float = 2.0,
    window_filter: Callable[[object, str], bool] | None = None,
) -> tuple[object | None, str, str]:
    """특정 PID 에 연결해 title_prefix 와 유사 매칭되는 첫 창을 반환한다."""
    search_started_at = time.time()
    for backend in backends:
        backend_started_at = time.time()
        app = Application(backend=backend)
        print(
            "[INFO] 로그인 창 PID 연결 시도 "
            f"backend={backend}, pid={process_id}, timeout={connect_timeout}s"
        )
        try:
            app.connect(process=process_id, timeout=connect_timeout)
        except Exception as exc:
            print(
                "[INFO] 로그인 창 PID 연결 실패 "
                f"backend={backend}, pid={process_id}, error={exc}"
            )
            continue

        try:
            windows = app.windows()
        except Exception as exc:
            print(
                "[INFO] 로그인 창 PID 조회 실패 "
                f"backend={backend}, pid={process_id}, error={exc}"
            )
            continue

        print(
            "[INFO] 로그인 창 PID 조회 "
            f"backend={backend}, pid={process_id}, window_count={len(windows)}, "
            f"elapsed={format_elapsed_ms(backend_started_at)}"
        )
        for win in windows:
            is_match, title, normalized_title, match_mode = _match_title_prefix(win, title_prefix)
            if not title:
                continue

            if not is_match:
                continue

            if window_filter is not None and not window_filter(win, title):
                print(
                    "[INFO] 로그인 창 PID 후보 제외 "
                    f"backend={backend}, pid={process_id}, title={title!r}, "
                    f"normalized_title={normalized_title!r}, match_mode={match_mode}"
                )
                continue
            print(
                "[INFO] 로그인 창 PID 발견 "
                f"backend={backend}, pid={process_id}, title={title!r}, "
                f"normalized_title={normalized_title!r}, match_mode={match_mode}, "
                f"total_elapsed={format_elapsed_ms(search_started_at)}"
            )
            return win, title, backend

    print(
        "[INFO] 로그인 창 PID 미발견 "
        f"pid={process_id}, title_prefix={title_prefix!r}, "
        f"elapsed={format_elapsed_ms(search_started_at)}"
    )
    return None, "", ""


def find_window_by_title_prefix(
    title_prefix: str,
    backends: tuple[str, ...] = ("uia", "win32"),
    *,
    visible_only: bool = True,
    window_filter: Callable[[object, str], bool] | None = None,
) -> tuple[object | None, str, str]:
    """top-level 창 중 title_prefix 와 유사 매칭되는 첫 창을 반환한다."""
    search_started_at = time.time()
    for backend in backends:
        backend_started_at = time.time()
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True,
                visible_only=visible_only,
            )
        except Exception as exc:
            print(f"[INFO] 로그인 창 조회 실패: backend={backend}, error={exc}")
            continue

        print(
            "[INFO] 로그인 창 조회 "
            f"backend={backend}, visible_only={visible_only}, "
            f"window_count={len(windows)}, "
            f"elapsed={format_elapsed_ms(backend_started_at)}"
        )
        for win in windows:
            is_match, title, normalized_title, match_mode = _match_title_prefix(win, title_prefix)
            if not title:
                continue

            if not is_match:
                continue

            if window_filter is not None and not window_filter(win, title):
                print(
                    "[INFO] 로그인 창 후보 제외 "
                    f"backend={backend}, title={title!r}, "
                    f"normalized_title={normalized_title!r}, match_mode={match_mode}, "
                    f"visible_only={visible_only}"
                )
                continue
            print(
                "[INFO] 로그인 창 발견 "
                f"backend={backend}, title={title!r}, "
                f"normalized_title={normalized_title!r}, match_mode={match_mode}, "
                f"total_elapsed={format_elapsed_ms(search_started_at)}"
            )
            return win, title, backend

    print(
        "[INFO] 로그인 창 미발견 "
        f"title_prefix={title_prefix!r}, visible_only={visible_only}, "
        f"elapsed={format_elapsed_ms(search_started_at)}"
    )
    return None, "", ""
