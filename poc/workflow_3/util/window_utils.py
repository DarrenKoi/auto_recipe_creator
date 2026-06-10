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
_SW_MAXIMIZE = 3
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


def window_rect_size(window) -> tuple[int, int] | None:
    """창 rect 의 (width, height) 논리 px. 조회 실패 시 None.

    캡처 시점과 클릭 시점 사이의 창 크기 드리프트(사용자 리사이즈) 감지용.
    위치 이동은 image_point_to_screen 이 live rect 로 흡수하지만, 크기 변화는
    내용물 reflow 라 좌표 보정이 불가능해 호출부가 중단/재캡처해야 한다.
    """
    try:
        rect = window.rectangle()
    except Exception:
        return None
    return (rect.right - rect.left, rect.bottom - rect.top)


def is_window_maximized(window) -> bool:
    """창이 최대화(maximize) 상태인지 확인한다 (Win32 IsZoomed)."""
    if os.name != "nt":
        return False
    handle = _extract_window_handle(window)
    if handle is None:
        return False
    try:
        return bool(ctypes.windll.user32.IsZoomed(handle))
    except Exception:
        return False


def maximize_window(
    window,
    *,
    debug_label: str = "window",
    settle_sec: float = 0.3,
) -> bool:
    """창을 최대화한다 (ShowWindow SW_MAXIMIZE, 실패 시 pywinauto fallback)."""
    if os.name != "nt":
        print(f"[INFO] 창 최대화 미지원 OS: {debug_label}")
        return False

    handle = _extract_window_handle(window)
    if handle is None:
        return False

    try:
        ctypes.windll.user32.ShowWindow(handle, _SW_MAXIMIZE)
        time.sleep(settle_sec)
    except Exception as exc:
        print(f"[INFO] Win32 창 최대화 실패: {debug_label}, error={exc}")

    if is_window_maximized(window):
        print(f"[INFO] 창 최대화 완료: {debug_label}")
        return True

    try:
        window.maximize()
        time.sleep(settle_sec)
    except Exception as exc:
        print(f"[INFO] 창 최대화 미확인: {debug_label}, error={exc}")
        return False

    maximized = is_window_maximized(window)
    if maximized:
        print(f"[INFO] 창 최대화 완료(pywinauto): {debug_label}")
    else:
        print(f"[INFO] 창 최대화 미확인(pywinauto): {debug_label}")
    return maximized


def _window_alive(handle: int | None) -> bool:
    """Win32 IsWindow 로 창이 아직 살아있는지 확인한다."""
    if os.name != "nt" or handle is None:
        return False
    try:
        return bool(ctypes.windll.user32.IsWindow(handle))
    except Exception:
        return False


def close_window(
    window,
    *,
    debug_label: str = "window",
    settle_sec: float = 0.5,
    try_red_x: bool = True,
    red_x_offset: tuple[int, int] = (-18, 18),
) -> bool:
    """창을 닫는다. 전략 사다리로 시도하고 매 단계 닫힘 여부를 검증한다.

    RCS 레거시 창은 window 객체 메서드(WM_CLOSE 계열)를 무시할 수 있어, 마지막에
    우상단 빨간 X 버튼을 직접 클릭하는 GUI 폴백까지 둔다.

    1) pywinauto ``close()``
    2) WM_CLOSE PostMessage
    3) WM_SYSCOMMAND/SC_CLOSE (타이틀바 우클릭→Close 의 프로그램적 등가)
    4) 우상단 X 버튼 click_input (창 기준 상대 좌표; ``red_x_offset`` 로 보정)

    ``red_x_offset`` = (오른쪽 모서리 기준 dx, 상단 기준 dy). 기본 (-18, 18) 은
    표준 close 버튼 위치 근사값이며, RCS 스킨에 맞춰 호출부에서 보정한다.
    """
    handle = _extract_window_handle(window)
    verifiable = handle is not None and os.name == "nt"

    def _closed_after_attempt() -> bool:
        # 핸들이 없으면 닫힘을 검증할 수 없다 → 성공으로 단정하지 않고(False) 다음
        # 전략(특히 GUI X 버튼 클릭)까지 모두 시도하게 한다. 잘못 닫힌 창을 닫혔다고
        # 보고해 다음 fail 과 경합시키는 일을 막는다.
        if not verifiable:
            return False
        time.sleep(settle_sec)
        return not _window_alive(handle)

    # 1) pywinauto close().
    try:
        window.close()
        if _closed_after_attempt():
            print(f"[INFO] 창 닫기 완료: {debug_label}, strategy=close")
            return True
        print(f"[INFO] close() 후에도 창 생존 — 다음 전략: {debug_label}")
    except Exception as exc:
        print(f"[INFO] pywinauto close 실패: {debug_label}, error={exc}")

    if os.name == "nt" and handle is not None:
        user32 = ctypes.windll.user32
        # 2) WM_CLOSE.
        try:
            user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE.
            if _closed_after_attempt():
                print(f"[INFO] 창 닫기 완료: {debug_label}, strategy=WM_CLOSE")
                return True
        except Exception as exc:
            print(f"[INFO] WM_CLOSE 실패: {debug_label}, error={exc}")
        # 3) WM_SYSCOMMAND / SC_CLOSE.
        try:
            user32.PostMessageW(handle, 0x0112, 0xF060, 0)  # WM_SYSCOMMAND, SC_CLOSE.
            if _closed_after_attempt():
                print(f"[INFO] 창 닫기 완료: {debug_label}, strategy=SC_CLOSE")
                return True
        except Exception as exc:
            print(f"[INFO] SC_CLOSE 실패: {debug_label}, error={exc}")

    # 4) GUI 폴백 — 우상단 X 버튼 클릭.
    if try_red_x:
        try:
            rect = window.rectangle()
            rel_x = int((rect.right - rect.left) + red_x_offset[0])
            rel_y = int(red_x_offset[1])
            window.click_input(coords=(rel_x, rel_y))
            if _closed_after_attempt():
                print(
                    f"[INFO] 창 닫기 완료: {debug_label}, strategy=red_x "
                    f"coords=({rel_x},{rel_y})"
                )
                return True
            print(f"[INFO] X 버튼 클릭 후에도 창 생존: {debug_label}")
        except Exception as exc:
            print(f"[INFO] X 버튼 클릭 실패: {debug_label}, error={exc}")

    if not verifiable:
        print(
            f"[WARNING] 창 닫힘을 검증할 수 없습니다(handle 없음) — 모든 전략을 시도했으나 "
            f"결과 미확인: {debug_label}"
        )
    else:
        print(f"[WARNING] 창 닫기 모든 전략 실패: {debug_label}")
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


def image_point_to_screen(
    window,
    image_point: dict,
    image_size: tuple[int, int] | None = None,
) -> dict[str, int] | None:
    """윈도우 이미지 좌표를 스크린 절대 좌표로 변환한다.

    캡처 이미지(mss, 물리 픽셀)와 window.rectangle()(DPI 배율 적용 시 논리 픽셀)의
    크기가 다를 수 있다. image_size(=캡처 이미지 크기)가 주어지면 rect/image 비율로
    보정해 DPI 배율 화면에서도 정확한 스크린 좌표를 만든다. 크기가 같으면 배율 1.0
    이라 기존 동작과 동일하다(100% 배율 무영향).
    """
    try:
        rect = window.rectangle()
    except Exception as exc:
        print(f"[ERROR] 창 rectangle 조회 실패: {exc}")
        return None

    rect_w = rect.right - rect.left
    rect_h = rect.bottom - rect.top
    scale_x = 1.0
    scale_y = 1.0
    if image_size is not None:
        img_w, img_h = image_size
        if img_w > 0 and img_h > 0:
            scale_x = rect_w / img_w
            scale_y = rect_h / img_h

    screen_x = int(round(rect.left + image_point["x"] * scale_x))
    screen_y = int(round(rect.top + image_point["y"] * scale_y))

    print(
        f"[INFO] image→screen: rect=({rect.left},{rect.top},{rect.right},{rect.bottom}) "
        f"{rect_w}x{rect_h}, image_size={image_size}, scale=({scale_x:.3f},{scale_y:.3f}), "
        f"image_point=({image_point['x']},{image_point['y']}) → screen=({screen_x},{screen_y})"
    )
    return {"x": screen_x, "y": screen_y}
