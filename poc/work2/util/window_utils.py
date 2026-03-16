"""윈도우 창 탐색 유틸리티."""

import time

from pywinauto import Desktop
from pywinauto.application import Application

from .time_utils import format_elapsed_ms


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
    backends: tuple[str, ...] = ("win32", "uia"),
    *,
    connect_timeout: float = 2.0,
) -> tuple[object | None, str, str]:
    """특정 PID 에 연결해 title_prefix 로 시작하는 첫 창을 반환한다."""
    search_started_at = time.time()
    for backend in backends:
        backend_started_at = time.time()
        app = Application(backend=backend)
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
            try:
                title = (win.window_text() or "").strip()
            except Exception:
                continue

            if not title:
                continue

            if title.startswith(title_prefix):
                print(
                    "[INFO] 로그인 창 PID 발견 "
                    f"backend={backend}, pid={process_id}, title={title!r}, "
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
    backends: tuple[str, ...] = ("win32", "uia"),
    *,
    visible_only: bool = True,
) -> tuple[object | None, str, str]:
    """top-level 창 중 title_prefix 로 시작하는 첫 창을 반환한다."""
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
            try:
                title = (win.window_text() or "").strip()
            except Exception:
                continue

            if not title:
                continue

            if title.startswith(title_prefix):
                print(
                    "[INFO] 로그인 창 발견 "
                    f"backend={backend}, title={title!r}, "
                    f"total_elapsed={format_elapsed_ms(search_started_at)}"
                )
                return win, title, backend

    print(
        "[INFO] 로그인 창 미발견 "
        f"title_prefix={title_prefix!r}, visible_only={visible_only}, "
        f"elapsed={format_elapsed_ms(search_started_at)}"
    )
    return None, "", ""
