"""윈도우 창 탐색 유틸리티."""

import time

from pywinauto import Desktop

from .time_utils import format_elapsed_ms


def find_window_by_title_prefix(
    title_prefix: str,
    backends: tuple[str, ...] = ("win32", "uia"),
) -> tuple[object | None, str, str]:
    """보이는 top-level 창 중 title_prefix 로 시작하는 첫 창을 반환한다."""
    search_started_at = time.time()
    for backend in backends:
        backend_started_at = time.time()
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True,
                visible_only=True,
            )
        except Exception as exc:
            print(f"[INFO] 로그인 창 조회 실패: backend={backend}, error={exc}")
            continue

        print(
            "[INFO] 로그인 창 조회 "
            f"backend={backend}, window_count={len(windows)}, "
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
        f"title_prefix={title_prefix!r}, elapsed={format_elapsed_ms(search_started_at)}"
    )
    return None, "", ""
