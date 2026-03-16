"""윈도우 창 탐색 유틸리티."""

import time
from typing import Callable

from pywinauto import Desktop
from pywinauto.application import Application

from .time_utils import format_elapsed_ms


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


def _tokens_appear_in_order(text: str, tokens: list[str]) -> bool:
    """토큰들이 순서대로 등장하는 느슨한 fallback 매칭."""
    if not tokens:
        return False

    search_pos = 0
    for token in tokens:
        index = text.find(token, search_pos)
        if index < 0:
            return False
        search_pos = index + len(token)
    return True


def _is_title_boundary(text: str, index: int) -> bool:
    """부분 문자열 매칭 시 시작 위치가 단어 중간이 아닌지 확인한다."""
    if index <= 0:
        return True
    return not text[index - 1].isalnum()


def _match_title_prefix(
    window,
    title_prefix: str,
) -> tuple[bool, str, str, str]:
    """창의 여러 title 후보 중 prefix 와 매칭되는 값을 찾는다."""
    candidates = _collect_window_title_candidates(window)
    if not candidates:
        return False, "", "", ""

    normalized_prefix = _normalize_window_title(title_prefix)
    if not normalized_prefix:
        raw_title, normalized_title = candidates[0]
        return False, raw_title, normalized_title, ""

    lowered_prefix = normalized_prefix.casefold()
    prefix_tokens = [token for token in lowered_prefix.split(" ") if token]
    best_fallback: tuple[str, str, str] | None = None

    for raw_title, normalized_title in candidates:
        lowered_title = normalized_title.casefold()
        if lowered_title.startswith(lowered_prefix):
            return True, raw_title, normalized_title, "startswith"

        index = lowered_title.find(lowered_prefix)
        if index >= 0:
            end_index = index + len(lowered_prefix)
            if _is_title_boundary(lowered_title, index) and (
                end_index >= len(lowered_title)
                or not lowered_title[end_index].isalnum()
            ):
                return True, raw_title, normalized_title, "boundary_contains"

        if best_fallback is None and _tokens_appear_in_order(lowered_title, prefix_tokens):
            best_fallback = (raw_title, normalized_title, "token_order")

    if best_fallback is not None:
        raw_title, normalized_title, match_mode = best_fallback
        return True, raw_title, normalized_title, match_mode

    raw_title, normalized_title = candidates[0]
    return False, raw_title, normalized_title, ""


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
