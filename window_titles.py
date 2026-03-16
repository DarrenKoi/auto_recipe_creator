"""현재 열려 있는 상위 창 제목을 빠르게 확인하는 테스트 스크립트.

사용법:
  uv run python window_titles.py

선택 환경변수:
  WINDOW_TITLES_BACKENDS=uia,win32
  WINDOW_TITLES_VISIBLE_ONLY=true
"""

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    from pywinauto import Desktop

    PYWINAUTO_AVAILABLE = True
    PYWINAUTO_IMPORT_ERROR = None
except ImportError as exc:
    Desktop = None
    PYWINAUTO_AVAILABLE = False
    PYWINAUTO_IMPORT_ERROR = exc

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


def _parse_backends_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """백엔드 우선순위 문자열을 tuple 로 정리한다."""
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default

    backends: list[str] = []
    seen: set[str] = set()
    for item in raw_value.split(","):
        backend = item.strip().lower()
        if not backend or backend in seen:
            continue
        seen.add(backend)
        backends.append(backend)

    return tuple(backends) or default


DEFAULT_BACKENDS = _parse_backends_env("WINDOW_TITLES_BACKENDS", ("uia", "win32"))
DEFAULT_VISIBLE_ONLY = _parse_bool_env("WINDOW_TITLES_VISIBLE_ONLY", True)


@dataclass(frozen=True)
class WindowTitlesSettings:
    """창 제목 조회 설정."""

    backends: tuple[str, ...] = DEFAULT_BACKENDS
    visible_only: bool = DEFAULT_VISIBLE_ONLY


@dataclass(frozen=True)
class WindowRow:
    """출력용 창 정보."""

    title: str
    handle: int | None


def _normalize_window_title(title: str) -> str:
    """창 제목 비교 전 공백/제어 문자를 정리한다."""
    cleaned = (
        title.replace("\x00", " ")
        .replace("\u200b", " ")
        .replace("\ufeff", " ")
        .strip()
    )
    return " ".join(cleaned.split())


def _collect_window_title_candidates(window) -> list[str]:
    """window_text 외 후보 필드까지 모아 제목 후보를 만든다."""
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

    candidates: list[str] = []
    seen: set[str] = set()
    for raw_title in raw_candidates:
        normalized = _normalize_window_title(raw_title)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)

    return candidates


def _read_best_window_title(window) -> str:
    """창 객체에서 가장 적절한 제목을 뽑는다."""
    candidates = _collect_window_title_candidates(window)
    if not candidates:
        return ""
    return candidates[0]


def _read_window_handle(window) -> int | None:
    """창 handle 값을 안전하게 읽는다."""
    try:
        handle = getattr(window, "handle", None)
    except Exception:
        handle = None

    if handle is None:
        return None

    try:
        return int(handle)
    except Exception:
        return None


def _read_foreground_window_handle() -> int | None:
    """현재 foreground 창 handle 을 읽는다."""
    try:
        import ctypes

        handle = int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        return None

    return handle or None


def _read_foreground_window_title(backend: str, handle: int | None) -> str:
    """foreground handle 에 해당하는 제목을 읽는다."""
    if handle is None:
        return ""

    try:
        window = Desktop(backend=backend).window(handle=handle)
    except Exception:
        return ""

    return _read_best_window_title(window)


def _collect_window_rows(
    backend: str,
    *,
    visible_only: bool,
) -> list[WindowRow]:
    """선택한 backend 의 top-level 창 제목 목록을 수집한다."""
    windows = Desktop(backend=backend).windows(
        top_level_only=True,
        visible_only=visible_only,
    )

    rows: list[WindowRow] = []
    for window in windows:
        title = _read_best_window_title(window)
        if not title:
            continue
        rows.append(WindowRow(title=title, handle=_read_window_handle(window)))

    return rows


def _format_handle(handle: int | None) -> str:
    """handle 값을 사람이 읽기 쉬운 형태로 변환한다."""
    if handle is None:
        return "N/A"
    return hex(handle)


def _merge_window_rows(
    base_rows: list[WindowRow],
    new_rows: list[WindowRow],
) -> list[WindowRow]:
    """backend 별 결과를 합치되 같은 창은 한 번만 남긴다."""
    merged = list(base_rows)
    seen_handles = {
        row.handle for row in merged if row.handle is not None
    }
    seen_title_fallbacks = {
        row.title.casefold() for row in merged if row.handle is None
    }

    for row in new_rows:
        if row.handle is not None:
            if row.handle in seen_handles:
                continue
            seen_handles.add(row.handle)
            merged.append(row)
            continue

        title_key = row.title.casefold()
        if title_key in seen_title_fallbacks:
            continue
        seen_title_fallbacks.add(title_key)
        merged.append(row)

    return merged


def _print_report(
    backends_used: tuple[str, ...],
    *,
    visible_only: bool,
    rows: list[WindowRow],
    foreground_handle: int | None,
    foreground_title: str,
) -> None:
    """조회 결과를 콘솔에 출력한다."""
    print(
        f"[INFO] backends_used={backends_used}, visible_only={visible_only}, "
        f"titled_window_count={len(rows)}"
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
        is_foreground = False
        if foreground_handle is not None and row.handle == foreground_handle:
            is_foreground = True
        elif foreground_title and row.title == foreground_title:
            is_foreground = True

        marker = "*" if is_foreground else " "
        print(
            f"[INFO] [{index:02d}] {marker} {row.title} "
            f"(handle={_format_handle(row.handle)})"
        )


def main() -> int:
    """현재 창 제목 목록을 출력한다."""
    if os.name != "nt":
        print(f"[WARNING] 이 스크립트는 Windows 전용입니다: platform={sys.platform}")
        return 1

    if not PYWINAUTO_AVAILABLE:
        print(
            "[ERROR] pywinauto 를 찾지 못했습니다. "
            f"error={PYWINAUTO_IMPORT_ERROR}"
        )
        return 1

    settings = WindowTitlesSettings()
    print(
        f"[INFO] 창 제목 조회 시작: backends={settings.backends}, "
        f"visible_only={settings.visible_only}"
    )

    foreground_handle = _read_foreground_window_handle()
    foreground_title = ""
    merged_rows: list[WindowRow] = []
    successful_backends: list[str] = []

    for backend in settings.backends:
        try:
            rows = _collect_window_rows(backend, visible_only=settings.visible_only)
        except Exception as exc:
            print(f"[WARNING] backend={backend} 조회 실패: error={exc}")
            continue

        successful_backends.append(backend)
        merged_rows = _merge_window_rows(merged_rows, rows)

        if not foreground_title:
            foreground_title = _read_foreground_window_title(backend, foreground_handle)

    if not successful_backends:
        print("[ERROR] 모든 backend 에서 창 제목 조회에 실패했습니다.")
        return 1

    _print_report(
        tuple(successful_backends),
        visible_only=settings.visible_only,
        rows=merged_rows,
        foreground_handle=foreground_handle,
        foreground_title=foreground_title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
