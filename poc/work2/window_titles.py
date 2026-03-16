"""현재 열려 있는 창 제목 목록을 빠르게 출력하는 테스트 스크립트.

사용법:
  uv run python poc/work2/window_titles.py

선택 환경변수:
  WINDOW_TITLES_VISIBLE_ONLY=true
  WINDOW_TITLES_TARGET_REGEX=^Remote Control System(?:\\b| .*)
  WINDOW_TITLES_TARGET_PREFIX=Remote Control System
  WINDOW_TITLES_RUN_BENCHMARK=true
  WINDOW_TITLES_BENCH_REPEATS=3
  WINDOW_TITLES_SEARCH_BACKENDS=uia,win32
"""

import ctypes
import os
import re
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

from dotenv import load_dotenv

try:
    from poc.work2.util import activate_window, find_window_by_title_prefix

    WINDOW_UTILS_AVAILABLE = True
    WINDOW_UTILS_IMPORT_ERROR = None
except Exception as exc:
    activate_window = None
    find_window_by_title_prefix = None
    WINDOW_UTILS_AVAILABLE = False
    WINDOW_UTILS_IMPORT_ERROR = exc

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


def _parse_int_env(env_name: str, default: int) -> int:
    """환경변수 int 값을 읽는다."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    try:
        return int(raw_value.strip())
    except ValueError:
        return default


def _parse_csv_env(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """쉼표 구분 환경변수를 tuple 로 읽는다."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default

    items = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return items or default


@dataclass(frozen=True)
class WindowTitlesSettings:
    """창 제목 조회 설정."""

    visible_only: bool = _parse_bool_env("WINDOW_TITLES_VISIBLE_ONLY", True)
    target_regex: str = os.getenv(
        "WINDOW_TITLES_TARGET_REGEX",
        r"^Remote Control System(?:\b| .*)",
    )
    target_prefix: str = os.getenv(
        "WINDOW_TITLES_TARGET_PREFIX",
        "Remote Control System",
    ).strip()
    run_benchmark: bool = _parse_bool_env("WINDOW_TITLES_RUN_BENCHMARK", True)
    bench_repeats: int = max(1, _parse_int_env("WINDOW_TITLES_BENCH_REPEATS", 3))
    search_backends: tuple[str, ...] = _parse_csv_env(
        "WINDOW_TITLES_SEARCH_BACKENDS",
        ("uia", "win32"),
    )


@dataclass(frozen=True)
class WindowRow:
    """출력용 창 정보."""

    title: str
    handle: int


@dataclass(frozen=True)
class BenchmarkStats:
    """탐색 벤치마크 결과."""

    name: str
    repeats: int
    found_count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    last_detail: str


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


def _format_elapsed_ms(elapsed_sec: float) -> str:
    """초 단위 경과 시간을 ms 문자열로 변환한다."""
    return f"{elapsed_sec * 1000:.1f}ms"


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


def _focus_window(user32, hwnd: int) -> bool:
    """창을 foreground 로 활성화한다.

    최소화 상태면 복원 후 SetForegroundWindow 를 호출한다.
    """
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)

    set_foreground_ok = bool(user32.SetForegroundWindow(hwnd))
    foreground_handle = _read_foreground_window_handle(user32)
    return set_foreground_ok and foreground_handle == hwnd


def _compile_target_regex(pattern_text: str) -> re.Pattern[str] | None:
    """대상 창 제목용 정규식을 compile 한다."""
    try:
        return re.compile(pattern_text, re.IGNORECASE)
    except re.error as exc:
        print(
            f"[ERROR] WINDOW_TITLES_TARGET_REGEX 컴파일 실패: "
            f"pattern={pattern_text!r}, error={exc}"
        )
        return None


def _find_and_focus_target_window(
    user32,
    rows: list[WindowRow],
    title_pattern: re.Pattern[str],
) -> bool:
    """rows 에서 정규식 prefix 와 매칭되는 창을 찾아 foreground 로 올린다."""
    for row in rows:
        if title_pattern.match(row.title):
            print(
                f"[INFO] 대상 창 발견: title={row.title!r}, "
                f"handle={_format_handle(row.handle)}"
            )
            ok = _focus_window(user32, row.handle)
            if ok:
                print("[INFO] 대상 창을 foreground 로 활성화했습니다.")
            else:
                print("[WARNING] SetForegroundWindow 실패 — 권한 또는 포커스 제한.")
            return ok

    print(
        f"[INFO] 대상 창을 찾지 못했습니다: "
        f"target_regex={title_pattern.pattern!r}"
    )
    return False


def _find_target_row(
    rows: list[WindowRow],
    title_pattern: re.Pattern[str],
) -> WindowRow | None:
    """rows 에서 정규식 prefix 에 맞는 첫 창을 반환한다."""
    for row in rows:
        if title_pattern.match(row.title):
            return row
    return None


def _benchmark_enum_search(
    settings: WindowTitlesSettings,
    title_pattern: re.Pattern[str],
) -> BenchmarkStats:
    """직접 EnumWindows 탐색을 반복 측정한다."""
    elapsed_ms_values: list[float] = []
    found_count = 0
    last_detail = ""

    for index in range(settings.bench_repeats):
        started_at = time.perf_counter()
        rows = _collect_window_rows(visible_only=settings.visible_only)
        matched_row = _find_target_row(rows, title_pattern)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        elapsed_ms_values.append(elapsed_ms)
        if matched_row is not None:
            found_count += 1
            last_detail = (
                f"title={matched_row.title!r}, "
                f"handle={_format_handle(matched_row.handle)}, "
                f"row_count={len(rows)}, run={index + 1}"
            )
        else:
            last_detail = f"not_found row_count={len(rows)}, run={index + 1}"

    return BenchmarkStats(
        name="enum_windows",
        repeats=settings.bench_repeats,
        found_count=found_count,
        avg_ms=sum(elapsed_ms_values) / len(elapsed_ms_values),
        min_ms=min(elapsed_ms_values),
        max_ms=max(elapsed_ms_values),
        last_detail=last_detail,
    )


def _benchmark_window_utils_search(settings: WindowTitlesSettings) -> BenchmarkStats | None:
    """window_utils 의 find_window_by_title_prefix 탐색을 반복 측정한다."""
    if not WINDOW_UTILS_AVAILABLE or find_window_by_title_prefix is None:
        return None

    elapsed_ms_values: list[float] = []
    found_count = 0
    last_detail = ""

    for index in range(settings.bench_repeats):
        started_at = time.perf_counter()
        window, window_title, backend = find_window_by_title_prefix(
            settings.target_prefix,
            settings.search_backends,
            visible_only=settings.visible_only,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        elapsed_ms_values.append(elapsed_ms)
        if window is not None:
            found_count += 1
            last_detail = (
                f"title={window_title!r}, backend={backend}, run={index + 1}"
            )
        else:
            last_detail = f"not_found run={index + 1}"

    return BenchmarkStats(
        name="window_utils.find_window_by_title_prefix",
        repeats=settings.bench_repeats,
        found_count=found_count,
        avg_ms=sum(elapsed_ms_values) / len(elapsed_ms_values),
        min_ms=min(elapsed_ms_values),
        max_ms=max(elapsed_ms_values),
        last_detail=last_detail,
    )


def _print_benchmark_stats(stats: BenchmarkStats) -> None:
    """벤치마크 요약을 출력한다."""
    print(
        "[INFO] benchmark "
        f"name={stats.name}, repeats={stats.repeats}, "
        f"found={stats.found_count}/{stats.repeats}, "
        f"avg={stats.avg_ms:.1f}ms, min={stats.min_ms:.1f}ms, "
        f"max={stats.max_ms:.1f}ms"
    )
    print(f"[INFO] benchmark detail: {stats.last_detail}")


def _run_benchmark(
    settings: WindowTitlesSettings,
    title_pattern: re.Pattern[str],
) -> None:
    """EnumWindows 와 shared util 탐색을 비교 측정한다."""
    if not settings.run_benchmark:
        print("[INFO] 벤치마크 생략: WINDOW_TITLES_RUN_BENCHMARK=false")
        return

    print(
        "[INFO] 탐색 벤치마크 시작: "
        f"repeats={settings.bench_repeats}, visible_only={settings.visible_only}, "
        f"target_prefix={settings.target_prefix!r}, "
        f"target_regex={settings.target_regex!r}, "
        f"search_backends={settings.search_backends}"
    )
    enum_stats = _benchmark_enum_search(settings, title_pattern)
    _print_benchmark_stats(enum_stats)

    window_utils_stats = _benchmark_window_utils_search(settings)
    if window_utils_stats is None:
        print(
            "[WARNING] window_utils 벤치마크 생략: "
            f"import_error={WINDOW_UTILS_IMPORT_ERROR}"
        )
        return

    _print_benchmark_stats(window_utils_stats)


def _focus_target_with_window_utils(settings: WindowTitlesSettings) -> bool:
    """shared util 탐색/활성화 경로로 대상 창을 foreground 로 올린다."""
    if not WINDOW_UTILS_AVAILABLE:
        print(
            "[WARNING] window_utils focus 생략: "
            f"import_error={WINDOW_UTILS_IMPORT_ERROR}"
        )
        return False

    if activate_window is None or find_window_by_title_prefix is None:
        print("[WARNING] window_utils focus 생략: required function missing")
        return False

    window, window_title, backend = find_window_by_title_prefix(
        settings.target_prefix,
        settings.search_backends,
        visible_only=settings.visible_only,
    )
    if window is None:
        print(
            "[INFO] window_utils 대상 창 미발견: "
            f"target_prefix={settings.target_prefix!r}, "
            f"search_backends={settings.search_backends}"
        )
        return False

    print(
        "[INFO] window_utils 대상 창 발견: "
        f"title={window_title!r}, backend={backend}"
    )
    ok = activate_window(
        window,
        debug_label=f"window_titles target backend={backend} title={window_title!r}",
    )
    if ok:
        print("[INFO] window_utils 경로로 대상 창을 foreground 로 활성화했습니다.")
    else:
        print("[WARNING] window_utils 경로 foreground 활성화 실패")
    return ok


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
    print(
        "[INFO] 창 제목 조회 시작: "
        f"visible_only={settings.visible_only}, "
        f"target_regex={settings.target_regex!r}, "
        f"target_prefix={settings.target_prefix!r}, "
        f"search_backends={settings.search_backends}"
    )
    title_pattern = _compile_target_regex(settings.target_regex)
    if title_pattern is None:
        return 1

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

    _run_benchmark(settings, title_pattern)

    focused = _focus_target_with_window_utils(settings)
    if not focused:
        print("[INFO] window_utils foreground 실패 → raw EnumWindows fallback")
        _find_and_focus_target_window(user32, rows, title_pattern)

    foreground_handle = _read_foreground_window_handle(user32)
    foreground_title = ""
    if foreground_handle is not None:
        foreground_title = _read_window_text(user32, foreground_handle, buf)
    rows = _collect_window_rows(visible_only=settings.visible_only)

    _print_report(
        visible_only=settings.visible_only,
        rows=rows,
        foreground_handle=foreground_handle,
        foreground_title=foreground_title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
