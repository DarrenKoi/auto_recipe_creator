"""현재 열려 있는 창 제목 목록을 출력하고 대상 창을 foreground 로 올린다.

사용법:
  uv run python poc/work2/window_titles.py

선택 환경변수:
  WINDOW_TITLES_VISIBLE_ONLY=true
  WINDOW_TITLES_TARGET_PREFIX=Remote Control System
  WINDOW_TITLES_RUN_BENCHMARK=true
  WINDOW_TITLES_BENCH_REPEATS=3
  WINDOW_TITLES_SEARCH_BACKENDS=uia,win32
"""

import os
import re
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from poc.work2.util import (
    WindowRow,
    activate_window,
    collect_window_rows,
    find_window_by_title_prefix,
    read_foreground_window_info,
)

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
class BenchmarkStats:
    """탐색 벤치마크 결과."""

    name: str
    repeats: int
    found_count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    last_detail: str


def _compile_title_prefix_pattern(title_prefix: str) -> re.Pattern[str] | None:
    """title_prefix 문자열을 regex prefix 패턴으로 변환한다."""
    normalized_prefix = " ".join(title_prefix.strip().split())
    if not normalized_prefix:
        return None

    pattern_text = rf"^{re.escape(normalized_prefix)}(?:\b| .*)"
    return re.compile(pattern_text, re.IGNORECASE)


def _format_handle(handle: int | None) -> str:
    """handle 값을 사람이 읽기 쉬운 형태로 변환한다."""
    if handle is None:
        return "N/A"
    return hex(handle)


def _find_target_row(
    rows: list[WindowRow],
    title_pattern: re.Pattern[str],
) -> WindowRow | None:
    """rows 에서 정규식 prefix 에 맞는 첫 창을 반환한다."""
    for row in rows:
        if title_pattern.match(row.title):
            return row
    return None


def _benchmark_collect_rows(
    settings: WindowTitlesSettings,
    title_pattern: re.Pattern[str],
) -> BenchmarkStats:
    """shared raw enumeration 기반 창 수집/매칭 시간을 반복 측정한다."""
    elapsed_ms_values: list[float] = []
    found_count = 0
    last_detail = ""

    for index in range(settings.bench_repeats):
        started_at = time.perf_counter()
        rows = collect_window_rows(visible_only=settings.visible_only)
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
        name="window_utils.collect_window_rows",
        repeats=settings.bench_repeats,
        found_count=found_count,
        avg_ms=sum(elapsed_ms_values) / len(elapsed_ms_values),
        min_ms=min(elapsed_ms_values),
        max_ms=max(elapsed_ms_values),
        last_detail=last_detail,
    )


def _benchmark_find_window(settings: WindowTitlesSettings) -> BenchmarkStats:
    """shared title finder 기반 탐색 시간을 반복 측정한다."""
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
    """shared util 탐색 경로를 비교 측정한다."""
    if not settings.run_benchmark:
        print("[INFO] 벤치마크 생략: WINDOW_TITLES_RUN_BENCHMARK=false")
        return

    print(
        "[INFO] 탐색 벤치마크 시작: "
        f"repeats={settings.bench_repeats}, visible_only={settings.visible_only}, "
        f"target_prefix={settings.target_prefix!r}, "
        f"search_backends={settings.search_backends}"
    )
    _print_benchmark_stats(_benchmark_collect_rows(settings, title_pattern))
    _print_benchmark_stats(_benchmark_find_window(settings))


def _print_report(
    *,
    visible_only: bool,
    rows: list[WindowRow],
    foreground_handle: int | None,
    foreground_title: str,
) -> None:
    """조회 결과를 콘솔에 출력한다."""
    print(f"[INFO] visible_only={visible_only}, titled_window_count={len(rows)}")

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
    """현재 창 제목 목록을 출력하고 대상 창을 foreground 로 올린다."""
    if os.name != "nt":
        print(f"[WARNING] 이 스크립트는 Windows 전용입니다: platform={sys.platform}")
        return 1

    settings = WindowTitlesSettings()
    title_pattern = _compile_title_prefix_pattern(settings.target_prefix)
    if title_pattern is None:
        print("[ERROR] WINDOW_TITLES_TARGET_PREFIX 가 비어 있습니다.")
        return 1

    print(
        "[INFO] 창 제목 조회 시작: "
        f"visible_only={settings.visible_only}, "
        f"target_prefix={settings.target_prefix!r}, "
        f"search_backends={settings.search_backends}"
    )

    try:
        foreground_handle, foreground_title = read_foreground_window_info()
        rows = collect_window_rows(visible_only=settings.visible_only)
    except Exception as exc:
        print(f"[ERROR] 창 제목 조회 실패: error={exc}")
        return 1

    _run_benchmark(settings, title_pattern)

    window, window_title, backend = find_window_by_title_prefix(
        settings.target_prefix,
        settings.search_backends,
        visible_only=settings.visible_only,
    )
    if window is None:
        print(
            "[INFO] 대상 창 미발견: "
            f"target_prefix={settings.target_prefix!r}, "
            f"search_backends={settings.search_backends}"
        )
    else:
        print(
            "[INFO] 대상 창 발견: "
            f"title={window_title!r}, backend={backend}"
        )
        if activate_window(
            window,
            debug_label=f"window_titles target backend={backend} title={window_title!r}",
        ):
            print("[INFO] 대상 창을 foreground 로 활성화했습니다.")
        else:
            print("[WARNING] 대상 창 foreground 활성화 실패.")

    foreground_handle, foreground_title = read_foreground_window_info()
    rows = collect_window_rows(visible_only=settings.visible_only)

    _print_report(
        visible_only=settings.visible_only,
        rows=rows,
        foreground_handle=foreground_handle,
        foreground_title=foreground_title,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
