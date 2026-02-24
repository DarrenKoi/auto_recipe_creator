"""RCS에서 툴 화면 창이 열렸는지 확인한다 (Windows 전용).

환경 변수:
    RCS_TOOL_NAME                대상 툴명 (기본: MCD018)
    RCS_TOOL_SCREEN_TIMEOUT      창 탐색 대기 시간(초, 기본: 15)
    RCS_TOOL_SCREEN_SETTLE_SEC   폴링 간격(초, 기본: 0.5)
    RCS_TOOL_SCREEN_BACKENDS     pywinauto 백엔드 리스트 (기본: win32,uia)
    RCS_TOOL_SCREEN_PROCESS_NAMES 감지 우선 프로세스명 목록 (기본: RcsViewerHD.exe)
    RCS_TOOL_SCREEN_ACTIVATE     감지 후 창 활성화 시도 여부 (기본: true)
    RCS_TOOL_SCREEN_DEBUG        1/true/yes/on 이면 미탐지 시 디버그 메시지 출력
"""

import os
import sys
import time
from dataclasses import dataclass

from pywinauto import Desktop

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env

DEFAULT_TOOL_NAME = "MCD018"
DEFAULT_TOOL_SCREEN_SETTLE_SEC = 0.5
DEFAULT_TOOL_PROCESS_NAMES = ("RcsViewerHD.exe",)


@dataclass(frozen=True)
class ToolScreenSettings:
    tool_name: str
    timeout: float
    debug: bool
    check_interval: float
    backends: tuple[str, ...]
    process_names: tuple[str, ...]
    activate_on_detect: bool


def _parse_backends() -> tuple[str, ...]:
    raw = [
        item.strip().lower()
        for item in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "win32,uia").split(",")
        if item.strip()
    ]
    backends = [b for b in raw if b in {"win32", "uia"}]
    return tuple(backends or ["win32", "uia"])


def _parse_process_names() -> tuple[str, ...]:
    raw = [item.strip() for item in os.environ.get("RCS_TOOL_SCREEN_PROCESS_NAMES", "").split(",")]
    names = tuple(item for item in raw if item) or DEFAULT_TOOL_PROCESS_NAMES
    return names


def _normalize_process_name(name: str) -> str:
    lowered = (name or "").strip().lower()
    if lowered.endswith(".exe"):
        lowered = lowered[:-4]
    return lowered


def _is_target_process(process_name: str, allowed_names: tuple[str, ...]) -> bool:
    current = _normalize_process_name(process_name)
    if not current:
        return False
    allowed = {_normalize_process_name(item) for item in allowed_names}
    return current in allowed


def _is_tool_window_title(title: str, tool_name: str) -> bool:
    """창 제목에 툴명이 포함되어 있으면 매칭 (대소문자 무시)."""
    title_lower = (title or "").lower()
    tool_lower = (tool_name.strip() or DEFAULT_TOOL_NAME).lower()
    return tool_lower in title_lower


def load_settings() -> ToolScreenSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME

    return ToolScreenSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        debug=env_flag("RCS_TOOL_SCREEN_DEBUG", False),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=_parse_backends(),
        process_names=_parse_process_names(),
        activate_on_detect=env_flag("RCS_TOOL_SCREEN_ACTIVATE", True),
    )


def _safe_process_id(window) -> int | None:
    try:
        pid = window.process_id()
        if isinstance(pid, int) and pid > 0:
            return pid
    except Exception:
        pass

    try:
        pid = int(window.element_info.process_id)
        if pid > 0:
            return pid
    except Exception:
        pass
    return None


def _resolve_process_name(pid: int | None, cache: dict[int, str]) -> str:
    if pid is None or pid <= 0:
        return ""
    if pid in cache:
        return cache[pid]
    if not PSUTIL_AVAILABLE:
        cache[pid] = ""
        return ""
    try:
        cache[pid] = psutil.Process(pid).name()
    except Exception:
        cache[pid] = ""
    return cache[pid]


def _find_target_processes(process_names: tuple[str, ...]) -> list[tuple[int, str]]:
    if not PSUTIL_AVAILABLE:
        return []

    matched: list[tuple[int, str]] = []
    try:
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            pid = proc.info.get("pid")
            name = proc.info.get("name") or ""
            if isinstance(pid, int) and pid > 0 and _is_target_process(name, process_names):
                matched.append((pid, name))
    except Exception as exc:
        print(f"[WARNING] 프로세스 목록 조회 실패: {exc}")
        return []

    matched.sort(key=lambda item: item[0])
    return matched


def _scan_window_list(
    windows,
    source_name: str,
    tool_name: str,
    process_cache: dict[int, str],
    debug_rows: list[str],
    allowed_pids: set[int] | None = None,
) -> list[tuple[str, str, int | None, str, object]]:
    found: list[tuple[str, str, int | None, str, object]] = []
    for idx, win in enumerate(windows, start=1):
        try:
            title = win.window_text() or ""
        except Exception as exc:
            debug_rows.append(f"{source_name}[{idx}] title-read-error={exc}")
            continue

        pid = _safe_process_id(win)
        if allowed_pids is not None and (pid is None or pid not in allowed_pids):
            continue

        process_name = _resolve_process_name(pid, process_cache)
        matched = _is_tool_window_title(title, tool_name)
        debug_rows.append(
            f"{source_name}[{idx}] matched={matched} pid={pid} "
            f"process={process_name or 'unknown'} title={title!r}"
        )
        if matched:
            found.append((source_name, title, pid, process_name, win))
    return found


def _find_tool_windows(
    backends: tuple[str, ...],
    tool_name: str,
    allowed_pids: set[int] | None = None,
) -> tuple[list[tuple[str, str, int | None, str, object]], str, list[str]]:
    matches: list[tuple[str, str, int | None, str, object]] = []
    seen_keys: set[tuple[str, int | None, str]] = set()
    process_cache: dict[int, str] = {}
    debug_rows: list[str] = []
    for backend in backends:
        try:
            windows = Desktop(backend=backend).windows(top_level_only=True, visible_only=True)
        except Exception as exc:
            print(f"[WARNING] backend={backend} 창 조회 실패: {exc}")
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        scanned = _scan_window_list(
            windows=windows,
            source_name=f"desktop[{backend}]",
            tool_name=tool_name,
            process_cache=process_cache,
            debug_rows=debug_rows,
            allowed_pids=allowed_pids,
        )
        for source, title, pid, process_name, win in scanned:
            row = (source, title, pid, process_name, win)
            dedupe_key = (_normalize_process_name(process_name), pid, title)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            matches.append(row)

    mode = "process->title" if allowed_pids is not None else "title"
    return matches, mode, debug_rows


def _activate_window(
    window,
    backend: str,
    pid: int | None,
    process_name: str,
    debug: bool,
) -> bool:
    try:
        if hasattr(window, "is_minimized") and window.is_minimized():
            window.restore()
            time.sleep(0.1)
    except Exception:
        pass

    if pid:
        try:
            process_window = Desktop(backend=backend).window(process=pid, handle=window.handle)
            process_window.set_focus()
            print(
                f"[INFO] 프로세스 기반 포커스 성공: pid={pid}, "
                f"process={process_name or 'unknown'}"
            )
            return True
        except Exception as exc:
            if debug:
                print(f"[DEBUG] 프로세스 기반 포커스 실패: {exc}")

    try:
        window.set_focus()
        print("[INFO] set_focus()로 툴 화면 활성화 완료")
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] set_focus 실패: {exc}")

    try:
        rect = window.rectangle()
        width = max(1, rect.right - rect.left)
        height = max(1, rect.bottom - rect.top)
        click_x = max(1, min(width - 1, width // 2))
        click_y = max(1, min(height - 1, 18))
        window.click_input(coords=(click_x, click_y), button="left")
        print("[INFO] click_input() 폴백으로 툴 화면 활성화 완료")
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] click_input 폴백 실패: {exc}")
        return False


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    print(f"[INFO] 감지 대상 툴: {settings.tool_name}")
    print(f"[INFO] 탐색 백엔드: {settings.backends}")
    print(f"[INFO] 우선 프로세스명: {settings.process_names}")
    if not PSUTIL_AVAILABLE:
        print("[WARNING] psutil 미설치: 프로세스명 기반 우선 탐지가 비활성화됩니다.")
    print(f"[INFO] 감지 후 활성화 시도: {settings.activate_on_detect}")

    deadline = time.time() + max(0.5, settings.timeout)
    seen: set[str] = set()
    seen_process_snapshots: set[str] = set()

    while time.time() < deadline:
        process_list = _find_target_processes(settings.process_names)
        pid_set = {pid for pid, _name in process_list}
        process_snapshot = ",".join(f"{pid}:{name}" for pid, name in process_list) or "__none__"
        if process_snapshot not in seen_process_snapshots:
            if process_list:
                print(
                    f"[INFO] 대상 프로세스 감지: count={len(process_list)}, "
                    f"items={[f'{pid}:{name}' for pid, name in process_list]}"
                )
            else:
                print("[INFO] 대상 프로세스 미감지: 전체 창 제목 스캔으로 폴백")
            seen_process_snapshots.add(process_snapshot)

        matches: list[tuple[str, str, int | None, str, object]] = []
        match_mode = "title"
        debug_rows: list[str] = []
        if pid_set:
            print("[INFO] 프로세스 기반 창 탐색 시작: PID별 툴 화면 제목 확인")
            matches, match_mode, process_rows = _find_tool_windows(
                settings.backends,
                settings.tool_name,
                allowed_pids=pid_set,
            )
            for row in process_rows:
                key = f"INFO_PROCESS_ROW::{row}"
                if key in seen:
                    continue
                seen.add(key)
                print(f"[INFO] {row}")
            debug_rows.extend(process_rows)
            if not matches:
                print("[INFO] 대상 프로세스 창에서 툴 제목 미탐지, 전체 창 스캔 폴백")

        if not matches:
            all_matches, fallback_mode, fallback_rows = _find_tool_windows(
                settings.backends,
                settings.tool_name,
            )
            matches = all_matches
            match_mode = fallback_mode if not pid_set else f"{match_mode}->fallback:{fallback_mode}"
            debug_rows.extend(fallback_rows)

        if matches:
            for idx, (source, title, pid, process_name, _win) in enumerate(matches, start=1):
                print(
                    f"[INFO] 감지됨 #{idx}: source={source}, pid={pid}, "
                    f"process={process_name or 'unknown'}, title={title!r}"
                )
            print(f"[INFO] 매칭 방식: {match_mode}")
            if settings.activate_on_detect:
                source, title, pid, process_name, window = matches[0]
                backend = source.split("[", 1)[-1].rstrip("]") if source.startswith("desktop[") else "win32"
                activated = _activate_window(window, backend, pid, process_name, settings.debug)
                if activated:
                    print(f"[INFO] 활성화 대상: title={title!r}")
                else:
                    print(f"[WARNING] 툴 화면 활성화 실패: title={title!r}")
            print(f"[INFO] 툴 화면이 열렸습니다: {len(matches)}개")
            return 0

        if settings.debug:
            if "__scan__" not in seen:
                print(f"[DEBUG] 감지 대상 툴명: {settings.tool_name!r}")
                seen.add("__scan__")
            for row in debug_rows:
                key = f"row::{row}"
                if key in seen:
                    continue
                seen.add(key)
                print(f"[DEBUG] {row}")

        time.sleep(max(0.1, settings.check_interval))

    print(f"[ERROR] {settings.timeout:.1f}초 내에 툴 화면을 감지하지 못했습니다.")
    print(f"[INFO] 감지 대상 툴명: {settings.tool_name!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
