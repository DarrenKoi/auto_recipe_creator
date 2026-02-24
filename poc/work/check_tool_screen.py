"""RCS에서 툴 화면 창이 열렸는지 확인한다 (Windows 전용).

환경 변수:
    RCS_TOOL_NAME                대상 툴명 (기본: MCD018)
    RCS_TOOL_SCREEN_TITLE_REGEX  감지용 정규식 (기본값은 툴명 자동 조합)
    RCS_TOOL_SCREEN_TIMEOUT      창 탐색 대기 시간(초, 기본: 15)
    RCS_TOOL_SCREEN_SETTLE_SEC   폴링 간격(초, 기본: 0.5)
    RCS_TOOL_SCREEN_BACKENDS     pywinauto 백엔드 리스트 (기본: win32,uia)
    RCS_TOOL_SCREEN_DEBUG        1/true/yes/on 이면 미탐지 시 디버그 메시지 출력
"""

import os
import re
import sys
import time
from dataclasses import dataclass

from pywinauto import Desktop

from poc.work.rcs_common import DEFAULT_TIMEOUT, env_float, env_flag, load_env

DEFAULT_TOOL_NAME = "MCD018"
DEFAULT_TOOL_SCREEN_SETTLE_SEC = 0.5


@dataclass(frozen=True)
class ToolScreenSettings:
    tool_name: str
    timeout: float
    tool_window_regex: str
    debug: bool
    check_interval: float
    backends: tuple[str, ...]


def _parse_backends() -> tuple[str, ...]:
    raw = [
        item.strip().lower()
        for item in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "win32,uia").split(",")
        if item.strip()
    ]
    backends = [b for b in raw if b in {"win32", "uia"}]
    return tuple(backends or ["win32", "uia"])


def _build_tool_window_regex(tool_name: str, custom_pattern: str = "") -> str:
    if custom_pattern.strip():
        return custom_pattern.strip()
    escaped_tool = re.escape(tool_name.strip() or DEFAULT_TOOL_NAME)
    return rf"^Remote Monitoring System - .* - {escaped_tool} Server\[[^\]]+\]$"


def _is_match(title: str, pattern: str) -> bool:
    return re.search(pattern, title or "", flags=re.IGNORECASE) is not None


def load_settings() -> ToolScreenSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME
    user_pattern = os.environ.get("RCS_TOOL_SCREEN_TITLE_REGEX", "").strip()
    regex = _build_tool_window_regex(tool_name, user_pattern)
    try:
        re.compile(regex, flags=re.IGNORECASE)
    except re.error:
        print(f"[ERROR] 잘못된 정규식 패턴: {regex!r}")
        print("[WARNING] 기본 패턴으로 폴백합니다.")
        regex = _build_tool_window_regex(tool_name, "")

    return ToolScreenSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        tool_window_regex=regex,
        debug=env_flag("RCS_TOOL_SCREEN_DEBUG", False),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=_parse_backends(),
    )


def _find_tool_windows(pattern: str, backends: tuple[str, ...]) -> list[tuple[str, str, object]]:
    results: list[tuple[str, str, object]] = []
    for backend in backends:
        try:
            windows = Desktop(backend=backend).windows(top_level_only=True, visible_only=True)
        except Exception as exc:
            print(f"[WARNING] backend={backend} 창 조회 실패: {exc}")
            continue

        for win in windows:
            try:
                title = win.window_text() or ""
            except Exception:
                title = ""
            if _is_match(title, pattern):
                results.append((backend, title, win))

    return results


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    print(f"[INFO] 감지 대상 툴: {settings.tool_name}")
    print(f"[INFO] 툴 화면 감지 패턴: {settings.tool_window_regex!r}")
    print(f"[INFO] 탐색 백엔드: {settings.backends}")

    deadline = time.time() + max(0.5, settings.timeout)
    seen = set()

    while time.time() < deadline:
        matches = _find_tool_windows(settings.tool_window_regex, settings.backends)
        if matches:
            for idx, (backend, title, _win) in enumerate(matches, start=1):
                print(f"[INFO] 감지됨 #{idx}: backend={backend}, title={title!r}")
            print(f"[INFO] 툴 화면이 열렸습니다: {len(matches)}개")
            return 0

        if settings.debug:
            if "__none__" not in seen:
                print("[DEBUG] 아직 감지된 창이 없습니다.")
                seen.add("__none__")

        time.sleep(max(0.1, settings.check_interval))

    print(f"[ERROR] {settings.timeout:.1f}초 내에 툴 화면을 감지하지 못했습니다.")
    print(f"[INFO] 시도한 패턴: {settings.tool_window_regex!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
