"""RCS에서 툴 화면 창이 열렸는지 확인한다 (Windows 전용).

환경 변수:
    RCS_TOOL_NAME                대상 툴명 (기본: MCD018)
    RCS_TOOL_SCREEN_TIMEOUT      창 탐색 대기 시간(초, 기본: 15)
    RCS_TOOL_SCREEN_SETTLE_SEC   폴링 간격(초, 기본: 0.5)
    RCS_TOOL_SCREEN_BACKENDS     pywinauto 백엔드 (기본: uia,win32)
    RCS_TOOL_SCREEN_ACTIVATE     감지 후 창 활성화 시도 여부 (기본: true)
    RCS_TOOL_SCREEN_DEBUG        디버그 모드 (기본: false)
"""

import os
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
    debug: bool
    check_interval: float
    backends: tuple[str, ...]
    activate_on_detect: bool


def load_settings() -> ToolScreenSettings:
    load_env()
    tool_name = os.environ.get("RCS_TOOL_NAME", "").strip() or DEFAULT_TOOL_NAME
    raw_backends = [
        b.strip().lower()
        for b in os.environ.get("RCS_TOOL_SCREEN_BACKENDS", "uia,win32").split(",")
        if b.strip().lower() in {"win32", "uia"}
    ]
    backends = tuple(raw_backends) if raw_backends else ("uia", "win32")

    return ToolScreenSettings(
        tool_name=tool_name,
        timeout=env_float("RCS_TOOL_SCREEN_TIMEOUT", DEFAULT_TIMEOUT),
        debug=env_flag("RCS_TOOL_SCREEN_DEBUG", False),
        check_interval=env_float("RCS_TOOL_SCREEN_SETTLE_SEC", DEFAULT_TOOL_SCREEN_SETTLE_SEC),
        backends=backends,
        activate_on_detect=env_flag("RCS_TOOL_SCREEN_ACTIVATE", True),
    )


def _title_matches(title: str, tool_name: str) -> bool:
    """창 제목에 툴명이 포함되어 있으면 매칭 (대소문자 무시)."""
    return tool_name.lower() in (title or "").lower()


def _try_activate(window, debug: bool) -> bool:
    """창을 포커스한다. 실패 시 False 반환."""
    try:
        if hasattr(window, "is_minimized") and window.is_minimized():
            window.restore()
            time.sleep(0.1)
    except Exception:
        pass

    try:
        window.set_focus()
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] set_focus 실패: {exc}")

    try:
        window.click_input(coords=(100, 18), button="left")
        return True
    except Exception as exc:
        if debug:
            print(f"[DEBUG] click_input 폴백 실패: {exc}")
        return False


def _scan_once(settings: ToolScreenSettings) -> object | None:
    """모든 백엔드에서 visible 창을 스캔하여 툴명이 포함된 첫 번째 창을 반환한다."""
    for backend in settings.backends:
        try:
            windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True,
            )
        except Exception as exc:
            if settings.debug:
                print(f"[DEBUG] backend={backend} 창 조회 실패: {exc}")
            continue

        for win in windows:
            try:
                title = win.window_text() or ""
            except Exception:
                continue

            if settings.debug:
                print(f"[DEBUG] backend={backend} title={title!r}")

            if _title_matches(title, settings.tool_name):
                print(
                    f"[INFO] 감지됨: backend={backend}, title={title!r}"
                )
                return win

    return None


def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()
    print(f"[INFO] 감지 대상 툴: {settings.tool_name}")
    print(f"[INFO] 탐색 백엔드: {settings.backends}")
    print(f"[INFO] 타임아웃: {settings.timeout}초")

    deadline = time.time() + max(0.5, settings.timeout)
    logged_debug_once = False

    while time.time() < deadline:
        if settings.debug and not logged_debug_once:
            print(f"[DEBUG] 스캔 시작 — 툴명={settings.tool_name!r}")
            logged_debug_once = True

        window = _scan_once(settings)

        if window is not None:
            if settings.activate_on_detect:
                activated = _try_activate(window, settings.debug)
                if activated:
                    print("[INFO] 툴 화면 활성화 완료")
                else:
                    print("[WARNING] 툴 화면 활성화 실패")
            print("[INFO] 툴 화면이 열렸습니다.")
            return 0

        time.sleep(max(0.1, settings.check_interval))

    print(f"[ERROR] {settings.timeout:.1f}초 내에 툴 화면을 감지하지 못했습니다.")
    print(f"[INFO] 감지 대상 툴명: {settings.tool_name!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
