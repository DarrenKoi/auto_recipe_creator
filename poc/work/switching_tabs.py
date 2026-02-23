"""RCS 메인 창에서 View / List 탭을 전환한다 (Windows 전용).

환경 변수:
    RCS_WINDOW_TITLE       연결할 RCS 창 제목 정규식
    RCS_TAB_NAME           전환할 탭 이름 (기본: List)
    RCS_TIMEOUT            창 탐색 대기 제한 시간(초, 기본: 15)
    RCS_SWITCH_TAB_DEBUG   1/true/yes/on 이면 컨트롤 트리 덤프
"""

import os
import sys
from dataclasses import dataclass

from poc.work.rcs_common import (
    DEFAULT_TAB,
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_TITLE_REGEX,
    PYWIN_AVAILABLE,
    connect_rcs_window,
    env_flag,
    env_float,
    load_env,
    normalize_tab_name,
    switch_tab,
)


@dataclass(frozen=True)
class SwitchTabSettings:
    window_title: str
    tab_name: str
    timeout: float
    debug: bool


def load_settings() -> SwitchTabSettings:
    load_env()
    raw_tab = os.environ.get("RCS_TAB_NAME", DEFAULT_TAB)
    return SwitchTabSettings(
        window_title=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        tab_name=normalize_tab_name(raw_tab, DEFAULT_TAB),
        timeout=env_float("RCS_TIMEOUT", DEFAULT_TIMEOUT),
        debug=env_flag("RCS_SWITCH_TAB_DEBUG", False),
    )


# ---------------------------------------------------------------------------
# 공통 공개 함수
# connect_rcs_window / switch_tab 은 rcs_common에서 임포트해 재사용한다.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1
    if not PYWIN_AVAILABLE:
        print("[ERROR] pywinauto가 필요합니다: pip install pywinauto")
        return 2

    settings = load_settings()

    try:
        rcs_win = connect_rcs_window(settings.window_title, settings.timeout)
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    if settings.debug:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=4):")
        rcs_win.print_control_identifiers(depth=4)

    ok = switch_tab(rcs_win, settings.tab_name)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
