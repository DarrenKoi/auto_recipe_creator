"""RCS List 탭에서 등록된 툴 목록을 조회한다 (Windows 전용).

UIA 컨트롤 타입 우선순위: List → Tree → DataGrid/Table
어느 것도 발견되지 않으면 디버그 모드로 컨트롤 트리를 확인해야 한다.

환경 변수:
    RCS_WINDOW_TITLE    연결할 RCS 창 제목 정규식
    RCS_TIMEOUT         창 탐색 대기 제한 시간(초, 기본: 15)
    RCS_LIST_NO_SWITCH  1/true/yes/on 이면 List 탭 자동 전환 생략
    RCS_LIST_DEBUG      1/true/yes/on 이면 컨트롤 트리 덤프
"""

import os
import sys
from dataclasses import dataclass
from typing import List

try:
    from .rcs_common import (
        DEFAULT_TIMEOUT,
        DEFAULT_WINDOW_TITLE_REGEX,
        PYWIN_AVAILABLE,
        TOOL_CONTAINER_ORDER,
        _is_visible,
        connect_rcs_window,
        env_flag,
        env_float,
        load_env,
        switch_tab,
    )
except ImportError:
    from rcs_common import (
        DEFAULT_TIMEOUT,
        DEFAULT_WINDOW_TITLE_REGEX,
        PYWIN_AVAILABLE,
        TOOL_CONTAINER_ORDER,
        _is_visible,
        connect_rcs_window,
        env_flag,
        env_float,
        load_env,
        switch_tab,
    )


@dataclass(frozen=True)
class ListToolsSettings:
    window_title: str
    timeout: float
    no_switch: bool
    debug: bool


def load_settings() -> ListToolsSettings:
    load_env()
    return ListToolsSettings(
        window_title=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        timeout=env_float("RCS_TIMEOUT", DEFAULT_TIMEOUT),
        no_switch=env_flag("RCS_LIST_NO_SWITCH", False),
        debug=env_flag("RCS_LIST_DEBUG", False),
    )


# ---------------------------------------------------------------------------
# 툴 목록 조회 (공개 함수 — select_tool.py 에서 임포트)
# ---------------------------------------------------------------------------

def get_tool_list(rcs_window) -> List[str]:
    """List 탭 영역에서 툴 이름 목록을 반환한다.

    UIA 컨트롤 탐색 순서:
        1. ListView (ListItem)
        2. TreeView (TreeItem)
        3. DataGrid / Table (DataItem)

    하나라도 이름이 발견되면 즉시 반환하므로 중복이 없다.

    Returns:
        툴 이름 문자열 리스트 (빈 리스트면 RCS_LIST_DEBUG=1로 확인 필요)
    """

    def _collect(container_type: str, child_type: str) -> List[str]:
        containers = [
            c for c in rcs_window.descendants(control_type=container_type)
            if _is_visible(c)
        ]
        for container in containers:
            names = []
            for child in container.children(control_type=child_type):
                try:
                    text = (child.window_text() or "").strip()
                except Exception:
                    text = ""
                if text:
                    names.append(text)
            if names:
                print(f"[INFO] {container_type}/{child_type} 에서 {len(names)}개 항목 발견")
                return names
        return []

    for container_type, child_type in TOOL_CONTAINER_ORDER:
        tools = _collect(container_type, child_type)
        if tools:
            return tools

    print("[WARNING] 알려진 컨트롤 타입에서 툴을 찾지 못했습니다.")
    print("[WARNING] RCS_LIST_DEBUG=1로 컨트롤 트리를 확인하고 실제 타입을 파악하세요.")
    return []


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

    if not settings.no_switch:
        ok = switch_tab(rcs_win, "List")
        if not ok:
            print("[WARNING] List 탭 전환 실패 — 현재 탭에서 계속 진행합니다.")

    if settings.debug:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=5):")
        rcs_win.print_control_identifiers(depth=5)

    tools = get_tool_list(rcs_win)

    if tools:
        print(f"\n[INFO] 발견된 툴 목록 ({len(tools)}개):")
        for i, name in enumerate(tools, 1):
            print(f"  {i:3}. {name}")
    else:
        print("[ERROR] 툴 목록이 비어 있습니다. RCS_LIST_DEBUG=1로 컨트롤 트리를 확인하세요.")
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
