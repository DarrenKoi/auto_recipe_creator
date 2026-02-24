"""RCS List 탭에서 특정 툴을 찾아 선택(클릭)한다 (Windows 전용).

툴 이름은 부분 일치(대소문자 무관)로 검색하므로, 전체 이름을 몰라도 된다.
더블클릭이 필요한 UI의 경우 환경 변수로 동작을 제어한다.

환경 변수:
    RCS_WINDOW_TITLE        연결할 RCS 창 제목 정규식
    RCS_TOOL_NAME           선택할 툴 이름 (부분 일치, 필수)
    RCS_TIMEOUT             창 탐색 대기 제한 시간(초, 기본: 15)
    RCS_SELECT_NO_SWITCH    1/true/yes/on 이면 List 탭 자동 전환 생략
    RCS_SELECT_DOUBLE_CLICK 1/true/yes/on 이면 더블클릭
    RCS_SELECT_LIST_FIRST   1/true/yes/on 이면 선택 전에 전체 목록 출력
    RCS_SELECT_DEBUG        1/true/yes/on 이면 컨트롤 트리 덤프
"""

import os
import sys
from dataclasses import dataclass

from poc.work.rcs_common import (
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_TITLE_REGEX,
    TOOL_CONTAINER_ORDER,
    _is_visible,
    connect_rcs_window,
    env_flag,
    env_float,
    load_env,
    switch_tab,
)
from poc.work.list_up_tools import get_tool_list


@dataclass(frozen=True)
class SelectToolSettings:
    window_title: str
    tool_name: str
    timeout: float
    no_switch: bool
    double_click: bool
    show_list_first: bool
    debug: bool


def load_settings() -> SelectToolSettings:
    load_env()
    return SelectToolSettings(
        window_title=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        tool_name=os.environ.get("RCS_TOOL_NAME", "").strip(),
        timeout=env_float("RCS_TIMEOUT", DEFAULT_TIMEOUT),
        no_switch=env_flag("RCS_SELECT_NO_SWITCH", False),
        double_click=env_flag("RCS_SELECT_DOUBLE_CLICK", False),
        show_list_first=env_flag("RCS_SELECT_LIST_FIRST", False),
        debug=env_flag("RCS_SELECT_DEBUG", False),
    )


# ---------------------------------------------------------------------------
# 툴 컨트롤 탐색
# ---------------------------------------------------------------------------

def _find_tool_control(rcs_window, tool_name: str):
    """툴 이름과 부분 일치하는 UIA 컨트롤을 반환한다.

    탐색 순서: List/ListItem → Tree/TreeItem → DataGrid/DataItem → Table/DataItem

    Args:
        rcs_window: pywinauto 창 래퍼
        tool_name: 검색할 툴 이름 (부분 일치, 대소문자 무관)

    Returns:
        일치하는 컨트롤 래퍼, 없으면 None
    """
    target = tool_name.strip().lower()

    for container_type, child_type in TOOL_CONTAINER_ORDER:
        containers = [
            c for c in rcs_window.descendants(control_type=container_type)
            if _is_visible(c)
        ]
        for container in containers:
            for child in container.children(control_type=child_type):
                try:
                    text = (child.window_text() or "").strip()
                except Exception:
                    text = ""
                if target in text.lower():
                    print(f"[INFO] 툴 발견 [{container_type}/{child_type}]: '{text}'")
                    return child

    return None


# ---------------------------------------------------------------------------
# 툴 선택 (공개 함수)
# ---------------------------------------------------------------------------

def select_tool(
    rcs_window,
    tool_name: str,
    double_click: bool = False,
) -> bool:
    """List 탭에서 지정한 이름의 툴을 찾아 클릭한다.

    Args:
        rcs_window: pywinauto 창 래퍼
        tool_name: 선택할 툴 이름 (부분 일치 허용)
        double_click: True이면 더블클릭으로 선택

    Returns:
        성공 여부
    """
    ctrl = _find_tool_control(rcs_window, tool_name)
    if ctrl is None:
        print(f"[ERROR] '{tool_name}' 에 해당하는 툴을 찾지 못했습니다.")
        print("[ERROR] RCS_SELECT_LIST_FIRST=1로 전체 목록을 확인하거나 RCS_SELECT_DEBUG=1로 컨트롤 트리를 확인하세요.")
        return False

    try:
        if double_click:
            ctrl.double_click_input()
            print(f"[INFO] 툴 더블클릭 완료: '{tool_name}'")
        else:
            ctrl.click_input()
            print(f"[INFO] 툴 클릭 완료: '{tool_name}'")
        return True
    except Exception as exc:
        print(f"[ERROR] 툴 클릭 중 오류: {exc}")
        return False


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> int:
    if os.name != "nt":
        print("[ERROR] 이 스크립트는 Windows 전용입니다.")
        return 1

    settings = load_settings()

    if not settings.tool_name:
        print("[ERROR] 환경변수 RCS_TOOL_NAME 이 필요합니다.")
        return 1

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

    if settings.show_list_first:
        if get_tool_list is not None:
            tools = get_tool_list(rcs_win)
            if tools:
                print(f"\n[INFO] 전체 툴 목록 ({len(tools)}개):")
                for i, name in enumerate(tools, 1):
                    print(f"  {i:3}. {name}")
            print()
        else:
            print("[WARNING] list_up_tools 임포트 실패 — 목록 출력 옵션을 사용할 수 없습니다.")

    ok = select_tool(rcs_win, settings.tool_name, double_click=settings.double_click)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
