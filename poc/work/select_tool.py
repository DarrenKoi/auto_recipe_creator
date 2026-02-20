"""RCS List 탭에서 특정 툴을 찾아 선택(클릭)한다 (Windows 전용).

툴 이름은 부분 일치(대소문자 무관)로 검색하므로, 전체 이름을 몰라도 된다.
더블클릭이 필요한 UI의 경우 --double-click 플래그를 사용한다.

Usage:
    python select_tool.py --tool-name "CD-SEM"
    python select_tool.py --tool-name "CD-SEM" --double-click
    python select_tool.py --tool-name "CD-SEM" --debug  # 구조 파악 후 재시도
"""

import argparse
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv

try:
    from pywinauto import Desktop
    PYWIN_AVAILABLE = True
except ImportError:
    PYWIN_AVAILABLE = False

# switching_tabs / list_up_tools 공개 함수 재사용
try:
    from switching_tabs import connect_rcs_window, switch_tab, _is_visible
    _SWITCHING_IMPORTED = True
except ImportError:
    _SWITCHING_IMPORTED = False

    def _is_visible(control) -> bool:
        try:
            return control.is_visible() and control.is_enabled()
        except Exception:
            return False

    def connect_rcs_window(title_regex: str, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                windows = Desktop(backend="uia").windows(title_re=title_regex)
                visible = [w for w in windows if _is_visible(w)]
                if visible:
                    print(f"[INFO] RCS 창 연결: '{visible[0].window_text()}'")
                    return visible[0]
            except Exception as exc:
                print(f"[WARNING] 창 탐색 중 오류: {exc}")
            time.sleep(0.5)
        raise TimeoutError(f"RCS 창을 {timeout:.0f}초 내에 찾지 못했습니다.")

    def switch_tab(rcs_window, tab_name: str) -> bool:
        print("[WARNING] switching_tabs 임포트 실패 — 탭 전환 불가")
        return False

try:
    from list_up_tools import get_tool_list
    _LIST_IMPORTED = True
except ImportError:
    _LIST_IMPORTED = False

DEFAULT_WINDOW_TITLE_REGEX = r".*RCS.*"


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser(description="RCS List 탭에서 툴 선택.")
    p.add_argument(
        "--window-title",
        default=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        help="연결할 RCS 창 제목 정규식 (기본: .*RCS.*)",
    )
    p.add_argument(
        "--tool-name",
        default=os.environ.get("RCS_TOOL_NAME", ""),
        help="선택할 툴 이름 (부분 일치, 대소문자 무관). 환경변수 RCS_TOOL_NAME 대체 가능.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="창 탐색 대기 제한 시간(초, 기본: 15)",
    )
    p.add_argument(
        "--no-switch",
        action="store_true",
        help="List 탭 자동 전환 건너뜀 (이미 List 탭에 있는 경우)",
    )
    p.add_argument(
        "--double-click",
        action="store_true",
        help="단일 클릭 대신 더블클릭으로 툴을 열기",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="선택 전에 전체 툴 목록을 먼저 출력 (확인용)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="전체 컨트롤 트리 덤프 (구조 파악용)",
    )
    return p.parse_args()


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

    for container_type, child_type in [
        ("List",     "ListItem"),
        ("Tree",     "TreeItem"),
        ("DataGrid", "DataItem"),
        ("Table",    "DataItem"),
    ]:
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
        print("[ERROR] --list 로 전체 목록을 확인하거나 --debug 로 컨트롤 트리를 확인하세요.")
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
    if not PYWIN_AVAILABLE:
        print("[ERROR] pywinauto가 필요합니다: pip install pywinauto")
        return 2

    args = parse_args()

    if not args.tool_name:
        print("[ERROR] --tool-name 또는 환경변수 RCS_TOOL_NAME 이 필요합니다.")
        return 1

    try:
        rcs_win = connect_rcs_window(args.window_title, args.timeout)
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    if not args.no_switch:
        ok = switch_tab(rcs_win, "List")
        if not ok:
            print("[WARNING] List 탭 전환 실패 — 현재 탭에서 계속 진행합니다.")

    if args.debug:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=5):")
        rcs_win.print_control_identifiers(depth=5)

    if args.list:
        if _LIST_IMPORTED:
            tools = get_tool_list(rcs_win)
            if tools:
                print(f"\n[INFO] 전체 툴 목록 ({len(tools)}개):")
                for i, name in enumerate(tools, 1):
                    print(f"  {i:3}. {name}")
            print()
        else:
            print("[WARNING] list_up_tools 임포트 실패 — --list 옵션을 사용할 수 없습니다.")

    ok = select_tool(rcs_win, args.tool_name, double_click=args.double_click)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
