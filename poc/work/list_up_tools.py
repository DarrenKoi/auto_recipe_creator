"""RCS List 탭에서 등록된 툴 목록을 조회한다 (Windows 전용).

UIA 컨트롤 타입 우선순위: List → Tree → DataGrid/Table
어느 것도 발견되지 않으면 --debug 플래그로 컨트롤 트리를 확인해야 한다.

Usage:
    python list_up_tools.py
    python list_up_tools.py --no-switch   # 탭 전환 생략 (이미 List 탭인 경우)
    python list_up_tools.py --debug       # 컨트롤 트리 덤프로 구조 파악
"""

import argparse
import os
import sys
import time
from typing import List

from dotenv import load_dotenv

try:
    from pywinauto import Desktop
    PYWIN_AVAILABLE = True
except ImportError:
    PYWIN_AVAILABLE = False

# switching_tabs의 공개 함수 재사용
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

DEFAULT_WINDOW_TITLE_REGEX = r".*RCS.*"


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser(description="RCS List 탭에서 툴 목록 조회.")
    p.add_argument(
        "--window-title",
        default=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        help="연결할 RCS 창 제목 정규식 (기본: .*RCS.*)",
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
        "--debug",
        action="store_true",
        help="전체 컨트롤 트리 덤프 — 툴이 어느 컨트롤에 있는지 파악할 때 사용",
    )
    return p.parse_args()


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
        툴 이름 문자열 리스트 (빈 리스트면 --debug 로 확인 필요)
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

    for container_type, child_type in [
        ("List",     "ListItem"),
        ("Tree",     "TreeItem"),
        ("DataGrid", "DataItem"),
        ("Table",    "DataItem"),
    ]:
        tools = _collect(container_type, child_type)
        if tools:
            return tools

    print("[WARNING] 알려진 컨트롤 타입에서 툴을 찾지 못했습니다.")
    print("[WARNING] --debug 플래그로 컨트롤 트리를 확인하고 실제 타입을 파악하세요.")
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

    args = parse_args()

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

    tools = get_tool_list(rcs_win)

    if tools:
        print(f"\n[INFO] 발견된 툴 목록 ({len(tools)}개):")
        for i, name in enumerate(tools, 1):
            print(f"  {i:3}. {name}")
    else:
        print("[ERROR] 툴 목록이 비어 있습니다. --debug 플래그로 컨트롤 트리를 확인하세요.")
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
