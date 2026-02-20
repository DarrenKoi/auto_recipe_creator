"""RCS 메인 창에서 View / List 탭을 전환한다 (Windows 전용).

Usage:
    python switching_tabs.py --tab List
    python switching_tabs.py --tab View --debug
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

try:
    from pywinauto import Desktop
    PYWIN_AVAILABLE = True
except ImportError:
    PYWIN_AVAILABLE = False

# 로그인 후 RCS 메인 창 제목 패턴 (로그인 창과 달리 서버명이 포함되는 경우가 많음)
DEFAULT_WINDOW_TITLE_REGEX = r".*RCS.*"
DEFAULT_TAB = "List"


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser(description="RCS View / List 탭 전환.")
    p.add_argument(
        "--window-title",
        default=os.environ.get("RCS_WINDOW_TITLE", DEFAULT_WINDOW_TITLE_REGEX),
        help="연결할 RCS 창 제목 정규식 (기본: .*RCS.*)",
    )
    p.add_argument(
        "--tab",
        default=DEFAULT_TAB,
        choices=["View", "List", "view", "list"],
        help="전환할 탭 이름 (기본: List)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="창 탐색 대기 제한 시간(초, 기본: 15)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="탭 컨트롤 트리 전체 덤프 (구조 파악용)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 공통 헬퍼 (list_up_tools.py / select_tool.py 에서도 동일 패턴 사용)
# ---------------------------------------------------------------------------

def _is_visible(control) -> bool:
    """컨트롤이 화면에 보이고 활성화된 상태인지 확인."""
    try:
        return control.is_visible() and control.is_enabled()
    except Exception:
        return False


def connect_rcs_window(title_regex: str, timeout: float):
    """실행 중인 RCS 메인 창에 연결하고 창 래퍼를 반환.

    Args:
        title_regex: 창 제목 정규식
        timeout: 대기 제한 시간(초)

    Returns:
        pywinauto 창 래퍼

    Raises:
        TimeoutError: 제한 시간 내 창을 찾지 못한 경우
    """
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


# ---------------------------------------------------------------------------
# 탭 전환 (공개 함수 — 다른 스크립트에서 임포트 가능)
# ---------------------------------------------------------------------------

def switch_tab(rcs_window, tab_name: str) -> bool:
    """RCS 창에서 지정한 탭으로 전환한다.

    Args:
        rcs_window: pywinauto 창 래퍼
        tab_name: 전환할 탭 이름 (대소문자 무관, 부분 일치)

    Returns:
        성공 여부
    """
    target = tab_name.strip().lower()

    try:
        tab_ctrls = [
            c for c in rcs_window.descendants(control_type="Tab")
            if _is_visible(c)
        ]
        if not tab_ctrls:
            print("[WARNING] TabControl을 찾지 못했습니다. --debug 로 컨트롤 트리를 확인하세요.")
            return False

        tab_ctrl = tab_ctrls[0]
        tab_items = [
            c for c in tab_ctrl.children(control_type="TabItem")
            if _is_visible(c)
        ]

        for item in tab_items:
            try:
                title = (item.window_text() or "").strip()
            except Exception:
                title = ""
            if target in title.lower():
                item.click_input()
                time.sleep(0.3)  # 탭 콘텐츠 로딩 대기
                print(f"[INFO] 탭 전환 완료: '{title}'")
                return True

        found = [i.window_text() for i in tab_items]
        print(f"[WARNING] '{tab_name}' 탭을 찾지 못했습니다. 발견된 탭: {found}")
        return False

    except Exception as exc:
        print(f"[ERROR] 탭 전환 중 오류: {exc}")
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

    try:
        rcs_win = connect_rcs_window(args.window_title, args.timeout)
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    if args.debug:
        print("[DEBUG] 전체 컨트롤 트리 덤프 (depth=4):")
        rcs_win.print_control_identifiers(depth=4)

    ok = switch_tab(rcs_win, args.tab)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
