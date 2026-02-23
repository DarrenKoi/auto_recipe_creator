"""Shared helpers for env-driven RCS UI scripts."""

import os
import time

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from pywinauto import Desktop
    PYWIN_AVAILABLE = True
except ImportError:
    Desktop = None  # type: ignore[assignment]
    PYWIN_AVAILABLE = False

DEFAULT_WINDOW_TITLE_REGEX = r".*RCS.*"
DEFAULT_TIMEOUT = 15.0
DEFAULT_TAB = "List"
TOOL_CONTAINER_ORDER = [
    ("List", "ListItem"),
    ("Tree", "TreeItem"),
    ("DataGrid", "DataItem"),
    ("Table", "DataItem"),
]


def load_env() -> None:
    """Load .env file if python-dotenv is available."""
    if DOTENV_AVAILABLE:
        load_dotenv()


def env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variable."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "y"}


def env_float(name: str, default: float) -> float:
    """Parse float environment variable with safe fallback."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"[WARNING] 잘못된 {name} 값 '{value}', 기본값 {default} 사용")
        return default


def normalize_tab_name(tab_name: str, default: str = DEFAULT_TAB) -> str:
    """Normalize tab name to View/List."""
    value = (tab_name or "").strip()
    if not value:
        return default
    return "View" if value.lower() == "view" else "List"


def _is_visible(control) -> bool:
    """Check control visibility and enabled state."""
    try:
        return control.is_visible() and control.is_enabled()
    except Exception:
        return False


def connect_rcs_window(title_regex: str, timeout: float):
    """Connect to a running RCS main window and return the wrapper."""
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
    """Switch to the target tab in an RCS window."""
    target = tab_name.strip().lower()

    try:
        tab_ctrls = [
            c for c in rcs_window.descendants(control_type="Tab")
            if _is_visible(c)
        ]
        if not tab_ctrls:
            print("[WARNING] TabControl을 찾지 못했습니다. RCS_SWITCH_TAB_DEBUG=1로 컨트롤 트리를 확인하세요.")
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
                time.sleep(0.3)
                print(f"[INFO] 탭 전환 완료: '{title}'")
                return True

        found = [i.window_text() for i in tab_items]
        print(f"[WARNING] '{tab_name}' 탭을 찾지 못했습니다. 발견된 탭: {found}")
        return False

    except Exception as exc:
        print(f"[ERROR] 탭 전환 중 오류: {exc}")
        return False
