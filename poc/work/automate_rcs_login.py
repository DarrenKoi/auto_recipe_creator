"""Run RCS and fill the login form automatically (Windows only)."""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

DEFAULT_RCS_EXE = r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"
DEFAULT_SERVER = "Dropbox"
WINDOW_TITLE_REGEX = ".*RCS|RcsMainHD|Login|로그인.*"

try:
    from pywinauto.application import Application
    PYWIN_AVAILABLE = True
except ImportError:
    PYWIN_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Launch RCS and automate login.")
    parser.add_argument(
        "--exe",
        default=os.environ.get("RCS_EXE_PATH", DEFAULT_RCS_EXE),
        help="Path to RcsMainHD.exe",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("RCS_SERVER", DEFAULT_SERVER),
        help="Server name shown in the drop-down (default: Dropbox)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("RCS_USERNAME", ""),
        help="RCS user id",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("RCS_PASSWORD", ""),
        help="RCS password",
    )
    parser.add_argument(
        "--window-title",
        default=WINDOW_TITLE_REGEX,
        help="Regex to find login window",
    )
    parser.add_argument(
        "--launch-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for login window after start",
    )
    parser.add_argument(
        "--post-login-wait",
        type=float,
        default=6.0,
        help="Seconds to observe window after submitting credentials",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass into RcsMainHD.exe",
    )
    return parser.parse_args()


def _is_visible(control) -> bool:
    try:
        return control.is_visible() and control.is_enabled()
    except Exception:
        return False


def _first_visible_windows(windows) -> List:
    return [win for win in windows if _is_visible(win)]


def _wait_for_login_window(app, title_regex: str, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        windows = app.windows()
        visible = _first_visible_windows(windows)
        for window in visible:
            title = ""
            try:
                title = window.window_text() or ""
            except Exception:
                title = ""

            if re.search(title_regex, title, re.IGNORECASE):
                return window

        if visible:
            return visible[0]

        time.sleep(0.5)

    raise TimeoutError(f"Login window not found within {timeout:.0f}s")


def _set_combo_value(combo, value: str) -> bool:
    wrapper = combo.wrapper_object()

    try:
        options = [text.strip() for text in wrapper.item_texts()]
        if value in options:
            wrapper.select(value)
            return True
    except Exception:
        pass

    try:
        wrapper.set_focus()
        wrapper.type_keys(f"{value}{{ENTER}}", set_foreground=False)
        return True
    except Exception:
        return False


def _set_edit_text(edit, value: str) -> bool:
    if value is None:
        return False

    wrapper = edit.wrapper_object()

    for method_name in ("set_edit_text", "set_text"):
        setter = getattr(wrapper, method_name, None)
        if callable(setter):
            try:
                setter(value)
                return True
            except Exception:
                pass

    try:
        wrapper.set_focus()
        wrapper.type_keys("^a{BACKSPACE}", set_foreground=False)
        wrapper.type_keys(value, set_foreground=False, with_spaces=True)
        return True
    except Exception:
        return False


def _find_button(window, label_candidates: List[str]):
    buttons = [b for b in window.descendants(control_type="Button") if _is_visible(b)]
    for button in buttons:
        text = ""
        try:
            text = (button.window_text() or "").strip().lower()
        except Exception:
            text = ""
        if not text:
            continue
        for candidate in label_candidates:
            if candidate in text:
                return button

    if buttons:
        return buttons[0]

    return None


def _submit(window):
    login_keywords = ["login", "log in", "sign in", "ok", "확인", "로그인", "submit", "continue"]
    button = _find_button(window, login_keywords)
    if button is not None:
        button.wrapper_object().click_input()
        return

    window.type_keys("{ENTER}", set_foreground=False)


def main() -> int:
    if os.name != "nt":
        print("[ERROR] This script only supports Windows.")
        return 1

    if not PYWIN_AVAILABLE:
        print("[ERROR] pywinauto is required. Install it first: pip install pywinauto")
        return 2

    args = parse_args()
    exe_path = Path(args.exe).expanduser()

    if not exe_path.exists():
        print(f"[ERROR] executable not found: {exe_path}")
        return 1

    print(f"[INFO] Starting RCS: {exe_path}")
    cmd_str = subprocess.list2cmdline([str(exe_path), *args.extra_args])
    app = Application(backend="uia").start(cmd_str, wait_for_idle=False)

    try:
        login_window = _wait_for_login_window(app, args.window_title, args.launch_timeout)
        print(f"[INFO] 로그인 창: '{login_window.window_text()}'")
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    try:
        combo_boxes = [c for c in login_window.descendants(control_type="ComboBox") if _is_visible(c)]
        if combo_boxes and args.server:
            selected = _set_combo_value(combo_boxes[0], args.server)
            print(f"[INFO] 서버 선택 ({args.server}): {'OK' if selected else 'FAIL'}")
    except Exception as exc:
        print(f"[WARN] 서버 선택 중 오류: {exc}")

    try:
        edits = [e for e in login_window.descendants(control_type="Edit") if _is_visible(e)]
        edits = sorted(edits, key=lambda control: (control.rectangle().top, control.rectangle().left))
        if args.username and len(edits) > 0:
            ok = _set_edit_text(edits[0], args.username)
            print(f"[INFO] User ID 입력: {'OK' if ok else 'FAIL'}")
        if args.password and len(edits) > 1:
            ok = _set_edit_text(edits[1], args.password)
            print(f"[INFO] Password 입력: {'OK' if ok else 'FAIL'}")
    except Exception as exc:
        print(f"[WARN] 입력 처리 중 오류: {exc}")

    _submit(login_window)
    print("[INFO] 로그인 제출 완료")

    time.sleep(args.post_login_wait)
    try:
        if not login_window.is_visible():
            print("[INFO] 로그인 창이 닫혔습니다. 로그인 완료로 추정합니다.")
            return 0
    except Exception:
        print("[INFO] 로그인 창 상태를 더 이상 확인할 수 없습니다.")
        return 0

    print("[WARN] 로그인 창이 아직 표시됩니다. 로그인 실패 또는 추가 인증 창일 수 있습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
