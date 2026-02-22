"""RCS 실행 후 로그인 폼을 자동으로 채워 넣는 스크립트 (Windows 전용)."""

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pywinauto.application import Application

load_dotenv()

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
SERVER = os.environ.get("RCS_SERVER", "Dropbox")
USERNAME = os.environ.get("RCS_USERNAME", "")
PASSWORD = os.environ.get("RCS_PASSWORD", "")
LAUNCH_TIMEOUT = 30.0
POST_LOGIN_WAIT = 6.0
WINDOW_TITLE_PREFIX = "Remote Control System"


def _wait_for_login_window(app):
    """'Remote Control System [Ver' 로 시작하는 창이 나타날 때까지 대기."""
    deadline = time.time() + LAUNCH_TIMEOUT
    while time.time() < deadline:
        for win in app.windows():
            try:
                title = win.window_text() or ""
            except Exception:
                continue
            if title.startswith(WINDOW_TITLE_PREFIX):
                return win
        time.sleep(0.5)
    raise TimeoutError(f"로그인 창을 {LAUNCH_TIMEOUT:.0f}초 내에 찾지 못했습니다")


def _select_server(window) -> None:
    """첫 번째 ComboBox에서 서버를 선택한다."""
    combos = window.descendants(control_type="ComboBox")
    if not combos:
        print("[WARNING] ComboBox를 찾을 수 없습니다")
        return
    combo = combos[0].wrapper_object()
    try:
        combo.select(SERVER)
    except Exception:
        combo.set_focus()
        combo.type_keys(f"{SERVER}{{ENTER}}", set_foreground=False)
    print(f"[INFO] 서버 선택: {SERVER}")


def _fill_credentials(window) -> None:
    """User ID, Password Edit 필드를 채운다 (위→아래 순서)."""
    edits = window.descendants(control_type="Edit")
    edits = sorted(edits, key=lambda c: (c.rectangle().top, c.rectangle().left))
    if len(edits) < 2:
        print(f"[WARNING] Edit 필드 {len(edits)}개 발견 (2개 필요)")
        return

    if USERNAME:
        edits[0].wrapper_object().set_edit_text(USERNAME)
        print("[INFO] User ID 입력 완료")
    if PASSWORD:
        edits[1].wrapper_object().set_edit_text(PASSWORD)
        print("[INFO] Password 입력 완료")


def _click_login(window) -> None:
    """'Log In' 버튼을 클릭한다."""
    for btn in window.descendants(control_type="Button"):
        text = (btn.window_text() or "").strip().lower()
        if "log in" in text or "login" in text:
            btn.wrapper_object().click_input()
            print("[INFO] Log In 버튼 클릭 완료")
            return
    window.type_keys("{ENTER}", set_foreground=False)
    print("[INFO] Log In 버튼 미발견, ENTER 키 전송")


def main() -> int:
    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return 1

    print(f"[INFO] RCS 시작: {RCS_EXE}")
    cmd_str = subprocess.list2cmdline([str(RCS_EXE)])
    app = Application(backend="uia").start(cmd_str, wait_for_idle=False)

    try:
        login_window = _wait_for_login_window(app)
        print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    _select_server(login_window)
    _fill_credentials(login_window)
    _click_login(login_window)

    time.sleep(POST_LOGIN_WAIT)
    try:
        if not login_window.is_visible():
            print("[INFO] 로그인 창이 닫혔습니다. 로그인 성공으로 추정합니다.")
            return 0
    except Exception:
        print("[INFO] 로그인 창 상태를 더 이상 확인할 수 없습니다.")
        return 0

    print("[WARNING] 로그인 창이 아직 표시됩니다. 로그인 실패 또는 추가 인증 필요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
