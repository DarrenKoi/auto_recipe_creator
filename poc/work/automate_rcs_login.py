"""RCS 실행 후 로그인 폼을 자동으로 채워 넣는 스크립트 (Windows 전용)."""

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pywinauto.application import Application
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False

load_dotenv()

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
SERVER = os.environ.get("RCS_SERVER", "Dropbox")
USERNAME = os.environ.get("RCS_USERNAME", "")
PASSWORD = os.environ.get("RCS_PASSWORD", "")
VLM_API_URL = (
    os.environ.get("VLM_API_URL", "").strip()
    or os.environ.get("VLM_API_BASE_URL", "").strip()
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
try:
    VLM_CHECK_TIMEOUT = float(os.environ.get("VLM_CHECK_TIMEOUT", "3.0"))
except ValueError:
    print("[WARN] VLM_CHECK_TIMEOUT 값이 유효하지 않아 3.0초로 대체합니다.")
    VLM_CHECK_TIMEOUT = 3.0
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


def _check_vlm_responsive() -> bool:
    """VLM API 서버가 응답 가능한지 가볍게 점검."""
    if not VLM_API_URL:
        print("[INFO] VLM_API_URL이 설정되지 않아 응답성 점검을 생략합니다.")
        return True

    if not REQUESTS_AVAILABLE:
        print("[WARNING] requests 패키지가 없어 VLM 응답성 점검을 생략합니다.")
        return True

    headers = {}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    base = VLM_API_URL.rstrip("/")
    if base.endswith("/v1"):
        candidates = [
            f"{base}/models",
            f"{base}/health",
            base,
        ]
    else:
        candidates = [
            f"{base}/v1/models",
            f"{base}/models",
            f"{base}/health",
            base,
        ]

    for url in dict.fromkeys(candidates):
        try:
            response = requests.get(url, headers=headers, timeout=VLM_CHECK_TIMEOUT)
            if response.status_code < 500:
                print(f"[INFO] VLM 응답 확인: {url} -> {response.status_code}")
                return True
            print(f"[WARNING] VLM 응답 상태 이상: {url} -> {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"[WARNING] VLM 타임아웃: {url} ({VLM_CHECK_TIMEOUT:.1f}s)")
        except requests.exceptions.RequestException as exc:
            print(f"[WARNING] VLM 연결 실패: {url} ({exc})")

    print("[ERROR] VLM이 응답하지 않습니다. 환경을 점검한 뒤 다시 실행하세요.")
    return False


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
    if not _check_vlm_responsive():
        return 4

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
