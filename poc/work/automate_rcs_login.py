"""Run RCS and fill the login form automatically (Windows only)."""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from pynput.mouse import Button as _MouseButton, Controller as _PynputMouse
    from pynput.keyboard import Controller as _PynputKeyboard, Key as _Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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


def _window_rect(window) -> dict:
    """pywinauto 창의 위치·크기를 mss 호환 dict로 반환."""
    r = window.rectangle()
    return {
        "left": r.left,
        "top": r.top,
        "width": r.right - r.left,
        "height": r.bottom - r.top,
    }


def _capture_window_png(rect: dict) -> Optional[bytes]:
    """mss로 창 영역을 캡처해 PNG bytes를 반환. mss 미설치 시 None."""
    if not MSS_AVAILABLE:
        return None
    with mss.mss() as sct:
        shot = sct.grab(rect)
        return mss.tools.to_png(shot.rgb, shot.size)


def _vlm_locate_controls(
    image_data: bytes,
    api_url: str,
    api_key: str,
    model_name: str,
) -> dict:
    """스크린샷을 VLM에 보내 UI 컨트롤 좌표(창 내 상대 픽셀)를 파싱해 반환.

    반환 형태::
        {
            "username_field": (x, y) | None,
            "password_field": (x, y) | None,
            "login_button":   (x, y) | None,
        }
    """
    import base64
    import json

    prompt = (
        "이 화면은 소프트웨어 로그인 창입니다.\n"
        "다음 UI 요소의 중심 픽셀 좌표를 JSON으로 반환해주세요.\n"
        "보이지 않는 요소는 null로 반환하세요.\n"
        '{"username_field": [x, y], "password_field": [x, y], "login_button": [x, y]}\n'
        "반드시 JSON만 반환하세요."
    )

    b64 = base64.b64encode(image_data).decode()
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    empty = {"username_field": None, "password_field": None, "login_button": None}
    try:
        resp = _requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        raw = json.loads(content)
    except Exception as exc:
        print(f"[WARN] VLM 응답 파싱 실패: {exc}")
        return empty

    result = {}
    for key in ("username_field", "password_field", "login_button"):
        val = raw.get(key)
        result[key] = tuple(val) if isinstance(val, (list, tuple)) and len(val) == 2 else None
    return result


def _vlm_login_fallback(login_window, username: str, password: str) -> bool:
    """UIA 실패 시 VLM 스크린샷 분석으로 자격증명을 입력하고 로그인을 시도.

    성공하면 True, 불가능하면 False를 반환.
    """
    api_url = os.environ.get("VLM_API_URL", "")
    api_key = os.environ.get("VLM_API_KEY", "")
    model_name = os.environ.get("VLM_MODEL_NAME", "")

    if not (api_url and model_name):
        print("[WARN] VLM API URL/모델 미설정 → VLM 폴백 불가")
        return False
    if not REQUESTS_AVAILABLE:
        print("[WARN] requests 미설치 → VLM 폴백 불가")
        return False
    if not MSS_AVAILABLE:
        print("[WARN] mss 미설치 → VLM 폴백 불가")
        return False
    if not PYNPUT_AVAILABLE:
        print("[WARN] pynput 미설치 → VLM 폴백 불가")
        return False

    rect = _window_rect(login_window)
    image_data = _capture_window_png(rect)
    if image_data is None:
        print("[WARN] 창 캡처 실패 → VLM 폴백 불가")
        return False

    print("[INFO] VLM으로 UI 컨트롤 좌표 탐색 중…")
    coords = _vlm_locate_controls(image_data, api_url, api_key, model_name)
    print(f"[INFO] VLM 좌표 결과: {coords}")

    mouse = _PynputMouse()
    keyboard = _PynputKeyboard()
    win_left, win_top = rect["left"], rect["top"]

    def _click_and_type(rel_coords, text: str) -> None:
        if rel_coords is None:
            return
        ax, ay = win_left + int(rel_coords[0]), win_top + int(rel_coords[1])
        mouse.position = (ax, ay)
        mouse.click(_MouseButton.left)
        time.sleep(0.1)
        # 기존 텍스트 전체 선택 후 삭제
        with keyboard.pressed(_Key.ctrl):
            keyboard.press("a")
            keyboard.release("a")
        keyboard.press(_Key.delete)
        keyboard.release(_Key.delete)
        time.sleep(0.05)
        keyboard.type(text)

    if username:
        _click_and_type(coords.get("username_field"), username)
        print("[INFO] VLM: User ID 입력 완료")
    if password:
        _click_and_type(coords.get("password_field"), password)
        print("[INFO] VLM: Password 입력 완료")

    login_btn = coords.get("login_button")
    if login_btn is not None:
        ax, ay = win_left + int(login_btn[0]), win_top + int(login_btn[1])
        mouse.position = (ax, ay)
        time.sleep(0.05)
        mouse.click(_MouseButton.left)
    else:
        try:
            login_window.type_keys("{ENTER}", set_foreground=False)
        except Exception:
            pass

    return True


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

    _uia_login_done = False
    try:
        edits = [e for e in login_window.descendants(control_type="Edit") if _is_visible(e)]
        edits = sorted(edits, key=lambda control: (control.rectangle().top, control.rectangle().left))
        if len(edits) >= 2:
            if args.username:
                ok = _set_edit_text(edits[0], args.username)
                print(f"[INFO] User ID 입력 (UIA): {'OK' if ok else 'FAIL'}")
            if args.password:
                ok = _set_edit_text(edits[1], args.password)
                print(f"[INFO] Password 입력 (UIA): {'OK' if ok else 'FAIL'}")
            _submit(login_window)
            print("[INFO] 로그인 제출 완료 (UIA)")
            _uia_login_done = True
        else:
            print(f"[INFO] UIA Edit {len(edits)}개 발견 (2개 필요) → VLM 폴백 시도")
    except Exception as exc:
        print(f"[WARN] UIA 입력 처리 중 오류: {exc}")

    if not _uia_login_done:
        vlm_ok = _vlm_login_fallback(login_window, args.username, args.password)
        if not vlm_ok:
            print("[WARN] VLM 폴백 실패. ENTER 키 전송.")
            try:
                login_window.type_keys("{ENTER}", set_foreground=False)
            except Exception:
                pass
        print("[INFO] 로그인 제출 완료 (VLM)")

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
