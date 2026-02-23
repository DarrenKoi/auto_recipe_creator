"""RCS 실행 후 VLM 기반으로 로그인 폼을 자동으로 채워 넣는 스크립트 (Windows 전용).

pywinauto는 창 실행·탐색에만 사용하고, 내부 컨트롤 조작은
스크린샷 → VLM 좌표 추출 → pynput 클릭/타이핑으로 수행한다.
"""

import base64
import json
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import mss
import mss.tools
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pynput.keyboard import Controller as KbdCtrl, Key
from pynput.mouse import Button as MouseButton, Controller as MouseCtrl
from pywinauto.application import Application

load_dotenv()

# ─────────────────────────── 설정 ───────────────────────────

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
SERVER = os.environ.get("RCS_SERVER", "Dropbox")
USERNAME = os.environ.get("RCS_USERNAME", "")
PASSWORD = os.environ.get("RCS_PASSWORD", "")
VLM_API_URL = (
    os.environ.get("VLM_API_URL", "").strip()
    or os.environ.get("VLM_API_BASE_URL", "").strip()
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen3-VL-30B-Instruct")

LAUNCH_TIMEOUT = 30.0
POST_LOGIN_WAIT = 6.0
WINDOW_TITLE_PREFIX = "Remote Control System"
ACTION_DELAY = 0.4

# 자격증명이 이미 입력되어 있으면 Log In 버튼만 클릭
CREDENTIALS_PREFILLED = True
# True이면 VLM 좌표 정확도 테스트만 실행 (클릭/타이핑 없이 디버그 이미지만 저장)
DEBUG_COORDINATE_TEST = True


# ─────────────────────────── 창 탐색 ───────────────────────────

def _wait_for_login_window(app):
    """'Remote Control System' 으로 시작하는 창이 나타날 때까지 대기."""
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


# ─────────────────────────── 스크린샷 ───────────────────────────

def _capture_window(window) -> Image.Image:
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
    rect = window.rectangle()
    region = {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_data))
    print(f"[INFO] 창 캡처 완료: {image.size[0]}x{image.size[1]} px")
    return image


# ─────────────────────────── VLM 호출 공통 ───────────────────────────

def _vlm_endpoint() -> str:
    """VLM chat completions 엔드포인트 URL을 반환한다."""
    base = VLM_API_URL.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _call_vlm(system_msg: str, prompt: str, img_b64: str) -> str:
    """VLM API를 호출하고 응답 텍스트를 반환한다."""
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    endpoint = _vlm_endpoint()
    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
        "temperature": 0.1,
    }

    print(f"[INFO] VLM API 호출 중... ({endpoint})")
    start = time.time()
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    print(f"[INFO] VLM 응답 수신 ({(time.time() - start) * 1000:.0f}ms)")
    return raw


def _encode_image(image: Image.Image) -> tuple[str, int, int]:
    """PIL Image를 base64 PNG로 인코딩하고 (b64, w, h)를 반환한다."""
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    w, h = image.size
    print(f"[INFO] VLM 전송 이미지: {w}x{h}, {len(buf.getvalue()) / 1024:.1f}KB")
    return b64, w, h


def _extract_json(text: str) -> dict:
    """VLM 응답 텍스트에서 JSON을 추출한다."""
    if "```json" in text:
        s = text.find("```json") + 7
        e = text.find("```", s)
        if e != -1:
            return json.loads(text[s:e].strip())
    if "{" in text:
        s = text.find("{")
        e = text.rfind("}")
        if e > s:
            return json.loads(text[s : e + 1])
    return json.loads(text)


def _normalize_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """좌표값을 정규화(0~1) → 픽셀 변환하고 범위를 클램핑한다."""
    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"[WARNING] VLM 응답에 '{key}' 누락")
            continue
        x, y = pt.get("x", 0), pt.get("y", 0)
        if isinstance(x, float) and 0 <= x <= 1.0 and isinstance(y, float) and 0 <= y <= 1.0:
            x, y = int(x * img_w), int(y * img_h)
        data[key] = {"x": max(0, min(int(x), img_w)), "y": max(0, min(int(y), img_h))}
    return data


# ─────────────────────── VLM 좌표 정확도 테스트 ───────────────────────

def _run_coordinate_debug(window) -> None:
    """VLM 좌표 정확도 테스트. 클릭/타이핑 없이 디버그 이미지만 저장."""
    print("[INFO] ===== VLM 좌표 정확도 테스트 모드 =====")
    image = _capture_window(window)
    img_b64, w, h = _encode_image(image)

    rect = window.rectangle()
    print(f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
          f"right={rect.right}, bottom={rect.bottom}, "
          f"size={rect.right - rect.left}x{rect.bottom - rect.top}")

    prompt = f"""You are a GUI screen analysis expert.

This image shows a Remote Control System login dialog.
Find the exact center pixel coordinate of each visible text label listed below.
I need the coordinate of the TEXT ITSELF (the rendered characters), not any input field or button area next to it.

Target texts:
1. "Server" — the label text
2. "User ID" — the label text
3. "Password" — the label text
4. "Log In" — the button text

Image resolution: {w}x{h} pixels
Coordinate range: x is 0~{w}, y is 0~{h}

Respond ONLY with this JSON format:
{{
    "Server": {{"x": integer, "y": integer}},
    "User ID": {{"x": integer, "y": integer}},
    "Password": {{"x": integer, "y": integer}},
    "Log In": {{"x": integer, "y": integer}}
}}"""

    system_msg = (
        f"You are a GUI coordinate extraction agent. "
        f"Image resolution is {w}x{h} pixels. "
        f"Return coordinates as pixel values in range 0~{w}(x), 0~{h}(y). "
        f"Respond ONLY in JSON format."
    )

    raw = _call_vlm(system_msg, prompt, img_b64)
    print(f"[INFO] VLM 원문 응답:\n{raw}")

    data = _extract_json(raw)
    labels = ["Server", "User ID", "Password", "Log In"]
    data = _normalize_coords(data, labels, w, h)

    for name in labels:
        pt = data.get(name)
        if not pt:
            continue
        x, y = pt["x"], pt["y"]
        in_bounds = 0 <= x <= w and 0 <= y <= h
        print(f"[INFO] '{name}': ({x}, {y}) — {'OK' if in_bounds else 'OUT OF BOUNDS'}")

    # 디버그 이미지 저장
    colors = {"Server": "red", "User ID": "blue", "Password": "green", "Log In": "orange"}
    _save_marked_image(image, data, colors, "debug_vlm_coords.png")
    print("[INFO] ===== 좌표 테스트 완료 — debug_vlm_coords.png 확인 =====")


# ─────────────────────────── 디버그 이미지 ───────────────────────────

def _save_marked_image(image: Image.Image, elements: dict, colors: dict, filename: str) -> None:
    """좌표를 원본 스크린샷 위에 십자선+원으로 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    r = 15
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        x, y = int(pt["x"]), int(pt["y"])
        color = colors.get(name, "white")
        draw.line([(x - r, y), (x + r, y)], fill=color, width=3)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=3)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=3)
        draw.text((x + r + 4, y - 8), f"{name} ({x},{y})", fill=color, font=font)

    out_path = Path(__file__).parent / filename
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


# ─────────────────────────── 클릭·타이핑 ───────────────────────────

def _click(abs_x: int, abs_y: int) -> None:
    """절대 좌표 클릭."""
    mouse = MouseCtrl()
    mouse.position = (abs_x, abs_y)
    time.sleep(0.1)
    mouse.click(MouseButton.left)
    time.sleep(ACTION_DELAY)


def _click_and_type(abs_x: int, abs_y: int, text: str) -> None:
    """절대 좌표로 클릭한 뒤 기존 내용을 지우고 타이핑한다."""
    _click(abs_x, abs_y)
    kbd = KbdCtrl()
    kbd.press(Key.ctrl)
    kbd.press("a")
    kbd.release("a")
    kbd.release(Key.ctrl)
    time.sleep(0.05)
    kbd.type(text)
    time.sleep(ACTION_DELAY)


# ─────────────────────────── VLM 로그인 ───────────────────────────

def _vlm_login(window) -> bool:
    """VLM 좌표 기반으로 로그인 폼을 채우고 Log In 클릭."""
    image = _capture_window(window)
    img_b64, w, h = _encode_image(image)

    system_msg = (
        f"당신은 GUI 자동화 에이전트입니다. "
        f"이 이미지의 해상도는 {w}x{h} 픽셀입니다. "
        f"좌표는 반드시 0~{w}(x), 0~{h}(y) 범위의 픽셀 값으로 반환하세요. "
        f"반드시 JSON 형식으로만 응답하세요."
    )

    if CREDENTIALS_PREFILLED:
        prompt = f"""당신은 GUI 화면 분석 전문가입니다.

이 이미지는 Remote Control System 로그인 화면입니다.
자격증명(Server, User ID, Password)은 이미 입력되어 있습니다.
"Log In" 버튼의 정확한 중심점 좌표만 찾아 주세요.

이미지 해상도: {w}x{h} 픽셀
좌표 범위: x는 0~{w}, y는 0~{h}

반드시 다음 JSON 형식으로만 응답하세요:
{{
    "login_button": {{"x": 정수, "y": 정수}}
}}"""
        keys = ["login_button"]
    else:
        prompt = f"""당신은 GUI 화면 분석 전문가입니다.

이 이미지는 Remote Control System 로그인 화면입니다.
화면 레이아웃: 왼쪽에 "Server", "User ID", "Password" 텍스트 라벨이 있고,
각 라벨의 오른쪽에 흰색 배경의 콤보박스/입력 필드가 위치합니다.

다음 4개 UI 요소의 좌표를 찾아 주세요.
중요: 라벨 텍스트가 아닌, 라벨 오른쪽에 있는 흰색 입력 영역의 좌측 1/3 지점을 클릭 좌표로 잡아 주세요.

1. server — "Server" 오른쪽 드롭다운 콤보박스 (흰색 영역의 왼쪽 1/3, 세로 중심)
2. user_id — "User ID" 오른쪽 텍스트 입력 필드 (흰색 영역의 왼쪽 1/3, 세로 중심)
3. password — "Password" 오른쪽 텍스트 입력 필드 (흰색 영역의 왼쪽 1/3, 세로 중심)
4. login_button — "Log In" 버튼 중심

이미지 해상도: {w}x{h} 픽셀
좌표 범위: x는 0~{w}, y는 0~{h}

반드시 다음 JSON 형식으로만 응답하세요:
{{
    "server": {{"x": 정수, "y": 정수}},
    "user_id": {{"x": 정수, "y": 정수}},
    "password": {{"x": 정수, "y": 정수}},
    "login_button": {{"x": 정수, "y": 정수}}
}}"""
        keys = ["server", "user_id", "password", "login_button"]

    raw = _call_vlm(system_msg, prompt, img_b64)
    data = _extract_json(raw)
    data = _normalize_coords(data, keys, w, h)
    print(f"[INFO] VLM 좌표: {json.dumps(data, indent=2)}")

    # 디버그 이미지 저장
    colors = {"server": "red", "user_id": "blue", "password": "green", "login_button": "orange"}
    _save_marked_image(image, data, colors, "debug_vlm_login.png")

    # 좌표 변환: 스크린샷 좌표 → 절대 스크린 좌표
    rect = window.rectangle()

    def to_abs(pt):
        return int(pt["x"]) + rect.left, int(pt["y"]) + rect.top

    if not CREDENTIALS_PREFILLED:
        if "server" in data and SERVER:
            sx, sy = to_abs(data["server"])
            print(f"[INFO] 서버 드롭다운 클릭: ({sx}, {sy})")
            _click(sx, sy)
            time.sleep(0.3)
            kbd = KbdCtrl()
            kbd.type(SERVER)
            time.sleep(0.2)
            kbd.press(Key.enter)
            kbd.release(Key.enter)
            time.sleep(ACTION_DELAY)

        if "user_id" in data and USERNAME:
            ux, uy = to_abs(data["user_id"])
            print(f"[INFO] User ID 필드 클릭: ({ux}, {uy})")
            _click_and_type(ux, uy, USERNAME)

        if "password" in data and PASSWORD:
            px, py = to_abs(data["password"])
            print(f"[INFO] Password 필드 클릭: ({px}, {py})")
            _click_and_type(px, py, PASSWORD)

    # Log In 버튼
    if "login_button" in data:
        lx, ly = to_abs(data["login_button"])
        print(f"[INFO] Log In 버튼 좌표: ({lx}, {ly})")
        print(f"[INFO] 창 영역: left={rect.left}, top={rect.top}, right={rect.right}, bottom={rect.bottom}")
        if not (rect.left <= lx <= rect.right and rect.top <= ly <= rect.bottom):
            print(f"[WARNING] Log In 좌표가 창 영역 밖입니다!")
        try:
            window.set_focus()
            time.sleep(0.3)
        except Exception as exc:
            print(f"[WARNING] 창 포커스 실패: {exc}")
        _click(lx, ly)
        print("[INFO] Log In 버튼 클릭 완료")
    else:
        print("[WARNING] login_button 좌표 없음, ENTER 전송")
        kbd = KbdCtrl()
        kbd.press(Key.enter)
        kbd.release(Key.enter)

    return True


# ─────────────────────────── 메인 ───────────────────────────

def main() -> int:
    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return 1

    print(f"[INFO] RCS 시작: {RCS_EXE}")
    cmd_str = subprocess.list2cmdline([str(RCS_EXE)])
    app = Application(backend="win32").start(cmd_str, wait_for_idle=False)

    try:
        login_window = _wait_for_login_window(app)
        print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    time.sleep(1.0)

    if DEBUG_COORDINATE_TEST:
        _run_coordinate_debug(login_window)
        print("[INFO] 디버그 모드 — 로그인 시도 없이 종료합니다")
        return 0

    if not _vlm_login(login_window):
        print("[ERROR] VLM 기반 로그인 실패")
        return 5

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
