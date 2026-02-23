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

from dotenv import load_dotenv
from pywinauto.application import Application

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from pynput.mouse import Button as MouseButton, Controller as MouseCtrl
    from pynput.keyboard import Controller as KbdCtrl
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

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
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen3-VL-30B-Instruct")
try:
    VLM_CHECK_TIMEOUT = float(os.environ.get("VLM_CHECK_TIMEOUT", "3.0"))
except ValueError:
    print("[WARN] VLM_CHECK_TIMEOUT 값이 유효하지 않아 3.0초로 대체합니다.")
    VLM_CHECK_TIMEOUT = 3.0
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "1280"))
LAUNCH_TIMEOUT = 30.0
POST_LOGIN_WAIT = 6.0
WINDOW_TITLE_PREFIX = "Remote Control System"
ACTION_DELAY = 0.4


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


# ─────────────────────────── VLM 헬스체크 ───────────────────────────

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


# ─────────────────────────── 스크린샷 ───────────────────────────

def _capture_window(window) -> "Image.Image | None":
    """pywinauto 창 영역을 mss로 캡처하여 PIL Image로 반환한다."""
    if not MSS_AVAILABLE or not PIL_AVAILABLE:
        print("[ERROR] mss 또는 Pillow 라이브러리가 필요합니다")
        return None

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


def _resize_for_vlm(image: "Image.Image") -> "tuple[Image.Image, float]":
    """VLM 입력용 리사이즈. (resized_image, scale) 반환."""
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= MAX_IMAGE_SIZE:
        return image.copy(), 1.0
    scale = MAX_IMAGE_SIZE / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    print(f"[INFO] VLM용 리사이즈: {w}x{h} → {new_w}x{new_h} (scale={scale:.4f})")
    return resized, scale


# ─────────────────────────── VLM 좌표 추출 ───────────────────────────

def _ask_vlm_login_elements(image: "Image.Image") -> "dict | None":
    """VLM에 로그인 화면의 UI 요소 좌표를 질의한다.

    Returns:
        {
            "server": {"x": int, "y": int},
            "user_id": {"x": int, "y": int},
            "password": {"x": int, "y": int},
            "login_button": {"x": int, "y": int}
        }
        또는 None (실패 시)
    """
    if not REQUESTS_AVAILABLE:
        print("[ERROR] requests 라이브러리가 필요합니다")
        return None

    w, h = image.size

    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[INFO] VLM 전송 이미지: {w}x{h}, PNG, {len(buf.getvalue()) / 1024:.1f}KB")

    prompt = f"""당신은 GUI 화면 분석 전문가입니다.

이 이미지는 Remote Control System 로그인 화면입니다.
화면 레이아웃: 왼쪽에 "Server", "User ID", "Password" 텍스트 라벨이 있고,
각 라벨의 오른쪽에 콤보박스/입력 필드가 위치합니다. Log In 버튼도 오른쪽에 있습니다.

다음 4개 UI 요소의 **입력 가능한 컨트롤** 중심점 좌표를 찾아 주세요 (라벨 텍스트가 아닌, 그 오른쪽의 실제 입력 영역):

1. server — "Server" 라벨 오른쪽의 드롭다운 콤보박스 중심
2. user_id — "User ID" 라벨 오른쪽의 텍스트 입력 필드 중심
3. password — "Password" 라벨 오른쪽의 텍스트 입력 필드 중심
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

    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    api_base = VLM_API_URL.rstrip("/")
    endpoint = f"{api_base}/v1/chat/completions" if not api_base.endswith("/v1") else f"{api_base}/chat/completions"

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 GUI 자동화 에이전트입니다. "
                    f"이 이미지의 해상도는 {w}x{h} 픽셀입니다. "
                    f"좌표는 반드시 0~{w}(x), 0~{h}(y) 범위의 픽셀 값으로 반환하세요. "
                    "반드시 JSON 형식으로만 응답하세요."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.1,
    }

    try:
        print(f"[INFO] VLM API 호출 중... ({endpoint})")
        start = time.time()
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        elapsed = (time.time() - start) * 1000
        raw = resp.json()["choices"][0]["message"]["content"]
        print(f"[INFO] VLM 응답 수신 ({elapsed:.0f}ms)")
        return _parse_elements_json(raw, w, h)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] VLM 서버 연결 실패: {endpoint}")
    except requests.exceptions.Timeout:
        print("[ERROR] VLM 요청 타임아웃 (60초)")
    except Exception as exc:
        print(f"[ERROR] VLM API 호출 실패: {exc}")
    return None


def _parse_elements_json(text: str, img_w: int, img_h: int) -> "dict | None":
    """VLM 응답에서 UI 요소 좌표 JSON을 파싱한다."""
    try:
        json_str = text
        if "```json" in text:
            s = text.find("```json") + 7
            e = text.find("```", s)
            if e != -1:
                json_str = text[s:e].strip()
        elif "{" in text:
            s = text.find("{")
            e = text.rfind("}")
            if e > s:
                json_str = text[s : e + 1]

        data = json.loads(json_str)

        # 정규화 좌표 감지 (0~1) → 픽셀 변환
        for key in ("server", "user_id", "password", "login_button"):
            pt = data.get(key)
            if not pt:
                print(f"[WARNING] VLM 응답에 '{key}' 누락")
                continue
            x, y = pt.get("x", 0), pt.get("y", 0)
            if isinstance(x, float) and 0 <= x <= 1.0 and isinstance(y, float) and 0 <= y <= 1.0:
                x, y = int(x * img_w), int(y * img_h)
            x = max(0, min(int(x), img_w))
            y = max(0, min(int(y), img_h))
            data[key] = {"x": x, "y": y}

        print(f"[INFO] VLM 좌표 파싱 완료: {json.dumps(data, indent=2)}")
        return data

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[ERROR] VLM 응답 파싱 실패: {exc}")
        print(f"[DEBUG] 원문: {text[:500]}")
        return None


# ─────────────────────────── 디버그 스크린샷 ───────────────────────────

ELEMENT_COLORS = {
    "server": "red",
    "user_id": "blue",
    "password": "green",
    "login_button": "orange",
}


def _save_debug_screenshot(image: "Image.Image", elements: dict, scale: float) -> None:
    """VLM이 반환한 좌표를 원본 스크린샷 위에 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()

    marker_r = 12

    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        # VLM 좌표(리사이즈) → 원본 스크린샷 좌표
        sx = int(pt["x"] / scale)
        sy = int(pt["y"] / scale)
        color = ELEMENT_COLORS.get(name, "white")

        # 십자선
        draw.line([(sx - marker_r, sy), (sx + marker_r, sy)], fill=color, width=2)
        draw.line([(sx, sy - marker_r), (sx, sy + marker_r)], fill=color, width=2)
        # 원
        draw.ellipse(
            [(sx - marker_r, sy - marker_r), (sx + marker_r, sy + marker_r)],
            outline=color, width=2,
        )
        # 라벨
        label = f"{name} ({sx},{sy})"
        draw.text((sx + marker_r + 4, sy - 8), label, fill=color, font=font)

    out_dir = Path(__file__).parent
    out_path = out_dir / "debug_vlm_login.png"
    debug_img.save(out_path)
    print(f"[INFO] 디버그 스크린샷 저장: {out_path}")


# ─────────────────────────── 클릭·타이핑 ───────────────────────────

def _click_and_type(abs_x: int, abs_y: int, text: str = "") -> None:
    """절대 좌표로 클릭한 뒤, text가 있으면 타이핑한다."""
    mouse = MouseCtrl()
    kbd = KbdCtrl()

    mouse.position = (abs_x, abs_y)
    time.sleep(0.1)
    mouse.click(MouseButton.left)
    time.sleep(ACTION_DELAY)

    if text:
        # 기존 내용 전체 선택 후 덮어쓰기
        from pynput.keyboard import Key
        kbd.press(Key.ctrl)
        kbd.press("a")
        kbd.release("a")
        kbd.release(Key.ctrl)
        time.sleep(0.05)
        kbd.type(text)
        time.sleep(ACTION_DELAY)


def _click(abs_x: int, abs_y: int) -> None:
    """절대 좌표 클릭만 수행한다."""
    mouse = MouseCtrl()
    mouse.position = (abs_x, abs_y)
    time.sleep(0.1)
    mouse.click(MouseButton.left)
    time.sleep(ACTION_DELAY)


# ─────────────────────────── VLM 로그인 ───────────────────────────

def _vlm_login(window) -> bool:
    """VLM 좌표 기반으로 로그인 폼을 채우고 Log In 클릭."""
    if not PYNPUT_AVAILABLE:
        print("[ERROR] pynput 라이브러리가 필요합니다")
        return False

    # 1) 창 캡처
    image = _capture_window(window)
    if image is None:
        return False

    # 2) 리사이즈 + VLM 질의
    resized, scale = _resize_for_vlm(image)
    elements = _ask_vlm_login_elements(resized)
    if elements is None:
        return False

    # 3) 디버그: VLM 좌표를 원본 스크린샷 위에 마킹하여 저장
    _save_debug_screenshot(image, elements, scale)

    # 4) 좌표 변환: VLM(리사이즈) → 스크린샷 → 절대 스크린 좌표
    rect = window.rectangle()
    win_left, win_top = rect.left, rect.top

    def to_abs(pt):
        """VLM 좌표 → 절대 스크린 좌표."""
        sx = int(pt["x"] / scale)  # 스크린샷 좌표
        sy = int(pt["y"] / scale)
        return sx + win_left, sy + win_top

    # 5) 서버 선택
    if "server" in elements and SERVER:
        sx, sy = to_abs(elements["server"])
        print(f"[INFO] 서버 드롭다운 클릭: ({sx}, {sy})")
        _click(sx, sy)
        time.sleep(0.3)
        # 드롭다운이 열린 후 서버명 타이핑 + Enter
        kbd = KbdCtrl()
        kbd.type(SERVER)
        time.sleep(0.2)
        from pynput.keyboard import Key
        kbd.press(Key.enter)
        kbd.release(Key.enter)
        time.sleep(ACTION_DELAY)
        print(f"[INFO] 서버 선택: {SERVER}")

    # 6) User ID
    if "user_id" in elements and USERNAME:
        ux, uy = to_abs(elements["user_id"])
        print(f"[INFO] User ID 필드 클릭: ({ux}, {uy})")
        _click_and_type(ux, uy, USERNAME)
        print("[INFO] User ID 입력 완료")

    # 7) Password
    if "password" in elements and PASSWORD:
        px, py = to_abs(elements["password"])
        print(f"[INFO] Password 필드 클릭: ({px}, {py})")
        _click_and_type(px, py, PASSWORD)
        print("[INFO] Password 입력 완료")

    # 8) Log In 버튼
    if "login_button" in elements:
        lx, ly = to_abs(elements["login_button"])
        print(f"[INFO] Log In 버튼 클릭: ({lx}, {ly})")
        _click(lx, ly)
        print("[INFO] Log In 버튼 클릭 완료")
    else:
        print("[WARNING] login_button 좌표 없음, ENTER 전송")
        from pynput.keyboard import Key
        kbd = KbdCtrl()
        kbd.press(Key.enter)
        kbd.release(Key.enter)

    return True


# ─────────────────────────── 메인 ───────────────────────────

def main() -> int:
    if not _check_vlm_responsive():
        return 4

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

    time.sleep(1.0)  # 컨트롤 렌더링 대기

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
