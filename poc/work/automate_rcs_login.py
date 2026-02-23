"""RCS 실행 후 VLM 기반으로 로그인 폼을 자동으로 채워 넣는 스크립트 (Windows 전용).

pywinauto는 창 실행·탐색에만 사용하고, 내부 컨트롤 조작은
스크린샷 → VLM 좌표 추출 → pynput 클릭/타이핑으로 수행한다.
"""

import base64
import json
import os
import platform
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
LAUNCH_TIMEOUT = 30.0
POST_LOGIN_WAIT = 6.0
WINDOW_TITLE_PREFIX = "Remote Control System"
ACTION_DELAY = 0.4
# 자격증명이 이미 입력되어 있으면 Log In 버튼만 클릭
CREDENTIALS_PREFILLED = True


# ─────────────────────────── DPI / 모니터 ───────────────────────────

def _ensure_dpi_aware() -> None:
    """Windows에서 DPI Awareness를 설정하여 mss가 물리 픽셀로 캡처하도록 한다."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        print("[INFO] DPI Awareness 설정 완료")
    except Exception as exc:
        print(f"[WARNING] DPI Awareness 설정 실패: {exc}")


def _log_monitor_info() -> None:
    """연결된 모니터 목록과 좌표 오프셋을 출력한다 (디버그용)."""
    if not MSS_AVAILABLE:
        return
    with mss.mss() as sct:
        monitors = sct.monitors
    print(f"[INFO] 감지된 모니터: {len(monitors) - 1}대")
    for i, mon in enumerate(monitors):
        label = "가상 전체" if i == 0 else f"모니터 {i}"
        print(
            f"  [{i}] {label}: "
            f"({mon['left']}, {mon['top']}) "
            f"{mon['width']}x{mon['height']}"
        )


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
    print(
        f"[INFO] 창 영역 (pywinauto): "
        f"left={rect.left}, top={rect.top}, "
        f"right={rect.right}, bottom={rect.bottom}, "
        f"size={region['width']}x{region['height']}"
    )

    with mss.mss() as sct:
        shot = sct.grab(region)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    image = Image.open(BytesIO(png_data))
    cap_w, cap_h = image.size
    # 캡처된 이미지 크기와 pywinauto 영역 크기가 다르면 DPI 스케일링 의심
    if cap_w != region["width"] or cap_h != region["height"]:
        print(
            f"[WARNING] 캡처 크기 불일치! "
            f"pywinauto: {region['width']}x{region['height']}, "
            f"mss 캡처: {cap_w}x{cap_h} — DPI 스케일링 가능성"
        )
    print(f"[INFO] 창 캡처 완료: {cap_w}x{cap_h} px")
    return image


# ─────────────────────────── VLM 좌표 추출 ───────────────────────────

def _ask_vlm_login_elements(image: "Image.Image") -> "tuple[dict, dict] | None":
    """VLM에 로그인 화면의 UI 요소 좌표를 질의한다.

    Returns:
        (processed, raw) 튜플 또는 None (실패 시).
        processed: 정규화/클램핑 후 좌표 dict
        raw: VLM이 반환한 원본 좌표 dict (디버깅용)
    """
    if not REQUESTS_AVAILABLE:
        print("[ERROR] requests 라이브러리가 필요합니다")
        return None

    w, h = image.size

    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[INFO] VLM 전송 이미지: {w}x{h}, PNG, {len(buf.getvalue()) / 1024:.1f}KB")

    if CREDENTIALS_PREFILLED:
        # 자격증명이 이미 입력됨 — Log In 버튼 좌표만 필요
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
    else:
        prompt = f"""당신은 GUI 화면 분석 전문가입니다.

이 이미지는 Remote Control System 로그인 화면입니다.
화면 레이아웃: 왼쪽에 "Server", "User ID", "Password" 텍스트 라벨이 있고,
각 라벨의 오른쪽에 흰색 배경의 콤보박스/입력 필드가 위치합니다.

다음 4개 UI 요소의 좌표를 찾아 주세요.
중요: 라벨 텍스트("Server", "User ID" 등)가 아닌, 라벨 오른쪽에 있는 흰색 입력 영역의 좌측 1/3 지점을 클릭 좌표로 잡아 주세요.
입력 필드의 세로 중심, 가로는 입력 영역의 왼쪽에서 약 1/3 지점이 이상적입니다.

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


def _parse_elements_json(text: str, img_w: int, img_h: int) -> "tuple[dict, dict] | None":
    """VLM 응답에서 UI 요소 좌표 JSON을 파싱한다.

    Returns:
        (processed, raw) 튜플 — processed는 정규화/클램핑 후 좌표,
        raw는 VLM이 반환한 원본 좌표. 실패 시 None.
    """
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
        raw = {}  # VLM 원본 좌표 보존

        # 정규화 좌표 감지 (0~1) → 픽셀 변환
        for key in ("server", "user_id", "password", "login_button"):
            pt = data.get(key)
            if not pt:
                print(f"[WARNING] VLM 응답에 '{key}' 누락")
                continue
            x, y = pt.get("x", 0), pt.get("y", 0)
            raw[key] = {"x": x, "y": y}  # 변환 전 원본 저장
            if isinstance(x, float) and 0 <= x <= 1.0 and isinstance(y, float) and 0 <= y <= 1.0:
                x, y = int(x * img_w), int(y * img_h)
            x = max(0, min(int(x), img_w))
            y = max(0, min(int(y), img_h))
            data[key] = {"x": x, "y": y}

        print(f"[INFO] VLM 원본(raw) 좌표: {json.dumps(raw, indent=2)}")
        print(f"[INFO] VLM 변환(processed) 좌표: {json.dumps(data, indent=2)}")
        return data, raw

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


def _save_debug_screenshot(
    image: "Image.Image", elements: dict, raw_elements: "dict | None" = None
) -> None:
    """VLM이 반환한 좌표를 원본 스크린샷 위에 마킹하여 저장한다.

    processed 좌표는 십자선+원으로, raw 원본 좌표는 사각형으로 표시한다.
    """
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

    # raw 원본 좌표 먼저 그리기 (사각형, 연한 색)
    if raw_elements:
        for name, pt in raw_elements.items():
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                continue
            rx, ry = pt["x"], pt["y"]
            # 정규화 좌표(0~1)인 경우 픽셀로 변환하여 표시
            img_w, img_h = image.size
            if isinstance(rx, float) and 0 <= rx <= 1.0 and isinstance(ry, float) and 0 <= ry <= 1.0:
                rx_px, ry_px = int(rx * img_w), int(ry * img_h)
                raw_label = f"raw:{name} ({pt['x']:.4f},{pt['y']:.4f})→({rx_px},{ry_px})"
            else:
                rx_px, ry_px = int(rx), int(ry)
                raw_label = f"raw:{name} ({rx_px},{ry_px})"
            color = ELEMENT_COLORS.get(name, "white")

            # 사각형 마커 (raw 좌표 표시용)
            draw.rectangle(
                [(rx_px - marker_r, ry_px - marker_r), (rx_px + marker_r, ry_px + marker_r)],
                outline=color, width=1,
            )
            # raw 라벨 (위쪽에 표시)
            draw.text((rx_px + marker_r + 4, ry_px - 24), raw_label, fill=color, font=font)

    # processed 좌표 (십자선 + 원)
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        sx = int(pt["x"])
        sy = int(pt["y"])
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

def _refocus_window(window) -> None:
    """창 타이틀바를 클릭하여 창을 다시 전면으로 가져온다.

    set_focus()가 레거시 앱에서 불안정할 수 있으므로,
    타이틀바 영역을 직접 클릭하여 확실하게 포커스를 확보한다.
    """
    try:
        window.set_focus()
        time.sleep(0.2)
    except Exception as exc:
        print(f"[WARNING] set_focus 실패: {exc}")

    try:
        rect = window.rectangle()
        # 타이틀바 중심 클릭 (상단에서 약 15px 아래)
        title_x = (rect.left + rect.right) // 2
        title_y = rect.top + 15
        print(f"[INFO] 타이틀바 클릭으로 창 재활성화: ({title_x}, {title_y})")
        _click(title_x, title_y)
        time.sleep(0.3)
    except Exception as exc:
        print(f"[WARNING] 타이틀바 클릭 실패: {exc}")


def _detect_dpi_scale(window) -> float:
    """mss 캡처 크기와 pywinauto 창 크기를 비교하여 DPI 스케일을 추정한다.

    pywinauto(win32)는 논리 좌표를, mss는 물리 픽셀을 반환하므로
    두 값이 다르면 DPI 스케일링이 적용된 환경이다.
    """
    rect = window.rectangle()
    logical_w = rect.right - rect.left
    logical_h = rect.bottom - rect.top

    if not MSS_AVAILABLE or logical_w <= 0:
        return 1.0

    region = {
        "left": rect.left, "top": rect.top,
        "width": logical_w, "height": logical_h,
    }
    with mss.mss() as sct:
        shot = sct.grab(region)
    physical_w = shot.size.width

    if physical_w == logical_w:
        return 1.0

    scale = physical_w / logical_w
    print(
        f"[INFO] DPI 스케일 감지: {scale:.2f}x "
        f"(논리 {logical_w}px → 물리 {physical_w}px)"
    )
    return scale


def _vlm_login(window) -> bool:
    """VLM 좌표 기반으로 로그인 폼을 채우고 Log In 클릭."""
    if not PYNPUT_AVAILABLE:
        print("[ERROR] pynput 라이브러리가 필요합니다")
        return False

    # 1) DPI 스케일 감지 (mss 물리 픽셀 vs pywinauto 논리 좌표)
    dpi_scale = _detect_dpi_scale(window)

    # 2) 창 캡처 (mss — 물리 픽셀)
    image = _capture_window(window)
    if image is None:
        return False

    # 3) VLM 질의
    result = _ask_vlm_login_elements(image)
    if result is None:
        return False
    elements, raw_elements = result

    # 4) 디버그: VLM 원본(raw) + 변환(processed) 좌표를 스크린샷 위에 마킹
    _save_debug_screenshot(image, elements, raw_elements)

    # 5) 좌표 변환 함수
    #    VLM 좌표는 mss 캡처 이미지(물리 픽셀) 기준이다.
    #    pynput은 논리 좌표를 사용하므로 DPI 보정이 필요하다.
    #    체인: VLM 좌표(물리px) ÷ dpi_scale → 논리px + win_offset → 절대 마우스 좌표
    rect = window.rectangle()

    def to_abs(pt):
        """VLM 좌표(물리 픽셀) → 절대 마우스 좌표(논리 좌표)."""
        phys_x = int(pt["x"])
        phys_y = int(pt["y"])
        # 물리 → 논리 변환
        local_x = int(phys_x / dpi_scale)
        local_y = int(phys_y / dpi_scale)
        # 창 오프셋 (pywinauto rect는 이미 논리 좌표)
        cur_rect = window.rectangle()
        abs_x = local_x + cur_rect.left
        abs_y = local_y + cur_rect.top
        if dpi_scale != 1.0:
            print(
                f"[DEBUG] 좌표 변환: 물리({phys_x},{phys_y}) "
                f"÷{dpi_scale:.2f} → 논리({local_x},{local_y}) "
                f"+ offset({cur_rect.left},{cur_rect.top}) "
                f"= 절대({abs_x},{abs_y})"
            )
        return abs_x, abs_y

    print(
        f"[INFO] 좌표 변환 정보: dpi_scale={dpi_scale:.2f}, "
        f"창 offset=({rect.left},{rect.top})"
    )

    if CREDENTIALS_PREFILLED:
        print("[INFO] 자격증명이 이미 입력됨 — Log In 버튼만 클릭합니다")
    else:
        # 6) 서버 선택
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

        # 7) User ID
        if "user_id" in elements and USERNAME:
            ux, uy = to_abs(elements["user_id"])
            print(f"[INFO] User ID 필드 클릭: ({ux}, {uy})")
            _click_and_type(ux, uy, USERNAME)
            print("[INFO] User ID 입력 완료")

        # 8) Password
        if "password" in elements and PASSWORD:
            px, py = to_abs(elements["password"])
            print(f"[INFO] Password 필드 클릭: ({px}, {py})")
            _click_and_type(px, py, PASSWORD)
            print("[INFO] Password 입력 완료")

    # 9) Log In 버튼 — 클릭 전 창을 타이틀바 클릭으로 재활성화
    if "login_button" in elements:
        print("[INFO] 로그인 클릭 전 창 재활성화 중...")
        _refocus_window(window)
        # 창이 이동했을 수 있으므로 to_abs가 최신 rect를 참조
        lx, ly = to_abs(elements["login_button"])
        cur_rect = window.rectangle()
        print(f"[INFO] Log In 버튼 좌표: ({lx}, {ly})")
        print(f"[INFO] 창 영역: left={cur_rect.left}, top={cur_rect.top}, right={cur_rect.right}, bottom={cur_rect.bottom}")
        # 좌표가 창 영역 밖이면 경고
        if not (cur_rect.left <= lx <= cur_rect.right and cur_rect.top <= ly <= cur_rect.bottom):
            print(f"[WARNING] Log In 좌표가 창 영역 밖입니다!")
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
    _ensure_dpi_aware()
    _log_monitor_info()

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
