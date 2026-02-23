"""RCS 로그인 화면에서 VLM 좌표 검출 정확도를 비교하는 벤치마크 스크립트 (Windows 전용).

여러 VLM 모델(Kimi-K2.5, Qwen3-VL-30B)에 동일한 스크린샷을 전송하여
텍스트 라벨·입력 필드(콤보박스/텍스트)·버튼 3개의 좌표를 추출하고, 모델별 디버그 이미지를 저장한다.
pywinauto는 창 실행·탐색에만 사용한다.
"""

import base64
import json
import os
import struct
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
from pywinauto.application import Application

load_dotenv()

# ─────────────────────────── 설정 ───────────────────────────

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
VLM_API_URL = (
    os.environ.get("VLM_API_URL", "").strip()
    or os.environ.get("VLM_API_BASE_URL", "").strip()
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "win32"

LAUNCH_TIMEOUT = 30.0
WINDOW_TITLE_PREFIX = "Remote Control System"

# 테스트할 VLM 모델 목록
TEST_MODELS = [
    "Kimi-K2.5",
    "Qwen3-VL-30B-Instruct",
]

# 검출 대상: 텍스트 라벨 + 입력 필드 + 버튼
TARGET_ELEMENTS = [
    "server_label",
    "server_input",
    "userid_label",
    "userid_input",
    "password_label",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
]

ELEMENT_COLORS = {
    "server_label": "red",
    "server_input": "salmon",
    "userid_label": "blue",
    "userid_input": "deepskyblue",
    "password_label": "green",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}

INPUT_X_OFFSET = 12
INPUT_X_OFFSET_KEYS = {
    "server_input",
    "userid_input",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
}


# ─────────────────────────── 창 탐색 ───────────────────────────


def _python_bitness() -> int:
    """현재 Python 인터프리터의 비트 수를 반환한다."""
    return 64 if sys.maxsize > 2**32 else 32


def _exe_bitness(exe_path: Path) -> int | None:
    """PE 헤더를 읽어 실행 파일 비트 수를 판별한다."""
    try:
        with exe_path.open("rb") as fp:
            if fp.read(2) != b"MZ":
                return None
            fp.seek(0x3C)
            e_lfanew = struct.unpack("<I", fp.read(4))[0]
            fp.seek(e_lfanew + 4)
            machine = struct.unpack("<H", fp.read(2))[0]
    except OSError:
        return None

    if machine == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
        return 64
    if machine == 0x14C:  # IMAGE_FILE_MACHINE_I386
        return 32
    return None


def _resolve_backend(exe_path: Path) -> str:
    """혼합 비트 환경에서 32/64비트 호환성이 높은 백엔드를 선택한다."""
    backend = PYWINAUTO_BACKEND
    exe_bits = _exe_bitness(exe_path)
    py_bits = _python_bitness()

    if exe_bits and exe_bits != py_bits and backend == "win32":
        print(
            f"[INFO] 비트 수 불일치 감지 (Python={py_bits}-bit, RCS EXE={exe_bits}-bit). "
            "win32 백엔드 대신 uia를 사용해 32비트 앱 자동화 이슈를 우회합니다."
        )
        return "uia"

    return backend


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


# ─────────────────────────── VLM 호출 ───────────────────────────

def _vlm_endpoint() -> str:
    base = VLM_API_URL.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _call_vlm(model: str, system_msg: str, prompt: str, img_b64: str) -> str:
    """VLM API를 호출하고 응답 텍스트를 반환한다."""
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    payload = {
        "model": model,
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
        "temperature": 0.0,
    }

    endpoint = _vlm_endpoint()
    print(f"[INFO] VLM 호출: model={model}, endpoint={endpoint}")
    start = time.time()
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    elapsed = (time.time() - start) * 1000
    print(f"[INFO] 응답 수신 ({elapsed:.0f}ms)")
    return raw


def _encode_image(image: Image.Image) -> tuple[str, int, int]:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    w, h = image.size
    print(f"[INFO] 이미지 인코딩: {w}x{h}, {len(buf.getvalue()) / 1024:.1f}KB")
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


def _parse_coords(data: dict, keys: list[str], img_w: int, img_h: int) -> dict:
    """VLM 응답 좌표를 정수로 변환하고 로그를 출력한다. 변환 로직 없이 원본 그대로 사용."""
    for key in keys:
        pt = data.get(key)
        if not pt:
            print(f"  [MISS] {key:20s} — VLM 응답에 없음")
            continue
        raw_x, raw_y = pt.get("x", 0), pt.get("y", 0)
        x, y = int(raw_x), int(raw_y)
        data[key] = {"x": x, "y": y}
        # 범위 체크
        out = ""
        if not (0 <= x <= img_w and 0 <= y <= img_h):
            out = " ← OUT OF BOUNDS"
        print(f"  [RAW ] {key:20s} — raw=({raw_x}, {raw_y}) → px=({x}, {y}){out}")
    return data


def _apply_control_bias(data: dict, img_w: int, img_h: int) -> dict:
    """지정된 사각형 요소 좌표를 오른쪽으로 이동해 클릭 정밀도를 높인다."""
    for key in INPUT_X_OFFSET_KEYS:
        pt = data.get(key)
        if not isinstance(pt, dict):
            continue
        if "x" not in pt or "y" not in pt:
            continue
        try:
            x = int(pt["x"]) + INPUT_X_OFFSET
            y = int(pt["y"])
        except (TypeError, ValueError):
            continue

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        data[key] = {"x": x, "y": y}
        print(f"  [SHIFT] {key:20s} — x +{INPUT_X_OFFSET} applied")
    return data


# ─────────────────────────── 프롬프트 ───────────────────────────

def _build_prompt(w: int, h: int) -> tuple[str, str]:
    """벤치마크용 시스템 메시지와 유저 프롬프트를 반환한다."""
    system_msg = (
        "You are a precise GUI element locator. "
        f"The image is {w}x{h} pixels. "
        "The origin (0, 0) is the top-left corner of the image. "
        "Return coordinates as integer pixel values. "
        "Respond ONLY with valid JSON — no explanation, no markdown."
    )

    prompt = f"""Locate GUI elements in this Remote Control System login dialog.

The dialog has three labeled rows and three buttons.

Find the pixel coordinates of these 9 elements:

TEXT LABELS — find the first letter of each label and return its center:
1. "server_label" — the first letter in "Server"
2. "userid_label" — the first letter in "User ID"
3. "password_label" — the first letter in "Password"

INPUT FIELDS — find a point on the left edge center of each white rectangular control:
4. "server_input" — the combobox/dropdown to the right of "Server" (has a dropdown arrow on its right side).
5. "userid_input" — the white-background, gray-bordered text input field to the right of "User ID"
6. "password_input" — the white-background, gray-bordered text input field to the right of "Password"

BUTTONS — find the left edge center of each clickable button:
7. "login_button" — the button labeled "Log In"
8. "cancel_button" — the button labeled "Cancel"
9. "shortcut_button" — the button with Korean text (for making shortcuts)

Image size: {w} x {h} pixels.
x range: 0 (left edge) to {w} (right edge).
y range: 0 (top edge) to {h} (bottom edge).

Return ONLY this JSON (all values are integers):
{{
    "server_label": {{"x": ..., "y": ...}},
    "server_input": {{"x": ..., "y": ...}},
    "userid_label": {{"x": ..., "y": ...}},
    "userid_input": {{"x": ..., "y": ...}},
    "password_label": {{"x": ..., "y": ...}},
    "password_input": {{"x": ..., "y": ...}},
    "login_button": {{"x": ..., "y": ...}},
    "cancel_button": {{"x": ..., "y": ...}},
    "shortcut_button": {{"x": ..., "y": ...}}
}}"""

    return system_msg, prompt


# ─────────────────────────── 디버그 이미지 ───────────────────────────

def _save_marked_image(
    image: Image.Image, elements: dict, colors: dict, filename: str
) -> None:
    """좌표를 원본 스크린샷 위에 십자선+원으로 마킹하여 저장한다."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    r = 12
    for name, pt in elements.items():
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            continue
        x, y = int(pt["x"]), int(pt["y"])
        color = colors.get(name, "white")
        # 십자선
        draw.line([(x - r, y), (x + r, y)], fill=color, width=2)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=2)
        # 원
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=2)
        # 라벨
        label = f"{name} ({x},{y})"
        # 라벨이 겹치지 않도록 input은 아래, label은 위에 표시
        if "input" in name or "button" in name:
            draw.text((x + r + 3, y + 4), label, fill=color, font=font)
        else:
            draw.text((x + r + 3, y - 16), label, fill=color, font=font)

    out_path = Path(__file__).parent / filename
    debug_img.save(out_path)
    print(f"[INFO] 디버그 이미지 저장: {out_path}")


# ─────────────────────────── 벤치마크 실행 ───────────────────────────

def _run_benchmark(window) -> None:
    """모든 VLM 모델로 좌표 검출을 실행하고 모델별 디버그 이미지를 저장한다."""
    image = _capture_window(window)
    img_b64, w, h = _encode_image(image)

    rect = window.rectangle()
    print(f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
          f"size={rect.right - rect.left}x{rect.bottom - rect.top}")

    system_msg, prompt = _build_prompt(w, h)
    results = {}

    for model in TEST_MODELS:
        print(f"\n{'=' * 60}")
        print(f"[INFO] 모델 테스트: {model}")
        print("=" * 60)

        try:
            raw = _call_vlm(model, system_msg, prompt, img_b64)
            print(f"[INFO] 원문 응답:\n{raw}\n")

            data = _extract_json(raw)
            print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
            data = _parse_coords(data, TARGET_ELEMENTS, w, h)
            data = _apply_control_bias(data, w, h)

            detected = sum(1 for k in TARGET_ELEMENTS if k in data and isinstance(data[k], dict))
            print(f"[INFO] 검출률: {detected}/{len(TARGET_ELEMENTS)}")

            # 모델명을 파일명에 안전하게 변환
            safe_name = model.replace("/", "_").replace(" ", "_")
            filename = f"debug_{safe_name}.png"
            _save_marked_image(image, data, ELEMENT_COLORS, filename)

            results[model] = {"detected": detected, "data": data}

        except Exception as exc:
            print(f"[ERROR] {model} 실패: {exc}")
            results[model] = {"detected": 0, "error": str(exc)}

    # 요약
    print(f"\n{'=' * 60}")
    print("[INFO] ===== 벤치마크 결과 요약 =====")
    print("=" * 60)
    for model, res in results.items():
        det = res["detected"]
        total = len(TARGET_ELEMENTS)
        status = f"{det}/{total} 검출" if "error" not in res else f"실패: {res['error']}"
        print(f"  {model:30s} — {status}")
    print("=" * 60)


# ─────────────────────────── 메인 ───────────────────────────

def main() -> int:
    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return 1

    print(f"[INFO] RCS 시작: {RCS_EXE}")
    backend = _resolve_backend(RCS_EXE)
    cmd_str = subprocess.list2cmdline([str(RCS_EXE)])
    print(f"[INFO] pywinauto 백엔드: {backend}")
    app = Application(backend=backend).start(cmd_str, wait_for_idle=False)

    try:
        login_window = _wait_for_login_window(app)
        print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        return 3

    time.sleep(1.0)
    _run_benchmark(login_window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
