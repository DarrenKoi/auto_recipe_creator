"""RCS 로그인 화면에서 VLM 좌표 검출 정확도를 비교하는 벤치마크 스크립트 (Windows 전용).

여러 VLM 모델(Kimi-K2.5, Qwen3-VL-30B)에 동일한 스크린샷을 전송하여
텍스트 라벨·입력 필드(콤보박스/텍스트)·버튼 3개의 좌표를 추출하고, 모델별 디버그 이미지를 저장한다.
pywinauto는 창 실행·탐색에만 사용한다.
"""

import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import mss
import mss.tools
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pywinauto.keyboard import send_keys
from pywinauto import mouse
from pywinauto import Desktop
from pywinauto.application import Application

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_openai_client import ChatImageRequest, LangChainOpenAIVLMClient  # noqa: E402
from prompts import build_rcs_login_locator_prompt  # noqa: E402

load_dotenv()

# ─────────────────────────── 설정 ───────────────────────────

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
VLM_API_URL = (
    os.environ.get("VLM_API_URL", "").strip()
    or os.environ.get("VLM_API_BASE_URL", "").strip()
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "win32"
MAIN_WINDOW_TITLE_REGEX = (
    os.environ.get("RCS_MAIN_WINDOW_REGEX", r"\brcs\b.*\[server\s*:[^\]]+\]").strip()
    or r"\brcs\b.*\[server\s*:[^\]]+\]"
)
DEBUG_MAIN_WINDOW_TITLES = (
    os.environ.get("RCS_DEBUG_MAIN_WINDOW_TITLES", "0").strip().lower()
    not in {"0", "false", "no", "off"}
)
_desktop_backends_raw = [
    item.strip().lower()
    for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "win32,uia").split(",")
    if item.strip()
]
_desktop_backends = _desktop_backends_raw + [PYWINAUTO_BACKEND]
DESKTOP_SCAN_BACKENDS = tuple(
    dict.fromkeys(b for b in _desktop_backends if b in {"uia", "win32"})
) or ("uia", "win32")

LAUNCH_TIMEOUT = 30.0
WINDOW_TITLE_PREFIX = "Remote Control System"

# 테스트할 VLM 모델 목록
KIMI_MODEL = "Kimi-K2.5"
TEST_MODELS = [
    KIMI_MODEL,
]
TARGET_CLICK_KEY = os.environ.get("RCS_TARGET_CLICK_KEY", "login_button").strip() or "login_button"

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

VLM_CLIENT = LangChainOpenAIVLMClient(
    base_url=VLM_API_URL,
    api_key=VLM_API_KEY,
    timeout_sec=120.0,
)

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0

try:
    POST_LOGIN_DELAY_SEC = float(os.getenv("RCS_POST_LOGIN_DELAY_SEC", "4.0"))
except ValueError:
    POST_LOGIN_DELAY_SEC = 4.0

try:
    POST_LOGIN_MAIN_TIMEOUT_SEC = float(os.getenv("RCS_POST_LOGIN_MAIN_TIMEOUT_SEC", "240.0"))
except ValueError:
    POST_LOGIN_MAIN_TIMEOUT_SEC = 240.0

try:
    POST_LOGIN_POLL_SEC = float(os.getenv("RCS_POST_LOGIN_POLL_SEC", "0.5"))
except ValueError:
    POST_LOGIN_POLL_SEC = 0.5

try:
    CLICK_RETRY_COUNT = int(os.getenv("RCS_CLICK_RETRY_COUNT", "2"))
except ValueError:
    CLICK_RETRY_COUNT = 2

try:
    CLICK_RETRY_DELAY_SEC = float(os.getenv("RCS_CLICK_RETRY_DELAY_SEC", "0.25"))
except ValueError:
    CLICK_RETRY_DELAY_SEC = 0.25

try:
    POST_LOGIN_SCROLL_STEPS = int(os.getenv("RCS_POST_LOGIN_SCROLL_STEPS", "0"))
except ValueError:
    POST_LOGIN_SCROLL_STEPS = 0

try:
    POST_LOGIN_SCROLL_INTERVAL = float(os.getenv("RCS_POST_LOGIN_SCROLL_INTERVAL", "0.3"))
except ValueError:
    POST_LOGIN_SCROLL_INTERVAL = 0.3

POST_LOGIN_SCROLL_MODE = (
    os.getenv("RCS_POST_LOGIN_SCROLL_MODE", "wheel").strip().lower() or "wheel"
)  # wheel | keys | combo

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
SERVER_INPUT_X_OFFSET = 50
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


def _is_main_window_title(title: str) -> bool:
    try:
        return re.search(MAIN_WINDOW_TITLE_REGEX, title, flags=re.IGNORECASE) is not None
    except re.error:
        t = title.lower()
        return "rcs" in t and "[server" in t


def _find_window_by_title(app, matcher):
    for win in app.windows():
        try:
            title = win.window_text() or ""
        except Exception:
            continue
        if matcher(title):
            return win, title
    return None, ""


def _scan_window_list(windows, source_name: str, debug_rows: list[str]):
    for idx, win in enumerate(windows, start=1):
        try:
            title = win.window_text() or ""
        except Exception as exc:
            debug_rows.append(f"{source_name}[{idx}] title-read-error={exc}")
            continue

        matched = _is_main_window_title(title)
        debug_rows.append(f"{source_name}[{idx}] matched={matched} title={title!r}")
        if matched:
            return win, title

    return None, ""


def _scan_main_window_candidates(app):
    """현재 app.windows()의 타이틀과 매칭 결과를 수집한다."""
    debug_rows = []

    # 1) 시작 프로세스(app)에서 먼저 탐색
    app_window, app_title = _scan_window_list(app.windows(), "app", debug_rows)
    if app_window is not None:
        return app_window, app_title, debug_rows

    # 2) RCS가 재기동되어 프로세스가 바뀌는 경우를 위해 데스크톱 전체 창에서 탐색
    for backend in DESKTOP_SCAN_BACKENDS:
        try:
            desktop_windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True
            )
        except Exception as exc:
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        desktop_window, desktop_title = _scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows
        )
        if desktop_window is not None:
            return desktop_window, desktop_title, debug_rows

    return None, "", debug_rows


def _find_existing_main_window():
    """이미 떠 있는 메인 RCS 창(기 로그인 상태)을 데스크톱 전체에서 탐색한다."""
    debug_rows = []
    for backend in DESKTOP_SCAN_BACKENDS:
        try:
            desktop_windows = Desktop(backend=backend).windows(
                top_level_only=True, visible_only=True
            )
        except Exception as exc:
            debug_rows.append(f"desktop[{backend}] windows-error={exc}")
            continue

        main_window, main_title = _scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows
        )
        if main_window is not None:
            return main_window, main_title, debug_rows

    return None, "", debug_rows


def _wait_for_post_login_windows(app):
    """로그인 버튼 클릭 후 메인 RCS 창이 나타날 때까지 대기한다."""
    if POST_LOGIN_DELAY_SEC > 0:
        print(f"[INFO] 로그인 직후 안정화 대기: {POST_LOGIN_DELAY_SEC:.1f}초")
        time.sleep(POST_LOGIN_DELAY_SEC)

    print(
        f"[INFO] 메인 RCS 창 대기 시작 (최대 {POST_LOGIN_MAIN_TIMEOUT_SEC:.0f}초, "
        f"poll={POST_LOGIN_POLL_SEC:.1f}s)"
    )
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")

    main_deadline = time.time() + POST_LOGIN_MAIN_TIMEOUT_SEC
    attempt = 0
    while time.time() < main_deadline:
        attempt += 1
        main_window, main_title, debug_rows = _scan_main_window_candidates(app)
        if DEBUG_MAIN_WINDOW_TITLES:
            elapsed = POST_LOGIN_MAIN_TIMEOUT_SEC - max(0.0, main_deadline - time.time())
            print(f"[DEBUG] title-scan attempt={attempt}, elapsed={elapsed:.1f}s")
            if not debug_rows:
                print("[DEBUG] app.windows() returned no visible top-level windows")
            else:
                for row in debug_rows:
                    print(f"[DEBUG] {row}")

        if main_window is not None:
            print(f"[INFO] 로그인 성공 창 발견: '{main_title}'")
            return main_window
        time.sleep(POST_LOGIN_POLL_SEC)

    print(f"[ERROR] 메인 RCS 창을 {POST_LOGIN_MAIN_TIMEOUT_SEC:.0f}초 내에 찾지 못했습니다.")
    return None


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
        
        # server_input은 더 큰 오프셋(50px) 적용, 나머지는 기본(12px)
        offset = SERVER_INPUT_X_OFFSET if key == "server_input" else INPUT_X_OFFSET
        
        try:
            x = int(pt["x"]) + offset
            y = int(pt["y"])
        except (TypeError, ValueError):
            continue

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        data[key] = {"x": x, "y": y}
        print(f"  [SHIFT] {key:20s} — x +{offset} applied")
    return data


def _click_at(element_key: str, window, elements: dict) -> bool:
    """VLM 좌표를 스크린 좌표로 변환해 요소를 클릭한다."""
    pt = elements.get(element_key)
    if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
        print(f"[ERROR] 클릭 대상 '{element_key}' 좌표가 없습니다.")
        return False

    rect = window.rectangle()
    x = int(pt["x"]) + rect.left
    y = int(pt["y"]) + rect.top
    x = max(rect.left, min(x, rect.right - 1))
    y = max(rect.top, min(y, rect.bottom - 1))
    rel_x = max(0, min(int(pt["x"]), rect.right - rect.left - 1))
    rel_y = max(0, min(int(pt["y"]), rect.bottom - rect.top - 1))

    print(f"[INFO] '{element_key}' 클릭: screen=({x}, {y})")
    for attempt in range(1, max(1, CLICK_RETRY_COUNT) + 1):
        try:
            window.set_focus()
        except Exception:
            pass

        try:
            # 창 핸들 기준 실클릭을 먼저 시도한다.
            window.click_input(coords=(rel_x, rel_y), button="left")
            print(f"[INFO] click_input 성공 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] click_input 실패 (attempt={attempt}): {exc}")

        try:
            # 전역 마우스 이벤트로 down/up을 강제 전송한다.
            mouse.move(coords=(x, y))
            time.sleep(0.08)
            mouse.press(button="left", coords=(x, y))
            time.sleep(0.05)
            mouse.release(button="left", coords=(x, y))
            print(f"[INFO] mouse press/release 실행 (attempt={attempt})")
            return True
        except Exception as exc:
            print(f"[WARN] mouse press/release 실패 (attempt={attempt}): {exc}")

        time.sleep(max(0.0, CLICK_RETRY_DELAY_SEC))

    return False


def _scroll_to_reveal_more(window) -> None:
    """로그인 후 큰 화면에서 더 많은 텍스트 항목을 보도록 스크롤한다."""
    if POST_LOGIN_SCROLL_STEPS <= 0:
        return

    mode = POST_LOGIN_SCROLL_MODE
    if mode not in {"wheel", "keys", "combo"}:
        mode = "wheel"

    rect = window.rectangle()
    center_x = (rect.left + rect.right) // 2
    center_y = (rect.top + rect.bottom) // 2
    steps = POST_LOGIN_SCROLL_STEPS

    if mode == "combo":
        # ComboBox 항목이 펼쳐졌을 때 내려가며 항목 노출을 시도
        print(f"[INFO] Combo mode scroll: 단계 수={steps}")
        send_keys("{F4}")
        time.sleep(0.25)
        for _ in range(steps):
            send_keys("{DOWN}")
            time.sleep(POST_LOGIN_SCROLL_INTERVAL)
        send_keys("{ESC}")
        return

    if mode == "keys":
        # 포커스된 스크롤 패널/문서에서 페이지 이동
        print(f"[INFO] Key mode scroll: 단계 수={steps}, 대상=PageDown")
        for _ in range(steps):
            send_keys("{PGDN}")
            time.sleep(POST_LOGIN_SCROLL_INTERVAL)
        return

    # wheel mode: 기본값. 창 중앙에서 마우스 휠로 내려감
    print(f"[INFO] Wheel mode scroll: 단계 수={steps}, 좌표=({center_x}, {center_y})")
    for _ in range(steps):
        mouse.scroll(coords=(center_x, center_y), wheel_dist=-1)
        time.sleep(POST_LOGIN_SCROLL_INTERVAL)


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

def _run_benchmark(window) -> dict | None:
    """모든 VLM 모델로 좌표 검출을 실행하고 모델별 디버그 이미지를 저장한다."""
    image = _capture_window(window)
    img_b64, w, h = _encode_image(image)

    rect = window.rectangle()
    print(f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
          f"size={rect.right - rect.left}x{rect.bottom - rect.top}")

    system_msg, prompt = build_rcs_login_locator_prompt(
        width=w,
        height=h,
        target_keys=TARGET_ELEMENTS,
    )
    results = {}

    for model in TEST_MODELS:
        print(f"\n{'=' * 60}")
        print(f"[INFO] 모델 테스트: {model}")
        print("=" * 60)

        try:
            request = ChatImageRequest(
                model=model,
                system_message=system_msg,
                user_text=prompt,
                image_b64=img_b64,
                temperature=VLM_TEMPERATURE,
            )
            print(f"[INFO] VLM 호출: model={model}, endpoint={VLM_CLIENT.endpoint}")
            start = time.time()
            raw = VLM_CLIENT.chat_with_image(request)
            elapsed = (time.time() - start) * 1000
            print(f"[INFO] 응답 수신 ({elapsed:.0f}ms)")
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

    return results.get(KIMI_MODEL, {}).get("data")


# ─────────────────────────── 메인 ───────────────────────────

def main() -> int:
    existing_window, existing_title, existing_debug_rows = _find_existing_main_window()
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")
        if not existing_debug_rows:
            print("[DEBUG] existing-check: no visible top-level windows")
        else:
            for row in existing_debug_rows:
                print(f"[DEBUG] existing-check {row}")
    if existing_window is not None:
        print(f"[INFO] 이미 로그인된 RCS 창 감지: '{existing_title}'")
        print("[INFO] 로그인 절차를 건너뜁니다.")
        return 0

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
    data = _run_benchmark(login_window)
    if not data:
        return 4

    if not _click_at(TARGET_CLICK_KEY, login_window, data):
        return 5

    if TARGET_CLICK_KEY != "login_button":
        print(
            f"[WARN] TARGET_CLICK_KEY='{TARGET_CLICK_KEY}' 이므로 "
            "로그인 성공 창 검증을 건너뜁니다."
        )
        return 0

    main_window = _wait_for_post_login_windows(app)
    if main_window is None:
        return 6

    _scroll_to_reveal_more(main_window)

    return 0


if __name__ == "__main__":
    sys.exit(main())
