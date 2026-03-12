"""RCS 로그인 화면에서 work2 VLM pipeline 좌표 검출을 테스트하는 스크립트 (Windows 전용).

Primary VLM은 UI-Venus를 사용하고, PaddleOCR-VL을 보조 OCR stage 로 사용한다.
로그인 화면의 텍스트 라벨·입력 필드·버튼 좌표를 추출하고 디버그 이미지를 저장한다.
pywinauto는 창 실행·탐색에만 사용한다.
"""

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

from pywinauto.keyboard import send_keys
from pywinauto import mouse
from pywinauto import Desktop
from pywinauto.application import Application

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient
from poc.work2.flask_vlm import apply_pipeline_env_defaults
from poc.work2.pipeline_ocr import build_ocr_extra_instructions, collect_ocr_hint_result
from poc.work2.prompts import build_rcs_login_locator_prompt
from poc.work2.rcs_utils import (
    capture_window,
    click_at,
    debug_image_path,
    encode_image_webp,
    extract_json,
    find_existing_main_window,
    is_main_window_title,
    parse_coords,
    save_marked_image,
    scan_window_list,
)

PIPELINE_CONFIG = apply_pipeline_env_defaults()

# ─────────────────────────── 설정 ───────────────────────────

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
VLM_API_URL = str(PIPELINE_CONFIG["primary_api_url"] or "")
VLM_API_KEY = str(PIPELINE_CONFIG["primary_api_key"] or "")
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

# 테스트할 primary VLM 모델 목록
PRIMARY_VLM_MODEL = str(PIPELINE_CONFIG["primary_model_name"] or "ui-venus-1.5-8b")
TEST_MODELS = [
    PRIMARY_VLM_MODEL,
]
TARGET_CLICK_KEY = os.environ.get("RCS_TARGET_CLICK_KEY", "login_button").strip() or "login_button"
LOGIN_OCR_FOCUS_WORDS = (
    "Server",
    "User ID",
    "Password",
    "Log In",
    "Cancel",
)

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

VLM_CLIENT = LangChainOpenAICompatibleVLMClient(
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

DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"

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


def _main_title_matcher(title: str) -> bool:
    """모듈 설정 MAIN_WINDOW_TITLE_REGEX 기준 매칭 함수."""
    return is_main_window_title(title, MAIN_WINDOW_TITLE_REGEX)


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


def _scan_main_window_candidates(app):
    """현재 app.windows()의 타이틀과 매칭 결과를 수집한다."""
    debug_rows = []

    # 1) 시작 프로세스(app)에서 먼저 탐색
    app_window, app_title = scan_window_list(
        app.windows(), "app", debug_rows, _main_title_matcher
    )
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

        desktop_window, desktop_title = scan_window_list(
            desktop_windows, f"desktop[{backend}]", debug_rows, _main_title_matcher
        )
        if desktop_window is not None:
            return desktop_window, desktop_title, debug_rows

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


# ─────────────────────────── 좌표 보정 ───────────────────────────


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


# ─────────────────────────── 스크롤 ───────────────────────────


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


# ─────────────────────────── 벤치마크 실행 ───────────────────────────

def _run_benchmark(window) -> dict | None:
    """Primary VLM + OCR assist pipeline 으로 좌표 검출을 실행한다."""
    image = capture_window(window)
    img_b64, w, h = encode_image_webp(image)

    rect = window.rectangle()
    print(f"[INFO] 창 영역: left={rect.left}, top={rect.top}, "
          f"size={rect.right - rect.left}x{rect.bottom - rect.top}")

    ocr_result = collect_ocr_hint_result(
        image_b64=img_b64,
        image_width=w,
        image_height=h,
        image_mime="image/webp",
        pipeline_config=PIPELINE_CONFIG,
        context_label="RCS login dialog",
        focus_words=LOGIN_OCR_FOCUS_WORDS,
    )
    prompt_extra_instructions = build_ocr_extra_instructions(ocr_result)

    system_msg, prompt = build_rcs_login_locator_prompt(
        width=w,
        height=h,
        target_keys=TARGET_ELEMENTS,
        extra_instructions=prompt_extra_instructions,
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

            data = extract_json(raw)
            print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
            data = parse_coords(data, TARGET_ELEMENTS, w, h)
            data = _apply_control_bias(data, w, h)

            detected = sum(1 for k in TARGET_ELEMENTS if k in data and isinstance(data[k], dict))
            print(f"[INFO] 검출률: {detected}/{len(TARGET_ELEMENTS)}")

            out_path = debug_image_path(
                DEBUG_IMAGE_DIR,
                "debug_login.png",
                model_name=model,
            )
            save_marked_image(image, data, ELEMENT_COLORS, out_path)

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

    return results.get(PRIMARY_VLM_MODEL, {}).get("data")


# ─────────────────────────── 메인 ───────────────────────────

def main() -> int:
    if VLM_API_URL:
        print(
            f"[INFO] pipeline: primary={PIPELINE_CONFIG['primary_service']} "
            f"({PRIMARY_VLM_MODEL}) -> ocr={PIPELINE_CONFIG['ocr_service']} "
            f"({PIPELINE_CONFIG['ocr_model_name']})"
        )
        print(
            f"[INFO] primary endpoint={VLM_API_URL}, "
            f"ocr endpoint={PIPELINE_CONFIG['ocr_api_url']}"
        )
    else:
        print("[WARNING] primary VLM API URL이 비어 있습니다. poc/work2/flask_vlm.py 의 공유 설정을 확인하세요.")

    existing_window, existing_title, existing_debug_rows = find_existing_main_window(
        DESKTOP_SCAN_BACKENDS, _main_title_matcher
    )
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

    if not click_at(
        TARGET_CLICK_KEY,
        login_window,
        data,
        retry_count=CLICK_RETRY_COUNT,
        retry_delay_sec=CLICK_RETRY_DELAY_SEC,
    ):
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
