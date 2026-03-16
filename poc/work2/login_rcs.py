"""RCS 로그인 창을 열고 ui-venus 로 입력창/버튼 좌표를 마킹하는 스크립트.

Windows 전용 스크립트다. RCS 실행 파일을 열어 로그인 다이얼로그를 캡처하고,
ui-venus VLM 에 좌표 검출을 요청한 뒤 입력창/버튼 중심점을 오버레이 이미지로 저장한다.

사용법:
  uv run python poc/work2/login_rcs.py
"""

import json
import os
import struct
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.prompts import build_rcs_login_locator_prompt
from poc.work2.rcs_utils import (
    capture_window,
    debug_image_path,
    encode_image_webp,
    extract_json,
    find_existing_main_window,
    format_elapsed_ms,
    is_main_window_title,
    launch_application,
    parse_coords,
    save_debug_webp,
    save_marked_image,
    wait_for_window_by_title_prefix,
)
from poc.work2.logger import log_work2_event
from poc.work2.vlm_client import Work2VLMClient

load_dotenv()

RCS_EXE = Path(os.environ.get("RCS_EXE_PATH", r"C:\Users\2067928\Documents\RCS\RcsMainHD.exe"))
PYWINAUTO_BACKEND = os.environ.get("PYWINAUTO_BACKEND", "").strip().lower() or "uia"
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
    for item in os.environ.get("RCS_DESKTOP_SCAN_BACKENDS", "uia").split(",")
    if item.strip()
]
_desktop_backends = _desktop_backends_raw + [PYWINAUTO_BACKEND]
DESKTOP_SCAN_BACKENDS = tuple(
    dict.fromkeys(item for item in _desktop_backends if item in {"uia", "win32"})
) or ("uia",)

LAUNCH_TIMEOUT = 10.0
WINDOW_TITLE_PREFIX = "Remote Control System"
LOGIN_SERVICE_SLUG = "ui-venus"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"
LOG_NAME = Path(__file__).stem
INPUT_BUTTON_TARGETS = [
    "server_input",
    "userid_input",
    "password_input",
    "login_button",
    "cancel_button",
    "shortcut_button",
]
ELEMENT_COLORS = {
    "server_input": "salmon",
    "userid_input": "deepskyblue",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0


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

    if machine == 0x8664:
        return 64
    if machine == 0x14C:
        return 32
    return None


def _resolve_backend(exe_path: Path) -> str:
    """혼합 비트 환경에서 32/64비트 호환성이 높은 백엔드를 선택한다."""
    backend = PYWINAUTO_BACKEND
    if backend == "uia":
        return "uia"
    exe_bits = _exe_bitness(exe_path)
    py_bits = _python_bitness()

    if exe_bits and exe_bits != py_bits and backend == "win32":
        print(
            f"[INFO] 비트 수 불일치 감지 (Python={py_bits}-bit, RCS EXE={exe_bits}-bit). "
            "win32 대신 uia 백엔드를 사용합니다."
        )
        return "uia"

    return backend


def _main_title_matcher(title: str) -> bool:
    """메인 RCS 창 제목인지 판별한다."""
    return is_main_window_title(title, MAIN_WINDOW_TITLE_REGEX)


def _save_debug_jpeg(image, out_path: Path) -> None:
    """원본 스크린샷을 JPEG 로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_img = image.convert("RGB") if image.mode != "RGB" else image
    debug_img.save(out_path, format="JPEG", quality=95)
    print(f"[INFO] 원본 캡처 저장: {out_path}")
    log_work2_event(
        component="debug_image",
        message="saved_jpeg",
        log_name=LOG_NAME,
        path=out_path,
        quality=95,
    )


def _locate_login_controls(login_window) -> int:
    """로그인 다이얼로그 스크린샷을 ui-venus 로 분석하고 overlay 를 저장한다."""
    locate_started_at = time.time()
    image = capture_window(login_window)
    client = Work2VLMClient(service_slug=LOGIN_SERVICE_SLUG, log_name=LOG_NAME)

    raw_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_capture.jpg",
        model_name=client.model_name,
    )
    _save_debug_jpeg(image, raw_path)
    vlm_input_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_vlm_input.webp",
        model_name=client.model_name,
    )
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)

    image_b64, width, height = encode_image_webp(image)
    system_message, user_text = build_rcs_login_locator_prompt(
        width=width,
        height=height,
        target_keys=INPUT_BUTTON_TARGETS,
    )

    print(
        f"[INFO] VLM 요청: service={client.service_slug}, "
        f"model={client.model_name}, endpoint={client.endpoint}"
    )
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        image_mime="image/webp",
        system_message=system_message,
        user_text=user_text,
        temperature=VLM_TEMPERATURE,
    )
    print(f"[INFO] VLM 응답 수신: tokens={response.token_usage or {}}")
    print(f"[INFO] 원문 응답:\n{response.text}\n")

    data = extract_json(response.text)
    print(f"[INFO] 파싱된 JSON:\n{json.dumps(data, indent=2)}\n")
    parsed = parse_coords(data, INPUT_BUTTON_TARGETS, width, height)

    detected = sum(
        1 for key in INPUT_BUTTON_TARGETS if key in parsed and isinstance(parsed[key], dict)
    )
    print(f"[INFO] 검출 결과: {detected}/{len(INPUT_BUTTON_TARGETS)}")

    overlay_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_rcs_overlay.jpg",
        model_name=response.model_name,
    )
    save_marked_image(image, parsed, ELEMENT_COLORS, overlay_path)
    print(
        "[INFO] 디버그 이미지: "
        f"capture={raw_path}, vlm_input={vlm_input_path}, overlay={overlay_path}"
    )
    print(f"[INFO] 로그인 이미지 분석 전체 소요: {format_elapsed_ms(locate_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="analysis_finished",
        log_name=LOG_NAME,
        service=client.service_slug,
        model=response.model_name or client.model_name,
        capture_path=raw_path,
        vlm_input_path=vlm_input_path,
        overlay_path=overlay_path,
        detected=detected,
        target_count=len(INPUT_BUTTON_TARGETS),
        elapsed_ms=f"{(time.time() - locate_started_at) * 1000:.1f}",
    )
    return 0 if detected > 0 else 4


def main() -> int:
    """스크립트 엔트리포인트."""
    script_started_at = time.time()
    precheck_started_at = time.time()
    log_work2_event(
        component="login_rcs",
        message="script_started",
        log_name=LOG_NAME,
        exe_path=RCS_EXE,
        backend_default=PYWINAUTO_BACKEND,
        desktop_backends=",".join(DESKTOP_SCAN_BACKENDS),
    )
    existing_window, existing_title, debug_rows = find_existing_main_window(
        DESKTOP_SCAN_BACKENDS,
        _main_title_matcher,
    )
    print(f"[INFO] 기존 메인 창 점검 소요: {format_elapsed_ms(precheck_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="existing_main_window_checked",
        log_name=LOG_NAME,
        elapsed_ms=f"{(time.time() - precheck_started_at) * 1000:.1f}",
        found=existing_window is not None,
        title=existing_title,
    )
    if DEBUG_MAIN_WINDOW_TITLES:
        print(f"[DEBUG] 메인 창 regex: {MAIN_WINDOW_TITLE_REGEX!r}")
        if not debug_rows:
            print("[DEBUG] existing-check: no visible top-level windows")
        else:
            for row in debug_rows:
                print(f"[DEBUG] existing-check {row}")

    if existing_window is not None:
        print(f"[WARNING] 이미 로그인된 RCS 메인 창이 떠 있습니다: '{existing_title}'")
        print("[WARNING] 로그인 다이얼로그 대신 메인 창이 활성 상태일 수 있으니 먼저 상태를 정리하세요.")
        return 2

    if not RCS_EXE.exists():
        print(f"[ERROR] 실행 파일을 찾을 수 없습니다: {RCS_EXE}")
        return 1

    backend = _resolve_backend(RCS_EXE)
    print(f"[INFO] RCS 시작: {RCS_EXE}")
    print(f"[INFO] pywinauto 백엔드: {backend}")
    launch_started_at = time.time()
    try:
        app = launch_application(RCS_EXE, backend, wait_for_idle=False, log_name=LOG_NAME)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        log_work2_event(
            component="login_rcs",
            message="launch_failed",
            level="error",
            log_name=LOG_NAME,
            error=exc,
        )
        return 3
    print(f"[INFO] RCS 프로세스 시작 단계 소요: {format_elapsed_ms(launch_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="launch_completed",
        log_name=LOG_NAME,
        backend=backend,
        elapsed_ms=f"{(time.time() - launch_started_at) * 1000:.1f}",
    )

    wait_started_at = time.time()
    try:
        login_window = wait_for_window_by_title_prefix(
            app,
            DESKTOP_SCAN_BACKENDS,
            WINDOW_TITLE_PREFIX,
            LAUNCH_TIMEOUT,
            log_name=LOG_NAME,
        )
    except TimeoutError as exc:
        print(f"[ERROR] {exc}")
        log_work2_event(
            component="login_rcs",
            message="login_window_wait_failed",
            level="error",
            log_name=LOG_NAME,
            error=exc,
            elapsed_ms=f"{(time.time() - wait_started_at) * 1000:.1f}",
        )
        return 3
    print(f"[INFO] 로그인 창 탐색 소요: {format_elapsed_ms(wait_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="login_window_found",
        log_name=LOG_NAME,
        title=login_window.window_text(),
        elapsed_ms=f"{(time.time() - wait_started_at) * 1000:.1f}",
    )

    print(f"[INFO] 로그인 창 발견: '{login_window.window_text()}'")
    time.sleep(1.0)
    result = _locate_login_controls(login_window)
    print(f"[INFO] login_rcs 전체 소요: {format_elapsed_ms(script_started_at)}")
    log_work2_event(
        component="login_rcs",
        message="script_finished",
        log_name=LOG_NAME,
        result=result,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return result


if __name__ == "__main__":
    sys.exit(main())
