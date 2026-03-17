"""RCS 로그인 창 UI-TARS 전용 분석 스크립트.

UI-TARS-1.5-7B 는 일반 VLM 과 달리 GUI agent 모델로,
`Thought: / Action: click(start_box='(x,y)')` 형식으로 응답한다.
좌표는 Qwen2.5-VL 의 smart-resize 된 이미지 공간의 절대 픽셀값이므로,
원본 이미지 좌표로 역변환이 필요하다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_ui_tars.py
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.flask_vlm import get_service_by_slug, resolve_service_proxy_url
from poc.work2.logger import log_work2_event
from poc.work2.prompts.login_rcs_ui_tars import (
    DEFAULT_UI_TARS_TARGET_KEYS,
    build_login_rcs_ui_tars_prompt,
    build_single_element_prompt,
)
from poc.work2.util.debug_image_utils import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_image,
)
from poc.work2.util.image_utils import encode_image_webp
from poc.work2.util.time_utils import format_elapsed_ms
from poc.work2.vlm_client import Work2VLMClient

load_dotenv()

SERVICE_SLUG = "ui-tars"
LOG_NAME = Path(__file__).stem
COMPONENT_NAME = LOG_NAME
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images"

ELEMENT_COLORS = {
    "window_title_text": "tomato",
    "close_button": "violet",
    "server_label": "gold",
    "server_input": "salmon",
    "userid_label": "dodgerblue",
    "userid_input": "deepskyblue",
    "password_label": "chartreuse",
    "password_input": "limegreen",
    "login_button": "orange",
    "cancel_button": "magenta",
    "shortcut_button": "cyan",
}

UI_TARS_MAX_TOKENS = 1000
UI_TARS_FREQUENCY_PENALTY = 1.0
UI_TARS_IMAGE_FACTOR = 28
UI_TARS_MIN_PIXELS = 100 * 28 * 28       # 78,400
UI_TARS_MAX_PIXELS = 16384 * 28 * 28     # 12,845,056

try:
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
except ValueError:
    VLM_TEMPERATURE = 0.0

# ── Action 파싱 패턴 ─────────────────────────────────────────────────
# click(start_box='(197,525)') 형태에서 좌표를 추출한다.
_ACTION_COORD_PATTERN = re.compile(
    r"click\s*\(\s*start_box\s*=\s*['\"]?\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*['\"]?\s*\)"
)
# element_name: click(...) 형태에서 요소 이름과 좌표를 추출한다.
_ELEMENT_LINE_PATTERN = re.compile(
    r"(\w+)\s*:\s*click\s*\(\s*start_box\s*=\s*['\"]?\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*['\"]?\s*\)"
)


# ── Qwen2.5-VL smart_resize 재현 ────────────────────────────────────
def smart_resize(
    height: int,
    width: int,
    factor: int = UI_TARS_IMAGE_FACTOR,
    min_pixels: int = UI_TARS_MIN_PIXELS,
    max_pixels: int = UI_TARS_MAX_PIXELS,
) -> tuple[int, int]:
    """Qwen2.5-VL 의 smart_resize 로직을 재현한다.

    vLLM 이 모델에 전달하기 전에 이미지를 이 크기로 리사이즈한다.
    UI-TARS 가 출력하는 절대 좌표는 이 리사이즈된 공간 기준이다.

    Returns:
        (resized_height, resized_width) 튜플.
    """
    if height < factor or width < factor:
        raise ValueError(f"height({height}) and width({width}) must be >= {factor}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def convert_ui_tars_coords(
    model_x: int,
    model_y: int,
    original_width: int,
    original_height: int,
) -> tuple[int, int]:
    """UI-TARS 절대 좌표를 원본 이미지 픽셀 좌표로 역변환한다."""
    resized_h, resized_w = smart_resize(original_height, original_width)
    actual_x = int(model_x / resized_w * original_width)
    actual_y = int(model_y / resized_h * original_height)

    actual_x = max(0, min(actual_x, original_width - 1))
    actual_y = max(0, min(actual_y, original_height - 1))
    return actual_x, actual_y


def parse_ui_tars_response(
    text: str,
    original_width: int,
    original_height: int,
    target_keys: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """UI-TARS 응답에서 요소별 좌표를 추출하고 원본 픽셀 좌표로 변환한다."""
    resized_h, resized_w = smart_resize(original_height, original_width)
    print(
        f"[INFO] UI-TARS smart_resize: "
        f"original={original_width}x{original_height} → "
        f"resized={resized_w}x{resized_h}"
    )

    coords: dict[str, dict[str, int]] = {}

    # 패턴 1: element_name: click(start_box='(x,y)') 형태
    for match in _ELEMENT_LINE_PATTERN.finditer(text):
        name = match.group(1).strip()
        mx, my = int(match.group(2)), int(match.group(3))
        ax, ay = convert_ui_tars_coords(mx, my, original_width, original_height)
        coords[name] = {"x": ax, "y": ay}
        print(
            f"  [TARS] {name:20s} - model=({mx}, {my}) → px=({ax}, {ay})"
        )

    # 패턴 2: 단일 click(start_box='(x,y)') — 단일 요소 모드일 때
    if not coords:
        match = _ACTION_COORD_PATTERN.search(text)
        if match and len(target_keys) == 1:
            mx, my = int(match.group(1)), int(match.group(2))
            ax, ay = convert_ui_tars_coords(mx, my, original_width, original_height)
            coords[target_keys[0]] = {"x": ax, "y": ay}
            print(
                f"  [TARS] {target_keys[0]:20s} - model=({mx}, {my}) → px=({ax}, {ay}) [single]"
            )

    # 매칭 안 된 target_keys 리포트
    for key in target_keys:
        if key not in coords:
            print(f"  [MISS] {key:20s} - UI-TARS 응답에서 미검출")

    return coords


def _write_debug_text(path: Path, text: str) -> None:
    """디버그 텍스트 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_debug_json(path: Path, payload: dict) -> None:
    """디버그 JSON 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_ui_tars_batch_analysis(
    *,
    image: "Image.Image",
    debug_image_dir: Path,
    debug_stamp: str,
    target_keys: tuple[str, ...] = DEFAULT_UI_TARS_TARGET_KEYS,
    element_colors: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict:
    """캡처된 로그인 이미지를 UI-TARS 에 한 번에 보내 전체 요소를 탐색한다."""
    colors = element_colors or ELEMENT_COLORS
    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        raise ValueError(f"서비스 {SERVICE_SLUG} 를 찾을 수 없습니다.")

    client = Work2VLMClient(service_slug=SERVICE_SLUG, log_name=LOG_NAME)

    raw_capture_path = debug_image_path(
        debug_image_dir, "login_rcs_capture.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    vlm_input_path = debug_image_path(
        debug_image_dir, "login_rcs_vlm_input.webp",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    raw_response_path = debug_image_path(
        debug_image_dir, "login_rcs_response.txt",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    parsed_json_path = debug_image_path(
        debug_image_dir, "login_rcs_parsed.json",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    overlay_path = debug_image_path(
        debug_image_dir, "login_rcs_overlay.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )

    save_debug_jpeg(image, raw_capture_path, log_name=LOG_NAME)
    save_debug_webp(image, vlm_input_path, log_name=LOG_NAME)

    image_b64, width, height = encode_image_webp(image)
    system_message, user_text = build_login_rcs_ui_tars_prompt(target_keys=target_keys)

    print(
        f"[INFO] UI-TARS 배치 분석 시작: "
        f"model={client.model_name}, endpoint={client.endpoint}, "
        f"image={width}x{height}"
    )

    started_at = time.time()
    try:
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            image_mime="image/webp",
            system_message=system_message,
            user_text=user_text,
            temperature=temperature,
            max_tokens=UI_TARS_MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        elapsed_ms = (time.time() - started_at) * 1000
        print(f"[ERROR] UI-TARS 요청 실패: {exc}")
        log_work2_event(
            component=COMPONENT_NAME,
            message="ui_tars_request_failed",
            level="error",
            log_name=LOG_NAME,
            service=SERVICE_SLUG,
            model=client.model_name,
            elapsed_ms=f"{elapsed_ms:.1f}",
            error=exc,
        )
        return {
            "status": "request_error",
            "error": str(exc),
            "detected_count": 0,
            "target_count": len(target_keys),
        }

    elapsed_ms = (time.time() - started_at) * 1000
    _write_debug_text(raw_response_path, response.text)
    print(f"[INFO] UI-TARS 응답 수신: tokens={response.token_usage or {}}")
    print(f"[INFO] UI-TARS 원문 응답:\n{response.text}\n")

    parsed_coords = parse_ui_tars_response(
        response.text, width, height, target_keys,
    )

    detected_count = len(parsed_coords)
    _write_debug_json(parsed_json_path, parsed_coords)
    save_marked_image(image, parsed_coords, colors, overlay_path)

    print(
        f"[INFO] UI-TARS 배치 분석 완료: "
        f"detected={detected_count}/{len(target_keys)}, "
        f"elapsed={elapsed_ms:.1f}ms"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="ui_tars_batch_finished",
        log_name=LOG_NAME,
        service=SERVICE_SLUG,
        model=response.model_name,
        detected=detected_count,
        target_count=len(target_keys),
        elapsed_ms=f"{elapsed_ms:.1f}",
        raw_response_path=raw_response_path,
        overlay_path=overlay_path,
    )
    return {
        "status": "ok",
        "detected_count": detected_count,
        "target_count": len(target_keys),
        "coords": parsed_coords,
        "raw_response": response.text,
        "elapsed_ms": elapsed_ms,
        "overlay_path": str(overlay_path),
    }


def run_ui_tars_per_element_analysis(
    *,
    image: "Image.Image",
    debug_image_dir: Path,
    debug_stamp: str,
    target_keys: tuple[str, ...] = DEFAULT_UI_TARS_TARGET_KEYS,
    element_colors: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> dict:
    """각 요소를 개별 요청으로 하나씩 탐색한다 (느리지만 더 정확할 수 있음)."""
    colors = element_colors or ELEMENT_COLORS
    client = Work2VLMClient(service_slug=SERVICE_SLUG, log_name=LOG_NAME)

    raw_capture_path = debug_image_path(
        debug_image_dir, "login_rcs_capture.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    overlay_path = debug_image_path(
        debug_image_dir, "login_rcs_overlay_per_elem.jpg",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )
    responses_dir = debug_image_path(
        debug_image_dir, "per_element_responses",
        model_name=client.model_name, timestamp_tag=debug_stamp,
    )

    save_debug_jpeg(image, raw_capture_path, log_name=LOG_NAME)

    image_b64, width, height = encode_image_webp(image)
    all_coords: dict[str, dict[str, int]] = {}
    total_started_at = time.time()

    for key in target_keys:
        system_message, user_text = build_single_element_prompt(key)
        print(f"[INFO] UI-TARS 단일 요소 탐색: {key}")

        started_at = time.time()
        try:
            response = client.chat_with_image_b64(
                image_b64=image_b64,
                image_mime="image/webp",
                system_message=system_message,
                user_text=user_text,
                temperature=temperature,
                max_tokens=200,
                stream=False,
            )
        except Exception as exc:
            print(f"  [ERROR] {key} 요청 실패: {exc}")
            continue

        elem_elapsed_ms = (time.time() - started_at) * 1000
        response_path = Path(str(responses_dir)) / f"{key}.txt"
        _write_debug_text(response_path, response.text)
        print(f"  [INFO] {key} 응답: {response.text.strip()!r} ({elem_elapsed_ms:.0f}ms)")

        parsed = parse_ui_tars_response(
            response.text, width, height, (key,),
        )
        all_coords.update(parsed)

    total_elapsed_ms = (time.time() - total_started_at) * 1000
    detected_count = len(all_coords)

    save_marked_image(image, all_coords, colors, overlay_path)
    print(
        f"[INFO] UI-TARS 개별 분석 완료: "
        f"detected={detected_count}/{len(target_keys)}, "
        f"total_elapsed={total_elapsed_ms:.1f}ms"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="ui_tars_per_element_finished",
        log_name=LOG_NAME,
        service=SERVICE_SLUG,
        model=client.model_name,
        detected=detected_count,
        target_count=len(target_keys),
        elapsed_ms=f"{total_elapsed_ms:.1f}",
        overlay_path=overlay_path,
    )
    return {
        "status": "ok",
        "detected_count": detected_count,
        "target_count": len(target_keys),
        "coords": all_coords,
        "elapsed_ms": total_elapsed_ms,
        "overlay_path": str(overlay_path),
    }


# ── UI-TARS 진단 프로브 ───────────────────────────────────────────
def probe_ui_tars_text_only() -> None:
    """UI-TARS 에 이미지 없이 간단한 텍스트만 보내 응답 생성을 확인한다.

    completion_tokens=1 문제 진단용. 이미지 없이도 1 토큰이면 모델/vLLM 설정 문제,
    이미지 있을 때만 1 토큰이면 이미지 처리 문제로 원인을 분리할 수 있다.
    """
    import requests as _requests

    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        print(f"[PROBE] 서비스 {SERVICE_SLUG} 를 찾을 수 없습니다.")
        return

    proxy_url = resolve_service_proxy_url(SERVICE_SLUG)
    endpoint = f"{proxy_url.rstrip('/')}/v1/chat/completions"

    # 1) 텍스트 전용 요청
    payload = {
        "model": service_entry.model_name,
        "messages": [
            {
                "role": "user",
                "content": "Hello, describe what you can do in one sentence.",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    print(f"[PROBE] UI-TARS 텍스트 전용 요청: endpoint={endpoint}")
    try:
        resp = _requests.post(endpoint, json=payload, timeout=30)
        print(f"[PROBE] status={resp.status_code}, body={resp.text[:600]}")
    except Exception as exc:
        print(f"[PROBE] 텍스트 전용 요청 실패: {exc}")


# ── 창 탐색은 login_rcs_Rev2 와 동일 로직 재사용 ────────────────────
def _find_login_window():
    """login_rcs_Rev2 의 창 탐색 로직을 재사용한다."""
    from poc.work2.login_rcs_Rev2 import _find_login_window as _find
    return _find()


def main() -> str:
    """UI-TARS 로 RCS 로그인 창을 분석한다."""
    script_started_at = time.time()
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_started",
        log_name=LOG_NAME,
        service=SERVICE_SLUG,
    )

    # 진단 프로브: UI-TARS 가 텍스트 전용으로도 생성하는지 확인
    if os.getenv("UI_TARS_PROBE", "1").strip() == "1":
        probe_ui_tars_text_only()

    # 배치 모드(기본) vs 개별 모드 선택
    mode = os.getenv("UI_TARS_MODE", "batch").strip().lower()

    login_window, window_title, backend = _find_login_window()
    if login_window is None:
        print(
            "[ERROR] 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        return "login_window_not_found"

    from poc.work2.util import (
        activate_window,
        capture_window,
        foreground_window,
        make_timestamp_tag,
    )

    debug_stamp = make_timestamp_tag(script_started_at)

    if not activate_window(login_window, debug_label=f"ui_tars backend={backend}"):
        print("[ERROR] 로그인 창 활성화 실패")
        return "login_window_activate_failed"

    if not foreground_window(login_window, debug_label=f"ui_tars backend={backend}"):
        print("[ERROR] 로그인 창 foreground 실패")
        return "login_window_activate_failed"

    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        return "capture_failed"

    if mode == "per_element":
        print("[INFO] UI-TARS 개별 요소 모드 실행")
        result = run_ui_tars_per_element_analysis(
            image=image,
            debug_image_dir=DEBUG_IMAGE_DIR,
            debug_stamp=debug_stamp,
            temperature=VLM_TEMPERATURE,
        )
    else:
        print("[INFO] UI-TARS 배치 모드 실행")
        result = run_ui_tars_batch_analysis(
            image=image,
            debug_image_dir=DEBUG_IMAGE_DIR,
            debug_stamp=debug_stamp,
            temperature=VLM_TEMPERATURE,
        )

    status = result.get("status", "unknown")
    detected = result.get("detected_count", 0)
    target = result.get("target_count", 0)
    print(
        f"[INFO] {LOG_NAME} 결과: status={status}, "
        f"detected={detected}/{target}, "
        f"end-to-end={format_elapsed_ms(script_started_at)}"
    )
    log_work2_event(
        component=COMPONENT_NAME,
        message="script_finished",
        log_name=LOG_NAME,
        result=status,
        detected=detected,
        target_count=target,
        mode=mode,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )

    return "success" if status == "ok" and detected > 0 else status


if __name__ == "__main__":
    exit_result = main()
    if exit_result != "success":
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
