"""RCS 로그인 창 텍스트 전용 GOT-OCR 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
`got-ocr` `/v1/ocr` 엔드포인트로 스크린샷 내 텍스트만 읽는다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_got_ocr.py
"""

import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from poc.work2.flask_vlm import get_service_by_slug, resolve_service_api_key, resolve_service_proxy_url
from poc.work2.login_rcs_common import DESKTOP_SCAN_BACKENDS, WINDOW_TITLE_PREFIX, find_login_window
from poc.work2.logger import log_work2_event
from poc.work2.util import (
    activate_window,
    capture_window,
    debug_image_path,
    foreground_window,
    format_elapsed_ms,
    make_timestamp_tag,
    save_debug_jpeg,
    save_debug_webp,
)
from poc.work2.util.debug_image_utils import save_debug_json, save_debug_text


load_dotenv()

SERVICE_SLUG = "got-ocr"
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images" / "login_rcs_got_ocr"
LOG_NAME = Path(__file__).stem
TIMEOUT_SEC = 120.0

EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"
EXIT_OCR_EMPTY = "ocr_empty"


def _env_flag(name: str, default: bool = False) -> bool:
    """bool 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _parse_got_box(raw: str) -> list[int] | None:
    """GOT-OCR box 환경변수를 `[x1, y1, x2, y2]`로 파싱한다."""
    text = raw.strip()
    if not text:
        return None

    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("LOGIN_RCS_GOT_OCR_BOX 는 x1,y1,x2,y2 형식이어야 합니다.")

    try:
        box = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("LOGIN_RCS_GOT_OCR_BOX 는 정수 4개여야 합니다.") from exc

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("LOGIN_RCS_GOT_OCR_BOX 는 x2>x1, y2>y1 이어야 합니다.")
    return box


def _normalize_lines(raw_text: str, max_items: int = 80) -> list[str]:
    """OCR 응답을 사람이 보기 쉬운 고유 줄 목록으로 정리한다."""
    if not raw_text.strip():
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_items:
            break
    return lines


def _print_ocr_preview(lines: list[str]) -> None:
    """OCR 결과 일부를 콘솔에 출력한다."""
    if not lines:
        print("[INFO] OCR 결과 preview 없음")
        return

    print("[INFO] OCR 텍스트 preview:")
    for index, line in enumerate(lines[:20], start=1):
        print(f"  {index:02d}. {line}")
    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more line(s))")


def _build_got_ocr_endpoint(base_url: str) -> str:
    """GOT-OCR `/v1/ocr` 엔드포인트를 구성한다."""
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/ocr"
    return f"{normalized}/v1/ocr"


def _save_input_artifacts(image, model_name: str, timestamp_tag: str) -> Path:
    """원본 JPEG와 전송 WebP를 저장한다."""
    capture_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_capture.jpg",
        model_name=model_name,
        timestamp_tag=timestamp_tag,
    )
    webp_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_input.webp",
        model_name=model_name,
        timestamp_tag=timestamp_tag,
    )
    save_debug_jpeg(image, capture_path, log_name=LOG_NAME)
    save_debug_webp(image, webp_path, quality=90, log_name=LOG_NAME)
    return webp_path


def _extract_response_text(body) -> str:
    """GOT-OCR 응답에서 텍스트 필드를 추출한다."""
    if isinstance(body, dict):
        for key in ("text", "output_text", "content", "result"):
            value = body.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return str(body).strip()


def _run_login_ocr(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 GOT-OCR로 읽고 텍스트 결과를 저장한다."""
    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)
    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        print(f"[ERROR] OCR 서비스 설정을 찾지 못했습니다: {SERVICE_SLUG}")
        return EXIT_OCR_REQUEST_ERROR

    if not activate_window(
        login_window,
        debug_label=f"login_window got-ocr recapture backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 재활성화 실패: title={window_title!r}, backend={backend}")
        log_work2_event(
            component="login_rcs_got_ocr",
            message="login_window_reactivate_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    if not foreground_window(
        login_window,
        debug_label=f"login_window got-ocr screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 foreground 활성화 실패: title={window_title!r}, backend={backend}")
        log_work2_event(
            component="login_rcs_got_ocr",
            message="login_window_foreground_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_LOGIN_WINDOW_ACTIVATE_FAILED

    try:
        image = capture_window(login_window)
    except Exception as exc:
        print(f"[ERROR] 로그인 창 캡처 실패: {exc}")
        log_work2_event(
            component="login_rcs_got_ocr",
            message="capture_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            error=exc,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_CAPTURE_FAILED

    try:
        got_box = _parse_got_box(os.environ.get("LOGIN_RCS_GOT_OCR_BOX", ""))
    except ValueError as exc:
        print(f"[ERROR] GOT-OCR box 설정 오류: {exc}")
        log_work2_event(
            component="login_rcs_got_ocr",
            message="got_box_invalid",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            error=exc,
        )
        return EXIT_OCR_REQUEST_ERROR

    got_format_output = _env_flag("LOGIN_RCS_GOT_OCR_FORMAT_OUTPUT", default=True)
    got_crop_to_patches = _env_flag("LOGIN_RCS_GOT_OCR_CROP_TO_PATCHES", default=False)

    webp_path = _save_input_artifacts(image, service_entry.model_name, debug_stamp)
    raw_response_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_ocr_response.txt",
        model_name=service_entry.model_name,
        timestamp_tag=debug_stamp,
    )
    result_json_path = debug_image_path(
        DEBUG_IMAGE_DIR,
        "login_ocr_result.json",
        model_name=service_entry.model_name,
        timestamp_tag=debug_stamp,
    )

    base_url = resolve_service_proxy_url(SERVICE_SLUG)
    endpoint = _build_got_ocr_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    api_key = resolve_service_api_key(SERVICE_SLUG)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    image_bytes = webp_path.read_bytes()
    payload: dict[str, object] = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "format_output": got_format_output,
        "crop_to_patches": got_crop_to_patches,
    }
    if got_box is not None:
        payload["box"] = got_box

    print(
        f"[INFO] 로그인 창 OCR 시작: backend={backend}, title={window_title!r}, "
        f"service_slug={SERVICE_SLUG}, format_output={got_format_output}, "
        f"crop_to_patches={got_crop_to_patches}, box={got_box}"
    )

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_SEC,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        print(f"[ERROR] GOT-OCR 요청 실패: {exc}")
        log_work2_event(
            component="login_rcs_got_ocr",
            message="ocr_request_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            service_slug=SERVICE_SLUG,
            error=exc,
            endpoint=endpoint,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_OCR_REQUEST_ERROR

    raw_text = _extract_response_text(body)
    normalized_lines = _normalize_lines(raw_text)
    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": SERVICE_SLUG,
            "model_name": str(body.get("model", service_entry.model_name)) if isinstance(body, dict) else service_entry.model_name,
            "api_url": base_url,
            "endpoint": endpoint,
            "window_title": window_title,
            "backend": backend,
            "got_box": got_box,
            "format_output": got_format_output,
            "crop_to_patches": got_crop_to_patches,
            "raw_text": raw_text,
            "normalized_lines": normalized_lines,
            "line_count": len(normalized_lines),
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            "raw_body": body,
        },
    )

    _print_ocr_preview(normalized_lines)
    print(f"[INFO] OCR raw response 저장: {raw_response_path}")
    print(f"[INFO] OCR summary 저장: {result_json_path}")
    print(f"[INFO] login_rcs_got_ocr 소요: {format_elapsed_ms(started_at)}")

    if not raw_text:
        log_work2_event(
            component="login_rcs_got_ocr",
            message="ocr_empty",
            level="warning",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            service_slug=SERVICE_SLUG,
            endpoint=endpoint,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_OCR_EMPTY

    log_work2_event(
        component="login_rcs_got_ocr",
        message="ocr_finished",
        log_name=LOG_NAME,
        title=window_title,
        backend=backend,
        service_slug=SERVICE_SLUG,
        endpoint=endpoint,
        line_count=len(normalized_lines),
        elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS


def main() -> str:
    """이미 열려 있는 로그인 창을 GOT-OCR로 읽는다."""
    script_started_at = time.time()
    log_work2_event(
        component="login_rcs_got_ocr",
        message="script_started",
        log_name=LOG_NAME,
        desktop_backends=",".join(DESKTOP_SCAN_BACKENDS),
        window_title_prefix=WINDOW_TITLE_PREFIX,
        service_slug=SERVICE_SLUG,
    )

    login_window, window_title, backend = find_login_window()
    if login_window is None:
        print(
            "[ERROR] 이미 떠 있는 로그인 창을 찾지 못했습니다. "
            "먼저 open_rcs.py 로 로그인 창을 열어 두세요."
        )
        log_work2_event(
            component="login_rcs_got_ocr",
            message="login_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=WINDOW_TITLE_PREFIX,
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    result = _run_login_ocr(login_window, window_title, backend)
    log_work2_event(
        component="login_rcs_got_ocr",
        message="script_finished",
        log_name=LOG_NAME,
        result=result,
        window_title=window_title,
        backend=backend,
        elapsed_ms=f"{(time.time() - script_started_at) * 1000:.1f}",
    )
    return result


if __name__ == "__main__":
    exit_result = main()
    if exit_result != EXIT_SUCCESS:
        print(f"[EXIT] {exit_result}")
        sys.exit(1)
    sys.exit(0)
