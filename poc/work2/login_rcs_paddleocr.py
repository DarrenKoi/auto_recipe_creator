"""RCS 로그인 창 텍스트 전용 OCR 스크립트.

이미 떠 있는 `Remote Control System` 로그인 창을 캡처하고,
`paddleocr-vl-1.5` 로 스크린샷 내 텍스트만 읽는다.

사용법:
  1. uv run python poc/work2/open_rcs.py
  2. uv run python poc/work2/login_rcs_paddleocr.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from poc.work2.flask_vlm import get_service_by_slug
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
from poc.work2.vlm_client import Work2VLMClient


load_dotenv()

SERVICE_SLUG = "paddleocr-vl-1.5"
OCR_PROMPT = os.getenv("LOGIN_RCS_PADDLEOCR_PROMPT", "OCR:").strip() or "OCR:"
MAX_TOKENS = int(os.getenv("LOGIN_RCS_PADDLEOCR_MAX_TOKENS", "4096"))
DEBUG_IMAGE_DIR = Path(__file__).parent / "debug_images" / "login_rcs_paddleocr"
LOG_NAME = Path(__file__).stem

EXIT_SUCCESS = "success"
EXIT_LOGIN_WINDOW_NOT_FOUND = "login_window_not_found"
EXIT_LOGIN_WINDOW_ACTIVATE_FAILED = "login_window_activate_failed"
EXIT_CAPTURE_FAILED = "capture_failed"
EXIT_OCR_REQUEST_ERROR = "ocr_request_error"
EXIT_OCR_EMPTY = "ocr_empty"


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


def _run_login_ocr(login_window, window_title: str, backend: str) -> str:
    """로그인 창을 OCR 전용으로 읽고 텍스트 결과를 저장한다."""
    started_at = time.time()
    debug_stamp = make_timestamp_tag(started_at)
    service_entry = get_service_by_slug(SERVICE_SLUG)
    if service_entry is None:
        print(f"[ERROR] OCR 서비스 설정을 찾지 못했습니다: {SERVICE_SLUG}")
        return EXIT_OCR_REQUEST_ERROR

    if not activate_window(
        login_window,
        debug_label=f"login_window paddleocr recapture backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 재활성화 실패: title={window_title!r}, backend={backend}")
        log_work2_event(
            component="login_rcs_paddleocr",
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
        debug_label=f"login_window paddleocr screenshot backend={backend} title={window_title!r}",
    ):
        print(f"[ERROR] 로그인 창 foreground 활성화 실패: title={window_title!r}, backend={backend}")
        log_work2_event(
            component="login_rcs_paddleocr",
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
            component="login_rcs_paddleocr",
            message="capture_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            error=exc,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_CAPTURE_FAILED

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

    print(
        f"[INFO] 로그인 창 OCR 시작: backend={backend}, title={window_title!r}, "
        f"service_slug={SERVICE_SLUG}, prompt={OCR_PROMPT!r}"
    )
    client = Work2VLMClient(
        service_slug=SERVICE_SLUG,
        timeout_sec=120.0,
        log_name=LOG_NAME,
    )

    try:
        response = client.chat_with_image_path(
            image_path=webp_path,
            system_message="",
            user_text=OCR_PROMPT,
            image_mime="image/webp",
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        print(f"[ERROR] PaddleOCR 요청 실패: {exc}")
        log_work2_event(
            component="login_rcs_paddleocr",
            message="ocr_request_failed",
            level="error",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            service_slug=SERVICE_SLUG,
            error=exc,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_OCR_REQUEST_ERROR

    raw_text = response.text.strip()
    normalized_lines = _normalize_lines(raw_text)
    save_debug_text(raw_response_path, raw_text)
    save_debug_json(
        result_json_path,
        {
            "service_slug": response.service_slug,
            "model_name": response.model_name,
            "api_url": response.api_url,
            "endpoint": client.endpoint,
            "window_title": window_title,
            "backend": backend,
            "prompt_text": OCR_PROMPT,
            "raw_text": raw_text,
            "normalized_lines": normalized_lines,
            "line_count": len(normalized_lines),
            "token_usage": response.token_usage,
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
        },
    )

    _print_ocr_preview(normalized_lines)
    print(f"[INFO] OCR raw response 저장: {raw_response_path}")
    print(f"[INFO] OCR summary 저장: {result_json_path}")
    print(f"[INFO] login_rcs_paddleocr 소요: {format_elapsed_ms(started_at)}")

    if not raw_text:
        log_work2_event(
            component="login_rcs_paddleocr",
            message="ocr_empty",
            level="warning",
            log_name=LOG_NAME,
            title=window_title,
            backend=backend,
            service_slug=SERVICE_SLUG,
            elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
        )
        return EXIT_OCR_EMPTY

    log_work2_event(
        component="login_rcs_paddleocr",
        message="ocr_finished",
        log_name=LOG_NAME,
        title=window_title,
        backend=backend,
        service_slug=SERVICE_SLUG,
        line_count=len(normalized_lines),
        elapsed_ms=f"{(time.time() - started_at) * 1000:.1f}",
    )
    return EXIT_SUCCESS


def main() -> str:
    """이미 열려 있는 로그인 창을 OCR 전용으로 읽는다."""
    script_started_at = time.time()
    log_work2_event(
        component="login_rcs_paddleocr",
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
            component="login_rcs_paddleocr",
            message="login_window_not_found",
            level="error",
            log_name=LOG_NAME,
            title_prefix=WINDOW_TITLE_PREFIX,
        )
        return EXIT_LOGIN_WINDOW_NOT_FOUND

    result = _run_login_ocr(login_window, window_title, backend)
    log_work2_event(
        component="login_rcs_paddleocr",
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
