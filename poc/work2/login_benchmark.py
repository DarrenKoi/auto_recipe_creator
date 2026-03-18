"""RCS 로그인 화면용 모델 비교 헬퍼.

Windows 의 실제 창 탐색/캡처는 `login_rcs.py` 가 담당하고,
이 모듈은 캡처된 이미지 1장을 여러 서비스에 동일 JSON 계약으로 보내어
모델별 debug artifact 와 로그를 분리 저장하는 역할만 맡는다.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from poc.work2 import resolve_debug_model_name
from poc.work2.flask_vlm import (
    get_enabled_services_by_role,
    get_service_by_slug,
    resolve_service_api_key,
    resolve_service_proxy_url,
)
from poc.work2.logger import log_work2_event
from poc.work2.prompts import (
    build_login_rcs_locator_prompt,
    build_login_rcs_ui_venus_prompt,
)
from poc.work2.util.debug_image_utils import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_json,
    save_debug_text,
    save_debug_webp,
    save_marked_image,
)
from poc.work2.util.image_utils import encode_image_webp
from poc.work2.util.json_utils import extract_json, parse_coords
from poc.work2.vlm_client import Work2VLMClient

PRIMARY_GUI_BENCHMARK_ROLE = "primary_gui"
DEFAULT_PRIMARY_GUI_SERVICE_SLUGS = tuple(
    service.route_slug
    for service in get_enabled_services_by_role(PRIMARY_GUI_BENCHMARK_ROLE)
)
OCR_LOGIN_EXTRA_INSTRUCTIONS = (
    "Read all visible text labels first, then map each label to the nearest matching control rectangle.",
    "For text-heavy rows, prioritize exact label-control pairing over coarse layout guesses.",
    "Keep the same JSON schema and coord_system as the GUI models.",
)

STATUS_OK = "ok"
STATUS_REQUEST_ERROR = "request_error"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SKIPPED_UNHEALTHY = "skipped_unhealthy"

HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("VLM_HEALTH_PROBE_TIMEOUT_SEC", "5.0"))


@dataclass(frozen=True)
class LoginBenchmarkResult:
    """모델 1회 실행 결과."""

    service_slug: str
    display_name: str
    model_name: str
    status: str
    detected_count: int
    target_count: int
    elapsed_ms: float
    raw_capture_path: Path
    vlm_input_path: Path
    overlay_path: Path | None
    raw_response_path: Path | None
    parsed_json_path: Path | None
    token_usage: dict[str, int]
    error: str = ""


def parse_service_slugs(raw: str | None, default: Iterable[str]) -> tuple[str, ...]:
    """환경변수/입력 문자열에서 서비스 slug 목록을 파싱한다."""
    if raw and raw.strip():
        candidates = raw.replace(";", ",").split(",")
    else:
        candidates = list(default)

    parsed: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for item in candidates:
        slug = str(item).strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        if get_service_by_slug(slug) is None:
            unknown.append(slug)
            continue
        parsed.append(slug)

    if unknown:
        raise ValueError(f"알 수 없는 service slug: {', '.join(unknown)}")

    return tuple(parsed)


def resolve_benchmark_service_slugs(raw: str | None = None) -> tuple[str, ...]:
    """primary GUI benchmark 기본 서비스 목록을 반환한다."""
    return parse_service_slugs(raw, DEFAULT_PRIMARY_GUI_SERVICE_SLUGS)


def build_prompt_for_service(
    service_slug: str,
    *,
    width: int,
    height: int,
    target_keys: Iterable[str],
) -> tuple[str, str]:
    """서비스 prompt family 에 따라 로그인 프롬프트를 구성한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        raise ValueError(f"알 수 없는 service slug: {service_slug}")

    if service_slug == "ui-venus":
        return build_login_rcs_ui_venus_prompt(
            width=width,
            height=height,
            target_keys=target_keys,
        )

    extra_instructions = None
    if service_entry.prompt_family == "ocr":
        extra_instructions = OCR_LOGIN_EXTRA_INSTRUCTIONS

    system_message, user_text = build_login_rcs_locator_prompt(
        width=width,
        height=height,
        target_keys=target_keys,
        extra_instructions=extra_instructions,
    )
    return system_message, user_text


def build_model_log_name(base_log_name: str, model_name: str) -> str:
    """모델명 기준 파일 로그명을 만든다."""
    return f"{(base_log_name or 'work2').strip() or 'work2'}_{resolve_debug_model_name(model_name)}"


def _probe_service_health(service_slug: str) -> tuple[bool, str]:
    """서비스의 사전 건강 상태를 빠르게 확인한다.

    Returns:
        (healthy, reason) 튜플. healthy=True 이면 모델이 서빙 중.
    """
    base_url = resolve_service_proxy_url(service_slug)
    if not base_url:
        return False, "proxy URL 미확인"

    svc = get_service_by_slug(service_slug)
    if svc is not None and svc.connection_mode == "direct":
        return True, f"direct preflight skipped: {svc.model_name}"

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        models_url = f"{normalized}/models"
    else:
        models_url = f"{normalized}/v1/models"

    headers: dict[str, str] = {}
    api_key = resolve_service_api_key(service_slug)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(models_url, headers=headers, timeout=HEALTH_PROBE_TIMEOUT_SEC)
    except requests.RequestException as exc:
        return False, f"probe 연결 실패: {exc}"

    if resp.status_code >= 400:
        return False, f"probe HTTP {resp.status_code}"

    try:
        body = resp.json()
    except ValueError:
        return False, "probe 응답이 JSON 이 아님"

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return False, f"probe 모델 목록 비어 있음: {str(body)[:120]}"

    model_ids = [item.get("id", "") for item in data if isinstance(item, dict)]
    return True, f"serving: {', '.join(model_ids)}"


def _make_skipped_result(
    service_slug: str,
    display_name: str,
    model_name: str,
    target_count: int,
    reason: str,
) -> LoginBenchmarkResult:
    """건강하지 않은 서비스를 건너뛸 때 사용하는 결과를 생성한다."""
    return LoginBenchmarkResult(
        service_slug=service_slug,
        display_name=display_name,
        model_name=model_name,
        status=STATUS_SKIPPED_UNHEALTHY,
        detected_count=0,
        target_count=target_count,
        elapsed_ms=0.0,
        raw_capture_path=Path(),
        vlm_input_path=Path(),
        overlay_path=None,
        raw_response_path=None,
        parsed_json_path=None,
        token_usage={},
        error=reason,
    )


def run_login_analysis_for_service(
    *,
    image: "Image.Image",
    service_slug: str,
    debug_image_dir: Path,
    debug_stamp: str,
    target_keys: Iterable[str],
    element_colors: dict[str, str],
    temperature: float,
    base_log_name: str,
    context_fields: dict[str, object] | None = None,
) -> LoginBenchmarkResult:
    """캡처된 로그인 이미지를 한 서비스로 분석한다."""
    context = context_fields or {}
    target_key_tuple = tuple(target_keys)
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        raise ValueError(f"알 수 없는 service slug: {service_slug}")

    log_name = build_model_log_name(base_log_name, service_entry.model_name)
    client = Work2VLMClient(service_slug=service_slug, log_name=log_name)

    raw_capture_path = debug_image_path(
        debug_image_dir,
        "login_rcs_capture.jpg",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    vlm_input_path = debug_image_path(
        debug_image_dir,
        "login_rcs_vlm_input.webp",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    raw_response_path = debug_image_path(
        debug_image_dir,
        "login_rcs_response.txt",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    parsed_json_path = debug_image_path(
        debug_image_dir,
        "login_rcs_response.json",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )
    overlay_path = debug_image_path(
        debug_image_dir,
        "login_rcs_overlay.jpg",
        model_name=client.model_name,
        timestamp_tag=debug_stamp,
    )

    save_debug_jpeg(image, raw_capture_path, log_name=log_name)
    save_debug_webp(image, vlm_input_path, log_name=log_name)

    image_b64, width, height = encode_image_webp(image)
    system_message, user_text = build_prompt_for_service(
        service_slug,
        width=width,
        height=height,
        target_keys=target_key_tuple,
    )

    print(
        f"[INFO] 로그인 모델 분석 시작: service={service_slug}, "
        f"model={client.model_name}, endpoint={client.endpoint}"
    )

    started_at = time.time()
    try:
        response = client.chat_with_image_b64(
            image_b64=image_b64,
            image_mime="image/webp",
            system_message=system_message,
            user_text=user_text,
            temperature=temperature,
        )
    except Exception as exc:
        elapsed_ms = (time.time() - started_at) * 1000
        log_work2_event(
            component="login_benchmark",
            message="vlm_request_failed",
            level="error",
            log_name=log_name,
            service=service_slug,
            model=client.model_name,
            elapsed_ms=f"{elapsed_ms:.1f}",
            error=exc,
            **context,
        )
        return LoginBenchmarkResult(
            service_slug=service_slug,
            display_name=service_entry.display_name,
            model_name=client.model_name,
            status=STATUS_REQUEST_ERROR,
            detected_count=0,
            target_count=len(target_key_tuple),
            elapsed_ms=elapsed_ms,
            raw_capture_path=raw_capture_path,
            vlm_input_path=vlm_input_path,
            overlay_path=None,
            raw_response_path=None,
            parsed_json_path=None,
            token_usage={},
            error=str(exc),
        )

    elapsed_ms = (time.time() - started_at) * 1000
    save_debug_text(raw_response_path, response.text)
    print(f"[INFO] VLM 응답 수신: tokens={response.token_usage or {}}")
    print(f"[INFO] 원문 응답:\n{response.text}\n")

    try:
        parsed_json = extract_json(response.text)
    except Exception as exc:
        log_work2_event(
            component="login_benchmark",
            message="vlm_json_parse_failed",
            level="error",
            log_name=log_name,
            service=service_slug,
            model=response.model_name or client.model_name,
            elapsed_ms=f"{elapsed_ms:.1f}",
            raw_response_path=raw_response_path,
            error=exc,
            **context,
        )
        return LoginBenchmarkResult(
            service_slug=service_slug,
            display_name=service_entry.display_name,
            model_name=response.model_name or client.model_name,
            status=STATUS_PARSE_ERROR,
            detected_count=0,
            target_count=len(target_key_tuple),
            elapsed_ms=elapsed_ms,
            raw_capture_path=raw_capture_path,
            vlm_input_path=vlm_input_path,
            overlay_path=None,
            raw_response_path=raw_response_path,
            parsed_json_path=None,
            token_usage=dict(response.token_usage or {}),
            error=str(exc),
        )

    print(f"[INFO] 파싱된 JSON:\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}\n")
    save_debug_json(parsed_json_path, parsed_json)
    parsed_coords = parse_coords(parsed_json, list(target_key_tuple), width, height)
    detected_count = sum(
        1 for key in target_key_tuple if key in parsed_coords and isinstance(parsed_coords[key], dict)
    )

    save_marked_image(image, parsed_coords, element_colors, overlay_path)
    log_work2_event(
        component="login_benchmark",
        message="analysis_finished",
        log_name=log_name,
        service=service_slug,
        model=response.model_name or client.model_name,
        detected=detected_count,
        target_count=len(target_key_tuple),
        elapsed_ms=f"{elapsed_ms:.1f}",
        raw_capture_path=raw_capture_path,
        vlm_input_path=vlm_input_path,
        overlay_path=overlay_path,
        raw_response_path=raw_response_path,
        parsed_json_path=parsed_json_path,
        **context,
    )
    return LoginBenchmarkResult(
        service_slug=service_slug,
        display_name=service_entry.display_name,
        model_name=response.model_name or client.model_name,
        status=STATUS_OK,
        detected_count=detected_count,
        target_count=len(target_key_tuple),
        elapsed_ms=elapsed_ms,
        raw_capture_path=raw_capture_path,
        vlm_input_path=vlm_input_path,
        overlay_path=overlay_path,
        raw_response_path=raw_response_path,
        parsed_json_path=parsed_json_path,
        token_usage=dict(response.token_usage or {}),
    )


def run_login_benchmark(
    *,
    image: "Image.Image",
    service_slugs: Iterable[str],
    debug_image_dir: Path,
    debug_stamp: str,
    target_keys: Iterable[str],
    element_colors: dict[str, str],
    temperature: float,
    base_log_name: str,
    context_fields: dict[str, object] | None = None,
) -> list[LoginBenchmarkResult]:
    """여러 서비스에 동일 로그인 이미지를 보내어 결과를 수집한다."""
    resolved_slugs = parse_service_slugs(None, service_slugs)
    target_key_list = list(target_keys)
    results: list[LoginBenchmarkResult] = []
    for service_slug in resolved_slugs:
        service_entry = get_service_by_slug(service_slug)
        if service_entry is None:
            continue

        healthy, reason = _probe_service_health(service_slug)
        if not healthy:
            print(
                f"[WARNING] {service_slug} 건강 점검 실패 → 건너뜀: {reason}"
            )
            log_work2_event(
                component="login_benchmark",
                message="service_skipped_unhealthy",
                level="warning",
                log_name=base_log_name,
                service=service_slug,
                model=service_entry.model_name,
                reason=reason,
            )
            results.append(
                _make_skipped_result(
                    service_slug=service_slug,
                    display_name=service_entry.display_name,
                    model_name=service_entry.model_name,
                    target_count=len(target_key_list),
                    reason=reason,
                )
            )
            continue

        print(f"[INFO] {service_slug} 건강 점검 통과: {reason}")
        results.append(
            run_login_analysis_for_service(
                image=image,
                service_slug=service_slug,
                debug_image_dir=debug_image_dir,
                debug_stamp=debug_stamp,
                target_keys=target_key_list,
                element_colors=element_colors,
                temperature=temperature,
                base_log_name=base_log_name,
                context_fields=context_fields,
            )
        )
    return results


def print_benchmark_summary(results: Iterable[LoginBenchmarkResult]) -> None:
    """모델별 로그인 벤치마크 결과를 표로 출력한다."""
    rows = list(results)
    print("\n" + "=" * 96)
    print("  로그인 primary GUI 벤치마크 요약")
    print("=" * 96)
    print(
        f"  {'service':<16} {'model':<22} {'status':<14} "
        f"{'detected':<10} {'latency':<12} {'response'}"
    )
    print(
        f"  {'─' * 16} {'─' * 22} {'─' * 14} "
        f"{'─' * 10} {'─' * 12} {'─' * 20}"
    )

    for item in rows:
        detected = f"{item.detected_count}/{item.target_count}"
        latency = f"{item.elapsed_ms:.1f}ms"
        if item.status == STATUS_SKIPPED_UNHEALTHY:
            note = item.error or "unhealthy"
        else:
            note = str(item.raw_response_path or item.error or "-")
        print(
            f"  {item.service_slug:<16} {item.model_name[:22]:<22} {item.status:<14} "
            f"{detected:<10} {latency:<12} {note}"
        )
    print("=" * 96)


def benchmark_has_success(results: Iterable[LoginBenchmarkResult]) -> bool:
    """하나 이상 의미 있는 검출 결과가 있으면 True."""
    return any(
        item.status == STATUS_OK and item.detected_count > 0
        for item in results
    )


def resolve_service_slugs_from_env(
    env_name: str = "RCS_LOGIN_SERVICE_SLUGS",
) -> tuple[str, ...]:
    """환경변수 또는 primary GUI 기본값에서 서비스 목록을 읽는다."""
    return resolve_benchmark_service_slugs(os.environ.get(env_name, ""))


__all__ = [
    "DEFAULT_PRIMARY_GUI_SERVICE_SLUGS",
    "LoginBenchmarkResult",
    "PRIMARY_GUI_BENCHMARK_ROLE",
    "STATUS_OK",
    "STATUS_PARSE_ERROR",
    "STATUS_REQUEST_ERROR",
    "STATUS_SKIPPED_UNHEALTHY",
    "benchmark_has_success",
    "build_model_log_name",
    "build_prompt_for_service",
    "parse_service_slugs",
    "print_benchmark_summary",
    "resolve_benchmark_service_slugs",
    "resolve_service_slugs_from_env",
    "run_login_analysis_for_service",
    "run_login_benchmark",
]
