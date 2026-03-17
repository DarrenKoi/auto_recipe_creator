"""Flask proxy VLM 연결 점검 스크립트.

Flask `/api/vlm_serve/health` 와 각 proxy route 의 `/v1/models` 응답을 점검한다.
VLM 서비스 registry 및 Flask URL 설정은 `flask_vlm.py` 에서 가져온다.

필요하면 `CONNECTION_CHECK_SERVICES=slug1,slug2,...` 환경변수로 대상 route 를 제한할 수 있다.

동작:
  1) Flask proxy health endpoint (/api/vlm_serve/health) 를 호출하여 전체 상태 확인
  2) health payload 의 서비스 목록을 기준으로 proxy route `/v1/models` 를 직접 확인
  3) `health_status` + proxy probe 결과를 함께 테이블 형태로 출력

사용법:
  uv run python poc/work2/connection_check.py
"""

import os
import time

import requests

from poc.work2.flask_vlm import (
    ALL_VLM_SERVICES,
    DEFAULT_OCR_MODEL_NAME,
    DEFAULT_OCR_SERVICE,
    DEFAULT_SCREEN_ANALYSIS_MODEL_NAME,
    DEFAULT_SCREEN_ANALYSIS_SERVICE,
    DEFAULT_VLM_HEALTH_TIMEOUT_SEC,
    fetch_vlm_health,
    get_service_by_slug,
    normalize_vlm_health_entries,
    resolve_flask_api_base_url,
    resolve_service_proxy_url,
    resolve_vlm_health_url,
)

TIMEOUT_SEC = DEFAULT_VLM_HEALTH_TIMEOUT_SEC
SEPARATOR = "-" * 80


def _parse_requested_services() -> tuple[str, ...]:
    """환경변수에서 요청된 service slug 목록을 파싱한다."""
    raw = os.environ.get("CONNECTION_CHECK_SERVICES", "").strip()
    if raw:
        return tuple(
            item.strip()
            for item in raw.replace(";", ",").split(",")
            if item.strip()
        )
    return ()


def _fallback_target_row(service_slug: str) -> dict[str, object]:
    """health endpoint 를 못 쓸 때 사용할 최소 대상 row 를 구성한다."""
    service_entry = get_service_by_slug(service_slug)
    proxy_registered = service_entry is not None and service_entry.enabled
    return {
        "service": service_slug,
        "display_name": service_entry.display_name if service_entry else service_slug,
        "expected_model": service_entry.model_name if service_entry else "",
        "health_status": "",
        "proxy_registered": proxy_registered,
        "config_known": service_entry is not None,
        "config_enabled": None if service_entry is None else service_entry.enabled,
        "api_url": resolve_service_proxy_url(service_slug) if proxy_registered else "",
        "upstream_base_url": "",
        "reason": "",
        "source_env": "",
        "runtime": "",
    }


def _resolve_target_services(health_body: dict | None) -> list[dict[str, object]]:
    """점검 대상 서비스 목록을 health payload 우선으로 결정한다."""
    requested_services = _parse_requested_services()

    if health_body:
        health_rows = normalize_vlm_health_entries(health_body)
        if requested_services:
            health_map = {str(row["service"]): row for row in health_rows}
            return [
                dict(health_map.get(service_slug) or _fallback_target_row(service_slug))
                for service_slug in requested_services
            ]
        return [dict(row) for row in health_rows]

    resolved: list[dict[str, object]] = []
    seen: set[str] = set()
    fallback_services = requested_services or tuple(
        service.route_slug for service in ALL_VLM_SERVICES if service.enabled
    )
    for service_slug in fallback_services:
        if service_slug in seen:
            continue
        seen.add(service_slug)
        resolved.append(_fallback_target_row(service_slug))
    return resolved


def _probe_url(url: str) -> dict:
    """URL 에 GET 요청을 보내고 결과를 반환한다."""
    result: dict = {"url": url, "ok": False, "status_code": None, "latency_ms": None, "body": None, "error": None}
    try:
        start = time.time()
        resp = requests.get(url, timeout=TIMEOUT_SEC)
        latency_ms = (time.time() - start) * 1000
        result["status_code"] = resp.status_code
        result["latency_ms"] = round(latency_ms, 1)
        try:
            result["body"] = resp.json()
        except ValueError:
            result["body"] = resp.text[:200]
        result["ok"] = resp.status_code < 400
    except requests.RequestException as exc:
        result["error"] = str(exc)
    return result


def _build_models_url(base_url: str) -> str:
    """OpenAI-compatible base URL 에서 `/models` probe URL 을 만든다."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def check_flask_health(flask_base_url: str) -> dict | None:
    """Flask /api/vlm_serve/health 엔드포인트를 호출한다."""
    health_url = resolve_vlm_health_url(flask_base_url=flask_base_url)
    print(f"\n[INFO] Flask VLM health endpoint 호출: {health_url}")
    started = time.time()
    try:
        payload = fetch_vlm_health(flask_base_url=flask_base_url, timeout_sec=TIMEOUT_SEC)
    except requests.RequestException as exc:
        error_message = str(exc)
        print(f"[ERROR] Flask health 호출 실패: {error_message}")
        return None
    except ValueError as exc:
        print(f"[ERROR] Flask health 응답 파싱 실패: {exc}")
        return None

    latency_ms = round((time.time() - started) * 1000, 1)
    print(f"[INFO] Flask health 응답 수신 ({latency_ms}ms)")
    return payload


def check_proxy_models(flask_base_url: str, target_services: list[dict[str, object]]) -> list[dict]:
    """각 VLM 서비스의 proxy route 를 통해 /v1/models 를 호출한다."""
    results = []
    for target in target_services:
        route_slug = str(target["service"])
        expected_model = str(target["expected_model"])
        proxy_base_url = str(target.get("api_url") or "").strip()
        if not proxy_base_url:
            proxy_base_url = resolve_service_proxy_url(
                route_slug,
                flask_base_url=flask_base_url,
            )
        proxy_url = _build_models_url(proxy_base_url)
        probe = dict(target)
        probe["url"] = proxy_url

        if not bool(target.get("proxy_registered")):
            probe.update(
                {
                    "ok": False,
                    "status_code": None,
                    "latency_ms": None,
                    "body": None,
                    "error": None,
                    "detected_models": [],
                    "model_match": False,
                    "skipped": True,
                }
            )
            results.append(probe)
            continue

        probe.update(_probe_url(proxy_url))
        probe["skipped"] = False

        detected_models = []
        if isinstance(probe.get("body"), dict):
            data = probe["body"].get("data", [])
            if isinstance(data, list):
                detected_models = [item.get("id", "") for item in data if isinstance(item, dict)]
        probe["detected_models"] = detected_models
        probe["model_match"] = expected_model in detected_models if detected_models else False
        results.append(probe)
    return results


def print_summary(
    health_body: dict | None,
    proxy_results: list[dict],
    flask_base_url: str,
    target_services: list[dict[str, object]],
) -> None:
    """점검 결과를 요약 출력한다."""
    registered_services = set(health_body.get("registered_vlms", [])) if health_body else set()
    target_service_names = ", ".join(str(item["service"]) for item in target_services) or "(없음)"

    print(f"\n{SEPARATOR}")
    print("  VLM 연결 점검 결과")
    print(f"{SEPARATOR}")
    print(f"  Flask API Base URL: {flask_base_url}")
    print(f"  점검 대상: {target_service_names}")

    # Flask health 요약
    if health_body:
        serving_now = health_body.get("serving_now", [])
        registered = health_body.get("registered_vlms", [])
        print(f"  등록 서비스: {', '.join(registered) if registered else '(없음)'}")
        print(f"  현재 서빙중: {', '.join(serving_now) if serving_now else '(없음)'}")
    else:
        print("  Flask health: 응답 없음")

    # 서비스별 상세 테이블
    print(
        f"\n  {'서비스':<18} {'모델':<22} {'health':<14} {'route':<8} "
        f"{'probe':<10} {'모델 일치':<10} {'응답시간':<10} {'비고'}"
    )
    print(
        f"  {'─' * 18} {'─' * 22} {'─' * 14} {'─' * 8} "
        f"{'─' * 10} {'─' * 10} {'─' * 10} {'─' * 18}"
    )

    all_ok = True
    for probe in proxy_results:
        service = str(probe["service"])
        health_status = str(probe.get("health_status") or "-")
        model_name = str(probe.get("expected_model") or probe.get("display_name") or service)
        route_registered = bool(probe.get("proxy_registered")) or service in registered_services
        if probe.get("skipped"):
            status = "SKIP"
        elif probe["ok"] and probe["model_match"]:
            status = "OK"
        elif probe["ok"] and not probe["model_match"]:
            status = "MISMATCH"
            all_ok = False
        else:
            status = "FAIL"
            all_ok = False

        if str(probe.get("expected_model", "")).strip():
            match_str = "O" if probe["model_match"] else "X"
        else:
            match_str = "-"
        registered_str = "O" if route_registered else "X"
        latency_str = f"{probe['latency_ms']}ms" if probe["latency_ms"] is not None else "-"
        note = ""
        if probe.get("skipped"):
            note = str(probe.get("reason") or "proxy route 미등록")[:18]
        elif probe.get("error"):
            note = probe["error"][:30]
        elif not probe["ok"]:
            note = f"HTTP {probe.get('status_code')}"
        elif probe["detected_models"] and not probe["model_match"]:
            note = f"got: {', '.join(probe['detected_models'][:2])}"
        elif probe.get("config_known") is False:
            note = "unknown service slug"
        elif probe.get("config_enabled") is False and service not in registered_services:
            note = "local enabled=False"

        print(
            f"  {service:<18} {model_name[:22]:<22} {health_status[:14]:<14} {registered_str:<8} "
            f"{status:<10} {match_str:<10} {latency_str:<10} {note}"
        )

    # 공유 pipeline 에서 사용 중인 서비스 표시
    print(
        f"\n  configured screen analysis VLM: "
        f"{DEFAULT_SCREEN_ANALYSIS_SERVICE} ({DEFAULT_SCREEN_ANALYSIS_MODEL_NAME})"
    )
    print(f"  configured OCR VLM:     {DEFAULT_OCR_SERVICE} ({DEFAULT_OCR_MODEL_NAME})")

    print(f"{SEPARATOR}")
    if all_ok:
        print("  결과: 모든 VLM 서비스 정상")
    else:
        print("  결과: 일부 서비스 연결 실패 — 위 비고 확인")
    print(SEPARATOR)


def main() -> None:
    """VLM 연결 점검을 수행한다."""
    flask_base_url = resolve_flask_api_base_url()

    print(f"[INFO] VLM 연결 점검 시작 (Flask API: {flask_base_url})")

    # 1) Flask health 호출
    health_body = check_flask_health(flask_base_url)
    target_services = _resolve_target_services(health_body)
    print(f"[INFO] 대상 서비스: {', '.join(str(item['service']) for item in target_services)}")

    # 2) 각 서비스 proxy route 직접 호출
    print(f"\n[INFO] 각 VLM 서비스 proxy route 직접 호출 중...")
    proxy_results = check_proxy_models(flask_base_url, target_services)

    # 3) 결과 출력
    print_summary(health_body, proxy_results, flask_base_url, target_services)


if __name__ == "__main__":
    main()
