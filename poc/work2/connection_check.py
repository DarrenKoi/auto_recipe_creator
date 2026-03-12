"""Flask API 경유 VLM 모델 연결 상태 점검 스크립트.

등록된 모든 VLM 서비스에 대해:
  1) Flask proxy health endpoint (/api/vlm_serve/health) 를 호출하여 전체 상태 확인
  2) 각 서비스의 proxy route 를 통해 /v1/models 직접 확인
  3) 결과를 테이블 형태로 출력

사용법:
  uv run python poc/work2/connection_check.py
"""

import time

import requests

from flask_api.vlm_serve.config import ENABLED_VLM_SERVICES
from poc.work2.flask_vlm import (
    DEFAULT_OCR_VLM_MODEL_NAME,
    DEFAULT_OCR_VLM_SERVICE,
    DEFAULT_PRIMARY_VLM_MODEL_NAME,
    DEFAULT_PRIMARY_VLM_SERVICE,
    load_work2_env,
    resolve_work2_flask_api_base_url,
)

# 점검 대상: config.py 에서 enabled=True 인 서비스만 점검
VLM_SERVICES = [
    (s.route_slug, s.model_name) for s in ENABLED_VLM_SERVICES
]

TIMEOUT_SEC = 5.0
SEPARATOR = "-" * 80


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


def check_flask_health(flask_base_url: str) -> dict | None:
    """Flask /api/vlm_serve/health 엔드포인트를 호출한다."""
    health_url = f"{flask_base_url}/vlm_serve/health"
    print(f"\n[INFO] Flask VLM health endpoint 호출: {health_url}")
    result = _probe_url(health_url)
    if not result["ok"]:
        error_message = result.get("error") or f"HTTP {result.get('status_code')}"
        print(f"[ERROR] Flask health 호출 실패: {error_message}")
        return None
    print(f"[INFO] Flask health 응답 수신 ({result['latency_ms']}ms)")
    return result.get("body")


def check_proxy_models(flask_base_url: str) -> list[dict]:
    """각 VLM 서비스의 proxy route 를 통해 /v1/models 를 호출한다."""
    results = []
    for route_slug, expected_model in VLM_SERVICES:
        proxy_url = f"{flask_base_url}/vlm_serve/{route_slug}/v1/models"
        probe = _probe_url(proxy_url)
        probe["service"] = route_slug
        probe["expected_model"] = expected_model

        detected_models = []
        if isinstance(probe.get("body"), dict):
            data = probe["body"].get("data", [])
            if isinstance(data, list):
                detected_models = [item.get("id", "") for item in data if isinstance(item, dict)]
        probe["detected_models"] = detected_models
        probe["model_match"] = expected_model in detected_models if detected_models else False
        results.append(probe)
    return results


def print_summary(health_body: dict | None, proxy_results: list[dict], flask_base_url: str) -> None:
    """점검 결과를 요약 출력한다."""
    print(f"\n{SEPARATOR}")
    print("  VLM 연결 점검 결과")
    print(f"{SEPARATOR}")
    print(f"  Flask API Base URL: {flask_base_url}")

    # Flask health 요약
    if health_body:
        serving_now = health_body.get("serving_now", [])
        registered = health_body.get("registered_vlms", [])
        print(f"  등록 서비스: {', '.join(registered) if registered else '(없음)'}")
        print(f"  현재 서빙중: {', '.join(serving_now) if serving_now else '(없음)'}")
    else:
        print("  Flask health: 응답 없음")

    # 서비스별 상세 테이블
    print(f"\n  {'서비스':<20} {'상태':<12} {'모델 일치':<10} {'응답시간':<10} {'비고'}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 30}")

    all_ok = True
    for probe in proxy_results:
        service = probe["service"]
        if probe["ok"] and probe["model_match"]:
            status = "OK"
        elif probe["ok"] and not probe["model_match"]:
            status = "MISMATCH"
            all_ok = False
        else:
            status = "FAIL"
            all_ok = False

        match_str = "O" if probe["model_match"] else "X"
        latency_str = f"{probe['latency_ms']}ms" if probe["latency_ms"] is not None else "-"
        note = ""
        if probe.get("error"):
            note = probe["error"][:30]
        elif not probe["ok"]:
            note = f"HTTP {probe.get('status_code')}"
        elif probe["detected_models"] and not probe["model_match"]:
            note = f"got: {', '.join(probe['detected_models'][:2])}"

        print(f"  {service:<20} {status:<12} {match_str:<10} {latency_str:<10} {note}")

    # work2 pipeline 에서 사용 중인 서비스 표시
    print(f"\n  work2 primary VLM: {DEFAULT_PRIMARY_VLM_SERVICE} ({DEFAULT_PRIMARY_VLM_MODEL_NAME})")
    print(f"  work2 OCR VLM:     {DEFAULT_OCR_VLM_SERVICE} ({DEFAULT_OCR_VLM_MODEL_NAME})")

    print(f"{SEPARATOR}")
    if all_ok:
        print("  결과: 모든 VLM 서비스 정상")
    else:
        print("  결과: 일부 서비스 연결 실패 — 위 비고 확인")
    print(SEPARATOR)


def main() -> None:
    """VLM 연결 점검을 수행한다."""
    load_work2_env()
    flask_base_url = resolve_work2_flask_api_base_url()

    print(f"[INFO] VLM 연결 점검 시작 (Flask API: {flask_base_url})")

    # 1) Flask health 호출
    health_body = check_flask_health(flask_base_url)

    # 2) 각 서비스 proxy route 직접 호출
    print(f"\n[INFO] 각 VLM 서비스 proxy route 직접 호출 중...")
    proxy_results = check_proxy_models(flask_base_url)

    # 3) 결과 출력
    print_summary(health_body, proxy_results, flask_base_url)


if __name__ == "__main__":
    main()
