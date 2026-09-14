"""workflow_3 에 등록된 VLM 서비스 연결 상태를 확인한다.

사용법:
  uv run python poc/workflow_3/check_vlm_connection.py
"""

import sys

import requests

from poc.workflow_3.vlm.flask_vlm import (
    ALL_VLM_SERVICES,
    VLMServiceEntry,
    resolve_service_api_key,
    resolve_service_proxy_url,
)


DEFAULT_TIMEOUT_SEC = 5.0


def models_url(api_url: str) -> str:
    """OpenAI-compatible API base URL 에서 models endpoint 를 만든다."""
    base_url = api_url.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/models"
    return f"{base_url}/v1/models"


def check_service(
    service: VLMServiceEntry,
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """서비스의 /v1/models 응답에 기대 모델이 있는지 확인한다."""
    api_url = resolve_service_proxy_url(service.route_slug)
    api_key = resolve_service_api_key(service.route_slug)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        response = requests.get(
            models_url(api_url),
            headers=headers,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return False, str(exc)
    except ValueError:
        return False, "JSON 응답 파싱 실패"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return False, "응답에 model data 목록이 없음"

    model_ids = [
        str(item.get("id") or "").strip()
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]
    if service.model_name in model_ids:
        return True, ""
    if model_ids:
        return (
            False,
            f"기대 모델 '{service.model_name}' 없음 (응답: {', '.join(model_ids)})",
        )
    return False, f"기대 모델 '{service.model_name}' 없음 (응답에 id 없음)"


def main() -> int:
    """활성 VLM 서비스를 모두 확인하고 실패가 있으면 1을 반환한다."""
    services = [service for service in ALL_VLM_SERVICES if service.enabled]
    if not services:
        print("[WARNING] 확인할 VLM 서비스가 없습니다.")
        return 1

    print(f"[INFO] VLM 연결 확인: {len(services)}개 서비스")
    failed = 0
    for service in services:
        ok, reason = check_service(service)
        if ok:
            print(
                f"[OK] {service.route_slug}: {service.model_name} "
                f"({models_url(resolve_service_proxy_url(service.route_slug))})"
            )
            continue
        failed += 1
        print(f"[FAIL] {service.route_slug}: {reason}")

    print(f"[INFO] 결과: {len(services) - failed} success / {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
