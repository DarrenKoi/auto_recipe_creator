"""code-server forward URL 경유 VLM 연결 점검 스크립트.

Flask proxy 를 거치지 않고 code-server 의 forwarded address
(`http://host/proxy/{port}/`) 로 직접 접근해 각 VLM 서비스 상태를 확인한다.

기본 대상 주소:
  http://itc-1stop-solution-gpu-image-vscode.aipp02.skhynix.com/proxy/{port}/

환경변수:
  CONNECTION_CHECK_SERVICES=ui-venus,paddleocr-vl-1.5
  CODE_SERVER_FORWARD_ROOT_URL=http://itc-1stop-solution-gpu-image-vscode.aipp02.skhynix.com
  CODE_SERVER_FORWARD_PATH_TEMPLATE=/proxy/{port}/
  CODE_SERVER_FORWARD_TIMEOUT_SEC=5.0
  CODE_SERVER_FORWARD_API_KEY=<optional default bearer token>
  CODE_SERVER_FORWARD_UI_VENUS_BASE_URL=http://host/proxy/8001/
  CODE_SERVER_FORWARD_UI_VENUS_API_KEY=<optional per-service bearer token>

호환용 환경변수:
  기존 `FORWARD_URL_*` 값을 이미 쓰고 있으면 그대로 재사용할 수 있다.

사용법:
  uv run python poc/work2/code_server_forward_connection_check.py
"""

from dataclasses import dataclass
import os
import time

import requests

from flask_api.vlm_serve.config import ENABLED_VLM_SERVICES, get_service_by_slug
from poc.work2.flask_vlm import (
    DEFAULT_OCR_VLM_MODEL_NAME,
    DEFAULT_OCR_VLM_SERVICE,
    DEFAULT_PRIMARY_VLM_MODEL_NAME,
    DEFAULT_PRIMARY_VLM_SERVICE,
)

DEFAULT_CODE_SERVER_FORWARD_ROOT_URL = (
    "http://itc-1stop-solution-gpu-image-vscode.aipp02.skhynix.com"
)
DEFAULT_PATH_TEMPLATE = "/proxy/{port}/"
DEFAULT_TIMEOUT_SEC = 5.0
SEPARATOR = "-" * 132
_PATH_TEMPLATE_WARNING_EMITTED = False


@dataclass(frozen=True)
class ProbeConfig:
    """code-server forward URL probe 설정."""

    root_url: str
    path_template: str
    timeout_sec: float
    default_api_key: str


@dataclass(frozen=True)
class ProbeTarget:
    """서비스별 probe 대상 정의."""

    service: str
    display_name: str
    expected_model: str
    port: int | None
    config_known: bool
    config_enabled: bool | None


def _service_env_prefix(service_slug: str) -> str:
    """service slug 를 env prefix 형태로 변환한다."""
    return service_slug.replace("-", "_").replace(".", "_").upper()


def _read_env(primary_name: str, fallback_name: str = "") -> str:
    """우선순위에 따라 환경변수 값을 읽는다."""
    primary_value = os.environ.get(primary_name, "").strip()
    if primary_value:
        return primary_value
    if fallback_name:
        return os.environ.get(fallback_name, "").strip()
    return ""


def _parse_timeout_sec(raw_value: str) -> float:
    """timeout 초 값을 파싱한다."""
    try:
        return max(float(raw_value), 0.1)
    except ValueError:
        print(
            f"[WARNING] timeout 값이 올바르지 않습니다: {raw_value!r} "
            f"-> {DEFAULT_TIMEOUT_SEC}초 사용"
        )
        return DEFAULT_TIMEOUT_SEC


def _resolve_probe_config() -> ProbeConfig:
    """스크립트 실행 설정을 계산한다."""
    root_url = (
        _read_env("CODE_SERVER_FORWARD_ROOT_URL", "FORWARD_URL_ROOT_URL").rstrip("/")
        or DEFAULT_CODE_SERVER_FORWARD_ROOT_URL
    )
    path_template = (
        _read_env("CODE_SERVER_FORWARD_PATH_TEMPLATE", "FORWARD_URL_PATH_TEMPLATE")
        or DEFAULT_PATH_TEMPLATE
    )
    timeout_raw = (
        _read_env("CODE_SERVER_FORWARD_TIMEOUT_SEC", "FORWARD_URL_TIMEOUT_SEC")
        or str(DEFAULT_TIMEOUT_SEC)
    )
    default_api_key = _read_env("CODE_SERVER_FORWARD_API_KEY", "FORWARD_URL_API_KEY")
    return ProbeConfig(
        root_url=root_url,
        path_template=path_template,
        timeout_sec=_parse_timeout_sec(timeout_raw),
        default_api_key=default_api_key,
    )


def _parse_requested_services() -> tuple[str, ...]:
    """환경변수에서 요청된 service slug 목록을 파싱한다."""
    raw = os.environ.get("CONNECTION_CHECK_SERVICES", "").strip()
    if not raw:
        return ()
    return tuple(
        item.strip()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    )


def _build_target(service_slug: str) -> ProbeTarget:
    """service slug 기준 대상 row 를 구성한다."""
    service_entry = get_service_by_slug(service_slug)
    if service_entry is None:
        return ProbeTarget(
            service=service_slug,
            display_name=service_slug,
            expected_model="",
            port=None,
            config_known=False,
            config_enabled=None,
        )

    return ProbeTarget(
        service=service_entry.route_slug,
        display_name=service_entry.display_name,
        expected_model=service_entry.model_name,
        port=service_entry.upstream_port,
        config_known=True,
        config_enabled=service_entry.enabled,
    )


def _resolve_targets() -> list[ProbeTarget]:
    """점검 대상 서비스를 결정한다."""
    requested_services = _parse_requested_services()
    if requested_services:
        seen: set[str] = set()
        targets: list[ProbeTarget] = []
        for service_slug in requested_services:
            if service_slug in seen:
                continue
            seen.add(service_slug)
            targets.append(_build_target(service_slug))
        return targets

    return [
        _build_target(service.route_slug)
        for service in ENABLED_VLM_SERVICES
    ]


def _render_forward_path(config: ProbeConfig, target: ProbeTarget) -> str:
    """forward path template 을 렌더링한다."""
    global _PATH_TEMPLATE_WARNING_EMITTED

    values = {
        "port": target.port,
        "service": target.service,
        "model": target.expected_model,
    }
    try:
        rendered = config.path_template.format(**values)
    except (KeyError, ValueError) as exc:
        if not _PATH_TEMPLATE_WARNING_EMITTED:
            print(
                f"[WARNING] CODE_SERVER_FORWARD_PATH_TEMPLATE 형식이 올바르지 않습니다: {exc} "
                f"-> 기본값 {DEFAULT_PATH_TEMPLATE!r} 사용"
            )
            _PATH_TEMPLATE_WARNING_EMITTED = True
        rendered = DEFAULT_PATH_TEMPLATE.format(**values)

    rendered = rendered.strip()
    if not rendered:
        return DEFAULT_PATH_TEMPLATE.format(**values)
    if not rendered.startswith("/"):
        return f"/{rendered}"
    return rendered


def _resolve_service_base_url(config: ProbeConfig, target: ProbeTarget) -> tuple[str, bool]:
    """서비스별 forward base URL 을 결정한다."""
    env_prefix = _service_env_prefix(target.service)
    override_base_url = (
        _read_env(
            f"CODE_SERVER_FORWARD_{env_prefix}_BASE_URL",
            f"FORWARD_URL_{env_prefix}_BASE_URL",
        ).rstrip("/")
    )
    if override_base_url:
        return override_base_url, True

    if target.port in {None, ""} or not config.root_url:
        return "", False

    path = _render_forward_path(config, target)
    return f"{config.root_url}{path}", False


def _resolve_service_api_key(config: ProbeConfig, target: ProbeTarget) -> str:
    """서비스별 bearer token 을 반환한다."""
    env_prefix = _service_env_prefix(target.service)
    return (
        _read_env(
            f"CODE_SERVER_FORWARD_{env_prefix}_API_KEY",
            f"FORWARD_URL_{env_prefix}_API_KEY",
        )
        or config.default_api_key
    )


def _health_endpoint(base_url: str) -> str:
    """vLLM health endpoint URL 을 계산한다."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    return f"{normalized}/health"


def _models_endpoint(base_url: str) -> str:
    """OpenAI-compatible /v1/models endpoint 를 계산한다."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _probe_url(url: str, *, api_key: str, timeout_sec: float) -> dict[str, object]:
    """URL 에 GET 요청을 보내고 결과를 반환한다."""
    result: dict[str, object] = {
        "url": url,
        "ok": False,
        "status_code": None,
        "latency_ms": None,
        "body": None,
        "error": None,
    }
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        started = time.time()
        response = requests.get(url, headers=headers, timeout=timeout_sec)
        latency_ms = (time.time() - started) * 1000
        result["status_code"] = response.status_code
        result["latency_ms"] = round(latency_ms, 1)
        try:
            result["body"] = response.json()
        except ValueError:
            result["body"] = response.text[:200]
        result["ok"] = response.status_code < 400
    except requests.RequestException as exc:
        result["error"] = str(exc)

    return result


def _extract_model_ids(payload: object) -> list[str]:
    """OpenAI /v1/models 응답에서 model id 목록을 추출한다."""
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []

    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


def check_forward_models(
    config: ProbeConfig,
    targets: list[ProbeTarget],
) -> list[dict[str, object]]:
    """code-server forward URL 을 통해 health + /v1/models 를 확인한다."""
    results: list[dict[str, object]] = []

    for target in targets:
        base_url, override_used = _resolve_service_base_url(config, target)
        api_key = _resolve_service_api_key(config, target)
        health_url = _health_endpoint(base_url)
        models_url = _models_endpoint(base_url)
        health_probe = (
            _probe_url(health_url, api_key=api_key, timeout_sec=config.timeout_sec)
            if health_url
            else {
                "ok": False,
                "status_code": None,
                "latency_ms": None,
                "body": None,
                "error": "health URL을 계산할 수 없습니다.",
            }
        )
        models_probe = (
            _probe_url(models_url, api_key=api_key, timeout_sec=config.timeout_sec)
            if models_url
            else {
                "ok": False,
                "status_code": None,
                "latency_ms": None,
                "body": None,
                "error": "/v1/models URL을 계산할 수 없습니다.",
            }
        )
        detected_models = _extract_model_ids(models_probe.get("body"))
        expected_model = target.expected_model.strip()

        results.append(
            {
                "service": target.service,
                "display_name": target.display_name,
                "expected_model": target.expected_model,
                "port": target.port,
                "config_known": target.config_known,
                "config_enabled": target.config_enabled,
                "base_url": base_url,
                "health_url": health_url,
                "models_url": models_url,
                "override_used": override_used,
                "health_ok": health_probe.get("ok", False),
                "health_status_code": health_probe.get("status_code"),
                "health_latency_ms": health_probe.get("latency_ms"),
                "health_error": health_probe.get("error"),
                "models_ok": models_probe.get("ok", False),
                "models_status_code": models_probe.get("status_code"),
                "models_latency_ms": models_probe.get("latency_ms"),
                "models_error": models_probe.get("error"),
                "detected_models": detected_models,
                "model_match": expected_model in detected_models if expected_model and detected_models else False,
            }
        )

    return results


def print_summary(config: ProbeConfig, targets: list[ProbeTarget], results: list[dict[str, object]]) -> None:
    """점검 결과를 요약 출력한다."""
    target_service_names = ", ".join(target.service for target in targets) or "(없음)"

    print(f"\n{SEPARATOR}")
    print("  code-server Forward URL VLM 연결 점검 결과")
    print(f"{SEPARATOR}")
    print(f"  Forward Root:      {config.root_url or '(미설정)'}")
    print(f"  Path Template:     {config.path_template}")
    print(f"  점검 대상:         {target_service_names}")
    print(f"  Timeout:           {config.timeout_sec}s")

    print("\n  Resolved URLs:")
    for result in results:
        service = str(result.get("service") or "")
        base_url = str(result.get("base_url") or "(계산 실패)")
        models_url = str(result.get("models_url") or "(계산 실패)")
        print(f"  - {service}: {base_url}")
        print(f"    models -> {models_url}")

    print(
        f"\n  {'서비스':<18} {'포트':<6} {'모델':<22} {'health':<10} {'models':<10} "
        f"{'모델 일치':<10} {'응답시간':<10} {'total path':<52} {'비고'}"
    )
    print(
        f"  {'─' * 18} {'─' * 6} {'─' * 22} {'─' * 10} {'─' * 10} "
        f"{'─' * 10} {'─' * 10} {'─' * 52} {'─' * 18}"
    )

    all_ok = True
    for result in results:
        service = str(result.get("service") or "")
        port = result.get("port")
        port_str = str(port) if port not in {None, ""} else "-"
        model_name = str(result.get("expected_model") or result.get("display_name") or service)
        models_url = str(result.get("models_url") or "")

        health_status = "OK" if result.get("health_ok") else "FAIL"
        if result.get("models_ok") and (
            not str(result.get("expected_model") or "").strip() or result.get("model_match")
        ):
            models_status = "OK"
        elif result.get("models_ok"):
            models_status = "MISMATCH"
            all_ok = False
        else:
            models_status = "FAIL"
            all_ok = False

        expected_model = str(result.get("expected_model") or "").strip()
        match_str = "O" if expected_model and result.get("model_match") else "-" if not expected_model else "X"
        latency_ms = result.get("models_latency_ms")
        latency_str = f"{latency_ms}ms" if latency_ms is not None else "-"

        note = ""
        if result.get("models_error"):
            note = str(result["models_error"])[:30]
        elif not result.get("models_ok"):
            note = f"HTTP {result.get('models_status_code')}"
        elif result.get("detected_models") and not result.get("model_match"):
            note = f"got: {', '.join(result['detected_models'][:2])}"
        elif result.get("health_error"):
            note = f"health: {str(result['health_error'])[:22]}"
        elif result.get("override_used"):
            note = "service override"
        elif result.get("config_known") is False:
            note = "unknown service slug"
        elif result.get("config_enabled") is False:
            note = "config enabled=False"

        print(
            f"  {service:<18} {port_str:<6} {model_name[:22]:<22} {health_status:<10} {models_status:<10} "
            f"{match_str:<10} {latency_str:<10} {models_url[:52]:<52} {note}"
        )

    print(f"\n  configured primary VLM: {DEFAULT_PRIMARY_VLM_SERVICE} ({DEFAULT_PRIMARY_VLM_MODEL_NAME})")
    print(f"  configured OCR VLM:     {DEFAULT_OCR_VLM_SERVICE} ({DEFAULT_OCR_VLM_MODEL_NAME})")

    print(f"{SEPARATOR}")
    if all_ok:
        print("  결과: code-server forward URL 로 직접 접근 가능")
    else:
        print("  결과: 일부 서비스 direct 연결 실패 — 위 비고 확인")
    print(SEPARATOR)


def main() -> None:
    """code-server forward URL 기반 VLM 연결 점검을 수행한다."""
    config = _resolve_probe_config()
    print(f"[INFO] code-server forward URL 연결 점검 시작 (root: {config.root_url})")

    targets = _resolve_targets()
    print(f"[INFO] 대상 서비스: {', '.join(target.service for target in targets)}")

    print("\n[INFO] 각 VLM 서비스 health + /v1/models 직접 호출 중...")
    results = check_forward_models(config, targets)

    print_summary(config, targets, results)


if __name__ == "__main__":
    main()
