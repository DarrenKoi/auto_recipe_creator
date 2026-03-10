"""VLM 서비스 blueprint template."""

import os
from dataclasses import dataclass
from typing import Iterator

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context


@dataclass(frozen=True)
class VLMServiceConfig:
    """VLM 서비스 route 설정."""

    blueprint_name: str
    route_slug: str
    display_name: str
    upstream_port: int

    @property
    def env_prefix(self) -> str:
        """환경변수 prefix 를 반환한다."""
        return self.route_slug.replace("-", "_").upper()

    @property
    def api_base_path(self) -> str:
        """API base path 를 반환한다."""
        return f"/api/vlm_serve/{self.route_slug}"

    @property
    def health_path(self) -> str:
        """Health path 를 반환한다."""
        return f"{self.api_base_path}/health"

    @property
    def upstream_base_url(self) -> str:
        """예상 upstream base URL 을 반환한다."""
        service_key = f"VLM_SERVE_{self.env_prefix}_BASE_URL"
        service_url = os.environ.get(service_key, "").strip().rstrip("/")
        if service_url:
            return service_url

        upstream_host = os.environ.get("VLM_SERVE_UPSTREAM_HOST", "127.0.0.1").strip()
        if not upstream_host:
            upstream_host = "127.0.0.1"
        return f"http://{upstream_host}:{self.upstream_port}"

    def to_dict(self) -> dict[str, object]:
        """직렬화용 dict 를 반환한다."""
        return {
            "service": self.route_slug,
            "model_name": self.display_name,
            "mode": "proxy",
            "upstream_port": self.upstream_port,
            "upstream_base_url": self.upstream_base_url,
            "api_base_path": self.api_base_path,
            "health_url": self.health_path,
        }


def _upstream_timeout() -> tuple[float, float]:
    """Upstream 요청 timeout 을 반환한다."""
    connect_timeout = float(os.environ.get("VLM_SERVE_CONNECT_TIMEOUT_SEC", "5.0"))
    read_timeout = float(os.environ.get("VLM_SERVE_READ_TIMEOUT_SEC", "300.0"))
    return connect_timeout, read_timeout


def _should_stream_response() -> bool:
    """스트리밍 응답 여부를 추정한다."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and payload.get("stream") is True:
        return True
    return False


def _build_upstream_headers() -> dict[str, str]:
    """Hop-by-hop 헤더를 제외한 upstream 요청 헤더를 구성한다."""
    blocked_headers = {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
    }
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked_headers
    }

    default_api_key = os.environ.get("VLM_SERVE_UPSTREAM_API_KEY", "").strip()
    if default_api_key and "authorization" not in {key.lower() for key in headers}:
        headers["Authorization"] = f"Bearer {default_api_key}"

    return headers


def _build_response_headers(upstream_headers: requests.structures.CaseInsensitiveDict) -> list[tuple[str, str]]:
    """Flask 응답으로 넘길 헤더를 정리한다."""
    blocked_headers = {
        "content-length",
        "connection",
        "transfer-encoding",
        "content-encoding",
    }
    return [
        (key, value)
        for key, value in upstream_headers.items()
        if key.lower() not in blocked_headers
    ]


def _stream_body(upstream_response: requests.Response) -> Iterator[bytes]:
    """Streaming body 를 그대로 전달한다."""
    try:
        for chunk in upstream_response.iter_content(chunk_size=8192):
            if chunk:
                yield chunk
    finally:
        upstream_response.close()


def _proxy_request(config: VLMServiceConfig, upstream_path: str):
    """현재 요청을 upstream vLLM 으로 프록시한다."""
    upstream_url = f"{config.upstream_base_url.rstrip('/')}/{upstream_path.lstrip('/')}"
    try:
        upstream_response = requests.request(
            method=request.method,
            url=upstream_url,
            params=request.args,
            headers=_build_upstream_headers(),
            data=request.get_data(),
            cookies=request.cookies,
            timeout=_upstream_timeout(),
            stream=True,
        )
    except requests.RequestException as exc:
        print(f"[ERROR] VLM upstream 요청 실패 ({config.route_slug}): {exc}")
        return jsonify(
            {
                "service": config.route_slug,
                "status": "error",
                "message": str(exc),
                "upstream_url": upstream_url,
            }
        ), 502

    response_headers = _build_response_headers(upstream_response.headers)
    if _should_stream_response() or upstream_response.headers.get("content-type", "").startswith("text/event-stream"):
        return Response(
            stream_with_context(_stream_body(upstream_response)),
            status=upstream_response.status_code,
            headers=response_headers,
        )

    body = upstream_response.content
    upstream_response.close()
    return Response(
        body,
        status=upstream_response.status_code,
        headers=response_headers,
    )


def create_vlm_service_blueprint(config: VLMServiceConfig) -> Blueprint:
    """VLM 서비스 proxy blueprint 를 생성한다."""
    service_blueprint = Blueprint(config.blueprint_name, __name__)

    @service_blueprint.route("/", methods=["GET"])
    def home():
        """서비스 안내 엔드포인트."""
        payload = config.to_dict()
        payload.update(
            {
                "status": "ok",
                "message": "VLM service proxy route is ready.",
            }
        )
        return jsonify(payload)

    @service_blueprint.route("/health", methods=["GET"])
    def health():
        """서비스 헬스 체크 엔드포인트."""
        return _proxy_request(config, "/v1/models")

    @service_blueprint.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    def proxy_v1(subpath: str):
        """OpenAI-compatible `/v1/*` 요청을 upstream 으로 프록시한다."""
        return _proxy_request(config, f"/v1/{subpath}")

    return service_blueprint


__all__ = ["VLMServiceConfig", "create_vlm_service_blueprint"]
