"""flask_api.vlm_serve proxy tests."""

import json
import logging
from pathlib import Path

import pytest
from flask import Flask
from requests import RequestException

from flask_api import register_flask_api
import flask_api.vlm_serve as vlm_serve
from flask_api.vlm_serve import logger as vlm_logger_module


class DummyResponse:
    """requests.Response 대체용 더미."""

    def __init__(
        self,
        status_code: int,
        body: bytes,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}
        self._chunks = chunks or [body]

    @property
    def content(self) -> bytes:
        return self._body

    def iter_content(self, chunk_size: int = 8192):
        yield from self._chunks

    def close(self):
        return None


class DummyHealthResponse:
    """health probe 용 requests.Response 대체 객체."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _create_test_app() -> Flask:
    app = Flask(__name__)
    register_flask_api(app)
    return app


def _clear_vlm_file_handlers() -> None:
    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    for handler in list(root_logger.handlers):
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()


def _fake_vlm_health_get(url: str, timeout: float):
    del timeout
    if url == "http://127.0.0.1:8001/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "ui-venus-1.5-8b"}]})
    if url == "http://127.0.0.1:8004/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "paddleocr-vl-1.5"}]})
    raise RequestException(f"connection refused: {url}")


@pytest.fixture(autouse=True)
def cleanup_vlm_file_handlers():
    _clear_vlm_file_handlers()
    yield
    _clear_vlm_file_handlers()


def test_vlm_serve_root_returns_live_health_payload(monkeypatch):
    monkeypatch.setattr("flask_api.vlm_serve.requests.get", _fake_vlm_health_get)

    app = _create_test_app()
    client = app.test_client()

    response = client.get("/api/vlm_serve/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["status"] == "ok"
    assert payload["mode"] == "proxy"
    assert payload["base_path"] == "/api/vlm_serve"
    assert set(payload["registered_vlms"]) == {
        "ui-venus",
        "mai-ui",
        "ui-tars",
        "paddleocr-vl-1.5",
        "got-ocr",
    }
    assert set(payload["serving_routes"]) == {
        "ui-venus",
        "paddleocr-vl-1.5",
    }


def test_deploy_model_env_root_defaults_to_repo_deploy_vlms(monkeypatch):
    monkeypatch.delenv("CONFIG_ROOT", raising=False)
    monkeypatch.delenv("DEPLOY_VLMS_ROOT", raising=False)

    expected = (
        Path(__file__).resolve().parents[2]
        / "deploy_vlms"
        / "config"
        / "models"
    )

    assert vlm_serve._deploy_model_env_root() == expected


def test_models_proxy_uses_expected_upstream(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=json.dumps({"data": [{"id": "ui-venus-1.5-8b"}]}).encode("utf-8"),
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    response = client.get("/api/vlm_serve/ui-venus/v1/models")

    assert response.status_code == 200
    assert response.get_json()["data"][0]["id"] == "ui-venus-1.5-8b"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:8001/v1/models"


def test_chat_proxy_injects_upstream_api_key(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                            }
                        }
                    ]
                }
            ).encode("utf-8"),
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)
    monkeypatch.setenv("VLLM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    response = client.post(
        "/api/vlm_serve/mai-ui/v1/chat/completions",
        json={
            "model": "mai-ui-8b",
            "messages": [{"role": "user", "content": "ping"}],
        },
        # Authorization 없이 X-VLM-Token 만 보낸다 - 업스트림 헤더는 프록시가 만들어야 한다.
        headers={"X-VLM-Token": "internal-key"},
    )

    assert response.status_code == 200
    assert response.get_json()["choices"][0]["message"]["content"] == "ok"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8002/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer internal-key"


def test_chat_proxy_logs_request_and_response_details(monkeypatch, caplog):
    def fake_request(**kwargs):
        return DummyResponse(
            status_code=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "analysis complete",
                            }
                        }
                    ]
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)
    monkeypatch.setenv("VLLM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/mai-ui/v1/chat/completions",
            json={
                "model": "mai-ui-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers={"X-VLM-Token": "internal-key"},
        )

    assert response.status_code == 200
    log_text = caplog.text
    assert "request service=mai-ui method=POST" in log_text
    assert "response service=mai-ui" in log_text
    assert "Bearer internal-key" not in log_text


def test_chat_proxy_logs_upstream_request_exception(monkeypatch, caplog):
    def fake_request(**kwargs):
        raise RequestException("connection refused")

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/ui-venus/v1/chat/completions",
            json={
                "model": "ui-venus-1.5-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 502
    assert response.get_json()["message"] == "connection refused"
    assert "upstream failed service=ui-venus" in caplog.text
    assert "connection refused" in caplog.text


def test_streaming_chat_proxy_logs_stream_summary(monkeypatch, caplog):
    def fake_request(**kwargs):
        chunks = [
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        ]
        return DummyResponse(
            status_code=200,
            body=b"".join(chunks),
            headers={"Content-Type": "text/event-stream"},
            chunks=chunks,
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/ui-tars/v1/chat/completions",
            json={
                "model": "ui-tars-1.5-7b",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
    )

    assert response.status_code == 200
    assert b"delta" in response.data
    assert "request service=ui-tars method=POST" in caplog.text
    assert "response service=ui-tars" in caplog.text


def test_get_vlm_logger_creates_cloud_repo_log_dir(monkeypatch, tmp_path):
    cloud_repo_root = tmp_path / "cloud_repo"
    expected_log_path = cloud_repo_root / "logs" / "vlm_service" / "vlm_serve.log"

    monkeypatch.setenv("VLM_SERVE_REPO_ROOT", str(cloud_repo_root))
    monkeypatch.delenv("VLM_SERVE_LOG_DIR", raising=False)

    logger = vlm_logger_module.get_vlm_logger("proxy")
    logger.info("cloud repo log smoke test")

    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False)
    ]

    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == expected_log_path
    assert expected_log_path.parent.is_dir()

    file_handlers[0].flush()
    assert "cloud repo log smoke test" in expected_log_path.read_text(encoding="utf-8")


def test_get_vlm_logger_reuses_existing_file_handler(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_SERVE_LOG_DIR", str(tmp_path / "logs" / "vlm_service"))

    vlm_logger_module.get_vlm_logger("proxy")
    vlm_logger_module.get_vlm_logger("proxy")

    root_logger = logging.getLogger(vlm_logger_module.LOGGER_NAME)
    file_handlers = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, vlm_logger_module.FILE_HANDLER_MARKER, False)
    ]

    assert len(file_handlers) == 1


# ── 팀 공용 키 인증 (VLLM_API_KEY) ──────────────────────────────────


_CAPTURED: dict[str, object] = {}


def _ok_response(*_args, **kwargs):
    """프록시가 upstream 까지 갔는지 보기 위한 더미."""
    _CAPTURED.update(kwargs)
    return DummyResponse(
        status_code=200,
        body=json.dumps({"data": [{"id": "mai-ui-8b"}]}).encode("utf-8"),
    )


@pytest.fixture
def proxy_client(monkeypatch):
    _CAPTURED.clear()
    monkeypatch.setattr(
        "flask_api.vlm_serve.service_template.requests.request", _ok_response
    )
    return _create_test_app().test_client()


def test_proxy_rejects_call_without_key_when_configured(proxy_client, monkeypatch):
    """키 없는 호출은 upstream 까지 가지 않는다 - 서버에 키를 채우기 전에 workflow_3 부터 옮길 것."""
    monkeypatch.setenv("VLLM_API_KEY", "team-secret")

    response = proxy_client.get("/api/vlm_serve/mai-ui/v1/models")

    assert response.status_code == 401
    assert _CAPTURED == {}


def test_proxy_rejects_wrong_key(proxy_client, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models", headers={"X-VLM-Token": "nope"}
    )

    assert response.status_code == 401


def test_proxy_accepts_openai_style_bearer_key(proxy_client, monkeypatch):
    """OpenAI 클라이언트(workflow_3 의 vlm_client 포함)는 api_key 를 Authorization 으로 보낸다."""
    monkeypatch.setenv("VLLM_API_KEY", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models",
        headers={"Authorization": "Bearer team-secret"},
    )

    assert response.status_code == 200
    assert _CAPTURED["headers"]["Authorization"] == "Bearer team-secret"


def test_caller_authorization_is_replaced_by_the_key(proxy_client, monkeypatch):
    """X-VLM-Token 으로 인증한 호출자가 딴 Authorization 을 들고 와도 업스트림엔 키가 간다."""
    monkeypatch.setenv("VLLM_API_KEY", "team-secret")

    response = proxy_client.get(
        "/api/vlm_serve/mai-ui/v1/models",
        headers={"X-VLM-Token": "team-secret", "Authorization": "Bearer placeholder"},
    )

    assert response.status_code == 200
    assert _CAPTURED["headers"]["Authorization"] == "Bearer team-secret"


def test_health_and_home_stay_open_when_key_is_set(proxy_client, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "team-secret")

    assert proxy_client.get("/api/vlm_serve/mai-ui/").status_code == 200
    assert proxy_client.get("/api/vlm_serve/mai-ui/health").status_code == 200


def test_proxy_stays_open_when_no_key_configured(proxy_client, monkeypatch):
    """키를 안 채운 배포(지금 서버)는 종전처럼 열려 있어야 한다."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    response = proxy_client.get("/api/vlm_serve/mai-ui/v1/models")

    assert response.status_code == 200
