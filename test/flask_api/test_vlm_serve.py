"""flask_api.vlm_serve proxy tests."""

from __future__ import annotations

import json
import logging

from flask import Flask
from requests import RequestException

from flask_api import register_flask_api


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


def _create_test_app() -> Flask:
    app = Flask(__name__)
    register_flask_api(app)
    return app


def test_vlm_serve_root_lists_registered_services():
    app = _create_test_app()
    client = app.test_client()

    response = client.get("/api/vlm_serve/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["base_path"] == "/api/vlm_serve"
    assert {item["service"] for item in payload["vlm_services"]} == {
        "ui-venus",
        "mai-ui",
        "ui-tars",
    }


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
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    response = client.post(
        "/api/vlm_serve/mai-ui/v1/chat/completions",
        json={
            "model": "mai-ui-8b",
            "messages": [{"role": "user", "content": "ping"}],
        },
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
    monkeypatch.setenv("VLM_SERVE_UPSTREAM_API_KEY", "internal-key")

    app = _create_test_app()
    client = app.test_client()

    with caplog.at_level(logging.INFO, logger="flask_api.vlm_serve"):
        response = client.post(
            "/api/vlm_serve/mai-ui/v1/chat/completions",
            json={
                "model": "mai-ui-8b",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    log_text = caplog.text
    assert "Proxy request started service=mai-ui" in log_text
    assert "Upstream response completed service=mai-ui" in log_text
    assert '"content": "analysis complete"' in log_text
    assert "Bearer internal-key" not in log_text
    assert "<redacted>" in log_text


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
    assert "VLM upstream request failed service=ui-venus" in caplog.text
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
    assert "Upstream streaming response started service=ui-tars" in caplog.text
    assert "Upstream streaming response completed service=ui-tars" in caplog.text
    assert "chunks=2" in caplog.text
