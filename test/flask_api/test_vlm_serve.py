"""flask_api.vlm_serve proxy tests."""

from __future__ import annotations

import json

from flask import Flask

from flask_api import register_flask_api


class DummyResponse:
    """requests.Response 대체용 더미."""

    def __init__(self, status_code: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}

    @property
    def content(self) -> bytes:
        return self._body

    def iter_content(self, chunk_size: int = 8192):
        yield self._body

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
