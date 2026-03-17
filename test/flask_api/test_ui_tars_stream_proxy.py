"""ui-tars proxy stream 보존 테스트."""

import json

from flask import Flask

from flask_api import register_flask_api


class DummyResponse:
    """requests.Response 대체용 최소 객체."""

    def __init__(self, *, status_code: int, body: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}

    @property
    def content(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


def _create_test_app() -> Flask:
    app = Flask(__name__)
    register_flask_api(app)
    return app


def test_ui_tars_proxy_preserves_stream_flag(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=(
                b'data: {"choices":[{"delta":{"content":"{\\"coord_system\\": \\"relative_1000\\"}"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

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
    payload = json.loads(captured["data"].decode("utf-8"))
    assert payload["stream"] is True


def test_ui_tars_proxy_forces_stream_when_client_sent_false(monkeypatch):
    captured: dict[str, object] = {}

    def fake_request(**kwargs):
        captured.update(kwargs)
        return DummyResponse(
            status_code=200,
            body=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr("flask_api.vlm_serve.service_template.requests.request", fake_request)

    app = _create_test_app()
    client = app.test_client()

    response = client.post(
        "/api/vlm_serve/ui-tars/v1/chat/completions",
        json={
            "model": "ui-tars-1.5-7b",
            "stream": False,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    assert response.status_code == 200
    payload = json.loads(captured["data"].decode("utf-8"))
    assert payload["stream"] is True
