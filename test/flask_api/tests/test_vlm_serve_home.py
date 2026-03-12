"""flask_api VLM route home 테스트."""

import pytest
from flask import Flask
from requests import RequestException

from flask_api import register_flask_api
from gpu_dashboard import gpu_dashboard_dp


@pytest.fixture
def client():
    """Flask test client 를 생성한다."""
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(gpu_dashboard_dp, url_prefix="/gpu-dashboard")
    register_flask_api(app)

    with app.test_client() as test_client:
        yield test_client


class DummyHealthResponse:
    """health probe 용 requests.Response 대체 객체."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fake_vlm_health_get(url: str, timeout: float):
    # 현재 응답하는 포트만 정상 응답: 8001 (ui-venus), 8004 (paddleocr-vl-1.5)
    if url == "http://127.0.0.1:8001/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "ui-venus-1.5-8b"}]})
    if url == "http://127.0.0.1:8004/v1/models":
        return DummyHealthResponse(200, {"data": [{"id": "paddleocr-vl-1.5"}]})
    raise RequestException(f"connection refused: {url}")


def test_gpu_dashboard_health_route(client):
    """GPU dashboard health route 가 함께 살아 있어야 한다."""
    response = client.get("/gpu-dashboard/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "service": "gpu_dashboard",
        "status": "ok",
    }


def test_vlm_serve_root_returns_live_health_payload(client, monkeypatch):
    """VLM root 엔드포인트가 live health payload 를 그대로 반환해야 한다."""
    monkeypatch.setattr("flask_api.vlm_serve.requests.get", _fake_vlm_health_get)

    response = client.get("/api/vlm_serve")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["status"] == "ok"
    assert payload["mode"] == "proxy"
    assert payload["base_path"] == "/api/vlm_serve"
    assert set(payload["registered_vlms"]) == {
        "ui-venus",
        "mai-ui",
        "paddleocr-vl-1.5",
        "got-ocr",
    }
    assert set(payload["serving_now"]) == {
        "UI-Venus-1.5-8B",
        "PaddleOCR-VL-1.5",
    }
    assert {
        item["service"]
        for item in payload["vlm_statuses"]
    } >= {
        "ui-venus",
        "paddleocr-vl-1.5",
        "got-ocr",
    }


def test_vlm_serve_health_lists_serving_models(client, monkeypatch):
    """VLM health 엔드포인트에 live serving 상태가 보여야 한다."""
    monkeypatch.setattr("flask_api.vlm_serve.requests.get", _fake_vlm_health_get)

    response = client.get("/api/vlm_serve/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["status"] == "ok"
    assert payload["mode"] == "proxy"
    assert payload["base_path"] == "/api/vlm_serve"
    # config.py 에서 enabled=True 인 서비스만 등록됨
    assert set(payload["registered_vlms"]) == {
        "ui-venus",
        "mai-ui",
        "paddleocr-vl-1.5",
        "got-ocr",
    }
    assert set(payload["serving_now"]) == {
        "UI-Venus-1.5-8B",
        "PaddleOCR-VL-1.5",
    }

    status_map = {
        item["service"]: item
        for item in payload["vlm_statuses"]
    }
    assert status_map["ui-venus"]["health_status"] == "serving"
    assert status_map["mai-ui"]["health_status"] == "unreachable"
    assert status_map["paddleocr-vl-1.5"]["health_status"] == "serving"
    assert status_map["got-ocr"]["health_status"] == "unreachable"


def test_api_health_wraps_vlm_health_payload(client, monkeypatch):
    """`/api/health`가 상위 health entrypoint 로 동작해야 한다."""
    monkeypatch.setattr("flask_api.vlm_serve.requests.get", _fake_vlm_health_get)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "api"
    assert payload["status"] == "ok"
    assert payload["base_path"] == "/api"
    assert payload["vlm_serve"]["service"] == "vlm_serve"
    assert "UI-Venus-1.5-8B" in payload["vlm_serve"]["serving_now"]
