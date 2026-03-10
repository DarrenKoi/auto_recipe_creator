"""flask_api VLM route home 테스트."""

import pytest
from flask import Flask

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


def test_gpu_dashboard_health_route(client):
    """GPU dashboard health route 가 함께 살아 있어야 한다."""
    response = client.get("/gpu-dashboard/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "service": "gpu_dashboard",
        "status": "ok",
    }


@pytest.mark.parametrize(
    "expected_slug",
    [
        "ui-venus",
        "mai-ui",
        "ui-tars",
    ],
)
def test_vlm_serve_root_lists_registered_services(client, expected_slug):
    """VLM root 엔드포인트에 등록된 서비스 목록이 보여야 한다."""
    response = client.get("/api/vlm_serve/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "vlm_serve"
    assert payload["status"] == "ok"
    assert payload["base_path"] == "/api/vlm_serve"
    assert expected_slug in {
        item["service"]
        for item in payload["vlm_services"]
    }


def test_vlm_serve_health_lists_registered_slugs(client):
    """VLM health 엔드포인트에 등록 slug 목록이 보여야 한다."""
    response = client.get("/api/vlm_serve/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "service": "vlm_serve",
        "status": "ok",
        "mode": "proxy",
        "base_path": "/api/vlm_serve",
        "registered_vlms": ["ui-venus", "mai-ui", "ui-tars"],
    }
