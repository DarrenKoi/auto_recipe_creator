"""flask_api VLM route home 테스트."""

import sys
from pathlib import Path

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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


def test_api_health_lists_registered_vlms(client):
    """루트 health 에 등록된 VLM 목록이 보여야 한다."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "service": "api",
        "status": "ok",
        "mode": "template",
        "registered_vlms": ["ui-venus", "mai-ui", "ui-tars"],
    }


@pytest.mark.parametrize(
    ("route_slug", "display_name", "upstream_port"),
    [
        ("ui-venus", "UI-Venus-1.5-8B", 8001),
        ("mai-ui", "MAI-UI-8B", 8002),
        ("ui-tars", "UI-TARS-1.5-7B", 8003),
    ],
)
def test_each_vlm_template_health_route(client, route_slug, display_name, upstream_port):
    """각 VLM template health route 가 포트 정보를 반환해야 한다."""
    response = client.get(f"/api/{route_slug}/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "service": route_slug,
        "model_name": display_name,
        "mode": "template",
        "upstream_port": upstream_port,
        "upstream_base_url": f"http://127.0.0.1:{upstream_port}",
        "api_base_path": f"/api/{route_slug}",
        "health_url": f"/api/{route_slug}/health",
        "status": "ok",
        "message": "Template health route is reachable.",
    }
