"""workflow_3 VLM 연결 확인 스크립트 테스트."""

import requests

from poc.workflow_3.check_vlm_connection import check_service, models_url
from poc.workflow_3.vlm.flask_vlm import VLMServiceEntry


SERVICE = VLMServiceEntry(
    route_slug="mai-ui",
    display_name="MAI-UI-8B",
    model_name="mai-ui-8b",
    api_url="http://example.test/api/vlm_serve/mai-ui",
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_models_url_handles_direct_and_proxy_base_urls():
    assert models_url("http://common.llm.test/v1") == "http://common.llm.test/v1/models"
    assert models_url(SERVICE.api_url) == f"{SERVICE.api_url}/v1/models"


def test_check_service_uses_resolved_url_key_and_expected_model(monkeypatch):
    seen = {}

    def fake_get(url, *, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        return _Response({"data": [{"id": "mai-ui-8b"}]})

    monkeypatch.setattr(
        "poc.workflow_3.check_vlm_connection.resolve_service_proxy_url",
        lambda slug: "http://proxy.test/mai-ui",
    )
    monkeypatch.setattr(
        "poc.workflow_3.check_vlm_connection.resolve_service_api_key",
        lambda slug: "team-key",
    )
    monkeypatch.setattr("poc.workflow_3.check_vlm_connection.requests.get", fake_get)

    ok, reason = check_service(SERVICE, timeout_sec=3.0)

    assert ok, reason
    assert seen == {
        "url": "http://proxy.test/mai-ui/v1/models",
        "headers": {"Authorization": "Bearer team-key"},
        "timeout": 3.0,
    }


def test_check_service_reports_connection_failure(monkeypatch):
    def fail_get(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("poc.workflow_3.check_vlm_connection.requests.get", fail_get)

    ok, reason = check_service(SERVICE)

    assert not ok
    assert "connection refused" in reason


def test_check_service_reports_model_mismatch(monkeypatch):
    monkeypatch.setattr(
        "poc.workflow_3.check_vlm_connection.requests.get",
        lambda *args, **kwargs: _Response({"data": [{"id": "another-model"}]}),
    )

    ok, reason = check_service(SERVICE)

    assert not ok
    assert "mai-ui-8b" in reason
    assert "another-model" in reason
