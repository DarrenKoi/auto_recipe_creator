"""poc.work2.connection_check 회귀 테스트."""

from poc.work2 import connection_check
from poc.work2.flask_vlm import resolve_service_proxy_url


def test_resolve_service_proxy_url_normalizes_base_url_variants():
    assert (
        resolve_service_proxy_url(
            "mai-ui",
            flask_base_url="http://example.com/api",
        )
        == "http://example.com/api/vlm_serve/mai-ui"
    )
    assert (
        resolve_service_proxy_url(
            "mai-ui",
            flask_base_url="http://example.com/api/vlm_serve",
        )
        == "http://example.com/api/vlm_serve/mai-ui"
    )
    assert (
        resolve_service_proxy_url(
            "mai-ui",
            flask_base_url="http://example.com/api/vlm_serve/mai-ui",
        )
        == "http://example.com/api/vlm_serve/mai-ui"
    )


def test_check_proxy_models_uses_normalized_proxy_route(monkeypatch):
    captured_urls: list[str] = []

    def fake_probe(url: str) -> dict:
        captured_urls.append(url)
        return {
            "url": url,
            "ok": True,
            "status_code": 200,
            "latency_ms": 1.0,
            "body": {"data": [{"id": "mai-ui-8b"}]},
            "error": None,
        }

    monkeypatch.setattr(connection_check, "_probe_url", fake_probe)

    results = connection_check.check_proxy_models(
        "http://example.com/api/vlm_serve",
        [
            {
                "service": "mai-ui",
                "expected_model": "mai-ui-8b",
                "proxy_registered": True,
                "api_url": "",
            }
        ],
    )

    assert captured_urls == ["http://example.com/api/vlm_serve/mai-ui/v1/models"]
    assert results[0]["model_match"] is True
    assert results[0]["detected_models"] == ["mai-ui-8b"]
