"""workflow_3 가 /api/vlm_serve 프록시에 서버와 같은 VLLM_API_KEY 를 싣는지 확인한다.

GPU 서버 site.env 에 키를 채우는 순간 프록시가 키를 요구한다. 이 키를 안 실으면
mai-ui / paddleocr / qwen 호출이 전부 401 로 멈춘다.

    pytest poc/workflow_3/vlm/test_flask_vlm_key.py
"""

from poc.workflow_3.vlm.flask_vlm import resolve_service_api_key


def test_proxy_services_send_vllm_api_key(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "team-key")
    monkeypatch.setenv("COMMON_LLM_API_KEY", "company-key")

    assert resolve_service_api_key("mai-ui") == "team-key"
    assert resolve_service_api_key("paddleocr-vl-1.5") == "team-key"
    assert resolve_service_api_key("qwen3.8-27b") == "team-key"
    # direct 회사 LLM 은 그대로 회사 키를 쓴다.
    assert resolve_service_api_key("kimi-k2.6") == "company-key"


def test_proxy_key_is_empty_when_unset(monkeypatch):
    """서버가 키를 안 쓰면 빈 값이 나가야 한다 - vlm_client 는 빈 키면 헤더를 만들지 않는다."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    assert resolve_service_api_key("mai-ui") == ""
