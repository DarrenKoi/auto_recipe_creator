"""poc.work2 현재 서비스 설정/프롬프트 home 테스트."""

from poc.work2 import connection_check
from poc.work2.connection_check import _build_models_url, check_proxy_models
from poc.work2.flask_vlm import (
    KIMI_K2_5_MODEL_NAME,
    SHARED_PIPELINE_SETTINGS,
    get_service_by_slug,
    resolve_company_llm_api_key,
    resolve_service_api_key,
)
from poc.work2.prompts.prompt_login_rcs import build_login_rcs_locator_prompt
from poc.work2.util.json_utils import extract_json
from poc.work2.vlm_client import Work2VLMClient


def test_generic_locator_prompt_keeps_relative_1000_json_contract():
    system_message, user_text = build_login_rcs_locator_prompt(
        width=800,
        height=600,
        target_keys=("login_button", "cancel_button"),
    )

    assert "coord_system='relative_1000'" in system_message
    assert '"login_button": {"x": ..., "y": ...}' in user_text
    assert '"cancel_button": {"x": ..., "y": ...}' in user_text


def test_kimi_service_uses_exact_company_model_name():
    assert KIMI_K2_5_MODEL_NAME == "Kimi-K2.5"
    assert get_service_by_slug("kimi-k2.5").model_name == "Kimi-K2.5"


def test_build_models_url_accepts_direct_v1_base_url():
    assert _build_models_url("http://common.llm.skhynix.com/v1") == (
        "http://common.llm.skhynix.com/v1/models"
    )
    assert _build_models_url("http://example.com/api/vlm_serve/ui-venus") == (
        "http://example.com/api/vlm_serve/ui-venus/v1/models"
    )


def test_company_llm_api_key_uses_common_llm_env(monkeypatch):
    monkeypatch.setenv("COMMON_LLM_API_KEY", "test-company-key")

    assert "company_llm_api_key" not in SHARED_PIPELINE_SETTINGS
    assert resolve_company_llm_api_key() == "test-company-key"
    assert resolve_service_api_key("kimi-k2.5") == "test-company-key"
    assert resolve_service_api_key("qwen3-vl-30b-instruct") == "test-company-key"

    client = Work2VLMClient(service_slug="kimi-k2.5")
    assert client.api_key == "test-company-key"


def test_check_proxy_models_skips_direct_company_services(monkeypatch):
    def fail_get(*args, **kwargs):
        raise AssertionError("direct company services should not probe /models")

    monkeypatch.setattr(connection_check.requests, "get", fail_get)

    results = check_proxy_models(
        flask_base_url="http://example.com/api",
        target_services=[
            {
                "service": "kimi-k2.5",
                "expected_model": "Kimi-K2.5",
                "connection_mode": "direct",
                "api_url": "http://common.llm.skhynix.com/v1",
                "proxy_registered": False,
            }
        ],
    )

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["skipped"] is True
    assert results[0]["reason"] == "direct company API: /models probe 미사용"


def test_extract_json_repairs_common_ui_model_almost_json_shapes():
    fenced = """```json
    {
      "coord_system": "relative_1000",
      "login_button": {"x": 512, "y": 824},
    }
    ```"""
    assert extract_json(fenced)["login_button"] == {"x": 512, "y": 824}

    python_like = """Here is the result:
    {'coord_system': 'relative_1000', 'cancel_button': {'x': 700, 'y': 820}}
    """
    assert extract_json(python_like)["cancel_button"] == {"x": 700, "y": 820}
