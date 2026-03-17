"""poc.work2.login_benchmark home 테스트."""

from poc.work2 import login_benchmark
from poc.work2.connection_check import _build_models_url
from poc.work2.flask_vlm import (
    KIMI_K2_5_MODEL_NAME,
    SHARED_PIPELINE_SETTINGS,
    get_service_by_slug,
    resolve_company_llm_api_key,
    resolve_service_api_key,
)
from poc.work2.login_benchmark import (
    build_model_log_name,
    build_prompt_for_service,
    resolve_benchmark_service_slugs,
)
from poc.work2.util.json_utils import extract_json
from poc.work2.vlm_client import Work2VLMClient


def test_resolve_benchmark_service_slugs_defaults_to_primary_gui_order():
    assert resolve_benchmark_service_slugs() == (
        "kimi-k2.5",
        "qwen3-vl-30b-instruct",
        "ui-venus",
        "ui-tars",
    )


def test_resolve_benchmark_service_slugs_parses_and_deduplicates_input():
    assert resolve_benchmark_service_slugs(" ui-venus ; kimi-k2.5, ui-venus, ui-tars ") == (
        "ui-venus",
        "kimi-k2.5",
        "ui-tars",
    )


def test_build_prompt_for_service_keeps_same_output_contract_for_ocr_and_gui():
    gui_system, gui_user = build_prompt_for_service(
        "ui-venus",
        width=800,
        height=600,
        target_keys=("login_button", "cancel_button"),
    )
    ocr_system, ocr_user = build_prompt_for_service(
        "paddleocr-vl-1.5",
        width=800,
        height=600,
        target_keys=("login_button", "cancel_button"),
    )

    assert "coord_system='relative_1000'" in gui_system
    assert "coord_system='relative_1000'" in ocr_system
    assert '"login_button": {"x": ..., "y": ...}' in gui_user
    assert '"login_button": {"x": ..., "y": ...}' in ocr_user
    assert "Keep the same JSON schema and coord_system as the GUI models." in ocr_user


def test_build_model_log_name_separates_files_by_model_name():
    assert build_model_log_name("login_rcs", "UI-Venus-1.5-8B") == "login_rcs_ui-venus-1.5-8b"


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


def test_direct_company_probe_skips_models_list_check(monkeypatch):
    def fail_get(*args, **kwargs):
        raise AssertionError("direct company services should not probe /models")

    monkeypatch.setattr(login_benchmark.requests, "get", fail_get)

    healthy, reason = login_benchmark._probe_service_health("kimi-k2.5")

    assert healthy is True
    assert reason == "direct preflight skipped: Kimi-K2.5"


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
