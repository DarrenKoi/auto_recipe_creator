"""poc.work2.login_benchmark home 테스트."""

from poc.work2.connection_check import _build_models_url
from poc.work2.login_benchmark import (
    build_model_log_name,
    build_prompt_for_service,
    resolve_benchmark_service_slugs,
)


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


def test_build_models_url_accepts_direct_v1_base_url():
    assert _build_models_url("http://common.llm.skhynix.com.com/v1") == (
        "http://common.llm.skhynix.com.com/v1/models"
    )
    assert _build_models_url("http://example.com/api/vlm_serve/ui-venus") == (
        "http://example.com/api/vlm_serve/ui-venus/v1/models"
    )
