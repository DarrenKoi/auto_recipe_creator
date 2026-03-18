from poc.work2.login_benchmark import build_prompt_for_service
from poc.work2.prompts.prompt_login_rcs import build_login_rcs_locator_prompt
from poc.work2.prompts.prompt_login_rcs_ui_venus import build_login_rcs_ui_venus_prompt


def test_ui_venus_prompt_uses_grounding_language_without_manual_point_formula() -> None:
    system_message, user_text = build_login_rcs_ui_venus_prompt(
        800,
        600,
        ("server_input", "login_button"),
    )

    assert system_message.startswith("GROUNDING task for a desktop GUI screenshot.")
    assert "point you would actually click" in system_message
    assert "Return the point that best grounds the visible element a user would click." in user_text
    assert "20-30% of the control width" not in user_text
    assert '"server_input": {"x": ..., "y": ...}' in user_text
    assert '"login_button": {"x": ..., "y": ...}' in user_text


def test_build_prompt_for_service_routes_ui_venus_to_dedicated_prompt() -> None:
    system_message, user_text = build_prompt_for_service(
        "ui-venus",
        width=800,
        height=600,
        target_keys=("login_button",),
    )
    expected_system, expected_user = build_login_rcs_ui_venus_prompt(
        800,
        600,
        ("login_button",),
    )

    assert system_message == expected_system
    assert user_text == expected_user


def test_non_ui_venus_gui_services_keep_generic_locator_prompt() -> None:
    system_message, user_text = build_prompt_for_service(
        "mai-ui",
        width=800,
        height=600,
        target_keys=("login_button",),
    )
    expected_system, expected_user = build_login_rcs_locator_prompt(
        width=800,
        height=600,
        target_keys=("login_button",),
    )

    assert system_message == expected_system
    assert user_text == expected_user
