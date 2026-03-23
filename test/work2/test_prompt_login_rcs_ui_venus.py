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


def test_ui_venus_prompt_and_generic_locator_share_same_json_schema() -> None:
    _, ui_venus_user_text = build_login_rcs_ui_venus_prompt(
        width=800,
        height=600,
        target_keys=("login_button",),
    )
    _, generic_user_text = build_login_rcs_locator_prompt(
        width=800,
        height=600,
        target_keys=("login_button",),
    )

    assert '"coord_system": "relative_1000"' in ui_venus_user_text
    assert '"coord_system": "relative_1000"' in generic_user_text
    assert '"login_button": {"x": ..., "y": ...}' in ui_venus_user_text
    assert '"login_button": {"x": ..., "y": ...}' in generic_user_text


def test_generic_locator_prompt_keeps_manual_left_inner_guidance() -> None:
    _, user_text = build_login_rcs_locator_prompt(
        width=800,
        height=600,
        target_keys=("server_input",),
    )

    assert "20-30% of the control width" in user_text
    assert "Return the point that best grounds the visible element a user would click." not in user_text
