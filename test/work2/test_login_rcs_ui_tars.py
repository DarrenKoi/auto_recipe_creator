from poc.work2.prompts.prompt_login_rcs_ui_tars import (
    build_login_rcs_ui_tars_prompt,
    build_single_element_prompt,
)

def test_batch_prompt_does_not_use_element_name_placeholder_format() -> None:
    system_message, user_text = build_login_rcs_ui_tars_prompt(("server_input", "login_button"))

    assert "element_name:" not in user_text
    assert "<target_key>: Action: click(point='x y')" in system_message
    assert "server_input" in user_text
    assert "login_button" in user_text


def test_single_element_prompt_uses_official_action_style() -> None:
    system_message, user_text = build_single_element_prompt("server_input")

    assert "## Output Format" in system_message
    assert "Action: ..." in system_message
    assert "click(point='x y')" in system_message
    assert "Target key: server_input" in user_text
    assert "Action: click(point='x y')" in user_text


def test_single_element_prompt_rejects_unknown_key() -> None:
    try:
        build_single_element_prompt("unknown_key")
    except ValueError as exc:
        assert "Unknown element key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown UI-TARS key")
