from poc.work2.login_rcs_ui_tars import convert_ui_tars_coords, parse_ui_tars_response
from poc.work2.prompts.login_rcs_ui_tars import (
    build_login_rcs_ui_tars_prompt,
    build_single_element_prompt,
)


def test_parse_single_action_point_format() -> None:
    width = 1280
    height = 720
    response = "Action: click(point='420 308')"

    parsed = parse_ui_tars_response(response, width, height, ("login_button",))

    assert parsed == {
        "login_button": {
            "x": convert_ui_tars_coords(420, 308, width, height)[0],
            "y": convert_ui_tars_coords(420, 308, width, height)[1],
        }
    }


def test_parse_labeled_batch_actions_with_mixed_formats() -> None:
    width = 1280
    height = 720
    response = "\n".join(
        [
            "server_input: Action: click(point='344 182')",
            "login_button: click(start_box='(412, 366)')",
        ]
    )

    parsed = parse_ui_tars_response(
        response,
        width,
        height,
        ("server_input", "login_button"),
    )

    expected_server = convert_ui_tars_coords(344, 182, width, height)
    expected_login = convert_ui_tars_coords(412, 366, width, height)

    assert parsed == {
        "server_input": {"x": expected_server[0], "y": expected_server[1]},
        "login_button": {"x": expected_login[0], "y": expected_login[1]},
    }


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
