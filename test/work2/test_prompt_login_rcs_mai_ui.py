from poc.work2.prompts import build_login_rcs_mai_ui_zoom_prompt


def test_mai_ui_zoom_prompt_is_crop_relative_and_json_only() -> None:
    system_message, user_text = build_login_rcs_mai_ui_zoom_prompt()

    assert "zoomed-in crop" in system_message
    assert "Return ONLY valid JSON." in system_message
    assert "relative to THIS cropped image only" in user_text
    assert '"coord_system": "relative_1000"' in user_text
    assert '"userid_input": {' in user_text
    assert '"userid_input": null' in user_text


def test_mai_ui_zoom_prompt_guides_left_inner_click_area() -> None:
    _, user_text = build_login_rcs_mai_ui_zoom_prompt()

    assert "left-inner text entry area" in user_text
    assert "Do not click on label text, borders, highlights" in user_text
