"""Prompt builders for `poc.work2`."""

from .prompt_ocr_assist import build_ocr_assist_prompt
from .prompt_login_rcs import build_login_rcs_locator_prompt
from .prompt_login_rcs_ui_tars import (
    build_login_rcs_ui_tars_prompt,
    build_single_element_prompt as build_login_rcs_ui_tars_single_prompt,
)
from .prompt_rcs_main_tabs import build_rcs_main_tab_locator_prompt
from .prompt_screen_analysis import (
    build_general_query_prompt,
    build_measurement_judgment_prompt,
    build_state_recognition_prompt,
)

__all__ = [
    "build_ocr_assist_prompt",
    "build_login_rcs_locator_prompt",
    "build_login_rcs_ui_tars_prompt",
    "build_login_rcs_ui_tars_single_prompt",
    "build_rcs_main_tab_locator_prompt",
    "build_state_recognition_prompt",
    "build_measurement_judgment_prompt",
    "build_general_query_prompt",
]
