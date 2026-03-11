"""Prompt builders for `poc.work2`."""

from .ocr_assist import build_ocr_assist_prompt
from .rcs_login import build_rcs_login_locator_prompt
from .rcs_main_tabs import build_rcs_main_tab_locator_prompt
from .screen_analysis import (
    build_general_query_prompt,
    build_measurement_judgment_prompt,
    build_state_recognition_prompt,
)

__all__ = [
    "build_ocr_assist_prompt",
    "build_rcs_login_locator_prompt",
    "build_rcs_main_tab_locator_prompt",
    "build_state_recognition_prompt",
    "build_measurement_judgment_prompt",
    "build_general_query_prompt",
]
