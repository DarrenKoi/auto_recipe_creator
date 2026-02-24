"""VLM prompt builders for RCS GUI automation."""

from .rcs_login import build_rcs_login_locator_prompt
from .rcs_main_tabs import build_rcs_main_tab_locator_prompt
from .rcs_select_tool import build_rcs_select_tool_prompt
from .rcs_tool_list import build_rcs_tool_list_reader_prompt

__all__ = [
    "build_rcs_login_locator_prompt",
    "build_rcs_main_tab_locator_prompt",
    "build_rcs_select_tool_prompt",
    "build_rcs_tool_list_reader_prompt",
]
