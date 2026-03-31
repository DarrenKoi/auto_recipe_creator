"""Phase-1 prompt builders.

이 패키지는 workflow_1 이 shared `poc.work2.prompts` 와 독립적으로
진화할 수 있도록 로그인 phase 에 필요한 프롬프트만 별도로 보관한다.
"""

from .prompt_login_rcs_mai_ui import build_mai_ui_zoom_prompt
from .prompt_login_rcs_ui_venus import (
    build_ui_venus_single_element_bbox_prompt,
    build_ui_venus_single_element_bbox_prompt_by_key,
    build_ui_venus_single_element_prompt,
    build_ui_venus_single_element_prompt_by_key,
)

__all__ = [
    "build_mai_ui_zoom_prompt",
    "build_ui_venus_single_element_bbox_prompt",
    "build_ui_venus_single_element_bbox_prompt_by_key",
    "build_ui_venus_single_element_prompt",
    "build_ui_venus_single_element_prompt_by_key",
]

