"""
PoC Work2 Module (Flask VLM proxy test)

기존 `poc.work` 자동화 흐름을 유지하면서 Flask server 경유 VLM 테스트를 위한
실험용 entrypoint 들을 모아둔 패키지.
"""

__all__ = [
    "flask_vlm",
    "pipeline_ocr",
    "prompts",
    "rcs_utils",
    "automate_rcs_login",
    "check_tool_screen",
    "click_rcs_view_mode",
    "vlm_screen_analysis",
]
