"""
Workflow 1 package.

`poc/work2` 에서 분리한 workflow 관련 모듈을 모아 둔 패키지.
공용 유틸리티와 VLM 인프라는 계속 `poc.work2` 를 재사용한다.
"""

from pathlib import Path

WORKFLOW_1_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW_1_DIR / "debug_images"
LOG_DIR = WORKFLOW_1_DIR / "logs"

__all__ = [
    "DEBUG_IMAGE_DIR",
    "LOG_DIR",
    "WORKFLOW_1_DIR",
    "logger",
    "login_rcs_common",
    "login_rcs_ui_venus_mai",
    "open_rcs",
    "workflow_config",
    "workflow_login",
    "workflow_runner",
    "workflow_types",
]
