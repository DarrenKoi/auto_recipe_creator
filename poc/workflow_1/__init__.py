"""
Workflow 1 package.

`poc/work2` 에서 분리한 workflow 관련 모듈과
독립 실행에 필요한 최소 공용 유틸리티를 함께 모아 둔 패키지.
"""

import os
import re
import time
from pathlib import Path

WORKFLOW_1_DIR = Path(__file__).resolve().parent
DEBUG_IMAGE_DIR = WORKFLOW_1_DIR / "debug_images"
LOG_DIR = WORKFLOW_1_DIR / "logs"
RECORDING_DIR = WORKFLOW_1_DIR / "recordings"

# 오피스 MES 가 align fail 시 생성하는 이미지 루트. align fail 핸들러는 여기에
# captured_img_from_rcs 를 함께 적재하고, workflow_2 는 align_fail_assets 로 읽는다.
#   align_images/<eqp_id>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr, captured_img_from_rcs}
ALIGN_IMAGES_DIR = WORKFLOW_1_DIR / "align_images"

_TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{6}_\d{6}_")


def _slugify_model_name(model_name: str) -> str:
    """모델명을 폴더명에 안전한 slug 로 변환한다."""
    safe_chars = []
    for char in model_name.strip().lower():
        if char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")

    slug = "".join(safe_chars).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default-model"


def resolve_debug_model_name(model_name: str | None = None) -> str:
    """디버그 이미지 저장에 사용할 모델명을 결정한다."""
    from poc.workflow_1.flask_vlm import DEFAULT_SCREEN_ANALYSIS_MODEL_NAME

    resolved = (
        (model_name or "").strip()
        or os.environ.get("VLM_MODEL_NAME", "").strip()
        or DEFAULT_SCREEN_ANALYSIS_MODEL_NAME
    )
    return _slugify_model_name(resolved or "default-model")


def debug_image_dir(debug_root: Path, model_name: str | None = None) -> Path:
    """모델명 기준 디버그 이미지 하위 디렉터리를 반환한다."""
    out_dir = debug_root / resolve_debug_model_name(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def debug_image_path(
    debug_root: Path,
    filename: str,
    model_name: str | None = None,
    timestamp_tag: str | None = None,
    now: float | None = None,
) -> Path:
    """모델명 하위 폴더를 포함한 디버그 이미지 파일 경로를 반환한다."""
    relative_path = Path(filename)
    resolved_name = relative_path.name
    if not _TIMESTAMP_PREFIX_PATTERN.match(resolved_name):
        if timestamp_tag:
            resolved_tag = str(timestamp_tag).strip()
        else:
            resolved_now = time.time() if now is None else now
            resolved_tag = time.strftime("%y%m%d_%H%M%S", time.localtime(resolved_now))
        resolved_name = f"{resolved_tag}_{resolved_name}"

    return (
        debug_image_dir(debug_root, model_name=model_name)
        / relative_path.with_name(resolved_name)
    )

__all__ = [
    "ALIGN_IMAGES_DIR",
    "DEBUG_IMAGE_DIR",
    "LOG_DIR",
    "RECORDING_DIR",
    "WORKFLOW_1_DIR",
    "debug_image_dir",
    "debug_image_path",
    "logger",
    "login_rcs_common",
    "login_rcs_ui_venus_mai",
    "open_rcs",
    "prompts",
    "resolve_debug_model_name",
    "util",
    "view_list_tab_rcs",
    "vlm_client",
    "workflow_select_tool",
    "workflow_config",
    "workflow_login",
    "workflow_runner",
    "workflow_types",
]
