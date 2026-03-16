"""
PoC Work2 Module (Flask VLM proxy test)

기존 `poc.work` 자동화 흐름을 유지하면서 Flask server 경유 VLM 테스트를 위한
실험용 entrypoint 들을 모아둔 패키지.
"""

import os
from pathlib import Path


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
    from poc.work2.flask_vlm import DEFAULT_SCREEN_ANALYSIS_MODEL_NAME

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
) -> Path:
    """모델명 하위 폴더를 포함한 디버그 이미지 파일 경로를 반환한다."""
    return debug_image_dir(debug_root, model_name=model_name) / filename

__all__ = [
    "debug_image_dir",
    "debug_image_path",
    "connection_check",
    "flask_vlm",
    "logger",
    "pipeline_ocr",
    "prompts",
    "rcs_utils",
    "resolve_debug_model_name",
    "automate_rcs_login",
    "login_rcs",
    "check_tool_screen",
    "click_rcs_view_mode",
    "vlm_client",
    "vlm_screen_analysis",
]
