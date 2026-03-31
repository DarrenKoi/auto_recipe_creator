"""workflow_1 용 UI-Venus + MAI-UI locator wrapper."""

from pathlib import Path

from PIL import Image

from poc.work2.ui_venus_mai_locator import (
    EXIT_CAPTURE_FAILED,
    EXIT_SUCCESS,
    EXIT_VLM_NO_DETECTION,
    EXIT_WINDOW_ACTIVATE_FAILED,
    TargetConfig,
    TargetResult,
    analyze_window_target as analyze_window_target_with_shared_pipeline,
)
from poc.work2.workflow_1.prompts import (
    build_mai_ui_zoom_prompt,
    build_ui_venus_single_element_bbox_prompt,
)


def analyze_window_target(
    window,
    window_title: str,
    backend: str,
    target: TargetConfig,
    *,
    debug_image_dir: Path,
    log_name: str,
    component_name: str,
    artifact_prefix: str,
    coarse_service_slug: str = "ui-venus",
    refine_service_slug: str = "mai-ui",
    result_mode: str = "workflow_1_ui_venus_then_mai_ui_single_target",
    image: Image.Image | None = None,
) -> TargetResult:
    """workflow_1 전용 prompt 세트를 사용해 타겟을 찾는다."""
    return analyze_window_target_with_shared_pipeline(
        window,
        window_title,
        backend,
        target,
        debug_image_dir=debug_image_dir,
        log_name=log_name,
        component_name=component_name,
        artifact_prefix=artifact_prefix,
        coarse_service_slug=coarse_service_slug,
        refine_service_slug=refine_service_slug,
        result_mode=result_mode,
        image=image,
        coarse_prompt_builder=build_ui_venus_single_element_bbox_prompt,
        refine_prompt_builder=build_mai_ui_zoom_prompt,
    )


__all__ = [
    "EXIT_CAPTURE_FAILED",
    "EXIT_SUCCESS",
    "EXIT_VLM_NO_DETECTION",
    "EXIT_WINDOW_ACTIVATE_FAILED",
    "TargetConfig",
    "TargetResult",
    "analyze_window_target",
]
