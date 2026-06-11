"""Recipe align image to AlignKeyTemplate materialization."""

from pathlib import Path

from poc.workflow_3.align.assets import AlignFailAssets, load_gray
from poc.workflow_3.align.cond_file import load_cond
from poc.workflow_3.align.cond_template import (
    CENTER_AREA_RATIO,
    centered_area_crop,
    check_cond_box,
    cond_align_offset,
    cond_template_crop,
)
from poc.workflow_3.align.matching.engine import AlignKeyTemplate, build_template


def load_template(
    path: Path, *, recipe_id: str, key_type: str, cond_box_crop: bool
) -> AlignKeyTemplate:
    """Load one registered recipe image as a cond-aware AlignKeyTemplate."""
    gray = load_gray(path)
    if not cond_box_crop:
        crop, offset = gray, (0, 0)
    else:
        cond = load_cond(path)
        box_ltrb = cond.box_ltrb if cond is not None else None
        if box_ltrb is None:
            status = "skip"
            reason = "cond 파일 없음" if cond is None else "box 없음"
        else:
            status, reason, _onorm = check_cond_box(box_ltrb, gray.shape)
        if status != "skip":
            crop, _bbox = cond_template_crop(gray, cond)
            offset = cond_align_offset(box_ltrb, gray.shape)
            level = "WARNING" if status == "warn" else "INFO"
            print(f"[{level}] {key_type} template cond box-crop: offset={offset} ({reason})")
        else:
            crop = centered_area_crop(gray, CENTER_AREA_RATIO)
            offset = (0, 0)
            print(f"[INFO] {key_type} template center-area crop ({reason})")
    return build_template(
        crop,
        recipe_id=recipe_id,
        version="v0",
        key_type=key_type,
        align_offset_xy=offset,
    )


def build_templates_from_assets(
    assets: AlignFailAssets, *, cond_box_crop: bool = True
) -> dict[str, AlignKeyTemplate]:
    """Convert available recipe OM/SEM images to template map."""
    templates: dict[str, AlignKeyTemplate] = {}
    if assets.recipe_om is not None:
        templates["OM"] = load_template(
            assets.recipe_om,
            recipe_id=assets.recipe_id,
            key_type="om",
            cond_box_crop=cond_box_crop,
        )
    if assets.recipe_sem is not None:
        templates["SEM"] = load_template(
            assets.recipe_sem,
            recipe_id=assets.recipe_id,
            key_type="sem",
            cond_box_crop=cond_box_crop,
        )
    return templates


# Internal name kept for tests that explicitly exercise the branch behavior.
_load_template = load_template

__all__ = ["build_templates_from_assets", "load_template"]
