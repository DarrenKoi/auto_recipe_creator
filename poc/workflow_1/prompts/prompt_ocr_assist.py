"""workflow_1 용 OCR assist 프롬프트 빌더."""

from typing import Iterable


def build_ocr_assist_prompt(
    width: int,
    height: int,
    context_label: str = "",
    focus_words: Iterable[str] | None = None,
    max_items: int = 12,
) -> tuple[str, str]:
    """PaddleOCR-VL 용 OCR 태스크 키워드 프롬프트를 반환한다."""
    _ = (width, height, context_label, focus_words, max_items)
    return "", "OCR:"


__all__ = ["build_ocr_assist_prompt"]
