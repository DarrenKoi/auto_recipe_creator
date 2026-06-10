"""workflow_3 용 OCR assist 프롬프트 빌더."""

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


def build_spotting_prompt() -> tuple[str, str]:
    """PaddleOCR-VL `Spotting:` 태스크 프롬프트를 반환한다.

    `OCR:` 가 평문 텍스트만 주는 것과 달리, `Spotting:` 은 검출 텍스트마다
    bbox 좌표를 함께 돌려준다 (클릭 좌표 산출용).
    """
    return "", "Spotting:"


__all__ = ["build_ocr_assist_prompt", "build_spotting_prompt"]
