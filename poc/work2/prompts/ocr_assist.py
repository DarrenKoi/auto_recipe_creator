"""OCR assist prompt builders for `poc.work2` pipeline.

PaddleOCR-VL-1.5 는 0.9B 파라미터 모델로 6가지 태스크 키워드에 대해 학습되었다:
OCR:, Table Recognition:, Formula Recognition:,
Chart Recognition:, Spotting:, Seal Recognition:

시스템 메시지나 복잡한 지시 없이 태스크 키워드만 전송해야 한다.
응답은 plain text 로 반환되므로 호출 측에서 후처리한다.
"""

from typing import Iterable


def build_ocr_assist_prompt(
    width: int,
    height: int,
    context_label: str = "",
    focus_words: Iterable[str] | None = None,
    max_items: int = 12,
) -> tuple[str, str]:
    """PaddleOCR-VL 용 OCR 태스크 키워드 프롬프트를 반환한다.

    PaddleOCR-VL-1.5 는 ``OCR:`` 키워드에 대해 학습되었으므로
    시스템 메시지 없이 키워드만 전송한다.
    ``context_label``, ``focus_words``, ``max_items`` 는
    호출 측에서 응답 후처리에 사용된다.
    """
    return "", "OCR:"
