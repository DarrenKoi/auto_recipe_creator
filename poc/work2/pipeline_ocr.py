"""OCR assist helpers for the `ui-venus -> paddleocr` work2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from poc.work.vlm_openai_client import ChatImageRequest, LangChainOpenAICompatibleVLMClient
from poc.work2.prompts import build_ocr_assist_prompt


@dataclass(frozen=True)
class OCRHintResult:
    texts: tuple[str, ...]
    focus_hits: tuple[str, ...]
    raw_response: str


def _parse_ocr_lines(raw: str, max_items: int) -> tuple[str, ...]:
    """PaddleOCR-VL plain text 응답을 줄 단위로 파싱한다."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        cleaned.append(text)
        seen.add(normalized)
        if len(cleaned) >= max_items:
            break
    return tuple(cleaned)


def _match_focus_words(
    texts: tuple[str, ...],
    focus_words: Iterable[str] | None,
) -> tuple[str, ...]:
    """추출된 텍스트 중 focus_words 에 매칭되는 항목을 반환한다."""
    if not focus_words:
        return ()
    targets = [w.strip().lower() for w in focus_words if w and w.strip()]
    if not targets:
        return ()
    hits: list[str] = []
    for text in texts:
        text_lower = text.lower()
        for target in targets:
            if target in text_lower:
                hits.append(text)
                break
    return tuple(hits)


def collect_ocr_hint_result(
    *,
    image_b64: str,
    image_width: int,
    image_height: int,
    pipeline_config: dict[str, object],
    image_mime: str = "image/webp",
    context_label: str = "",
    focus_words: Iterable[str] | None = None,
    max_items: int = 12,
    temperature: float = 0.0,
) -> OCRHintResult | None:
    """Run PaddleOCR stage and return extracted OCR hints."""
    if pipeline_config.get("ocr_pipeline_enabled") is False:
        print("[INFO] work2 OCR pipeline 비활성화 — OCR 힌트 수집 건너뜀")
        return None

    api_url = str(pipeline_config.get("ocr_api_url", "") or "").strip()
    model_name = str(pipeline_config.get("ocr_model_name", "") or "").strip()
    api_key = str(pipeline_config.get("ocr_api_key", "") or "").strip()
    service_name = str(pipeline_config.get("ocr_service", "") or "").strip()

    if not api_url or not model_name:
        print("[INFO] OCR API URL 또는 OCR model name 미설정 — OCR 힌트 수집 건너뜀")
        return None

    system_message, user_text = build_ocr_assist_prompt(
        width=image_width,
        height=image_height,
        context_label=context_label,
        focus_words=focus_words,
        max_items=max_items,
    )
    client = LangChainOpenAICompatibleVLMClient(
        base_url=api_url,
        api_key=api_key,
        timeout_sec=120.0,
    )
    request = ChatImageRequest(
        model=model_name,
        system_message=system_message,
        user_text=user_text,
        image_b64=image_b64,
        image_mime=image_mime,
        temperature=temperature,
    )

    print(
        f"[INFO] OCR assist 호출: service={service_name}, model={model_name}, "
        f"endpoint={client.endpoint}"
    )

    try:
        raw = client.chat_with_image(request)
    except Exception as exc:
        print(f"[WARNING] OCR assist 호출 실패: {exc}")
        return None

    print(f"[INFO] OCR assist 원문 응답:\n{raw}\n")

    texts = _parse_ocr_lines(raw, max_items=max_items)
    focus_hits = _match_focus_words(texts, focus_words)
    if texts:
        print(f"[INFO] OCR 추출 텍스트: {', '.join(texts)}")
    if focus_hits:
        print(f"[INFO] OCR focus hit: {', '.join(focus_hits)}")

    return OCRHintResult(
        texts=texts,
        focus_hits=focus_hits,
        raw_response=raw,
    )


def build_ocr_extra_instructions(
    result: OCRHintResult | None,
    *,
    max_items: int = 8,
) -> tuple[str, ...]:
    """Convert OCR result into short prompt instructions for the primary VLM."""
    if result is None:
        return ()

    instructions: list[str] = [
        "Use OCR only as auxiliary context. Final coordinates and layout judgment must still come from the actual pixels."
    ]
    if result.texts:
        instructions.append(
            "OCR observed these visible texts: " + ", ".join(result.texts[:max_items]) + "."
        )
    if result.focus_hits:
        instructions.append(
            "OCR confirmed these target texts are visible: "
            + ", ".join(result.focus_hits[:max_items])
            + "."
        )
    return tuple(instructions)
