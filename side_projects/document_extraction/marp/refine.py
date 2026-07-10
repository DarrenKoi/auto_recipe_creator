"""Marp deck LLM 다듬기 (선택 단계; 값 창작 금지 검증 + 실패 시 원본 유지).

Stage 5 의 결정론적 deck 은 정확하지만 구조가 단조롭다(모든 본문이 최상위 불릿).
이 모듈은 슬라이드 단위로 LLM(kimi-k2.6 비전 겸용 / glm-5.2 텍스트)에 넘겨
헤딩 레벨·불릿 중첩·강조 같은 "구조"만 다듬는다. LLM 출력은 제안일 뿐이고,
기계 검증을 통과해야만 채택된다(값 창작 방지 — 프로젝트 원칙: 모델은 제안,
결정은 결정론 로직):

    1) 원본의 표 행(`|` 라인) / 수식($$...$$) / 이미지 참조(![...](...)) 가
       전부 그대로 살아 있어야 채택
    2) 원본에 없는 숫자가 새로 생기면 기각
    3) 빈 출력/호출 실패/오프라인이면 원본 유지

슬라이드 단위로 처리하므로 토큰 창이 유계이고, 한 슬라이드 실패가 나머지
채택을 막지 않는다.
"""

import os
import re
from pathlib import Path


REFINE_SERVICE_DEFAULT = "glm-5.2"   # 텍스트 다듬기라 로컬 텍스트 LLM 이 기본
REFINE_MAX_TOKENS = 2048

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _offline_env() -> bool:
    return os.getenv("DOC_EXTRACT_OFFLINE", "").strip() in {"1", "true", "True"}


def _table_lines(md: str) -> list[str]:
    """마크다운 표 라인(| 로 시작)들을 반환(순수)."""
    return [line.strip() for line in md.splitlines() if line.strip().startswith("|")]


def _formula_blocks(md: str) -> list[str]:
    """$$ ... $$ 수식 라인들을 반환(순수). generate.py 는 한 줄 수식만 만든다."""
    return [line.strip() for line in md.splitlines()
            if line.strip().startswith("$$") and line.strip().endswith("$$")]


def _image_paths(md: str) -> list[str]:
    return _IMG_RE.findall(md)


def validate_refined_slide(original_md: str, refined_md: str) -> tuple[bool, list[str]]:
    """다듬어진 슬라이드가 원본의 사실을 훼손하지 않았는지 검증한다(순수).

    반환: (채택 가능 여부, 기각 사유 목록).
    """
    reasons: list[str] = []
    refined = (refined_md or "").strip()
    if not refined:
        return False, ["빈 출력"]

    # 1) 표 행 verbatim 보존(순서 무관, 존재 필수)
    refined_tables = set(_table_lines(refined))
    for line in _table_lines(original_md):
        if line not in refined_tables:
            reasons.append(f"표 행 소실/변형: {line[:60]}")

    # 2) 수식 verbatim 보존
    for block in _formula_blocks(original_md):
        if block not in refined:
            reasons.append(f"수식 소실/변형: {block[:60]}")

    # 3) 이미지 참조(crop 재삽입) 경로 보존
    refined_imgs = set(_image_paths(refined))
    for path in _image_paths(original_md):
        if path not in refined_imgs:
            reasons.append(f"이미지 참조 소실: {path}")

    # 4) 새 숫자 금지(값 창작 방지). 원본에 있는 숫자 집합만 허용.
    allowed = set(_NUM_RE.findall(original_md))
    for num in _NUM_RE.findall(refined):
        if num not in allowed:
            reasons.append(f"원본에 없는 숫자 생성: {num}")

    return (not reasons), reasons


def _refine_prompt(slide_md: str) -> tuple[str, str]:
    """슬라이드 1장 다듬기 프롬프트(순수). (system, user) 튜플."""
    system = (
        "You improve the STRUCTURE of one Marp markdown slide: heading levels, "
        "bullet nesting, bold emphasis for key phrases, and logical grouping. "
        "STRICT RULES: keep every markdown table line, every $$formula$$ block, "
        "and every image reference EXACTLY as-is (byte-identical). Do not add, "
        "remove, or change any number, label, or fact. Do not add new content. "
        "Output ONLY the improved markdown for this single slide - no fences, "
        "no commentary, no slide separators (---)."
    )
    user = "SLIDE MARKDOWN:\n" + slide_md
    return system, user


def _default_llm_call(service_slug: str):
    """service slug -> (system, user) -> 응답 텍스트 함수를 만든다(lazy import)."""
    from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

    client = Workflow1VLMClient(service_slug)

    def _call(system_message: str, user_text: str) -> str:
        resp = client.chat_text(
            system_message=system_message,
            user_text=user_text,
            max_tokens=REFINE_MAX_TOKENS,
        )
        return resp.text

    return _call


def _strip_md_fence(text: str) -> str:
    """모델이 ```markdown fence 로 감싸 보낸 경우 벗긴다(순수)."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1 and stripped.rstrip().endswith("```"):
            return stripped[first_nl + 1:].rstrip().removesuffix("```").rstrip()
    return stripped


def split_deck(deck_md: str) -> tuple[str, list[str]]:
    """deck(.md) -> (프론트매터, 슬라이드 목록) (순수).

    generate.results_to_deck 의 출력 형식(프론트매터 + `\\n\\n---\\n\\n` 구분)을 가정.
    프론트매터가 없으면 빈 문자열을 돌려준다.
    """
    front = ""
    body = deck_md
    if deck_md.startswith("---\n"):
        end = deck_md.find("\n---\n", 4)
        if end != -1:
            front = deck_md[: end + 5]
            body = deck_md[end + 5:]
    slides = [s.strip() for s in body.split("\n\n---\n\n")]
    return front, [s for s in slides if s]


def join_deck(front: str, slides: list[str]) -> str:
    """(프론트매터, 슬라이드 목록) -> deck(.md) (순수). split_deck 의 역."""
    body = "\n\n---\n\n".join(slides)
    if front:
        return front + "\n" + body + "\n"
    return body + "\n"


def refine_deck(
    deck_md: str,
    *,
    service_slug: str = REFINE_SERVICE_DEFAULT,
    llm_call=None,
    offline: bool | None = None,
) -> tuple[str, int]:
    """deck 전체를 슬라이드 단위로 LLM 다듬기 한다.

    llm_call: (system, user) -> text 함수 override(테스트용). None 이면 실제 클라이언트.
    offline: None 이면 DOC_EXTRACT_OFFLINE env 로 결정. 오프라인이면 원본 그대로.
    반환: (다듬어진 deck, 채택된 슬라이드 수).
    """
    if _offline_env() if offline is None else bool(offline):
        print("[INFO] refine 오프라인 - deck 원본 유지")
        return deck_md, 0

    front, slides = split_deck(deck_md)
    if not slides:
        return deck_md, 0

    if llm_call is None:
        try:
            llm_call = _default_llm_call(service_slug)
        except Exception as exc:
            print(f"[WARNING] refine 클라이언트 생성 실패({service_slug}) - 원본 유지: {exc}")
            return deck_md, 0

    adopted = 0
    refined_slides: list[str] = []
    for idx, slide in enumerate(slides, start=1):
        # 전체 래스터 강등 슬라이드(![bg ...]) 는 다듬을 텍스트가 없다 - 그대로 둠
        if slide.startswith("![bg"):
            refined_slides.append(slide)
            continue
        try:
            system, user = _refine_prompt(slide)
            candidate = _strip_md_fence(llm_call(system, user))
        except Exception as exc:
            print(f"[WARNING] refine 호출 실패(슬라이드 {idx}) - 원본 유지: {exc}")
            refined_slides.append(slide)
            continue

        ok, reasons = validate_refined_slide(slide, candidate)
        if ok:
            refined_slides.append(candidate)
            adopted += 1
        else:
            print(
                f"[WARNING] refine 기각(슬라이드 {idx}): "
                + "; ".join(reasons[:3])
            )
            refined_slides.append(slide)

    print(f"[INFO] refine 완료: {adopted}/{len(slides)} 슬라이드 채택")
    return join_deck(front, refined_slides), adopted


__all__ = [
    "REFINE_SERVICE_DEFAULT",
    "join_deck",
    "refine_deck",
    "split_deck",
    "validate_refined_slide",
]
