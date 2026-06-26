"""RAG chunk 검색 + quality gate (rag_db_plan.md).

embedding backend 없이 동작하는 첫 단계 retrieval: JSONL chunk 를 읽어 keyword +
metadata 필터로 검색하고, quality gate 로 trusted / lower-trust tier 를 나눈다.
순수 함수라 집에서 검증된다(embedding/vector DB 는 후속 단계).
"""

import json
import re
from pathlib import Path

from side_projects.document_extraction.extraction.schemas import CHUNK_TYPES


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# quality gate 의 confidence 임계 (rag_chunks.TRUST_CONFIDENCE 와 동일)
TRUST_CONFIDENCE = 0.7

# 알려진 chunk type 은 schemas.CHUNK_TYPES 단일 출처에서 파생(중복/드리프트 방지).
_KNOWN_REGION_TYPES = set(CHUNK_TYPES)


def _safe_float(value, default: float = 0.0) -> float:
    """None/문자열/garbage 를 안전하게 float 로 강제(실패 시 default)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_chunks(jsonl_path: str | Path) -> list[dict]:
    """rag_chunks.jsonl 을 읽어 dict 목록으로 반환(빈 줄 무시)."""
    path = Path(jsonl_path)
    chunks: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[WARNING] JSONL 파싱 실패(무시): {exc}")
    return chunks


def quality_gate(chunk: dict, *, check_file_exists: bool = False) -> tuple[str, list]:
    """chunk 를 rag_db_plan.md Quality Gates 로 평가해 (tier, reasons) 반환.

    tier: "trusted" (모든 게이트 통과) | "lower_trust" (하나라도 실패).
    reasons: 실패 사유 목록(통과면 빈 리스트).
    """
    reasons: list = []

    content = str(chunk.get("content") or "").strip()
    if not content:
        reasons.append("empty content")

    source_image = str(chunk.get("source_image") or "").strip()
    if not source_image:
        reasons.append("missing source_image")
    elif check_file_exists and not Path(source_image).exists():
        reasons.append("source_image file not found")

    region_type = str(chunk.get("region_type") or "")
    if region_type not in _KNOWN_REGION_TYPES:
        reasons.append(f"unknown region_type: {region_type or '(empty)'}")

    confidence = _safe_float(chunk.get("confidence"))
    approved = str(chunk.get("review_status") or "") == "approved"
    if confidence < TRUST_CONFIDENCE and not approved:
        reasons.append("low confidence and not approved")

    # table/chart chunk 는 screenshot 없이 이해할 만큼 label 보유해야 함
    if region_type in {"table_summary", "chart_summary", "table_row"}:
        has_labels = bool(str(chunk.get("parent_heading") or "").strip()) or len(content) >= 20
        if not has_labels:
            reasons.append("table/chart chunk lacks labels/heading")

    return ("trusted" if not reasons else "lower_trust"), reasons


def _searchable_text(chunk: dict) -> str:
    """검색 대상 텍스트: content + heading + embedding_text + raw_ocr_text."""
    return " ".join(
        str(chunk.get(k) or "")
        for k in ("content", "parent_heading", "embedding_text", "raw_ocr_text")
    )


def _tokens(text: str) -> list:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _passes_filters(chunk: dict, filters: dict) -> bool:
    """metadata 필터 통과 여부. filters 의 빈/None 값은 무시."""
    for key in ("source_type", "document_id", "screenshot_id", "region_type", "review_status"):
        want = filters.get(key)
        if want and str(chunk.get(key) or "") != str(want):
            return False
    min_conf = filters.get("min_confidence")
    if min_conf is not None:
        if _safe_float(chunk.get("confidence")) < _safe_float(min_conf):
            return False
    return True


def keyword_search(
    chunks: list,
    query: str,
    *,
    filters: dict | None = None,
    tier: str = "all",
    check_file_exists: bool = False,
    limit: int = 20,
) -> list:
    """keyword(AND) + metadata 필터 + quality tier 로 chunk 를 검색한다.

    - query 의 모든 토큰이 chunk 검색 텍스트에 있어야 매칭(AND).
    - tier: "all" | "trusted" (trusted 만) | "lower_trust" (저신뢰만).
    - 랭킹: heading 매칭 가중 + 토큰 빈도. 동점은 confidence 내림차순.
    반환: [{chunk, score, tier, gate_reasons}] (score 내림차순, limit 까지).
    """
    filters = filters or {}
    q_tokens = set(_tokens(query))
    results = []

    for chunk in chunks:
        if not _passes_filters(chunk, filters):
            continue
        chunk_tier, reasons = quality_gate(chunk, check_file_exists=check_file_exists)
        if tier != "all" and chunk_tier != tier:
            continue

        if q_tokens:
            body_tokens = _tokens(_searchable_text(chunk))
            body_set = set(body_tokens)
            if not q_tokens.issubset(body_set):
                continue
            # 빈도 점수 + heading 매칭 가중
            freq = sum(body_tokens.count(t) for t in q_tokens)
            heading_tokens = set(_tokens(str(chunk.get("parent_heading") or "")))
            heading_hits = len(q_tokens & heading_tokens)
            score = freq + 3.0 * heading_hits
        else:
            score = 0.0  # 빈 query = 필터/tier 브라우징

        results.append(
            {
                "chunk": chunk,
                "score": round(score, 3),
                "tier": chunk_tier,
                "gate_reasons": reasons,
            }
        )

    results.sort(
        key=lambda r: (r["score"], _safe_float(r["chunk"].get("confidence"))),
        reverse=True,
    )
    return results[:limit]


def partition_by_tier(chunks: list, *, check_file_exists: bool = False) -> dict:
    """chunk 들을 trusted / lower_trust 로 분할한다(개수 요약 포함)."""
    trusted, lower = [], []
    for chunk in chunks:
        chunk_tier, _ = quality_gate(chunk, check_file_exists=check_file_exists)
        (trusted if chunk_tier == "trusted" else lower).append(chunk)
    return {
        "trusted": trusted,
        "lower_trust": lower,
        "counts": {"trusted": len(trusted), "lower_trust": len(lower), "total": len(chunks)},
    }


__all__ = [
    "TRUST_CONFIDENCE",
    "keyword_search",
    "load_chunks",
    "partition_by_tier",
    "quality_gate",
]
