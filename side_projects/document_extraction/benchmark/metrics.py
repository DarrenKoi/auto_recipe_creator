"""채점 메트릭 (benchmark_plan.md). 모두 순수 함수.

입력 규약:
    extraction : extraction/ 가 만든 ExtractionResult.to_dict() (또는 raw_evidence JSON)
    gt         : GroundTruth

각 메트릭은 (score: float in 0..1, detail: dict) 를 반환한다. detail 에는 무엇이
맞고 빠졌는지 근거를 담아 사람이 review 할 수 있게 한다. Hallucination 만 rate
(낮을수록 좋음)와 count 를 함께 돌려준다.
"""

import re

from side_projects.document_extraction.benchmark.ground_truth import GroundTruth


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# 단어 경계 숫자만(예: "Q2" 의 "2" 는 잡지 않음, 공백으로 둘러싼 "30" 은 잡음).
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?(?![A-Za-z0-9])")


def _norm(text: str) -> str:
    """소문자 + 공백 collapse 정규화."""
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set:
    return set(m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))


def _phrase_found(phrase: str, haystack_norm: str, haystack_tokens: set) -> bool:
    """구가 텍스트에 있는지: 정규화 substring 또는 토큰 0.7 이상 포함."""
    p_norm = _norm(phrase)
    if not p_norm:
        return True
    if p_norm in haystack_norm:
        return True
    p_tokens = _tokens(phrase)
    if not p_tokens:
        return False
    overlap = len(p_tokens & haystack_tokens) / len(p_tokens)
    return overlap >= 0.7


def _all_extraction_text(extraction: dict) -> str:
    """region/table/chart/formula/summary 의 모든 텍스트를 한 덩어리로."""
    parts: list[str] = [extraction.get("summary_markdown") or ""]
    for r in extraction.get("regions") or []:
        parts.append(r.get("text") or "")
    for t in extraction.get("tables") or []:
        parts.append(t.get("title") or "")
        parts.extend(t.get("header") or [])
        for row in t.get("cells") or []:
            parts.extend(row)
    for c in extraction.get("charts") or []:
        parts.append(c.get("title") or "")
        parts.extend(c.get("axis_labels") or [])
        parts.extend(c.get("legend_labels") or [])
        parts.extend(c.get("visible_values") or [])
        parts.append(c.get("trend_summary") or "")
    for f in extraction.get("formulas") or []:
        parts.append(f.get("latex") or "")
    return " ".join(str(p) for p in parts)


# --- Text Recall -----------------------------------------------------------

def text_recall(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """중요한 visible text 가 추출에 나타나는 비율(unreadable 은 제외)."""
    targets = list(gt.important_texts)
    if gt.title:
        targets.append(gt.title)
    if not targets:
        return 1.0, {"targets": 0, "note": "no important_texts in GT"}

    blob = _all_extraction_text(extraction)
    blob_norm = _norm(blob)
    blob_tokens = _tokens(blob)

    found, missed = [], []
    for phrase in targets:
        if _phrase_found(phrase, blob_norm, blob_tokens):
            found.append(phrase)
        else:
            missed.append(phrase)

    score = len(found) / len(targets)
    return score, {"found": found, "missed": missed, "recall": round(score, 3)}


# --- Table Accuracy --------------------------------------------------------

def _best_table_match(gt_table, ex_tables: list) -> dict | None:
    """header 토큰 겹침이 가장 큰 추출 표를 고른다."""
    gt_header_tokens = set()
    for h in gt_table.header:
        gt_header_tokens |= _tokens(h)
    best, best_overlap = None, -1.0
    for ex in ex_tables:
        ex_tokens = set()
        for h in ex.get("header") or []:
            ex_tokens |= _tokens(str(h))
        overlap = len(gt_header_tokens & ex_tokens)
        if overlap > best_overlap:
            best, best_overlap = ex, overlap
    return best


def table_accuracy(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """표별 header + cell 정확도의 평균 (0..1)."""
    if not gt.tables:
        return 1.0, {"tables": 0, "note": "no GT tables"}

    ex_tables = extraction.get("tables") or []
    per_table = []
    for gt_table in gt.tables:
        match = _best_table_match(gt_table, ex_tables)
        if match is None:
            per_table.append({"title": gt_table.title, "score": 0.0, "matched": False})
            continue

        # header correctness: GT header 중 매칭표 header 에 있는 비율
        ex_header_norm = {_norm(h) for h in (match.get("header") or [])}
        hdr_hits = sum(1 for h in gt_table.header if _norm(h) in ex_header_norm)
        hdr_score = hdr_hits / len(gt_table.header) if gt_table.header else 1.0

        # cell correctness: GT cell 텍스트가 매칭표 어딘가에 있는 비율
        ex_cell_norm = set()
        for row in match.get("cells") or []:
            for c in row:
                ex_cell_norm.add(_norm(str(c)))
        gt_cells = [c for row in gt_table.rows for c in row]
        cell_hits = sum(1 for c in gt_cells if _norm(c) in ex_cell_norm)
        cell_score = cell_hits / len(gt_cells) if gt_cells else 1.0

        # extra/missing column penalty: header 길이 차이를 약하게 반영
        gt_cols, ex_cols = len(gt_table.header), len(match.get("header") or [])
        col_penalty = abs(gt_cols - ex_cols) / max(gt_cols, ex_cols, 1)

        score = max(0.0, (hdr_score + cell_score) / 2 - 0.25 * col_penalty)
        per_table.append(
            {
                "title": gt_table.title,
                "matched": True,
                "header_score": round(hdr_score, 3),
                "cell_score": round(cell_score, 3),
                "col_penalty": round(col_penalty, 3),
                "score": round(score, 3),
            }
        )

    mean = sum(t["score"] for t in per_table) / len(per_table)
    return mean, {"per_table": per_table, "mean": round(mean, 3)}


# --- Chart Understanding ---------------------------------------------------

def chart_understanding(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """차트의 title/axis/legend/visible_value/trend 복원 비율 (보이는 것만)."""
    if not gt.charts:
        return 1.0, {"charts": 0, "note": "no GT charts"}

    ex_charts = extraction.get("charts") or []
    # 모든 추출 차트 텍스트를 한 덩어리로 묶어 존재 여부를 본다(차트 매칭 단순화).
    blob = ""
    for c in ex_charts:
        blob += " " + (c.get("title") or "")
        blob += " " + " ".join(c.get("axis_labels") or [])
        blob += " " + " ".join(c.get("legend_labels") or [])
        blob += " " + " ".join(c.get("visible_values") or [])
        blob += " " + (c.get("trend_summary") or "")
    blob_norm = _norm(blob)
    blob_tokens = _tokens(blob)

    per_chart = []
    for gt_chart in gt.charts:
        checks: list[tuple[str, bool]] = []
        if gt_chart.title:
            checks.append(("title", _phrase_found(gt_chart.title, blob_norm, blob_tokens)))
        for a in gt_chart.axis_labels:
            checks.append((f"axis:{a}", _phrase_found(a, blob_norm, blob_tokens)))
        for l in gt_chart.legend_labels:
            checks.append((f"legend:{l}", _phrase_found(l, blob_norm, blob_tokens)))
        for v in gt_chart.visible_values:
            checks.append((f"value:{v}", _norm(v) in blob_norm))
        if gt_chart.trend:
            checks.append(("trend", _phrase_found(gt_chart.trend, blob_norm, blob_tokens)))

        if not checks:
            per_chart.append({"title": gt_chart.title, "score": 1.0})
            continue
        score = sum(1 for _, ok in checks if ok) / len(checks)
        per_chart.append(
            {
                "title": gt_chart.title,
                "score": round(score, 3),
                "missed": [name for name, ok in checks if not ok],
            }
        )

    mean = sum(c["score"] for c in per_chart) / len(per_chart)
    return mean, {"per_chart": per_chart, "mean": round(mean, 3)}


# --- Layout Accuracy -------------------------------------------------------

def layout_accuracy(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """기대 region type(multiset) 이 추출 region type 에 얼마나 들어맞는지."""
    if not gt.region_types:
        return 1.0, {"note": "no GT region_types"}

    ex_types: list[str] = [str(r.get("type", "")) for r in extraction.get("regions") or []]
    # multiset 교집합 비율
    ex_pool = list(ex_types)
    hits = 0
    for want in gt.region_types:
        if want in ex_pool:
            ex_pool.remove(want)
            hits += 1
    score = hits / len(gt.region_types)
    return score, {
        "expected": gt.region_types,
        "got": ex_types,
        "matched": hits,
        "score": round(score, 3),
    }


# --- Hallucination Rate ----------------------------------------------------

def _allowed_visible_text(gt: GroundTruth) -> str:
    """GT 의 '보이는' 텍스트를 한 덩어리로 모은다(hallucination 판정 기준)."""
    parts: list[str] = [gt.title, *gt.important_texts, *gt.expected_summary_keywords,
                        *gt.visible_tokens]
    for tab in gt.tables:
        parts.append(tab.title)
        parts.extend(tab.header)
        for row in tab.rows:
            parts.extend(row)
    for ch in gt.charts:
        parts.append(ch.title)
        parts.extend(ch.axis_labels)
        parts.extend(ch.legend_labels)
        parts.extend(ch.visible_values)
        parts.append(ch.trend)
    return " ".join(str(p) for p in parts)


def hallucination_rate(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """추출이 만든 숫자 중 GT 의 보이는 근거에 없는 것의 비율(낮을수록 좋음).

    숫자(number)를 hallucination proxy 로 본다: 만들어낸 수치가 가장 위험.
    summary_markdown + chart visible_values 의 숫자를 검사한다.
    """
    allowed_numbers = set(_NUM_RE.findall(_allowed_visible_text(gt)))

    claimed: list[str] = []
    claimed += _NUM_RE.findall(extraction.get("summary_markdown") or "")
    for c in extraction.get("charts") or []:
        for v in c.get("visible_values") or []:
            claimed += _NUM_RE.findall(str(v))

    if not claimed:
        return 0.0, {"claimed_numbers": 0, "hallucinated": [], "rate": 0.0}

    hallucinated = [n for n in claimed if n not in allowed_numbers]
    rate = len(hallucinated) / len(claimed)
    return rate, {
        "claimed_numbers": len(claimed),
        "hallucinated": hallucinated,
        "count": len(hallucinated),
        "rate": round(rate, 3),
    }


# --- RAG Readiness ---------------------------------------------------------

def rag_readiness(extraction: dict, gt: GroundTruth) -> tuple[float, dict]:
    """chunk 들이 retrieval 에 충분한 provenance/context 를 갖는 비율 (benchmark_plan.md).

    chunk 별 체크:
        - source_image 존재
        - screenshot_index >= 1
        - region_type 알려짐(빈 문자열 아님)
        - content 가 충분히 구체적(>= 20자) 또는 heading 보유
        - table/chart chunk 는 라벨(heading/숫자) 유지
        - 저신뢰(<0.7)는 review_status 로 명시 표시
    """
    chunks = extraction.get("rag_chunks") or []
    if not chunks:
        return 0.0, {"chunks": 0, "note": "no rag_chunks"}

    known_types = {
        "document_summary", "region_text", "table_summary",
        "table_row", "chart_summary", "formula", "unresolved",
    }
    per_chunk_scores = []
    failures: list[str] = []
    for ch in chunks:
        checks = []
        checks.append(bool(ch.get("source_image")))
        checks.append(int(ch.get("screenshot_index") or 0) >= 1)
        rtype = str(ch.get("region_type") or "")
        checks.append(rtype in known_types)

        content = str(ch.get("content") or "")
        specific = len(content.strip()) >= 20 or bool(ch.get("parent_heading"))
        checks.append(specific)

        if rtype in {"table_summary", "chart_summary"}:
            checks.append(bool(ch.get("parent_heading")) or bool(ch.get("content")))

        conf = float(ch.get("confidence") or 0.0)
        if conf < 0.7:
            checks.append(str(ch.get("review_status") or "") in {"needs_review", "rejected"})

        passed = sum(1 for c in checks if c)
        score = passed / len(checks) if checks else 0.0
        per_chunk_scores.append(score)
        if score < 1.0:
            failures.append(f"{ch.get('chunk_id', '?')}: {passed}/{len(checks)} checks")

    mean = sum(per_chunk_scores) / len(per_chunk_scores)
    return mean, {"chunks": len(chunks), "mean": round(mean, 3), "failures": failures[:10]}


# --- Latency (보고용, 점수 아님) -------------------------------------------

def latency_summary(extraction: dict) -> dict:
    """stage_log 에서 service 별 latency 합/평균을 집계한다."""
    log = extraction.get("stage_log") or []
    total_ms = 0.0
    per_service: dict[str, float] = {}
    calls = 0
    for entry in log:
        ms = float(entry.get("latency_ms") or 0.0)
        total_ms += ms
        svc = str(entry.get("service") or "?")
        per_service[svc] = per_service.get(svc, 0.0) + ms
        if entry.get("mode", "").startswith("online"):
            calls += 1
    return {
        "total_ms": round(total_ms, 1),
        "online_calls": calls,
        "per_service_ms": {k: round(v, 1) for k, v in per_service.items()},
    }


__all__ = [
    "chart_understanding",
    "hallucination_rate",
    "latency_summary",
    "layout_accuracy",
    "rag_readiness",
    "table_accuracy",
    "text_recall",
]
