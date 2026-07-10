"""Phase 1.5 검색 벤치용 golden 질의셋 스키마 + 로더 + 예시 템플릿 writer.

rag_chart_heavy_architecture.md Phase 1.5: golden 질의셋에 **차트-온리 질의**
(값/추세를 차트 픽셀에서만 읽을 수 있는 것) 계층을 별도로 두고 Recall@k /
parser-loss recovery 를 잰다. 질의는 tier 로 계층화한다:

    chart_only  답이 차트 픽셀에만 있음(본문/표에 없음) - 비전 arm 의 존재 이유
    table       답이 표 셀에 있음
    text        답이 일반 본문 텍스트에 있음
    mixed       텍스트+차트/표를 함께 봐야 답이 됨

relevance 매처는 두 형태를 허용한다(사내 GT 작성 비용 최소화):
    {"chunk_id": "..."}                             정밀(청크 단위)
    {"screenshot_id": "...", "region_type": "..."}  페이지 수준(+선택 region 제한)
       - region_type 생략 시 해당 스크린샷의 모든 chunk 가 정답 취급
       - "document_id" 를 추가로 제한 조건으로 둘 수 있다

실제 golden JSON 은 사내 데이터라 repo 에 커밋하지 않는다(예시만 커밋:
golden_retrieval_queries.example.json). GT 작성은 사내, 채점은 어디서나.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


# 질의 tier (chart_only 가 이 벤치의 존재 이유 - 비중 있게 작성할 것)
TIERS: tuple[str, ...] = ("chart_only", "table", "text", "mixed")

# relevance 매처에 허용되는 키
_MATCHER_KEYS = {"chunk_id", "screenshot_id", "region_type", "document_id"}


@dataclass
class GoldenQuery:
    """golden 질의 1개."""

    query_id: str
    tier: str
    query_text: str
    # 한/영 변형 질의(선택). 평가 시 대표 query_text 만 쓰되 확장 실험용으로 보존.
    query_variants: list[str] = field(default_factory=list)
    # relevance 매처 목록(OR). 형태는 모듈 docstring 참고.
    relevant: list[dict] = field(default_factory=list)
    # QA 확장용 정답 텍스트(선택; Phase 1.5 는 retrieval 만 채점)
    expected_answer: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "tier": self.tier,
            "query_text": self.query_text,
            "query_variants": list(self.query_variants),
            "relevant": [dict(m) for m in self.relevant],
            "expected_answer": self.expected_answer,
            "notes": self.notes,
        }


def validate_query(raw: dict, *, index: int = 0) -> list[str]:
    """질의 dict 1개의 스키마 오류 목록을 반환(비었으면 유효, 순수)."""
    errors: list[str] = []
    where = f"queries[{index}]"

    if not str(raw.get("query_id") or "").strip():
        errors.append(f"{where}: query_id 누락")
    tier = str(raw.get("tier") or "")
    if tier not in TIERS:
        errors.append(f"{where}: tier '{tier}' 는 {TIERS} 중 하나여야 함")
    if not str(raw.get("query_text") or "").strip():
        errors.append(f"{where}: query_text 누락")

    relevant = raw.get("relevant") or []
    if not relevant:
        errors.append(f"{where}: relevant 매처가 최소 1개 필요")
    for midx, matcher in enumerate(relevant):
        if not isinstance(matcher, dict):
            errors.append(f"{where}.relevant[{midx}]: dict 가 아님")
            continue
        unknown = set(matcher) - _MATCHER_KEYS
        if unknown:
            errors.append(
                f"{where}.relevant[{midx}]: 알 수 없는 키 {sorted(unknown)}"
            )
        if not (
            str(matcher.get("chunk_id") or "").strip()
            or str(matcher.get("screenshot_id") or "").strip()
        ):
            errors.append(
                f"{where}.relevant[{midx}]: chunk_id 또는 screenshot_id 필요"
            )
    return errors


def load_golden(path: str | Path) -> list[GoldenQuery]:
    """golden JSON -> GoldenQuery 목록. 스키마 오류가 있으면 ValueError.

    query_id 중복도 오류(채점 dict 키 충돌 방지 - recipe_id 충돌 교훈).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_queries = data.get("queries") or []
    if not raw_queries:
        raise ValueError(f"golden 질의가 없습니다: {path}")

    errors: list[str] = []
    seen_ids: set[str] = set()
    queries: list[GoldenQuery] = []
    for idx, raw in enumerate(raw_queries):
        errors.extend(validate_query(raw, index=idx))
        qid = str(raw.get("query_id") or "")
        if qid in seen_ids:
            errors.append(f"queries[{idx}]: query_id 중복 '{qid}'")
        seen_ids.add(qid)
        queries.append(
            GoldenQuery(
                query_id=qid,
                tier=str(raw.get("tier") or ""),
                query_text=str(raw.get("query_text") or ""),
                query_variants=[str(v) for v in (raw.get("query_variants") or [])],
                relevant=[dict(m) for m in (raw.get("relevant") or [])
                          if isinstance(m, dict)],
                expected_answer=str(raw.get("expected_answer") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )
    if errors:
        raise ValueError(
            f"golden 스키마 오류 {len(errors)}건:\n" + "\n".join(errors)
        )

    tier_counts = {t: sum(1 for q in queries if q.tier == t) for t in TIERS}
    print(f"[INFO] golden 로드: {len(queries)}질의, tier={tier_counts}")
    if tier_counts["chart_only"] == 0:
        print("[WARNING] chart_only 질의가 0건 - 이 벤치의 핵심 계층이 비어 있음")
    return queries


def matcher_matches_hit(matcher: dict, hit: dict) -> bool:
    """relevance 매처 1개 <-> 검색 hit 1개 매칭(순수).

    chunk_id 가 있으면 정확 일치만 본다(가장 정밀).
    아니면 screenshot_id 일치 + (있다면) region_type/document_id 제한을 함께 확인.
    """
    want_chunk = str(matcher.get("chunk_id") or "").strip()
    if want_chunk:
        return str(hit.get("chunk_id") or hit.get("_id") or "") == want_chunk

    want_sid = str(matcher.get("screenshot_id") or "").strip()
    if not want_sid or str(hit.get("screenshot_id") or "") != want_sid:
        return False
    want_region = str(matcher.get("region_type") or "").strip()
    if want_region and str(hit.get("region_type") or "") != want_region:
        return False
    want_doc = str(matcher.get("document_id") or "").strip()
    if want_doc and str(hit.get("document_id") or "") != want_doc:
        return False
    return True


def is_relevant(query: GoldenQuery, hit: dict) -> bool:
    """hit 이 질의의 정답(any 매처)인지 판정(순수)."""
    return any(matcher_matches_hit(m, hit) for m in query.relevant)


def write_example_golden(path: str | Path) -> None:
    """사내 GT 작성 시작점이 될 예시 golden JSON 을 쓴다(합성 데이터).

    chart_only 를 비중 있게(핵심 계층) + 각 tier 예시 1개씩.
    """
    example = {
        "_readme": (
            "Phase 1.5 검색 golden 질의셋. tier=chart_only 가 핵심(답이 차트 "
            "픽셀에만 있는 질의). relevant 는 chunk_id 정밀 매칭 또는 "
            "screenshot_id(+region_type) 페이지 수준 매칭. 실데이터 golden 은 "
            "커밋 금지(사내 전용)."
        ),
        "queries": [
            {
                "query_id": "q001",
                "tier": "chart_only",
                "query_text": "2분기 수율 추세는 개선됐는가?",
                "query_variants": ["Q2 yield trend improvement"],
                "relevant": [
                    {"screenshot_id": "doc1_s003", "region_type": "chart_summary"}
                ],
                "expected_answer": "개선(하락 추세 -> AI 적용 후 회복)",
                "notes": "값이 라인 차트에만 있음. 본문/표에 수치 없음",
            },
            {
                "query_id": "q002",
                "tier": "chart_only",
                "query_text": "장비별 알람 건수가 가장 많은 장비는?",
                "relevant": [{"chunk_id": "doc2_s005_c001"}],
                "expected_answer": "EQP-07",
                "notes": "막대 차트의 최대 막대 - 축 라벨만 텍스트로 추출됨",
            },
            {
                "query_id": "q003",
                "tier": "table",
                "query_text": "manual 모드 셋업 시간은 몇 분인가?",
                "relevant": [
                    {"screenshot_id": "doc1_s001", "region_type": "table_row"}
                ],
                "expected_answer": "30분",
            },
            {
                "query_id": "q004",
                "tier": "text",
                "query_text": "레시피 자동화 프로젝트의 목적은?",
                "relevant": [
                    {"screenshot_id": "doc1_s001", "region_type": "region_text"}
                ],
            },
            {
                "query_id": "q005",
                "tier": "mixed",
                "query_text": "셋업 시간 단축 효과를 수치와 추세로 설명하라",
                "relevant": [
                    {"screenshot_id": "doc1_s001"},
                    {"screenshot_id": "doc1_s003", "region_type": "chart_summary"},
                ],
                "notes": "표(수치) + 차트(추세) 둘 다 필요",
            },
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] 예시 golden 저장: {out}")


__all__ = [
    "GoldenQuery",
    "TIERS",
    "is_relevant",
    "load_golden",
    "matcher_matches_hit",
    "validate_query",
    "write_example_golden",
]
