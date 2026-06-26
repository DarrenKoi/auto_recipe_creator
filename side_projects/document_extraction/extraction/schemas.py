"""추출 파이프라인 출력 계약(schema).

pipeline_plan.md 의 Output Contract 와 rag_db_plan.md 의 Chunk Schema 를
dataclass 로 옮긴 것. 첫 구현은 dict 로 시작해도 되지만, schema 가 흔들리지
않도록 타입을 고정해 둔다. 모든 dataclass 는 `to_dict()` 로 JSON 직렬화 가능한
순수 dict 를 반환한다(파이프라인 산출물을 JSON/JSONL 로 떨구기 위함).
"""

from dataclasses import asdict, dataclass, field


def _coerce_float(value, default: float = 0.0) -> float:
    """None/문자열/garbage 를 안전하게 float 로(실패 시 default). '3.0' 도 허용."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int = 0) -> int:
    """안전한 int 강제: float 경유라 '3.0'/3.0 도 3 으로. 0 도 보존(`or` 미사용)."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# 알려진 region type (pipeline_plan.md Stage 3). 그 외는 "other".
REGION_TYPES: tuple[str, ...] = (
    "title",
    "body",
    "table",
    "chart",
    "formula",
    "footer",
    "legend",
    "other",
)

# RAG chunk type (rag_db_plan.md "Chunk Types" + Phase 1 digital-harvest 확장).
CHUNK_TYPES: tuple[str, ...] = (
    "document_summary",
    "region_text",
    "table_summary",
    "table_row",
    "chart_summary",
    "formula",
    "unresolved",
    # Phase 1 (harvest digital 레이어) 신규: 절차 / 에러코드 / 그림
    "procedure",
    "error_code",
    "figure",
)

SOURCE_TYPES: tuple[str, ...] = ("powerpoint", "pdf", "excel", "unknown")


@dataclass
class BBox:
    """스크린샷 픽셀 좌표계의 bounding box (좌상단 원점)."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BBox":
        data = data or {}
        def _int(key: str) -> int:
            try:
                return int(data.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0
        return cls(_int("left"), _int("top"), _int("right"), _int("bottom"))


@dataclass
class Region:
    """layout/OCR 이 식별한 한 영역의 추출 결과."""

    region_id: str
    type: str = "other"
    bbox: BBox = field(default_factory=BBox)
    text: str = ""
    surrounding_context: str = ""
    confidence: float = 0.0
    model_sources: list[str] = field(default_factory=list)
    # merge 단계에서 모델 간 충돌을 발견하면 비우지 말고 여기에 기록한다.
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Table:
    """복원된 표 한 개. cells 는 행(list) 의 list(2D)."""

    region_id: str
    title: str = ""
    header: list[str] = field(default_factory=list)
    cells: list[list[str]] = field(default_factory=list)
    confidence: float = 0.0
    model_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chart:
    """복원된 차트 한 개의 요약 + 보이는 라벨."""

    region_id: str
    title: str = ""
    axis_labels: list[str] = field(default_factory=list)
    legend_labels: list[str] = field(default_factory=list)
    visible_values: list[str] = field(default_factory=list)
    trend_summary: str = ""
    confidence: float = 0.0
    model_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Formula:
    """복원된 수식 한 개 (LaTeX 우선)."""

    region_id: str
    latex: str = ""
    nearby_label: str = ""
    confidence: float = 0.0
    model_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagChunk:
    """retrieval store 에 들어갈 RAG-ready chunk (rag_db_plan.md)."""

    chunk_id: str
    collection_id: str = ""
    document_id: str = ""
    screenshot_id: str = ""
    screenshot_index: int = 0
    source_type: str = "unknown"
    source_image: str = ""
    region_id: str = ""
    region_type: str = "other"
    bbox: BBox = field(default_factory=BBox)
    parent_heading: str = ""
    content: str = ""
    raw_ocr_text: str = ""
    context_before: str = ""
    context_after: str = ""
    keywords: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    review_status: str = "needs_review"
    # 별도로 생성하는 embedding 입력 텍스트 (rag_chunks.build_embedding_text)
    embedding_text: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class ExtractionResult:
    """스크린샷 1장에 대한 최종 추출 산출물 (pipeline_plan.md Output Contract)."""

    source_image: str
    source_type: str = "unknown"
    document_id: str = ""
    collection_id: str = ""
    screenshot_id: str = ""
    screenshot_index: int = 1
    overall_confidence: float = 0.0
    summary_markdown: str = ""
    # summary_markdown 을 만든 출처 (예: ["deterministic"] 또는 ["kimi-k2.6"])
    summary_model_sources: list[str] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    formulas: list[Formula] = field(default_factory=list)
    rag_chunks: list[RagChunk] = field(default_factory=list)
    # 해소되지 않은 충돌/저신뢰 항목 (사람 review 용)
    unresolved: list[str] = field(default_factory=list)
    # stage 별 latency/모델 호출 로그 (benchmark_plan.md Latency)
    stage_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_image": self.source_image,
            "source_type": self.source_type,
            "document_id": self.document_id,
            "collection_id": self.collection_id,
            "screenshot_id": self.screenshot_id,
            "screenshot_index": self.screenshot_index,
            "overall_confidence": self.overall_confidence,
            "summary_markdown": self.summary_markdown,
            "summary_model_sources": list(self.summary_model_sources),
            "regions": [r.to_dict() for r in self.regions],
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts],
            "formulas": [f.to_dict() for f in self.formulas],
            "rag_chunks": [c.to_dict() for c in self.rag_chunks],
            "unresolved": list(self.unresolved),
            "stage_log": list(self.stage_log),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        """raw_evidence JSON(dict) -> ExtractionResult. rag_chunks 는 복원하지 않는다
        (Marp/리포팅은 구조 evidence 만 필요; chunk 는 별도 JSONL 에서 읽는다)."""
        result = cls(
            source_image=str(data.get("source_image", "")),
            source_type=str(data.get("source_type", "unknown")),
            document_id=str(data.get("document_id", "")),
            collection_id=str(data.get("collection_id", "")),
            screenshot_id=str(data.get("screenshot_id", "")),
            screenshot_index=_coerce_int(data.get("screenshot_index"), 1),
            overall_confidence=_coerce_float(data.get("overall_confidence")),
            summary_markdown=str(data.get("summary_markdown", "")),
            summary_model_sources=[str(s) for s in (data.get("summary_model_sources") or [])],
            unresolved=[str(u) for u in (data.get("unresolved") or [])],
            stage_log=list(data.get("stage_log") or []),
        )
        for r in data.get("regions") or []:
            result.regions.append(
                Region(
                    region_id=str(r.get("region_id", "")),
                    type=str(r.get("type", "other")),
                    bbox=BBox.from_dict(r.get("bbox")),
                    text=str(r.get("text", "")),
                    surrounding_context=str(r.get("surrounding_context", "")),
                    confidence=_coerce_float(r.get("confidence")),
                    model_sources=[str(s) for s in (r.get("model_sources") or [])],
                    conflicts=[str(c) for c in (r.get("conflicts") or [])],
                )
            )
        for t in data.get("tables") or []:
            result.tables.append(
                Table(
                    region_id=str(t.get("region_id", "")),
                    title=str(t.get("title", "")),
                    header=[str(h) for h in (t.get("header") or [])],
                    cells=[[str(c) for c in row] for row in (t.get("cells") or [])],
                    confidence=_coerce_float(t.get("confidence")),
                    model_sources=[str(s) for s in (t.get("model_sources") or [])],
                )
            )
        for c in data.get("charts") or []:
            result.charts.append(
                Chart(
                    region_id=str(c.get("region_id", "")),
                    title=str(c.get("title", "")),
                    axis_labels=[str(a) for a in (c.get("axis_labels") or [])],
                    legend_labels=[str(l) for l in (c.get("legend_labels") or [])],
                    visible_values=[str(v) for v in (c.get("visible_values") or [])],
                    trend_summary=str(c.get("trend_summary", "")),
                    confidence=_coerce_float(c.get("confidence")),
                    model_sources=[str(s) for s in (c.get("model_sources") or [])],
                )
            )
        for f in data.get("formulas") or []:
            result.formulas.append(
                Formula(
                    region_id=str(f.get("region_id", "")),
                    latex=str(f.get("latex", "")),
                    nearby_label=str(f.get("nearby_label", "")),
                    confidence=_coerce_float(f.get("confidence")),
                    model_sources=[str(s) for s in (f.get("model_sources") or [])],
                )
            )
        return result


__all__ = [
    "BBox",
    "Chart",
    "CHUNK_TYPES",
    "ExtractionResult",
    "Formula",
    "RagChunk",
    "Region",
    "REGION_TYPES",
    "SOURCE_TYPES",
    "Table",
]
