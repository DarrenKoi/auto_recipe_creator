"""추출 파이프라인 출력 계약(schema).

pipeline_plan.md 의 Output Contract 와 rag_db_plan.md 의 Chunk Schema 를
dataclass 로 옮긴 것. 첫 구현은 dict 로 시작해도 되지만, schema 가 흔들리지
않도록 타입을 고정해 둔다. 모든 dataclass 는 `to_dict()` 로 JSON 직렬화 가능한
순수 dict 를 반환한다(파이프라인 산출물을 JSON/JSONL 로 떨구기 위함).
"""

from dataclasses import asdict, dataclass, field


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

# RAG chunk type (rag_db_plan.md "Chunk Types").
CHUNK_TYPES: tuple[str, ...] = (
    "document_summary",
    "region_text",
    "table_summary",
    "table_row",
    "chart_summary",
    "formula",
    "unresolved",
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
