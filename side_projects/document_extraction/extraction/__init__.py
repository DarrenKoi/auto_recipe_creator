"""스크린샷 문서 추출 파이프라인 (skeleton).

`document_extraction/` 의 캡처 단계(ppt/excel/word/pdf -> page WebP)가 만든
스크린샷 이미지를 입력으로 받아, VLM tiered pipeline 으로 텍스트/표/차트/수식을
추출하고 RAG-ready chunk 까지 생성하는 단계(Stage 1~8)를 담는다.

설계 근거 문서:
    side_projects/document_extraction/docs/pipeline_plan.md
    side_projects/document_extraction/docs/rag_db_plan.md

현 시점은 *뼈대(skeleton)* 다. 실제 VLM 호출은 사내 PC + Flask proxy 가 있어야
의미가 있으므로, 모델 서버가 없을 때는 OFFLINE(dry-run) 경로로 stub evidence 를
생성해 파이프라인 골격과 schema/merge/chunk 로직을 서버 없이 검증할 수 있게 한다.

VLM 클라이언트는 production 컨벤션을 따라 poc.workflow_3.vlm 을 재사용한다
(docs 의 `poc.work2` 표기는 stale: poc/workflow_3 가 현 production 경로).
"""

from side_projects.document_extraction.extraction.schemas import (
    BBox,
    Chart,
    ExtractionResult,
    Formula,
    RagChunk,
    Region,
    Table,
)

__all__ = [
    "BBox",
    "Chart",
    "ExtractionResult",
    "Formula",
    "RagChunk",
    "Region",
    "Table",
]
