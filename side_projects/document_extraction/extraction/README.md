# extraction — 스크린샷 문서 추출 파이프라인 (skeleton)

캡처 단계(`document_extraction/extract.py` → page WebP)가 만든 스크린샷
이미지를 입력으로 받아, VLM tiered pipeline 으로 텍스트/표/차트/수식을 추출하고
RAG-ready chunk 까지 만든다.

설계 근거: [`../docs/pipeline_plan.md`](../docs/pipeline_plan.md),
[`../docs/rag_db_plan.md`](../docs/rag_db_plan.md).

> **현 상태 = 뼈대(skeleton).** 실제 VLM 호출은 사내 PC + Flask proxy 가 있어야
> 의미가 있다. 모델 서버가 없으면 OFFLINE(dry-run) 경로로 stub evidence 를
> 만들어 파이프라인 골격·schema·merge·chunk 로직을 서버 없이 검증한다.

## 모듈 구성

| 파일 | 역할 | 서버 필요? |
|---|---|---|
| `schemas.py` | 출력 계약 dataclass (ExtractionResult/Region/Table/Chart/Formula/RagChunk) | ✗ |
| `prompts.py` | 스테이지별 `(system, user)` 프롬프트 빌더 | ✗ |
| `models.py` | `StageRunner` — service slug 별 VLM 호출 + offline 폴백 | 호출 시 |
| `merge.py` | Stage 5 evidence merge (순수 함수, 충돌 표시) | ✗ |
| `rag_chunks.py` | Stage 8 chunk 생성 + embedding_text + JSONL writer | ✗ |
| `extract_screenshot.py` | Stage 1~8 오케스트레이터 (폴더 단위) | 호출 시 |
| `test_extraction_smoke.py` | OFFLINE e2e + merge + chunk + keyword 검색 스모크 | ✗ |

## 스테이지 ↔ 모델 (poc/workflow_3/vlm)

| Stage | 모델 | service slug |
|---|---|---|
| 2 OCR/parse | PaddleOCR-VL-1.5 | `paddleocr-vl-1.5` |
| 3 layout | UI-Venus-1.5-8B | `ui-venus` |
| 4 crop refine | MAI-UI-8B | `mai-ui` |
| 6 synthesis | Kimi-K2.6 | `kimi-k2.6` |

> docs 의 `poc.work2` / `kimi-k2.5` 표기는 stale. 현 production VLM 클라이언트는
> `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient` 이고 Kimi slug 는 `kimi-k2.6`.

## 실행

CLI 인자 없음. `extract_screenshot.py` 상단 상수를 직접 수정한다.

```bash
# 스모크 테스트 (서버 불필요)
uv run python -m side_projects.document_extraction.extraction.test_extraction_smoke

# 폴더 추출 (상단 상수 INPUT_IMAGE_DIR/OUTPUT_DIR 채운 뒤)
uv run python -m side_projects.document_extraction.extraction.extract_screenshot

# 서버 없이 강제 dry-run
DOC_EXTRACT_OFFLINE=1 uv run python -m side_projects.document_extraction.extraction.extract_screenshot
```

출력:
- `<OUTPUT_DIR>/rag_chunks.jsonl` — retrieval store (chunk JSONL)
- `<OUTPUT_DIR>/raw_evidence/<screenshot_id>.json` — raw evidence (debug/reprocess)

## 아직 안 된 것 (office 캘리브레이션 후)

- **Stage 4 crop 재병합**: 현재 crop 후보 region 식별만(훅). bbox crop 저장 →
  `run_crop_refine` → ocr 재병합은 미구현 (`extract_screenshot._apply_crop_refine` TODO).
- **table_row 단위 chunk**, 좌표 기반 nearest-heading 매칭.
- **실제 VLM 호출 검증**: 사내 PC 에서 paddleocr/ui-venus/kimi 1콜 스모크
  (benchmark_plan.md "다음 단계").
- **벤치마크 채점 하네스** (benchmark_plan.md) — 별도.
- **Marp roundtrip** (marp_roundtrip_design.md) — 별도 트랙.
