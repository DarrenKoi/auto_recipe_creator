# 스크린샷 문서 추출 파이프라인 개요

> 이 문서는 `side_projects/document_extraction/` 에 **실제로 구현된** 파이프라인을
> end-to-end 로 설명한다. 설계 의도는 [`pipeline_plan.md`](./pipeline_plan.md),
> [`rag_db_plan.md`](./rag_db_plan.md), [`benchmark_plan.md`](./benchmark_plan.md),
> [`marp_roundtrip_design.md`](./marp_roundtrip_design.md) 에 있고, 여기서는 그것들이
> 코드로 어떻게 이어지는지 한눈에 보여준다.

## 1. 목적

DRM 보호 때문에 원본 PPT/PDF/Excel 파일을 직접 파싱할 수 없을 때, **화면에 보이는
스크린샷에서만** 정보를 복원해 RAG(retrieval-augmented generation)에 바로 쓸 수 있는
database 를 만든다. 단순 요약이 아니라 content + context + provenance(출처) +
confidence 를 함께 보존해, 나중에 답변이 원본 스크린샷 근거까지 추적되도록 한다.

핵심 원칙: **보이는 것만 추출하고, 값을 창작하지 않는다.** 읽을 수 없으면 비워 둔다.

## 2. 전체 데이터 흐름

```
[원본 문서]                         [사내 PC, Windows]
 PPT/PDF/Excel
     │  Stage 0  캡처 (document_extraction/extract.py + *_handler.py)
     ▼
 page_NNN.webp  (페이지별 WebP, 1MB 캡)
     │
     │  Stage 1~8  추출 (extraction/extract_screenshot.py)
     ▼
 ┌─ raw_evidence/<screenshot_id>.json   (ExtractionResult; debug/reprocess)
 └─ rag_chunks.jsonl                     (retrieval store; RAG chunk)
     │
     ├─ retrieval/search   (extraction/search.py)        keyword + quality gate 검색
     ├─ benchmark          (benchmark/run_benchmark.py)   ground-truth 대비 채점
     └─ marp               (marp/build_marp.py)           evidence -> Marp 슬라이드 복원
```

- **Stage 0 캡처**는 Windows COM(PowerPoint/Excel/Word 슬라이드쇼·인쇄) + PyMuPDF 로
  페이지를 WebP 로 떨군다. 사내 PC 전용. **DRM 폴백**: DRM/암호화 PDF(fitz 열기
  실패/needs_pass)와 export 차단된 Word 는 `util/viewer_capture.py` 로 기본 뷰어를
  띄워 페이지별 화면 캡처(frame-diff 로 마지막 페이지 판정) — PPT 슬라이드쇼
  캡처와 동일 원리라 DRM 안전.
- **Stage 1~8 추출**부터는 모두 순수 로직 + VLM 호출이라, VLM 서버가 없으면 OFFLINE
  폴백으로 골격이 그대로 돈다(아래 §6).

## 3. 출력 계약 (schema)

추출 산출물은 `extraction/schemas.py` 의 dataclass 로 고정돼 있다.

| 타입 | 의미 |
|---|---|
| `ExtractionResult` | 스크린샷 1장의 최종 산출물(regions/tables/charts/formulas/rag_chunks/summary) |
| `Region` | 한 영역(title/body/table/chart/formula/footer/legend/other) + bbox + text + 충돌 |
| `Table` | header + cells(2D) |
| `Chart` | title/axis/legend/visible_values/trend |
| `Formula` | LaTeX + nearby_label |
| `RagChunk` | retrieval store 한 줄: content + provenance(source_image/bbox/index) + confidence + review_status |

모든 dataclass 는 `to_dict()`(JSON 직렬화) / `from_dict()`(라운드트립 복원)를 제공한다.

## 4. 단계별 설명

### Stage 1 — Preprocess
이미지 크기(width/height)를 재고 source type 을 추정한다. 캡처가 이미 WebP 라 재인코딩은
생략한다. (`extract_screenshot._image_size`, `_infer_source_type`)

### Stage 2 — OCR / Document Parsing  (`paddleocr-vl-1.5`)
보이는 텍스트, reading order, 표/차트/수식 후보를 뽑는다. `StageRunner.run_ocr`.

### Stage 3 — Layout / Region Detection  (`ui-venus`)
스크린샷 종류(ppt/pdf/excel)와 영역(bbox + type)을 식별한다. `StageRunner.run_layout`.
layout 이 source_type 을 모르면 폴더 힌트로 보강한다.

### Stage 4 — Crop Refinement  (`mai-ui` / `paddleocr-vl-1.5`)  — `crop.py`
dense 영역(table/chart/formula/legend)을 bbox 로 잘라(margin 추가 + frame clamp) 재인식한다.
- `crop.compute_crop_box` / `crop.crop_region`: 자르기는 순수 CV(집에서 검증).
- `_merge_crop_refine`: 재인식 결과를 ocr evidence 에 **손실 없이** 병합(별개 표/차트는
  보존, 정확히 같은 header/legend 만 중복 제거).

### Stage 5 — Evidence Merge  — `merge.py`
OCR + layout 을 `ExtractionResult` 로 합친다. 텍스트는 OCR 우선, region type/bbox 는 layout
우선. 모델 출력이 충돌하면 조용히 고르지 않고 `region.conflicts` / `unresolved` 로 표시한다.
순수 함수.

### Stage 6 — Synthesis  — `synthesis.py`
요약(summary_markdown) + overall_confidence + unresolved 를 만든다. 모드 선택:
- **`deterministic`(기본)**: 모델 0콜. evidence 만으로 제목/본문/표/차트/수식을 Markdown 으로
  조립. 값 창작 위험 없음.
- **`kimi`**: `kimi-k2.6` 비전 합성(이미지+evidence). **실패 시 `glm-5.2` 텍스트
  폴백 -> 그래도 실패해야 offline stub**(폴백 체인, models.run_synthesis).
- **`glm`**: `glm-5.2` 텍스트 전용 합성(evidence-only, 이미지 미전송) — 사내
  로컬 API 의 텍스트 LLM 을 직접 지정.
- **`none`**: 합성 생략.

요약 출처는 `summary_model_sources`(예: `["deterministic"]` / `["kimi-k2.6"]` /
`["glm-5.2"]` / stub 이면 `["offline"]`)로 실제 사용 서비스를 정직하게 기록한다.

### Stage 7 — Human Review Packet
`raw_evidence/<screenshot_id>.json` 으로 전체 evidence(요약/충돌/저신뢰 포함)를 저장해
사람이 검토·수정할 수 있게 한다. `rag_chunks.write_raw_evidence`.

### Stage 8 — RAG Chunk Generation  — `rag_chunks.py`
evidence 를 retrieval-ready chunk 로 변환한다. chunk type:
`region_text` / `table_summary` / `table_row`(행 단위, header 맥락 보존, 상한 있음) /
`chart_summary` / `formula` / `document_summary`. 각 chunk 는 `embedding_text`(heading +
region type + context 포함)를 별도로 갖는다. heading 은 bbox 기반 nearest-title 로 붙인다.
결과는 `rag_chunks.jsonl` 에 append.

## 5. 후속 소비 단계

### 검색 — `retrieval.py` / `search.py`
embedding 없이 동작하는 첫 단계 검색.
- `quality_gate`: rag_db_plan Quality Gates(content 비어있지 않음 / source_image 존재 /
  region_type 알려짐 / confidence≥0.7 또는 approved / 표·차트 chunk 의 label 보유)로
  `trusted` vs `lower_trust` 분류.
- `keyword_search`: query 토큰 AND 매칭 + metadata 필터(source_type/document/region_type/
  review_status/min_confidence) + tier 필터, heading 매칭 가중 랭킹.

### 벤치마크 — `benchmark/`
모델이 만든 추출 JSON 을 사람이 만든 ground-truth JSON 과 대조해 채점한다(순수 Python).
- 메트릭: Text Recall / Table Accuracy / Chart Understanding / Layout Accuracy /
  Hallucination Rate / RAG Readiness / Latency.
- `run_benchmark.py`: `extractions/<pipeline>/` 의 4개 파이프라인을 비교해 `scores.json`,
  `comparison_matrix.md`, `acceptance.json` 을 쓴다. 추출은 사내, 채점은 어디서나.

### Marp 복원 — `marp/`
evidence 를 Marp(Markdown 슬라이드)로 되돌린다(marp_roundtrip_design Stage 5). 텍스트류
(제목/본문/표/수식)는 Marp 네이티브로, 래스터류(차트)는 원본 crop 이미지 재삽입(crop_lookup)
또는 데이터 표 대체 + 화자 노트로 보존. 표 셀의 `|`·줄바꿈은 escape 하고, row 가 header 보다
길면 colN 으로 보강해 데이터를 잘라내지 않는다. `build_marp.py` 가 `raw_evidence/*.json` ->
`deck.md` 를 만든다. 추가 기능(전부 선택, 기본 off):
- **crop 자동 대응**(`crop_map.py`): Stage 4 가 남긴 `_crops/<sid>/crops.json`
  (없으면 파일명 스캔)에서 chart crop 을 찾아 evidence 의 cNNN 차트에 순서 대응 —
  CROP_LOOKUPS 수동 작성 불필요(status 계획 8번 해소).
- **커스텀 테마**(`themes/doc_restore.css`, `THEME="doc-restore"`): 한국어 글꼴
  스택 고정 + 사내 슬라이드 룩 근사(시각 보정, status 계획 7번). 렌더 시
  `render_deck(..., theme_css=DOC_RESTORE_THEME_CSS)`.
- **LLM 구조 다듬기**(`refine.py`, `REFINE_SERVICE="glm-5.2"|"kimi-k2.6"`):
  슬라이드 단위로 헤딩/불릿 구조만 다듬고, 표 행·수식·이미지 참조 verbatim 보존
  + 새 숫자 금지 검증을 통과한 슬라이드만 채택(실패 시 원본 유지).

## 6. OFFLINE(dry-run) 폴백

`StageRunner` 는 VLM 서버에 못 붙으면(사외 dev PC) 결정론적 stub evidence 를 돌려준다.
- 강제: 환경변수 `DOC_EXTRACT_OFFLINE=1`.
- 자동: 실제 호출 실패 시 해당 alarm 만 offline 으로 강등.

덕분에 Stage 1~8 골격 + merge/crop/synthesis/chunk/검색/벤치/Marp 로직 전부를 모델 없이
집에서 스모크 테스트로 검증할 수 있다.

## 7. 실행 (CLI 인자 없음 — 모듈 상단 상수/.env 수정)

```bash
# 0) 캡처 (사내 Windows)
uv run python side_projects/document_extraction/extract.py

# 1) 추출 (Stage 1~8). 서버 없으면 DOC_EXTRACT_OFFLINE=1
uv run python -m side_projects.document_extraction.extraction.extract_screenshot

# 2) 검색
uv run python -m side_projects.document_extraction.extraction.search

# 3) 벤치 채점
uv run python -m side_projects.document_extraction.benchmark.run_benchmark

# 4) Marp deck 생성
uv run python -m side_projects.document_extraction.marp.build_marp

# 스모크 테스트 (서버 불필요)
uv run python -m side_projects.document_extraction.extraction.test_extraction_smoke
uv run python -m side_projects.document_extraction.extraction.test_retrieval_smoke
uv run python -m side_projects.document_extraction.benchmark.test_benchmark_smoke
uv run python -m side_projects.document_extraction.marp.test_marp_smoke
```

## 8. 모델 역할 (poc/workflow_3/vlm 재사용)

| 단계 | 모델 | service slug |
|---|---|---|
| Stage 2 OCR/parse | PaddleOCR-VL-1.5 | `paddleocr-vl-1.5` |
| Stage 3 layout | UI-Venus-1.5-8B | `ui-venus` |
| Stage 4 crop refine | MAI-UI-8B | `mai-ui` |
| Stage 6 synthesis | Kimi-K2.6 (비전) | `kimi-k2.6` |
| Stage 6 폴백/대체 + Marp refine | GLM-5.2 (텍스트 전용) | `glm-5.2` |

> 사내 게이트웨이에서 Kimi-K2.5/Qwen3-VL 은 deprecated → `kimi-k2.6` 으로 통일됨.
> GLM-5.2 는 같은 게이트웨이의 텍스트 LLM(direct 모드) — 이미지 없이
> `Workflow1VLMClient.chat_text()` 로 호출한다. VLM 클라이언트는
> `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient` 를 그대로 쓴다.

## 9. 패키지 레이아웃

```
side_projects/document_extraction/
├─ extract.py, *_handler.py, util/   # Stage 0 캡처 (Windows)
├─ extraction/                       # Stage 1~8 + 검색
│  ├─ schemas.py  prompts.py  models.py  merge.py
│  ├─ crop.py     synthesis.py  rag_chunks.py
│  ├─ extract_screenshot.py          # 오케스트레이터
│  ├─ retrieval.py  search.py        # RAG 검색 + quality gate
│  └─ test_*_smoke.py
├─ benchmark/                        # 채점 하네스
│  ├─ ground_truth.py  metrics.py  scorer.py  run_benchmark.py
│  └─ test_benchmark_smoke.py
├─ marp/                             # evidence -> Marp 복원
│  ├─ generate.py  build_marp.py  test_marp_smoke.py
└─ docs/                            # 설계 + 본 개요 문서
```

## 10. 집에서 완료 vs 사내 PC 필요

- **집에서(모델 없이) 완료·검증**: Stage 1~8 골격, merge, crop 기하, 결정론적 합성,
  chunk(+table_row/heading), 검색+quality gate, 벤치 채점 하네스, Marp 생성. 스모크 4종 통과.
- **사내 PC 필요(잔여)**: 실제 VLM 1콜 검증 + crop 재인식 튜닝, 9장 캡처(Windows COM) +
  ground-truth 작성(그 뒤 채점은 집에서), Marp Stage 6/7(marp-cli 렌더 + SSIM 검증/자동 강등),
  crop_path ↔ chart region 좌표 자동 대응.
