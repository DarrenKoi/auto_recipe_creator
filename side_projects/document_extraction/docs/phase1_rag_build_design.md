# Phase 1 — Harvest 번들에서 RAG 빌드 (design spec)

> 작성: 2026-06-26. Phase 0(`harvest/harvest_pdf.py`)이 떠둔 **디지털 harvest 번들**을
> 입력으로, 2000쪽 tool manual 을 검색 가능한 RAG 로 만드는 빌드 설계.
> 배경 연구: [`study/pixelrag_application_methods.md`](./study/pixelrag_application_methods.md),
> 기존 파이프라인: [`pipeline_overview.md`](./pipeline_overview.md), 저장 계약:
> [`rag_db_plan.md`](./rag_db_plan.md).

## 0. 한 문장

DRM 전에 떠둔 **정확한 digital 레이어**(text+좌표 / native table / figure / render PNG)를
질의유형별 chunker 로 잘라 **text arm + pixel arm 2-arm hybrid retriever**(RRF 융합)로
검색하고, Kimi-K2.6 reader 가 인용과 함께 답한다.

## 1. 무엇이 바뀌나 (기존 파이프라인 대비)

기존 `extraction/`(Stage 1~8)은 **screenshot-first**다: DRM 때문에 화면 캡처밖에 없어
VLM(OCR/layout)이 모든 페이지를 다시 읽는다. Phase 0 harvest 덕분에 이제 **digital-first**로
뒤집는다:

| | 기존(screenshot-first) | Phase 1(digital-first) |
|---|---|---|
| 텍스트 출처 | VLM OCR (lossy, 2000쪽 × 호출) | **harvest text dict (정확·무료)** |
| 표 | VLM 재인식 | **harvest native table cells (정확)** |
| 그림 | render 안의 픽셀 | **harvest figure 원본 바이트 + render** |
| 구조/heading | layout VLM 추정 | **harvest TOC + font-size (권위)** |
| VLM 역할 | 전 페이지 추출 | **수술적**(figure caption, 모호 표 reconcile 만) |

핵심 원칙(기존과 동일): **정확한 데이터가 권위, VLM 은 식별/설명만.** "보이는 것만,
값은 창작 안 함." harvest 가 비어 있는(텍스트 0) 페이지만 기존 screenshot VLM 경로로
폴백한다(post-DRM 잔여 페이지 처리 경로와 동일).

## 2. 입력 계약 — harvest 번들

`harvest_pdf.py` 가 만든 `<OUTPUT>/<stem>/` 를 그대로 읽는다(반출 금지 자산):

```
<stem>/
├─ manifest.json          # per_page 레코드 + summary(빌드가 신뢰하는 인덱스)
├─ metadata.json, toc.json
├─ text/page_NNNN.json    # get_text("dict") block/line/span + bbox + font/size
│  page_NNNN.txt          # reading-order 평문
├─ tables/page_NNNN.json  # [{source, bbox, rows, row_count, col_count}, ...]
├─ figures/page_NNNN.json # [{xref, file, bboxes_on_page, w, h}, ...]
│  figures/by_xref/*.png  # 원본 바이트(dedup)
├─ render/page_NNNN.png   # 무손실 페이지 렌더
└─ links/page_NNNN.json
```

빌드는 이 번들만 의존한다 → **사외(Mac)에서도 chunk 단계까지 전부 검증 가능**(VLM/embedding
없이). embedding/index/reader 만 사내 게이트.

## 3. 아키텍처 (component DAG)

```
harvest 번들
   │  C1 harvest_loader   (번들 → PageModel; 순수)
   ▼
PageModel[]  (blocks+bbox+font, tables, figures, render path, links)
   │  C2 structure        (TOC + font-size → 섹션/heading 계층; 순수)
   ▼
구조화 PageModel (각 block 에 section_path + parent_heading)
   │  C3 chunkers         (질의유형별; 순수)            ┌─ C4 enrich_vlm (수술적, 게이트)
   ▼                                                    │   figure caption / 모호 표 reconcile
RagChunk[]  (procedure/table_row/error_code/figure/region_text/…)
   │
   ├─► rag_chunks.jsonl  (retrieval store; 기존 schema 재사용)
   │
   ├─ C5a text embed  → text_index/ (FAISS)
   └─ C5b tile/figure embed → pixel_index/ (FAISS)   ← vision-embed 게이트
        │
        ▼  C6 fuse (keyword ⊕ text-vec ⊕ pixel-tile, RRF)
      top-K (혼합)
        │  C7 reader = Kimi-K2.6 (멀티모달)
        ▼
     답 + 인용(page / section_path / region_id / source_image / bbox)
        │  C8 benchmark (golden Q&A, 4유형, A/B)
```

각 컴포넌트는 **단일 책임 + 명확한 인터페이스**로 독립 테스트 가능하게 만든다.

## 4. 컴포넌트 명세

### C1 `extraction/harvest_loader.py` — 번들 → PageModel (순수)
`manifest.json` 을 권위 인덱스로 페이지별 산출물을 `PageModel` dataclass 로 로드.
`PageModel`: `page_no, size, blocks[{text,bbox,font,size,flags}], tables[], figures[],
render_path, links[], has_text`. 손상/누락 파일은 조용히 비우지 않고 `load_warnings` 에 기록.
스크린샷 폴백 대상(=`has_text=False`)을 표시.

### C2 `extraction/structure.py` — 섹션/heading 복원 (순수)
- **TOC(authoritative)**: `toc.json` 의 `[level, title, page]` → 페이지→섹션경로 매핑
  (`Ch 3 > 3.2 Alignment > 3.2.1 Procedure`).
- **In-page heading**: block font-size 분포를 클러스터링해 body 대비 큰 span 을 heading 후보로.
- 각 block 에 `section_path`(TOC 기반) + `parent_heading`(bbox 기준 nearest 위쪽 heading) 부여.
- heading 연결은 `rag_db_plan.md` 의 context 보존 규칙을 그대로 만족(검색·인용 핵심).

### C3 `extraction/chunkers.py` — 질의유형별 chunker (순수)
4개 질의유형 → chunk 매핑. 모두 기존 `RagChunk` schema(§5)를 채운다.

| 질의유형 | chunker | chunk type | 규칙 |
|---|---|---|---|
| 절차/how-to | `procedure_chunker` | `procedure`(신규) | "Step N"/"1. 2. 3." 순번 시퀀스를 **한 chunk 로 묶음**(중간 분할 금지), 같은 페이지 figure 참조 포함 |
| 파라미터/스펙 | `table_chunker` | `table_row` / `table_summary` | native table → 행 단위, header+단위+section 맥락 유지(**flatten 금지**), 표 전체 요약 1개 |
| 에러/알람 코드 | `error_code_chunker` | `error_code`(신규) | header 에 code/error/alarm/alid 키워드인 표 → `코드→의미→조치` 단위 chunk |
| 개념/다이어그램 | `figure_chunker` | `figure`(신규) | figure 1개=1 chunk, caption(근접 block)+section 맥락+`source_image`(원본 바이트 경로) |
| (일반 prose) | `region_chunker` | `region_text` | 문단 → heading+section 맥락 |

`table_row` 는 기존 `rag_chunks.py` 에 이미 있다(재사용·확장). `table_chunker` 는 같은 표를
두 출처(pymupdf/pdfplumber)로 받았을 때 **행수·열수 일치하면 한쪽 채택, 불일치하면 둘 다
보존 + `conflicts` 표시**(merge.py 의 무손실 병합 원칙 동일). 불일치 표는 C4 reconcile 후보.

### C4 `extraction/enrich_vlm.py` — 수술적 VLM (게이트, 선택)
**전 페이지 OCR 아님.** 두 경우만:
1. **figure caption**: 근접 caption 이 없는 figure → Kimi-K2.6 멀티모달이 figure crop 에
   1줄 설명 생성(개념/다이어그램 검색용 `embedding_text` 보강). 출처를 `model_sources` 에 정직 기록.
2. **모호 표 reconcile**: C3 가 `conflicts` 표시한 표만 표 crop 을 VLM 에 보내 셀 정합.

호출 수는 figure 수 + 모호 표 수로 **상한**(2000쪽 전수 아님). `DOC_EXTRACT_OFFLINE=1` 이면
stub(결정론적) → 사외 골격 검증. 기존 `poc.workflow_3.vlm.vlm_client` 재사용.

### C5 embedding + index (사내 게이트)
- **C5a text arm** `extraction/text_index.py`: chunk `embedding_text`(heading+section+content+
  context) → text embedding → FAISS `text_index/`.
- **C5b pixel arm** `extraction/pixel_index.py`: render PNG 타일 + figure crop → vision
  embedding → FAISS `pixel_index/`. 타일 정책은 `pixelrag_application_methods.md §5`(PDF 세로
  2~3 타일, overlap 10~15%). 타일 메타에 `page_no + tile_index + bbox` → provenance 역추적.
- **1차 게이트**: 사내 multimodal embedding 엔드포인트 유무(`pixelrag_application §3`). 없으면
  Qwen3-VL-Embedding-2B 를 `deploy_vlms/`+`vlm_serve` 로 배포(오프라인 반입). text embedding 도
  사내 엔드포인트 확인 필요.

### C6 `extraction/fuse.py` — hybrid 검색 + RRF
`keyword_search`(기존 `retrieval.py`) ⊕ text-vector ⊕ pixel-tile → **RRF 융합**(workflow_3
ensemble 의 RRF 동일 개념) → top-K(text chunk + figure/tile 혼합). metadata 필터(section/
source_type/review_status/confidence/chunk_type) 유지. text arm 이 provenance/confidence/quality
gate 의 권위, pixel arm 이 recall(parser-loss 회수) 담당.

### C7 `extraction/reader.py` — 답변 + 인용
top-K(텍스트 + 이미지)를 Kimi-K2.6 가 읽어 답 생성. **인용 필수**: `page / section_path /
region_id / source_image / bbox`. reader 는 4B+ frontier 고정(PixelRAG 한계 §5: 작은 모델 금지).

### C8 benchmark 확장
기존 `benchmark/` 에 manual 골든 Q&A(4유형 의도적 포함, 특히 "표/그림에서만 답이 보이는"
parser-loss 케이스) 추가. 지표: **Recall@k**, **parser-loss recovery**(text arm miss → pixel arm
회수율), **table/param exactness**, **error-code lookup accuracy**, 기존 Hallucination/Latency.
A/B: text-only vs hybrid. 가설: layout/표/그림 질의에서 hybrid > text-only, 순수 prose 는 동률.

## 5. schema 확장

`extraction/schemas.py` `CHUNK_TYPES` 에 **`procedure`, `error_code`, `figure`** 추가.
기존 `RagChunk` 필드 그대로 사용하되 의미 매핑:
- `source_image` = figure chunk 는 `figures/by_xref/*.png`, 그 외는 `render/page_NNNN.png`.
- `parent_heading` = C2 가 부여한 nearest heading. `bbox` = 원본 페이지 좌표(타일/표/그림 위치).
- `screenshot_index` → **`page_no`** 로 사용(문서가 1개라 page = 검색 단위 순서).
- 신규 메타(옵션): `section_path`, `tile_index`. `RagChunk` 에 `section_path` 필드 추가 검토.

## 6. 모듈 레이아웃 (기존 정합)

```
extraction/
├─ harvest_loader.py   (C1, 신규)   test_harvest_loader_smoke.py
├─ structure.py        (C2, 신규)   test_structure_smoke.py
├─ chunkers.py         (C3, 신규)   test_chunkers_smoke.py
├─ enrich_vlm.py       (C4, 신규)
├─ text_index.py       (C5a, 신규)
├─ pixel_index.py      (C5b, 신규)  ┐ pixelrag_application §4 와 동일 배치
├─ tile_search.py      (검색)       │
├─ fuse.py             (C6, 신규)   ┘ test_pixel_smoke.py(embed stub + FAISS 라운드트립)
├─ reader.py           (C7, 신규)
├─ schemas.py          (CHUNK_TYPES 확장)
├─ rag_chunks.py       (table_row/heading 재사용·확장)
└─ retrieval.py        (keyword + quality gate 재사용)
benchmark/             (C8: manual 골든 + parser-loss 지표 추가)
```

## 7. 로드맵 (사외 가능 vs 사내 게이트)

| 단계 | 작업 | 어디서 | 게이트 |
|---|---|---|---|
| **B0** | harvest 번들 확보·검증(Phase 0) | 사내 | DRM 전 |
| **B1** | C1 loader + C2 structure + C3 chunkers + 스모크(VLM/embed 없이) | **사외 OK** | 없음 |
| **B2** | 사내 text/vision embedding 엔드포인트 확인/배포 | 사내 | 오프라인 반입 |
| **B3** | C5 실 임베딩 → text_index + pixel_index 빌드 | 사내 | B2 |
| **B4** | C6 hybrid + RRF | 사내 | B3 |
| **B5** | C7 reader + 인용 | 사내 | B4 |
| **B6** | C8 골든 Q&A + A/B 튜닝(채점은 사외 OK) | 사내/사외 | B3 |

B1 은 지금 사외에서 바로 가능(harvest 번들 샘플 또는 합성 stub 으로). C4(수술적 VLM)는
`DOC_EXTRACT_OFFLINE=1` stub 으로 골격까지 사외 검증.

## 8. 리스크 / 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| embedding 엔드포인트 부재 | C5+ 블록 | B2 먼저(`pixelrag_application §3`); text arm 만 우선 가동 가능 |
| 한국어 문서 도메인 갭 | pixel embed 전이 불확실 | B6 한국어 골든 먼저 측정, 부족 시 base/LoRA |
| 표 merge cell / 회전 표 | param 정확도 | C3 두 출처 보존 + C4 reconcile, 그래도 모호하면 `unresolved` |
| 절차 경계 오검출 | procedure chunk 분절 | 순번 패턴 + heading 경계 휴리스틱, 실패 시 region_text 폴백 |
| 인덱스 크기(2000쪽) | single-vector 라 작음(FAISS OK) | 타일 수만 모니터, ColQwen 은 필요 시만 |
| 반출/보안 | 데이터 반출 금지 | 모델만 반입, 번들/인덱스/figure 는 사내 잔류 |

## 9. 비목표 (YAGNI)

- 2000쪽 전수 VLM OCR(디지털 레이어가 있으니 불필요).
- MongoDB/OpenSearch 등 벡터 DB(첫 빌드는 JSONL + FAISS, `rag_db_plan` 원칙 그대로).
- Marp roundtrip 변경(text arm evidence 기반, 그대로 둠).
- 멀티문서 collection(현재는 manual 1종; `document_id` 만 유지해 확장 여지).

## 10. 수용 기준 (Phase 1 완료)

1. B1 스모크 4종(loader/structure/chunkers/pixel-stub) 사외 통과.
2. 4 질의유형 각각에 대해 골든 질문에서 hybrid Recall@k 가 text-only 이상.
3. 모든 답이 page/section/region 인용을 동반(인용 없는 답 0).
4. param/error-code 질의에서 표 값이 flatten/창작 없이 정확 일치.
