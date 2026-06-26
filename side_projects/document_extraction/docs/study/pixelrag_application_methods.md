# PixelRAG 적용 방법 — document_extraction (사내 DRM 스크린샷 RAG)

> PixelRAG 특성은 [`pixelrag_characteristics.md`](./pixelrag_characteristics.md).
> 이 문서는 **우리 프로젝트 + 사내 환경**에 어떻게 가져올지의 구체적 방법·로드맵·게이트.

## 0. 결론 먼저

- **무엇을**: 기존 text-extraction(Stage 1~8 → `rag_chunks.jsonl` → keyword search)은
  **그대로 두고**, **pixel 검색 arm 을 하나 추가**해 둘을 **RRF 로 융합**한다(hybrid).
- **왜 hybrid 인가**: PixelRAG 는 retrieval **recall** 을 올리지만 provenance/
  confidence/structured field/Marp roundtrip 을 안 준다. 우리는 그게 필요하다 →
  대체가 아니라 **2nd arm**.
- **1차 게이트**: 사내에 **vision embedding 엔드포인트**가 있어야 한다(현재 없음, §3).

## 1. 왜 우리 프로젝트에 특히 잘 맞나 (적합성 근거)

| PixelRAG 가 요구하는 것 | 우리가 이미 가진 것 |
|---|---|
| 페이지를 스크린샷으로 render | **Stage 0 가 이미 `page_NNN.webp` 생산**(PPT 슬라이드쇼/Excel·Word PDF/PDF 직접). 추가 캡처 0 |
| 표/차트/layout 보존이 가치 있는 corpus | 사내 발표자료·recipe 문서 = layout-heavy. text-chunk 가 가장 손해 보는 유형 |
| FAISS 인덱스 + GPU | 사내 GPU 클러스터 + FAISS 사용 이력(`test/video_frame_parser`) |
| reader 는 4B+ frontier VLM | **Kimi-K2.6**(multimodal, 사내 게이트웨이 확인됨)이 reader 로 적합 |
| 소규모 corpus 이면 인제스트 비용 무시 가능 | 벤치 9장 ~ 수백 페이지 규모. 800CPU/5TB 우려는 Wikipedia-scale 얘기, 우리완 무관 |

또한 PixelRAG 의 "타일이 곧 화면 영역" 특성은 우리 `rag_db_plan.md` 의
**provenance(source_image/bbox) 추적 목표와 자연스럽게 일치** — 검색 결과가 곧
원본 스크린샷 조각이라 "근거 픽셀까지 trace" 가 공짜다.

## 2. 목표 아키텍처 — Hybrid 2-arm retriever

```
                 page_NNN.webp  (Stage 0, 공용 입력)
                        │
        ┌───────────────┴────────────────┐
        ▼ (기존)                          ▼ (신규: PixelRAG arm)
  Stage 1~8 VLM 추출                 Tile → vision embed → FAISS
  → rag_chunks.jsonl                 → pixel_index/
  → keyword_search (retrieval.py)    → tile_search
        │                                  │
        └──────────────► RRF 융합 ◄────────┘
                        │  (workflow_3 의 ensemble/consensus 와 동일한 RRF)
                        ▼
              top-K (text chunk + pixel tile 혼합)
                        │
                        ▼  reader = Kimi-K2.6 (멀티모달)
        답 + 인용(document_id/screenshot_index/region_id/source_image/bbox)
```

설계 원칙(우리 프로젝트의 기존 원칙과 정합):

- **text arm 은 권위 있는 structured evidence/provenance/confidence 를 책임**(quality
  gate, review_status, Marp roundtrip 은 전부 text arm 산출물에서 나옴).
- **pixel arm 은 recall 을 책임**(parser-loss·layout-heavy 케이스 회수).
- **CV/수치가 최종 결정, 모델은 영역 식별만** 이라는 workflow_3 철학과 동일하게,
  **융합 점수(RRF)가 랭킹을 결정**하고 reader VLM 은 "읽기"만 한다.

## 3. 1차 게이트 — vision embedding 엔드포인트 (반드시 먼저 해결)

현재 사내 VLM 4종(`paddleocr-vl-1.5`, `ui-venus`, `mai-ui`, `kimi-k2.6`)은 전부
**generative** 모델이다. PixelRAG 는 **embedding** 모델이 필요하다. 선택지:

| 옵션 | 내용 | 장단 |
|---|---|---|
| **A. Qwen3-VL-Embedding-2B 자체 배포** | `deploy_vlms/` 에 vLLM/transformers 로 임베딩 서비스 추가, Flask proxy 에 `vlm_serve` 엔트리 1개 등록 | PixelRAG 원본과 동일. 단 가중치(+ LoRA adapter)를 **air-gapped 망으로 오프라인 반입** 필요 |
| **B. ColQwen / ColPali 계열 배포** | late-interaction(MaxSim) embedding | 소규모 corpus 라 인덱스 무게 감당 가능, localization 정밀. 단 검색 코드가 single-vector 보다 복잡 |
| **C. 사내 게이트웨이 embedding 유무 확인** | company LLM gateway 에 이미 multimodal embedding 이 있는지 먼저 조회 | 있으면 가장 빠름. 없을 가능성 큼 |

권장: **C 로 확인 → 없으면 A**(원본과 동일, `flask_api/vlm_serve` 패턴 재사용). corpus 가
작아 정밀도가 더 필요하면 B(ColQwen) 를 벤치에서 A/B.

> ⚠️ **반입 제약**: 가중치/adapter 다운로드는 사외에서, 반입은 회사 정책 따라
> 오프라인 전송. 코드/모델은 반입 OK, **데이터(스크린샷)는 반출 금지** 원칙 유지.

## 4. 재사용 가능한 자산 vs 신규로 만들 것

| 우리가 이미 가진 것 (재사용) | PixelRAG 용으로 신규 |
|---|---|
| Stage 0 캡처(`extract.py`, `*_handler.py`) = Render | **tile 슬라이서**(고정 높이 타일; 슬라이드는 1타일, 긴 PDF/넓은 Excel 은 분할) |
| `util/screen_capture.save_webp_capped`(1MB 캡, 품질 사다리) | **embedding 클라이언트**(신규 service slug 호출) |
| `extraction/schemas.py`(provenance/bbox/screenshot_index) | **`pixel_index.py`**(타일 임베딩 → FAISS build/load) |
| `extraction/retrieval.py`(keyword + quality gate) | **`tile_search.py`**(FAISS top-N) + **RRF 융합기** |
| `benchmark/`(채점 하네스, metrics) | benchmark 에 **pixel-RAG / hybrid 파이프라인 2종 추가**(text-RAG 와 A/B) |
| `marp/`(roundtrip) | 변경 없음 — 여전히 text arm evidence 기반 |

신규 모듈 배치 제안(기존 레이아웃과 정합):

```
extraction/
├─ pixel_index.py     # 타일 슬라이스 + vision embed + FAISS build/load  (신규)
├─ tile_search.py     # FAISS top-N 타일 검색                           (신규)
├─ fuse.py            # RRF: keyword_search 결과 ⊕ tile_search 결과       (신규)
└─ test_pixel_smoke.py# 오프라인 스모크(임베딩 stub, FAISS 라운드트립)    (신규)
```

## 5. 타일링 정책 (우리 데이터에 맞춤)

PixelRAG 는 "fixed-width render → fixed-height tile". 우리는 소스 타입별로 자연 단위가 다름:

- **PPT**: 슬라이드 1장 = `page_NNN.webp` = **타일 1개**(이미 검색 단위로 충분). 추가 분할 불필요.
- **PDF**: 페이지가 길거나 multi-column → 페이지를 **세로 2~3 타일**로 분할(읽기 순서 보존),
  타일 경계 **overlap 10~15%** 로 표/문단 잘림 방지.
- **Excel**: 인쇄 분할 페이지가 이미 있음(`ExportAsFixedFormat`). 넓은 시트는 **가로 타일**도 고려.

타일 메타에 기존 `screenshot_index` + `tile_index` + `bbox(타일의 원본 내 위치)`를 달아
**검색 결과 → 원본 페이지 좌표** 역추적을 보장(= provenance 일치).

## 6. 단계별 로드맵 (집/사외에서 가능한 것 vs 사내 필요)

| 단계 | 작업 | 어디서 | 게이트 |
|---|---|---|---|
| **P0** | 사내 게이트웨이에 embedding 엔드포인트 유무 확인(§3-C) | 사내 | — |
| **P1** | `pixel_index.py`/`tile_search.py`/`fuse.py` **골격 + 임베딩 stub** + 스모크(FAISS 라운드트립). 모델 없이 검증 | 사외 OK | 없음(stub) |
| **P2** | 임베딩 모델 배포(§3-A/B) — `deploy_vlms/` + `vlm_serve` 엔트리 1개 | 사내 | 가중치 오프라인 반입 |
| **P3** | 실제 타일 임베딩 → FAISS 인덱스 빌드(벤치 9장) | 사내 | P2 |
| **P4** | **benchmark A/B**: text-RAG vs pixel-RAG vs hybrid — Retrieval recall + parser-loss 회수율 측정 | 채점은 사외 OK | P3 |
| **P5** | RRF 가중치 튜닝 + reader(Kimi-K2.6) 인용 경로 | 사내 | P4 |
| **P6** | (옵션) 한국어 문서에서 base embedding 성능 부족 시 LoRA fine-tune 검토 | 사내 GPU | P4 결과 |

P1 은 지금 사외에서 바로 가능(우리 프로젝트의 "OFFLINE stub 으로 골격 검증" 패턴 그대로).

## 7. 평가 방법 (기존 benchmark 하네스 확장)

`benchmark/run_benchmark.py` 는 이미 "4개 파이프라인 비교 → comparison_matrix" 구조다.
여기에 **pixel-RAG**, **hybrid** 파이프라인을 얹어 같은 9장 golden set 으로 A/B:

- **추가 지표**: *Retrieval Recall@k*(질문의 정답이 top-k 타일/청크에 들어오나),
  *Parser-loss recovery*(text arm 이 놓친 표/차트 질문을 pixel arm 이 회수한 비율 —
  PixelRAG 논문의 핵심 우위 지표).
- **기존 지표 유지**: Hallucination Rate, RAG Readiness, Latency, Table/Chart Accuracy.
- 가설: **layout-heavy 슬라이드/차트 질문에서 hybrid > text-only**, **순수 텍스트
  질문에선 거의 동률**(PixelRAG 한계 §5 와 일치) → 그래서 **대체가 아닌 융합**이 정답.

## 8. 리스크 / 주의

| 리스크 | 영향 | 완화 |
|---|---|---|
| **embedding 엔드포인트 부재** | 적용 자체 블록 | P0 먼저. 없으면 P2 배포(원본과 동일 경로) |
| **한국어 문서 도메인 갭** | LoRA 가 영문 Wikipedia screenshot 학습 → 한글 layout/글꼴 전이 불확실 | P4 에서 한국어 golden 으로 먼저 측정, 부족 시 base embedding 또는 P6 fine-tune |
| **reader 4B 미만 금지** | 작은 VLM 이면 정확도 12.5점↓ | reader 는 Kimi-K2.6(frontier) 고정 |
| **provenance/confidence 미제공** | 단독 사용 시 우리 품질 게이트 무력화 | text arm 을 권위 소스로 유지(hybrid 의 핵심 이유) |
| **인덱스 크기** | single-vector 라 작음. ColQwen(late-interaction) 택하면 커짐 | 소규모 corpus 라 둘 다 감당. 벤치로 결정 |
| **반입/보안** | 데이터 반출 금지 원칙 | 모델만 반입, 인덱스/스크린샷은 사내 잔류. `research_notes.md` 안전 경계 준수 |

## 9. 다음 액션 (구체)

1. **P0**: 사내 게이트웨이에 multimodal embedding 서비스가 있는지 확인(없으면 Qwen3-VL-Embedding-2B 배포 계획 수립).
2. **P1**: `extraction/pixel_index.py` + `tile_search.py` + `fuse.py` 골격 + 임베딩 stub + 스모크 테스트(사외에서 지금 가능).
3. **P4 준비**: benchmark 9장 golden set 에 "표/차트에서만 답이 보이는" 질문 몇 개를 의도적으로 추가(parser-loss 회수 측정용).

## 출처

- PixelRAG GitHub: <https://github.com/StarTrail-org/PixelRAG>
- PixelRAG 해설(워크플로/한계): <https://www.theaiautomators.com/pixelrag-visual-rag-without-text-parsing/>
- VisRAG (ICLR 2025, single-vector 비교): <https://arxiv.org/abs/2410.10594>
- 프로젝트 내부 근거: [`../rag_db_plan.md`](../rag_db_plan.md), [`../pipeline_overview.md`](../pipeline_overview.md), [`../research_notes.md`](../research_notes.md)
