# 차트/그래프 중심 문서의 RAG 아키텍처 — 저장 표상 + Hybrid 검색 설계

> 질문: "문서 대부분이 차트/그래프일 때, 그 정보를 RAG 에 어떻게 저장하는 것이
> 최선인가? image-based RAG 인가, hybrid RAG 인가?"
>
> 결론 먼저: **hybrid — 차트 하나당 3+1 표상(래스터/구조 표/컨텍스트 텍스트
> [+vision embedding])을 함께 저장하고, 검색은 텍스트 arm 과 비전 arm 을 RRF 로
> 융합, 답변은 reader VLM 이 원본 이미지(crop+페이지)를 직접 읽는다.**
> 순수 image-RAG 도, 순수 text-RAG 도 차트 corpus 에서는 각각 치명적 결손이 있다.
>
> 선행 문서: [`rag_context_architecture.md`](./rag_context_architecture.md)(텍스트
> RAG 코어), [`study/pixelrag_*.md`](./study/pixelrag_characteristics.md)(비전 arm
> 후보). 이 문서는 둘을 **차트 중심 corpus** 관점에서 통합·갱신한다(2026-07-10,
> 웹 재조사 반영).

---

## 1. 왜 hybrid 인가 — 증거 요약

### 1-1. 순수 text RAG(추출 텍스트만 색인)의 천장 — 검증됨

| 근거 | 수치 | 출처 |
|---|---|---|
| 비전 검색 vs OCR+BM25 vs **고품질 VLM 캡션**+BM25 (과학 figure/차트 subset) | nDCG@5 **79.1 vs 31.6 vs 40.1** — 캡션을 아무리 잘 만들어도 ~39pt 차이 | ColPali, arXiv 2407.01449 (ViDoRe) |
| 페이지 이미지 검색 + VLM 생성 vs text-parse RAG (멀티모달 문서 e2e) | **+20~40%** | VisRAG, arXiv 2410.10594 (ICLR'25) |
| 차트 많은 재무 슬라이드에서 이미지 임베딩 vs GPT 캡션+텍스트 임베딩 직접 A/B | mAP@5 **0.523 vs 0.396**, 캡션이 이긴 케이스 0 | arXiv 2511.16654 |

즉 **차트가 지배하는 corpus 에서 "추출한 텍스트만 색인" 은 검색 recall 의
구조적 상한**이 있다. 우리가 이미 아는 parser-loss 문제의 정량판.

### 1-2. 순수 image RAG 의 결손 — 우리 요구와 충돌

- **provenance/confidence/quality-gate/Marp roundtrip 을 못 준다** — 검색 결과가
  "이미지 타일"뿐이라 rag_db_plan 의 근거추적·재사용 요구를 충족 못함
  (pixelrag_application_methods §0 결론과 동일).
- **CLIP류 통합 임베딩 함정 — 검증됨**: Chart-MRAG/CHARGE(arXiv 2502.14864)에서
  CLIP/SigLIP/JINA-CLIP 는 텍스트↔차트 교차 질의 Recall@5 **0%**. "vision arm"
  은 반드시 **문서 스크린샷으로 학습된 VDR 계열**(ColPali/ColQwen/DSE/
  Qwen3-VL-Embedding)이어야 하고, CLIP 계열 유용은 금지.
- **분포 이동 리스크(단일 출처)**: OCR 기반 RAG 가 미지 도메인/열화 문서에
  더 잘 일반화한다는 신호(arXiv 2505.05666) — 비전 arm 단독 의존 금지 사유.
- 정확 토큰(파라미터명/에러코드) 질의는 BM25 가 여전히 우위
  (rag_context_architecture 레버 #2).

### 1-3. 종합

- 차트-온리 질의: 비전 arm 이 크게 이김 / 텍스트-온리 질의: 텍스트 arm 이 이김
  → **대체가 아니라 융합**(Chart-MRAG 의 최고 구성도 "분리 저장 + 결합").
- 저장은 "무엇으로 검색하나(retrieval 표상)"와 "무엇으로 답하나(answer 표상)"를
  **분리**해야 한다 — "Index Light, Reason Deep"/DVI(arXiv 2602.14162) 방향과 일치.

---

## 2. 저장 설계 — 차트 1개당 3+1 표상

```
chart record (rag_chunks + payload)
├─ R1 래스터 (answer 표상, 색인 안 함)
│    page_NNN.webp (Stage 0) + _crops/<sid>/rNNN_chart.jpg (Stage 4, crops.json 메타)
├─ R2 구조 표 (retrieval 키 + 답변 교차검증)
│    PaddleOCR-VL 차트→표 (chart_summary / table_row chunk; 이미 구현)
├─ R3 컨텍스트 텍스트 (retrieval 키)
│    제목·축라벨·legend·trend + 슬라이드 heading = embedding_text/bm25 (이미 구현)
└─ R4 vision embedding (retrieval 키; 게이트 = 임베딩 엔드포인트 배포)
     페이지(슬라이드=1타일) 단위 벡터 -> OpenSearch dense kNN 3번째 arm
```

원칙:

1. **R1 은 진실, R2 는 근사** — 차트→표 추출은 복잡한 실전 차트에서 값 누락/환각이
   확인돼 있고(arXiv 2305.18641, 2605.27298, CharXiv 2406.18521) 오류가 다운스트림에
   전파된다. **R2 를 단독 진실로 쓰지 말 것**(검색 키 + 교차검증 역할까지만).
2. R2/R3 는 **이미 구현돼 있다**(schemas.Chart, chart_summary chunk, provenance
   bbox, embedding_text). R1 연결은 crops.json + marp/crop_map 으로 이번에 닫힘.
   **신규는 R4 하나뿐** — 기존 B1 산출물 폐기 없음.
3. 캡션-온리 인덱싱 금지(1-1 근거) / CLIP류 임베딩 금지(1-2 근거).

## 3. 검색 아키텍처 — 3-arm RRF + rerank + 이미지 reader

```
질의 (한/영, GLM-5.2 query planner 로 표현 확장 - 선택)
  │
  ├─ arm A: BM25 (R2 표 텍스트 + R3 컨텍스트)      ← 정확 토큰/코드
  ├─ arm B: dense kNN bge-m3 (R3 embedding_text)   ← 의미 검색
  └─ arm C: vision kNN (R4, 페이지 벡터)           ← layout/차트 질의 [게이트]
  │
  ▼  RRF(k=60) 융합  (arm 별 가중은 질의유형별 튜닝 필수 - Chart-MRAG 교훈:
  │   교차문서 질의에선 비전 arm 기여 0 인 케이스 존재, 고정 가중 금지)
  ▼  rerank: bge-reranker-v2-m3(텍스트) -> [게이트] Qwen3-VL-Reranker(비전)
  │   비전 reranker 가 추가되면 최대 단일 레버(ViDoRe 계열 검증)
  ▼  top-K
  ▼  reader = Kimi-K2.6: R1 crop + 저해상 전체 페이지 + R2 표(라벨 명시) 직접 읽기
답 + 인용(document_id/screenshot_index/region_id/bbox)
```

### reader 입력 규약 (차트 특화 — 검증된 실무 규칙)

- **crop + 전체 페이지를 함께** 준다: crop 단독은 맥락 상실, region 단위 입력은
  QA 정확도/토큰 모두 개선(RegionRAG arXiv 2510.27261: +10.0% R@1, 토큰 71%).
- R2 표는 **"기계 추출본, 오류 가능 — 이미지로 검증하라"** 라벨을 붙여 보조로만.
  reader 는 텍스트가 있으면 이미지보다 텍스트를 믿는 **text-over-visual bias** 가
  확인돼 있어(Chart-MRAG), 틀린 표가 옳은 차트를 덮어쓸 수 있다.
- 캡션/표만 먹인 reader 는 수치 충실도에서 이미지 reader 에 대패
  (0.20 vs 0.80, arXiv 2511.16654) — **표-온리 입력 금지**.
- reader 는 frontier 급 유지(Kimi-K2.6). 4B 미만 비전 모델은 타일 읽기에서
  큰 폭 하락(PixelRAG 논문 -12.5pt).

## 4. 비전 임베딩 모델 선택 (R4 게이트의 해소 방법)

| 옵션 | 품질 | 우리 스택 적합성 |
|---|---|---|
| **Qwen3-VL-Embedding(single-vector) + Qwen3-VL-Reranker** (arXiv 2601.04720) | single-vector + reranker 로 late-interaction 과의 격차 대부분 회수(Nemotron ColEmbed V2 실험: 48.7→54.4 vs LI 55.5) | **최적** — 벡터 1개/페이지라 OpenSearch dense kNN 필드에 그대로 들어감. 권장 |
| ColQwen3 계열 (late-interaction MaxSim) | 리더보드 최상(ViDoRe V3 61~63 nDCG@10, single-vector 대비 +2~6pt) | OpenSearch 는 multi-vector 필드가 **아직 RFC 단계**(k-NN #2706) — MaxSim rescore 스크립트나 사이드카 서비스 필요. 수백~수천 페이지면 brute-force ~1ms/1천页 로 가능은 함 |
| Qwen3-VL-Embedding-2B + PixelRAG LoRA | 웹 스크린샷 특화 | 한국어 사내 슬라이드 도메인 갭 미검증 — A/B 후보로만 |

권장: **Phase 2 에서 Qwen3-VL-Embedding 자체 배포**(`deploy_vlms/` + `vlm_serve`
엔트리 1개, air-gap 반입은 가중치만). late-interaction 은 +2~6pt 를 위해 별도
인프라를 얹을 가치가 벤치로 증명될 때만.

## 5. 단계별 로드맵 (기존 P0~P6 갱신)

| 단계 | 작업 | 게이트 | 어디서 |
|---|---|---|---|
| **Phase 1 (지금, 배포 0)** | 3-표상 저장 완성: chart chunk 에 `crop_path` provenance 연결(crops.json), OpenSearch 색인(BM25+bge-m3+sparse, RRF), bge-reranker, reader 에 R1 이미지 첨부(DVI) — **코드 골격 구현됨(2026-07-10, 아래 §5-1)** | 없음 | 색인/검색 사내 |
| Phase 1.5 | golden 질의셋에 **차트-온리 질의**(값/추세를 차트에서만 읽을 수 있는 것) 별도 계층 추가 + Recall@k / parser-loss recovery 측정 — **벤치 골격 구현됨(2026-07-10, benchmark/retrieval_*)**; GT 작성만 사내 잔여 | GT 작성 | 사내 |
| **Phase 2 (배포 1개, 최대 ROI)** | Qwen3-VL-Embedding(+Reranker) 배포 → R4 arm C 추가, 3-arm RRF, 질의유형별 가중 튜닝 | 가중치 반입 | 사내 |
| Phase 3 (선택) | ColQwen3 late-interaction A/B(사이드카 MaxSim), 부족 시 LoRA 도메인 적응 | Phase 2 결과 | 사내 GPU |

Phase 1 만으로도 "차트를 이미지로 답하는" DVI 경로는 열린다(검색 recall 천장은
남지만, R3 컨텍스트가 캡션 역할로 완화). Phase 2 가 recall 천장을 걷어낸다.

### 5-1. Phase 1 코드 골격 (구현됨, 2026-07-10)

```
extraction/
├─ schemas.py         RagChunk.crop_path 필드 신설 (R1 래스터 provenance)
├─ crop.py            map_charts_to_crop_paths (CropMeta -> cNNN 순서 대응)
├─ rag_chunks.py      generate_chunks(chart_crop_lookup=...) -> chart_summary 에 crop_path
├─ embeddings.py      bge-m3 dense 클라이언트 (OpenAI-호환 /embeddings) + offline 결정론 stub
├─ opensearch_index.py  매핑/문서변환/bulk(순수) + REST 클라이언트(transport 주입) + 색인 엔트리
├─ hybrid_search.py   BM25/kNN 쿼리 빌더 + rrf_fuse(k=60, arm 가중) + rerank 훅(passthrough)
│                     + build_reader_payload (DVI: 관련최고=꼬리, crop 우선 이미지, 기계추출 라벨)
└─ test_opensearch_smoke.py  9 테스트 (fake transport 로 색인->검색 e2e, 서버 불필요)

benchmark/  (Phase 1.5 검색 벤치)
├─ retrieval_golden.py     golden 질의 스키마(tier: chart_only/table/text/mixed)
│                          + relevance 매처(chunk_id 정밀 | screenshot_id 페이지 수준)
├─ retrieval_metrics.py    Recall@k / MRR tier 집계 + parser-loss recovery(baseline=bm25)
├─ run_retrieval_benchmark.py  3-arm(bm25/dense/hybrid) 채점 + [DIGEST]/digest.txt
├─ golden_retrieval_queries.example.json  사내 GT 작성 템플릿(합성 5질의)
└─ test_retrieval_benchmark_smoke.py      7 테스트 (stub searcher e2e)
```

- env: `DOC_EXTRACT_OPENSEARCH_URL/INDEX/USER/PASSWORD`,
  `DOC_EXTRACT_EMBED_API_URL/MODEL/API_KEY`(미설정=offline stub),
  `DOC_EXTRACT_RERANK_API_URL`(미설정=passthrough).
- 사내 잔여 배선: 실제 bge-m3 embeddings 엔드포인트 URL 확인, bge-reranker 훅
  구현(`hybrid_search.rerank_hits`), neural sparse 파이프라인(`sparse_features`
  필드는 예약됨), arm 가중 질의유형별 튜닝(Phase 1.5 벤치).

## 6. 리스크 / 캐비엇

- 수치들은 대부분 영문 문서/차트 벤치 — **한국어 사내 슬라이드로 절대치 이식
  금지**, 방향 신호로만(기존 rag_context_architecture §7 과 동일 원칙).
- 비전 arm 의 교차문서(멀티홉) 질의 기여는 확인 안 됨(Chart-MRAG 에서 0 사례) —
  RRF 가중을 질의유형별로 두고, 실측 없이 비전 arm 을 키우지 말 것.
- "RRF(텍스트 arm, 비전 arm)" 자체의 정량 ablation 은 공개 문헌이 얇다
  (실무 표준이지만 수치는 미검증) — Phase 1.5 벤치가 우리 수치를 만든다.
- DRM 캡처 화면(안티앨리어싱/장식 배경)에 대한 임베딩 robustness 미검증 —
  9장 미니 벤치 먼저.

## 7. 출처 (핵심만)

- ColPali/ViDoRe: arXiv 2407.01449 · VisRAG: 2410.10594 · 캡션 vs 이미지 A/B: 2511.16654
- Chart-MRAG/CHARGE(CLIP 0%, 분리저장 우위, text-bias): 2502.14864
- RegionRAG(crop 입력): 2510.27261 · DVI: 2602.14162 · 차트→표 오류 전파: 2305.18641, 2605.27298, CharXiv 2406.18521
- Qwen3-VL-Embedding/Reranker: 2601.04720 · Nemotron ColEmbed V2(저장/정확도 트레이드오프): 2602.03992
- OpenSearch multi-vector 상태: k-NN issue #2706 (RFC) + late-interaction rescore 블로그
- 사내 선행: rag_context_architecture.md, study/pixelrag_*.md, rag_db_plan.md
