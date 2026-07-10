# 문서 추출 side project — 현황 (계획 / 완료 / 필요 테스트)

> 빠른 확인용 현황표. 상세 절차는 [`runbook.md`](./runbook.md), 구현 개요는
> [`pipeline_overview.md`](./pipeline_overview.md). 갱신: 2026-07-10.

---

## 1. 한눈에

```
Stage 0 캡처 ──▶ Stage 1~8 추출 ──▶ ┌ 검색(search)
 (Windows)        (VLM/offline)      ├ 벤치(benchmark)
                                     ├ Marp Stage 5(deck 생성)
                                     ├ Marp Stage 6(렌더)        ← 신규
                                     └ Marp Stage 7(SSIM/강등)    ← 신규
```
- **코드 골격**: Stage 0~8 + 검색 + 벤치 + Marp 5/6/7 모두 구현 완료.
- **남은 건 거의 다 "오피스 실데이터 검증"** — 코드가 아니라 정확도/충실도 측정·보정.

---

## 2. 단계별 현황

| 단계 | 상태 | 비고 |
|---|---|---|
| Stage 0 캡처 | ✅ 구현 | Windows COM + PyMuPDF. **사내에서만 실행 가능** |
| Stage 1 전처리 | ✅ 구현 | 크기/소스타입 추정 |
| Stage 2 OCR (`paddleocr-vl-1.5`) | ✅ 구현 / ⏳ 실호출 검증 | 서버 ready 필요 |
| Stage 3 layout (`ui-venus`) | ✅ 구현 / ⏳ 실호출 검증 | |
| Stage 4 crop refine (`mai-ui`) | ✅ 구현 / ⏳ 캘리브레이션 | `ENABLE_CROP_REFINE` 기본 off |
| Stage 5 merge | ✅ 구현 | 순수, 충돌 보존 |
| Stage 6 synthesis | ✅ 구현 | `deterministic`/`kimi`/`none` |
| Stage 7 review packet | ✅ 구현 | raw_evidence JSON |
| Stage 8 RAG chunk | ✅ 구현 | rag_chunks.jsonl |
| 검색 search/retrieval | ✅ 구현 | quality gate + keyword |
| 벤치 benchmark | ✅ 구현 / ⏳ ground-truth | 채점은 어디서나, GT 작성은 사내 |
| **Marp Stage 5 생성** | ✅ 구현 | evidence -> deck.md |
| **Marp Stage 6 렌더** | ✅ 구현 (이번 작업) | marp-cli, npx 폴백, graceful |
| **Marp Stage 7 SSIM/강등** | ✅ 구현 (이번 작업) | floor 0.90, 전체 래스터 안전망 |

범례: ✅ 구현 완료 · ⏳ 오피스 실데이터/캘리브레이션 대기

---

## 3. 이번에 한 일 (차트 RAG Phase 1 골격, 2026-07-10 오후)

- **3-표상 저장 완성(R1 고리)**: `RagChunk.crop_path` 신설, `crop.map_charts_to_crop_paths`
  (CropMeta -> cNNN 순서 대응), `generate_chunks(chart_crop_lookup=...)` 로
  chart_summary chunk 에 crop provenance 연결(extract_one 자동 배선).
- **OpenSearch 색인기**: `extraction/opensearch_index.py` — 매핑(BM25 + knn_vector
  1024 + sparse/R4 예약)/문서변환/bulk 순수 빌더 + REST 클라이언트(transport 주입)
  + `index_chunks_jsonl` 엔트리. `extraction/embeddings.py` — bge-m3 클라이언트
  (offline 결정론 stub).
- **hybrid 검색**: `extraction/hybrid_search.py` — 2-arm(BM25/kNN) + 클라이언트측
  RRF(k=60, arm 가중 열림) + rerank 훅(passthrough) + DVI reader payload(관련최고=
  꼬리, crop 우선 이미지 첨부, 기계추출 라벨).
- 스모크: `test_opensearch_smoke` 9 테스트(fake transport 색인->검색 e2e) 추가,
  전 스위트 10종 통과. 상세: [rag_chart_heavy_architecture.md §5-1](./rag_chart_heavy_architecture.md).

## 3-1. 이전 (GLM-5.2 + DRM 폴백 + Marp 강화, 2026-07-10)

- **GLM-5.2 로컬 API 통합**: `flask_vlm.py` 에 `glm-5.2`(direct) 등록,
  `Workflow1VLMClient.chat_text()`(텍스트 전용) 신설. Stage 6 합성 폴백 체인
  kimi(비전) -> glm(텍스트) -> offline stub + `SYNTHESIS_MODE="glm"` 직접 모드.
  `summary_model_sources` 가 실사용 서비스를 기록.
- **DRM 캡처 폴백**: `util/viewer_capture.py`(뷰어 열기 + 키 전송 + frame-diff
  종료 판정). PDF(열기 실패/needs_pass), Word(export 실패) 자동 폴백, Excel 은
  경고만. 판정 로직은 Mac 스모크로 검증(4 테스트).
- **Marp 강화(계획 7·8번 해소)**: `themes/doc_restore.css` 커스텀 테마 +
  `--theme` 렌더 인자, `crop_map.py`(crops.json/파일명 -> cNNN 차트 자동 대응;
  extract_screenshot 이 crops.json 저장), `refine.py`(LLM 구조 다듬기 + 표/수식/
  이미지/숫자 verbatim 검증, 기각 시 원본 유지). verify 강등 경로 테마 유지.
- **차트 중심 RAG 설계 문서**: [`rag_chart_heavy_architecture.md`](./rag_chart_heavy_architecture.md)
  — 3+1 표상 저장 + 3-arm hybrid(RRF) + 이미지 reader(DVI), 웹 재조사 근거 포함.
- 스모크: extraction 10, marp 5종(marp 9/render 7/verify 11/crop_map 6/refine 9),
  viewer_capture 4 — 전부 통과(Mac, 서버 불필요).

### 이전 (Marp Stage 6/7, 2026-06-26)

- `marp/render.py` — `build_render_args`(순수) + `resolve_marp_command` + `render_deck`
  (marp 부재 시 graceful).
- `marp/verify.py` — `ssim`(numpy, skimage 비의존) + `slide_fidelity` + `flag_low_fidelity`
  + `plan_downgrade` + `whole_slide_marp` + `apply_downgrade_plans` + `verify_and_downgrade`.
- `__init__.py`/`build_marp.py`/`marp_roundtrip_design.md` 갱신.
- **Mac e2e 확인**: 합성 2-슬라이드 deck -> PNG 렌더 -> 자기비교 SSIM=[1.0,1.0], 강등 없음.
- 커밋 `db7391e` (main push 완료).

---

## 4. 계획 (남은 일 — 대부분 오피스)

| 우선 | 일 | 종류 | 어디서 |
|---|---|---|---|
| 1 | VLM 서비스 ready 확인 + 실호출 1콜 스모크 (**glm-5.2 chat_text 포함**) | 검증 | 사내 |
| 2 | 9장 캡처(PPT3/PDF3/Excel3) + ground-truth 작성 | 데이터 | 사내 |
| 3 | 4 파이프라인 벤치 채점 -> comparison_matrix | 측정 | 채점 어디서나 |
| 4 | Marp Stage 7 실행 -> **SSIM floor 0.90 보정** (+ doc-restore 테마 A/B) | 캘리브레이션 | 사내(캡처 필요) |
| 5 | crop 분기 임계(Stage 3) 튜닝 + `ENABLE_CROP_REFINE` on | 캘리브레이션 | 사내 |
| 6 | Kimi Platform 엔드포인트 base64 1콜 스모크 | 검증 | 사내 |
| 7 | ~~Marp 커스텀 테마 1종~~ ✅ `themes/doc_restore.css` (2026-07-10) | 코드 | 완료 |
| 8 | ~~crop_path <-> chart region 자동 대응~~ ✅ `crop_map.py` (2026-07-10) | 코드 | 완료 |
| 9 | **DRM 폴백 실검증**: DRM PDF/Word 1개씩 뷰어 캡처 + 키 시퀀스 보정 | 검증 | 사내 |
| 10 | Marp refine 실호출 A/B(glm vs kimi, 채택률/기각 사유 확인) | 검증 | 사내 |
| 11 | ~~차트 RAG Phase 1 골격(3-표상 + OpenSearch 색인)~~ ✅ (2026-07-10) | 코드 | 완료 |
| 12 | Phase 1 사내 배선: bge-m3 embeddings URL + bge-reranker 훅 + 실색인 1회 | 검증 | 사내 |
| 13 | 차트 RAG Phase 2: Qwen3-VL-Embedding 배포(R4 vision arm) | 배포 | [설계](./rag_chart_heavy_architecture.md) |

---

## 5. 필요한 테스트

### 5-1. 이미 있는 스모크 (서버/marp 불필요, 회귀 가드)
| 테스트 | 개수 | 범위 |
|---|---|---|
| `extraction/test_extraction_smoke` | 10 | Stage 1~8 골격 + 합성 폴백 체인 + crops.json |
| `extraction/test_retrieval_smoke` | — | 검색 + quality gate |
| `extraction/test_opensearch_smoke` | 9 | 색인기/RRF/hybrid/reader payload (fake transport) |
| `benchmark/test_benchmark_smoke` | — | 채점 하네스 |
| `marp/test_marp_smoke` | 9 | Stage 5 생성 + 테마 프론트매터 |
| `marp/test_render_smoke` | 7 | Stage 6 인자 빌더 + --theme + graceful |
| `marp/test_verify_smoke` | 11 | Stage 7 SSIM/강등 결정 |
| `marp/test_crop_map_smoke` | 6 | crop <-> chart 자동 대응 |
| `marp/test_refine_smoke` | 9 | LLM 다듬기 검증/채택·기각 |
| `util/test_viewer_capture_smoke` | 4 | DRM 폴백 frame-diff 판정 |

→ 전부 `uv run python -m ...` 로 사외에서도 회귀 확인 가능(런북 §7).

### 5-2. 아직 없는/필요한 테스트
| 필요 테스트 | 왜 | 막는 요소 |
|---|---|---|
| 실 VLM 1콜 정확도(서비스별) | OFFLINE stub 가 아닌 진짜 출력 품질 | 사내 서버 ready |
| crop refine 전/후 dense recall A/B | Stage 4 켤지 결정 | 사내 캡처 + 서버 |
| 벤치 ground-truth 대조 채점 | 4 파이프라인 정확도 수치 | 사람 GT 작성 |
| **Marp Stage 7 실캡처 SSIM 분포** | floor 0.90 보정 + 강등 빈도 | 사내 캡처 + marp/Chromium |
| marp-cli 실렌더 통합(Chromium) | 오피스 렌더 경로 자체 | Chromium 설치(런북 §1-3) |
| Kimi base64 이미지 1콜 | Stage 6 kimi 합성 경로 | Platform 엔드포인트 |

> 핵심: **코드 단위 테스트는 충분**(스모크 25+). 부족한 건 전부 **실데이터 정확도/충실도
> 측정** — 순수 로직이 아니라 모델·렌더 출력 품질이라 사내 실행으로만 닫힌다.
