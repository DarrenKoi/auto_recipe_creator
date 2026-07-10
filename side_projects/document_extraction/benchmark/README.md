# benchmark — 추출 벤치마크 채점 하네스

`benchmark_plan.md` 의 메트릭을 순수 Python 으로 구현한다. 모델이 만든 추출
산출물(JSON)과 사람이 만든 ground-truth(JSON)를 대조해 채점한다.

> **채점은 어디서나(집에서) 돈다.** 실제 스크린샷 캡처 + VLM 추출은 사내 PC 에서
> 돌려 `extractions/` 에 떨궈 두고, 채점/집계/acceptance 판정은 모델 없이 실행.

## 모듈

| 파일 | 역할 |
|---|---|
| `ground_truth.py` | `GroundTruth` 스키마 + 로더 + `write_template` |
| `metrics.py` | 메트릭 순수 함수 (아래) |
| `scorer.py` | 스크린샷 채점 + 비교 매트릭스 + acceptance 판정 |
| `run_benchmark.py` | 폴더 단위 실행 엔트리 (CLI 인자 없음) |
| `test_benchmark_smoke.py` | 합성 GT/추출로 메트릭 검증 (서버 불필요) |

### Phase 1.5 검색 벤치 (2026-07-10 신규 — 추출 벤치와 별개 축)

| 파일 | 역할 |
|---|---|
| `retrieval_golden.py` | golden 질의 스키마(tier: **chart_only**/table/text/mixed) + 로더/검증 + 예시 writer. relevance = chunk_id 정밀 또는 screenshot_id(+region_type) 페이지 수준 |
| `retrieval_metrics.py` | Recall@k / MRR tier 집계 + **parser-loss recovery**(baseline=bm25 가 놓친 chart_only 를 후보 arm 이 회수한 비율) |
| `run_retrieval_benchmark.py` | 3-arm(bm25/dense/hybrid) 실행 엔트리. `retrieval_scores.json` + `retrieval_matrix.md` + `digest.txt` + 콘솔 `[DIGEST]` 한 줄 |
| `golden_retrieval_queries.example.json` | 사내 GT 작성 시작 템플릿(합성 예시 5질의). **실데이터 golden 은 커밋 금지** |
| `test_retrieval_benchmark_smoke.py` | 7 테스트: 스키마/매처/메트릭 수학/recovery/stub-searcher e2e |

사내 실행 순서: 색인(`extraction/opensearch_index.py`) → golden 작성(예시 복사,
chart_only 비중 있게) → `run_retrieval_benchmark.py` (GOLDEN_JSON 상수 지정) →
`digest.txt` 한 줄로 결과 전달. Phase 2 에서 vision arm 이 추가되면 searchers
dict 에 arm 하나만 더 얹으면 같은 하네스로 A/B 된다.

## 메트릭 (benchmark_plan.md)

| 메트릭 | 범위 | 방식 |
|---|---|---|
| Text Recall | 0..1 | GT important_texts/title 이 추출 텍스트에 나타나는 비율(정규화 substring + 토큰 0.7 overlap) |
| Table Accuracy | 0..1 | header 겹침으로 표 매칭 → header/cell 정확도 평균 − column 차이 penalty |
| Chart Understanding | 0..1 | title/axis/legend/visible_value/trend 복원 비율 (보이는 것만) |
| Layout Accuracy | 0..1 | 기대 region type(multiset) 교집합 비율 (첫 벤치는 approximate box 허용 → type 존재만) |
| Hallucination Rate | 0..1 (낮을수록 좋음) | 추출이 만든 숫자 중 GT 보이는 근거에 없는 것의 비율 (단어 경계 숫자만) |
| RAG Readiness | 0..1 | chunk 가 provenance(source_image/index/bbox/type) + context + 저신뢰 표시를 갖는 비율 |
| Latency | 보고용 | `stage_log` 에서 service 별 latency 집계 |

## 벤치 폴더 레이아웃

```
<BENCH_DIR>/
├─ ground_truth/<screenshot_id>.json          # 사람이 작성 (write_template 로 초안)
├─ extractions/<pipeline_name>/<screenshot_id>.json   # 추출 산출물(extraction/ raw_evidence)
└─ results/                                    # 실행이 생성
   ├─ scores.json
   ├─ comparison_matrix.md                     # Pipeline x 메트릭 표
   └─ acceptance.json                          # acceptance criteria 판정
```

`pipeline_name` 예: `ocr_only`, `ocr_uivenus`, `ocr_uivenus_crops`, `full`
(benchmark_plan.md 의 4개 파이프라인 비교).

## 실행

```bash
# 스모크 테스트 (서버 불필요)
uv run python -m side_projects.document_extraction.benchmark.test_benchmark_smoke

# 벤치 채점 (run_benchmark.py 상단 BENCH_DIR 채운 뒤)
uv run python -m side_projects.document_extraction.benchmark.run_benchmark
```

## Acceptance Criteria (benchmark_plan.md)

- PPT/PDF text recall ≥ 0.7
- Excel table accuracy ≥ 0.7
- RAG readiness ≥ 0.7

벤치셋에 없는 source 카테고리는 `null`(n/a) 로 표시되어 `all_passed` 판정에서 제외된다.

## 집에서 못 하는 것 (사내 PC)

- 실제 9장 스크린샷 캡처(Windows COM) + paddleocr/ui-venus/mai-ui/kimi 추출.
  → `extractions/<pipeline>/` 에 산출물을 떨궈 두면 채점은 집에서 가능.
- ground-truth 작성은 실제 스크린샷을 보고 사람이 해야 함(`write_template` 로 초안만).
