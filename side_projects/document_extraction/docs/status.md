# 문서 추출 side project — 현황 (계획 / 완료 / 필요 테스트)

> 빠른 확인용 현황표. 상세 절차는 [`runbook.md`](./runbook.md), 구현 개요는
> [`pipeline_overview.md`](./pipeline_overview.md). 갱신: 2026-06-26.

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

## 3. 이번에 한 일 (Marp Stage 6/7, 2026-06-26)

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
| 1 | VLM 4서비스 ready 확인 + 실호출 1콜 스모크 | 검증 | 사내 |
| 2 | 9장 캡처(PPT3/PDF3/Excel3) + ground-truth 작성 | 데이터 | 사내 |
| 3 | 4 파이프라인 벤치 채점 -> comparison_matrix | 측정 | 채점 어디서나 |
| 4 | Marp Stage 7 실행 -> **SSIM floor 0.90 보정** | 캘리브레이션 | 사내(캡처 필요) |
| 5 | crop 분기 임계(Stage 3) 튜닝 + `ENABLE_CROP_REFINE` on | 캘리브레이션 | 사내 |
| 6 | Kimi Platform 엔드포인트 base64 1콜 스모크 | 검증 | 사내 |
| 7 | Marp 커스텀 테마 1종(시각 보정) | 코드 | 어디서나 |
| 8 | crop_path <-> chart region 좌표 자동 대응 | 코드 | 어디서나 |

---

## 5. 필요한 테스트

### 5-1. 이미 있는 스모크 (서버/marp 불필요, 회귀 가드)
| 테스트 | 개수 | 범위 |
|---|---|---|
| `extraction/test_extraction_smoke` | — | Stage 1~8 골격 |
| `extraction/test_retrieval_smoke` | — | 검색 + quality gate |
| `benchmark/test_benchmark_smoke` | — | 채점 하네스 |
| `marp/test_marp_smoke` | 8 | Stage 5 생성 |
| `marp/test_render_smoke` | 6 | Stage 6 인자 빌더 + graceful |
| `marp/test_verify_smoke` | 11 | Stage 7 SSIM/강등 결정 |

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
