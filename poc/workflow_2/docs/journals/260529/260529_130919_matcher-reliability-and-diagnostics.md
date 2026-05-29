# 260529 — Align point correction 진단 도구 + matcher 신뢰도 작업

> 세션: 2026-05-29
> 주제: 100+ 테스트셋 디버깅 인프라 구축 → 실데이터 진단 → matcher 신뢰도 Slice 1·2 + 코드 리뷰
> 관련: [recovery plan](260529_align-point-correction-recovery-plan.md), [golden dataset plan](../../align_success_dataset_plan.md)

---

## 1. 진행 사항

이번 세션은 "왜 align point(green mark)가 엉뚱한 위치에 찍히나"를 **계측 → 실데이터 진단 → matcher 보강** 순으로 다뤘다.

1. **review/triage 인프라** — 100+ recipe 를 폴더별로 못 보는 문제 해결.
   - `align_review.py`: 최신 batch 결과를 **status 축**으로 재배치 (`by_status/` thumbnail + 단일 `index.html`, worst-first, lazy-load). `align_point_correction.run()` 끝에서 자동 호출.
   - `align_digest.py`: 스크린샷 대체용 **텍스트 집계** (status counts, rcp-box 검출률, best_scale 분포, suspect/not_distinctive 목록). 오피스에서 텍스트로 회신받는 1차 채널.
2. **crosshair 검출 v2** (`crosshair_detect.py`) — 구판 top-hat(국소 대비)이 흰 배경에서 죽는 문제.
   - 절대 saturation + 방향성 형태학 line extraction + 다중 임계 ladder + center-gap close. `probe()` 가 old vs v2 montage + S/E 분리 카운트 덤프.
3. **유사도/참조 진단** (`align_similarity.py`) — rcp-center(align point) ↔ msr(S/E) 유사도.
   - 세 위치(at_crosshair / at_center / free_best) × metric(matcher/MI/NCC) 분리도(bACC).
   - **상대 staleness**: rcp-vs-S-consensus 를 S-vs-S 일관성과 비교(MI). `stale_replace` / `S_inconsistent`(판단 불가) / `insufficient_S` 분류 → rcp 재등록 후보 리스트.
   - **truth-forced sweep**: 정답(S-crosshair)에서 wide scale(0.5~1.4)로 chamfer 강제 측정 → 병목(edge/metric vs scale-band vs reference) 분리. 진단 5종.
4. **matcher Slice 1** (`align_key_matcher.py`) — Codex 합의: 점수를 올리기보다 **wrong top-1 채택 방지**가 우선.
   - score map → NMS **top-N 후보**(`compute_chamfer_candidates`) + **distinctiveness gate**(best vs 2nd). 기존 동작 보존(10/10 호환).
5. **matcher 신호 연결 (Item 2)** (`align_point_correction.py`) — 엔진 distinctiveness 를 A-safe(병렬 계측 + 보수적 OR)로 status 에 연결.
6. **docs 재구성** — 사용자가 `study/`·`weekly_report/`·`journals/` 로 재정리. recovery plan 을 journals 로 이동 + **§5 라이브 align-key 탐색**(matcher 신뢰가 선행조건) 추가.
7. **코드 리뷰** — high-effort 7-angle 멀티에이전트 리뷰로 10개 findings 도출.

### 실데이터로 확정된 핵심 사실 (오피스 339장)
- **E(fail) 이미지는 crosshair 가 거의 없음** (with_crosshair=0/182). → crosshair 존재 여부 자체가 S/E 를 가르는 신호. "E=crosshair 틀린 위치" 유형은 이 데이터셋엔 사실상 없음.
- **정답(S-at-crosshair)에서조차 matcher median=0.62 ≈ match_threshold.** → free 검색이 다른 곳에서 우연히 더 높게 나와 green mark 가 틀림 = matcher 변별력 부족.
- crosshair 는 full-span 흰 직선. v2 검출률 ≈ S 79%, E 0%(정상).

---

## 2. 수정 내용

신규 파일:
- `poc/workflow_2/align_review.py`
- `poc/workflow_2/align_digest.py`
- `poc/workflow_2/crosshair_detect.py`
- `poc/workflow_2/align_similarity.py`
- `poc/workflow_2/test_align_key_distinctiveness.py`
- `poc/workflow_2/test_truth_forced_sweep.py`
- `poc/workflow_2/docs/align_success_dataset_plan.md`

수정 파일:
- `poc/workflow_2/align_key_matcher.py` — `AlignKeyCandidate` 추가, `_chamfer_score_map_at_scale`/`_extract_peaks`/`compute_chamfer_candidates` 추가, `compute_align_key_score` 가 top-N+distinctiveness 반환(하위호환), `MatchPolicy` 에 top_n/min_distinct_gap/max_second_ratio.
- `poc/workflow_2/align_point_correction.py` — `_ModalityScore` engine_* 필드, `_match_against` 엔진 신호 복사 + engine_scope, `_process_msr_image` not_distinctive = attempt OR engine(prior_roi 제외), `not_distinctive_source`, summary 에 source counts + 임계, `run()` 에 review 자동 생성.
- recovery plan 이동 + §5 추가.

커밋(main 직접): `edc6361`(review) `88576d7`(digest) `2cc6ce9`·`7ee9c73`(crosshair v2) `a0026f2`·`d65b700`·`4c26ab9`(similarity/staleness/golden plan) `8ccf15b`(matcher Slice1) `9b98b10`(docs reorg+§5) `9a415ce`(truth-forced) `dd2bba8`(engine 연결).

검증(Mac, blind): matcher 10/10 호환 + distinctiveness PASS + truth-forced(scale-band) PASS + 각 모듈 합성 self-test. 실데이터 full 실행은 오피스.

---

## 3. 다음 단계

### 오피스 실행 → 텍스트 회신 대기 (방향 결정용)
```bash
uv run python poc/workflow_2/crosshair_detect.py        # v2 검출률 S/E split
uv run python poc/workflow_2/align_similarity.py        # 유사도 표 + staleness + TRUTH-FORCED
uv run python poc/workflow_2/align_point_correction.py  # 보정 batch (engine_* + not_distinctive_source)
```
- **TRUTH-FORCED 진단 counts** → 병목 결정: scale_band_problem 多 → C4(scale band 확장), metric/edge 多 → MI/Canny, template_weak/reference → rcp 재등록.
- **유사도 표**(free_best/MI/NCC bACC, free_best vs free_best_box) → MI 채택·template 확정.
- **not_distinctive_source_counts**(engine_only 비율) → status gate 재조정 여부.

### 코드 리뷰 fix (권장 우선순위 — 사용자 선택 대기)
- **#1 (correctness)** `align_key_matcher.compute_chamfer_candidates` cross-scale NMS 가 현재 후보 merge_r 사용 → 인접 scale 중복 후보로 false not_distinctive. fix: kept peak 의 merge_r(또는 max) 사용.
- **#7 (efficiency)** `_truth_forced` 가 scale·modality 마다 전처리(distanceTransform) 재실행(~16회/이미지). fix: ROI 1회 전처리 / 단일 multi-scale 호출. (오피스 실행 속도 직결)
- **#3 (diagnostic correctness)** `build_probe_montage` 가 detector 와 다른 파이프라인으로 mask 재계산 → montage 가 검출 실제 mask 와 불일치.
- 기타: #2(_extract_peaks min_score), #4(crosshair line argmax가 밝은 bar 오선택), #5(masked_retry 엔진값 오염 → audit-only), #6(row 스키마 not_distinctive_source 누락), #8(_process_msr 전처리 중복), #9(COMPARE_SCALES 중복), #10(crosshair v2 미swap).

### 후속 단계 (합의된 순서)
- Phase 2 contour verifier → Phase 3 MI reranker(top-N crop, tiebreaker) — 유사도 표 확인 후.
- crosshair v2 → 생산 파이프라인 swap (오피스 v2>old 확인 후).
- **golden(항상 성공) 데이터셋 수집 — 다음 주(2026-06-01 주)**, staleness 임계 실측 calibration.

> 열린 결정(사용자 확인 필요): (a) 코드 리뷰 fix 중 어느 것을 먼저 적용할지, (b) 오피스 표를 먼저 받고 방향을 정할지. 현재는 (b) → 표 수신 후 Phase 3·fix 동시 결정 권장.

---

## 4. 메모리 업데이트

- `MEMORY.md` 갱신:
  - `feedback_commit_directly_to_main.md` (브랜치 안 만들고 main 직접 커밋)
  - `feedback_no_office_data_to_mac.md` (오피스 데이터 Mac 반입 불가 → blind+텍스트 회신)
  - `project_e_images_no_crosshair.md` (E=crosshair 없음, 실데이터 도메인 사실) — 이번 세션 신규.
- 그 외 코드 구조/모듈은 CLAUDE.md 와 이 저널이 기록하므로 MEMORY.md 추가 불필요.
