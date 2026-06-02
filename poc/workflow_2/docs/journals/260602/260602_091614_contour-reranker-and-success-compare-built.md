# 260602 — contour reranker + success_vs_fail_compare 구현, golden 워크스트림 재조정

> 세션: 2026-06-02 (오전, 이어서)
> 주제: test1 판정 이후 — (1) 재등록 워크스트림 재조정(fail S → golden), (2) contour reranker A/B 구현,
>        (3) success_vs_fail_compare.py 신규 구현. 둘 다 Mac self-test 통과 + push 완료.
> 선행 저널(중복 금지, 참조): [test1 분석·MI 폐기·contour 근거](260602_075313_mi-reranker-ruled-out-contour-next.md)
> 관련 문서: `docs/align_success_dataset_plan.md`, `docs/study/runbooks/workflow_2_procedure.md` §6.1

---

## 1. 진행 사항

### (A) 재등록 워크스트림 재조정 — fail S 폐기 → golden 특성화
- 처음엔 fail 폴더의 S 로 consensus 템플릿을 만드는 **재등록 도구**를 brainstorm(결정 3개 확정:
  standalone / `ALIGN_USE_CONSENSUS` opt-in / S≥3+일관성 게이트).
- 사용자 지적으로 방향 전환: **fail 폴더의 S 는 표본 적고 편향**(결국 실패한 run 의 성공 step)이라
  신뢰도 낮음 → 재등록(템플릿 교체) 보류. 대신 **golden(항상 성공, 9:1 로 풍부) 데이터로 success vs
  fail 의 rcp↔msr 차이를 지표화 → 엔지니어 가이드라인** 으로 목적 이동.
- 핵심 함정 합의: recipe 가 두 트리에서 다르므로 **절대 차이 비교 불가 → 상대 ratio**
  (rcp_vs_consensus / S_internal) 로만 비교. golden 은 S 표본이 많아 이 상대 지표가 안정적.
- 규모 가이드 확정: **recipe당 S 8~10장(wafer/lot/시간 분산), recipe 10(부트스트랩)→30(사용가능,
  fail ~26 과 균형)→50+(OM/SEM 별)**.

### (B) contour reranker A/B 구현 (align_similarity.py) — 우선 워크스트림
- MI rerank 가 test1 에서 폐기된 뒤, topk_not_rank1 갭을 **형상(contour/Hu)** 으로 메우는지 같은 A/B
  하네스로 잴 컬럼 추가. chamfer=멤버십 게이트 유지, contour=순서만(하드 reorder) → in_topk 불변.
- `_contour_sim`/`_hu_log`: Otsu 이진화 + log-Hu moment L1 거리(높을수록 유사).
- self-test: 같은 패턴 self=−0.000 vs 다른 패턴 diff=−37.85 변별 확인 + 멤버십 불변 assertion. 통과.

### (C) success_vs_fail_compare.py 신규 구현 — golden 워크스트림
- golden/fail 양 트리를 `_build_templates`+`_process_msr`(재사용)로 crop 수집 → `_reference_quality`
  (재사용)로 recipe별 상대 ratio·status 산출.
- 신규 로직: `_calibrate`(golden healthy 하위 p10 → stale 임계 제안 + 현재 임계 거짓양성),
  `_apply_threshold`(fail 에 적용 → drift recipe), `_build_comparison`, `_build_guideline`(한국어 md).
- self-test(합성 per-recipe 행, 이미지 트리 불요): golden median 0.925 > fail 0.525,
  현재 임계 거짓양성 0, S_inconsistent 는 scorable 제외, 빈 입력 안전. 통과.

### (D) 결과지 채움 양식 + 문서
- console INFO 타이핑 부담 → **숫자만 채우는 양식** 2종 신설(`_TEMPLATE_align_similarity.txt`,
  `_TEMPLATE_success_vs_fail_compare.txt`). contour/compare 줄 포함.
- plan §2/§3/§6/§7 재조정, procedure §6.1(선행 세션), 메모리 갱신.

## 2. 수정 내용 (커밋, main push 완료)
- `c1eb570` — align_similarity contour reranker A/B 컬럼(`_contour_sim`/`_hu_log`,
  `topk_rank_reranked_contour`, consensus_ab/gt_topk 집계, 콘솔, self-test).
- `a0c98e2` — align_similarity 템플릿에 contour 칸.
- `f99adbb` — `success_vs_fail_compare.py` 신규 + plan §6 체크 갱신.
- `ba939d3` — success_vs_fail 템플릿.
- (선행) `e4f8555` — test1 저널 + procedure §6.1 + plan 재조정 + test1 결과지.
- 검증(Mac): 두 모듈 self-test 통과, syntax OK, working tree clean.

## 3. 다음 단계

**⏳ office (사용자, fab 데이터 필요) — 둘은 독립, 순서 무관:**
1. **contour A/B 실행** — pull → `uv run python poc/workflow_2/align_similarity.py` →
   CONSENSUS A/B 의 `+contour rerank ... rerank_lift=____` 채워 회신(`260602_test2.txt`).
   - 판정: `rerank_rank1_lift_contour ≥ +0.10` → contour reranker 승격(재등록+contour 로드맵).
   - `≈0/음수` → reranker 레버 사망 → **proposer 교체 / live-search 분리**로 escalation.
2. **golden 수집 + compare 실행** — `align_images_golden/<eqp>/<class>/<recipe>/{from_rcp,from_msr}`
   에 부트스트랩 ~10 recipe(recipe당 8~10 S) 적재 → `success_vs_fail_compare.py` →
   golden 템플릿 채워 회신 → calibration + drift 가이드라인 판독.

**다음 내 작업(결과 회신 후 분기):**
- contour 통과 시: production 승격 경로(`align_point_correction` 후보 reorder) 설계.
- contour 실패 시: 2차 proposer(contour/AKAZE/MI-coarse 후보 생성기) 또는 live-search 분리 검토.
- golden 결과 후: 실측 percentile 로 `RELATIVE_STALE_RATIO`/`S_INCONSISTENT_CV` 확정, 30+ 로 확장.

> ❓ 확인 필요: 없음(다음은 office 실행 → 결과 회신 대기). 결과 들어오면 판정 게이트대로 판독.

## 4. 메모리 업데이트
- `project_matcher_flat_chamfer_distinctiveness.md` — test1 섹션(MI 폐기·WHY·contour)은 선행 세션에 반영됨.
  이번 세션의 contour 구현/golden 재조정은 코드·plan·저널에 기록되어 별도 메모리 추가 불필요.
- `.remember/now.md` — 07:53 항목에 test1 판정·contour 전환 반영됨.
- MEMORY.md 인덱스 — 변경 없음(기존 줄 유효).
