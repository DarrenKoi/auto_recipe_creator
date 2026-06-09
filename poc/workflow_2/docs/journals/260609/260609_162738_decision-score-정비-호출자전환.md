# decision/score 정비 + 호출자 4개 전환 (ensemble production 완성)

> 2026-06-09. NCC reranker production 배선(hit 0.608) 후속. ensemble 의 결정 score/decision 을
> selection 과 일치시키고, fallback/static 호출자 4개를 ensemble 로 전환·검증.

## 1. 진행 사항

선행 상태: `compute_align_key_score_ensemble` 의 candidate selection 은 chamfer+NCC(검증된
e2e hit 0.608)인데, **결정 score/decision 은 공유 `_finalize_match` 가 여전히 chamfer+ORB** 로
계산 → NCC 가 저-chamfer 정답을 고르면 score 가 낮아 `decision="low"` 오판 → 호출자 하드게이트가
정답 좌표를 버릴 위험. 이번 세션에서 이 블로커를 해소하고 호출자를 전환.

- **brainstorm(안 A 확정)** — 설계 4점 결정: ORB 완전 제거 / threshold Youden J / 호출자 직접
  교체 / vlm_align_key_region 이번 제외. spec 작성.
- **calib 러너** — `reranker_rule_ab._calibrate_thresholds`(Youden J + 고-recall) 추가, 오피스
  1회 실행으로 threshold 산출: **match=0.6053**(prec0.894/recall0.838), **adjust=0.4727**(recall0.95).
- **score 정비 구현(TDD)** — `_finalize_match` score_override/decision_thresholds 옵션,
  `_decision_for_score` threshold override, MatchPolicy ensemble 임계 필드, ensemble ORB 폐지.
- **code-review(7-angle)** — guard 범위 불일치 버그 등 5건 수정.
- **호출자 4개 전환** — match_recipe_key_on_crop · compare_align_images · align_fail_correct ·
  align_point_correction.
- **codex 리뷰 반영** — `_ensemble_pool` production 정합(cap + chamfer round6).
- **e2e 재확인(오피스)** — hit 0.608 유지, decision=match 과대확신 해소 검증.

## 2. 수정 내용

### 신규
- `poc/workflow_2/docs/specs/2026-06-09-ensemble-decision-score-design.md` — 안 A 설계.

### `poc/workflow_2/align_key_matcher.py`
- `MatchPolicy`: `ensemble_match_threshold=0.6053` / `ensemble_adjust_threshold=0.4727`(Youden 캘리브,
  rerank 가중 0.5/0.5 분포에 묶임 — 변경 시 재캘리브 경고 주석).
- `_decision_for_score(...)`: keyword `match_threshold`/`adjust_threshold` override(미지정 시 policy).
- `_finalize_match(...)`: `score_override`(None→baseline chamfer+ORB 바이트 동일, 주어지면 그 값) +
  `decision_thresholds`. **compute_align_key_score 무변경 보존(회귀가드 10/10).**
- `compute_align_key_score_ensemble`: 결정 score = selection sel(score_override=best_sel) + ensemble
  임계, ORB 계산 폐지(orb=0). **ens.fused(shadow24)를 top_n 으로 먼저 cap** → guard·selection·pool
  동일 범위(shadow 양-chamfer 가 guard 통과시키나 zero-chamfer top_n 픽하던 불일치 해소).

### `poc/workflow_2/reranker_rule_ab.py`
- `_calibrate_thresholds(pairs, recall_target=0.95)` — ROC Youden J(match) + 고-recall(adjust),
  round6, adjust==match 붕괴 warning. `_ens_ncc_sel` 헬퍼로 sel 공식 단일화(production 과 drift 방지).
- run(): ens_ncc sel 덤프 + 캘리브 산출·콘솔 출력(오피스 1회 실행이면 threshold 바로 찍힘).

### 호출자 전환 (4)
- `match_recipe_key_on_crop.py` / `compare_align_images.py` — drop-in(`_verdict_for`·match 게이트는
  decision 기반이라 ensemble 내부 재판정으로 안전). orb 표시 0.
- `align_fail_correct.py` — **HARD BLOCKER 해소**: `key_visibility_gate` 의 adjust 분기 `orb>0`
  (ensemble orb=0 → adjust→primary 전멸)을 `result.distinctive` 로 대체(presence 2차 확인). test_gate 갱신.
- `align_point_correction.py` — 두 call site 전환 + 절대 score 게이트 2곳을 `ensemble_adjust_threshold`로
  교체(:1148 crosshair-prior 채택, :1607 low_match_both). 상대 비교(modality race·distinctiveness ratio)는 불변.

### `poc/workflow_2/localization_regression_diag.py`
- `_ensemble_pool`: production 정합(`ens.fused[:top_n]` cap, chamfer round6).

### 테스트
- `test_align_key_score_ensemble.py`: score_override/threshold override/guard cap 단언 추가(9 pass).
- `test_reranker_rule_ab.py`: `_calibrate_thresholds` 3 케이스(separable/overlap/degenerate).
- `test_align_fail_correct.py`: 게이트 distinctive 기반 갱신.
- 전체 **144 pass**, baseline 회귀가드 **10/10**.

## 3. 검증 결과 (오피스 e2e, localization_ab_ensemble)

- **hit_rate**: baseline 0.422 → **ensemble 0.608** (guard cap 변경 후에도 유지, 회귀 0).
- confusion: both_hit=297, only_ens(gain)=163, only_base(regress)=22, both_miss≈274 (n≈756).
- **decision=match: base 657 → ens 432** — 핵심. 이전 ensemble chamfer+ORB score 의 match 과대확신
  (657@0.422 = 거짓 match 대량)이 해소되고, ens 432 가 실제 hit(460)과 정합 → **decision=match
  게이트가 신뢰 가능**해짐. 정비의 목표 달성.

## 4. 다음 단계

- (선택) 가중치 sweep — 0.608 이 rerank 0.5/0.5; 바꾸면 ensemble 임계 **재캘리브 필수**.
- live 경로(live_align_search/search_align_key)는 의도적으로 baseline 유지(프레임당 ~1s 비용).
- 후순위: 하드 플로어/consensus 재등록, 보류된 consensus 출력 정리(modality race).

## 5. 메모리 업데이트

`project_ensemble_proposer_and_consensus_race.md` 갱신 — decision/score 정비 완료·검증 + 호출자
전환 결과(hit 0.608 유지, decision 657→432) 반영.
