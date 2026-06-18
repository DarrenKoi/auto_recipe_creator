# OM/SEM modality-split 평가 + 조건부 lever — 설계 (combined eval Step 2)

> 상태: draft (브레인스토밍 합의 완료, 사용자 검토 대기)
> 선행: ADR 0004 (routed combined eval + S 수집, Step 1 = modality 층화 측정)
> 위치: 오프라인 CV 벤치(`poc/workflow_2/`). production 엔진은 lever 승자 확정 후 별도 포팅.

## 1. 문제

production 엔진은 OM(저배율·반복패턴 多)과 SEM(box/직선·edge 희소·contrast 다름)에 **동일 단일 CV
정책**을 쓴다: 전역 Canny 임계, 단일 scale band, OM+SEM 섞어 calibration 한 공용 Youden
임계(`MatchPolicy.ensemble_match_threshold=0.6053` / `ensemble_adjust_threshold=0.4727`).
`route_template` 은 *어느 템플릿*(OM/SEM)을 고를 뿐, 매칭 정책은 안 가른다.

ADR 0004 Step 1 으로 combined eval 에 modality 층화(`by_modality`)는 들어갔지만, 지금 내보내는
건 OM/SEM 각각의 `rank1_rate / in_topk_rate / n_frames` **집계값**뿐이다. 판정 주석은 "OM rank1
이 SEM 보다 확연히 낮고 **실패유형이 다르면** → split 검토"라고만 적혀 있는데, **현재 eval 은
'실패유형이 다른지'를 전혀 측정하지 않는다.** rank1 이 낮은 이유가

- 주기적 닮은꼴에 1등을 뺏김 (look_alike, OM 의 전형 — wrong_local_peak), 인지
- 진실이 후보 pool 에 아예 없음 (recall_miss, SEM edge-sparse 의 전형), 인지

를 모르면 split 여부도, 어떤 lever 를 쓸지도 정할 수 없다. 이 둘은 **완전히 다른 lever** 를 부른다.

## 2. 목표

"OM/SEM CV 정책을 쪼갤까"를 **증거 기반**으로 판정한다:

1. combined eval 에 **per-modality 실패유형 진단** + **명시 verdict** 를 넣어 오피스 1회 실행으로
   "split 할지 / 어느 modality 에 어떤 lever 인지"가 한 줄로 떨어지게 한다.
2. modality별 lever 3종(L1 per-mod Youden, L2 OM 주기억제, L3 SEM recall proposer)을 **설계**하되,
   구현은 증거에 **게이트**한다(페이즈드).

**비목표(out of scope):** consensus arm 엔진 변경(production-fixed), 오피스 데이터 수집,
production 엔진 modality-split 배선(lever 승자 확정 후 별도 포팅).

## 3. 데이터 사실 — plumbing 비용 (거의 0)

두 arm 모두 **추가 plumbing 없이** 기존 산출물로 실패유형을 잰다:

- **rcp_only arm** — `_score_rcp_only` → `_process_msr_cond` → `_localize` 의 per-frame 셀이
  `mod, score, align_xy, dist_norm, hit, topk_rank, in_topk, second_ratio, score_gap, distinctive`
  를 다 담는다. 실패유형 3분류·Youden 분리 모두 이 셀 집계로 끝.
- **consensus arm** — `_consensus_template_ab` 의 per_recipe row 는 `cons_in_topk_rate`,
  `cons_rank1_rate`, `n_S_loo` 를 노출한다. 실패유형 개수를 **rate 에서 재구성**한다(기존
  `_calibrate_periodicity` 가 이미 같은 방식으로 miss 수를 복원):
  `recall_miss ≈ n·(1−in_topk)`, `look_alike ≈ n·(in_topk−rank1)`, `rank1_hit ≈ n·rank1`.
- **periodic 신호** — 기존 `template_periodicity(raw_image, ...)` 재사용(consensus eval 의
  reregister-candidate 신호로 검증됨, `PERIODICITY_TAU` + ablation grid 존재). recipe-level 로
  "이 등록 key 가 주기/대칭인가"를 주고, 실패유형 집계 때 recipe 별로 join 한다.

따라서 **`_localize` 변경 불필요.** Phase 1 은 combined 드라이버 순수 추가 + `template_periodicity`
재사용으로 끝난다. GT(`crosshair_xy`)는 S 프레임마다 이미 있고(cond 또는 검출), S 만 채점 대상이라
GT 가 항상 존재한다.

## 4. 설계 — 평가 강화 (Phase 1)

대상 파일: `poc/workflow_2/golden_combined_eval_cond.py`(집계·verdict),
`poc/workflow_2/golden_localization_eval.py`(`_localize` 의 periodic 필드).
순수 헬퍼는 모두 `test_golden_combined_eval_cond.py` 에 합성 row 로 단위테스트(golden 불요, Mac 실행).

### 4.1 per-modality 실패유형 히스토그램

각 채점 셀을 기존 필드로 분류한다:

| 유형 | 조건 | 의미 |
| --- | --- | --- |
| `rank1_hit`   | `topk_rank == 1`        | 성공 |
| `look_alike`  | `topk_rank` ∈ 2..k      | 진실은 후보에 있으나 다른 위치(주로 주기적)에 1등 뺏김 |
| `recall_miss` | `topk_rank is None`     | 진실이 후보 pool 에 아예 없음(proposer 가 못 띄움) |

`look_alike` 에 **`periodic` 서브플래그**(§4.2). 집계 단위: modality(OM/SEM) × arm
(consensus / rcp_only / routed).

- **rcp_only arm**: per-frame 셀 집계. 순수 헬퍼 `_classify_cell(cell) -> str` +
  `_failure_hist_by_modality(cells_by_mod) -> {mod: {type: {n, share}}}`.
- **consensus arm**: per_recipe row 의 rate 에서 재구성(§3). 순수 헬퍼
  `_failure_hist_from_rates(per_recipe) -> {mod: {type: {n, share}}}` —
  `recall_miss = round(n·(1−in_topk))`, `look_alike = round(n·(in_topk−rank1))`,
  `rank1_hit = n − recall_miss − look_alike`(음수 클램프). bit-parity: 매칭 수학 불변, rate 만 읽음.

매칭 수학·`_localize` 모두 불변(순수 후처리 집계).

### 4.2 periodic 서브플래그 (기존 `template_periodicity` 재사용)

새 검출기를 짜지 않는다. consensus eval 에 검증된 `template_periodicity(raw_image, win_frac=,
min_lag_frac=)` 가 등록 key 의 주기/대칭성 점수를 준다(reregister-candidate + `PERIODICITY_TAU` +
ablation grid 로 이미 사용 중). combined 드라이버가 recipe 별로 modality 템플릿의 periodicity 를
계산해(consensus eval 과 동일: modality max) `periodic_recipe = score > PERIODICITY_TAU` 로 라벨하고,
실패유형 집계 때 join 한다:

- **periodic-look_alike** = `periodic_recipe == True` 인 recipe 의 look_alike. (주기 템플릿이 격자
  닮은꼴에 1등 뺏기는 OM 의 전형 — L2 의 직접 증거.)

recipe-level 신호라 per-frame plumbing 0. (더 정밀한 per-frame "이 변위가 pitch 배수인가"가 필요하면
`template_periodicity` 를 dominant pitch 벡터까지 반환하도록 확장하는 후속 refinement 으로 둔다 —
SPLIT 판정엔 recipe-level 로 충분.)

단위테스트: 합성 주기/비주기 템플릿으로 `periodic_recipe` 라벨 + look_alike join 검증.

### 4.3 per-modality Youden 분리 (분류 축, rcp_only arm)

L1 의 증거이자 평가. per-frame `score`+`hit` 가 필요하므로 **rcp_only arm 셀**에서 잰다(consensus
arm 의 per_recipe row 는 per-frame score 를 안 남김 — rcp localization 경로가 production rcp 폴백과
일치하니 분류 축은 거기서 충분). modality별로 (score, hit) [및 (second_ratio, hit)] 표본을 모아:

- modality별 **최적 Youden J 임계** 산출(score sweep, J = TPR − FPR 최대점).
- 공용 임계(0.6053 / 0.4727) 대비 per-mod 최적 임계의 **차이** + 그 임계에서의 혼동행렬
  (TP/FP/TN/FN)·정확도 변화.

순수 헬퍼 `_youden_threshold(samples) -> {thr, J, tpr, fpr}` +
`_youden_by_modality(cells_by_mod) -> {mod: {...}}`. 합성 분리/비분리 표본으로 단위테스트.

> 축 구분(eval_guide_01): L1 은 **분류**(임계가 feasible/infeasible 를 잘 가르나)이지 localization
> rank1 이 아니다. 그래서 eval 도 분류 축(혼동행렬)과 localization 축(rank1/recall)을 **따로** 낸다.

### 4.4 split verdict (판정 규칙)

per modality 쌍(OM vs SEM)에 대해 다음 게이트를 순서대로 적용한다. 임계값은 모두
**`golden_eval_config` 의 명명 상수**(오피스에서 데이터 보고 튜닝). 기본값:

```
SPLIT_MIN_FRAMES   = 30     # 신뢰 게이트: modality당 최소 채점 프레임
SPLIT_MIN_RECIPES  = 5      # 신뢰 게이트: modality당 최소 recipe
SPLIT_RANK1_GAP    = 0.10   # 헤드라인 격차(10pp)
SPLIT_RANK1_FLOOR  = 0.70   # 약한 쪽 절대 바닥
SPLIT_DOMINANCE    = 0.40   # 지배 실패유형 최소 비중
```

규칙:

```
1. 신뢰 게이트:  n_frames(m) ≥ SPLIT_MIN_FRAMES  AND  n_recipes(m) ≥ SPLIT_MIN_RECIPES
                 → 미달 modality 가 있으면 verdict = "insufficient" (판정 보류)
2. 헤드라인 격차: |rank1(OM) − rank1(SEM)| ≥ SPLIT_RANK1_GAP
                 OR  min(rank1(OM), rank1(SEM)) < SPLIT_RANK1_FLOOR
3. 실패유형 분기: 두 modality 의 *지배적* 실패유형이 다름
                 (예: OM 의 periodic-look_alike 비중 ≥ SPLIT_DOMINANCE
                  AND SEM 의 recall_miss 비중 ≥ SPLIT_DOMINANCE)

verdict:
  • 격차 AND 분기   → "SPLIT"        (+ 지배유형 → 권장 lever: OM→L2, SEM→L3)
  • 격차 BUT 분기X  → "shared_tune"  (같은 실패유형, 공용 정책 강화)
  • 격차 없음       → "no_split"
```

순수 함수 `_split_verdict(by_mod_failure_hist, by_mod_rates, cfg) -> {verdict, weaker_mod,
dominant_om, dominant_sem, suggested_levers}`. 합성 시나리오(SPLIT/shared/no_split/insufficient
각각)로 단위테스트.

### 4.5 출력

`summary.json` 에 `by_modality.failure_modes`(§4.1), `by_modality.youden`(§4.3),
`split_verdict`(§4.4) 추가. `_print_report` 에 실패유형 표(modality × 유형, share) + Youden 표 +
verdict 줄 추가. `_digest_line` 에 `verdict=SPLIT om[la/pm/rm=..] sem[..] | youden om=thr/J ...`
한 줄 노출(사용자가 이 줄만 복사해 전달).

## 5. 설계 — lever 3종 + A/B 하니스

모든 lever 는 `ensemble_lab.py` 의 lab-mode 패턴(`ALIGN_ENSEMBLE_LAB_MODE`, 현재 `edge_ncc`/`c4`)
으로 꽂고, combined eval 의 **rcp-only arm**(consensus 가 못 돕는 영역 = lever testbed)에서
by-modality 실패유형 히스토그램 전후 비교로 측정한다. consensus arm 은 production 고정.

| Lever | 증거(트리거) | 메커니즘 | 평가 축 | A/B |
| --- | --- | --- | --- | --- |
| **L1** per-modality Youden | OM/SEM score↔hit 최적 임계가 공용과 유의하게 다름(§4.3) | 엔진 `distinctive`/reject 임계를 modality 조건부로 | 분류(feasibility go/no-go 혼동행렬) | eval 이 per-mod 최적 Youden 산출 → 공용 대비 개선. orthogonal·무조건 |
| **L2** OM 주기억제 | OM 에서 periodic-look_alike 지배 | lattice-NMS(상위 peak 의 pitch 배수 후보 억제) 또는 기존 **C4 context-distinctiveness 채널**(`2026-06-10` 설계) 재사용 | localization(look_alike→rank1_hit 전환) | 신규 lab mode `om_periodicity`, by-modality 히스토그램 전후 |
| **L3** SEM recall proposer | SEM 에서 recall_miss 지배 | ensemble proposer 에 box/corner(Shi-Tomasi)/Hough-line 추가(SEM 한정) | localization(recall_miss↓ = gt_in_topk↑) | 신규 lab mode `sem_struct`, by-modality recall 전후 |

CV 가 좌표 권위 유지(2026-05-25 룰) — lever 는 proposer/임계만 건드리고 최종 좌표는 NCC rerank.

## 6. 구현 페이징 (plan 단계)

- **Phase 1 — 증거 eval.** §4 전부(실패유형 히스토그램 + periodic + per-mod Youden + verdict).
  Mac 합성 row 단위테스트 통과 → 오피스 1회 실행 → verdict 확보. **출시 게이트.**
- **Phase 2 — L1.** per-modality Youden(싸고 orthogonal). 증거와 무관하게 진행 가능하나, §4.3 가
  per-mod 임계 차이를 확인해준 뒤 production 임계를 modality 조건부로.
- **Phase 3 — L2/L3.** by-modality 증거의 **지배 실패유형이 가리키는 lever 만** lab mode 로 채움.
  OM periodic 지배면 L2, SEM recall 지배면 L3. 둘 다면 둘 다, 분기 없으면 shared_tune.

## 7. 테스트

- 순수 헬퍼(`_classify_cell`, `_failure_hist_by_modality`, `_failure_hist_from_rates`,
  `_youden_threshold`, `_youden_by_modality`, `_split_verdict`)는 합성 row 단위테스트
  (`test_golden_combined_eval_cond.py` 확장; golden 불요, Mac 실행). 기존 73 passed 회귀 가드.
- periodic join 은 합성 row(`periodic_recipe` True/False)로 look_alike 분류가 갈리는지 검증.
  `template_periodicity` 자체는 consensus eval 에서 이미 검증됨(재사용).
- 오피스: `uv run python poc/workflow_2/golden_combined_eval_cond.py` 가 실제 수치 + verdict 산출.
  `ALIGN_ENSEMBLE_LAB_MODE=om_periodicity|sem_struct` 로 Phase 3 lever A/B.

## 8. 인터페이스 요약

- `_localize` / 매칭 경로: **불변** (periodic 은 `template_periodicity` 재사용 + rate 재구성으로
  순수 후처리 — production 인접 코드 안 건드림).
- `golden_combined_eval_cond` 신규 순수 헬퍼: `_classify_cell`, `_failure_hist_by_modality`,
  `_failure_hist_from_rates`(§4.1), `_youden_threshold`/`_youden_by_modality`(§4.3),
  `_split_verdict`(§4.4). 기존 `_arm_rates`/`_routed_overall`/`_consensus_by_modality` 등은 불변.
  `summary` dict 에 `by_modality.failure_modes` / `by_modality.youden` / `split_verdict` 키 추가.
  `template_periodicity` 는 `golden_consensus_eval_cond` 에서 import 재사용.
- `golden_eval_config`(+example): §4.4 의 5개 SPLIT_* 상수 신규 블록. combined 드라이버가
  import-with-fallback(loader 패턴)로 직접 읽음(env 브리지 불필요 — 이 드라이버 전용 판정 knob).
- `ensemble_lab`: Phase 3 에서 `om_periodicity` / `sem_struct` lab mode 추가(Phase 1/2 에선 미추가).

## 9. ADR 0004 에서 이어받는 wart

consensus arm 의 rcp counterfactual 은 center 템플릿, rcp-only arm 은 box 템플릿이라 lift 비교는
arm 별로만 해석한다. by-modality 분해도 이 제약을 그대로 상속 — modality별 verdict 도 arm 내에서만
rcp 와 비교한다(routed pick 은 일관: eligible=consensus, rest=rcp box).
