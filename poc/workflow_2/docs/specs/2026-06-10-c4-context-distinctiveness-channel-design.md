# C4 context/distinctiveness proposer 채널 — 설계 (Codex 리뷰 반영판)

작성: 2026-06-10 · 대상 코드: `poc/workflow_3/vision/ensemble_proposer.py`,
`poc/workflow_3/vision/align_key_matcher.py`,
`poc/workflow_2/align_similarity.py`, `poc/workflow_2/golden_consensus_eval_cond.py`
선행: ensemble proposer(C1/C2/C3 + RRF) 프로덕션 통합 완료
(`2026-06-09-ensemble-proposer-design.md`), consensus eval ensemble 토글
(`CONSENSUS_USE_ENSEMBLE`, 본 작업과 함께 추가됨).

---

## 1. 배경 / 문제

consensus·localization 의 align point 후보 생성은 전부 **edge 기반 chamfer** 다.
ensemble 의 세 채널(C1 canny / C2 scharr / C3 orientation)도 *edge 추출 방식*만
다를 뿐, 점수는 모두 "template edge 가 frame edge 에 얼마나 겹치나"(appearance
overlap)를 잰다. 이 점수에는 **유일성(uniqueness)** 개념이 없다.

→ 반복 패턴(line/space grating, contact array — SEM 에 흔함)에서는 주기마다 edge 가
동일하게 겹쳐 **점수면이 평평**해지고, matcher 가 "옳은" 주기를 고를 수 없다
(실데이터 `wrong_local_peak 67%`, [[project_matcher_flat_chamfer_distinctiveness]]).
edge 채널을 더하거나 바꾸는 것(=ensemble)으로는 이 모호성을 깰 수 없다 —
*edge 검출 방식*이 아니라 *매칭 위치의 자기-유일성*이 빠진 신호이기 때문이다.

기존 `distinctiveness_ratio`(`align_point_correction.py`)는 이를 일부 가드하지만
(a) match-time(template-vs-frame, 매칭 후에만 발화), (b) 여전히 chamfer-on-edge,
(c) terminal(모호하면 포기) 라 위 문제를 *해결*하진 못한다.

## 2. 목표 / 비목표

**목표**
- edge 채널이 버리는 두 신호를 ensemble 에 추가한다: (a) **배경 포함 intensity
  컨텍스트**, (b) **매칭의 유일성(반복 패턴 페널티)**.
- 같은 LOO harness 에서 C1 / C1+C2+C3 / +C4 **3단 ablation** 을 측정해 C4 의
  *한계 기여*를 분리한다.
- C4 가 구조적 miss(far/veryfar)를 실제로 끌어오는지 **miss-distance bin 별로**
  측정한다.

**비목표 (이번 범위 아님)**
- production 매칭 경로 기본 동작 변경 — C4 는 **opt-in**, 기본 채널셋 불변.
- live broad-scan 에 NCC 투입 — ensemble 은 기존대로 fallback/static-compare 전용.
- consensus/template-bank 증강·ROI 마스킹 — 더 강한/싼 레버지만 **별도 작업**(§9 참조).

## 3. Codex 리뷰 반영 요약

`/codex:rescue` 리뷰(2026-06-10)의 5개 지적을 다음과 같이 설계에 반영한다.

| Codex 지적 | 본 spec 반영 |
|---|---|
| **#1** NCC isolation 은 periodicity 를 못 본다; 인공물/경계가 가짜 유일성 생성 | §6.2 **template autocorr periodicity pre-score** 추가 → isolation 을 `trust=1−periodicity` 로 스케일. isolation 은 candidate-level feature 로만 사용 |
| **#2** TM_CCOEFF_NORMED 는 focus blur/contamination/charging 에 약함; 가짜 isolated peak | §6.3 **C4-only 후보는 chamfer rescore 생존 필수**. §7 에서 S/E NCC 분포 계측. opt-in·A/B 한정 |
| **#3** `_rrf_fuse` 대표 선택이 raw 이종 score 기준 → C4 가 cluster 대표 탈취 + 잘못된 scale 전달 | §5 **Phase 0 prerequisite**: 대표를 **C1 chamfer rescore(또는 medoid)** 로 선택하도록 `_rrf_fuse` 수정. C4 보다 먼저 |
| **#4** far/veryfar 구조적 miss 를 살릴지 미지수 | §7 **C4-only truth hit 을 near/mid/far/veryfar bin 별 보고**. 집계 recall 단독 금지 |
| **#5** C4 는 1순위 레버 아님; consensus/template-bank·ROI·autocorr 가 더 싸고 근본 | §4 **시퀀싱**으로 prereq(autocorr periodicity, RRF fix)를 먼저. C4 는 exploratory opt-in. §9 에 우선순위 명시 |

## 4. 시퀀싱 (Codex #5 반영)

C4 는 "최우선 레버"가 아니라 **edge 천장이 context/uniqueness 로 뚫리는지 측정하는
탐색 채널**이다. 구현 순서는 *싸고 근본적인 것 먼저*:

1. **Phase 0** — `_rrf_fuse` 대표 선택 robustness 수정(이종 채널 합류 전제). C4 없이도
   유효한 버그 수정이며 C4 의 선결 조건.
2. **Phase 1** — template autocorr **periodicity pre-score**. C4 와 독립적으로도
   "이 key 가 내재적으로 반복적/모호 → 재등록 후보"라는 무료 신호를 낸다.
3. **Phase 2** — C4 context-patch NCC proposer 채널 + ablation 토글 + 계측.

각 Phase 는 독립적으로 테스트·커밋한다(TDD). Phase 2 가 흐려도 Phase 0/1 은 남는다.

> **실행 결정 (2026-06-10)**: **Phase 0 + Phase 1 을 먼저 구현·측정**한 뒤, 그 결과로
> **Phase 2 진입을 게이트**한다. 즉 (a) RRF 대표 robustness 가 3채널 무회귀로 안착하고,
> (b) periodicity pre-score 가 golden set 에서 의미 있는 `template_periodic_rate` 와
> 재등록 신호를 내는지 확인한 다음에 C4-NCC(Phase 2)를 별도 plan 으로 착수한다.
> 본 plan 의 범위는 Phase 0 + Phase 1 까지.

## 5. Phase 0 — RRF 대표 선택 robustness (선결)

**문제**: `_rrf_fuse` 는 좌표 거리(Chebyshev ≤ `match_radius`)로 클러스터를 묶은 뒤,
**raw `score` 최댓값 멤버**의 `xy/scale` 을 대표로 채택한다(`ensemble_proposer.py`).
C1/C2/C3 는 score 가 모두 chamfer 계열이라 비교 가능하지만, C4 의 NCC-isolation 점수는
척도가 달라 대표를 부당하게 탈취하고 **잘못된 scale 을 downstream chamfer rescore 에
전달**할 수 있다(Codex #3). `match_radius=max(8, 0.05·short)` 는 pitch 무관이라
≈8px 주기 array 에서 인접 주기가 한 클러스터로 병합될 위험도 있다.

**수정**:
- 클러스터링은 좌표 기준 유지(rank-기반 RRF 가중 합도 유지 — 융합 자체는 불변).
- **대표 `xy/scale` 은 raw score 가 아니라 공통 yardstick 으로 선택**: 클러스터
  멤버들의 위치를 C1 chamfer 로 rescore 해 최고점을 대표로(없으면 geometric medoid).
  → 모든 채널이 동일 척도(chamfer)로 비교되어 이종 합류가 안전.
- `_rrf_fuse` 에 **per-channel weight** 인자를 추가(기본 1.0). Phase 2 가 C4 의 RRF
  기여를 `trust` 로 줄이는 데 사용(§6.2). C1/C2/C3 는 1.0 → 기존 동작 보존.

**behavior-preservation**: 3채널(C1/C2/C3)만일 때 대표 선택이 바뀔 수 있으므로,
`test_ensemble_proposer.py` 와 오피스 ablation 으로 **3채널 in_topk/rank1 무회귀**를
확인한 뒤 채택. 회귀 시 대표 선택 변경을 C4 존재 시에만 적용하도록 범위 축소.

## 6. Phase 2 — C4 context-patch NCC proposer

### 6.1 후보 생성 (NCC)
modality 별 template `raw_image`(=consensus/rcp crop, **이미 배경 grey 컨텍스트 포함**)에:
1. 각 scale s 에서 `cv2.matchTemplate(frame_gray, scaled_template, TM_CCOEFF_NORMED)`
   → NCC 응답맵(평균 차감·분산 정규화 → 밝기/대비 affine 드리프트 흡수).
2. `_extract_peaks` 로 NMS top-k peak → 후보 위치. **좌표는 `+(tw//2, th//2)` 로
   center 화**(§6.4) → `_Cand.xy` 계약(template-center, frame px) 준수.
3. scale 교차 global NMS → C4 solo 후보 리스트(ncc score 내림차순).

### 6.2 template periodicity pre-score (Codex #1)
template 당 **1회**(scale 무관 — 상대 비율) autocorrelation 으로 내재 주기성을 잰다:
```
g = template - mean(template)
AC = fftshift(real(ifft2(fft2(g) * conj(fft2(g)))))      # 2D autocorrelation
AC /= AC[center]                                          # zero-lag = 1.0
periodicity = max(AC outside central radius r_excl)      # 0..1, 주기 peak 높이
trust = 1 - periodicity
```
- `periodicity` 高(off-center peak 강함) → 반복 패턴 template → **어떤 위치도
  유일하지 않음** → C4 를 신뢰하면 안 됨.
- 사용처: C4 의 RRF 기여 가중 = `trust`(§5 의 per-channel weight). 반복 template 에서
  C4 가 *조용해진다*(graceful degrade). `periodicity > τ_p` 면
  `template_periodic=True` 진단 플래그 → **재등록 후보** 신호(무료 부산물,
  [[project_rcp_white_box_unique_area]] 연결).
- NCC peak-isolation(best − 비국소 차순위 peak)은 **candidate-level feature 로만**
  기록/정렬에 쓰고, "내재 유일성의 증거"로 격상하지 않는다(Codex #1).

### 6.3 C4-only 후보 chamfer 생존 게이트 (Codex #2)
RRF 융합 후, **C4 가 단독 surface 한 후보**(edge 채널 어느 것도 `match_radius` 내에
제안하지 않은 위치)는 contamination/charging 가짜 isolated peak 일 수 있다. 따라서:
- 그 위치를 **C1 chamfer 로 rescore** 해 `chamfer ≥ τ_c` 인 것만 최종 후보로 유지.
- (이는 §5 의 "대표=chamfer rescore" 와 같은 yardstick 으로 자연스럽게 합쳐짐.)
- TM_CCOEFF_NORMED 는 focus blur/contamination 에 약하므로(Codex #2), C4 단독 신호는
  edge-구조 생존을 통과해야 신뢰.

### 6.4 좌표 계약 (불변식)
NCC peak 도 `matchTemplate` 결과의 **top-left** 라 `+(tw//2, th//2)` 로 center 화해야
C1~C3 와 동일 계약(template-center, frame px)이 된다. 어긋나면 RRF `match_radius`
병합이 C4 를 별개 클러스터로 취급 → 채널 무력화. (이 불변식은 `CONSENSUS_USE_ENSEMBLE`
배선에서 검증한 것과 동일 — `_Cand.xy` 주석 `ensemble_proposer.py:125`.)

### 6.5 ensemble 통합 + 토글
- `_channel_solo_candidates` 에 `'context'` 분기 추가(C4 후보 생성).
- `compute_ensemble_candidates(..., channels=(...))` 파라미터 + env
  `ENSEMBLE_CHANNELS`(예: `"canny,scharr,orient"` **기본** / `"...,context"`).
  → 한 harness 에서 C1 / C1+C2+C3 / +C4 ablation.
- **기본 채널셋은 현행 3채널** → production(`compute_align_key_score_ensemble`) 불변,
  C4 는 opt-in. (`CONSENSUS_USE_ENSEMBLE` 와 동일한 무해-토글 방식.)

## 7. 계측 (Codex #2, #4)

`golden_consensus_eval_cond.py` / `align_similarity._gt_in_topk` 확장:
- **C4-only truth hit 의 miss-distance bin 분포**: C4 가 *새로* 들어오게 만든 정답을
  near / mid / far / veryfar 로 층화 보고. (집계 recall 단독 금지 — Codex #4.)
  far/veryfar 가 줄면 구조적 천장을 건드린 것, near 만 줄면 reshuffle.
- **S/E NCC 분포**: C4 채널의 true(S, GT 근처) vs false(E/decoy) NCC 점수 분포를
  로깅 → drift 로 true NCC 가 무너지는지 가시화(Codex #2 게이트 근거).
- **ablation 표**: `proposer ∈ {c1, c1c2c3, c1c2c3c4}` × {in_topk, rank1,
  miss-dist bins}. summary.json 에 `proposer`, `ensemble_channels`,
  `template_periodic_rate`, `c4_only_hits_by_bin` 키 추가.

## 8. 테스트

**Mac 합성(오피스 데이터 불요, [[feedback_no_office_data_to_mac]])**
- `repeating_grating` fixture: 주기 line/space + 한쪽 끝 unique 랜드마크. 기대:
  C1 은 grating 따라 다중 peak(truth rank≠1), **+C4 가 truth 를 top-k 로** 끌어옴.
- `contact_array` fixture: 동일 dot 격자 + 격자 *종단*. 기대: C4 도 array 내부는
  isolation 낮음(정직한 한계) → `template_periodic=True` 플래그 발화, 종단만 살림.
- `grey_area` fixture: 무특징 영역 → C4 가 헛 lock 없이 저신뢰(graceful).
- `coord_parity`: unique fixture 에서 C4.xy == C1.xy(±NMS) — §6.4 불변식.
- `rrf_representative`: 이종 score 클러스터에서 대표가 chamfer 최고/medoid 인지(§5).

**오피스 A/B**: `ENSEMBLE_CHANNELS` ablation 으로 §7 지표 비교.

## 9. 우선순위 컨텍스트 & 미해결 (Codex #5)

C4 는 **탐색 채널**이다. 더 강하거나 싼 레버가 먼저 검증되어 있거나 존재한다:
1. **consensus/template-bank 증강** — 문서상 유일하게 검증된 강한 레버
   (`in_topk +0.282`, 0.436→0.718). [[project_consensus_sparse_golden_and_recipe_id_collision]]
2. **ROI/context 마스킹** — `roi_hint`/`_window_roi` 이미 존재(탐색 공간 축소).
3. **autocorr periodicity suppression** — 본 spec Phase 1 이 그 첫 조각.
4. **C4-NCC** — 본 spec Phase 2, opt-in exploratory.

**미해결 리스크**
- C4 가 far/veryfar 를 못 살리면(stale/periodic 우세) §7 데이터로 *기각*하고 Phase 0/1
  만 남긴다(여전히 순이익). — [[project_ensemble_on_consensus_rejected]] 의 재판.
- `τ_p`(periodicity), `r_excl`, `τ_c`(chamfer 생존), per-channel weight 곡선은
  cold-start → 오피스 sweep 으로 보정(상수는 합성에서만 검증).

## 10. 개발 위치 & 변경 파일

**개발 위치 규약 (2026-06-10 사용자 확정)**: ensemble 개선은 **workflow_2 에서**
실험·측정한다(drivers: `golden_localization_eval_cond.py`,
`golden_consensus_eval_cond.py`). 검증된 개선만 **workflow_3/vision(production
엔진)으로 포팅**한다. 따라서 실험 단계에서는 workflow_3 을 건드리지 않는다.

- **실험 코드(workflow_2)** — `poc/workflow_2/ensemble_lab.py`(신규): 실험용 융합
  (`rrf_fuse` rescore 대표·§5), `template_periodicity`(§6.2), [Phase 2] `'context'`
  C4 채널·chamfer 생존 게이트(§6.3). 채널 solo·primitive 는 workflow_3 에서 import 재사용.
- **드라이버(workflow_2)** — `golden_consensus_eval_cond.py`/`golden_localization_eval_cond.py`:
  `ENSEMBLE_CHANNELS` 노출, ablation/계측·`template_periodic_rate`(§7). `align_similarity.py`
  `_propose_topk`/`_gt_in_topk` 가 lab 융합 호출 + C4-only hit bin 집계(§7).
- **테스트(workflow_2)** — `poc/workflow_2/test_ensemble_lab.py`(신규).
- **포팅 대상(workflow_3, 검증 후 별도 작업)** — `vision/ensemble_proposer.py`
  (`_rrf_fuse` rescore·`'context'` 채널·`template_periodicity`),
  `vision/align_key_matcher.py`(C4-only chamfer 게이트). production 매칭 경로
  (`align_fail_correct`)에 기본 on 전환은 그 다음 단계.
