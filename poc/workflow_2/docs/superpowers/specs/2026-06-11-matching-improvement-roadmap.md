# Align-Key 매칭 알고리즘 개선 로드맵 (2026-06-11)

> 본 문서는 "rcp/msr 이미지로 align point 를 구하는 현재 두 방식 위에서 매칭을
> 더 개선할 방법"에 대한 **연구 종합 + 우선순위 플랜**이다. 코드 작업은 아직
> 없으며, 트랙 선택 후 `ensemble_lab` 에서 lab-first 검증 → workflow_3 포팅
> 순서를 따른다 (`CLAUDE.md` 의 workflow_2 → workflow_3 규약).
>
> 입력 소스: **Codex rescue (gpt-5-codex)** 1회 질의 + 자체 코드/문서 리서치
> (엔진 내부 맵 + eval 이력 맵). 두 결과를 하나의 우선순위 리스트로 병합했다.

---

## 0. 핵심 재정의 — 병목은 reranking 이 아니라 proposer recall

이번 리서치에서 가장 중요한 발견. "매칭을 개선한다"는 직관은 보통 **점수 재정렬
(reranking)** 로 향하지만, 데이터는 그 레버가 이미 죽었다고 말한다.

- **Truth-forced 실험이 결정적 증거.** ground-truth 위치에 템플릿을 정확히
  올려놓아도 best chamfer peak 이 **67% (153/229)** 엉뚱한 곳으로 흐른다
  (`reranker_ab_failure_analysis.md`). score surface 자체가 평평하다.
- **Reranker 두 개가 이미 명확한 부검과 함께 기각됨.**
  - MI reranker: lift **−0.013** (no lift). 이유가 구조적이라 중요 — **MI 는
    공간 비민감 global statistic.** pixel co-occurrence 가 같고 공간 배치만 다른
    두 crop 은 같은 MI 값을 갖는다 → registration 품질을 **원리적으로** 못 본다.
  - Contour/Hu-moments reranker: lift **−0.167** (baseline 아래로 추락). Otsu 가
    contrast drift 에 불안정 + Hu moment 가 너무 global/회전불변이라 미세 정합
    정보를 파괴.
- **Proposer membership 천장 = in_topk ≈ 0.594** (fail frame). 약 40% 는 truth 가
  애초에 candidate pool 에 못 들어온다 → 재정렬로 절대 복구 불가.

### 결론: 진짜 레버는 3개뿐

| 레버 | 의미 | 현재 상태 |
|------|------|-----------|
| **L1. 템플릿 변별력 ↑** | 평평한 score surface 를 붕괴시켜 옳은 peak 가 이기게 | 미착수 (본 로드맵 Tier 1·2 핵심) |
| **L2. 탐색 공간 ↓** | proposer 가 건초더미에서 바늘 찾지 않게 | peak-isolation router + VLM-region (부분 착수) |
| **L3. 현재 공정 외형 반영** | stale rcp 대신 최근 success 외형으로 매칭 | **consensus, +0.44 in_topk 검증 완료, 프로덕션화 중** |

Codex 의 8개 아이디어, 자체 리서치 아이디어 모두 이 3개 레버로 환원된다.
"점수를 **재계산**하려 하지 말고 변별력 있는 candidate 를 **생성**하라"가 관통선이다.

---

## 1. 현재 엔진 / eval 상태 (근거 스냅샷)

### 1.1 매칭 엔진 (`poc/workflow_3/vision/`)

- **Chamfer (주 geometric signal)**: edge → distance transform → `matchTemplate(TM_CCORR)`
  → `score = exp(-mean_dt / DT_TAU_PX)`, `DT_TAU_PX=10.0`. pixel intensity drift 에 불변.
- **ORB**: Lowe 0.75 + RANSAC(5px) inlier ratio. **검증용만**, primary searcher 아님.
- **Ensemble proposer (`ensemble_proposer.py`)**: C1 Canny / C2 Scharr / C3
  orientation-binned chamfer → **RRF fusion** (k0=10, Chebyshev ≤8px 군집) → top-N.
- **NCC reranker**: top-N pool 안에서만 `0.5·chamfer + 0.5·max(0,ncc)` 로 선택.
  full-frame primary 금지 (texture 에 lock 됨).
- **second_ratio (= peak-isolation)**: `chamfer_2nd / chamfer_best`. 현재
  `distinctive` bool 로 과적재되어 노출.
- **Threshold 현황**: ensemble 0.6053(Youden)/0.4727(recall@95) 는 golden(n=756)
  로 calibrated. 반면 `STRUCTURE_POLICY`(0.62/0.40), `min_distinct_gap=0.04`,
  `max_second_ratio=0.94`, `MIN_CONFIRM_SCALE=0.6` 는 **cold-start 가설값, 미보정.**

### 1.2 eval 이력 (`poc/workflow_2/docs/`)

| 지표 | baseline | best | 출처 |
|------|----------|------|------|
| in_topk (proposer recall) | 0.434 (rcp) | **0.876** (consensus LOO) | 260608 |
| rank1 | 0.269 (LOO) | **0.764** (consensus LOO) | 260608 |
| gt_in_topk 천장 | 0.594 | proposer 한계 | 260602 |
| ensemble recall@8 | 0.557 | **0.698** (+0.141) | 260609 |
| MI rerank lift | — | −0.013 (기각) | 260602 |
| Contour rerank lift | — | −0.167 (기각) | 260602 |
| peak-isolation AUC | — | **0.9108** (tau\*≈0.9825, recall 97% @ flag 34%) | ensemble_lab |
| truth-forced wrong_local_peak | — | **67%** | 260602 |

- **VLM vs OpenCV 설계 규칙 (불변)**: *"좌표는 CV 가 결정한다. VLM 은 영역 식별·
  애매 상황 설명·feasibility 에만."* low CV score 를 VLM 이 override 금지. E 이미지
  직접 좌표 VLM 금지.

---

## 2. 통합 우선순위 플랜 (Codex + 자체 리서치 병합)

각 항목에 레버(L1/L2/L3)와 출처(Codex # / 자체)를 표기.

### Tier 0 — 저비용, 이번 주: 신규 CV 없음, wiring + calibration 만

| # | 액션 | 레버 | 출처 |
|---|------|------|------|
| 0.1 | **`second_ratio` 를 notify-flag → router 로 승격.** `key_visibility_gate()` 가 bool 대신 `act / fallback_search / vlm_region / engineer_review` 반환. 검증된 tau\*≈0.98 사용. 평평한 surface 에서 **확신 오정렬 대신 abstain** 가능해져 이후 모든 항목을 배포 가능하게 만드는 안전망. | L2 | Codex #1 + peak-isolation port spec |
| 0.2 | **cold-start threshold 를 modality 별 재보정.** `STRUCTURE_POLICY`, `min_distinct_gap`, `max_second_ratio`, `MIN_CONFIRM_SCALE` 를 golden label + AUC-0.91 signal 로 OM/SEM 분리 fit. | L1 | 자체 (엔진 맵에서 미보정 플래그) |

### Tier 1 — 근본 원인 최대 payoff: 평평한 surface 죽이기

| # | 액션 | 레버 | 출처 |
|---|------|------|------|
| 1.1 | **White-box unique-region 템플릿 cropping.** 등록 시 rcp 템플릿을 `cond.box_ltrb`(엔지니어가 표시한 unique area)로 crop → 그걸 매칭 → `align_offset` 으로 center 복원. 흰 박스는 주기적 field 의 모호성을 깨라고 엔지니어가 찍어준 **국소 고유 영역**이며, `cond_file.load_cond()` 가 이미 `box_ltrb` 를 파싱한다. `ensemble_lab` 에서 far/very-far miss bin 별로 검증. | L1 | 자체 (Codex #3 가 닿지만 본 로드맵은 headline 으로 승격) |
| 1.2 | **Anchor-relative matching (주기적 key).** 1.1 일반화 — key 가 주기적 field 에 있으면 근처 **globally-unique landmark**(쉽고 sharp 한 peak)를 매칭 후 알려진 geometric offset 적용. 건초더미 탐색 자체를 우회. | L1+L2 | 자체 |
| 1.3 | **등록 시 periodicity gate.** 기존 `ensemble_lab.template_periodicity()`(FFT autocorr) 재사용해 score surface 가 평평할 key 를 **사전 flag**. periodic key → context-expansion(1.1/1.2) 강제 또는 bank(Tier 3) 로 라우팅. | L2(routing) | 자체 (Codex #5 의 FFT primitive 를 quality gate 로 용도 전환) |

### Tier 2 — 구조적 miss 절반 복구 (proposer recall)

| # | 액션 | 레버 | 출처 |
|---|------|------|------|
| 2.1 | **Constellation / geometric-layout verifier.** top-N 을 단일 patch 유사도가 아니라 **여러 feature 의 상대 기하**(mutual-NN + spatial-layout consistency)로 재정렬. 이것이 MI/Hu-moments 가 **원리적으로 못 본** 공간 배치 축을 보는 reranker. | L1 | 자체 (기각된 두 reranker 가 *왜* 실패했는지에 직접 답) |
| 2.2 | **Descriptor-channel 확장(검증용)**: AKAZE/SIFT geometry, phase-congruency edge 를 top-N 에만 — full-frame primary 절대 금지. lab-first, far/very-far bin 움직이는 채널만 포팅. | L1 | Codex #4 |
| 2.3 | **Fourier-Mellin / log-polar phase correlation** 으로 broad zoom-out scan 의 scale+rotation 처리. candidate 생성에만 bound, 최종 권한 아님. | L2 | Codex #5 |

### Tier 3 — 어려운 잔여 + 장기 베팅

| # | 액션 | 레버 | 출처 |
|---|------|------|------|
| 3.1 | **VLM-region → CV fine-coord**, Tier-0 router 의 `second_ratio` 모호도로 gating. `vlm_align_key_region.py` 가 올바른 contract(VLM=ROI 제안, CV=좌표 소유) 이미 보유. 구조적 절반의 채택된 escalation 경로. | L2 | Codex #6 + 설계 규칙 |
| 3.2 | **Multi-template bank.** periodicity gate(1.3)가 본질적 모호로 flag 한 recipe 는 K anchor 유지 → live-confirm/VLM 이 disambiguate, 하나로 조기 commit 안 함. | L1 | Codex #2·#7 |
| 3.3 | **소형 Siamese/metric descriptor** — bank+gate 가 천장 친 *후에만*, label 허용 시. top-N reranker, recipe-split validation. | L1 | Codex #7 |

### 양측 명시적 보류

- **Learned full-frame feature + 모든 NCC/pixel-identity reranker 는 보류.**
  도메인 제약(geometry > pixel)이 이미 C4 `ncc_ratio_top2` 의 AUC 를 0.72(vs 0.91)로
  깎았고, classical 레버가 소진되기 전 CNN 을 정당화할 만큼 label 이 충분치 않다.

---

## 3. Codex vs 자체 리서치 — 차이 요약

- **공통**: peak-isolation gate 를 먼저(Tier 0). reranking 단독 무용. VLM=region only.
  learned/NCC 는 후순위.
- **강조 차이**: Codex 는 peak-isolation gate 후 white-box/cond ROI 를 template-bank
  의 medium 의존 항목으로 둠. 자체 리서치는 **white-box cue 를 최고 근본 레버로 보고
  큐를 점프시킴** — 이미 보유한 데이터(`cond.box_ltrb`)로 평평한 surface 를 직접 공격
  하는 유일한 항목이기 때문.
- **자체 고유 추가**: anchor-relative matching(1.2), periodicity gate 용도 전환(1.3),
  constellation verifier(2.1, 기각된 reranker 실패 축에 정확히 대응).

---

## 3.5. 진행 상황

- **Tier 0.1 — 게이트 라우터 승격: ✅ 완료 (2026-06-11, TDD).**
  `key_visibility_gate()` 가 bool → route intent(`act`/`fallback_search`/`engineer_review`)로
  승격. present 라도 `second_ratio > reregister tau` 면 auto-act 대신 `engineer_review` 로
  보류해 평평한 surface 에서의 확신 오정렬+오클릭을 차단한다. `reregister_ratio_threshold`
  None(기본)이면 과거 act/fallback 2분기를 byte 동일하게 보존(opt-in). 운영 루프(`cycle.py`)는
  `Workflow3Settings.reregister_second_ratio_threshold`(0.98)를 `CorrectionConfig` 로 주입해
  게이트를 활성화한다. 모호 보류는 `escalated_ambiguous_key` status → notify 가 cube 로 알리고
  `재등록 권장(모호 키)` 한 줄을 싣는다. 변경: `align_fail_correct.py`, `config`(기존 필드 재사용),
  `cycle.py`; 테스트 `test_align_fail_correct.py` 10/10(gate 라우팅 10케이스 + engineer_review
  caller 통합), 회귀 ensemble/notify 그린.
- **Tier 0.2 — modality 별 threshold 재보정: ⏸ office-data-gated.**
  `STRUCTURE_POLICY`/`min_distinct_gap`/`max_second_ratio`/`MIN_CONFIRM_SCALE` 의 실제 fit 은
  fab golden 데이터가 필요(Mac 반입 불가). plumbing/harness 는 별도 착수 시 blind 작성 →
  오피스 실행으로 숫자 확보.
- **Tier 1.1 — box-crop localization 검증: ✅ GREEN-LIGHT (2026-06-11, office run).**
  `golden_localization_eval_cond.py` bin×arm 게이트 결과 — box-crop(cond.box_ltrb + decoupled
  offset)이 center-crop 대비 **모든 displacement bin 에서 gt_in_topk/rank1 동반 상승**(회귀 0):
  near +0.110/+0.159, mid +0.130/+0.160, far +0.085/+0.176 (veryfar 표본 0). 가설(far/veryfar
  구조적 rescue)보다 강함 — **균일한 template 변별력 향상**이라 displacement 라우팅 없이
  **무조건 포팅**. production headline = rank1 **+0.16~0.18**(올바른 reposition 비율). 검증
  설계 `specs/2026-06-11-box-crop-localization-validation-design.md`, 구현 플랜
  `plans/2026-06-11-box-crop-localization-validation.md` (lab 리포팅 4 tasks, 37 tests green).
  → **production 포팅 spec 진행**. 확인 필요: per-bin n(center/box) 커버리지(box 는 cond box
  valid recipe 한정, center-crop fallback).

## 4. 다음 단계

트랙은 서로 독립적이라 어느 하나부터 시작 가능. 권장 순서는
**Tier 0 → Tier 1.1** (저비용 안전망 먼저 깔고, 최고 payoff 근본 수정).
실제 착수 시 `poc/workflow_2/ensemble_lab.py` lab-first → `golden_localization_eval_cond.py`
로 far/very-far miss bin 별 lift 확인 → workflow_3 포팅.

### 참고 파일

- 엔진: `poc/workflow_3/vision/align_key_matcher.py`, `ensemble_proposer.py`,
  `live_align_search.py`, `align_fail_correct.py`, `clean_align_image.py`, `cond_file.py`
- lab/eval: `poc/workflow_2/ensemble_lab.py`, `golden_localization_eval_cond.py`,
  `golden_consensus_eval_cond.py`, `proposer_recall_ab.py`, `align_similarity.py`,
  `vlm_align_key_region.py`
- 근거 문서: `poc/workflow_2/docs/study/reranker_ab_failure_analysis.md`,
  `docs/journals/260608/260608_163302_*.md`, `docs/journals/260609/*`,
  `specs/2026-06-10-peak-isolation-reregister-notify-design.md`,
  `specs/2026-06-09-consensus-productization-design.md`
