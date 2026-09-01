---
status: synthesis
date: 2026-09-02
scope: 2026-09-02 병렬 리서치 4편(A 알고리즘 감사 / B 딥러닝 타당성 / C 데이터·라벨 감사 / D VLM·대안)의 종합. 코드/데이터 변경 없음.
inputs:
  - 2026-09-02-A-algorithm-audit.md
  - 2026-09-02-B-deep-learning-feasibility.md
  - 2026-09-02-C-training-data-audit.md
  - 2026-09-02-D-vlm-and-alternatives.md
supersedes-in-part: ../cv/align_fail_vlm_deep_learning_addendum_ko.md (2026-07-10) 의 실험 순서
---

# [DIGEST] 네 보고서가 독립적으로 같은 결론에 닿았다 - 잔여 실패의 87% 가 recall miss 이므로 pair-ranker/verifier 축은 접고, 학습은 **후보 생성(proposer)** 에 건다. 첫 모델은 tracker 계열 heatmap localizer(1점 라벨 그대로, 손실 하나가 recall+rank1 을 동시에 최적화). VLM 좌표 fine-tune 은 구조적으로 기각(양자화 ~5px/bin, 소형 타깃 8~40%). 그러나 착수 전 세 가지 게이트가 있다 - ① "수백만 장" 전제는 코드로 **미검증**(golden 298 recipe 중 S≥4 는 1개, MES 벌크를 옮기는 코드 없음) ② B 의 Step 0(무비용 버킷 재분류)이 "학습이 되는 문제인가"를 가른다 ③ A 가 찾은 벤치 결함(box offset scale 누락, scale band drift, top-8 cap, C3 중복 투표)을 고치지 않으면 어떤 A/B 도 읽을 수 없다.

---

## 1. 네 보고서가 합의한 것 (강한 합의)

| 결론 | A | B | C | D |
|---|---|---|---|---|
| pair-ranker / top-K verifier 는 잘못된 과녁 (상한 = 잔여 실패의 13%, SEM rank1 최대 +0.044) | ✓ §0 | ✓ §1.1 | ✓ §0 | ✓ 델타1 |
| 학습 투자는 proposer(후보 생성) 축 | ✓ §5 P0 | ✓ §2.3 | ✓ §4.5 | ✓ 3.2 |
| 형태는 dense heatmap (좌표 회귀 아님, 다중 가설·모호도 보존) | ✓ §5 | ✓ §1.3a | - | ✓ 3.2 |
| VLM 은 좌표를 내지 않는다 (feasibility/설명/분류에 한정) | - | ✓ §0 | - | ✓ §1 |
| 안전 경계 유지: CV 가 좌표 권한, 저신뢰는 engineer_review, 롤아웃 offline→shadow→router→prod | ✓ §5 | ✓ §0,§5.4 | - | ✓ 델타4 |
| split 은 recipe-disjoint + eqp/time holdout, 키는 leaf 가 아니라 `(eqp, class, recipe)` | ✓ §5 | ✓ §4.2 | ✓ §4.2 | - |
| S 라벨은 self-report 라 의심 대상; 최소 필터 = crosshair 존재 | - | ✓ §2.6 | ✓ §3.1 | - |
| 재등록 워크스트림은 학습과 **병행**이지 대체가 아니다 | - | ✓ §5.4-6 | - | ✓ 3.1 |

7월 문서(`align_fail_vlm_deep_learning_addendum_ko.md`)와 갈리는 지점은 B §6 표가 정확하다. 요지는 "무엇이 클릭을 결정하는가"는 안 바뀌고 "무엇을 먼저 시험하는가"만 바뀐다.

## 2. 보고서 간 긴장 - 조정이 필요한 다섯 지점

### 2.1 D 의 1순위(distinctiveness predictor) vs B 의 1순위(learned proposer)

경쟁이 아니라 **Step 0 의 결과가 비중을 정한다**. B §3 Step 0 는 consensus arm 의 SEM rank-1 오류를 `template_bank_lab.classify_winner` 로 correct / near_periodic / far_wrong 으로 재분류하는 무비용 실험이다.

- far_wrong 우세(ADR 0006 의 heatmap arm 에서는 0.456 vs near_periodic 0.052) → 정보는 있는데 현 매처가 못 보는 것 → **proposer 학습이 주축**, distinctiveness predictor 는 보조.
- near_periodic 우세 → 정보 부재 → **재등록 자동화(D 3.1)가 주축**, 학습 프로그램 중단.

단 B 자신이 단 조건: `classify_winner` 는 period=None 이면 전부 far_wrong 으로 떨어지므로 **period 추정 성공률을 같이 보고**해야 한다. 이게 빠지면 Step 0 결과는 무효다.

### 2.2 B 의 "drop-in" 주장 vs A 의 "top-8 cap + Canny rescore 가 학습 후보를 지운다"

B §2.3 은 `argmax R -> best_xy` 가 기존 좌표 계약과 같아 drop-in 이라 한다. A §4.1 #1-2 는 production engine 이 fused 후보를 **Canny Chamfer 로 재점수하고 앞 8개만** NCC/MIND/ECC 에 넘기므로 classical edge 로 약한 정답을 학습 proposer 가 새로 찾아도 downstream 이 다시 버린다고 확정한다.

**조정**: A 가 맞다. 실험 seam 은 `compute_ensemble_candidates` 교체가 아니라 A §5 의 `LearnedProposerResult -> candidate scorer(baseline arm / learned arm) -> calibrated result` 구조여야 하고 learned proposal_score 를 보존하는 arm 과 mode 별 threshold 재보정을 **한 묶음으로** A/B 한다. B 의 Step 2 통과선(SEM rank1 ≥0.73 등)은 이 seam 위에서 잰다.

### 2.3 crosshair 처리 - C "지워라" vs B "증강으로 건드리지 마라"

둘은 다른 얘기다. C §3.2 는 **학습 입력**에서 crosshair 를 inpaint 하지 않으면 모델이 crosshair 자체를 찾는 치팅을 배운다는 것이고 B §2.4 는 **증강**으로 crosshair 를 그리거나 지우지 말라는 것이다(제거 A/B -2%p, 가짜 lock).

**조정**: 학습 입력 msr S 프레임은 반드시 `clean_image` 로 crosshair 를 지운다(라벨 위치에 십자선이 있는 프레임으로 학습하면 결과가 무의미). 근거는 production 이 상대하는 paused fail 프레임에 crosshair 가 없다는 것(C §3.3: E 는 crosshair 0/182). -2%p 실험은 CV matcher rerank 맥락이라 학습에 이식할 근거가 아니다(C 도 같은 판단). 증강으로 crosshair 를 합성하지 않는다는 B 의 규칙은 그대로.

### 2.4 SSL 사전학습의 위치

C 는 "unlabeled S+E 전부 쓸 수 있으니 SSL 이 데이터 분포에 더 맞다"는 쪽으로 기울고 B 는 "라벨이 많으니 supervised 먼저, SSL 은 E 프레임 활용 수단으로 Step 3b" 로 둔다. D 는 둘을 짝으로 본다.

**조정**: B 의 순서를 채택한다. 단 조건부 - C §4.1 이 지적하듯 **fine-tune 용 라벨 물량은 MES 전체 S 커버리지에 달렸고 그 규모가 미확인**이다. 규모 probe(§4) 결과 S 라벨이 10^5 미만이면 SSL(3b)을 앞당긴다.

### 2.5 "수백만 장" 전제

A/B/D 는 이 전제 위에서 설계했고 C 만 이를 **코드 기준 미검증**으로 못박았다. golden 298 recipe 의 실측 분포는 fail-only 151 / 정확히 S=3 이 135 / S≥4 가 1 이며, MES 원본을 학습 파이프라인이 소비 가능한 형태로 옮기는 코드는 저장소에 없다(알람 트리거 저장과 consensus 캐시 두 좁은 통로뿐). `ALIGN_IMAGES_DIR` 과 MES 출력 경로 불일치도 그대로다.

**조정**: 이것이 전체 프로그램의 첫 병목이다. §4 의 오피스 작업 0번.

## 3. A 가 찾은 벤치/엔진 결함 - 학습과 무관하게 먼저 고칠 것

학습 A/B 이전에 **현재 headline 수치 자체가 production 을 대표하지 못하는** 원인들이다. 코드로 확정된 것만 적는다(A §3.3, §4.1).

| # | 결함 | 영향 | 위치 |
|---|---|---|---|
| 1 | golden `_localize` 가 box offset 에 scale 을 안 곱함 (production/KPI 는 곱함) | scale≠1 인 box arm 의 hit/rank 가 틀림 | `golden_localization_eval.py:322-341` |
| 2 | consensus KPI scale band `(0.6,0.75,0.85,1.0)` vs production `(0.7,0.85,1.0,1.2,1.4)` | 다른 search problem 을 재고 있음 | `align_similarity.py:48-51` |
| 3 | top-24 fused 를 만들고 top-8 만 rescore | 직접적 recall cap; rank 9-24 정답은 reranker 가 못 봄 | `engine.py:906-914` |
| 4 | C3 만 cross-scale NMS 없음 + RRF 가 같은 채널 중복 투표 허용 | orientation 중복 peak 과대가중 | `ensemble.py:145-202` |
| 5 | combined eligibility 가 recipe string 단위, 실제 cell 은 (recipe, modality) | 빠진 sibling modality 가 어려우면 routed 과대평가 | `golden_combined_eval_cond.py:492-572` |
| 6 | `result.candidates` 순서 = Chamfer 순, `best_xy` = NCC/MIND/ECC pick | 학습 A/B·debug 소비자가 `candidates[0]` 을 최종으로 오독 | `engine.py:951-967` |
| 7 | threshold 0.6053/0.4727 이 OM/SEM 혼합 756 S 에서 보정, mode 별 reranker 뒤에도 공통 적용 | mode 별 calibration 없음 | `engine.py:148-159` |
| 8 | engine docstring 이 Chamfer+ORB 0.6/0.4 를 주 알고리즘처럼 서술 (ORB 미계산) | 유지보수 mental model 오류 | `engine.py:1-14` |

1, 2, 5 는 벤치 정정이고 3, 4 는 production 에서 recall 을 막는 결함이다. 3 은 `policy.top_n` 을 24 로 올리는 A/B 만으로 즉시 잴 수 있다(NCC/MIND/ECC 비용 3배, 오프라인이라 무관).

## 4. 실행 순서

### 오피스 (모델 없음, 지금)

0. **규모 probe** (C §4.3 스펙): MES 접근 가능한 최대 트리 루트, recipe 수, measurement-event 당 이미지 수, dominant-modality S 히스토그램, `rcp_box_status` 분포, `target_xy=null` 비율, msr cond 의 `Magnification` 존재율. 이미지는 싣지 않고 경로만. 이 결과가 §2.4/§2.5 를 결정한다.
1. **Step 0 버킷 재분류** (B §3): consensus arm rank-1 오류에 `classify_winner` + period 추정 유효성. 학습 프로그램 착수/중단 판정.
2. **벤치 정정** (§3 의 1, 2, 5) 후 headline 재산출. 그리고 `top_n=24` A/B (§3 의 3).
3. E 프레임의 cond.txt `crosshair_xy` 가 실제로 `(-1,-1)` 인지 확인 (C §3.3 은 CV 검출 기반 수치라 cond 기반 재확인 필요).

### Mac (지금, 합성 데이터)

- **HITL 라벨 회수 배선** (D 3.3): `awaiting_engineer_ok` / `engineer_done` 시점에 엔지니어 행동(OK / 재등록 / 포기)과 재등록 시 새 key 위치를 기록. 비용 최저, 1·2 의 데이터 인프라.
- **DINOv2 zero-shot C5 채널** (B Step 1) 을 `ensemble_lab` 에 배선 (`_channel_solo_candidates` 동형, `ALIGN_LAB_ENSEMBLE_CHANNELS` 로 켬). 오피스 1회 실행용.
- **heatmap localizer 학습 파이프라인** 뼈대 (B §2.3): 입력 `(rcp box crop 또는 consensus 템플릿, clean msr S 프레임)`, 라벨 = crosshair Gaussian, Gaussian-weighted focal loss, 합성 smoke 테스트. 데이터 추출은 C §2.4 대로 `cond_for_image`+`cursor_to_image`, `cond_template_crop`+`cond_align_offset` **재사용**(재구현 금지).
- **phase-slip 라벨 필터** (B §2.6): 같은 recipe S crosshair 들의 consensus 정합 잔차가 격자 주기 정수배로 튀는 표본을 `audit` 집합으로 격리. 학습 착수의 전제 조건.

### 이후 (게이트 통과 시)

- Step 1 (오피스 하루): C5 solo recall@8, C1-3+C5 RRF recall@8 **및** NCC rerank 후 rank1. SEM recall@8 +5pp & rank1 무손실이면 통과, +2pp 미만이면 "자연영상 SSL 은 SEM 에 전이 안 됨" 으로 종결.
- Step 2 (2~4주): heatmap 모델. 통과선 SEM rank1 ≥0.73 / conversion(rank1/in_topk) ≥0.78 / OM 무손실 / **gt_in_topk=false 부분집합 회수율**을 주 근거. high-confidence off-target 증가 시 즉시 중단.
- Step 3 (조건부, 동시 진행 금지): 3a EfficientLoFTR fine-tune(배율 가정 3개 검증 선행) 또는 3b 도메인 SSL(Step 1 기각 시 먼저).
- 병행: distinctiveness predictor + 재등록 워크리스트 자동화 (D 3.1). Step 0 결과와 무관하게 필요하다(SEM 실패 recipe 104 중 84 가 비변별 키, 재등록 실측 rank1 +0.269).

## 5. 오피스에서만 답할 수 있는 질문

1. MES 원본 규모와 접근 경로 (§4-0).
2. msr cond.txt 에 `Magnification` 이 항상 있는가, 배율 비가 픽셀 스케일 비와 맞는가 (B §1.3c 가정 ①③ - `DEFAULT_SCALES` ±40% 탐색이 존재한다는 것 자체가 반증 신호).
3. E 프레임 crosshair 가 cond 기준으로도 부재인가.
4. 학습 데이터로더의 호스트 RAM 16GB 대응 (사전 절단 memmap/tar shard, worker ≤2, 또는 학습 전용 장비).
5. DINOv3 등 사전학습 가중치 반입 라이선스.

## 6. 기각/유예 확정 목록

VLM 좌표 fine-tune(D §1) / VLM-ROI(D 델타3, SEM 에서 oracle 상한 무용) / pair-ranker 재시도(전원) / MASt3R·DUSt3R(B, 평면 웨이퍼) / SAM·Grounding DINO(유지) / template bank 재시도(ADR 0006 유지) / 생성 모델로 등록 이미지 갱신(유지) / full 실패 원인 분류(D 2.3, 이진 gate 만 먼저) / Doppelgangers(rank-only, Step 3 이후).
