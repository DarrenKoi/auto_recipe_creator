[DIGEST] 학습은 타당하다. 단 "후보 순위"가 아니라 "후보 생성"을 학습해야 하고(잔여 실패의 87%가 recall miss), cond.txt 수백만 라벨은 1점 라벨로 쓰지 말고 tracker 계열 heatmap 감독(argmax = rank-1)으로 써야 우리가 배포하는 지표를 직접 최적화한다. 다만 ADR 0006 의 SEM 버킷(near_periodic 0.052 vs correct 0.492, 나머지 ~0.46 이 far_wrong)은 "SEM 실패 = 주기성 정보 부재" 라는 통념을 지지하지 않으므로, 모델을 만들기 전에 그 버킷을 consensus arm 에서 재확인하는 무비용 Step 0 로 프로그램 착수 여부를 가른다.

---

status: research
date: 2026-09-02
scope: office-only offline feasibility study. 코드/데이터 변경 없음
supersedes-in-part: cv/align_fail_vlm_deep_learning_addendum_ko.md (2026-07-10) 의 우선순위
related:
  - cv/align_fail_vlm_deep_learning_addendum_ko.md
  - cv/align_fail_cv_methods_research_ko.md
  - adr/0005-whitebox-box-crop-consensus-arm-rejected.md
  - adr/0006-template-bank-matcher-rejected-fusion-exhausted.md
  - reranker_ab_failure_analysis.md
  - ../../../../docs/project_progress/03_workflow_2.md

---

# 브리프 B: 딥러닝 기반 align-key localization 타당성 연구

## 0. 이 문서의 위치 - 7월 문서 대비 델타만 쓴다

2026-07-10 의 두 문서(`align_fail_cv_methods_research_ko.md`, `align_fail_vlm_deep_learning_addendum_ko.md`)는
방법 카탈로그로서 여전히 유효하다. 이 문서는 그것을 반복하지 않고, 그 뒤 확정된 사실 두 가지 위에서
우선순위와 학습 설계를 다시 계산한다.

| 항목 | 2026-07-10 판단 | 2026-09-02 판단 | 바뀐 이유 |
|---|---|---|---|
| P0-E patch-pair ranker | 가장 현실적인 첫 학습 실험 | **강등**. 첫 실험 아님 | rank-only 방법인데 잔여 실패의 87%가 후보에 정답이 없는 recall miss (03 문서 §7). 과녁이 틀렸다 |
| P1-F frozen DINOv2 | pair-ranker 의 feature(R3)로 제한 | **승격**. 독립 proposer 채널로 zero-shot 측정 | 학습 없이 recall 축을 건드리는 유일한 후보. `ensemble_lab` 4번째 채널 = 하루짜리 실험 |
| P1-G learned matcher | "CV top-K crop pair 안에서만" 검증 (common gate 1) | **그 게이트를 푼다**. full-frame proposer 로 평가 | top-K 안에서만 보면 정의상 recall miss 를 못 고친다. 7월 게이트는 87% 확정 전에 세운 것 |
| P3-I self-supervised pretrain | 마지막 순위(라벨 부족 가정) | **승격**. 3단계 본선 | 라벨이 수백만 장이면 "라벨이 적어서 SSL" 이 아니라 "라벨이 많아서 supervised 가 된다". SSL 은 E frame 활용 축으로 성격이 바뀐다 |
| 학습 감독 신호 | golden 수백 장의 독립 target point | **cond.txt 수백만 + dense 로 확장 가능** | §1.3 |
| MASt3R/DUSt3R | 미언급 | **기각** | 평면 웨이퍼는 3D grounding 이 퇴화. 외부 벤치에서도 protocol 민감·불안정 |

7월 문서의 안전 경계는 그대로 유지한다. 모델은 좌표 권한을 갖지 않고, 실패/저신뢰는 full-FOV CV
또는 `engineer_review` 로 결정론적으로 되돌아가며, live broad-search 에 무거운 모델을 넣지 않는다.
이 문서는 무엇을 먼저 시험하는가를 바꿀 뿐, 무엇이 클릭을 결정하는가는 건드리지 않는다.

---

## 1. 두 델타가 실제로 바꾸는 것

### 1.1 87% recall miss - rank-only 방법은 전부 잘못된 과녁이다

`docs/project_progress/03_workflow_2.md` §7: 모드별 rerank 반영 후 남은 실패의 약 87%가 "정답이
애초에 후보 목록에 없어서 어떤 재순위로도 고칠 수 없는" 경우다.

이 한 줄이 7월 문서의 실험 목록 대부분을 무효화한다. 다음은 전부 이미 뽑힌 top-K 안에서만 동작한다.

- P0-A ECC, P0-B SIFT/AKAZE-RANSAC, P1-C phase correlation, P1-D MIND verifier (CV 문서)
- P0-E patch-pair ranker (R0~R3), P1-B VLM abstention gate (DL 문서)
- P1-G 의 공통 게이트 1번 ("full frame global search 가 아니라 CV top-K crop pair 에서 먼저 검증")

이들의 이론적 상한은 잔여 실패의 13%다. SEM rank-1 0.665 기준으로 잔여 실패 0.335 중 13%
= 최대 +0.044. 그마저 완벽한 reranker 를 가정한 값이고, MI/contour 두 계열이 독립적으로
음수를 낸 전례가 있다(`reranker_ab_failure_analysis.md` §4: MI -0.013, contour -0.167).

**따라서 학습을 도입한다면 proposer(후보 생성) 축이어야 한다.** 이것이 이번 연구의 1차 결론이다.

### 1.2 그러나 recall 과 rank-1 은 시소다 - 새 지표를 먼저 정의해야 한다

여기서 멈추면 ADR 0006 이 이미 밟은 함정을 다시 밟는다. 같은 데이터를 재해석하면:

| arm | SEM in_topk | SEM rank-1 | 변환율(rank1/in_topk) | 출처 |
|---|---:|---:|---:|---|
| consensus (production) | 0.849 | (03 문서 0.665) | ~0.78 | ADR 0006, 03 문서 §6 |
| template bank heatmap | 0.943 | 0.492 | 0.52 | ADR 0006 |
| template bank RRF | 0.916 | 0.524 | 0.57 | ADR 0006 |

recall 을 +9.4pp 되찾자 rank-1 이 -17pp 무너졌다. 되찾은 후보들은 시스템이 순위를 못 매기는
후보였다. recall miss 를 고치면 된다는 명제는 참이지만 불완전하다. 정확히는:

> 되찾은 후보에 변별력 있는 점수까지 함께 붙이는 방법만 유효하다.

ADR 0006 의 방법론 교훈이 이것이다: "bench A/B 는 rank-1 로 비교할 것. in_topk 는 천장이고 rank-1 이
배포 산출물 - high in_topk / low rank-1 은 recall 착시다."

**이 문서가 제안하는 판정 지표**: 어떤 새 arm도 다음 두 값을 함께 보고한다.

1. `recall@8` (proposer 성능, 기존 `proposer_recall_ab.py` 정의)
2. `rank1` (동일 후보집합에 표준 NCC rerank 를 적용한 뒤) - 착시 방지
3. 파생: `conversion = rank1 / in_topk`. consensus SEM 의 0.78 보다 낮으면 **recall 이 올라도 기각**

이 지표 정의가 §3 실험 계획의 성공 기준이 된다.

### 1.3 수백만 cond.txt - 이건 "1점 라벨"이 아니다

브리프는 "rcp 박스 중심 <-> msr crosshair 대응 = 1점 대응만 있음. 이것으로 학습 가능한 loss 는?"
이라고 묻는다. 답은 세 층위이며 세 번째가 이 연구의 핵심 발견이다.

#### (a) 1점 그대로 쓰는 법: heatmap 감독 (가장 안전, 가정 없음)

template crop `T` 와 search 이미지 `F` 를 받아 `F` 위의 dense response map `R` 을 내는 모델을 둔다.
정답 crosshair 위치에 Gaussian 을 놓은 것을 라벨로 per-pixel loss 를 건다. 이것이 SiamFC 이후
tracking 계열의 표준 감독이며 정확히 1점만 필요하다. 손실은 logistic(SiamFC) 또는 Gaussian-weighted
focal loss(OSTrack 계열).

이 형태는 §1.2 의 시소를 구조적으로 회피한다. `argmax R` 이 rank-1 이고
`top-K peaks of R` 이 후보 집합이므로 **하나의 손실이 recall 과 rank-1 을 동시에 최적화한다.**
proposer 와 scorer 를 따로 만들어 붙이는 현재 구조(chamfer 후보 -> NCC rerank)가 시소를 만드는
원인인데, heatmap 구조에는 그 이음매가 없다.

#### (b) hard negative 를 명시적으로 쓰는 법: contrastive (7월 P0-E 와 동일, 강등)

같은 프레임의 decoy peak 를 negative 로 삼는 supervised contrastive. rank-only 라 §1.1 로 강등.
단 (a) 의 보조 손실로는 유효하다(응답맵의 2등 봉우리를 눌러 `second_ratio` 를 벌린다).

#### (c) 1점을 dense 대응으로 승격하는 법 - LoFTR/RoMa fine-tune 을 여는 열쇠

LoFTR 계열의 학습에는 dense 대응 라벨이 필요하다. EfficientLoFTR 은 coarse 단에서 depth map 과
camera pose 로 A 의 격자점을 B 로 보낸 대응을, fine 단에서 sub-pixel 오프셋 회귀를 감독한다
([Efficient LoFTR](https://arxiv.org/html/2403.04765v2)). 자연영상에서는 depth+pose 가 있어야 하지만
우리 문제는 그 자리에 훨씬 강한 사전지식이 있다.

rcp 템플릿과 msr 프레임의 관계는 (회전 없는) 2D similarity 다. 평행이동 `(tx, ty)` 와 배율 `s` 두
자유도뿐이므로 1점 대응 + 배율 하나면 템플릿 전 픽셀의 dense 대응이 해석적으로 결정된다:

```
p_F = s * (p_T - c_T) + c_F
      c_T = 템플릿의 align point (rcp box center + align_offset)
      c_F = msr crosshair (cond.txt)
      s   = msr_magnification / rcp_magnification
```

배율은 cond.txt 에 있다. `cond_file.py:178` 이 `raw.get("magnification")` 을 읽고
`grid_search.registered_magnification` 이 rcp cond 에서 등록 배율을 뽑는다
(`poc/workflow_3/align/cond_file.py:155-178`, `align/correction.py:580-584`).

> **가정이며 오피스 확인이 필요하다(추정)**: ① msr 쪽 cond.txt 에도 `Magnification` 이 항상 있는가
> (현재 코드는 modality 추론의 *보조* 신호로만 쓰므로 결측 가능성이 있다), ② stage 회전이 정말 0인가,
> ③ 배율 비가 실제 픽셀 스케일 비와 일치하는가. ③이 특히 의심스러운 이유는 production 엔진이
> `DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)` 로 ±40% 스케일 밴드를 탐색한다는 것이다
> (`align/matching/engine.py:47`) - 배율을 신뢰했다면 그 탐색이 필요 없다. s 는 cond 로 초기값을
> 잡되 pair 별로 정밀화해야 할 값으로 봐야 한다.

가정 ③이 깨져도 회피로가 있다: 현재 매처를 돌려 고득점(NCC > Youden match 임계 0.6053)으로 수렴한
pair 에서만 `s` 를 추정해 dense 라벨을 만드는 self-training. 수백만 장이 있으므로 엄격한 필터로
90%를 버려도 수십만 pair 가 남는다. 라벨이 많다는 말의 진짜 가치가 이것이다. 품질 필터를 사치스럽게 걸 수 있다.

#### 그래서 (a) 와 (c) 중 무엇을 먼저 하나

**(a) 먼저.** (c) 는 배율 가정에 의존하고 사전학습 가중치의 도메인 갭까지 얹지만, (a) 는 가정이 0이고
구조적으로 rank-1 을 최적화하며 모델이 작다. (c) 는 §3 의 3단계에 둔다.

---

## 2. 후보별 판정

각 후보를 원리 / 우리 데이터로 학습하는 법 / OM vs SEM 강약 / 추론 비용 / 근거로 나눈다.

추론 비용의 공통 전제부터 정리한다. 지연시간은 제약이 아니다. 알람은 분 단위로 오고 사이클당 추론은
1회이며 보정 사이클은 이미 VLM 호출로 수 초를 쓴다. 진짜 제약은 두 가지다.

- 호스트 RAM 16GB (H200 140GB x2 이지만 binding constraint 는 GPU 가 아니라 프로세스 수;
  `deploy_vlms/` 배치 기준, 2026-08-11 에 ui-venus/ui-tars/got-ocr 를 이 이유로 내렸다). 새 상주
  서비스 프로세스를 추가하려면 모델 하나를 내리는 것과 교환해야 한다.
- 학습 시 데이터로더의 호스트 RAM. 수백만 JPEG 을 도는 num_workers 다중 프로세스는 16GB 를 바로
  먹는다. 완화책: 고정 크기 uint8 타일로 사전 절단해 memmap 또는 tar shard 로 굽고 worker 2개 이하.
  또는 학습만 별도 장비. 이건 알고리즘 리스크가 아니다. 일정 리스크로 계상해야 한다.

### 2.1 학습 기반 dense/semi-dense matcher (LoFTR, EfficientLoFTR, RoMa, DKM, LightGlue)

**원리.** detector-free matcher 는 keypoint 검출 단계를 없애고 두 이미지의 feature grid 사이에
self/cross-attention 을 반복해 대응을 직접 낸다. LoFTR 가 이 패러다임을 열었고
([LoFTR, CVPR 2021](https://arxiv.org/pdf/2104.00680)), EfficientLoFTR 이 같은 정확도를 대폭 싼 값에
낸다([Efficient LoFTR](https://arxiv.org/html/2403.04765v2)). RoMa 계열은 dense warp + confidence 를
회귀하며 DINOv2/DINOv3 backbone 을 쓴다([RoMa](https://arxiv.org/pdf/2305.15404),
[RoMa v2, 2025](https://arxiv.org/abs/2511.15706)). LightGlue 는 sparse 축의 최적화된 matcher 다
([LightGlue, ICCV 2023](https://arxiv.org/abs/2306.13643)).

**우리 데이터로 학습하는 법.** §1.3(c) 의 dense 승격이 성립하면 원 논문의 감독을 그대로 쓸 수 있다:
coarse 단은 템플릿 격자점 -> 프레임 격자셀의 assignment(focal loss), fine 단은 sub-pixel 회귀(L2).
입력 쌍은 `(rcp 또는 consensus 템플릿, msr S 프레임)`. 회전이 없으므로 rotation augmentation 은
금지(7월 문서와 동일 판단).

**강점/약점 (OM vs SEM 분리).**
- OM: 이미 rank-1 0.852 라 얻을 것이 적다. OM 에는 쓰지 않는 것을 기본으로 한다.
- SEM: 여기가 가설의 전부다. transformer 는 self-attention 으로 FOV 전역을 본다. 이것이 chamfer/NCC 가
  구조적으로 못 하는 일이고, ADR 0005 가 간접 증거를 준다(§5.2).
- 약점 1: 자연영상 사전학습의 도메인 갭. 다만 반증 증거가 있다 - SAR-optical 위성 정합에서
  cross-modal 학습을 전혀 하지 않은 RoMa 가 평균 3.0px 로 XoFTR 과 공동 1위였다
  ([Are Pretrained Image Matchers Good Enough for SAR-Optical Registration?](https://arxiv.org/html/2604.10217v4)).
  SAR 은 SEM 만큼이나 자연영상이 아니다. 반대 방향 증거도 있다: EM 도메인에서는 자연영상 사전학습이
  도메인 특화 SSL 에 밀린다는 결과가 반복 보고된다
  ([SEM foundation model, MoE+MAE, 2026](https://arxiv.org/html/2604.05960v1)). zero-shot 은 걸어볼
  만하고 fine-tune 은 아마 필요하다.
- 약점 2: 반복 구조에서 many-to-one 대응. 7월 문서가 지적한 그대로이며 여전히 유효한 우려다.

**추론 비용.** EfficientLoFTR 최적화 모델 27.0ms(mixed precision) / 640x480, 1200x1200 에서 35.6ms(FP32),
RTX 3090 기준. LoFTR 대비 ~2.5x, RoMa 대비 ~7.5x 빠름(RoMa 302.7ms -> 40.1ms @640x480).
H200 에서는 더 빠르다. 연산 비용은 무시 가능하고 문제는 상주 프로세스 1개다.

**판정: 3단계 본선 후보(SEM 전용).** 단 §1.3(c) 의 배율 가정 검증이 선행 조건.

### 2.2 Foundation feature 기반 template matching (DINOv2 / DINOv3, SAM)

**원리.** self-supervised ViT 의 patch token 은 별도 학습 없이 patch 대 patch 대응에 쓸 수 있다.
템플릿 patch 토큰과 프레임 patch 토큰의 cosine 상관맵을 만들고 peak 를 후보로 낸다. DINOv3 는 dense
locality 를 명시적으로 강화했고 zero-shot 대응에서 DINOv2 를 전 해상도에서 앞선다(960x960 에서
PCK@0.1 58.4%)([DINOv3 정리](https://www.emergentmind.com/topics/self-supervised-vision-transformers-dinov3)).
산업 텍스처에서 frozen DINOv2 patch feature 가 실제로 통한다는 사례도 있다
([AnomalyDINO](https://arxiv.org/abs/2405.14529)).

**우리 데이터로 학습하는 법.** 학습하지 않는 것이 요점이다. 1단계는 zero-shot 상관맵을
`ensemble_lab` 의 4번째 채널로 넣는다. 배선 비용이 거의 0이다 - 채널 하나는 `_Cand(xy, score, scale)`
리스트를 반환하기만 하면 RRF 융합에 그대로 들어간다(`poc/workflow_3/align/matching/ensemble.py:145-174`).
효과가 보이면 그때 LoRA 또는 마지막 블록만 fine-tune 한다.

**강점/약점.**
- 강점: 유일하게 학습 0으로 recall 축을 건드리는 후보. patch stride(14 또는 16)가 크므로 좌표 정밀도는
  낮지만 우리 판정 tolerance 가 템플릿 short side 의 20%(`align_similarity.py:88`, `GT_TOL_NORM=0.20`)
  라 proposer 로서는 충분하다. 정밀 좌표는 기존 NCC/ECC 에 넘기면 된다.
- 약점: patch 토큰은 의미(semantic) 표현이라 이 라인과 저 라인은 같은 종류라고 말한다. 주기 구조의
  phase 를 가르는 데 필요한 것은 정반대의 성질이다. SEM 에서 상관맵이 통째로 평평할 위험이 실재한다.
  이건 1단계 실험이 싸게 반증해 줄 가설이다.
- SAM/SAM2: **기각 유지**(7월 판단과 동일). segmentation boundary 는 align target 을 뜻하지 않는다.
  추가 근거를 찾지 못했고 새로 밀 이유도 없다.

**추론 비용.** ViT-B/14 한 장 forward 는 H200 급에서 수십 ms 수준(추정). 오프라인 벤치에서는 무관.
production 도입 시엔 §2 서두의 프로세스 제약이 그대로 걸린다.

**판정: 1단계 실험으로 채택.** 가장 싸고 가장 빨리 틀릴 수 있다.

### 2.3 Siamese / one-shot detection / tracking heatmap - **이 연구의 추천안**

**원리.** "이 crop 을 저 이미지에서 찾아라"는 정확히 SiamFC 이후 single object tracking 의 정의다.
템플릿과 search 영역을 공유 encoder 에 넣고 cross-correlation 으로 단일 채널 response map 을 만들며
학습은 정답 위치의 Gaussian label 에 건다. OS2D 는 이것을 학습에 없던 클래스로 일반화시킨다:
dense correlation matching + feed-forward 기하 변환 + bilinear resampling 을 end-to-end 미분 가능하게
묶어 치약으로 학습하지 않고도 치약을 찾는다([OS2D, ECCV 2020](https://arxiv.org/abs/2003.06800)).

이 unseen class 일반화가 우리에게 결정적인 이유는 이렇다. production 은 학습에 없던 recipe 를 상대한다.
recipe = class 로 보면 OS2D 의 평가 프로토콜(train/test class 비중첩)이 우리의 recipe-disjoint split 과
정확히 같은 조건이다. 다른 후보들에는 이 대응이 없다.

**우리 데이터로 학습하는 법.**
- 입력 쌍: `T` = rcp 등록 이미지의 cond box crop (또는 consensus 템플릿), `F` = msr S 프레임 전체.
- 라벨: cond.txt crosshair 1점 -> Gaussian heatmap. 이것뿐이다. 배율 가정 불필요.
- 손실: Gaussian-weighted focal loss (OSTrack 계열) + 선택적 sub-pixel 오프셋 회귀 head.
- 보조 손실: §1.3(b) 의 contrastive 로 2등 봉우리 억제.
- 출력 계약: `argmax R` -> `best_xy`. 기존 chamfer 후보와 같은 좌표계, 같은 의미이므로
  `align_offset_xy * scale` 계약을 그대로 지킨다. drop-in 이다.

**데이터 규모의 타당성.** SOTA tracker 는 LaSOT(1,400 비디오 / 352만 프레임), TrackingNet(train 30,130
시퀀스 / 1,400만 프레임), GOT-10k(1만 세그먼트 / 150만 bbox)를 합쳐 학습한다. 우리가 접근 가능한
수백만 쌍은 tracker 를 처음부터 학습시키는 데 쓰이는 규모와 같은 자릿수다. 학습이 현실적인가라는
물음의 정량적 답이 이것이다.

**강점/약점.**
- 강점 1: recall 과 rank-1 을 하나의 손실로 최적화한다(§1.3a). 시소를 만드는 이음매가 없다.
- 강점 2: response map 이 native 하게 모호도를 준다. 2등/1등 봉우리 비, 엔트로피 -> `second_ratio`,
  `not_distinctive` 를 학습된 신호로 대체 가능. 우리는 이미 그 필드를 쓰고 있다.
- 강점 3: 모델이 작다(RepVGG/ResNet + xcorr). 16GB 호스트 제약과 가장 잘 맞는다.
- 약점 1: tracker 는 연속 프레임의 작은 변화를 전제로 설계됐다. 우리 pair 는 수 개월 공정 drift 를
  건널 수 있다. 완화: 학습 쌍을 시간차로 층화(§2.4).
- 약점 2: cross-correlation 은 결국 상관이다. 학습된 chamfer 로 끝날 위험이 실재한다. 반론은 학습된
  feature 위의 상관은 raw edge 위의 상관과 다르다는 것이지만 이건 주장이 아니라 실험이 답할 문제다.
- OM: 잘 되겠지만 얻을 게 적다. SEM: 이 축의 유일한 시험대.

**추론 비용.** 512x512 search 에서 GPU 10ms 내외, CPU 수백 ms(추정). CPU 추론이 가능한 것이
실무적으로 가장 큰 장점이다 - 상주 GPU 프로세스를 추가하지 않고 monitor 프로세스 안에서 돌릴 수 있다.

**판정: 2단계 본선. 학습을 한다면 이것부터.**

### 2.4 Cross-modal / 시간차 pair - 등록 이미지가 낡은 경우

문제 정의부터 교정한다. 우리 문제는 modality 간 정합(가시광-열화상)이 아니라 같은 modality 의 시간차
정합이다. 등록 이미지는 수 개월 전, 프레임은 지금이고, 그 사이 공정 drift 로 외관이 변한다.

**학습에서 커버하는 법 - 데이터가 이미 답을 갖고 있다.**

시간차를 학습 쌍의 축으로 명시하면 된다. 수백만 장이 있으므로 rcp-msr 쌍만 쓸 이유가 없다.

- `(S_t1, S_t2)` 쌍: 같은 recipe 의 두 성공 프레임. 양쪽 다 crosshair 가 있으므로 대응이 정확히 알려진
  쌍이 공짜로 조합폭발한다. recipe 당 S 가 n 장이면 쌍은 n(n-1)/2 개.
- 시간차 `|t2 - t1|` 로 층화해 긴 시간차 쌍을 hard example 로 과대표집한다. drift robustness 를
  직접 학습시키는 방법이며 별도 augmentation 설계보다 정직하다(실제 drift 를 쓰므로).
- `(rcp, S_t)` 쌍은 그중 시간차가 가장 큰 극단으로 자연스럽게 포함된다.

**증강은 최소로.** XoFTR 은 pseudo-thermal 증강으로 modality 갭을 건너 LoFTR 의 AUC@5 2.6% -> 22% 를
만들었다([XoFTR](https://ar5iv.labs.arxiv.org/html/2404.09692)). 우리에게 대응되는 것은 SEM 물리를
보존하는 증강뿐이다: 밝기/대비 drift, scan noise, charging gradient, 약한 blur, 작은 평행이동/스케일.
**금지**(7월 문서와 동일): 90도 회전, elastic warp, 합성 결함 삽입. 여기에 하나 추가한다 -
crosshair 를 증강으로 지우거나 그리지 말 것. crosshair 제거 A/B 가 이미 -2%p 로 음성이고 가짜 lock
위험이 확인됐다(03 문서 §7).

**판정: 후보가 아니라 데이터 구성 규칙.** 위 층화는 §2.3 과 §2.1 어느 쪽을 하든 적용한다.

### 2.5 SEM aperture 문제에 global context 가 답인가 - 근거 점검

브리프의 질문: "global context 를 쓰는 transformer matcher 가 local NCC 보다 나은 근거가 있는가?"

**일반 근거(있다, 그러나 약하다).** detector-free transformer 는 self/cross-attention 의 전역 수용장으로
저텍스처/반복 패턴에서 sparse detector 가 실패하는 대응을 찾는다고 보고된다(LoFTR 및 후속 survey).
그러나 이건 자연영상 벤치의 일반 진술이고 CD-SEM 성능을 보장하지 않는다.

**우리 저장소 안의 근거(더 강하다).** ADR 0005 가 결정적이다. whitebox box-crop 으로 템플릿을 줄이자
OM -0.042, SEM -0.110 으로 나빠졌다. ADR 0005 의 해석: "결과는 오히려 더 많은 주변 context 가 도움임을
시사한다." 판별 정보가 align key 안이 아니라 그 바깥 FOV 에 있다는 직접 증거다.

여기서 논리를 완성하면:

> 판별 정보가 key 바깥에 있는데, 현재 파이프라인의 점수 함수는 template 크기 window 안에서만 계산된다.
> chamfer/NCC 는 정의상 window 밖을 못 본다. 우리는 정보가 있는 곳을 보지 않는 매처를 쓰고 있다.

이 문서가 제시하는 가장 강한 학습 착수 이유가 이것이다. 그리고 이건 반증 가능하다: 전역 attention
matcher 가 recall-miss 부분집합에서 이득을 못 내면 가설이 틀린 것이다(§3 Step 1/2 의 판정 기준).

**반도체 도메인의 선행 연구(존재하나 직접 대응은 없다).** 검색 결과 다음이 확인된다.
- die-to-database 정합에 DNN 을 학습시켜 CAD 이미지와 실제 SEM 이미지를 맞추는 특허 계열
  ([US 12045996](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12045996),
  [US 10901391 multi-SEM wafer alignment](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10901391)).
- 리소그래피 정렬 마크의 subpixel keypoint localization + 각도 예측을 딥러닝으로 푸는 연구
  ([J Intell Manuf 2024](https://link.springer.com/article/10.1007/s10845-024-02400-8), 본문 접근 실패 -
  제목/초록 수준 인용). 마크 정렬+초점면을 한 이미지에서 동시 예측해 50ms -> 12ms 로 줄인 사례도 보고된다.
- ASML 계열의 alignment/overlay ML 은 마크 좌표 예측이 아니라 virtual metrology/보정 모델 축이다.

**정직한 요약**: "정렬 마크를 딥러닝으로 찾는다"는 산업적으로 이미 하는 일이다. 그러나 그 마크들은
설계상 유일하게 만들어진 전용 마크다. 우리 조건은 엔지니어가 제품 패턴 위 임의 영역을 key 로
등록해 그 영역이 유일하지 않을 수 있다는 것이고, 이 조건을 다룬 공개 연구는 찾지 못했다. 선행 연구는
방법의 실현 가능성을 지지하지 verdict 를 주지 않는다.

### 2.6 라벨 노이즈 - S 라벨의 false positive

**일반 도구는 있다.** 소손실(small-loss) 기준으로 깨끗한 표본을 고르는 co-teaching 계열, 암기 효과를
막는 early-learning regularization 이 표준이다
([ELR, NeurIPS 2020](https://arxiv.org/pdf/2007.00151),
[small-loss criterion 분석, IJCAI 2021](https://www.ijcai.org/proceedings/2021/0340.pdf)).

**그러나 우리 노이즈는 랜덤이 아니다. 이것이 진짜 위험이다.**

장비가 주기 구조에서 wrong-phase 위치에 정렬하고도 자기 기준으로 성공했다면, crosshair 는 decoy 를
가리키고 그 라벨은 정답으로 저장된다. 이 노이즈는:

- 하드 케이스와 완벽히 상관된다. 주기적 SEM 일수록 발생 확률이 높다. 정확히 우리가 고치려는 케이스에서만 틀린다.
- 소손실 기준이 무력하다. 모델은 decoy 를 쉽게 학습한다(구조적으로 정답과 동등하게 잘 맞으므로).
  손실이 작으므로 co-teaching 이 "깨끗한 표본"으로 선택한다. 표준 도구가 정확히 반대로 작동한다.
- 결과: 모델이 decoy 를 예측하도록 학습되고, 오프라인 지표는 좋아 보인다(같은 노이즈로 평가하므로).

`feedback_doubt_s_labels` 가 이미 "도구의 self-reported success 도 false-positive 가능"을 기록했지만,
학습을 도입하면 이 위험의 성격이 달라진다. 지금은 노이즈가 평가를 흐리는 정도지만, 학습하면 노이즈가
모델의 목표가 된다.

**대응 - 도메인 구조를 쓰는 필터가 일반 도구보다 낫다.**

1. **recipe 내 기하 일관성**: 같은 recipe 의 S crosshair 들은 다이/스테이지 좌표계에서 서로 정합되어야
   한다. 각 S 를 recipe consensus 에 정합한 뒤 crosshair 잔차를 보고 잔차가 격자 주기의 정수배로
   튀는 표본을 phase-slip 후보로 격리한다. 소손실 기준과 달리 노이즈의 실제 생성 메커니즘을 겨냥한다.
2. **격리는 별도 집합으로**: phase-slip 후보를 버리면 하드 케이스를 통째로 버린다.
   `train` 에서 빼되 `audit` 집합으로 보관하고, 그 위에서 모델이 어떻게 행동하는지 별도 보고한다.
3. **평가는 반드시 오염되지 않은 부분집합에서도 낸다.** recipe 내 일관성이 확인된 S 만으로 구성한 clean
   test 를 하나 더 둔다. 두 집합의 점수 차이가 노이즈 규모의 하한이다.
4. 일반 도구(ELR)는 위 필터 **다음에** 잔여 랜덤 노이즈용으로 얹는다.

**판정: 학습 프로그램의 전제 조건이다.** §3 Step 2 착수 전에 1번을 측정해야 한다.

### 2.7 기각/보류

| 후보 | 판정 | 이유 |
|---|---|---|
| MASt3R / DUSt3R | **기각** | 웨이퍼는 평면이고 고정 working distance 라 3D pointmap grounding 이 퇴화한다. 외부 cross-modal 벤치에서도 "protocol 민감, 튜닝 없이는 불안정"으로 보고([SAR-optical 벤치](https://arxiv.org/html/2604.10217v4)). 우리 문제에 3D 축을 도입할 근거가 없다 |
| SAM / SAM2 feature | **기각 유지** | 7월 판단 유지. semantic boundary != align target |
| Grounding DINO 텍스트 프롬프트 | **기각 유지** | align key 는 일반 객체 범주가 아니다 |
| template bank 재시도(어떤 형태든) | **금지 유지** | ADR 0006. member-fusion 축 소진 |
| 생성 모델(diffusion/GAN)로 등록 이미지 갱신 | **금지 유지** | 생성된 edge 는 웨이퍼 증거가 아니다 |
| Doppelgangers 식 이진 판별기 | **보류(흥미롭지만 rank-only)** | 시각적으로 닮은 두 구조가 실제로 같은지 판별하는 학습기. 입력이 raw 픽셀이 아니라 **keypoint/match 의 공간 분포**라는 점이 기존 reranker 들과 다르다([Doppelgangers](https://arxiv.org/pdf/2309.02420)). 그러나 여전히 후보 순위 축이라 §1.1 로 13% 상한. Step 3 이후 |

---

## 3. 추천 실험 순서 - 싼 것부터

각 단계는 앞 단계의 판정을 통과해야 착수한다. 판정 기준은 §1.2 의 3개 지표(recall@8, rank1, conversion)
로 쓰고, 비교는 **recipe-level paired bootstrap CI** 로 낸다.

### Step 0. 무비용 - "정보 부재" 가설을 먼저 반증하거나 확인한다

**왜 이게 먼저인가.** 모든 학습 계획은 "SEM 실패의 상당 부분이 정보 부재가 아니다"에 걸려 있다.
그 명제를 검증하는 데 모델이 필요 없다. 코드는 이미 있다.

**할 일.** `template_bank_lab.classify_winner`(`poc/workflow_2/template_bank_lab.py:205`)를 consensus
arm 의 rank-1 오류에 적용해 correct / near_periodic / far_wrong 버킷을 낸다. 지금 이 분류는 template
bank arm 에서만 돌았다.

**왜 결정적인가.** ADR 0006 의 heatmap SEM 수치는 correct 0.492, near_periodic 0.052 다.
나머지 약 0.456 이 far_wrong 이다. 그 arm 에서 SEM 오류의 압도적 다수는 격자 주기의 정수배
phase slip 이 아니라 전혀 엉뚱한 곳이었다. 이것이 사실이라면 "SEM 은 주기성 때문에 정보가 없다"는
설명은 관측된 실패의 주된 원인이 아니다.

**필수 부대 점검(이걸 안 하면 위 해석이 무효다)**: `classify_winner` 는 `period` 를 인자로 받고 period 가
None/0 이면 모든 오류가 far_wrong 으로 떨어진다. 따라서 period 추정(`ensemble_lab.template_periodicity`)의
성공률과 타당성을 함께 보고해야 한다. period 추정이 대부분 실패했다면 far_wrong 0.456 은 아무 의미가 없다.

**판정.**
- far_wrong 이 SEM 오류의 과반 && period 추정이 유효 -> **학습 프로그램 착수**. 전역 context 가설이 살아 있다.
- near_periodic 이 과반 -> **학습 프로그램 중단**. 재등록이 답이라는 기존 결론이 강화된다. 이 문서의 나머지는
  실행하지 않는다.

**비용**: 기존 드라이버에 분류 호출 추가. 새 모델 없음, 새 데이터 없음.

### Step 1. 하루 - zero-shot DINOv2/v3 상관 채널 (학습 0)

**할 일.** frozen DINOv2(또는 DINOv3) patch feature 상관맵의 peak 를 `_Cand` 리스트로 내는 채널을
`ensemble_lab` 에 C5 로 추가한다. 배선점은 `_channel_solo_candidates` 와 동형
(`poc/workflow_3/align/matching/ensemble.py:145`). production engine 무수정, `ALIGN_LAB_ENSEMBLE_CHANNELS`
에 이름 추가로 켠다(ADR 0003 의 기존 패턴).

**측정.** 두 arm 을 낸다.
- C5 solo 의 recall@8 (proposer 능력 격리, `proposer_recall_ab.py` 정의)
- C1/C2/C3 + C5 RRF 의 recall@8 **및** 표준 NCC rerank 후 rank1

**판정 기준(SEM 기준으로 읽는다. OM 은 무손실만 확인).**

| 결과 | 판정 |
|---|---|
| C5 를 넣은 RRF 의 SEM recall@8 이 C1/C2/C3(0.698, ADR 0003) 대비 **+5pp 이상** 이고 rank1 이 consensus SEM 0.665 대비 **손실 없음** | 통과 -> Step 2 |
| recall 은 오르는데 rank1 이 떨어짐 | **§1.2 시소 재현**. 채널로는 기각하되 "전역 feature 에 recall 이 있다"는 정보는 확보 -> Step 2 를 조건부 진행(학습 모델이 순위까지 책임지는 구조라면 의미 있음) |
| SEM recall@8 이 +2pp 미만 | **기각. 여기서 멈춘다.** 자연영상 SSL feature 는 SEM 에 전이되지 않는다는 결론 |

**왜 recall@8 만 보지 않는가.** 브리프의 예시는 recall@8 만 쟀지만, ADR 0006 이 정확히 그 함정을
기록했다. 같은 하네스가 rank1 을 거의 공짜로 주므로 둘 다 낸다.

**비용**: 채널 1개 구현 + 오피스 1회 실행. GPU 는 오프라인이므로 제약 없음.

### Step 2. 2~4주 - tracker 계열 heatmap 모델 학습 (본선)

**할 일.** §2.3 의 모델을 수백만 쌍으로 학습한다. §2.6 의 phase-slip 필터를 먼저 돌린다.

**측정.** 기존 golden 200-recipe 를 **완전히 hold out** 해서 같은 지표를 낸다.

**판정 기준.**

| 지표 | 현재(consensus) | Step 2 통과선 | 실패선 |
|---|---:|---:|---:|
| SEM rank1 | 0.665 | **>= 0.73** (+6.5pp, 잔여 실패의 20% 회수) | < 0.68 이면 기각 |
| SEM in_topk | 0.789 (03) / 0.849 (ADR 0006 set) | >= 0.85 | - |
| conversion (rank1/in_topk) | ~0.78 | **>= 0.78** | 0.70 미만이면 recall 이 올라도 기각 |
| OM rank1 | 0.852 | **무손실**(>= 0.84) | 손실이면 SEM 전용 라우팅 |
| recall-miss 부분집합 회수율 | - | **이 지표를 주 근거로 삼는다** | 0 이면 가설 반증 |

마지막 행이 핵심이다. 기존에 `gt_in_topk=False` 였던 행에서 몇 %를 되찾았는가가 §1.1 의 87% 를 직접 겨냥한
유일한 숫자다. 전체 평균 상승은 OM 의 쉬운 행으로 만들어질 수 있으므로 신뢰하지 않는다.

**중단 조건**: high-confidence off-target 이 늘어나면 rank1 이 올라도 즉시 중단. 이건 안전 지표이고
정확도와 교환하지 않는다.

### Step 3. 조건부 - EfficientLoFTR/RoMa fine-tune 또는 도메인 SSL 사전학습

Step 2 가 통과하되 SEM rank1 이 0.80 에 못 미치면 진행한다. 두 갈래이며 동시에 하지 않는다.

- **3a. §1.3(c) dense 승격 + EfficientLoFTR fine-tune.** 선행 조건: 배율 가정 3개(§1.3c) 검증.
  전역 attention 이 §2.5 의 key 바깥 정보를 실제로 쓰는지 보는 직접 실험.
- **3b. 도메인 SSL 사전학습 후 Step 2 모델 초기화.** E 프레임(라벨 없음)을 여기서 쓴다. MAE 계열이
  SEM 에서 자연영상 사전학습을 이긴다는 보고가 근거([SEM foundation model](https://arxiv.org/html/2604.05960v1)).
  7월의 P3-I 가 여기로 온다. 단 목적이 바뀌었다. 이제 이유는 라벨 부재가 아니다. E 프레임을 쓰려는 것이다.

Step 1 이 통과했다면 3a 를, 기각이었다면 3b 를 먼저 한다(Step 1 기각 = 자연영상 사전학습 무용 = 도메인
사전학습이 필요하다는 뜻).

---

## 4. 데이터 준비 요구

### 4.1 수량과 쌍 구성

| 용도 | 쌍 구성 | 목표 수량 | 비고 |
|---|---|---:|---|
| Step 2 학습 | `(rcp/consensus 템플릿, msr S 프레임)` | 10^5 ~ 10^6 | tracker 학습 관례와 같은 자릿수(LaSOT 3.52M, TrackingNet 14M 프레임) |
| Step 2 학습(증폭) | `(S_t1, S_t2)` 동일 recipe 쌍 | 조합폭발, 시간차로 층화 샘플 | **양쪽 crosshair 로 대응이 정확** (§2.4) |
| Step 3b SSL | E 프레임 + 라벨 없는 전체 | 가능한 전량 | 라벨 불필요 |
| 최종 test | 기존 golden 200 recipe | 고정 | **학습에 절대 넣지 않는다.** 0.876/0.764 와 비교 가능성 유지 |

**modality 층화 필수.** SEM 이 목표인데 OM 이 수적으로 우세하면 손실이 OM 에 지배된다. 배치 내 비율을
고정하거나 SEM 전용 모델을 따로 둔다(추천: SEM 전용 모델. OM 은 이미 0.852 이고 rerank 도 모드별로
갈렸다 - 03 문서 §7 의 "한 심판이 지배할 때는 종합이 아니라 전환" 교훈과 같은 논리).

### 4.2 hold-out 규칙 (전부 필수, 하나라도 어기면 숫자가 무의미하다)

1. **recipe 단위 분리**: 같은 recipe 의 프레임이 train 과 test 에 동시에 있으면 안 된다. consensus 와
   외관이 누수된다(7월 문서와 동일, 재확인).
2. **장비(eqp) 분리 보조 표**: consensus pool 은 eqp 무관이지만 캡처 특성은 장비별로 다르다.
   최소한 별도 표로 보고한다.
3. **시간 분할**: test 를 train 보다 미래로 둔다. 실제 배포는 항상 미래 프레임을 상대한다.
   랜덤 분할은 drift robustness 를 과대평가한다.
4. **unseen recipe 가 주 지표**: production 은 학습에 없던 recipe 를 만난다. seen-recipe 점수는 보조.
5. golden 200 은 최종 1회만 본다. 하이퍼파라미터는 dev split 에서 고정.

### 4.3 평가 대상 분포의 구조적 간극 (기존 벤치에도 있는 문제, 학습이 악화시킨다)

`_consensus_template_ab` 는 S 프레임을 LOO 로 순회하며 평가한다
(`poc/workflow_2/align_similarity.py:888-916`, `by_recipe[rec]["s_frames"]`). E 프레임은 generic 가드의
비교값으로만 쓰인다.

0.876 / 0.764 는 성공 프레임 위에서 측정된 숫자다. production 은 align fail 프레임 위에서
돈다. E 프레임에는 crosshair 가 없어 GT 가 없으므로 채점 자체가 불가능하다. 이건 기존 벤치의
한계이지 이 문서가 만든 문제가 아니다. 그러나 학습을 도입하면 모델이 S 분포에 최적화되고 E 분포에서는
평가되지 않는 상태가 되므로 위험이 커진다.

**요구사항**:
- 최소한 E 프레임에서 점수 분포와 abstention 율을 보고한다(정확도는 못 내도 분포는 낸다).
- S 와 E 의 응답맵 통계(최대값, 2등 비, 엔트로피)가 유의하게 다르면 그것 자체가 배포 위험 신호다.
- 진짜 종단 지표는 운영 루프의 corrected vs escalated 비율이다. 오프라인 rank1 로는 대신하지 못한다.
  Step 2 통과 후에도 shadow 단계를 건너뛰지 않는 이유가 이것이다.

### 4.4 데이터 반출 경계

오피스 fab 이미지는 반출 불가다(`feedback_no_office_data_to_mac`). 따라서:
- 학습/평가는 전부 오피스 내부. Mac 에서는 합성 데이터 단위 테스트만.
- 사전학습 가중치는 오피스로 반입(단방향). 라이선스는 도입 전 확인 필요 -
  특히 DINOv3 는 Meta 자체 라이선스이고 LoFTR/LightGlue 계열과 조건이 다르다(확인 필요).
- 결과는 `[DIGEST]` 텍스트로만 회신(기존 규약).

---

## 5. 정직한 리스크 - "학습해도 SEM 의 aperture 문제는 정보 부재라 못 푼다"

이 반론에 정면으로 답한다. **부분적으로 맞고, 결론으로는 틀렸을 가능성이 높다.**

### 5.1 맞는 부분 - 양보한다

템플릿 `T` 가 FOV 안에서 주기 `p` 로 정확히 반복하면, `T(x) = T(x + kp)` 이고 그 패치만의 함수로는
어떤 모델도 `k` 를 정할 수 없다. 이건 모델링 실패가 아니라 정보이론적 사실이며 학습은 없는 정보를
만들지 못한다. 이 경우 Bayes-optimal 출력 자체가 점 하나가 아닌 집합이다. 올바른 운영 행동은
`NO_ACTION` + 재등록이다. 이 입장은 유지되며 어떤 모델도 이걸 우회할 권한을 갖지 않는다.

ADR 0005 와 0006 이 이 벽을 경험적으로 확인했다: median-consensus, soft-voting heatmap, one-vote RRF
세 fusion 이 모두 in_topk ~0.9 를 만들면서 rank-1 ~0.5 라는 같은 벽에 부딪혔다.

### 5.2 틀린 부분 - 반론이 전제하는 사실이 데이터와 맞지 않는다

반론은 "관측된 SEM 실패 = 주기성 phase 모호"를 전제한다. **저장소의 데이터가 그 전제를 지지하지 않는다.**

**증거 1 (직접).** ADR 0006 의 kill-test 버킷, SEM heatmap arm: correct 0.492, near_periodic 0.052.
나머지 약 0.456 은 far_wrong 계열이다. 격자 주기의 정수배 오류는 SEM 오류의 약 10분의 1이고
나머지는 주기성과 무관한 엉뚱한 위치다. 주기성 정보 부재는 관측 실패의 지배적 원인이 아니다.
(단서: 이 값은 template bank arm 의 것이며 period 추정 유효성 확인이 필요하다. 그래서 Step 0 이 있다.)

**증거 2 (간접, 그러나 메커니즘이 명확하다).** ADR 0005: 템플릿을 key 안쪽으로 줄이자 SEM 이 -0.110 으로
가장 크게 나빠졌다. ADR 0005 자신의 해석이 "더 많은 주변 context 가 도움". 판별 정보가 key 바깥에
존재한다는 뜻이고 실제 다이에는 그런 것이 있다: 어레이 경계, 다이 경계, dummy fill 경계, 스캔/차징
그라디언트. 유한한 어레이는 무한 격자가 아니다.

**증거 3 (구조적).** 현재 점수 함수는 template 크기 window 안에서만 계산된다. chamfer 도 NCC 도 window
밖을 정의상 못 본다. **증거 2 가 가리키는 정보를 현재 매처는 원리적으로 사용하지 않는다.**
"재등록 필요"로 분류된 SEM key 84/104 는 *정보가 없다*는 판정이 아니라 *chamfer/NCC 점수면이 평평하다*는
판정이다. 이 둘은 지금까지 구분된 적이 없다.

### 5.3 그래서 종합하면

> **강한 형태의 반론("SEM 은 정보가 없다")은 검증된 적이 없고, 데이터는 오히려 약한 형태
> ("현재 매처가 보는 window 안에는 정보가 없다")를 지지한다.** 후자라면 전역 수용장을 가진 방법에
> 회수 여지가 있다. 전자라면 없다. **Step 0 이 이 둘을 가른다.**

### 5.4 남는 리스크는 정직하게

1. **천장은 1.0 이 아니다.** near_periodic 잔차(~5pp)와 진짜 featureless 프레임은 어떤 방법으로도
   안 풀린다. 낙관적 상한 추정: SEM rank1 0.665 -> 0.85~0.90 (**추정**, 증거 2/3 이 가리키는 far_wrong
   부분을 절반 이상 회수한다는 가정). 이 추정이 틀릴 경로가 여럿이라 Step 2 통과선을 0.73 으로 낮게 잡았다.
2. **학습 모델은 새로운 실패 양식을 들여온다.** chamfer 의 평평한 점수면은 최소한 모호함을 *보이게*
   했다. 학습 모델은 모호한 프레임에서도 뾰족한 응답맵을 낼 수 있다 - 자신 있는 오답. 이건 지금보다
   나쁜 상태다(보정이 조용히 틀린다). 그래서 응답맵의 2등 비/엔트로피를 abstention 신호로 반드시
   내보내고 `engineer_review` 로 라우팅해야 한다. 선택 사항이 아니다.
3. **라벨 노이즈가 하드 케이스와 상관된다**(§2.6). 표준 노이즈 대응 도구가 역작동한다.
   대응이 안 되면 오프라인 지표는 좋은데 현장에서 decoy 로 가는 상태가 될 수 있고, 이건 오프라인
   평가로는 안 잡힌다.
4. **평가 분포가 배포 분포가 아니다**(§4.3). S 로 학습하고 S 로 평가하는데 E 에서 돈다.
5. **운영 비용이 알고리즘 비용보다 크다.** 16GB 호스트에서 상주 프로세스 하나는 VLM 하나와 교환이다.
   Step 2 의 모델을 CPU 인프로세스로 돌릴 수 있게 설계해야 하는 실무적 이유가 이것이다.
6. **재등록은 어느 쪽이든 필요하다.** 학습이 성공해도 chronic-ambiguous key 는 재등록이 정답이다.
   이 연구는 재등록 워크스트림과 병행하는 축이다.

---

## 6. 7월 문서와 판단이 갈리는 지점 (요약)

| # | 7월 | 지금 | 근거 |
|---|---|---|---|
| 1 | "learned matcher 는 top-K crop pair 에서만 검증"(P1-G 게이트 1) | **full-frame proposer 로 평가한다** | top-K 안에서는 정의상 87% 를 못 고친다. 7월 게이트는 87% 확정 이전 |
| 2 | patch-pair ranker 가 첫 학습 실험(P0-E) | **첫 학습 실험은 tracker heatmap** | 손실 하나가 recall+rank1 을 동시에 최적화. ranker 는 rank-only |
| 3 | DINOv2 는 ranker 의 feature(R3) | **독립 proposer 채널로 zero-shot 측정** | 학습 0으로 recall 축을 건드리는 유일한 후보 |
| 4 | SSL 사전학습은 "라벨 부족" 대안(P3-I, 최후순위) | **E 프레임 활용 수단으로 재정의, 3단계 본선** | 라벨은 이제 부족하지 않다. 부족한 건 fail 분포 커버리지 |
| 5 | 라벨 노이즈 언급 없음(golden 은 사람 검증 전제) | **학습의 전제 조건으로 승격** | 수백만 자동 라벨은 검증되지 않았고, 노이즈가 하드 케이스와 상관된다 |
| 6 | "SEM 반복 패턴이면 어떤 모델도 근거 없는 확신" (결론부) | **동의하되, 그것이 현재 실패의 주 원인이라는 근거는 없다** | ADR 0006 near_periodic 0.052 vs far_wrong ~0.456 |

7월 문서의 안전 경계, 롤아웃 4단계(offline -> shadow -> router-only -> production), 평가 표 항목은
그대로 채택한다. 바뀐 것은 실험 순서와 학습 설계뿐이다.

---

## 7. 한 문단 요약

Step 0(무비용)이 far_wrong 우세를 확인하면 학습은 타당하고, 그때 첫 모델은 patch-pair ranker 도
LoFTR fine-tune 도 아닌 **tracker 계열 heatmap localizer** 다. 이유는 세 가지다 -
① cond.txt 1점 라벨을 가정 없이 그대로 쓴다, ② 하나의 손실이 recall 과 rank-1 을 동시에 최적화해
ADR 0006 이 기록한 시소를 구조적으로 피한다, ③ 모델이 작아 16GB 호스트 제약과 CPU 인프로세스 추론에
맞는다. 그 전에 zero-shot DINOv2 채널로 하루 만에 "자연영상 SSL feature 가 SEM 에 전이되는가"를
반증 가능하게 물어보고 학습에 들어가기 전에 phase-slip 라벨 필터를 반드시 통과시킨다.

---

## 참고 자료

**학습 기반 matcher**
- [LoFTR: Detector-Free Local Feature Matching with Transformers (CVPR 2021)](https://arxiv.org/pdf/2104.00680)
- [Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed](https://arxiv.org/html/2403.04765v2)
- [RoMa: Robust Dense Feature Matching (CVPR 2024)](https://arxiv.org/pdf/2305.15404)
- [RoMa v2: Harder Better Faster Denser Feature Matching (2025)](https://arxiv.org/abs/2511.15706)
- [LightGlue: Local Feature Matching at Light Speed (ICCV 2023)](https://arxiv.org/abs/2306.13643)
- [Grounding Image Matching in 3D with MASt3R (ECCV 2024)](https://arxiv.org/abs/2406.09756)

**도메인 전이 증거 (양방향)**
- [Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](https://arxiv.org/html/2604.10217v4) - RoMa 가 cross-modal 학습 없이 3.0px, MASt3R/DUSt3R 는 불안정
- [XoFTR: Cross-modal Feature Matching Transformer](https://ar5iv.labs.arxiv.org/html/2404.09692) - pseudo-modality 증강으로 AUC@5 2.6% -> 22%
- [A Mixture of Experts Foundation Model for SEM Image Analysis (2026)](https://arxiv.org/html/2604.05960v1) - SEM 도메인 MAE 사전학습이 자연영상 사전학습을 앞선다

**one-shot / tracking 계열 (추천안의 근거)**
- [OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features (ECCV 2020)](https://arxiv.org/abs/2003.06800) - 학습에 없던 클래스로 일반화
- [LaSOT (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_LaSOT_A_High-Quality_Benchmark_for_Large-Scale_Single_Object_Tracking_CVPR_2019_paper.pdf), [GOT-10k](https://arxiv.org/pdf/1810.11981) - 학습 데이터 규모 기준점

**foundation feature**
- [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv3 dense feature 정리](https://www.emergentmind.com/topics/self-supervised-vision-transformers-dinov3)
- [AnomalyDINO: Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529) - 산업 텍스처에서 frozen patch feature 유효 사례

**모호성 판별**
- [Doppelgangers: Learning to Disambiguate Images of Similar Structures](https://arxiv.org/pdf/2309.02420) - 입력이 픽셀이 아니라 match 의 공간 분포

**라벨 노이즈**
- [Early-Learning Regularization Prevents Memorization of Noisy Labels (NeurIPS 2020)](https://arxiv.org/pdf/2007.00151)
- [Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion (IJCAI 2021)](https://www.ijcai.org/proceedings/2021/0340.pdf)

**반도체 도메인 선행 (직접 대응 없음)**
- [Methods and systems for registering images for electronic designs (US 12045996)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12045996)
- [Multi-scanning electron microscopy for wafer alignment (US 10901391)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10901391)
- [Subpixel keypoint localization and angle prediction for lithography marks (J Intell Manuf 2024)](https://link.springer.com/article/10.1007/s10845-024-02400-8) - 본문 접근 실패, 제목/초록 수준 인용

**저장소 내부 근거**
- `docs/project_progress/03_workflow_2.md` §6-8 - 0.876/0.764, OM/SEM 분해, 87% recall miss
- `poc/workflow_2/docs/study/adr/0005-*.md` - box-crop 기각, "더 많은 context 가 도움"
- `poc/workflow_2/docs/study/adr/0006-*.md` - template bank 기각, 버킷 수치, rank-1 비교 원칙
- `poc/workflow_2/docs/study/reranker_ab_failure_analysis.md` - reranker 축 사망
- `poc/workflow_2/align_similarity.py:88, 888-916` - GT_TOL_NORM, S 프레임 LOO 평가
- `poc/workflow_2/template_bank_lab.py:205` - `classify_winner` (Step 0 의 도구)
- `poc/workflow_3/align/matching/ensemble.py:145-174` - 채널 배선점 (Step 1)
- `poc/workflow_3/align/matching/engine.py:47` - `DEFAULT_SCALES` (배율 가정 반증)
- `poc/workflow_3/align/cond_file.py:155-178` - cond.txt Magnification

외부 연구는 방법의 기제와 제약을 뒷받침할 뿐 CD-SEM align-key 성능을 보장하지 않는다. 이 문서의 모든
판정은 오피스 recipe-disjoint 오프라인 평가와 shadow 안전 평가를 통과하기 전까지 reposition/OK 권한을
얻지 못한다.
