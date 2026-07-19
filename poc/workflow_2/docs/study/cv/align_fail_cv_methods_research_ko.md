---
status: proposed
date: 2026-07-10
scope: offline research and experiment plan only
---

# Align Fail 정렬 위치 보정을 위한 추가 CV 방법 조사와 실험 우선순위

## 결론

현재의 가장 강한 기준선은 최근 성공 이미지의 leave-one-out `consensus`를 만들고, Canny/Scharr/orientation Chamfer proposer를 RRF로 합친 뒤 NCC로 재순위하는 방식입니다. 이 기준선의 오프라인 결과는 `in_topk=0.876`, `rank1=0.764`이며, 특히 SEM의 `rank1`은 약 `0.665`입니다. 따라서 다음 실험은 새로운 전역 탐색기를 무작정 추가하는 일이 아니라, 아래 두 실패를 분리해 다뤄야 합니다.

1. **정답이 이미 top-K에 있는데 1순위만 틀린 경우**: 후보별 국소 registration과 기하 검증으로 개선할 수 있습니다.
2. **정답이 top-K에도 없는 경우**: proposer 자체를 바꾸거나, 현재 reference와 live frame 사이의 contrast/geometry 차이를 줄여야 합니다.
3. **반복 패턴이라 어느 후보도 유일하지 않은 경우**: CV 점수만으로 해결할 수 없습니다. `NO_ACTION`과 재등록 권고가 올바른 결과입니다.

권장 순서는 **(P0) ECC 국소 registration**, **(P0) SIFT/AKAZE + RANSAC 기하 검증**, **(P1) phase correlation 국소 정밀화**입니다. 모두 현재 top-K 후보에서만 동작시키므로, 이미 효과가 없었던 box-crop, template bank, 전역 MI, 추가 edge-channel을 다시 구현하지 않습니다. MIND 계열 descriptor와 learned correspondence matcher는 그 다음의 offline-only 가설입니다.

> 이 문서는 알고리즘의 채택 결정이 아닙니다. 모든 제안은 오피스 golden data의 leave-one-out 평가를 통과하기 전까지 live reposition 또는 OK 동작에 연결하지 않습니다.

## 1. 현재 작업에서 확인한 전제

### 1.1 자산과 좌표 계약

- `align_img_from_rcp/`의 엔지니어 박스 안 구조가 reference align key이며, target point는 recipe-box center입니다.
- live crosshair는 현재 잘못 시도한 위치이고, matcher의 `best_xy`는 live frame 안 target point를 뜻합니다.
- PRIMARY는 key가 paused SEM 화면에 보일 때 `best_xy`로 double-click recenter 후 OK를 누르는 흐름입니다. pan/zoom은 key가 보이지 않을 때만 FALLBACK입니다.
- 따라서 새 방법은 `best_xy`의 **template-center / live-frame pixel** 의미와 `align_offset_xy` 보정 계약을 유지해야 합니다. 다른 좌표계의 confidence만으로 클릭을 만들면 안 됩니다.

근거는 루트 `CONTEXT.md`와 `poc/workflow_2/docs/study/adr/0002-primary-reposition-vs-fallback-search.md`입니다.

### 1.2 확인된 기준선과 제한

`docs/project_progress/03_workflow_2.md`의 200-recipe offline bench는 다음을 확인했습니다.

| 항목 | 확인된 결과 | 이 조사에 주는 의미 |
|---|---:|---|
| 단일 등록 이미지 | `in_topk=0.434`, `rank1=0.318` | 단일 stale reference를 기준선으로 삼지 않습니다. |
| 성공 사례 consensus | `in_topk=0.876`, `rank1=0.764` | 이후 모든 A/B의 기준선입니다. |
| OM consensus | `in_topk≈0.911`, `rank1≈0.852` | OM은 개선 여지가 작으므로 SEM과 분리 평가합니다. |
| SEM consensus | `in_topk≈0.789`, `rank1≈0.665` | 반복 구조에서 후보 miss와 rank ambiguity가 남은 주 대상입니다. |

현 코드도 이 구분을 이미 계측합니다. `align_similarity.py`의 `_gt_in_topk()`은 proposer recall과 rank-1 오류를 분리하고, `ensemble_lab.py`은 기본 production 경로를 건드리지 않는 opt-in 실험 통로입니다. 실험은 그 경계를 그대로 사용해야 합니다.

### 1.3 재실험하지 않을 항목

다음은 이미 실험되어 효과가 없거나 위험이 확인됐습니다. 아래를 새 이름으로 다시 제안하지 않습니다.

- recipe white-box만 쓰는 box-crop: OM 약 `-0.04`, SEM 약 `-0.11`.
- 성공 이미지 개별 template bank: proposer recall은 높아도 두 모드 rank-1이 약 `0.5`로 붕괴.
- 단순 contour/edge-NCC 채널 추가: OM 약 `+0.9%p`, SEM 효과 없음.
- consensus 위의 추가 ensemble: 남은 구조적 miss 때문에 기대 상한이 약 `+2~6%p`.
- crosshair inpaint/removal: 원본보다 약 `-2%p`이며 가짜 peak 위험.
- dense/global MI를 1차 matcher로 쓰는 방향: 비용이 높고, 현재 분석에서도 MI는 top-K verifier일 때만 의미가 있습니다.

## 2. 어떤 실패에 어느 방법을 적용할지

| 실패 분류 | 현재 관측 방법 | 맞는 실험 | 맞지 않는 실험 |
|---|---|---|---|
| proposer miss | `gt_in_topk=False` | MIND descriptor proposer, learned correspondence | 후보 후단 NCC/ECC만 추가 |
| 순위 오류 | `gt_in_topk=True`, `rank1=False` | ECC, phase correlation, SIFT/AKAZE-RANSAC, learned verifier | 새 template bank |
| 작은 위치/scale 오차 | truth 위치에서 score는 있으나 local peak가 빗나감 | ECC, phase correlation | 넓은 전역 pan/zoom |
| contrast/texture drift | edge/ORB가 모두 약하지만 위치 단서는 남음 | MIND, consensus quality gate, learned matcher | raw intensity NCC 단독 |
| 반복 패턴의 비변별 key | second ratio/gap이 모호하고 다수 peak가 동점 | `NO_ACTION`, 재등록 우선순위 | score를 강제로 하나 고르는 reranker |

`in_topk`가 낮은 row와 높은 row를 섞어 rank-1만 비교하면, 후보 후단 method가 proposer miss를 고친 것처럼 보이는 해석 오류가 생깁니다. 모든 결과 표는 위 다섯 분류와 OM/SEM을 함께 기록해야 합니다.

## 3. 우선 실험할 방법

### P0-A. top-K 후보별 ECC(Enhanced Correlation Coefficient) 국소 registration

**가설**: 현재 Chamfer/RRF가 target 근처까지는 찾았지만 contrast 변화, sub-pixel translation 또는 작은 scale/rotation 오차 때문에 순위가 틀린 경우, 후보 crop을 ECC로 정밀 정렬하면 rank-1이 개선됩니다.

OpenCV `findTransformECC`는 대략 맞춰진 초기 transform에서 intensity 기반 area alignment를 반복 개선합니다. 문서도 큰 displacement에는 rough initialization이 필요하다고 명시하므로, 이를 전역 search로 사용하면 안 되고 현재 top-K의 crop에서만 사용해야 합니다. [OpenCV ECC 문서](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)

**실험 설계**

1. consensus baseline이 만든 후보별 `(xy, scale)`에서 raw template 크기와 같은 live crop을 뽑습니다.
2. crop과 template에 동일한 CLAHE/gradient 또는 raw-gray arm을 각각 적용합니다. crosshair를 지우거나 inpaint하지 않습니다.
3. identity 또는 candidate scale을 초기값으로 두고, 변환 모델을 `translation -> Euclidean -> affine` 순으로 늘립니다. wafer SEM은 perspective camera가 아니므로 homography는 첫 실험에서 제외합니다.
4. ECC coefficient, 수렴 실패 여부, refined center, translation 크기, rotation, uniform scale 또는 affine determinant를 후보 메타데이터로 저장합니다.
5. 기존 Chamfer/NCC와 ECC를 후보 후단에서만 비교합니다. `best_xy`는 refined transform으로 template center를 live frame에 투영해 계산합니다.

**필수 reject gate**

- `cv2.error` 또는 미수렴이면 후보를 버리고 baseline 결과를 유지합니다.
- refined 이동량이 후보 crop margin을 넘거나, 물리적으로 불가능한 rotation/scale/affine shear면 버립니다.
- ECC가 높은 단일 후보만으로 action하지 않습니다. 기존 `distinctive`, second-gap, key visibility gate와 post-reposition verification을 모두 유지합니다.

**성공/실패 해석**: `gt_in_topk=True` subset의 rank-1만 오르면 의도대로 verifier가 작동한 것입니다. 이 subset에서도 개선이 없으면 contrast drift보다 반복 구조가 주원인이므로 ECC를 production에 넣지 않습니다.

### P0-B. SIFT 또는 AKAZE correspondence + RANSAC limited-affine verifier

**가설**: ORB inlier ratio가 약한 reference drift 상황에서도 scale-space keypoint와 기하적 일관성으로 잘못된 local peak를 거를 수 있습니다.

SIFT는 scale/rotation에 불변인 지역 feature를 만들고 illumination 및 제한된 affine 변화에도 강하도록 설계되었습니다. [Lowe, 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) OpenCV의 `estimateAffinePartial2D`는 translation, rotation, uniform scale의 4-DoF transform을 RANSAC으로 추정하고 inlier만으로 재정밀화합니다. [OpenCV limited-affine 문서](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

**실험 설계**

1. 현재 top-K live candidate crop마다 `SIFT`와 `AKAZE`를 별도 arm으로 실행합니다. ORB를 교체하지 않고 comparator로 남깁니다.
2. descriptor의 mutual nearest-neighbor와 ratio test를 적용한 뒤, `estimateAffinePartial2D(..., RANSAC)`으로 transform을 찾습니다.
3. 후보 score는 단순 match 수가 아니라 `inlier count`, `inlier ratio`, reprojection error, template 위 inlier의 spatial coverage, 추정 transform의 물리적 범위로 구성합니다.
4. template 중심을 transform으로 live 좌표에 보낸 값이 기존 candidate center와 과도하게 다르면 reject합니다.

**왜 spatial coverage가 필수인가**: 반복 line/edge 한 조각에서 일치한 keypoint만 많아도 RANSAC은 그 조각을 지지할 수 있습니다. inlier가 template의 한 모서리 또는 한 stripe에 몰리면 target point의 위치를 증명하지 못하므로 reject해야 합니다.

**성공/실패 해석**: SEM에서 inlier coverage가 확보되고 `gt_in_topk=True` rank-1이 오르면 ECC와 조합할 가치가 있습니다. keypoint가 거의 생기지 않거나 wrong peak도 동일한 inlier geometry를 만들면, 이 방법은 비변별 SEM key에 적합하지 않습니다.

### P1-C. phase correlation 기반 sub-pixel translation 정밀화

**가설**: 후보 주변에서 translation만 남은 경우, gradient/whitened crop의 Fourier phase peak가 NCC보다 위치 보정에 안정적일 수 있습니다.

OpenCV `phaseCorrelate`는 같은 크기의 두 float image에서 Fourier shift theorem으로 translation을 구하고, peak의 5x5 weighted centroid로 sub-pixel shift를 반환합니다. response는 single peak일수록 1에 가까워집니다. [OpenCV phase correlation 문서](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)

**실험 설계**

1. candidate scale로 template와 live crop의 크기를 같게 맞춘 뒤, raw-gray, gradient magnitude, Canny distance-friendly representation의 세 arm을 만듭니다.
2. Hann window를 적용하고 `(dx, dy, response)`를 얻습니다.
3. response와 shift 크기를 verifier feature로 기록하고, response가 낮거나 shift가 crop margin을 넘으면 후보를 reject합니다.
4. phase shift만으로 후보 순위를 뒤집지 말고, baseline/NCC/ECC와 함께 top-K 후단 score로 A/B합니다.

**제한**: 이 방법은 translation 전용이며 scale/rotation/큰 capture range를 해결하지 않습니다. 정답이 top-K에 없는 경우의 proposer로 쓰지 않습니다.

### P1-D. MIND-like local self-similarity descriptor

**가설**: SEM의 brightness/contrast/texture 변화가 raw NCC와 edge map 모두를 흔드는 row에서는, 각 이미지 내부의 patch self-similarity를 비교하는 descriptor가 더 안정적인 local similarity를 제공할 수 있습니다.

MIND는 modality별 absolute intensity 관계 대신 local self-similarity descriptor를 조밀하게 만들어 비교하는 registration 방법입니다. 원 논문은 multi-modal medical image를 대상으로 하므로 CD-SEM 성능의 근거는 아니며, 여기서는 contrast drift에 대한 **offline 가설**로만 사용합니다. [Heinrich et al., 2012](https://www.sciencedirect.com/science/article/pii/S1361841512000643)

**실험 설계**

1. 2-D template/live에 4 또는 8개 fixed offset의 Gaussian patch distance를 계산해 MIND-like descriptor map을 만듭니다.
2. 현재 top-K crop에서 descriptor SSD/NCC를 verifier로 먼저 비교합니다. 그 결과가 좋을 때만 same descriptor map의 dense score map을 proposer arm으로 시험합니다.
3. raw intensity와 edge map은 그대로 보존해, 어느 drift class에서만 이득이 생기는지 분리합니다.

**중단 조건**: descriptor 계산 시간이 현 bench에서 부담스럽거나 `gt_in_topk=True` rank-1에 이득이 없으면 dense proposer는 만들지 않습니다. MIND도 반복 패턴의 유일성을 만들지 못합니다.

### P2-E. learned local correspondence: LoFTR 또는 ALIKED/SIFT + LightGlue

**가설**: detector-free 또는 learned correspondence가 classical keypoint가 거의 나오지 않는 low-texture pair에서 추가 correspondence를 제공할 수 있습니다.

LoFTR는 detector/descriptor/matching 순서를 분리하지 않고 coarse-to-fine dense correspondence를 만드는 Transformer matcher이며 low-texture 영역을 목표로 합니다. [LoFTR, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) LightGlue는 sparse local feature matcher이며 pair 난이도에 따라 추론량을 줄이도록 설계됐습니다. [LightGlue, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)

**실험 설계와 경계**

- HCP 또는 office 환경에서만, 사내 image가 외부 API로 나가지 않는 local weights 조건을 확인한 뒤 실행합니다.
- 첫 실험은 full-frame search가 아니라 current top-K crop pair의 verifier입니다. LoFTR 1개와 LightGlue 1개를 동시에 production 후보로 만들지 말고, low-texture row에는 LoFTR, 충분한 SIFT/ALIKED keypoint가 있는 row에는 LightGlue라는 사전 routing 가설을 검증합니다.
- learned match 수가 아니라 RANSAC/limited-affine inlier geometry, center projection, candidate-to-candidate gap으로 평가합니다.
- natural-image pretrained model의 SEM domain shift, GPU/weight/license, worst-case latency를 별도 기록합니다.

**중단 조건**: SEM wrong peak에 대해 높은 confidence를 내거나, offline rank-1은 올라도 high-confidence off-target가 늘면 바로 reject합니다. learned model은 최종 action authority가 아니며 현 deterministic gate를 우회하지 않습니다.

### P3-F. 동일 FOV의 무동작 multi-frame denoising/consensus

**가설**: paused 상태에서 stage를 움직이지 않고 연속 캡처한 3~5 frame의 shot noise/scan noise가 독립적이면, sub-pixel registration 후 median 또는 robust mean으로 현재 frame 품질이 좋아질 수 있습니다.

**실험 설계**

- 먼저 static screenshot이 실제로 pixel-level variation을 갖는지 확인합니다. 동일 bitmap을 반복 캡처하는 환경이면 실험 가치가 없습니다.
- 개별 frame, median frame, robust mean frame에 동일 baseline matcher를 적용해 비교합니다.
- frame fusion은 key visibility가 낮을 때만 시도하고, capture budget과 timeout을 명시합니다.

**제한**: noise를 줄일 수 있을 뿐 반복 패턴의 공간적 모호성은 해결하지 못합니다. 이 방법은 primary action 직전의 화질 보강이지 새로운 위치 결정자가 아닙니다.

## 4. 실험 공통 설계

### 4.1 구현 위치와 변경 경계

- production `poc/workflow_3/align/matching/engine.py`는 수정하지 않습니다.
- 새 registration helper와 result dataclass는 `poc/workflow_2/`의 lab-only module에 둡니다. 현재 `ensemble_lab.py`의 opt-in 패턴, env/in-code default, no-`argparse` 규칙을 따릅니다.
- golden driver는 새 결과를 추가 column으로만 저장합니다. 기존 `rank1`, `in_topk`, `best_xy`, `scale`, `align_offset_xy` 의미와 baseline 열은 바꾸지 않습니다.
- live `align_fail_correct.py`에는 offline acceptance 전 어떠한 import도 넣지 않습니다.

### 4.2 동일 조건 A/B matrix

| Arm | 후보 생성 | 후보 후단 | 답할 질문 |
|---|---|---|---|
| B0 | 현재 consensus 3-channel RRF | 현재 NCC/결정 정책 | 기준선 |
| A1 | B0와 동일 | ECC | 국소 registration이 rank 오류를 줄이는가 |
| A2 | B0와 동일 | SIFT-RANSAC 또는 AKAZE-RANSAC | 기하 검증이 wrong local peak를 줄이는가 |
| A3 | B0와 동일 | phase correlation | sub-pixel translation이 유효한가 |
| A4 | B0와 동일 | MIND-like verifier | contrast drift에만 효과가 있는가 |
| A5 | B0와 동일 | LoFTR 또는 LightGlue verifier | learned correspondence가 SEM에 이득이 있는가 |
| P1 | MIND-like proposer | 가장 좋은 후단 verifier | proposer miss도 줄일 수 있는가 |

candidate verifier arm은 `gt_in_topk=True` row를 우선 분석합니다. proposer arm만 전체 `in_topk`와 `rank1`을 다시 비교합니다.

### 4.3 데이터 누수와 결과 기록 방지

- 기존 leave-one-out consensus 규칙을 유지합니다. 평가 대상 S frame이 template에 들어가면 안 됩니다.
- recipe 단위와 OM/SEM 단위로 분리한 결과를 냅니다. 가능하면 equipment/time bucket도 보조 표로 분리해 한 장비 또는 한 시점의 연속 image가 결과를 부풀리지 않는지 확인합니다.
- hyperparameter와 reject threshold는 development split에서만 고정하고, holdout split은 마지막 한 번만 확인합니다.
- 각 row에 `baseline_rank`, `candidate_rank`, `gt_in_topk`, `error_px`, `ECC`, `phase_response`, `inlier_count`, `inlier_ratio`, `inlier_coverage`, `transform`, `reject_reason`, `runtime_ms`를 남깁니다. false positive overlay도 저장합니다.

### 4.4 채택 기준

새 method는 다음을 모두 만족할 때만 live shadow mode 후보가 됩니다.

1. 같은 leave-one-out set에서 baseline보다 `rank1`이 개선되고, recipe-level paired bootstrap 신뢰구간이 0보다 큽니다.
2. SEM 결과를 별도로 보고, 전체 평균 상승이 OM의 쉬운 row만으로 만들어진 것이 아님을 보입니다.
3. `gt_in_topk=True` subset에서 verifier의 실질 개선이 보이며, proposer arm이면 `in_topk`도 함께 개선하거나 적어도 악화 원인을 설명합니다.
4. high-confidence off-target, `not_distinctive`를 무시한 action, coordinate-contract 위반이 증가하지 않습니다.
5. office 환경의 worst-case runtime이 primary safety budget 안에 있고, 실패/timeout은 baseline 또는 `NO_ACTION`으로 deterministically fallback합니다.

이 기준을 못 넘으면 score를 합성해 억지로 선택하지 않고, `not_distinctive`/재등록 분류를 강화합니다.

## 4.5 구현 현황 (2026-07-20)

`registration_lab.py` + `golden_registration_eval_cond.py` 에 아래가 구현되어 있고, 합성 self-test(24개)와 드라이버 합성 테스트(pytest 5개)를 통과했습니다. 오피스 golden A/B 는 아직 미실행입니다.

- P0-A `ecc`, P0-B `sift`/`akaze`, P1-C `phase`(raw) — 최초 구현분.
- P1-C 확장 `grad_phase`(Sobel magnitude 표상 phase), cascade `phase_ecc`(phase 전역 추정으로 ECC warp 초기화 — ECC 의 좁은 capture range 보완, 합성에서 20px 초기 오차 복원 확인).
- P1-D `mind`(MIND-like self-similarity, score-only verifier — 문서 설계 단계 1 그대로, shift 없음).
- `fuse` 의사-arm: fallback 아닌 arm 들의 재정렬 순열 RRF 합의(`rrf_fuse_orders`) — §4.3 의 단일-arm false-positive 상쇄 장치. 드라이버가 arm 2개 이상이면 자동 집계(`ALIGN_REG_FUSE=0` 으로 끔).
- 드라이버 summary 에 hit 행 GT 거리 중앙값(`err_b0_med_px`→`err_ref_med_px`)을 추가 — rank 지표와 별개로 sub-pixel 정밀화 효과를 분리 관찰.

## 5. 추천 실행 순서

1. **ECC lab**: candidate crop extraction과 transform-to-`best_xy` contract부터 synthetic unit test로 고정하고, golden LOO에서 `translation` arm만 실행합니다. 효과가 있을 때만 Euclidean/affine을 추가합니다.
2. **SIFT/AKAZE-RANSAC lab**: same top-K에서 inlier spatial-coverage gate를 포함해 ECC와 독립 A/B합니다. ORB를 제거하지 않습니다.
3. **phase correlation lab**: Hann window와 response를 포함해 sub-pixel refinement만 시험합니다. ECC보다 단순하고 빠른 대안인지 확인합니다.
4. **failure atlas 작성**: 세 P0/P1 결과를 proposer miss, rank error, contrast drift, non-distinctive로 묶어 visual overlay와 함께 저장합니다. 이 단계에서 재등록 대상 비율을 다시 산정합니다.
5. **MIND-like verifier**: contrast drift bucket에서만 value가 확인되면 dense proposer로 확장합니다.
6. **learned matcher feasibility**: office-local weight, license, latency, data retention 조건이 충족될 때만 LoFTR 또는 LightGlue 하나를 선택해 offline verifier로 검증합니다.

## 6. 참고 자료

- [OpenCV `findTransformECC`: initial transform을 개선하는 area-based registration](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- [OpenCV `phaseCorrelate`: Fourier-domain translation과 response](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)
- [OpenCV `estimateAffinePartial2D`: RANSAC limited-affine transform](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [Lowe (2004), SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [Heinrich et al. (2012), MIND local self-similarity descriptor](https://www.sciencedirect.com/science/article/pii/S1361841512000643)
- [Sun et al. (2021), LoFTR](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)
- [Lindenberger et al. (2023), LightGlue](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)

외부 자료는 각 방법의 일반적인 기제와 제약을 뒷받침할 뿐, CD-SEM align-key 성능을 보장하지 않습니다. 이 프로젝트에서의 유효성은 위의 golden leave-one-out과 오피스 실데이터 검증으로만 판단합니다.
