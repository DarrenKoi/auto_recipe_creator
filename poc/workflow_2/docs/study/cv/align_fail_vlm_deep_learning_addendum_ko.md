---
status: proposed
date: 2026-07-10
scope: office-only research and offline/shadow evaluation plan
related:
  - align_fail_cv_methods_research_ko.md
  - ../../../../workflow_3/docs/specs/vlm_roi_prior_design_260616.md
  - ../../../../workflow_3/docs/study/align_point_accuracy_ml_vlm_research_260616.md
---

# Align Fail 보정의 VLM·Deep Learning 추가 연구

## 결론

VLM과 Deep Learning을 쓸 여지는 있습니다. 그러나 둘 다 "이미지 한 장을 보고 좌표를 내는 자동 클릭기"로 쓰면 안 됩니다. 현재 `workflow_3`의 안전 경계는 다음처럼 유지해야 합니다.

```text
VLM / Deep model: ROI, 후보 점수, 품질/이상 상태, abstain 근거
Deterministic CV: 후보 좌표, scale/offset 보정, distinctiveness, 최종 click 좌표
Safety router: CV 저신뢰 또는 모델 불일치 -> full-FOV fallback / engineer_review
```

현 시점의 권장 순서는 다음과 같습니다.

1. **VLM ROI prior의 office A/B**: 이미 `roi_hint`와 throwaway VLM probe가 있어 가장 낮은 구현 비용으로 검증할 수 있습니다. VLM 실패는 반드시 full-FOV CV로 폴백합니다.
2. **작은 domain patch-pair ranker**: golden의 독립 target-point 라벨로 positive와 hard negative를 만들고, top-K 안 wrong-local-peak만 점수화합니다. 단일 VLM보다 실제 SEM texture에 맞출 가능성이 높습니다.
3. **frozen DINOv2 descriptor와 SuperPoint/LightGlue 또는 LoFTR/RoMa의 offline 비교**: natural-image pretrained model을 바로 신뢰하지 않고, candidate proposer/verifier로 같은 평가 집합에서만 비교합니다.
4. **VLM candidate-card abstention**: 후보 번호가 표시된 panel에서 VLM이 "CV 후보가 확실히 잘못됐는가"만 판단하는 shadow 안전 gate입니다. VLM이 고른 번호로 좌표를 바꾸지 않습니다.

가장 중요한 결론은 변하지 않습니다. SEM의 반복 패턴으로 진짜 위치와 decoy가 정보적으로 구별되지 않으면 어떤 VLM이나 deep matcher도 근거 없는 확신을 낼 수 있습니다. 이 경우 기대하는 정답은 `NO_ACTION`/`engineer_review`와 align-key 재등록입니다.

> 이 문서는 `docs/project_progress/03_workflow_2.md`의 현재 consensus benchmark를 기준선으로 삼습니다. 2026-06-16의 workflow_3 연구 문서는 방향성 및 기존 prototype의 근거로 참조하되, 그 문서의 당시 수치를 현재 baseline 수치로 재사용하지 않습니다.

## 1. 이미 있는 자산과 변경하면 안 되는 경계

### 1.1 현재 VLM 자산

| 자산 | 현재 상태 | 이 보고서에서의 역할 |
|---|---|---|
| `flask_api/vlm_serve/config.py` | UI-Venus, MAI-UI, UI-TARS, PaddleOCR-VL, GOT-OCR service 등록 | 사내 proxy/local inference 재사용. 외부 API 의존은 별도 승인 전 금지. |
| `poc/workflow_2/vlm_align_key_box.py` | 단일 image align-key bbox feasibility probe | ROI prior prompt의 출발점. matcher 좌표에 직접 연결하지 않음. |
| `poc/workflow_2/vlm_align_key_region.py` | reference + live scene multi-image region probe | cross-image VLM ROI의 office-only evidence generator. |
| `poc/workflow_3/docs/specs/vlm_roi_prior_design_260616.md` | VLM ROI -> `roi_hint` -> CV full-FOV fallback 설계 | 가장 먼저 검증할 VLM 통합 후보. |
| `poc/workflow_3/align/correction.py` | VLM은 OK dialog locator에만 사용, matcher는 CV | `best_xy`와 `align_offset_xy`의 좌표 권한을 유지할 기준. |

UI-Venus/MAI-UI는 UI grounding 용도로 검증된 모델입니다. SEM texture correspondence 능력은 아직 증명되지 않았으므로, 모델 이름이나 response confidence를 SEM 정확도의 근거로 쓰면 안 됩니다.

### 1.2 불변 안전 조건

- VLM/deep model은 raw screen 좌표 또는 `best_xy`를 직접 authoritative하게 반환하지 않습니다.
- 후보를 바꾸는 모든 method는 기존 CV candidate의 **frame-coordinate** 안에서만 작동하고, `align_offset_xy * scale` 보정을 거칩니다.
- VLM ROI, learned matcher, deep ranker의 오류/timeout/low-confidence는 기존 full-FOV CV 또는 `engineer_review`로 돌아갑니다.
- live broad-search에는 무거운 model을 넣지 않습니다. paused primary frame의 offline 또는 shadow path만 대상으로 합니다.
- 사내 align image는 승인된 local/office service 밖으로 전송하지 않습니다. 현재 `vlm_align_key_region.py`의 gateway 호출도 데이터 보존·망 경계가 명시되기 전에는 연구 실행 대상이 아닙니다.

## 2. VLM 방법

### P0-A. 기존 VLM ROI prior를 먼저 실제로 측정

**질문**: VLM이 reference key 또는 live FOV에서 넉넉한 영역만 반환해도, CV의 far/veryfar decoy가 줄어 `rank1`이 개선되는가?

현재 엔진은 이미 `roi_hint=(x, y, w, h)`를 받고 결과 좌표를 full frame 좌표로 되돌립니다. 따라서 새로운 matcher가 아니라 다음 두 arm의 비교입니다.

| Arm | VLM 출력 | CV 동작 | 실패 시 |
|---|---|---|---|
| B0 | 없음 | 기존 full-FOV consensus matcher | 기존 동작 |
| V1 | single-image key ROI | ROI matcher | low/invalid ROI면 B0 재실행 |
| V2 | reference + live multi-image ROI | ROI matcher | low/invalid ROI면 B0 재실행 |

**기록값**: `roi_found`, `roi_area_fraction`, GT가 padded ROI 안에 있는지, ROI-first의 rank, full-FOV fallback 여부, 최종 rank, VLM latency, response parse/timeout reason.

**합격 조건**: full-FOV baseline 대비 `in_topk` 손실이 없고, SEM `rank1` 또는 far/veryfar 오류가 개선되어야 합니다. ROI가 GT를 자주 배제하지만 fallback이 우연히 회복하는 경우는 가속/안전 효과가 없으므로 채택하지 않습니다.

VLM visual grounding에 region mark를 얹는 접근 자체는 Set-of-Mark(SoM) 연구가 보여 주지만, 해당 결과는 일반 visual grounding benchmark의 결과입니다. CD-SEM에서는 ROI recall을 별도 측정해야 합니다. [SoM 연구](https://arxiv.org/abs/2310.11441)

### P1-B. 후보 번호 panel을 이용한 VLM abstention gate

**목표**: VLM이 후보를 "선택"하도록 만들지 않고, CV가 제시한 top-K의 위험한 오선택을 보류할 근거가 있는지 shadow에서 확인합니다.

**입력 panel**

1. reference consensus/template 1장
2. live FOV 위 CV top-K center에 `A`~`H` 표기한 overview 1장
3. 각 후보의 raw crop과 edge overlay를 같은 크기로 배열한 candidate-card 1장

**VLM 출력 contract**

```json
{
  "baseline_candidate": "A",
  "verdict": "supports" | "ambiguous" | "reject" | "reference_not_visible",
  "confidence": 0.0,
  "evidence_type": ["corner", "box", "cross", "repeating_texture", "insufficient_detail"]
}
```

VLM은 좌표를 출력하지 않고, `A`는 항상 CV rank-1입니다. `supports`가 아니거나 동일 input 반복 응답이 불일치하면 action을 더 강하게 만드는 대신 `engineer_review` 후보로만 기록합니다. VLM이 `B`를 더 낫다고 말해도 곧바로 `B`로 reposition하지 않습니다.

**왜 이 형식인가**: 원래 이미지 전체에서 자유 좌표를 묻게 되면 pixel grounding error와 candidate ranking error가 섞입니다. 후보 ID로 제한하면 VLM을 "고위험 action 승인"이 아니라 독립된 ambiguity witness로 평가할 수 있습니다.

**평가**: `reject`가 실제 off-target row에서 얼마나 자주 나오는지(recall), true rank-1을 얼마나 잘못 막는지(false abstain), response consistency, latency를 측정합니다. 최종 목표는 rank-1 증가가 아니라 **high-confidence off-target 감소**입니다.

### P1-C. VLM UI/scene-state quality gate

UI-oriented VLM이 SEM 구조 자체를 읽는 것보다 적합할 수 있는 역할은 다음 state classification입니다.

- SEM Monitor viewport가 완전히 보이는지
- modal dialog, tooltip, selection overlay, zoom/pan UI가 FOV를 가리는지
- OM/SEM label 또는 scale bar가 reference routing과 모순되는지
- key가 보이지 않는지, 아니면 화면은 맞지만 periodic texture라 판별할 수 없는지

출력은 `usable_for_matching`, `occluded`, `wrong_panel_or_mode`, `ambiguous_scene` 같은 strict JSON enum이어야 합니다. `usable_for_matching=false`면 VLM은 click을 막거나 fallback을 요청할 뿐 target location을 만들지 않습니다. 이 역할은 project의 crop-first OCR, VLM OK-button locator, key-visibility gate와 일관됩니다.

### P2-D. semiconductor VLM fine-tuning은 분류/설명에 한정

반도체 electron micrograph에 instruction-tuned small VLM을 적용한 연구는 domain-specific instruction data로 VQA와 classification을 맞추는 방향을 제시합니다. [Srinivas et al., 2024](https://arxiv.org/abs/2409.07463) 이 접근은 아래 보조 task에는 후보가 될 수 있습니다.

- `key_visible` / `key_absent` / `repeating_non_distinctive` / `overlay_occluded` 분류
- engineer_review 시 evidence 설명과 debug artifact 태깅
- manual review queue의 우선순위 분류

그러나 "같은 key의 정확한 pixel 좌표"는 이 논문의 task가 아닙니다. 사내 사람이 확인한 frame-state label이 충분히 쌓이기 전에는 VLM fine-tuning을 align-point locator로 확장하지 않습니다.

## 3. Deep Learning 방법

### P0-E. 작은 patch-pair ranker: 가장 현실적인 학습 실험

**문제 정의**: full frame에서 좌표를 회귀하지 않습니다. 현재 CV가 낸 top-K 후보 `c_i`와 reference template `t`의 쌍 `(t, crop(c_i))`이 정답 target인지 점수화합니다.

**라벨 생성**

- positive: matcher 결과와 독립적으로 기록된 target point/crosshair 또는 golden condition에서 잘라낸 crop만 사용합니다.
- hard negative: 같은 frame의 top-K 중 GT tolerance 밖인 Chamfer/RRF high-score 후보를 우선 사용합니다.
- unknown: independent target이 없는 E frame은 supervised train/metric 계산에서 제외하거나 separate unlabeled pool으로만 씁니다.
- split: recipe 단위로 train/validation/test를 완전히 나누고, 가능하면 equipment와 time bucket holdout도 추가합니다. 같은 recipe의 success frame이 train과 test에 섞이면 consensus와 appearance가 누수됩니다.

**모델 arm**

| Arm | 입력 | 출력 | 목적 |
|---|---|---|---|
| R0 | Chamfer + NCC 기존값 | 순위 | baseline |
| R1 | Siamese small CNN embedding + cosine | pair score | 최소 학습 baseline |
| R2 | shared encoder + correlation/cross-attention head | pair score | 같은 texture의 미세 구조 차이를 직접 비교 |
| R3 | frozen DINOv2 patch features + lightweight MLP | pair score | 먼저 training 없이 foundation descriptor의 전이 가능성 확인 |

Supervised contrastive learning은 같은 class/sample family embedding을 가깝게, 다른 것을 멀게 만드는 objective를 제공하므로, positive와 **같은 FOV hard negative**가 있는 이 문제에 맞습니다. [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

**출력 경계**: ranker의 output은 `learned_pair_score` 하나입니다. 기존 candidate geometry, `distinctive`, second-gap, action gate를 재계산하지 않고 덮어쓰면 안 됩니다. top-K 밖 새 좌표를 만들지 않습니다.

**성공 판정**: `gt_in_topk=true` subset의 SEM rank-1, wrong-local-peak rate, high-confidence off-target를 baseline과 paired 비교합니다. top-K에 없는 정답은 ranker의 실패가 아니라 proposer 한계로 별도 표에 둡니다.

### P1-F. frozen DINOv2 descriptor와 retrieval-guided consensus

DINOv2는 self-supervised visual feature를 제공하는 foundation encoder입니다. [DINOv2](https://arxiv.org/abs/2304.07193) 첫 실험은 fine-tuning 없이 다음 두 가지로 제한합니다.

1. template/live candidate crop의 patch-token similarity를 R3 pair-ranker feature로 사용합니다.
2. current frame과 historical success frame의 global descriptor로 가장 가까운 success cluster를 찾고, 그 **선택된 cluster만** median consensus로 만듭니다.

두 번째는 template bank의 재시도가 아닙니다. 개별 template match를 fuse하지 않고, current appearance와 가까운 S subset을 먼저 고른 뒤 기존 consensus matcher 하나를 실행하는 가설입니다.

**강한 중단 조건**: `docs/project_progress/03_workflow_2.md`에서 individual template bank는 rank-1을 크게 악화시켰습니다. retrieval-guided consensus도 기존 all-S consensus를 유의하게 이기지 못하거나 selected cluster가 너무 작으면 즉시 중단합니다. 자연 이미지로 pretrain된 representation은 EM/장비 domain 간 mismatch가 클 수 있다는 최근 EM VFM 평가 결과도 이 보수적 순서를 뒷받침합니다. [EM foundation-model transfer study](https://arxiv.org/abs/2602.08505)

### P1-G. learned correspondence matcher: sparse와 dense를 분리 비교

| 후보 | 적합한 실패 | 출력 사용법 | 주요 위험 |
|---|---|---|---|
| SuperPoint + LightGlue | 충분한 keypoint가 있으나 ORB가 불안정 | correspondence -> limited-affine RANSAC -> CV candidate verifier | texture가 약하면 keypoint 자체가 부족 |
| LoFTR | low-texture이지만 두 crop에 넓은 overlap | semi-dense correspondence -> transform/inlier score | repeated lattice에서 many-to-one decoy match |
| RoMa | contrast/scale/texture 변화가 큰 pair | dense correspondence confidence -> top-K verifier 또는 proposer | natural-image/DINO prior의 SEM domain shift, GPU 비용 |

SuperPoint는 self-supervised interest point와 descriptor를 한 forward pass에 만드는 방법입니다. [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html) RoMa는 DINOv2 feature와 fine feature pyramid를 조합한 dense matcher입니다. [RoMa](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf) LightGlue와 LoFTR의 기본 후보 성격은 앞선 CV report에 정리했습니다.

**공통 gate**

1. full frame global search가 아니라 CV top-K crop pair에서 먼저 검증합니다.
2. correspondence count가 아니라 limited-affine RANSAC inlier ratio, template-side spatial coverage, reprojection error, transform plausibility를 씁니다.
3. match confidence map이 여러 동등 peak를 보이면 `not_distinctive`를 강화합니다. 가장 큰 confidence 하나를 강제로 선택하지 않습니다.
4. only `gt_in_topk=false` subset에서 proposer 효과를 별도로 보이고, verifier 결과와 섞지 않습니다.

EM registration에 learned feature와 spatial transformer를 사용한 선행 사례는 있으나, serial-section EM의 nonrigid deformation 문제입니다. CD-SEM paused FOV의 align-key에는 처음부터 nonrigid warp를 허용하는 근거가 아닙니다. [ssEMnet](https://arxiv.org/abs/1707.07833)

### P2-H. one-class anomaly/reference-drift gate

이 method는 위치를 찾지 않습니다. 같은 recipe의 정상 S/consensus reference와 비교해 현재 crop 또는 full FOV가 "matching input으로 쓸 수 없는 상태"인지 감지합니다.

- **입력**: recipe별 정상 S crop 몇 장, current candidate crop 또는 paused frame
- **출력**: `in_distribution`, `drifted_but_matchable`, `out_of_distribution`, anomaly heatmap
- **router**: `out_of_distribution`이면 primary auto-act를 강화하지 않고 fallback 또는 engineer_review로 보냅니다.

WinCLIP은 few-normal-shot industrial anomaly classification/segmentation을 다루지만, 그 benchmark 성능을 CD-SEM에 전이할 수는 없습니다. [WinCLIP](https://arxiv.org/abs/2303.14814) 여기서는 local normal-image bank가 있을 때 false certainty를 줄일 수 있는지 검증하는 보조 gate로만 둡니다.

### P3-I. self-supervised pretraining 후 small ranker fine-tune

unlabeled S/E frame이 충분하지만 target label이 적으면, office 내부에서 SimCLR/DINO-style self-supervised pretraining을 하고 P0-E의 pair-ranker만 small supervised data로 fine-tune할 수 있습니다. 장점은 recipe-specific 대형 model 대신 cross-recipe representation을 학습할 수 있다는 점입니다.

단, augmentation은 SEM 물리를 보존해야 합니다. random crop, intensity jitter, 약한 blur는 후보가 될 수 있어도 90-degree rotation, arbitrary elastic warp, synthetic defect insertion은 실제 stage/magnification contract를 깨므로 label-preserving augmentation으로 가정하면 안 됩니다.

## 4. 우선순위 밖 또는 금지할 방법

| 방법 | 판단 | 이유 |
|---|---|---|
| VLM 자유 좌표 출력 또는 VLM이 고른 candidate로 바로 click | 금지 | pixel error와 model variation을 측정/제어할 수 없음 |
| Grounding DINO에 "align key" text prompt | 보류 | align key는 일반 object category가 아니라 recipe-specific fabricated pattern; zero-shot semantic detector의 대상이 아님 |
| SAM/SAM2로 key를 segment해 target center 산출 | 보류 | semantic object boundary가 align target을 뜻하지 않으며, 기존 box/crosshair 처리 실패를 다른 형태로 반복할 위험 |
| diffusion/CycleGAN/super-resolution output을 matcher 또는 click 근거로 사용 | 금지 | 생성된 edge/texture가 실제 wafer evidence가 아닐 수 있음 |
| nonrigid spatial transformer/optical-flow warp를 primary matcher에 사용 | 금지 | paused FOV에 허용할 physical transform을 넘어 wrong peak를 더 잘 맞춰 보이게 할 수 있음 |
| template bank의 개별 score fuse 재시도 | 금지 | 이미 rank-1 붕괴가 확인됨; retrieval-guided single consensus만 제한적으로 다룸 |

## 5. 공통 평가와 rollout

### 5.1 offline 평가 표

모든 arm은 current consensus baseline과 같은 leave-one-out data split에서 다음을 함께 냅니다.

- 전체/OM/SEM `in_topk`, `rank1`, target error pixel
- `gt_in_topk=true` rank-1과 `gt_in_topk=false` proposer recall을 별도 집계
- `wrong_local_peak`, `not_distinctive`, `NO_ACTION`, high-confidence off-target 수
- recipe/equipment/time bucket별 결과와 recipe-level paired bootstrap CI
- per-row model latency, timeout, fallback, model/version/prompt hash

VLM은 추가로 ROI-GT containment, abstain calibration, repeated-input consistency를 기록합니다. 학습 model은 recipe-disjoint test score만 release candidate 지표로 인정합니다.

### 5.2 출시 순서

1. **offline lab**: static artifact와 golden LOO에서 score/ROI만 저장합니다.
2. **office shadow**: live paused frame에서 VLM/deep output을 로그와 overlay로 저장하지만, `best_xy`, move, OK click에 사용하지 않습니다.
3. **router-only pilot**: `occluded`/`out_of_distribution`/model disagreement를 `engineer_review`로 보내는 no-action gate만 제한 적용합니다.
4. **CV-assisted production**: full-FOV fallback, deterministic CV coordinate, post-reposition verification을 포함한 뒤에만 ROI/ranker의 긍정 결과를 반영합니다.

각 단계는 기본 off env flag와 versioned debug artifact를 가져야 하며, failure가 나면 baseline CV 또는 engineer escalation으로 deterministic하게 복귀해야 합니다.

## 6. 다음 실험 순서

1. `vlm_align_key_region.py`의 office artifact로 V1/V2 ROI containment와 full-FOV fallback A/B를 실행합니다.
2. golden data에서 independent target을 가진 row만 추려 positive/hard-negative manifest를 만들고 recipe-disjoint split을 고정합니다.
3. R1 small Siamese baseline과 R3 frozen-DINOv2 baseline을 먼저 비교합니다. 어느 쪽도 `gt_in_topk=true` SEM rank-1을 개선하지 못하면 deep ranker expansion을 멈춥니다.
4. SuperPoint+LightGlue, LoFTR, RoMa를 **같은 top-K crop pair**에서 transform verifier로 비교합니다. full-frame deployment는 이 단계에서 금지합니다.
5. VLM candidate-card abstention을 shadow로 돌려 true match를 막는 비율과 unsafe candidate를 막는 비율을 측정합니다.
6. 위 결과가 누적된 뒤에만 retrieval-guided consensus, anomaly gate, self-supervised pretraining으로 진행합니다.

## 참고 자료

- [Set-of-Mark prompting for visual grounding](https://arxiv.org/abs/2310.11441)
- [DINOv2 self-supervised visual features](https://arxiv.org/abs/2304.07193)
- [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- [SuperPoint](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html)
- [RoMa robust dense matching](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf)
- [WinCLIP few-normal-shot anomaly detection](https://arxiv.org/abs/2303.14814)
- [Domain-specific semiconductor electron-micrograph VLM](https://arxiv.org/abs/2409.07463)
- [EM foundation-model transfer and domain-mismatch evidence](https://arxiv.org/abs/2602.08505)
- [ssEMnet learned-feature registration](https://arxiv.org/abs/1707.07833)

외부 연구는 방법의 출발점일 뿐입니다. 자연 이미지, serial-section EM, biomedical EM, semiconductor CD-SEM의 capture physics와 align-key semantics는 다릅니다. 여기서 제안한 각 model은 office data의 recipe-disjoint offline evaluation과 shadow safety evaluation을 통과하기 전에는 reposition/OK authority를 얻지 못합니다.
