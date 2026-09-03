[DIGEST] **초판(one-shot 전제)의 "YOLO 는 논외" 는 철회한다.** align key 가 recipe 마다 임의의 구조가
아니라 **DFT(Die Fit Target) 라는 반복되는 표준 fiducial** 안/주변에 있다면 학습 가능한 class 가 실재하고,
`cond.txt` 가 이미 **box(LTRB) + keypoint(crosshair)** 를 들고 있으므로 사람 라벨 0 으로 YOLO-pose
라벨이 만들어진다 - 데이터도 class 도 없다던 §1~§3 의 반대는 이 전제에서 무너진다(§8). 새 verdict 는
"안 된다" 가 아니라 **"검증되지 않은 전제 하나에 전부 걸려 있고, 그 전제는 모델 없이 한 시간이면
판정된다"** 이다. 그 전제는 **decoy 반복 구조가 DFT box 밖에 있는가** 다 - 밖이면 DFT 검출이 탐색공간을
잘라 rank-1 이 그냥 오르고(초판이 놓친 진짜 rank-1 mechanism), 안이면 아무리 좋은 detector 도 0 을
더한다. 판정 실험은 학습도 weight 도 필요 없다: golden set 에서 `cond.box_ltrb` 를 **완벽한 DFT
detector 의 oracle** 로 놓고 기존 matcher 의 탐색 범위를 그 박스로 제한해 rank-1 을 다시 재면 된다(§11).
**단, 배포 게이트가 하나 더 늘었다 - Ultralytics 공식 licensing 페이지가 "Use YOLO only internally or
for R&D" 와 "Internal business tools or private company applications" 를 Enterprise License 필요
항목으로 명시한다(§9). fab 사내 사용은 vendor 의 공표된 입장으로는 유료다.** 초판의 나머지 결론
(learned matcher + RANSAC 의 기하 일관성, abstain 안 하는 실패, XFeat probe)은 그대로 유효하되 §8 의
전제가 무너질 때의 **대안 경로**로 격하한다.

---

status: research
date: 2026-09-03
scope: office-only offline feasibility. 코드/데이터 변경 없음. primary source(공식 docs / 논문 / repo) 만 인용
question: "Ultralytics YOLO 로 align point 한 점을 낼 수 있는가"
revision: 2026-09-03 2판 - align key 가 DFT(표준 fiducial) 위에 있다는 사용자 정정 반영. §0/§8~§11 추가,
  §1~§7 은 1판 그대로 보존(전제가 달라졌을 뿐 사실관계는 유효). **§1~§3 의 "class 가 없다/데이터가 없다"
  는 반대는 §8 이 무효화한다** - 읽을 때 §8 을 먼저 볼 것
related:
  - 2026-09-02-B-deep-learning-feasibility.md
  - 2026-09-02-C-training-data-audit.md
  - 2026-09-02-synthesis.md
  - ../adr/0005-whitebox-box-crop-consensus-arm-rejected.md
  - ../adr/0006-template-bank-matcher-rejected-fusion-exhausted.md
  - ../../../../workflow_3/docs/study/align_point_accuracy_ml_vlm_research_260616.md

---

# 브리프 E: YOLO 로 align point 를 낼 수 있는가 (2판 - DFT 전제 반영)

## 0. 결론 먼저 (2판, blunt)

**1판은 틀린 질문에 정확히 답했다.** 1판은 "align key = recipe 마다 임의의 구조" 를 전제로 깔고
"class 가 없으니 YOLO 는 논외" 라고 결론냈다. 그 전제가 틀렸다면 결론도 같이 무너지고, 실제로
무너진다 - align key 가 **DFT 라는 재발하는 표준 fiducial** 위에 있다면 class 는 **하나**이고,
데이터는 recipe 를 가로질러 pool 되며(298 recipe × recipe 당 rcp 2 + S 몇 장), 라벨은 `cond.txt` 의
box + crosshair 로 **이미 존재한다**. 이 조건에서 "YOLO-pose 로 DFT box 를 잡고 그 안의 align point 를
keypoint 로 회귀" 는 그럴듯한 설계이고, 진지하게 다뤄야 한다. §8 이 그 작업이다.

**그러나 그럴듯함이 곧 레버는 아니다.** 우리 병목은 여전히 rank-1(≈0.5) 이고, DFT detector 가 그것을
올리는 경로는 하나뿐이다 - **탐색공간 축소**. 즉 "정답과 헷갈리는 반복 line(decoy)들이 DFT box **밖**에
있어서, box 안으로 자르면 decoy 가 사라진다" 가 참이어야 한다. 참이면 rank-1 은 **matcher 를 하나도
안 바꾸고** 오른다(1판이 놓친 진짜 mechanism 이다 - 1판은 이 경로를 "recall 축" 으로 잘못 분류했다).
거짓이면 - 즉 aperture problem 이 DFT **안에서** 일어나면 - 완벽한 detector 도 rank-1 에 0 을 더한다.

**그리고 이건 모델을 만들기 전에, 학습도 weight 도 없이 판정된다.** `cond.box_ltrb` 를 "완벽한 DFT
detector" 로 놓고 기존 matcher 의 탐색을 그 박스로 제한해 golden rank-1 을 다시 재면 끝이다(§11).
oracle 이 안 올리면 실제 detector 는 그보다 못하다 - 학습 프로그램 전체가 그 한 줄로 닫힌다.
**이 실험이 §7 의 XFeat probe 보다 싸고 결정적이다.** 순서를 바꾼다.

**부수적으로 새 blocker 가 하나 생겼다.** Ultralytics 공식 licensing 페이지가 "Use YOLO only
internally or for R&D" 조차 Enterprise License 대상으로 명시한다(§9). 기술 판정과 무관하게 fab 배포
전에 Legal 을 거쳐야 하는 항목이고, 1판에서 내가 빠뜨렸다.

**1판(§1~§7)은 지우지 않는다.** 거기 적힌 사실 - 지원 task set, 최소 데이터 권장치, learned matcher
라이선스/크기, microscopy 실측, SuperPoint 비상업 조항 - 은 전제와 무관하게 참이고, §8 의 전제가
무너졌을 때 **가야 할 대안 경로**가 정확히 그것이다. 다만 §1~§3 의 "298 recipe = 298 class" 논증은
DFT 전제에서 **무효**이므로, 그 절들은 "one-shot 프레이밍에서는 이랬다" 는 기록으로 읽는다.

---

## 1. Ultralytics YOLO 가 실제로 지원하는 task set

### 1.1 shipped task 목록

공식 [Tasks](https://docs.ultralytics.com/tasks/) 페이지가 YOLO26 기준으로 나열하는 것은 다음뿐이다.

| task | 공식 문서의 정의 | 출력 |
|---|---|---|
| Detection | "identifying objects in an image or video frame and drawing bounding boxes around them" | box + class |
| Instance Segmentation | "producing pixel-level masks for each object" | mask |
| Semantic Segmentation | "assigns a class label to every pixel in an image, producing a dense class map" | dense class map |
| Monocular Depth | "predicts a per-pixel depth map in meters from a single RGB image" | depth map |
| Classification | "categorizing entire images based on their content" | class |
| Pose | "detects specific keypoints in images or video frames" | box + keypoints |
| OBB | "adding an orientation angle to better locate rotated objects" | rotated box |

**one-shot / template-conditioned / reference-image-conditioned 라는 mode 는 이 페이지에 없다.**
전부 학습 시점에 class 집합이 고정된 closed-set task 다.

### 1.2 YOLO-World: text prompt 전용

[YOLO-World](https://docs.ultralytics.com/models/yolo-world/) 는 "prompt-then-detect" 로
**text** class 를 받는다. 같은 페이지가 명시적으로 넘긴다 - "For open-vocabulary work that also needs
instance masks, visual prompts, or a prompt-free mode, see YOLOE." zero-shot COCO 는
YOLOv8x-worldv2 기준 mAP 47.1. **우리 target 은 말로 표현 불가**("이 recipe 의 등록된 align key")
이므로 text prompt 는 애초에 배선이 안 된다.

### 1.3 YOLOE: visual prompt 는 있다 - 그러나 proposer 다

[YOLOE 문서](https://docs.ultralytics.com/models/yoloe/)가 세 mode 를 표로 정리한다. visual prompt 행
그대로:

> Visual prompt | `*-seg.pt` | Example boxes on a reference image | Generic `object0`, `object1`, … |
> You cannot put the target into words: a specific part, logo, or defect

API 도 우리가 원하는 모양이다 - `visual_prompts={"bboxes":…, "cls":…}` 를 target image 에 주거나,
**별도 reference image** 를 `refer_image=` 로 넘길 수 있다("Or on a separate reference image passed as
`refer_image`, in which case `bboxes` and `cls` describe objects in that reference, not in the target").
구조는 SAVPE 다 - 공식 설명: "Semantic-Activated Visual Prompt Encoder (SAVPE) encodes semantic and
activation features from an example box, **conditioning the model on objects that look like it**.
This is the one-shot path for targets that are hard to name, such as a logo or a specific part."
(원 논문 [arXiv:2503.07465](https://arxiv.org/abs/2503.07465): "employs decoupled semantic and
activation branches to bring improved visual embedding and accuracy with minimal complexity";
LVIS 에서 "YOLOE-v8-S surpasses YOLO-Worldv2-S by 3.5 AP" with "3x less training cost".)

문서가 공개한 LVIS minival zero-shot 수치(각 cell 은 `text prompt / visual prompt`):

| Model | mAP50-95 | mAP_r (rare) | params (M) |
|---|---|---|---|
| YOLOE-26n | 24.7 / **21.9** | 20.5 / 17.6 | 3.9 / 3.1 |
| YOLOE-26s | 30.8 / **28.6** | 23.9 / 25.1 | 10.7 / 11.0 |
| YOLOE-26m | 35.4 / **33.9** | 31.1 / 33.4 | 21.3 / 25.1 |
| YOLOE-26l | 37.8 / **36.3** | 35.1 / 37.6 | 25.5 / 29.3 |
| YOLOE-26x | 40.6 / **38.5** | 37.4 / 35.3 | 55.2 / 65.2 |

**visual prompt 가 text prompt 보다 모든 scale 에서 낮다.** 공식 Limitations 절이 못 박는 문장들:

- "Zero-shot accuracy is well below a model trained on your classes. The prompted checkpoints land
  roughly in the 22-40 mAP band on LVIS minival … **Reach for YOLOE to cover classes you cannot train
  for, not to replace training.**"
- "Rare categories are the weak spot."
- "Visual prompts do not carry your labels. The class IDs in `visual_prompts` group examples together;
  the model reports them as `object0`, `object1`, and so on."
- Deployment: "Inference needs an NVIDIA GPU with 4-8 GB of VRAM."

**해석.** YOLOE visual prompt 는 우리 task 의 *형태* 는 만족한다(reference crop → target 에서 찾기).
그러나 출력은 confidence 가 붙은 box **집합**이고, 그 confidence 는 "example 과 얼마나 닮았나" 다.
반복 line 위에서 정답과 decoy 는 닮은 정도가 같다. 즉 **후보를 더 뽑아 줄 수는 있어도 순위를 가를
근거를 새로 만들지 못한다.** 게다가 4~8 GB VRAM 상주 프로세스를 하나 더 요구하는데, 호스트 RAM
16 GB 에 vLLM 3개가 이미 상주한 조건에서 이건 그 자체로 blocker 다.

참고로 Ultralytics 가 함께 배포하는 [SAM 3](https://docs.ultralytics.com/models/sam-3/) 는 image
exemplar("Bounding boxes around example objects (positive or negative) for fast generalization")를 받아
개념적으로는 가장 가깝지만, 문서 표에 **473.6 M params / 3450 MB / 2921 ms/im (NVIDIA RTX PRO 6000)**
로 적혀 있다. 실시간 loop 은 물론이고 offline bench 로도 무겁다.

---

## 2. 문서화된 최소 학습 데이터

folklore 가 아니라 공식 문서에 있는 수치는 두 곳에서 일치한다.

- [Data Collection and Annotation FAQ](https://docs.ultralytics.com/guides/data-collection-and-annotation/):
  "A few hundred annotated objects per class is enough to start experimenting with transfer learning,
  but for reliable real-world performance Ultralytics recommends **at least 1,500 images and 10,000
  labeled instances per class**. Pair a sufficiently large dataset with a reasonable training schedule -
  around 300 epochs is a common starting point."
- [Tips for Best Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/):
  "**≥ 1500 images per class recommended**", "**≥ 10000 instances (labeled objects) per class
  recommended**", "We recommend about 0-10% background images to help reduce FPs (COCO has 1000
  background images for reference, 1% of the total)."

**우리 데이터와의 거리.** recipe 를 class 로 놓으면 class 당 rcp 2 장 + S 3~10 장이다. 298 recipe 중
135 개가 S=3 장이다. 권장치의 **1/300 수준**이며, "a few hundred … to start experimenting" 라는 가장
느슨한 기준에도 두 자릿수 못 미친다. transfer learning 으로도 메워지는 격차가 아니다.

recipe 를 class 로 안 쓰고 "align key" 라는 **단일 class** 로 놓는 우회는 데이터 문제는 완화하지만
task 를 바꿔 버린다 - "align key 같은 것이 어디 있나" 를 배우는 detector 는 한 frame 안의 여러 반복
구조를 전부 찾아 줄 뿐, **그중 어느 것이 이 recipe 의 것인가** 를 답하지 않는다. 그 질문이 정확히
우리 rank-1 병목이다.

---

## 3. pose head 를 1점 regressor 로 쓰는 것 (1판; **§10.1 이 갱신**)

> **2판 주석.** 아래 마지막 문단의 "pose 도 class 를 고정한다 → recipe 마다 다른 class" 는
> DFT 전제에서 **무효**다(class 는 1개). 라벨 포맷/`kpt_shape`/visibility 에 대한 사실은 유효하며
> §10.1 이 그 위에 "1점은 스펙상 표현 가능" 과 visibility flag 의 실용적 쓸모를 더한다.


**구조적으로 가능하나 문서화된 use case 가 아니고, bbox 를 없애 주지도 않는다.**

- 라벨 포맷([Pose Datasets](https://docs.ultralytics.com/datasets/pose/)):
  `<class-index> <x> <y> <width> <height> <px1> <py1> … <pxn> <pyn>` -
  **bounding box 가 필수 필드다.** keypoint 만 주는 포맷은 없다. 즉 "점 하나만 라벨" 로는 학습이 안 되고
  bbox 를 지어내야 한다(우리는 human annotation 이 없고 cond.txt 의 crosshair 좌표뿐이다).
- `kpt_shape: [17, 3] # number of keypoints, number of dims` 는 dataset YAML 에서 설정 가능하므로
  `[1, 2]` 도 구조적으로 유효하다. 그러나 **1-keypoint 사용 사례를 다룬 공식 문서는 찾지 못했다
  (primary source not found).** 배포된 pose dataset 은 COCO-Pose 17점(58,945 장), Dog-Pose 24점,
  Hand Keypoints 21점(26,768 장), Tiger-Pose 12점(263 장)뿐이다.
- pose task 에 대한 **최소 이미지 수 FAQ 도 공식 문서에 없다(primary source not found).** §2 의
  일반 권장치(1,500/class)가 유일한 기준선이다.

그리고 근본 문제는 그대로다 - pose 도 class 를 고정한다. "이 recipe 의 align point" 를 keypoint 로
정의하는 순간 recipe 마다 다른 class 가 되어 §2 의 데이터 벽으로 돌아간다.

---

## 4. 형태가 실제로 맞는 계열: learned matching

우리 task 는 detection 이 아니라 **one-shot correspondence** 다. 그 계열의 primary source 정리.

| 방법 | license (primary) | 배포 weight 크기 | zero-shot(도메인 학습 불필요) | 출력 | 저텍스처/반복 구조 근거 |
|---|---|---|---|---|---|
| **XFeat** (CVPR'24) | [Apache-2.0](https://github.com/verlab/accelerated_features/blob/main/LICENSE) | `xfeat.pt` **6.2 MB** (+`xfeat-lighterglue.pt` 10.8 MB) | O | keypoint heatmap + 64-D descriptor + reliability heatmap; sparse 또는 semi-dense **집합** | "much better robustness to viewpoint and illumination changes than classic local features as ORB and SIFT" (README). textureless 에 대한 **정량 주장은 없음** |
| **LightGlue** (+SuperPoint/DISK/ALIKED/SIFT) | [Apache-2.0](https://github.com/cvg/LightGlue) (코드+LightGlue weight) | feature 별 **~47.6 MB** | O | "the indices of corresponding points" = 대응 **집합** | 논문이 "adaptive to the difficulty of the problem" 이라고만 기술([arXiv:2306.13643](https://arxiv.org/abs/2306.13643)). 반복 구조 언급 없음 |
| **LoFTR** | [Apache-2.0](https://github.com/zju3dv/LoFTR/blob/master/LICENSE) | - | O | semi-dense 대응 집합(sub-pixel refine) | **저자 주장 있음**: "LoFTR can extract high-quality semi-dense matches even in indistinctive regions with **low-textures, motion blur, or repetitive patterns**" ([project page](https://zju3dv.github.io/loftr/)) |
| **EfficientLoFTR** | [Apache-2.0](https://github.com/zju3dv/EfficientLoFTR) | - | O | 동일 + confidence | "~2.5x faster than LoFTR", "can surpass … SuperPoint + LightGlue in terms of speed" ([arXiv:2403.04765](https://arxiv.org/abs/2403.04765)). README: "trained on MegaDepth and works best for outdoor scenes. There may be a domain gap for indoor environments." |
| **RoMa** | [MIT](https://github.com/Parskatt/RoMa/blob/main/LICENSE) (DINOv2 부분은 Apache-2) | - (frozen **DINOv2 ViT-L** 백본 포함) | O | **dense warp + certainty**; "transformer match decoder that predicts anchor probabilities, enabling … multimodality" ([arXiv:2305.15404](https://arxiv.org/abs/2305.15404)) | 논문이 DINOv2 frozen feature 를 "significantly more robust than local features trained from scratch" 로 기술. 반복 구조 정량 근거 없음 |
| **MASt3R** | [CC BY-NC-SA 4.0 - **비상업**](https://github.com/naver/mast3r) (checkpoint 는 추가로 training dataset license 누적, "The mapfree dataset license in particular is very restrictive") | ViT-L enc / ViT-B dec | O | 2D-2D match + 3D point | **기각.** 라이선스가 fab 배포를 막고, 평면 wafer 에서 3D grounding 이 퇴화한다(브리프 B 와 동일 판단) |
| **MatchAnything** (TPAMI'26) | **LICENSE 파일 없음** - GitHub API `license: null`, repo 내용물이 `README.md` + `video` 뿐([repo](https://github.com/zju3dv/MatchAnything)) | HuggingFace 배포 | O (RoMa/ELoFTR 를 cross-modal 재학습) | base matcher 와 동일 | §5 참조 |

**부수 라이선스 함정 (배포 전 반드시 확인).** LightGlue README 원문:

> "The pre-trained weights of LightGlue and the code provided in this repository are released under the
> Apache-2.0 license. DISK follows this license as well but **SuperPoint follows a different, restrictive
> license** (this includes its pre-trained weights and its inference file). ALIKED was published under a
> BSD-3-Clause license."

그 "restrictive license" 실물([Magic Leap LICENSE](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE))
첫 줄이 **"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"** 이고
"The Software may be used for your own noncommercial internal research purposes" 다.
**사내 fab 라인 투입은 이 조항에 걸린다.** 상업 배포 가능한 조합은
**ALIKED(BSD-3) 또는 DISK(Apache-2.0) + LightGlue(Apache-2.0)**, 혹은 **XFeat 단독(Apache-2.0)** 이다.
(DISK/ALIKED 라이선스는 GitHub API 로 각각 `Apache-2.0`, `BSD-3-Clause` 확인.)

**공통 성질 두 가지 - 우리 task 와의 정합.**

1. **전부 zero-shot 이다.** per-recipe 학습이 없다. 298 recipe / 신규 recipe 문제가 통째로 사라진다.
   이것이 YOLO 계열과의 결정적 차이다.
2. **전부 "한 점" 이 아니라 "대응 집합" 을 낸다.** 이것은 결함이 아니라 우리에게 필요한 바로 그
   성질이다 - 집합에 `cv2.estimateAffinePartial2D(…, RANSAC)` 같은 기하 모델을 씌우면
   **여러 대응점이 같은 변환에 투표해야만** inlier 가 되고, 반복 구조의 decoy 들은 *서로 다른*
   변환에 투표하므로 상쇄된다. 지금 파이프라인(후보를 독립적으로 NCC/MIND/ECC rescore)에는 이
   메커니즘이 없다. **ADR 0006 이 "세 fusion 이 같은 벽" 이라 한 그 세 방법은 전부 member 를 독립
   점수로 합치는 방법이었다** - 기하 일관성은 아직 시도된 적이 없다.

**정직한 경고.** LoFTR 의 "repetitive patterns" 문장은 **저자 자신의 자연영상 주장**이지 SEM 실측이
아니다. 그리고 서베이 [Deep Learning Reforms Image Matching](https://arxiv.org/html/2506.04619v1) 은
반대 방향을 적는다 - "the performance ceiling of learnable sparse matchers is inherently limited by
detected keypoint quality, yet robust and repeatable detection remains challenging - particularly in
low-texture scenes", 그리고 dense matcher 에 대해 "significantly slower than sparse or semi-dense ones
and remain impractical for high-resolution cases".

---

## 5. 도메인 전이 증거 - 얇다, 그러나 0 은 아니다

**CD-SEM align key / recipe alignment 에 learned matcher 를 적용한 문헌은 찾지 못했다
(primary source not found).** 반도체 + YOLO 조합의 문헌은 전부 **고정 class 결함 검출**이다 -
예: [Optimizing YOLOv7 for Semiconductor Defect Detection](https://arxiv.org/abs/2302.09565) 은
"semiconductor line space pattern defects" 의 **defect class 집합**에 대한 mAP 를 개선한 연구다.
우리 문제(=recipe 마다 다른 구조를 한 점으로 지목)와는 다른 task 다. **이 사실 자체가 §1 의 결론을
독립적으로 뒷받침한다: 업계가 YOLO 를 쓰는 자리는 closed-set 검출이지 one-shot localization 이 아니다.**

가장 가까운 실측은 **correlative materials microscopy** 쪽에 하나 있다 - 프리프린트
["For foundation-model registration in correlative microscopy, cross-modal appearance matters more than
field of view"](https://www.researchsquare.com/article/rs-10311464/v1). 초록 원문에서:

> "Correlative materials microscopy pairs images of one specimen across modalities (**SEM**, EBSD, TEM,
> optical) that share little appearance and can differ in field of view (FOV) by orders of magnitude,
> defeating classical registration. We tested whether a scale-aware pyramidal wrapper around pretrained
> dense matchers (**RoMa, ELoFTR, MatchAnything**) could lift cross-modal registration on **AmalgaMatch
> (187 pairs, 19 subsets)**. A naive tiling pyramid degrades matchers badly: **because they never abstain,
> every tile floods robust estimation with thousands of confident matches (median error 76→1794 px)**.
> A redesigned verified coarse-to-fine wrapper recovers a small but significant gain (**SR@10 0.10→0.12**,
> p = 0.017) …, yet FOV ≤ 5% success stays zero. The largest lever was the backbone: cross-modal-trained
> MatchAnything-RoMa gave the only significant gain over zero-shot (**SR@10 + 0.032**, p = 0.018) …
> Decoder-only fine-tuning cut in-distribution TEM error ~5×, but across eight seeds regressed SR@20
> (0.393→0.26) by **forgetting untrained modalities**, which L2-SP did not fix."

여기서 우리가 실제로 가져갈 것은 세 가지이고, 셋 다 성공률 숫자가 아니다.

1. **"they never abstain."** 이것이 우리 loop 에 가장 위험한 실패 양식이다. 우리 파이프라인은
   저신뢰 시 `engineer_review` 로 되돌아가는 것이 안전 계약인데, 자신 있게 틀린 좌표를 내는 matcher 는
   그 계약을 조용히 무력화한다. **어떤 probe 든 rank-1 과 함께 confident-wrong 비율을 재야 한다.**
2. **fine-tuning 은 forgetting 을 부른다.** 우리처럼 recipe 다양성이 큰 데이터에서 소수 recipe 로
   fine-tune 하면 나머지가 무너진다는 직접 증거다. zero-shot 으로 먼저 재는 것이 옳다.
3. **이 수치는 우리 문제의 예측치가 아니다.** 그 벤치는 *cross-modal*(SEM↔EBSD 처럼 물리가 다른 짝)
   이고 FOV 가 자릿수로 다르다. 우리는 **same-modality**(SEM 템플릿 ↔ SEM 라이브, OM↔OM)이고 배율도
   cond.txt 로 알고 있다 - 훨씬 쉬운 조건이다. SR@10 ≈ 0.10 을 우리 예상치로 옮겨 적으면 부정직하다.
   **하한이 아니라 "이 계열이 microscopy 에서 자동으로 잘 되지는 않는다" 는 경고로만 읽는다.**

그 밖에 microscopy 정합 문헌은 CLEM(광학↔전자) 쪽에 있으나
([DeepCLEM](https://pmc.ncbi.nlm.nih.gov/articles/PMC10311120/)) 세포 이미지 대상이라 반복 line
패턴 문제와 무관하다. **반복 line/aperture problem regime 에서 learned matcher 를 정량 평가한
1차 문헌은 찾지 못했다(primary source not found).**

---

## 6. 정직한 판정 - YOLO 는 rank-1 을 못 올린다 (1판, **§10 이 부분 철회**)

> **2판 주석.** 아래 표는 one-shot 전제에서 쓴 것이다. DFT 전제에서는 `YOLOE visual prompt` 행이
> 아니라 **DFT-pose 행**이 추가돼야 하고, 그 행의 rank-1 칸은 `0` 이 아니라 **조건부**다(§10).
> 나머지 행(YOLO detect per-recipe / learned matcher / 재등록)의 판정은 그대로 유효하다.


질문을 정확히 두 축으로 갈라 답한다.

| | proposal recall (현재 ≈0.70, **병목 아님**) | rank-1 (현재 ≈0.5, **진짜 병목**) |
|---|---|---|
| **YOLO detect / pose (per-recipe 학습)** | 학습 불가(class 당 3~10 장 vs 권장 1,500). 논외 | 논외 |
| **YOLO detect (단일 'align key' class)** | 소폭 개선 가능성 있음 - 반복 구조 후보를 잘 뽑을 것이다 | **0.** 한 frame 의 반복 line 을 전부 찾아 줄 뿐, 그중 어느 것인지에 대한 정보를 담지 않는다 |
| **YOLOE visual prompt** | 개선 가능성 있음(reference crop conditioning) | **≈0.** 점수가 "example 과 닮은 정도" 인데 정답과 decoy 는 정의상 동일하게 닮았다. 공식 문서도 "22-40 mAP band", "not to replace training" |
| **learned matcher + RANSAC** | 부수 효과로 오를 수 있음 | **여기가 유일하게 새로운 mechanism** - 여러 대응점의 기하 일관성은 국소 점수와 독립인 정보다 |
| **align key 재등록 (ADR 0006 결론)** | 무관 | **원리적으로 확실** - 유일하지 않은 것을 유일하게 만든다 |

**따라서: YOLO 는 틀린 도구다.** 최선의 경우에도 이미 0.70 인 recall 축만 건드리고, rank-1 은
구조적으로 못 건드린다. 게다가 4~8 GB VRAM 상주 프로세스를 요구해 16 GB 호스트 제약에 직접 충돌한다.

**맞는 도구는 두 개이고 서로 대체가 아니다.**

- **단기(측정 가능):** learned matcher 의 **대응 집합 + 기하 일관성**. 지금까지 시도된 세 fusion 이
  전부 "후보를 독립적으로 재점수" 였으므로 이건 네 번째 fusion 이 아니라 **처음 쓰는 정보**다.
  구현 후보는 **XFeat**(Apache-2.0, 6.2 MB, "running in real-time on an inexpensive laptop CPU without
  specialized hardware optimizations") - GPU 프로세스를 추가하지 않으므로 16 GB 제약을 안 건드린다.
  더 강한 조합이 필요하면 **ALIKED(BSD-3) 또는 DISK(Apache-2.0) + LightGlue(Apache-2.0)** 로 간다.
  **SuperPoint 는 라이선스 때문에 배포 경로에서 제외한다.**
- **근본(원리적으로 확실):** **재등록.** ADR 0006 이 이미 "SEM = ranking/distinctiveness problem
  unrankable by ANY member-fusion" 이라고 결론지었다. 그 문장이 참이면 어떤 matcher 로도 안 풀린다.
  matcher probe 는 그 문장이 *member-fusion 에 한정된 참인지* 아니면 *정보 부재의 참인지* 를 가르는
  실험이기도 하다.

---

## 7. 가장 싼 실험 - 하루, 라벨 0, 장비 0, GPU 프로세스 0 (1판; **§11 이 더 싸다**)

> **2판 주석.** 이 probe 는 여전히 유효하지만 **1순위가 아니다**. §11 의 oracle-ROI 실험이 학습도
> weight 도 새 의존성도 없이 더 결정적인 질문에 답하므로 그것을 먼저 돌린다. 이 XFeat probe 는
> §11 이 "DFT 로는 안 된다" 로 끝났을 때 가는 다음 갈래다.


기존 golden harness 를 그대로 쓴다. `golden_localization_eval_cond.py` 는 이미 (template, frame,
gt_xy) 쌍을 열거하고 `gt_in_topk` / `rank1` 을 집계하며 `lever_verdict` 로 proposer/reranker 축을
가른다. **새 지표를 만들 필요가 없다 - 같은 자에 재야 비교가 성립한다.**

**probe 설계 (standalone driver 하나, production 무수정):**

1. `pip install torch` 만으로 되는 **XFeat** weight 하나(6.2 MB)를 office PC 에 둔다. CPU 추론.
2. 각 golden 쌍에 대해: template(=rcp 또는 consensus align key) 과 live frame 에서 XFeat sparse
   feature → MNN match → `cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)`.
3. 얻은 변환으로 **template 의 align point 를 frame 으로 사영** → 좌표 하나. 기존과 같은 tolerance 로
   hit 판정.
4. 출력은 세 숫자 + 한 줄 `[DIGEST]`(office 규약):
   - **rank1** (기존 값과 직접 비교: SEM 0.665 / OM 0.852)
   - **abstain rate** = inlier 수가 임계 미만이라 좌표를 안 낸 비율
   - **confident-wrong rate** = inlier 충분한데 틀린 비율 ← **§5 가 경고한 바로 그 실패. 이게 크면
     이 계열은 우리 안전 계약과 양립 불가이고 숫자가 좋아도 채택 못 한다.**
5. modality 별로 나눠 본다(SEM/OM). 벽은 SEM 이다.

**판정 규칙 (실험 전에 못 박는다):**

| 결과 | 결론 |
|---|---|
| SEM rank-1 ≥ 0.73 **이고** confident-wrong 이 현 fallback 발동률 이하 | 기하 일관성이 진짜 레버. `ensemble_lab.parse_ensemble_channels` 에 채널로 승격해 정식 A/B(브리프 A 가 지적한 top-8 cap / Canny rescore seam 을 함께 손봐야 읽힌다) |
| rank-1 ≈ 0.5 근방 | 세 fusion 이 부딪힌 그 벽이 fusion 방식이 아니라 **정보 부재**였다는 뜻. matcher 계열 종료, **재등록이 유일한 레버**로 확정 |
| rank-1 은 올랐는데 confident-wrong 도 큼 | 채택 보류. abstention 설계(inlier 수·잔차 기반 gate)를 먼저 만들고 재측정 |

**비용:** 오피스 PC 에서 offline 실행, 이미지 반출 없음, 새 라벨 0, RCS/장비 접근 0, GPU 상주
프로세스 0(CPU). 비교 대상인 baseline 수치는 이미 있다.

**이 probe 전에 YOLO/YOLOE 를 시험하지 않는다** - §6 표대로 그것은 recall 축이고, recall 은 이미
병목이 아니기 때문이다. YOLOE 를 굳이 재고 싶다면 이 probe 다음에, 그것도 rank-1 이 아니라
`gt_in_topk` 를 과녁으로 삼아야 한다.

---

---

## 8. 정정 - align key 는 DFT(표준 fiducial) 위에 있다

사용자 정정: align key 는 recipe 마다 임의로 생긴 구조가 아니라 **DFT(Die Fit Target)** 라는
표준 정렬 target 안/주변에 있다. 비유는 "고양이는 다 다르게 생겼지만 공통 특징을 공유한다" 다.
**이 전제를 받으면 §1~§3 의 반대 논거 세 개가 동시에 무너진다.**

| 1판의 반대 | DFT 전제에서 | 남는 것 |
|---|---|---|
| "class 가 recipe 마다 다르다 → 298 class" | class 는 **1개**(DFT). recipe 는 class 가 아니라 **appearance 변이**다 | 변이 폭이 detector 가 감당할 수준인가 (미검증) |
| "class 당 3~10 장뿐" | recipe 를 가로질러 pool → **수천 장 규모** | 실제 장수 (§8.2) |
| "human bbox annotation 이 없다" | `cond.txt` 가 box + keypoint 를 **이미** 들고 있다 | 라벨 커버리지·품질 (§8.3) |

### 8.1 `cond.txt` 는 실제로 YOLO-pose 라벨의 모양이다 (repo 확인)

`poc/workflow_3/align/cond_file.py` docstring 원문:

> `!Cursor_info` : crosshair / white box 좌표가 한 줄에 들어 있다.
>   `elements[4],[5]` = crosshair (cx, cy) — 둘 다 -1 이 아니면 존재
>   `elements[6],[7],[8],[9]` = white box (left, top, right, bottom) — `[8],[9]` 가 -1 이 아니면 존재
>   cursor 좌표는 `Pixel` 의 **10배 oversample** 프레임이다(이미지 px = cursor/10).

즉 이미지 한 장당 **box(LTRB) 하나 + point 하나**가 이미 파싱된 채로 있다. 이것이 정확히
YOLO-pose 의 라벨 단위다(`<class-index> <x> <y> <w> <h> <px1> <py1> <p1-visibility>`,
[Pose Datasets](https://docs.ultralytics.com/datasets/pose/)). **사람 라벨링 0.**

주의점 셋 (전부 repo 에 이미 근거가 있다):

1. **`cond_for_image()` 를 반드시 거친다.** `/10` 을 손으로 계산하면 안 된다 - `cond.Pixel` 이 실제
   로드 해상도와 다르면 축별 스케일 보정이 필요하고, 안 하면 **모든 프레임에 동일하게 걸리는 계통
   오차**가 된다(`cond_file.py:112-114` 주석). 브리프 C §1 이 "학습 데이터 추출 시 반드시 이 함수를
   거쳐야 한다"고 이미 못 박았다.
2. **align point ≠ box 중심.** `cond_template.cond_align_offset` 이
   `(round(w/2 - box_cx), round(h/2 - box_cy))` 로 정의돼 있고, rcp 이미지의 align point 는
   **이미지 중심**이다(브리프 C §2). 반면 msr S 이미지의 GT 는 **crosshair 그 자체**다. 두 좌표계를
   섞으면 라벨이 조용히 오염된다. **pose keypoint 로 쓸 것은 msr S 의 crosshair 뿐이다.**
3. **crosshair 는 입력에서 지운다.** 안 지우면 모델이 "십자선 교차점을 찾아라" 를 배우는 치팅이 된다.
   production 이 이미 `clean_align_image.clean_image` 로 지운다(브리프 C §3.2). E 프레임에는 crosshair 가
   없으므로(실측 0/182, 브리프 C §3.3) **배포 시점 입력에는 십자선이 없다** - 학습 입력도 같아야 한다.

### 8.2 데이터 물량 - repo 실측으로 계산

golden set(298 recipe) 실측(브리프 C §4, `probe_recipe_s_counts.py` 결과):

| 항목 | 실측 |
|---|---:|
| recipe 수 | 298 |
| dominant-modality S ≥ 4장 recipe | **1개** |
| dominant-modality S = 정확히 3장 | 135개 |
| dominant-modality S = 0장 (fail-only) | **151개** |
| recipe 당 평균 S 장수 | ~2.6장 |

- **bbox 라벨 원천 = rcp 이미지**: recipe 당 OM/SEM 2장 → 최대 **~596장**.
- **keypoint 라벨 원천 = msr S 이미지**: 298 × ~2.6 ≈ **~775장**.
- 합계 **~1,370장** 규모. 정정 메시지의 "1,500~3,500" 은 golden set 기준으로는 **상단이 낙관적이나
  자릿수는 맞다**. MES 전체를 쓰면 훨씬 커지겠지만 **그 규모는 이 저장소 어디에도 실측이 없다**
  (브리프 C §4: "MES 원본 규모, recipe 개수, recipe당 평균 측정 빈도 어느 것도 코드나 문서로 확인되지
  않는다"). 학습 착수 전 `probe_recipe_s_counts.py` 동형 스크립트를 MES 경로에 돌리는 것이 선결이다.

**단, 위 두 원천은 서로 다른 이미지다.** rcp 에는 box 는 있어도 crosshair 가 없고(align point =
이미지 중심), msr S 에는 crosshair 가 있다. **하나의 이미지에 box + keypoint 가 동시에 있어야
YOLO-pose 라벨이 완성되므로, "msr S 이미지가 box_ltrb 를 갖는가" 가 이 설계의 급소다.**
간접 증거는 있다 - ADR 0005 의 box-crop A/B 가 consensus(=msr S) pool 에서
`history_crops_box` 를 실제로 빌드했고 `box_no_cand=0` 으로 "box 템플릿 항상 후보 생성" 을 보고했다.
**그러나 msr S 중 몇 %가 유효 box 를 갖는지 세어 둔 곳은 저장소에 없다(primary source not found).**
§11 의 probe 가 이 숫자를 같이 찍는다.

### 8.3 라벨 품질 게이트 - `check_cond_box` 가 이미 거른다

`cond_template.check_cond_box()` 가 4단계로 box 를 거른다(브리프 C §2.5):
`degenerate`(w/h ≤ 0) → `out_of_bounds` → `too_small`(대칭 inset 후 16px 미만) →
`offset_too_far`(대각선 정규화 offset > 0.38). 이 넷은 `skip` 이다.
**`skip` 을 학습 positive 로 그냥 쓰면 offset 오염이 들어간다**(브리프 C 의 경고 그대로).
학습 큐레이션은 이 함수를 필터로 재사용하면 되고 새로 만들 것이 없다.

또 하나 - **S 라벨 자체를 의심해야 한다.** 저장소 원칙이 "도구의 self-reported success 도 false
positive 가능; `tool_label` 은 metadata, CV 입력 금지" 이고, 최소 방어선은
`golden_localization_eval_cond.py:430` 의 `label != "S" or crosshair_xy is None` 게이트다.
학습 라벨도 같은 게이트를 통과시킨다.

### 8.4 아직 검증되지 않은 것 - 이것이 전부다

- **"DFT 가 하나의 시각적 class 로 성립하는가."** 저장소에 `DFT` / `Die Fit Target` 이라는 문자열은
  **한 번도 등장하지 않는다**(전체 `.py`/`.md` grep 0건). 즉 이 전제는 사용자의 도메인 지식이지
  코드/문서로 검증된 사실이 아니다. **정직하게 적어 둔다: 이 브리프는 그 전제를 참으로 가정하고
  전개하며, 참임을 확인한 바 없다.**
- **decoy 가 DFT box 안인가 밖인가** - §10 의 갈림길이자 §11 의 실험.

---

## 9. Ultralytics 라이선스 - fab 배포의 실제 게이트

**공식 [Licensing 페이지](https://www.ultralytics.com/license) FAQ 원문:**

> "If you use Ultralytics YOLO code, models, architectures, training pipelines, or trained/fine-tuned
> models, you must either: Open-source your entire project under AGPL-3.0, or Obtain an Ultralytics
> Enterprise License. **This applies even if you: Train your own model from scratch / Do not use
> pretrained weights / Use YOLO only internally or for R&D** / Deploy through a SaaS platform, API, or
> other private system / Embed it in hardware, edge devices, or commercial products."
>
> "Under Ultralytics AGPL-3.0 guidance, compliance means publicly releasing the complete corresponding
> source code for the entire derivative work, including the larger application, modifications, scripts,
> configuration files, and, where applicable, **model weights**. If you do not want to open-source the
> full project, you need an Enterprise License."

같은 페이지의 Enterprise 필요 항목 목록에 **"Internal business tools or private company applications"**,
**"Using custom-trained or fine-tuned YOLO models in a proprietary or commercial setting"**,
**"R&D projects that are not fully open-sourced"** 가 그대로 들어 있다.

**우리 상황에 그대로 적용하면:** workflow_3 은 사내 비공개 코드이고, 학습된 weight 는 fab 데이터에서
나온다. Ultralytics 가 공표한 해석대로면 **Enterprise License 필요**이며, 그렇지 않으려면
workflow_3 전체와 weight 를 AGPL-3.0 으로 공개해야 한다. **후자는 선택지가 아니다.**

**여기서 정직하게 갈라 둘 것 - 라이선스 *본문* 과 vendor 의 *해석* 은 같지 않다.**
[AGPL-3.0 원문](https://www.gnu.org/licenses/agpl-3.0.txt) §13 은 이렇게만 적는다:

> "Notwithstanding any other provision of this License, **if you modify the Program**, your modified
> version must prominently offer **all users interacting with it remotely through a computer network**
> (if your version supports such interaction) an opportunity to receive the Corresponding Source …"

즉 license text 상의 trigger 는 ① conveying(배포) ② **수정본**의 **원격 네트워크 상호작용**이다.
수정 없이, 배포 없이, 네트워크로 외부 사용자에게 노출하지 않고 사내에서만 돌리는 사용이 §13 을
발동시키는지는 **본문만으로는 자명하지 않다** - 이것은 널리 다투어지는 지점이고, 여기서 내가
"괜찮다" 고 답하면 그건 법률 의견이지 리서치가 아니다. 확실한 것은 두 가지뿐이다.

1. **저작권자인 Ultralytics 가 공개적으로 "internal only 도 Enterprise 필요" 라고 공표했다.**
   집행 주체의 공표된 입장이므로, 기술적 해석과 별개로 **분쟁 리스크는 실재한다.**
2. **이 판단은 개발자가 아니라 Legal 이 한다.** 이 브리프는 "게이트가 있다" 까지만 확정한다.

**대조군.** §4 의 learned matcher 계열은 **XFeat / LoFTR / EfficientLoFTR / DISK / LightGlue =
Apache-2.0**, **ALIKED = BSD-3-Clause**, **RoMa = MIT** 로 전부 permissive 다(SuperPoint 와 MASt3R 만
제외 - 각각 비상업 조항). **즉 라이선스 축에서는 대안 계열이 명백히 유리하며, 이건 성능과 무관한
독립 축이다.**

---

## 10. DFT-pose 설계 검토 - 어떤 head 를, 어떤 정밀도로

### 10.1 pose head 로 "box + 점 하나" 가 표현 가능한가 - 가능하다

공식 [Pose Datasets](https://docs.ultralytics.com/datasets/pose/) 의 라벨 스펙:

> Format with keypoint visibility (includes visibility per point):
> `<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> …`
> "`<class-index>` is the index of the class for the object, `<x> <y> <width> <height>` are the
> normalized coordinates of the **bounding box**"

그리고 dataset YAML:

> `kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)`

**`kpt_shape` 의 첫 원소가 keypoint 개수이므로 `[1, 2]` 또는 `[1, 3]` 은 스펙상 표현 가능하다.**
1판에서 "1-keypoint 사용 사례 문서 없음" 이라고 적은 것은 유지한다(배포 dataset 은 COCO-Pose 17점 /
Dog-Pose 24점 / Hand 21점 / Tiger-Pose 12점뿐이고, **1점 예제는 공식 문서에 없다 -
primary source not found**). 그러나 **"문서화된 예제가 없다" 와 "표현할 수 없다" 는 다르고, 스펙은
표현 가능 쪽이다.** 1판의 서술을 그만큼 완화한다.

**visibility flag.** `p1-visibility` 는 선택 차원이며 COCO 규약에서 0/1/2(미라벨/가려짐/보임)이다.
우리에게 실용적 쓸모가 있다 - **E 프레임처럼 crosshair 가 없어 GT 점을 모르는 이미지**를
`visibility=0` 으로 넣으면 **box 는 학습에 쓰고 keypoint loss 는 끄는** 것이 가능하다. 즉
"151개 fail-only recipe" 와 E 프레임이 detector 학습에는 살아난다. (COCO 평가에서 `v=0` keypoint 가
OKS 계산에서 제외되는 것은 `cocoeval.py` 의 `e = e[vg > 0]` 로 확인된다.)

**bbox 는 필수다.** keypoint 만 있는 라벨 포맷은 없다. 우리는 box 가 있으므로 문제되지 않지만,
**box 가 없는 이미지(rcp `skip` 케이스, box 없는 msr)는 pose 학습에 못 쓴다.**

### 10.2 OBB 가 나은가 - fiducial 이 실제로 회전한다면 그렇다, 그러나 라벨이 없다

[OBB 문서](https://docs.ultralytics.com/tasks/obb/): "Oriented object detection goes a step further …
by introducing an extra angle to locate objects more accurately", 라벨은 네 꼭짓점
`class_index x1 y1 x2 y2 x3 y3 x4 y4` 이며 내부적으로 `xywhr`(rotation in radians,
`[-π/4, 3π/4)`) 로 처리된다. 권장 상황은 "objects appear at various angles … where traditional
axis-aligned bounding boxes may include unnecessary background".

**우리에게의 판정: 지금은 OBB 를 쓸 수 없다.** `cond.txt` 의 `!Cursor_info` box 는 **LTRB 4숫자,
축정렬**이다(`cond_file.py:_BOX_IDX = (6,7,8,9)`). 회전각 라벨이 **원천에 없다.** OBB 로 가려면
사람이 각을 새로 라벨하거나 각을 추정해야 하는데, 그건 "사람 라벨 0" 이라는 이 설계의 최대 장점을
버리는 것이다. 웨이퍼 stage 가 회전을 잘 통제한다면 축정렬 box 로 충분하고, 회전이 실제로 크다면
그건 **detector head 문제가 아니라 라벨 원천 문제**다. 오피스에서 "S 프레임들 사이에서 DFT 가 눈에
띄게 기울어 보이는가" 를 눈으로 한 번 보는 것이 OBB 검토의 선결이다.

### 10.3 정밀도 - mAP/OKS 는 픽셀 오차를 말해주지 않는다

**이게 이 절에서 가장 중요한 문장이다.** 출력이 클릭 좌표 하나이므로 필요한 것은 픽셀 오차인데,
공식 지표는 픽셀이 아니다.

COCO keypoint 평가의 실제 구현(`cocoapi/PythonAPI/pycocotools/cocoeval.py`, `computeOks`):

```python
vars = (sigmas * 2)**2
...
e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
```

**`gt['area']` 로 나눈다.** 즉 OKS 는 **객체 크기로 정규화된 상대 지표**이고, 같은 픽셀 오차라도
객체가 크면 OKS 가 높게 나온다. 따라서 **어떤 pose mAP 수치도 그 자체로는 "몇 px 안에 들어온다" 를
의미하지 않는다.** 우리 SEM 은 key 가 프레임의 80~100%를 채우므로(=`area` 가 매우 큼) OKS 는
**관대해지는 쪽**이다 - 즉 높은 pose mAP 가 우리 tolerance 를 만족한다는 보장이 되지 못한다.

- **Ultralytics 공식 문서에 pose 의 픽셀 정밀도 수치는 없다.** `yolo-performance-metrics` 가이드는
  pose 에 대해 "pose writes both `Box*` and `Pose*` curves" 라고만 적고 OKS 설명도, 픽셀 오차도 없다
  ([performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)).
  **primary source not found.**
- **box center 로 점을 만드는 것은 더 나쁘다.** detection 지표는 IoU 기준이고, IoU 0.5~0.95 를
  만족하는 박스도 중심은 객체 크기에 비례해 흔들린다. 큰 객체(우리 SEM)에서 IoU 는 중심 오차에 특히
  둔감하다. **box center 를 align point 로 쓰는 설계는 배제한다** - 반드시 keypoint head 를 쓰거나,
  box 로 ROI 만 자르고 좌표는 기존 CV matcher 가 낸다(후자가 §11 의 구조다).

**따라서 정밀도는 문헌에서 못 가져온다. 우리 golden set 에서 우리 tolerance 로 직접 재야 한다.**
이것 또한 §11 이 답한다.

### 10.4 1,500장 / 10,000 instance 권장치를 DFT 프레이밍에 정직하게 적용하면

권장치 원문과 **그 바로 옆에 붙은 이유**를 같이 읽어야 한다
([tips for best training results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)):

> "**Images per class.** ≥ 1500 images per class recommended
> **Instances per class.** ≥ 10000 instances (labeled objects) per class recommended
> **Image variety.** Must be representative of deployed environment. For real-world use cases we
> recommend images from **different times of day, different seasons, different weather, different
> lighting, different angles, different sources** (scraped online, collected locally, different
> cameras) etc."

**공정한 독해:**

- 권장치를 정당화하는 항목이 전부 **자연영상의 nuisance 변이**다 - 시간대/계절/날씨/조명/카메라.
  우리는 **한 계열의 장비, 통제된 배율과 조명, 회색조 단일 소스**다. 그 변이 예산이 통째로 없으므로
  **10,000 instance 라는 숫자를 그대로 적용하는 것은 과하다.** 같은 페이지가 "A few hundred annotated
  objects per class is enough to start experimenting with transfer learning"
  ([data collection guide](https://docs.ultralytics.com/guides/data-collection-and-annotation/))
  라고 적은 것도 이 방향을 지지한다. **~1,370장 / instance 1개씩은 "시작하기엔 충분, 신뢰 배포엔
  미달" 구간이며, 우리 조건에서는 그 격차가 COCO 기준보다 작다.**
- **그러나 우리에겐 저 목록에 없는 변이가 있다 - DFT 내부의 패턴 내용이 recipe 마다 다르다.**
  이건 조명 변이가 아니라 **class-내 구조 변이**이고, 오히려 더 어려운 종류다. 그러니 "우리 조건이
  통제돼 있으니 적어도 된다" 를 무한정 밀 수는 없다. **정직한 답: 권장치는 우리에게 그대로 구속력이
  없지만, 그것을 감량해 주는 근거(통제된 촬영)와 늘려야 하는 근거(recipe 간 구조 변이)가 동시에
  있어서 문헌으로는 결론이 안 난다. 학습곡선(recipe 수를 늘려가며 held-out recipe 성능)을 그리는 것이
  유일한 답이다.**
- **한 가지는 문헌으로 확정된다 - 부분 라벨 금지.**
  > "**Label consistency.** All instances of all classes in all images must be labeled.
  > **Partial labeling will not work.**"

  **이건 우리 라벨 원천의 직접적 위험이다.** `cond.txt` 는 이미지당 box 를 **하나** 준다. 한 프레임
  안에 DFT 가 여러 개 보이면(반복 die 배열이면 그럴 수 있다) 나머지는 라벨 없는 positive 가 되어
  모델에 "저건 DFT 가 아니다" 를 가르친다. 공식 문서가 명시적으로 "will not work" 라고 적은 조건이다.
  **학습 착수 전에 "한 프레임에 DFT 가 몇 개 보이는가" 를 눈으로 확인해야 한다.** 여러 개면
  라벨링 전략을 바꾸거나(추가 라벨 필요 → 사람 라벨 0 이라는 장점 소멸) crop 단위를 DFT 하나로 좁혀야 한다.

---

## 11. 프리어 아트, 그리고 진짜 첫 실험

### 11.1 반도체 정렬 마크에 DL 을 쓴 선행 연구 - 있다, 그리고 우리 설계와 다른 곳을 짚는다

1판은 "반도체 + DL = 결함 검출뿐" 이라고 적었는데, **정렬 마크 쪽으로 좁히면 문헌이 실제로 있다.**
1판의 그 서술은 검색 범위가 좁았던 것이고, 여기서 정정한다.

| 연구 | task | 데이터 | 측정된 정밀도 | 방식 |
|---|---|---|---|---|
| [Subpixel keypoint localization and angle prediction for lithography marks based on deep learning](https://link.springer.com/article/10.1007/s10845-024-02400-8) (J. Intell. Manuf.) | lithography **정렬 마크의 subpixel keypoint 좌표 + 회전각** 동시 예측 | 논문 본문 비공개(paywall). 코드/데이터 공개: [HRNet8-SVD](https://github.com/YuLungLee/HRNet8-SVD) (`DataSet.7z` 포함, **LICENSE 파일 없음**) | 각도 예측 **RMSE 0.093 / R² 0.976**. 좌표의 px 오차는 초록 수준에서 확인 불가 | **CNN backbone(HRNet) + keypoint head**, 그리고 **전체 이미지가 아니라 image patch 입력** ("takes image patches rather than the entire original image as input to focus only on the unobstructed keypoint regions") |
| [Improving Laser Direct Writing Overlay Precision Based on a Deep Learning Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC12388746/) | **결함 있는 십자(crosshair) fiducial** 의 중심 좌표 오차 예측/보정 | **합성 데이터 66,000 세트**(train 60,000 / val 3,000 / test 3,000) | "prediction errors **below 100 nm** in both X/Y", FNN 대비 90% 오차 감소 | conv 2층 + pooling 1층의 **작은 CNN**, 회귀 |

**여기서 우리가 가져갈 것 세 가지.**

1. **정렬 마크 keypoint 회귀는 이 산업에서 실제로 하는 일이다.** DFT 프레이밍은 문헌적으로 이상하지
   않다. 1판의 회의는 이 지점에서 과했다.
2. **두 연구 모두 "전체 프레임에서 마크를 찾는" 문제를 풀지 않는다.** 첫 연구는 **patch 입력**이고,
   둘째는 **이미 잘린 십자 마크**에 대한 좌표 보정이다. 즉 둘 다 **"어디를 볼지는 이미 정해졌고,
   그 안에서 정밀 좌표를 낸다"** 는 구조다 - 이것이 §11.3 에서 제안하는 우리 배치(**detector 가 ROI,
   좌표는 그 안에서**)와 정확히 같은 모양이고, "detector 가 최종 좌표를 낸다" 와는 다르다.
3. **둘째 연구는 66,000장을 합성으로 만들었다.** fiducial 이 **설계도가 있는 인공 구조**라는 사실이
   데이터 문제를 우회하는 정식 경로임을 보여준다. §10.4 의 물량 걱정에 대한 실질적 답이 될 수 있다
   (DFT 도형을 파라메트릭하게 그려 pretrain → 실데이터로 fine-tune).

**여전히 못 찾은 것(정직하게):** CD-SEM 의 recipe align key / DFT 를 detector 로 잡은 공개 문헌은
**찾지 못했다(primary source not found)**. 위 둘은 lithography/laser-writing 쪽 광학 마크다.

### 11.2 DFT detector 가 rank-1 을 올리는 유일한 경로 - 그리고 그것이 참인지 모른다

DFT box 를 완벽히 잡았다고 하자. 그것이 **rank-1** 에 기여하는 경로는 하나뿐이다:
**탐색공간에서 decoy 를 제거한다.**

- decoy 반복 line 이 **box 밖**에 있다 → box 로 자르면 decoy 가 사라진다 → **matcher 를 하나도 안 바꾸고
  rank-1 이 오른다.** 1판 §6 표에서 이 칸을 `0` 으로 적은 것은 **틀렸다.** 철회한다.
- decoy 가 **box 안**에 있다(= aperture problem 이 DFT 내부의 반복 구조에서 온다) → box 로 잘라도
  같은 decoy 가 그대로 남는다 → **완벽한 detector 도 rank-1 에 0 을 더한다.**

**어느 쪽인지 저장소에 근거가 없다.** 그리고 이게 프로그램 전체의 분기점이다.

**ADR 0005 가 이미 답했다고 읽으면 안 된다 - 방향이 반대다.**
ADR 0005 는 **template** 을 box 로 줄였다(작은 템플릿 → 프레임 안 wrong-phase 위치가 늘어 recall 하락,
OM -0.042 / SEM -0.110). 여기서 제안하는 것은 **search frame** 을 box 로 줄이는 것이다 - 후보 위치가
줄기만 하므로 원리상 나빠질 수 없다. **둘은 같은 박스를 쓸 뿐 서로 다른 실험이다.** 오히려 ADR 0005 의
결론("더 많은 주변 context 가 도움")은 template 은 크게 유지하고 frame 만 자르는 이 배치와 모순되지 않는다.

### 11.3 첫 실험 - oracle-ROI. 학습 0, weight 0, 새 의존성 0

**질문:** "DFT 를 완벽히 검출했다면 rank-1 이 오르는가?"
**방법:** 완벽한 검출기를 학습하는 대신 **`cond.box_ltrb` 를 정답 박스로 쓴다.**

1. golden set 의 각 (template, msr S frame, gt crosshair) 에 대해, **frame 쪽** cond 의
   `box_ltrb` 를 `cond_for_image()` 로 이미지 px 로 변환한다(직접 `/10` 금지).
2. 기존 matcher(`compute_align_key_score_ensemble`)의 **후보 탐색 범위를 그 박스로 제한**한다.
   템플릿은 지금 것 그대로 - 아무것도 안 바꾼다.
3. `golden_localization_eval_cond.py` 의 기존 집계(`gt_in_topk` / `rank1` / `lever_verdict`)를 그대로
   써서 **baseline vs oracle-ROI** 를 **고정 분모**로 A/B 한다(ADR 0005 가 확립한 규율:
   `n_eval` 동일, no-candidate = miss).
4. 같은 실행에서 **전제 검증 세 숫자**를 함께 찍는다 - 이게 절반의 가치다:
   - **msr S 중 유효 `box_ltrb` 보유 비율** (§8.2 의 미확인 급소)
   - **crosshair 가 그 box 안에 들어오는 비율** ("align point 는 DFT 에 대해 정해진다" 는 전제의 직접 검사.
     낮으면 DFT 프레이밍 자체가 틀린 것이다)
   - **box 면적 / 프레임 면적** 비율 (탐색공간이 실제로 얼마나 줄어드는가. 0.8 이면 자를 게 없다)
5. modality 별로 나눈다. 벽은 SEM 이다.

**판정 규칙 (실험 전에 못 박는다):**

| oracle-ROI 결과 | 결론 |
|---|---|
| SEM rank-1 이 유의하게 상승 | **DFT detector 는 진짜 레버.** 그때 비로소 YOLO-pose 학습(또는 더 가벼운 detector)을 설계한다. 단 §9 라이선스 게이트를 Legal 로 먼저 보낸다 |
| rank-1 거의 불변 | **decoy 가 box 안에 있다.** 완벽한 detector 도 0 이므로 **DFT 축 종료.** 남는 레버는 §4 의 기하 일관성(§7 XFeat probe) 과 재등록 |
| crosshair-in-box 비율이 낮음 | DFT 전제 자체가 성립하지 않는다. 사용자와 도메인 확인부터 |
| msr S 의 box 보유율이 낮음 | pose 라벨 원천이 rcp 뿐 → "사람 라벨 0" 장점 소멸. 라벨 전략 재설계 |

**비용:** 오피스 offline, 이미지 반출 0, 새 라벨 0, 장비 0, GPU 0, 새 의존성 0. 기존 드라이버에
arm 하나. **§7 의 XFeat probe 보다 싸고, 더 상위 질문에 답한다.** 순서는 **§11 → (필요시) §7** 이다.

**그리고 이 실험은 detector 를 훈련하는 것보다 항상 먼저 와야 한다** - detector 는 oracle 보다 잘할 수
없으므로, oracle 이 0 이면 detector 도 0 이다. 이 논증에 예외가 없다는 점이 이 실험을 결정적으로 만든다.

---

## 부록 - 인용한 primary source

**Ultralytics 공식 문서**
- Licensing (AGPL-3.0 vs Enterprise, internal-use 문구): https://www.ultralytics.com/license
- OBB task: https://docs.ultralytics.com/tasks/obb/
- Performance metrics (pose 는 `Pose*` curve 만 언급): https://docs.ultralytics.com/guides/yolo-performance-metrics/
- Tasks: https://docs.ultralytics.com/tasks/
- YOLOE (visual prompt / SAVPE / LVIS 표 / Limitations / Deployment): https://docs.ultralytics.com/models/yoloe/
- YOLO-World: https://docs.ultralytics.com/models/yolo-world/
- SAM 3: https://docs.ultralytics.com/models/sam-3/
- Pose task: https://docs.ultralytics.com/tasks/pose/
- Pose datasets (label format, `kpt_shape`): https://docs.ultralytics.com/datasets/pose/
- Data Collection and Annotation FAQ (1,500 / 10,000): https://docs.ultralytics.com/guides/data-collection-and-annotation/
- Tips for Best Training Results: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/

**라이선스 원문 / 평가 구현**
- GNU AGPL-3.0 전문 (§13 Remote Network Interaction): https://www.gnu.org/licenses/agpl-3.0.txt
- COCO keypoint 평가 구현(OKS 가 `gt['area']` 로 정규화됨): https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

**논문**
- YOLOE: https://arxiv.org/abs/2503.07465
- LoFTR: https://arxiv.org/abs/2104.00680 · project page https://zju3dv.github.io/loftr/
- Efficient LoFTR: https://arxiv.org/abs/2403.04765
- LightGlue: https://arxiv.org/abs/2306.13643
- XFeat: https://arxiv.org/abs/2404.19174
- RoMa: https://arxiv.org/abs/2305.15404
- MatchAnything: https://arxiv.org/abs/2501.07556
- Deep Learning Reforms Image Matching (survey): https://arxiv.org/html/2506.04619v1
- Optimizing YOLOv7 for Semiconductor Defect Detection: https://arxiv.org/abs/2302.09565
- Correlative microscopy foundation-model registration (preprint, SEM 포함): https://www.researchsquare.com/article/rs-10311464/v1
- Subpixel keypoint localization and angle prediction for lithography marks (J. Intell. Manuf., **paywall**): https://link.springer.com/article/10.1007/s10845-024-02400-8 · 코드 https://github.com/YuLungLee/HRNet8-SVD (LICENSE 없음)
- Improving Laser Direct Writing Overlay Precision Based on a Deep Learning Method: https://pmc.ncbi.nlm.nih.gov/articles/PMC12388746/

**repo / license (원문 확인)**
- LightGlue: https://github.com/cvg/LightGlue (Apache-2.0; SuperPoint 예외 명시)
- SuperPoint LICENSE: https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE
- DISK: https://github.com/cvlab-epfl/disk (Apache-2.0)
- ALIKED: https://github.com/Shiaoming/ALIKED (BSD-3-Clause)
- XFeat: https://github.com/verlab/accelerated_features (Apache-2.0; `weights/xfeat.pt` 6,247,949 B)
- LoFTR: https://github.com/zju3dv/LoFTR (Apache-2.0)
- EfficientLoFTR: https://github.com/zju3dv/EfficientLoFTR (Apache-2.0)
- RoMa: https://github.com/Parskatt/RoMa (MIT; DINOv2 Apache-2)
- MASt3R: https://github.com/naver/mast3r (CC BY-NC-SA 4.0, 비상업)
- MatchAnything: https://github.com/zju3dv/MatchAnything (LICENSE 파일 없음)

**저장소 내부 근거 (2판에서 확인)**
- `poc/workflow_3/align/cond_file.py` - `!Cursor_info` box/crosshair 파싱, `cond_for_image()` 스케일 보정
- `poc/workflow_3/align/cond_template.py` - `cond_align_offset`, `check_cond_box` 4단계 게이트
- `poc/workflow_2/docs/study/research/2026-09-02-C-training-data-audit.md` §2~§4 - 라벨 공식, S 분포 실측
- `poc/workflow_2/docs/study/adr/0005-whitebox-box-crop-consensus-arm-rejected.md` - box **template** crop A/B (frame ROI 와 별개)
- repo 전체 grep: `DFT` / `Die Fit` **0건** - DFT 전제는 코드/문서에 근거가 없다

**primary source 를 찾지 못한 항목 (추측으로 메우지 않음)**
- Ultralytics 의 pose task 전용 최소 이미지 수 기준
- 1-keypoint pose 모델을 공식적으로 다룬 문서/예제 (**단 `kpt_shape` 스펙상 표현은 가능**, §10.1)
- Ultralytics pose 의 **픽셀 단위** 정밀도 수치 (공식 문서는 OKS 설명조차 없음)
- CD-SEM align key / DFT 를 detector 로 검출한 공개 문헌
- msr S 이미지 중 유효 `box_ltrb` 보유 비율 (저장소에 집계 없음 - §11.3 이 찍는다)
- XFeat / LightGlue 의 공식 파라미터 수(README·논문 모두 미기재. weight 파일 크기만 확인)
- EfficientLoFTR / RoMa 의 공식 VRAM 요구치
- learned matcher 를 CD-SEM align key / recipe alignment 에 적용한 문헌
- 반복 line 구조(aperture problem regime)에서 learned matcher 를 정량 평가한 벤치마크
