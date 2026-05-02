# 이미지 처리 배경

이 문서는 `poc/workflow_2`에 쓰인 image processing 개념을 기초부터 설명한다.
OpenCV, matching, score, template처럼 코드에서 그대로 쓰는 기술 용어는 영어를
유지한다.

## 이미지가 코드에서 의미하는 것

OpenCV/Python에서 이미지는 NumPy array다.

- grayscale image는 2D matrix다: `height x width`.
- 각 pixel은 보통 `0`부터 `255`까지의 밝기값을 가진다.
- `0`은 검정, `255`는 흰색, 중간값은 gray다.
- color image는 보통 `height x width x channels` 형태다.
- `workflow_2`는 color input도 grayscale로 바꾼다. SEM monitor image와
  align-key matching에서 중요한 신호는 color가 아니라 intensity와 shape이기
  때문이다.

`align_key_matcher.py`의 `_to_grayscale()`이 이 normalization을 담당한다.

## SEM 이미지를 직접 matching하기 어려운 이유

같은 physical align key라도 매번 같은 pixel value로 찍히지 않는다. SEM image는
일반 camera image가 아니라 electron-sample interaction 결과다. 일반적인 SEM
관찰에서는 secondary electron이 surface/topography 정보를 많이 담는다. 그래서
edge와 surface shape는 중요하지만, acquisition condition이 조금만 달라져도
image appearance가 바뀔 수 있다.

`workflow_2`에서 중요한 변화 요인:

- detector gain, beam setting, dwell time 차이로 brightness/contrast가 바뀐다.
- low beam current나 fast imaging mode에서는 noise가 커질 수 있다.
- charging은 low contrast, local bright/dark patch, scan-direction tail,
  distortion, drift를 만들 수 있다.
- focus, scan rotation, stage motion 때문에 mark가 blur, rotation, scale
  change를 보일 수 있다.
- recipe template resolution과 live SEM monitor resolution이 다를 수 있다.

따라서 질문을 "두 이미지의 pixel이 같은가?"로 잡으면 실패하기 쉽다. 더 올바른
질문은 "두 이미지에 같은 geometric edge structure가 있는가?"다.

## 현재 전처리 파이프라인

```text
raw image
  -> grayscale
  -> CLAHE local contrast normalization
  -> Gaussian blur
  -> Canny edge detection
  -> distance transform
```

`preprocess_for_matching()`은 raw image를 바로 비교하지 않고, matching에 더
안정적인 representation으로 바꾼다. 이 단계는 SEM image의 brightness drift와
noise에 덜 민감하게 만들기 위한 것이다.

### 그레이스케일

Color channel 복잡도를 제거한다. SEM align-key matching에서는 geometry와
intensity가 주된 신호이므로 one-channel grayscale이면 충분하다.

### 국소 대비 보정

CLAHE는 Contrast Limited Adaptive Histogram Equalization이다. 전체 image를
한 번에 밝기 보정하지 않고 작은 tile 단위로 local contrast를 개선한다.
"Contrast Limited"가 중요한 이유는 adaptive equalization이 noise까지 과하게
키울 수 있기 때문이다.

현재 코드:

```python
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
```

의미:

- `8 x 8` tile grid 기준으로 local contrast를 보정한다.
- `clipLimit=2.0`으로 noise amplification을 보수적으로 제한한다.

### 노이즈 완화

Gaussian blur는 작은 random noise를 smoothing한다. Edge detector는 noise에
민감해서, noise가 작은 false edge처럼 보일 수 있다.

현재 코드:

```python
GAUSSIAN_SIGMA = 1.0
```

의미:

- 작은 noise는 줄인다.
- align mark의 큰 edge structure는 지우지 않는다.

### Canny 엣지 검출

Canny는 image를 binary edge map으로 바꾼다.

- edge pixel: mark boundary 또는 강한 intensity transition
- non-edge pixel: background 또는 mark 내부

현재 threshold:

```python
CANNY_T_LOW = 60
CANNY_T_HIGH = 160
```

핵심은 align key를 exact brightness texture가 아니라 edge로 된 shape로 본다는
점이다.

### 거리 변환

Distance transform은 binary edge image를 "nearest edge까지의 거리" map으로
바꾼다.

```text
edge pixel                  -> distance 0
1 pixel away from an edge   -> distance 1
10 pixels away from an edge -> distance 10
```

`workflow_2`는 live SEM frame의 edge map에 `cv2.distanceTransform()`을 적용한다.
그 다음 recipe-template edge mask를 이 distance map 위에서 sliding한다. Template
edge가 frame edge 가까이에 놓이면 average distance가 낮고, match가 좋다.

## Chamfer 매칭

Chamfer matching은 `workflow_2`의 main structural matcher다. Exact pixel value를
비교하지 않고 edge geometry가 얼마나 잘 겹치는지 본다.

동작:

1. Recipe template을 binary edge mask로 바꾼다.
2. SEM frame을 edge map과 distance transform으로 바꾼다.
3. Template edge mask를 frame distance transform 위에서 sliding한다.
4. 각 위치에서 template edge pixel 아래의 frame distance를 평균낸다.
5. Average distance가 낮을수록 edge geometry가 잘 맞는다.
6. Average distance를 0-1 score로 바꾼다.

```text
chamfer_score = exp(-mean_distance_px / DT_TAU_PX)
```

현재 `DT_TAU_PX = 10.0`이다. Template edge가 frame edge에서 평균적으로 멀어질수록
score가 exponential하게 감소한다.

장점:

- raw brightness 차이에 덜 민감하다.
- 작은 edge gap이나 noise에 비교적 강하다.
- "template edge가 frame edge에서 평균 N pixel 떨어져 있다"는 식으로 해석할 수
  있다.

약점:

- 비슷한 edge가 많은 frame에서는 false candidate가 생길 수 있다.
- 단독으로는 full mark geometry가 맞는지 충분히 증명하지 못한다.

## 여러 scale 탐색

Recipe template과 SEM frame의 physical resolution이 다르면 같은 mark라도 pixel
size가 달라진다. 예를 들어 template에서 128 px인 mark가 live frame에서는 110 px
또는 150 px로 보일 수 있다.

`nm_per_pixel` metadata가 있으면 physical resolution으로 scale을 계산하는 것이
가장 좋다. Metadata가 없으면 현재 코드는 다음 scale들을 시도한다.

```python
DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)
```

의미:

- template을 70%, 85%, 100%, 120%, 140% 크기로 바꿔가며 찾는다.

## ORB 특징점 매칭

ORB는 두 번째 matcher다. Chamfer가 "전체 edge 배치"를 보는 반면, ORB는 corner,
junction, dot, asymmetric mark 같은 local feature를 본다.

`compute_orb_inlier_ratio()` flow:

```text
template crop
  -> detect keypoints
  -> compute binary descriptors

frame crop near Chamfer candidate
  -> detect keypoints
  -> compute binary descriptors

match descriptors with Hamming distance
  -> keep unambiguous matches with Lowe ratio test
  -> fit homography with RANSAC
  -> return inlier ratio
```

용어:

- Keypoint: corner처럼 반복 검출될 가능성이 높은 image location.
- Descriptor: keypoint 주변 모양을 요약한 numeric signature.
- Hamming distance: binary descriptor끼리의 거리.
- Lowe ratio test: best match가 second-best match보다 충분히 좋을 때만 유지하는
  ambiguity filter.
- RANSAC: wrong match가 섞여 있어도 geometry model을 robust하게 맞추는 방법.
- Inlier: 최종 geometry와 일치하는 match.
- Outlier: 잘못되었거나 geometry와 맞지 않는 match.

ORB가 유용한 이유:

- Chamfer candidate 주변에 실제 matching local detail이 있는지 확인한다.
- random edge-like false positive를 줄인다.
- 제한적 rotation/scale 변화에 도움을 준다.

ORB의 약점:

- 대칭적인 nested box pattern은 distinctive keypoint가 부족할 수 있다.
- 매우 smooth한 SEM image에서는 stable feature가 적을 수 있다.
- 어떤 low-texture industrial image에서는 AKAZE나 SIFT가 더 나을 수 있다.

## 최종 점수와 decision

현재 score는 Chamfer score와 ORB inlier ratio를 weighted average로 합친다.

```text
score = 0.6 * chamfer_score + 0.4 * orb_inlier_ratio
```

Decision 기준:

| 점수 범위 | Decision | 의미 |
| --- | --- | --- |
| `score >= 0.75` | `match` | 자동 진행할 만큼 충분히 강한 match. |
| `0.55 <= score < 0.75` | `adjust` | 후보는 있지만 애매함. 이동/보조 판단 필요. |
| `score < 0.55` | `low` | 현재 FOV에서 신뢰할 match 없음. |

이 threshold는 engineering policy다. 실제 SEM data로 calibration하기 전에는
production truth로 보면 안 된다.

## VLM의 적절한 역할

VLM은 이 matching step의 primary tool로 적합하지 않다.

이유:

- 이 task는 calibrated, repeatable pixel/geometry evidence가 필요하다.
- 현재 project VLM들은 SEM fiducial registration보다 GUI grounding/OCR에 더
  맞다.
- Sparse SEM frame에서 VLM coordinate를 final stage movement decision으로 쓰기엔
  신뢰도가 부족하다.

좋은 VLM 역할:

- ambiguous debug frame 설명 또는 분류
- SEM view가 비어 있거나 structure가 한쪽에 몰렸을 때 rough direction 제안
- engineer review helper
- RCS GUI control처럼 SEM image matching 바깥의 작업 지원

## 빠른 용어 정리

| 용어 | 설명 |
| --- | --- |
| FOV | 현재 검색 중인 SEM monitor 영역. Field of View. |
| Template | 찾고 싶은 recipe align-key image. |
| Pixel | Image matrix의 한 칸. |
| Contrast | 어두운 영역과 밝은 영역의 차이. |
| Edge | 밝기가 급격히 바뀌는 boundary. |
| Keypoint | corner처럼 반복 검출 가능한 local point. |
| Descriptor | keypoint 주변을 요약한 numeric signature. |
| Inlier | geometric transform과 일치하는 match. |
| Outlier | 잘못되었거나 geometry와 맞지 않는 match. |
| Distance transform | 각 pixel에 nearest edge까지의 거리를 저장한 image. |
| Chamfer score | distance transform에서 계산한 edge-structure alignment score. |
| Homography | 한 image plane의 point를 다른 plane으로 mapping하는 2D transform. |
| RANSAC | wrong match가 많아도 robust하게 model을 fitting하는 방법. |

## 참고 자료

- OpenCV histogram equalization and CLAHE:
  <https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html>
- OpenCV Canny edge detector:
  <https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html>
- OpenCV distance transform:
  <https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html>
- OpenCV template matching:
  <https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html>
- OpenCV ORB class reference:
  <https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html>
- OpenCV feature matching + homography:
  <https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html>
- JEOL SEM secondary-electron image glossary:
  <https://www.jeol.com/words/semterms/20121024.070958.php>
- JEOL SEM charging glossary:
  <https://www.jeol.com/words/semterms/20121024.054158.php>
- NIST SEM noise note:
  <https://www.nist.gov/publications/nanomanufacturing-concerns-about-measurements-made-sem-part-v-dealing-noise>

