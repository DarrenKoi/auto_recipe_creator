# `poc/workflow_2` Research Guide / 연구 가이드

이 디렉터리는 `poc/workflow_2/`에 쓰인 과학/공학 배경을 설명한다. 이미지
처리를 잘 모르는 독자도 읽을 수 있도록 한국어 설명을 먼저 두고, 같은 의미의
English summary를 함께 적었다.

This directory explains the science and engineering behind `poc/workflow_2/`.
Each section gives Korean first, then a matching English summary.

## What `workflow_2` Solves / 해결하려는 문제

```text
Recipe align-key image
        +
Current SEM monitor frame
        |
        v
Find whether the same physical fiducial pattern is visible,
where it is in the frame, and how the stage/search loop should continue.
```

한국어:

- 레시피에 저장된 align-key 이미지를 기준으로 현재 SEM monitor frame 안에서
  같은 물리적 fiducial pattern이 보이는지 판단한다.
- 보인다면 frame 안의 좌표를 반환한다.
- 보이지 않거나 애매하면 stage/search loop가 다음에 어디로 움직일지
  결정한다.

English:

- Use the recipe align-key image as the reference.
- Decide whether the same physical fiducial pattern is visible in the current
  SEM monitor frame.
- Return the frame coordinate if found, or continue the stage/search loop if
  not found or ambiguous.

## Current Design / 현재 설계

한국어:

현재 구현은 VLM-first matcher가 아니다. 중심은 deterministic computer vision
matcher이고, 그 바깥을 stage-search loop가 감싼다.

English:

The current implementation is not a VLM-first matcher. It is a deterministic
computer-vision matcher wrapped by a stage-search loop.

| Module | 한국어 역할 | English role |
| --- | --- | --- |
| `poc/workflow_2/align_key_matcher.py` | 레시피 template과 SEM frame을 전처리하고 Chamfer + ORB 점수를 계산한다. | Preprocesses the recipe template and SEM frame, then computes Chamfer + ORB scores. |
| `poc/workflow_2/search_align_key.py` | FOV capture, score 계산, stage 이동, match/escalation 종료를 반복한다. | Repeats FOV capture, scoring, stage movement, and match/escalation termination. |
| `poc/workflow_2/test_align_key_match.py` | 합성 SEM-like data로 positive/negative smoke test를 수행한다. | Builds synthetic SEM-like data for positive/negative smoke tests. |

## Science And Engineering Used / 사용된 과학과 공학

| Layer | 한국어 설명 | English summary | Code |
| --- | --- | --- | --- |
| SEM image formation | 같은 wafer mark도 brightness, noise, sharpness, scale이 달라질 수 있음을 설명한다. | Explains why the same wafer mark can change brightness, noise, sharpness, and apparent scale. | Input assumptions |
| Digital image preprocessing | raw grayscale image를 matching에 더 안정적인 표현으로 바꾼다. | Converts raw grayscale images into a more stable representation before matching. | `preprocess_for_matching()` |
| Contrast normalization | CLAHE로 local contrast를 보정해 edge detection을 안정화한다. | CLAHE improves local contrast before edge detection. | `cv2.createCLAHE()` |
| Noise reduction | Gaussian blur로 작은 random noise가 false edge가 되는 것을 줄인다. | Gaussian blur suppresses high-frequency noise that would create false edges. | `cv2.GaussianBlur()` |
| Edge detection | Canny가 align mark 구조를 binary edge pixels로 바꾼다. | Canny turns the align mark structure into binary edge pixels. | `cv2.Canny()` |
| Distance-transform geometry | frame edge map을 "nearest edge까지의 거리" map으로 바꾼다. | Converts a frame edge map into distance-to-nearest-edge values. | `cv2.distanceTransform()` |
| Template matching | template edge mask를 frame distance map 위에서 sliding하며 score를 계산한다. | Slides the template edge mask over the frame distance map and scores geometric alignment. | `cv2.matchTemplate(..., TM_CCORR)` |
| Feature matching | ORB keypoint/descriptor로 corners, dots, asymmetric marks를 비교한다. | ORB keypoints/descriptors compare distinctive corners and small structures. | `cv2.ORB_create()` |
| Robust geometry | RANSAC homography가 outlier match를 버리고 geometry가 맞는 inlier만 남긴다. | RANSAC homography keeps geometrically consistent matches and rejects outliers. | `cv2.findHomography(..., cv2.RANSAC)` |
| Search/orchestration | match/adjust/low decision에 따라 stage search를 진행한다. | A stateful loop handles match/adjust/low decisions and stage movement. | `search_align_key()` |

## Documents / 문서 구성

| Document | 한국어 목적 | English purpose |
| --- | --- | --- |
| [01-image-processing-background.md](01-image-processing-background.md) | SEM image, pixel, contrast, edge, distance transform, ORB, RANSAC, score decision을 기초부터 설명한다. | Beginner background for SEM images, pixels, contrast, edges, distance transforms, ORB, RANSAC, and score decisions. |
| [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md) | `poc/workflow_2`의 주요 함수가 어떤 알고리즘 개념을 구현하는지 연결한다. | Maps important functions in `poc/workflow_2` to the algorithmic concepts they implement. |
| [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md) | research finding, 현재 설계 평가, production 전에 고려할 better options를 정리한다. | Research findings, current design assessment, and better options to consider before production use. |

Related existing design notes / 관련 기존 설계 문서:

- [docs/search_align_key.md](../search_align_key.md): original algorithm design
  and operation note / 원래 알고리즘 설계와 운영 노트.
- [docs/test_align_key_match.md](../test_align_key_match.md): synthetic
  smoke-test plan and expected output / 합성 smoke test 계획과 기대 결과.

## Recommended Reading Order / 권장 읽기 순서

1. [01-image-processing-background.md](01-image-processing-background.md)
   먼저 읽는다. Canny, Chamfer, ORB, RANSAC, distance transform이 익숙하지
   않을 때 가장 중요하다.
   Read this first if terms like Canny, Chamfer, ORB, RANSAC, and distance
   transform are unfamiliar.
2. [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md)를
   `poc/workflow_2/align_key_matcher.py`와 함께 본다.
   Read it while looking at `poc/workflow_2/align_key_matcher.py`.
3. [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md)를
   threshold 변경이나 새 matcher 추가 전에 본다.
   Read it before changing thresholds or adding a new matcher.

## Key External References / 주요 외부 참고자료

- OpenCV template matching tutorial:
  <https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html>
- OpenCV Canny edge detector tutorial:
  <https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html>
- OpenCV distance transform tutorial:
  <https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html>
- OpenCV ORB class reference:
  <https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html>
- OpenCV feature matching + homography tutorial:
  <https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html>
- OpenCV contour features tutorial:
  <https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html>
- JEOL SEM secondary-electron image glossary:
  <https://www.jeol.com/words/semterms/20121024.070958.php>
- JEOL SEM charging glossary:
  <https://www.jeol.com/words/semterms/20121024.054158.php>
- NIST note on SEM noise in nanomanufacturing:
  <https://www.nist.gov/publications/nanomanufacturing-concerns-about-measurements-made-sem-part-v-dealing-noise>

