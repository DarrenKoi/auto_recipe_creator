# `poc/workflow_2` 연구 가이드

이 디렉터리는 `poc/workflow_2/`에 쓰인 과학/공학 배경을 설명한다. 이미지
처리를 잘 모르는 독자도 읽을 수 있도록 한국어로 정리했다. 다만 OpenCV 함수명,
matching, score, template처럼 코드와 논의에서 그대로 쓰는 기술 용어는 영어를
유지한다.

## 해결하려는 문제

```text
Recipe align-key image
        +
현재 SEM monitor frame
        |
        v
같은 physical fiducial pattern이 보이는지,
frame 안 위치와 stage/search loop의 다음 동작을 판단한다.
```

- 레시피에 저장된 align-key 이미지를 기준으로 현재 SEM monitor frame 안에서
  같은 물리적 fiducial pattern이 보이는지 판단한다.
- 보인다면 frame 안의 좌표를 반환한다.
- 보이지 않거나 애매하면 stage/search loop가 다음에 어디로 움직일지
  결정한다.

## 현재 설계

현재 구현은 VLM-first matcher가 아니다. 중심은 deterministic computer vision
matcher이고, 그 바깥을 stage-search loop가 감싼다.

| 모듈 | 역할 |
| --- | --- |
| `poc/workflow_2/align_key_matcher.py` | 레시피 template과 SEM frame을 전처리하고 Chamfer + ORB 점수를 계산한다. |
| `poc/workflow_2/search_align_key.py` | FOV capture, score 계산, stage 이동, match/escalation 종료를 반복한다. |
| `poc/workflow_2/test_align_key_match.py` | 합성 SEM-like data로 positive/negative smoke test를 수행한다. |

## 사용된 과학과 공학

| 개념 | 설명 | 관련 코드 |
| --- | --- | --- |
| SEM image formation | 같은 wafer mark도 brightness, noise, sharpness, scale이 달라질 수 있음을 설명한다. | Input assumptions |
| Digital image preprocessing | raw grayscale image를 matching에 더 안정적인 표현으로 바꾼다. | `preprocess_for_matching()` |
| Contrast normalization | CLAHE로 local contrast를 보정해 edge detection을 안정화한다. | `cv2.createCLAHE()` |
| Noise reduction | Gaussian blur로 작은 random noise가 false edge가 되는 것을 줄인다. | `cv2.GaussianBlur()` |
| Edge detection | Canny가 align mark 구조를 binary edge pixels로 바꾼다. | `cv2.Canny()` |
| Distance-transform geometry | frame edge map을 "nearest edge까지의 거리" map으로 바꾼다. | `cv2.distanceTransform()` |
| Template matching | template edge mask를 frame distance map 위에서 sliding하며 score를 계산한다. | `cv2.matchTemplate(..., TM_CCORR)` |
| Feature matching | ORB keypoint/descriptor로 corners, dots, asymmetric marks를 비교한다. | `cv2.ORB_create()` |
| Robust geometry | RANSAC homography가 outlier match를 버리고 geometry가 맞는 inlier만 남긴다. | `cv2.findHomography(..., cv2.RANSAC)` |
| Search/orchestration | match/adjust/low decision에 따라 stage search를 진행한다. | `search_align_key()` |

## 문서 구성

| 문서 | 목적 |
| --- | --- |
| [01-image-processing-background.md](01-image-processing-background.md) | SEM image, pixel, contrast, edge, distance transform, ORB, RANSAC, score decision을 기초부터 설명한다. |
| [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md) | `poc/workflow_2`의 주요 함수가 어떤 알고리즘 개념을 구현하는지 연결한다. |
| [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md) | research finding, 현재 설계 평가, production 전에 고려할 better options를 정리한다. |

관련 기존 설계 문서:

- [docs/search_align_key.md](../search_align_key.md): 원래 알고리즘 설계와 운영 노트.
- [docs/test_align_key_match.md](../test_align_key_match.md): 합성 smoke test 계획과 기대 결과.

## 권장 읽기 순서

1. [01-image-processing-background.md](01-image-processing-background.md)
   먼저 읽는다. Canny, Chamfer, ORB, RANSAC, distance transform이 익숙하지
   않을 때 가장 중요하다.
2. [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md)를
   `poc/workflow_2/align_key_matcher.py`와 함께 본다.
3. [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md)를
   threshold 변경이나 새 matcher 추가 전에 본다.

## 주요 외부 참고자료

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

