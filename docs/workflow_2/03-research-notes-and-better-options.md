# 연구 노트와 개선 옵션

이 문서는 `workflow_2` 조사 결과와 production 적용 전에 고려할 더 나은 옵션을
정리한다. OpenCV, matching, score, template처럼 코드에서 그대로 쓰는 기술 용어는
영어를 유지한다.

## 핵심 결론

현재 방향은 맞다.

```text
classical CV matcher
  + deterministic score thresholds
  + debug images
  + stage-search loop
  + optional VLM assistance only for ambiguous/high-level cases
```

SEM align-key search는 VLM-only 접근보다 classical CV 중심 접근이 더 적합하다.
이 task는 "비슷해 보인다"는 자연어 판단이 아니라 repeatable geometric evidence가
필요하다. 중요한 신호는 exact brightness가 아니라 fiducial mark의 안정적인
structure다: boxes, crosses, corners, dots, relative layout.

현재 Chamfer + ORB 구현은 first prototype으로 적절하다. 가장 큰 risk는 더 복잡한
model이 없는 것이 아니라, real SEM positive/negative calibration data가 아직
부족하다는 점이다.

## Classical CV가 baseline으로 맞는 이유

OpenCV template matching은 작은 template을 큰 image의 모든 위치에 sliding하면서
가장 잘 맞는 위치를 찾는 기본 search pattern을 제공한다. Raw template matching은
scale, rotation, local contrast 변화에 약하지만, "search location" 문제의 기본
형태를 잘 설명한다.

`workflow_2`는 raw pixel matching 대신 edge geometry를 사용해 이 약점을 줄인다.

- Canny가 contrast-normalized image에서 edge pixel을 추출한다.
- Distance transform이 live FOV를 nearest-edge distance map으로 바꾼다.
- Chamfer matching이 template edge와 live-frame edge가 얼마나 가까운지 본다.
- ORB + RANSAC이 local detail이 하나의 geometry로 일치하는지 검증한다.

이는 SEM physical problem과 맞다. SEM brightness는 drift할 수 있지만, align-key
edge layout은 비교적 안정적이어야 한다.

## 실제 calibration data 먼저 수집

우선순위: 최고

Algorithm을 더 복잡하게 만들기 전에 작은 real dataset이 먼저 필요하다.

| 데이터 종류 | 최소 | 권장 |
| --- | ---: | ---: |
| Positive pair: 같은 key가 보이는 recipe key + live FOV | 20 | 50+ |
| Negative FOV: 같은 recipe/tool family에서 key가 없는 FOV | 20 | 50+ |
| Hard negative: similar boxes/wrong marks/partial marks | 10 | 30+ |
| focus/gain/scan 변화 조건의 반복 capture | 5 per recipe | 10+ per recipe |

각 sample에 저장할 것:

- recipe id/version
- template image
- live SEM frame
- true key center if visible
- `nm_per_pixel` if available
- tool/EQP id
- acquisition notes if known
- matcher output JSON and overlay JPEG

그 다음 distribution을 본다.

```text
positive chamfer_score vs negative chamfer_score
positive orb_inlier_ratio vs negative orb_inlier_ratio
positive fused score vs negative fused score
localization error in pixels
best_scale distribution
```

이 plot으로 threshold를 정해야 한다. 현재 `0.75`와 `0.55`는 reasonable cold-start
default일 뿐, production-validated threshold가 아니다.

## Box mark용 contour/geometry matcher 추가

우선순위: 높음

사용자 도메인 설명상 많은 align key는 "large boxes", "3-4 boxes", "marks"처럼
보인다. 이 경우 contour-based geometry가 잘 맞는다.

추가 matcher 제안:

```text
frame/template
  -> CLAHE
  -> blur
  -> Canny or adaptive threshold
  -> findContours()
  -> filter rectangular contours by area/aspect ratio
  -> approximate polygons with approxPolyDP()
  -> compare box centers, sizes, nesting, and relative layout
```

Box-in-box mark에 좋은 이유:

- 사람이 말하는 "same mark"를 직접 측정한다: box 개수, nesting, relative geometry.
- ORB가 keypoint를 충분히 못 잡을 때도 해석 가능한 evidence가 나온다.
- Chamfer가 edge clutter에 속는 false positive를 줄일 수 있다.

권장 score 구조:

```text
score_struct = chamfer_score
score_feature = orb_inlier_ratio
score_shape = contour_layout_score
```

단순 weighted average보다 gated decision이 더 안전하다.

```text
match if chamfer_score >= C_high
     and (orb_inlier_ratio >= O_min or contour_layout_score >= G_min)
     and localization_error_estimate가 허용 범위 안에 있음
```

이 방식은 한 metric이 높아서 다른 metric의 complete failure를 가리는 문제를 줄인다.

## 가능하면 `nm_per_pixel` metadata 사용

우선순위: 높음

현재 fallback은 5개 scale을 검색한다.

```python
(0.7, 0.85, 1.0, 1.2, 1.4)
```

하지만 physical metadata가 있으면 아래가 더 낫다.

```text
scale = template.nm_per_pixel / frame_nm_per_pixel
```

장점:

- scale을 알고 있으므로 matching이 빠르다.
- 잘못된 scale을 덜 검색하므로 false positive가 줄어든다.
- 예상 밖 `best_scale`이 warning signal이 되어 debug가 쉬워진다.

Metadata를 항상 받을 수 없다면, successful match 후 best scale을 저장하고 다음
frame부터는 좁은 scale window만 재탐색하는 것이 좋다.

## AKAZE 또는 SIFT fallback 추가

우선순위: 중간

ORB는 빠르고 first choice로 적절하지만, 일부 SEM fiducial에서는 약할 수 있다.

- symmetric nested boxes
- low-texture surfaces
- blurred captures
- repeated corner patterns with ambiguous descriptors

AKAZE는 좋은 fallback 후보이다. OpenCV가 local feature matching과
homography/inlier checking 예제를 제공하고, 일부 nonlinear scale-space structure에
ORB보다 강할 수 있다. SIFT도 offline 또는 slower verification path에서는 고려할
수 있다. 단, deployment/license/performance 조건을 확인해야 한다.

실행 방식:

```text
먼저 ORB 실행
ORB Lowe match가 너무 적거나 inlier ratio가 낮으면:
    같은 Chamfer candidate crop에서 AKAZE 실행
```

Latency 측정 없이 모든 detector를 full frame에 매번 실행하는 것은 피한다.

## Coarse-to-fine 탐색

우선순위: 중간

현재 multi-scale Chamfer는 scale마다 full frame을 scan한다. Prototype에는 단순하고
충분하지만, frame이 커지거나 scale 수가 늘면 coarse-to-fine이 낫다.

```text
downsample frame/template
  -> coarse Chamfer search
  -> top-N candidate boxes
  -> full-resolution Chamfer only near candidates
  -> ORB/AKAZE/contour verification only near final candidates
```

장점:

- latency 감소
- 더 많은 scale 탐색 가능
- best candidate 하나만이 아니라 top-N alternative를 debug할 수 있음

## VLM을 final scoring에서 제외

우선순위: 높음

VLM은 유용하지만 primary matcher나 final score source로 쓰면 안 된다.

좋은 VLM 사용 방식:

- SEM monitor가 empty/off-target/blocked인지 classification
- engineer review용 debug overlay summary
- `adjust` case에서 coarse movement direction 제안
- SEM image matching 바깥의 RCS GUI control 지원

피해야 할 사용 방식:

- VLM에게 final align-key coordinate를 맡김
- VLM confidence를 calibrated match confidence처럼 사용
- Classical score가 low인데 VLM output만으로 override

운영 규칙:

```text
match 여부는 OpenCV score가 결정한다.
VLM은 다음 탐색 위치를 제안하거나 ambiguous evidence를 설명한다.
```

## 합성 테스트 개선

우선순위: 중간

현재 synthetic smoke test는 유용하지만 실제 SEM artifact는 더 복잡하다. Test data에
다음을 추가하면 좋다.

- Poisson-like shot noise approximation
- scan-direction streaks
- local charging blobs and brightness tails
- blur/focus variation
- partial occlusion/cropping
- neighboring wafer structure에서 오는 similar but wrong marks
- Canny threshold가 흔들리는 low-contrast marks

Synthetic test는 계속 유지하되, "algorithm plumbing test"로 명확히 구분한다.
Production threshold는 real captured data에서 나와야 한다.

## 주요 해결책으로 비추천

### VLM-only 매칭

Stage movement에 필요한 calibration된 coordinate/score를 안정적으로 제공하지
못한다. 같은 input에서도 failure behavior를 재현하고 분석하기 어렵다.

### Raw pixel NCC 단독 사용

SEM brightness, contrast, charging, focus drift에 취약하다. Real data에서
분리력이 입증되면 fast coarse filter로는 쓸 수 있지만, primary matcher로는
부족하다.

### 조용한 template 자동 업데이트

Recipe align key는 human-verified reference로 유지해야 한다. Best score trend가
나빠지면 template을 조용히 바꾸지 말고 engineer에게 갱신 권고를 보내야 한다.

## 다음 구현 제안

1. Real-data evaluation output을 추가한다.
   JSONL row에 `recipe_id`, `score`, `chamfer`, `orb`, `best_xy`, `best_scale`,
   `decision`, ground-truth label을 저장한다.
2. Office Windows run에서 real positive/negative SEM sample을 수집한다.
3. Score distribution을 plot하고 threshold를 업데이트한다.
4. Nested-box align key용 contour/geometry matcher를 추가한다.
5. Final decision을 weighted average only에서 gated policy로 바꾼다.
6. Server가 제공할 수 있으면 `nm_per_pixel` metadata를 사용한다.
7. Real ORB failure가 확인될 때만 AKAZE fallback을 추가한다.

## 참고 자료

OpenCV 공식 문서:

- Template matching:
  <https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html>
- Canny edge detector:
  <https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html>
- Distance transform:
  <https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html>
- ORB class reference:
  <https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html>
- Feature matching + homography:
  <https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html>
- AKAZE matching:
  <https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html>
- Contour features:
  <https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html>

Computer vision 논문과 publication 기록:

- Rublee, Rabaud, Konolige, Bradski, "ORB: An efficient alternative to SIFT or
  SURF", ICCV 2011, DOI `10.1109/ICCV.2011.6126544`.
- Fischler and Bolles, "Random Sample Consensus", Communications of the ACM,
  1981, DOI `10.1145/358669.358692`.
- Borgefors, "Distance transformations in digital images", Computer Vision,
  Graphics, and Image Processing, 1986, DOI `10.1016/S0734-189X(86)80047-0`.
- Borgefors, "Hierarchical chamfer matching: a parametric edge matching
  algorithm", IEEE TPAMI, 1988, DOI `10.1109/34.9107`.

SEM 배경:

- JEOL secondary-electron image glossary:
  <https://www.jeol.com/words/semterms/20121024.070958.php>
- JEOL charging glossary:
  <https://www.jeol.com/words/semterms/20121024.054158.php>
- JEOL secondary-electron glossary:
  <https://www.jeol.com/words/semterms/20121024.070758.php>
- NIST SEM noise note:
  <https://www.nist.gov/publications/nanomanufacturing-concerns-about-measurements-made-sem-part-v-dealing-noise>
- NIST SEM interpretation note:
  <https://www.nist.gov/news-events/news/2025/04/nist-study-aims-improve-utility-scanning-electron-microscope>

