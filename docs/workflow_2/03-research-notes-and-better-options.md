# Research Notes And Better Options / 연구 노트와 개선 옵션

이 문서는 `workflow_2` 조사 결과와 production 적용 전에 고려할 더 나은 옵션을
정리한다. 한국어 설명을 먼저 두고 English summary를 함께 둔다.

This document summarizes research findings behind `workflow_2` and better
options to consider before production use. Korean explanations come first,
followed by English summaries.

## 1. Main Finding / 핵심 결론

한국어:

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

English:

The current direction is correct.

```text
classical CV matcher
  + deterministic score thresholds
  + debug images
  + stage-search loop
  + optional VLM assistance only for ambiguous/high-level cases
```

For SEM align-key search, a classical-CV-centered approach is more appropriate
than VLM-only matching. The task needs repeatable geometric evidence, not a
natural-language impression that two images "look similar." The important
signal is stable fiducial structure: boxes, crosses, corners, dots, and relative
layout.

The current Chamfer + ORB implementation is a good first prototype. The largest
risk is not the lack of a more complex model; it is the lack of real SEM
positive/negative calibration data.

## 2. Why Classical CV Is The Right Baseline / Classical CV가 baseline으로 맞는 이유

한국어:

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

English:

OpenCV template matching provides the basic search pattern: slide a smaller
template over a larger image and find the best location. Raw template matching
is fragile under scale, rotation, and local contrast changes, but it explains
the core search-location problem.

`workflow_2` reduces these weaknesses by using edge geometry instead of raw
pixel matching.

- Canny extracts edge pixels from contrast-normalized images.
- Distance transform converts the live FOV into a nearest-edge distance map.
- Chamfer matching measures how close template edges land to live-frame edges.
- ORB + RANSAC verifies that local details agree under one geometry.

This matches the SEM physical problem: SEM brightness can drift, but align-key
edge layout should remain relatively stable.

## 3. Better Option 1: Collect Real Calibration Data First / 실제 calibration data 먼저 수집

Priority: highest / 우선순위: 최고

한국어:

Algorithm을 더 복잡하게 만들기 전에 작은 real dataset이 먼저 필요하다.

| Data type | Minimum | Better |
| --- | ---: | ---: |
| Positive pairs: recipe key + live FOV containing same key | 20 | 50+ |
| Negative FOVs from same recipe/tool family | 20 | 50+ |
| Hard negatives: similar boxes/wrong marks/partial marks | 10 | 30+ |
| Repeated captures under focus/gain/scan changes | 5 per recipe | 10+ per recipe |

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

English:

Before making the algorithm more complex, collect a small real dataset.

Save recipe id/version, template image, live SEM frame, true key center if
visible, `nm_per_pixel` if available, tool/EQP id, acquisition notes, matcher
JSON, and overlay JPEG.

Then plot positive/negative distributions for Chamfer score, ORB inlier ratio,
fused score, localization error, and best scale. Thresholds should be set from
these plots. The current `0.75` and `0.55` values are reasonable cold-start
defaults, not production-validated thresholds.

## 4. Better Option 2: Add A Contour/Geometry Matcher For Box Marks / Box mark용 contour/geometry matcher 추가

Priority: high / 우선순위: 높음

한국어:

사용자 도메인 설명상 많은 align key는 "large boxes", "3-4 boxes", "marks"처럼
보인다. 이 경우 contour-based geometry가 잘 맞는다.

Proposed added matcher:

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
     and localization_error_estimate is acceptable
```

이 방식은 한 metric이 높아서 다른 metric의 complete failure를 가리는 문제를 줄인다.

English:

Many align keys look like large boxes, 3-4 boxes, or other structured marks.
Contour-based geometry fits this case well.

Suggested matcher:

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

Why it helps:

- directly measures what humans mean by "same mark": same box count, nesting,
  and relative geometry
- provides interpretable evidence when ORB has too few keypoints
- rejects false positives where Chamfer sees edge clutter but the contour layout
  is wrong

A gated decision is safer than a pure weighted average because it prevents one
strong metric from hiding a complete failure in another metric.

## 5. Better Option 3: Use `nm_per_pixel` Metadata Whenever Possible / 가능하면 `nm_per_pixel` metadata 사용

Priority: high / 우선순위: 높음

한국어:

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

English:

The current fallback searches:

```python
(0.7, 0.85, 1.0, 1.2, 1.4)
```

Physical metadata is better when available:

```text
scale = template.nm_per_pixel / frame_nm_per_pixel
```

Benefits: faster matching, fewer false positives from wrong scales, and easier
debugging because unexpected `best_scale` values become warning signals. If
metadata is not guaranteed, store the successful best scale and search only a
narrow scale window on the next frame.

## 6. Better Option 4: Add AKAZE Or SIFT Fallback / AKAZE 또는 SIFT fallback 추가

Priority: medium / 우선순위: 중간

한국어:

ORB는 빠르고 first choice로 적절하지만, 일부 SEM fiducial에서는 약할 수 있다.

- symmetric nested boxes
- low-texture surfaces
- blurred captures
- repeated corner patterns with ambiguous descriptors

AKAZE는 좋은 fallback 후보이다. OpenCV가 local feature matching과
homography/inlier checking 예제를 제공하고, 일부 nonlinear scale-space structure에
ORB보다 강할 수 있다. SIFT도 offline 또는 slower verification path에서는 고려할
수 있다. 단, deployment/license/performance 조건을 확인해야 한다.

Practical approach:

```text
run ORB first
if ORB has too few Lowe matches or low inlier ratio:
    run AKAZE on the same Chamfer candidate crop
```

Latency 측정 없이 모든 detector를 full frame에 매번 실행하는 것은 피한다.

English:

ORB is fast and appropriate as a first choice, but it can be weak on symmetric
nested boxes, low-texture surfaces, blurred captures, or repeated corner
patterns.

AKAZE is a good fallback candidate because OpenCV supports local-feature
matching with homography/inlier checking, and AKAZE may work better than ORB on
some nonlinear scale-space structures. SIFT can also be considered for offline
or slower verification paths if deployment, licensing, and performance
constraints allow it.

Practical approach:

```text
run ORB first
if ORB has too few Lowe matches or low inlier ratio:
    run AKAZE on the same Chamfer candidate crop
```

Avoid running every detector on the full frame every loop unless latency
measurements justify it.

## 7. Better Option 5: Coarse-To-Fine Search / Coarse-to-fine 탐색

Priority: medium / 우선순위: 중간

한국어:

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

English:

Current multi-scale Chamfer scans the full frame per scale. That is simple and
fine for a prototype, but coarse-to-fine search is better for larger frames or
more scales.

```text
downsample frame/template
  -> coarse Chamfer search
  -> top-N candidate boxes
  -> full-resolution Chamfer only near candidates
  -> ORB/AKAZE/contour verification only near final candidates
```

Benefits: lower latency, more searchable scales, and top-N debug alternatives
instead of only the best candidate.

## 8. Better Option 6: Keep VLMs Out Of Final Scoring / VLM을 final scoring에서 제외

Priority: high / 우선순위: 높음

한국어:

VLM은 유용하지만 primary matcher나 final score source로 쓰면 안 된다.

좋은 VLM use:

- SEM monitor가 empty/off-target/blocked인지 classification
- engineer review용 debug overlay summary
- `adjust` case에서 coarse movement direction 제안
- SEM image matching 바깥의 RCS GUI control 지원

피해야 할 use:

- VLM에게 final align-key coordinate를 맡김
- VLM confidence를 calibrated match confidence처럼 사용
- Classical score가 low인데 VLM output만으로 override

Production rule:

```text
OpenCV score decides the match.
VLM can suggest where to look next or explain ambiguous evidence.
```

English:

VLMs are useful, but should not be the primary matcher or final score source.

Good uses:

- classify whether the SEM monitor is empty, off-target, or blocked by UI
- summarize debug overlays for engineer review
- suggest coarse movement direction in `adjust` cases
- help RCS GUI control outside SEM-image matching

Avoid:

- asking a VLM for final align-key coordinates
- using VLM confidence as calibrated match confidence
- overriding a low classical score based only on VLM output

Production rule:

```text
OpenCV score decides the match.
VLM can suggest where to look next or explain ambiguous evidence.
```

## 9. Better Option 7: Improve Synthetic Tests / 합성 테스트 개선

Priority: medium / 우선순위: 중간

한국어:

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

English:

The current synthetic smoke test is useful, but real SEM artifacts are richer.
Add Poisson-like shot noise, scan-direction streaks, charging blobs/tails,
blur/focus variation, partial occlusion/cropping, similar but wrong neighboring
marks, and low-contrast cases where Canny thresholds become fragile.

Keep synthetic tests, but label them as algorithm-plumbing tests. Production
thresholds must come from real captured data.

## 10. Not Recommended As Primary Solutions / primary solution으로 비추천

### VLM-Only Matching / VLM-only 매칭

한국어:

Stage movement에 필요한 calibration된 coordinate/score를 안정적으로 제공하지
못한다. 같은 input에서도 failure behavior를 재현하고 분석하기 어렵다.

English:

Not recommended because it does not provide calibrated coordinates/scores
reliably enough for stage movement, and failure behavior is hard to reproduce
and analyze.

### Raw Pixel NCC Only / Raw pixel NCC 단독 사용

한국어:

SEM brightness, contrast, charging, focus drift에 취약하다. Real data에서
분리력이 입증되면 fast coarse filter로는 쓸 수 있지만, primary matcher로는
부족하다.

English:

Not recommended as the primary matcher because SEM brightness, contrast,
charging, and focus can drift. It can be used as a fast coarse filter only if
real data proves it separates positives and negatives.

### Silent Template Auto-Update / 조용한 template 자동 업데이트

한국어:

Recipe align key는 human-verified reference로 유지해야 한다. Best score trend가
나빠지면 template을 조용히 바꾸지 말고 engineer에게 갱신 권고를 보내야 한다.

English:

The recipe align key should remain a human-verified reference. If best-score
trends degrade, notify an engineer rather than silently updating the template
from a possibly wrong frame.

## 11. Suggested Next Implementation Plan / 다음 구현 제안

한국어:

1. Real-data evaluation output을 추가한다.
   JSONL row에 `recipe_id`, `score`, `chamfer`, `orb`, `best_xy`, `best_scale`,
   `decision`, ground-truth label을 저장한다.
2. Office Windows run에서 real positive/negative SEM sample을 수집한다.
3. Score distribution을 plot하고 threshold를 업데이트한다.
4. Nested-box align key용 contour/geometry matcher를 추가한다.
5. Final decision을 weighted average only에서 gated policy로 바꾼다.
6. Server가 제공할 수 있으면 `nm_per_pixel` metadata를 사용한다.
7. Real ORB failure가 확인될 때만 AKAZE fallback을 추가한다.

English:

1. Add real-data evaluation output.
   Write JSONL rows with `recipe_id`, `score`, `chamfer`, `orb`, `best_xy`,
   `best_scale`, `decision`, and ground-truth label if known.
2. Collect real positive/negative SEM samples from office Windows runs.
3. Plot score distributions and update thresholds.
4. Add a contour/geometry matcher for nested-box align keys.
5. Move final decision from weighted-average-only to a gated policy.
6. Use `nm_per_pixel` metadata when the server can provide it.
7. Add AKAZE fallback only if real ORB failures justify the extra cost.

## 12. References Consulted / 참고 자료

OpenCV official docs:

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

Computer-vision papers and publication records:

- Rublee, Rabaud, Konolige, Bradski, "ORB: An efficient alternative to SIFT or
  SURF", ICCV 2011, DOI `10.1109/ICCV.2011.6126544`.
- Fischler and Bolles, "Random Sample Consensus", Communications of the ACM,
  1981, DOI `10.1145/358669.358692`.
- Borgefors, "Distance transformations in digital images", Computer Vision,
  Graphics, and Image Processing, 1986, DOI `10.1016/S0734-189X(86)80047-0`.
- Borgefors, "Hierarchical chamfer matching: a parametric edge matching
  algorithm", IEEE TPAMI, 1988, DOI `10.1109/34.9107`.

SEM background:

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

