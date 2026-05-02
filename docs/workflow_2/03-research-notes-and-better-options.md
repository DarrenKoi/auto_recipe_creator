# Research Notes And Better Options

This document summarizes the research findings behind `workflow_2` and suggests
better options to consider before turning the prototype into a production
automation path.

## 1. Main Finding

The current architecture is directionally correct:

```text
classical CV matcher
  + deterministic score thresholds
  + debug images
  + stage-search loop
  + optional VLM assistance only for ambiguous/high-level cases
```

For SEM align-key search, this is stronger than a VLM-only approach because the
task needs repeatable geometric evidence. The important signal is not exact
brightness. It is the stable structure of the fiducial mark: boxes, crosses,
corners, dots, and their relative layout.

The current Chamfer + ORB implementation is a good first prototype. The largest
remaining risk is not the lack of a more complex model. It is the lack of real
SEM positive/negative calibration data.

## 2. Why Classical CV Is The Right Baseline

OpenCV template matching formalizes the sliding-window idea: compare a smaller
template against every possible region in a larger image and find the best
location. Raw template matching is fragile under scale, rotation, and local
contrast changes, but it establishes the core search pattern.

`workflow_2` improves on raw pixel matching by using edge geometry:

- Canny extracts edge pixels from contrast-normalized images.
- Distance transform turns the live FOV into a map of distance-to-nearest-edge.
- Chamfer matching measures how close template edges land to live-frame edges.
- ORB + RANSAC checks whether local details agree geometrically.

This matches the physical problem: SEM brightness can drift, but align-key edge
layout should remain relatively stable.

## 3. Better Option 1: Collect Real Calibration Data First

Priority: highest.

Before changing the algorithm, collect a small but real dataset:

| Data type | Minimum | Better |
| --- | ---: | ---: |
| Positive pairs: recipe key + live FOV containing same key | 20 | 50+ |
| Negative FOVs from same recipe/tool family | 20 | 50+ |
| Hard negatives: similar boxes/wrong marks/partial marks | 10 | 30+ |
| Repeated captures under focus/gain/scan changes | 5 per recipe | 10+ per recipe |

For each sample, save:

- recipe id/version
- template image
- live SEM frame
- true key center if visible
- `nm_per_pixel` if available
- tool/EQP id
- acquisition notes if known
- matcher output JSON and overlay JPEG

Then plot distributions:

```text
positive chamfer_score vs negative chamfer_score
positive orb_inlier_ratio vs negative orb_inlier_ratio
positive fused score vs negative fused score
localization error in pixels
best_scale distribution
```

Use those plots to set thresholds. The current `0.75` and `0.55` thresholds are
reasonable cold-start values, not validated production values.

## 4. Better Option 2: Add A Contour/Geometry Matcher For Box Marks

Priority: high.

The user-domain description says many align keys look like "large boxes,
3-4 boxes, or marks." That is exactly where contour-based geometry helps.

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

Why it is better for box-in-box marks:

- It measures what humans mean by "same mark": same number of boxes, same
  nesting, same relative geometry.
- It is more interpretable than ORB when ORB has too few keypoints.
- It can reject false positives where Chamfer sees edge clutter but the contour
  layout is wrong.

Recommended score contribution:

```text
score_struct = chamfer_score
score_feature = orb_inlier_ratio
score_shape = contour_layout_score
```

Then use a gated decision:

```text
match if chamfer_score >= C_high
     and (orb_inlier_ratio >= O_min or contour_layout_score >= G_min)
     and localization_error_estimate is acceptable
```

This is safer than a pure weighted average because it prevents one strong metric
from hiding another metric that completely failed.

## 5. Better Option 3: Use `nm_per_pixel` Metadata Whenever Possible

Priority: high.

The current fallback searches 5 scales:

```python
(0.7, 0.85, 1.0, 1.2, 1.4)
```

That is useful, but physical metadata is better:

```text
scale = template.nm_per_pixel / frame_nm_per_pixel
```

Benefits:

- Faster matching because the scale is known.
- Fewer false positives because the matcher searches fewer wrong sizes.
- Easier debugging because unexpected best-scale values become warning signals.

If metadata cannot be guaranteed, store the best scale after a successful match
and use a narrow scale window on the next frame.

## 6. Better Option 4: Add AKAZE Or SIFT Fallback

Priority: medium.

ORB is fast and appropriate as a first choice, but it may be weak on some SEM
fiducials:

- symmetric nested boxes
- low-texture surfaces
- blurred captures
- repeated corner patterns where descriptors are ambiguous

AKAZE is a good next fallback because OpenCV provides local feature matching
examples with homography/inlier checking, and it often works better on nonlinear
scale-space image structures than ORB. SIFT can also be considered for offline
or slower verification paths when licensing/deployment constraints are
acceptable in the target environment.

Practical approach:

```text
run ORB first
if ORB has too few Lowe matches or low inlier ratio:
    run AKAZE on the same Chamfer candidate crop
```

Do not run every feature detector on the whole frame every loop unless latency
measurements show it is acceptable.

## 7. Better Option 5: Coarse-To-Fine Search

Priority: medium.

Current multi-scale Chamfer scans the full frame per scale. That is simple and
fine for a prototype. For larger frames or more scales, use coarse-to-fine:

```text
downsample frame/template
  -> coarse Chamfer search
  -> top-N candidate boxes
  -> full-resolution Chamfer only near candidates
  -> ORB/AKAZE/contour verification only near final candidates
```

Benefits:

- Lower latency.
- More scales can be searched.
- Debug output can show top-N alternatives, not just the best candidate.

## 8. Better Option 6: Keep VLMs Out Of Final Scoring

Priority: high.

VLMs can be useful, but not as the primary matcher.

Good VLM uses:

- classify whether the SEM monitor appears empty, off-target, or blocked by UI
- summarize debug overlay for engineer review
- suggest coarse movement direction in `adjust` cases
- help with GUI/RCS controls outside the SEM image itself

Avoid:

- asking a VLM to provide final align-key coordinates
- using VLM confidence as calibrated match confidence
- letting VLM output override a low classical score without evidence

The production rule should be:

```text
OpenCV score decides the match.
VLM can suggest where to look next or explain ambiguous evidence.
```

## 9. Better Option 7: Improve Test Data Beyond Synthetic Gaussian Noise

Priority: medium.

The current synthetic smoke test is useful, but real SEM artifacts are richer.
Improve tests with:

- Poisson-like shot noise approximation
- scan-direction streaks
- local charging blobs and brightness tails
- blur/focus variation
- partial occlusion/cropping
- similar but wrong marks from neighboring wafer structures
- low-contrast marks where Canny thresholds become fragile

Keep synthetic tests, but mark them as "algorithm plumbing" tests. Production
thresholds should come from real captured data.

## 10. Not Recommended As Primary Solutions

### VLM-Only Matching

Not recommended because the output is not calibrated enough for stage movement.
It is also difficult to reproduce failure behavior from one run to another.

### Raw Pixel NCC Only

Not recommended because SEM brightness, contrast, charging, and focus can drift.
It can be used as a fast coarse filter only if real data proves it separates
positives and negatives.

### Silent Template Auto-Update

Not recommended. The recipe align key should remain a human-verified reference.
If the best score trend degrades, notify an engineer rather than silently
updating the template from a possibly wrong frame.

## 11. Suggested Next Implementation Plan

1. Add real-data evaluation output:
   write one JSONL row per frame with `recipe_id`, `score`, `chamfer`, `orb`,
   `best_xy`, `best_scale`, `decision`, and ground-truth label if known.
2. Collect real positive/negative SEM samples from office Windows runs.
3. Plot score distributions and update thresholds.
4. Add contour/geometry matcher for nested-box align keys.
5. Change final decision from weighted average only to a gated policy.
6. Use `nm_per_pixel` metadata when the server can provide it.
7. Add AKAZE fallback only if real ORB failures justify the extra cost.

## 12. References Consulted

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

