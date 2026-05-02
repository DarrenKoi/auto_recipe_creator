# Image Processing Background For `workflow_2`

This document explains the concepts used in `poc/workflow_2` without assuming
prior image-processing knowledge.

## 1. What An Image Is In This Code

In Python/OpenCV, an image is a NumPy array.

- A grayscale image is a 2D matrix: `height x width`.
- Each pixel usually has an integer brightness from `0` to `255`.
- `0` means black, `255` means white, and middle values are gray.
- A color image is usually `height x width x channels`; this workflow converts
  color input to grayscale because SEM monitor images and align-key structure
  are mainly intensity/shape problems.

In `align_key_matcher.py`, `_to_grayscale()` normalizes inputs into this simple
grayscale form before any matching.

## 2. Why SEM Images Are Hard To Match Directly

The same physical align key will not produce identical pixel values every time.
SEM images are created from electron-sample interactions, not from ordinary
visible-light photography. In typical SEM viewing, secondary electrons carry
surface/topography information. That makes edges and surface shape important,
but it also means the image can change when acquisition conditions change.

Common changes that matter for this workflow:

- Brightness and contrast change with detector gain, beam settings, and dwell
  time.
- Noise can increase when the frame is captured quickly or at low beam current.
- Charging can create low contrast, bright/dark patches, scan-direction tails,
  distortion, or drift.
- Focus, scan rotation, and stage motion can slightly move, blur, rotate, or
  rescale the mark.
- Recipe template resolution and live SEM monitor resolution may differ.

Therefore `workflow_2` should not ask "are these pixels equal?" It should ask
"do these two images contain the same geometric edge structure?"

## 3. The Current Preprocessing Pipeline

`preprocess_for_matching()` applies this pipeline:

```text
raw image
  -> grayscale
  -> CLAHE local contrast normalization
  -> Gaussian blur
  -> Canny edge detection
  -> distance transform
```

### Grayscale

Grayscale removes color-channel complexity. For SEM images, the main signal is
geometry and intensity, so one channel is enough.

### CLAHE

CLAHE means Contrast Limited Adaptive Histogram Equalization. It improves local
contrast by processing small tiles rather than stretching the whole image with
one global brightness curve. The "contrast limited" part matters because plain
adaptive equalization can amplify noise too much.

In this code:

```python
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
```

Meaning: use modest contrast enhancement in `8 x 8` local tiles.

### Gaussian Blur

Gaussian blur smooths small random noise before edge detection. Edge detectors
are very sensitive to noise because noise can look like tiny edges. The current
code uses:

```python
GAUSSIAN_SIGMA = 1.0
```

Meaning: blur enough to reduce small noise, but not enough to erase the align
mark structure.

### Canny Edge Detection

Canny converts the image into a binary edge map. In the edge map:

- edge pixel = mark boundary or strong intensity transition
- non-edge pixel = background/interior

The current thresholds are:

```python
CANNY_T_LOW = 60
CANNY_T_HIGH = 160
```

The important idea is that the align key is mostly a shape made of edges, not a
texture patch made of exact brightness values.

### Distance Transform

A distance transform converts a binary edge image into a map where each pixel
stores its distance to the nearest edge.

For example:

```text
edge pixel                  -> distance 0
1 pixel away from an edge   -> distance 1
10 pixels away from an edge -> distance 10
```

`workflow_2` applies `cv2.distanceTransform()` to the live SEM frame edge map.
Then it slides the recipe-template edge mask across that distance map. If the
template edges land close to frame edges, the average distance is low and the
match is good.

## 4. Chamfer Matching

Chamfer matching is the main structural matcher in `workflow_2`.

It works like this:

1. Convert the recipe template into a binary edge mask.
2. Convert the SEM frame into an edge map and then a distance transform.
3. Slide the template edge mask over the frame distance transform.
4. At each location, average the frame distances under the template edge pixels.
5. Lower average distance means better geometric alignment.
6. Convert average distance to a score:

```text
chamfer_score = exp(-mean_distance_px / DT_TAU_PX)
```

In the current code, `DT_TAU_PX = 10.0`. This means the score decays as the
average template-edge-to-frame-edge error grows.

Why this is useful:

- It ignores most raw brightness differences.
- It tolerates small edge gaps or noise.
- It is explainable: "the template edges are about N pixels from frame edges."

Main weakness:

- It can still produce false candidates in images with many similar edges.
- It does not by itself prove that the full mark geometry is correct.

## 5. Multi-Scale Search

If the recipe template and SEM frame use different physical resolution, a
128-pixel-wide mark in the template might appear as 110 or 150 pixels in the
live frame.

If `nm_per_pixel` metadata is available, the correct scale should be computed
from physical resolution. If not, the current code tries:

```python
DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)
```

That means: search for the template at 70%, 85%, 100%, 120%, and 140% size.

## 6. ORB Feature Matching

ORB is the second matcher. It looks for distinctive local points such as
corners, junctions, dots, and asymmetric marks.

The ORB flow in `compute_orb_inlier_ratio()` is:

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

Important terms:

- Keypoint: an image location that is likely repeatable, such as a corner.
- Descriptor: a compact signature around a keypoint.
- Hamming distance: distance between two binary descriptors.
- Lowe ratio test: keeps a match only when the best match is clearly better
  than the second-best match.
- RANSAC: fits a geometric model while rejecting wrong matches.
- Inlier: a match that agrees with the final geometry.

Why ORB is useful here:

- It checks whether the Chamfer candidate has matching local details.
- It helps reject random edge-like false positives.
- It adds rotation/scale tolerance within limits.

Main weakness:

- Symmetric box patterns may not have enough distinctive keypoints.
- Very smooth SEM images can produce too few stable features.
- ORB may be weaker than AKAZE or SIFT on some low-texture industrial images.

## 7. Final Score And Decisions

The current score is:

```text
score = 0.6 * chamfer_score + 0.4 * orb_inlier_ratio
```

Decision thresholds:

| Score range | Decision | Meaning |
| --- | --- | --- |
| `score >= 0.75` | `match` | Good enough to return the coordinate. |
| `0.55 <= score < 0.75` | `adjust` | Possible candidate; move or ask for assistance. |
| `score < 0.55` | `low` | No reliable match in this FOV. |

This is an engineering policy, not a law of nature. It must be calibrated with
real recipe/FOV data before relying on it in production.

## 8. Where VLMs Fit

VLMs are not the right primary tool for this specific matching step.

Reason:

- The task needs calibrated, repeatable pixel/geometry evidence.
- Current project VLMs are better at GUI grounding or OCR than SEM fiducial
  registration.
- VLM coordinates are not reliable enough for final stage movement decisions in
  a sparse SEM frame.

Better VLM role:

- Explain or classify ambiguous debug frames.
- Suggest a rough direction when the SEM view is clearly empty or structure is
  concentrated on one side.
- Act as a human-review helper, not the ground-truth matcher.

## 9. Quick Glossary

| Term | Plain meaning |
| --- | --- |
| FOV | Field of view. The current SEM monitor area being searched. |
| Template | The recipe align-key image to find. |
| Pixel | One cell in the image matrix. |
| Contrast | Difference between dark and bright regions. |
| Edge | Boundary where brightness changes sharply. |
| Keypoint | Repeatable local point such as a corner. |
| Descriptor | Numeric signature describing a keypoint neighborhood. |
| Inlier | A matched point that agrees with the geometric transform. |
| Outlier | A wrong or inconsistent match. |
| Distance transform | Image where each pixel stores distance to the nearest edge/background pixel. |
| Chamfer score | Edge-structure alignment score derived from distance transform. |
| Homography | A 2D projective transform mapping points from one image plane to another. |
| RANSAC | Robust model fitting that tolerates many wrong candidate matches. |

## Sources Consulted

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

