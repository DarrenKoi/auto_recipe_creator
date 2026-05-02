# Code And Algorithm Map

This document maps `poc/workflow_2` code to the image-processing concepts in
[01-image-processing-background.md](01-image-processing-background.md).

## 1. Package Structure

```text
poc/workflow_2/
|-- __init__.py
|-- align_key_matcher.py
|-- search_align_key.py
|-- test_align_key_match.py
`-- debug_images/
```

| File | Role |
| --- | --- |
| `__init__.py` | Defines package directories such as `DEBUG_IMAGE_DIR`. |
| `align_key_matcher.py` | Main classical CV matcher: preprocess, Chamfer score, ORB score, fused decision, overlay output. |
| `search_align_key.py` | Search-loop orchestration: capture FOV, score FOV, move stage, stop/escalate. |
| `test_align_key_match.py` | Synthetic data generator and smoke test for positive/negative separation. |

## 2. Main Data Flow

```text
Recipe align-key image
  -> build_template()
  -> _to_grayscale()
  -> preprocess_for_matching()
  -> AlignKeyTemplate

Current SEM frame
  -> compute_align_key_score()
  -> _to_grayscale()
  -> preprocess_for_matching()
  -> compute_chamfer_score()
  -> compute_orb_inlier_ratio()
  -> fused score + decision
  -> AlignKeyMatchResult
```

Then `search_align_key()` uses the result:

```text
decision == "match"  -> return final position
decision == "adjust" -> smaller nudge or VLM-assisted move
decision == "low"    -> spiral move; escalate after repeated low results
```

## 3. `AlignKeyTemplate`

Defined in `align_key_matcher.py`.

```python
@dataclass
class AlignKeyTemplate:
    recipe_id: str
    version: str
    raw_image: np.ndarray
    edge_map: np.ndarray
    distance_transform: np.ndarray
    nm_per_pixel: float | None
    key_type: str | None
    fetched_at: datetime
```

Purpose:

- Store the original recipe image for debugging and ORB.
- Store the precomputed template edge map so repeated frame scoring is faster.
- Carry optional physical scale metadata (`nm_per_pixel`).
- Carry recipe/version metadata so future caching can detect stale templates.

Note: `distance_transform` is stored but the current Chamfer implementation uses
the live frame distance transform plus the template edge mask. The template
distance transform may become useful if symmetric checks or contour/shape
analysis are added later.

## 4. `_to_grayscale()`

Concept: normalize input image shape and dtype.

What it does:

- Rejects `None`.
- Converts non-`uint8` arrays into `uint8`.
- Keeps 2D grayscale images.
- Converts BGR/BGRA images to grayscale.

Why it matters:

- OpenCV functions used later expect consistent image format.
- It removes ambiguity between screenshot formats.

## 5. `preprocess_for_matching()`

Concept: make SEM/template images more comparable.

Pipeline:

```text
grayscale
  -> CLAHE
  -> Gaussian blur
  -> Canny edge map
  -> inverted edge map
  -> distance transform
```

Code constants:

```python
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSSIAN_SIGMA = 1.0
CANNY_T_LOW = 60
CANNY_T_HIGH = 160
```

Important implementation detail:

`cv2.distanceTransform()` computes distance to zero-valued pixels, so the code
inverts the edge map before calling it. That makes edge locations become
distance `0`, which is what Chamfer scoring needs.

## 6. `build_template()`

Concept: convert a recipe image into reusable matching data.

It:

1. Converts `raw_image` to grayscale.
2. Builds `edge_map` and `distance_transform`.
3. Returns `AlignKeyTemplate`.

This should be called once per fetched recipe align-key image, not once per
frame.

## 7. `_chamfer_score_at_scale()`

Concept: single-scale edge-geometry matching.

Steps:

1. Resize the template edge map for the requested scale.
2. Convert template edges into a float mask.
3. Slide that mask over the frame distance transform with `cv2.matchTemplate()`.
4. Since the mask multiplies the frame distance map, the result is the sum of
   frame distances under template edge pixels.
5. Divide by edge count to get average edge distance.
6. Convert average distance into a score with:

```text
score = exp(-mean_dt / DT_TAU_PX)
```

Return values:

- `score`
- best center coordinate `(cx, cy)`
- scaled template size `(tw, th)`

Why `TM_CCORR` is used:

The template mask is `1` on template edges and `0` elsewhere. Correlation
therefore computes the sum of distance-transform values at template-edge
locations. The best match is the location with the minimum sum.

## 8. `compute_chamfer_score()`

Concept: multi-scale fallback.

It loops through:

```python
DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)
```

and keeps the scale/location with the highest Chamfer score.

This is a practical fallback when `nm_per_pixel` is unavailable. If physical
resolution metadata is available, `compute_align_key_score()` uses a single
scale instead.

## 9. `compute_orb_inlier_ratio()`

Concept: feature-level verification.

Steps:

1. Detect ORB keypoints and descriptors in the template and frame crop.
2. Match binary descriptors with `cv2.BFMatcher(cv2.NORM_HAMMING)`.
3. Apply Lowe ratio test:

```python
if m.distance < LOWE_RATIO * n.distance:
    good.append(m)
```

4. Require at least `MIN_LOWE_MATCHES = 8`.
5. Fit homography with `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
6. Return:

```text
inlier_ratio = number_of_RANSAC_inliers / number_of_good_matches
```

Interpretation:

- High ratio: many local features agree with one geometric transform.
- Low ratio: candidate likely false, too symmetric, too noisy, or feature-poor.

## 10. `compute_align_key_score()`

Concept: frame-level decision.

Inputs:

- `template`: preprocessed recipe align key.
- `frame`: current SEM frame.
- `frame_nm_per_pixel`: optional physical scale.
- `roi_hint`: optional `(x, y, w, h)` crop.

Important behavior:

- If both template and frame physical scale are known:

```text
scale = template.nm_per_pixel / frame_nm_per_pixel
```

- If scale is unknown, use `DEFAULT_SCALES`.
- If `roi_hint` is present, matching is limited to that crop but output
  coordinates are converted back to full-frame coordinates.
- ORB is run only around the best Chamfer candidate, not over the whole frame.

Final scoring:

```text
score = 0.6 * chamfer_score + 0.4 * orb_inlier_ratio
```

Decision:

```text
score >= 0.75 -> match
score >= 0.55 -> adjust
else          -> low
```

Output:

```python
AlignKeyMatchResult(
    score=...,
    chamfer_score=...,
    orb_inlier_ratio=...,
    best_xy=...,
    best_scale=...,
    decision=...,
    debug_overlay=...,
)
```

## 11. Debug Overlay

`_render_overlay()` draws:

- candidate rectangle
- center cross
- decision text
- final score
- Chamfer score
- ORB inlier ratio
- best scale

`save_overlay_jpeg()` stores the overlay as JPEG under `debug_images`. This is
important because SEM failures are hard to diagnose from logs alone.

## 12. `search_align_key()`

Concept: deterministic observe-score-act loop.

Inputs are injected:

```python
capture_fn(state) -> np.ndarray
move_stage_fn(state, dx, dy) -> AlignKeySearchState
```

That design lets the same search code run in two environments:

- Mac/offline demo: synthetic wafer image and mock stage coordinates.
- Office Windows/RCS: real monitor capture and real stage movement.

Loop behavior:

1. Capture current FOV.
2. Compute align-key score.
3. Save overlay if debug directory is provided.
4. If `match`, return success.
5. If `adjust`, use VLM move hint if provided, otherwise do a small spiral
   nudge.
6. If `low`, increment `low_streak`.
7. If repeated low results reach `low_streak_limit`, escalate.
8. Otherwise, move with square-spiral search.

## 13. Square-Spiral Search

`_square_spiral_step()` generates movement deltas in this pattern:

```text
start
  -> right
  -> up
  -> left, left
  -> down, down
  -> right, right, right
  -> ...
```

This is useful when the current FOV may be near the target but the direction is
unknown. It searches nearby space before moving farther away.

## 14. `test_align_key_match.py`

Concept: synthetic smoke test.

It creates:

- A `box` template similar to nested align boxes.
- Positive frames where the template exists with scale, rotation, brightness,
  contrast, charging-like gradients, and noise.
- Negative frames with no template, random blobs, wrong pattern, strong
  charging-like gradient, or out-of-scale template.

Pass criteria:

- Positives must be `match` or `adjust` and localize within 20 px.
- Negatives must be `low`.

This is a smoke test, not production validation. Real SEM data is still needed
to calibrate thresholds and failure modes.

## 15. Current Assumptions To Remember

- The align-key signal is mostly edge geometry.
- Rotation is expected to be small.
- Scale mismatch is covered only from `0.7x` to `1.4x` unless metadata exists.
- The fused threshold values are cold-start defaults.
- ORB verification can fail on very symmetric or very low-texture marks.
- Synthetic tests prove code plumbing and basic separation, not real SEM
  production accuracy.
