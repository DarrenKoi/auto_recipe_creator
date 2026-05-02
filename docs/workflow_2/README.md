# `poc/workflow_2` Research Guide

This directory explains the science and engineering behind `poc/workflow_2/`.
It is written for readers who know the automation goal but are new to image
processing.

`workflow_2` solves this problem:

```text
Recipe align-key image
        +
Current SEM monitor frame
        |
        v
Find whether the same physical fiducial pattern is visible,
where it is in the frame, and how the stage/search loop should continue.
```

The current implementation is not a VLM-first matcher. It is a deterministic
computer-vision matcher wrapped by a stage-search loop:

- `poc/workflow_2/align_key_matcher.py`
  preprocesses images and scores a recipe template against a SEM frame.
- `poc/workflow_2/search_align_key.py`
  repeatedly captures a FOV, calls the matcher, and moves the stage in a
  square-spiral search pattern until a match or escalation.
- `poc/workflow_2/test_align_key_match.py`
  creates synthetic SEM-like data to smoke-test positive/negative separation.

## Science And Engineering Used

| Layer | What it contributes | Where in code |
| --- | --- | --- |
| SEM image formation | Explains why the same wafer mark can change brightness, noise, sharpness, and apparent scale between captures. | Input assumptions in all modules |
| Digital image preprocessing | Converts raw grayscale images into a more stable representation before matching. | `preprocess_for_matching()` |
| Contrast normalization | CLAHE improves local contrast before edge detection. | `cv2.createCLAHE()` |
| Noise reduction | Gaussian blur suppresses high-frequency noise that would create false edges. | `cv2.GaussianBlur()` |
| Edge detection | Canny turns the align mark structure into binary edge pixels. | `cv2.Canny()` |
| Distance-transform geometry | Converts a frame edge map into "distance to nearest edge" values for Chamfer matching. | `cv2.distanceTransform()` |
| Template matching | Slides the template edge mask over the frame distance map and scores geometric alignment. | `cv2.matchTemplate(..., TM_CCORR)` |
| Feature matching | ORB keypoints/descriptors compare distinctive corners and small structures. | `cv2.ORB_create()` |
| Robust geometry | RANSAC homography keeps geometrically consistent matches and rejects outliers. | `cv2.findHomography(..., cv2.RANSAC)` |
| Search/orchestration | A stateful loop handles match/adjust/low decisions and stage movement. | `search_align_key()` |

## Documents

| Document | Purpose |
| --- | --- |
| [01-image-processing-background.md](01-image-processing-background.md) | Beginner background: SEM images, pixels, contrast, edges, distance transforms, ORB, RANSAC, and score decisions. |
| [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md) | Maps each important function in `poc/workflow_2` to the algorithmic concept it implements. |
| [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md) | Research findings, current design assessment, and better options to consider before production use. |

Related existing design notes:

- [docs/search_align_key.md](../search_align_key.md) is the original algorithm
  design and operation note.
- [docs/test_align_key_match.md](../test_align_key_match.md) documents the
  synthetic smoke-test plan and recent expected output.

## Recommended Reading Order

1. Read [01-image-processing-background.md](01-image-processing-background.md)
   first if terms like Canny, Chamfer, ORB, RANSAC, and distance transform are
   unfamiliar.
2. Read [02-code-and-algorithm-map.md](02-code-and-algorithm-map.md) while
   looking at `poc/workflow_2/align_key_matcher.py`.
3. Read [03-research-notes-and-better-options.md](03-research-notes-and-better-options.md)
   before changing thresholds or adding a new matcher.

## Key External References

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
