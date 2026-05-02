# 코드와 알고리즘 맵

이 문서는 `poc/workflow_2`의 각 코드 블록이 어떤 image-processing 개념을
구현하는지 연결한다. OpenCV, matching, score, template처럼 코드에서 그대로 쓰는
기술 용어는 영어를 유지한다.

## 패키지 구조

```text
poc/workflow_2/
|-- __init__.py
|-- align_key_matcher.py
|-- search_align_key.py
|-- test_align_key_match.py
`-- debug_images/
```

| 파일 | 역할 |
| --- | --- |
| `__init__.py` | `DEBUG_IMAGE_DIR`, `LOG_DIR` 같은 package 기준 경로를 정의한다. |
| `align_key_matcher.py` | Classical CV matcher. Preprocess, Chamfer score, ORB score, fused decision, overlay 생성을 담당한다. |
| `search_align_key.py` | Capture FOV, score 계산, stage 이동, match/escalation 종료를 관리하는 search loop다. |
| `test_align_key_match.py` | 합성 data generator와 positive/negative smoke test다. |

## 주요 데이터 흐름

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

Recipe align-key image는 한 번 template으로 만든 뒤 반복 사용한다. Live SEM
frame은 매 loop마다 capture되어 같은 preprocessing/matching path를 지난다. 최종
결과는 `match`, `adjust`, `low` decision으로 search loop에 전달된다.

Decision 흐름:

```text
decision == "match"  -> 최종 position 반환
decision == "adjust" -> 작은 보정 이동 또는 VLM-assisted move
decision == "low"    -> spiral move; 반복 low 이후 escalation
```

## `AlignKeyTemplate`

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

`AlignKeyTemplate`은 서버나 recipe source에서 받은 align key를 matching에 바로
쓸 수 있는 형태로 보존한다.

- `raw_image`: debug와 ORB input에 사용한다.
- `edge_map`: Chamfer matching에 쓰는 template edge mask다.
- `distance_transform`: 현재 구현에서는 직접 scoring에 쓰이지 않지만, 향후
  symmetric check나 shape analysis에 쓸 수 있다.
- `nm_per_pixel`: physical scale metadata다. 있으면 multi-scale fallback보다
  정확하고 빠르게 scale을 정할 수 있다.
- `recipe_id`, `version`, `fetched_at`: cache invalidation과 debug 추적에 필요하다.

## `_to_grayscale()`

Input image의 dtype과 channel shape를 matching 가능한 grayscale `uint8`로
정리한다.

하는 일:

- `None` input은 reject한다.
- non-`uint8` array는 `0..255`로 clip하고 `uint8`로 바꾼다.
- 2D grayscale은 그대로 사용한다.
- BGR/BGRA image는 grayscale로 변환한다.

왜 중요한가:

- 이후 OpenCV 함수들이 일관된 format을 기대한다.
- screenshot source마다 shape/dtype이 달라도 matcher code가 단순해진다.

## `preprocess_for_matching()`

SEM/template image를 matching에 더 안정적인 representation으로 바꾼다.

처리 흐름:

```text
grayscale
  -> CLAHE
  -> Gaussian blur
  -> Canny edge map
  -> inverted edge map
  -> distance transform
```

현재 상수:

```python
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSSIAN_SIGMA = 1.0
CANNY_T_LOW = 60
CANNY_T_HIGH = 160
```

중요한 구현 detail:

`cv2.distanceTransform()`은 zero-valued pixel까지의 거리를 계산한다. 그래서 code는
edge map을 invert한 뒤 distance transform을 호출한다. 이렇게 해야 edge 위치가
distance `0`이 되고, Chamfer score 계산에 맞는 형태가 된다.

## `build_template()`

Recipe image를 한 번 preprocessing해서 reusable `AlignKeyTemplate`로 만든다.

동작:

1. `raw_image`를 grayscale로 변환한다.
2. `edge_map`과 `distance_transform`을 만든다.
3. metadata와 함께 `AlignKeyTemplate`를 반환한다.

Frame마다 호출하면 낭비다. 같은 recipe template에 대해서는 한 번만 호출하고,
live frame마다 `compute_align_key_score()`를 호출하는 구조가 맞다.

## `_chamfer_score_at_scale()`

하나의 scale에서 edge-geometry matching을 수행한다.

동작:

1. 요청된 `scale`로 template edge map을 resize한다.
2. Template edge를 float mask로 만든다. Edge는 `1`, non-edge는 `0`.
3. `cv2.matchTemplate(frame_dt, template_mask, cv2.TM_CCORR)`로 mask를 frame
   distance transform 위에서 sliding한다.
4. 결과값은 각 위치에서 `sum(frame_distance_under_template_edges)`가 된다.
5. Edge count로 나누어 average edge distance를 구한다.
6. Average distance를 score로 변환한다.

```text
score = exp(-mean_dt / DT_TAU_PX)
```

반환값:

- `score`
- best center coordinate `(cx, cy)`
- scaled template size `(tw, th)`

왜 `TM_CCORR`인가:

Template mask가 edge 위치에서만 `1`이므로 correlation 결과는 template edge가 놓인
frame distance 값의 합이다. 그 합이 가장 작은 위치가 best geometry alignment다.

## `compute_chamfer_score()`

Multi-scale fallback을 수행한다.

```python
DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)
```

각 scale에서 `_chamfer_score_at_scale()`을 실행하고, 가장 높은 Chamfer score를
준 scale/location을 선택한다.

이 fallback은 `nm_per_pixel`이 없을 때 필요하다. Physical resolution metadata가
있으면 `compute_align_key_score()`에서 single scale을 계산해 쓰는 것이 더 좋다.

## `compute_orb_inlier_ratio()`

Feature-level verification을 수행한다. Chamfer가 찾은 candidate가 local feature
수준에서도 맞는지 확인한다.

동작:

1. Template image와 frame crop에서 ORB keypoint/descriptor를 검출한다.
2. `cv2.BFMatcher(cv2.NORM_HAMMING)`으로 binary descriptor를 match한다.
3. Lowe ratio test로 ambiguous match를 제거한다.

```python
if m.distance < LOWE_RATIO * n.distance:
    good.append(m)
```

4. `MIN_LOWE_MATCHES = 8`개 미만이면 score `0.0`을 반환한다.
5. `cv2.findHomography(..., cv2.RANSAC, 5.0)`로 geometry를 맞춘다.
6. RANSAC inlier ratio를 반환한다.

```text
inlier_ratio = number_of_RANSAC_inliers / number_of_good_matches
```

해석:

- 높음: local features가 하나의 geometry transform에 잘 맞는다.
- 낮음: false candidate, symmetry, noise, feature 부족 가능성이 높다.

## `compute_align_key_score()`

Frame마다 호출되는 main scoring function이다.

입력:

- `template`: preprocessed recipe align key.
- `frame`: current SEM frame.
- `frame_nm_per_pixel`: optional physical scale.
- `roi_hint`: optional `(x, y, w, h)` crop.

Scale 처리:

```text
if template.nm_per_pixel and frame_nm_per_pixel:
    scale = template.nm_per_pixel / frame_nm_per_pixel
else:
    scales = DEFAULT_SCALES
```

ROI 처리:

- `roi_hint`가 있으면 해당 crop 안에서만 matching한다.
- 결과 좌표는 다시 full-frame coordinate로 변환한다.

ORB 처리:

- ORB는 full frame 전체가 아니라 Chamfer best candidate 주변 crop에서만 실행한다.
- 이렇게 해야 속도가 빠르고 false feature가 줄어든다.

최종 score:

```text
score = 0.6 * chamfer_score + 0.4 * orb_inlier_ratio
```

Decision 기준:

```text
score >= 0.75 -> match
score >= 0.55 -> adjust
else          -> low
```

출력:

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

## 디버그 오버레이

`_render_overlay()`는 SEM frame 위에 다음 정보를 그린다.

- 후보 사각형
- 중심 십자 표시
- decision 문자열
- 최종 score
- Chamfer score
- ORB inlier ratio
- best scale

`save_overlay_jpeg()`는 overlay를 JPEG로 저장한다. SEM matching failure는 log
number만 보면 원인 파악이 어렵기 때문에, overlay JPEG는 중요한 debug artifact다.

## `search_align_key()`

Deterministic observe-score-act loop다.

주입 함수:

```python
capture_fn(state) -> np.ndarray
move_stage_fn(state, dx, dy) -> AlignKeySearchState
```

이 구조 덕분에 같은 search code를 두 환경에서 재사용할 수 있다.

- Mac/offline demo: synthetic wafer image와 mock stage coordinates.
- Office Windows/RCS: real monitor capture와 real stage movement.

Loop 동작:

1. 현재 FOV를 capture한다.
2. `compute_align_key_score()`로 score/decision을 계산한다.
3. Debug directory가 있으면 overlay를 저장한다.
4. `match`면 success로 종료한다.
5. `adjust`면 VLM hint가 있으면 사용하고, 없으면 small spiral nudge를 한다.
6. `low`면 `low_streak`를 증가시킨다.
7. Repeated low가 `low_streak_limit`에 도달하면 escalate한다.
8. 아니면 square-spiral step으로 다음 위치로 이동한다.

## 사각 spiral 탐색

`_square_spiral_step()`은 다음 pattern으로 stage delta를 만든다.

```text
start
  -> right
  -> up
  -> left, left
  -> down, down
  -> right, right, right
  -> ...
```

Target이 현재 FOV 근처에 있을 수 있지만 방향을 모를 때 유용하다. 가까운 곳부터
점점 넓게 훑는다.

## `test_align_key_match.py`

Synthetic smoke test다. 실제 SEM data가 없어도 algorithm plumbing과 basic
positive/negative separation을 검증한다.

생성하는 것:

- nested box와 비슷한 `box` template
- scale, rotation, brightness, contrast, charging-like gradient, noise가 들어간
  positive frames
- template 없음, random blobs, wrong pattern, strong charging-like gradient,
  out-of-scale template 같은 negative frames

통과 기준:

- Positive: `match` 또는 `adjust`이고 ground truth에서 20 px 이내.
- Negative: `low`.

주의:

이 test는 production validation이 아니다. 실제 SEM data로 threshold와 failure
mode를 반드시 다시 확인해야 한다.

## 현재 가정

- Align-key signal은 주로 edge geometry라고 가정한다.
- Rotation은 크지 않다고 가정한다.
- Metadata가 없으면 scale mismatch는 `0.7x` to `1.4x`만 본다.
- `0.75` / `0.55` threshold는 cold-start default다.
- ORB는 symmetric 또는 low-texture mark에서 실패할 수 있다.
- Synthetic test 통과는 production accuracy를 보장하지 않는다.

