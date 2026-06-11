# Tier 1.1 cond-box-crop Template Porting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the office-verified cond-box-crop template (box-interior crop + decoupled align offset) from the workflow_2 lab into the workflow_3 production correction path, default-ON, so reposition lands on the true align point (verified rank1 +0.16–0.18).

**Architecture:** A new `vision/cond_template.py` promotes the verified cond geometry primitives byte-identical from the lab (single source of truth; the lab re-imports them). `AlignKeyTemplate` gains an `align_offset_xy` field so each template carries its own offset. `build_templates_from_assets` becomes cond-aware (box-crop when a usable cond box exists, else center-area crop, else whole-image rollback). `correct_align_fail` applies `offset × best_scale` to the matched center before clamping. The ensemble matcher itself is unchanged — only the input template and the reposition arithmetic change.

**Tech Stack:** Python 3.10+, OpenCV (cv2), NumPy, the existing Chamfer/NCC ensemble matcher, `uv run` + `uv run pytest`. macOS-verifiable throughout (synthetic data only; no RCS, no office images).

**Dependency direction (CLAUDE.md):** workflow_2 (lab) → workflow_3.vision (prod). Never the reverse. The lab re-imports from `cond_template`; `cond_template` never imports workflow_2.

---

## Source-of-truth references (read before starting)

- Spec: `poc/workflow_3/docs/superpowers/specs/2026-06-11-box-crop-template-porting-design.md`
- Lab primitives being promoted (byte-identical): `poc/workflow_2/golden_localization_eval_cond.py`
  - `_cond_box_to_xywh` (lines 117–121), `_cond_box_center` (255–259), `cond_align_offset` (262–270), `cond_offset_norm` (273–278), `check_cond_box` (281–303), `cond_template_crop` (306–326), constants `CROP_INSET_PX`/`MIN_INNER_PX`/`WARN_INNER_PX`/`OFFSET_WARN`/`OFFSET_SKIP` (130–134).
- Reused primitives (do NOT duplicate): `poc/workflow_3/vision/clean_align_image.py` (`OVERSAMPLE`, `cursor_to_image`, `clean_image`), `poc/workflow_3/vision/cond_file.py` (`CondInfo`, `load_cond`), `poc/workflow_3/vision/align_point_correction.py` (`_centered_area_crop_bbox`, line 804).
- Prod targets: `poc/workflow_3/vision/align_key_matcher.py`, `poc/workflow_3/vision/align_fail_correct.py`, `poc/workflow_3/config.py`, `poc/workflow_3/monitor/cycle.py`.

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `poc/workflow_3/vision/cond_template.py` | **create** | Promoted cond geometry primitives (offset/crop/guard) + `centered_area_crop` fallback helper. Single source of truth. |
| `poc/workflow_3/vision/test_cond_template.py` | **create** | Pytest-style unit tests for the primitives + `build_template` offset preservation. |
| `poc/workflow_3/vision/align_key_matcher.py` | modify | Add `AlignKeyTemplate.align_offset_xy` field + `build_template(..., align_offset_xy=)` param. |
| `poc/workflow_2/golden_localization_eval_cond.py` | modify | Re-import the promoted primitives from `cond_template` (delete local defs) so the lab stays bit-parity; its existing test keeps passing. |
| `poc/workflow_3/vision/align_fail_correct.py` | modify | cond-aware `build_templates_from_assets`/`_load_template`; remove `extract_annotation_box`; apply offset in `correct_align_fail`; rename config field; wire `correct_align_fail_auto`. |
| `poc/workflow_3/vision/test_align_fail_correct.py` | modify | Add `_load_template` branch test + offset-application test to the existing `main()` runner. |
| `poc/workflow_3/config.py` | modify | Add `Workflow3Settings.cond_box_crop` + env override `ALIGN_FAIL_COND_BOX_CROP`. |
| `poc/workflow_3/monitor/cycle.py` | modify | Pass `cond_box_crop=settings.cond_box_crop` into `CorrectionConfig`. |

---

## Task 1: `cond_template.py` — promote cond geometry primitives

**Files:**
- Create: `poc/workflow_3/vision/cond_template.py`
- Test: `poc/workflow_3/vision/test_cond_template.py`

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/vision/test_cond_template.py`:

```python
"""cond_template primitive 합성 테스트 (Mac, 실데이터 불필요).

실행: uv run pytest poc/workflow_3/vision/test_cond_template.py -q
"""

import cv2
import numpy as np

from poc.workflow_3.vision.align_point_correction import _centered_area_crop_bbox
from poc.workflow_3.vision.cond_file import CondInfo
from poc.workflow_3.vision.cond_template import (
    CENTER_AREA_RATIO,
    CROP_INSET_PX,
    OFFSET_SKIP,
    centered_area_crop,
    check_cond_box,
    cond_align_offset,
    cond_offset_norm,
    cond_template_crop,
)


def _cond(box_ltrb, crosshair_xy=None):
    return CondInfo(scope="OM", pixel=(512, 512),
                    box_ltrb=box_ltrb, crosshair_xy=crosshair_xy)


def test_centered_box_has_zero_offset():
    # box 중심 (256,256) == image center → offset (0,0). cursor ×10: (2060..3060).
    assert cond_align_offset((2060, 2060, 3060, 3060), (512, 512)) == (0, 0)


def test_offcenter_box_offset_is_image_center_minus_box_center():
    # box 중심 (200,256) → offset (256-200, 0) = (56, 0).
    assert cond_align_offset((1500, 2060, 2500, 3060), (512, 512)) == (56, 0)


def test_offset_norm_uses_image_diagonal():
    onorm = cond_offset_norm((1500, 2060, 2500, 3060), (512, 512))
    assert abs(onorm - 56.0 / float(np.hypot(512, 512))) < 1e-6


def test_check_ok_for_normal_centered_box():
    status, reason, onorm = check_cond_box((2060, 2060, 3060, 3060), (512, 512))
    assert (status, reason) == ("ok", "ok") and onorm == 0.0


def test_check_skip_for_tiny_box():
    status, reason, _ = check_cond_box((2510, 2510, 2610, 2610), (512, 512))
    assert (status, reason) == ("skip", "box:too_small")


def test_check_skip_for_out_of_bounds_box():
    status, reason, _ = check_cond_box((4800, 2060, 5600, 3060), (512, 512))
    assert (status, reason) == ("skip", "box:out_of_bounds")


def test_check_skip_for_far_offcenter_box():
    status, reason, onorm = check_cond_box((150, 150, 650, 650), (512, 512))
    assert (status, reason) == ("skip", "offset:too_far") and onorm > OFFSET_SKIP


def test_cond_template_crop_centered_and_inset():
    # 200px box → 대칭 inset 후 crop = (200-2*inset)변, crop 중심 == box 중심, stroke 제거.
    box_ltrb = (1560, 1560, 3560, 3560)  # px box (156,156)-(356,356) = 200px
    gray = np.full((512, 512), 110, dtype=np.uint8)
    cv2.rectangle(gray, (156, 156), (356, 356), 255, 1)
    crop, (x0, y0, w, h) = cond_template_crop(gray, _cond(box_ltrb))
    assert w == 200 - 2 * CROP_INSET_PX and h == 200 - 2 * CROP_INSET_PX
    assert abs((x0 + w / 2.0) - 256) <= 0.5 and abs((y0 + h / 2.0) - 256) <= 0.5
    assert int(crop.max()) < 200  # inpaint + 대칭 inset 로 밝은 stroke(255) 제거.


def test_centered_area_crop_matches_bbox_helper():
    gray = np.full((512, 512), 90, dtype=np.uint8)
    x, y, cw, ch = _centered_area_crop_bbox(gray, CENTER_AREA_RATIO)
    crop = centered_area_crop(gray, CENTER_AREA_RATIO)
    assert crop.shape == (ch, cw)
    assert cw < 512 and ch < 512  # 중심부 축소 crop.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/vision/test_cond_template.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.workflow_3.vision.cond_template'` (the module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/vision/cond_template.py` (geometry functions byte-identical to the lab; only the imports point at workflow_3 modules and `centered_area_crop` is new):

```python
"""cond.txt 기하 primitive — box-crop template + decoupled align offset (production).

office 검증(workflow_2 golden_localization_eval_cond, 2026-06-11): cond.box_ltrb 로 crop 한
box template(+분리된 offset)이 center-area crop 대비 모든 displacement bin 에서 localization
동반 상승(rank1 +0.16~0.18). 그 검증된 cond 기하 함수를 lab 에서 production 으로 byte-identical
승격한다([[project_align_cond_files_and_coords]], [[project_rcp_white_box_unique_area]]).

핵심 분리(decoupled offset): align point 는 *이미지 중심* 이지 box 중심이 아니다. box 는
유니크 영역 단서일 뿐이다. offset = image_center - box_center 를 crop 과 분리해 cond 기하로만
계산하고(검출 inner-crop 의 off-center 오염 제거), template 은 box stroke 를 inpaint 로 지운 뒤
box 내부를 *대칭* inset 해 만든다(crop 중심 == box 중심 → offset 과 일관).

좌표계: cond.txt cursor 좌표는 이미지 px 의 10배(OVERSAMPLE). 변환은 clean_align_image 의
cursor_to_image 를 재사용한다(중복 생성 금지). 의존 방향: lab → 이 모듈(prod), 역방향 금지.
"""

import numpy as np

from poc.workflow_3.vision.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.vision.cond_file import CondInfo

# box-없음 fallback 의 center-area crop 비율(검증된 center arm; lab CENTER_AREA_RATIO 동일 값).
CENTER_AREA_RATIO = 0.15

# --- cond box → template/offset 가드 상수 (lab byte-parity) ---
CROP_INSET_PX = 2       # inpaint 후 edge-smear 가 template 에 안 들어오게 하는 대칭 inset.
MIN_INNER_PX = 16       # 대칭 inset 후 box 내부 하한(미만이면 skip — 매칭 신호 불안정).
WARN_INNER_PX = 24      # 작은 box 경고 임계(skip 아님).
OFFSET_WARN = 0.25      # offset_norm(÷대각선) 경고 임계(box 가 중심에서 멂).
OFFSET_SKIP = 0.38      # offset_norm 하드 skip(=box≠center 가정 붕괴, 엔지니어 검토 필요).


def _cond_box_to_xywh(box_ltrb):
    """cond.box_ltrb(cursor frame, ×10) → 이미지 px (x, y, w, h)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (int(round(l)), int(round(t)), int(round(r - l)), int(round(b - t)))


def _cond_box_center(box_ltrb):
    """cond.box_ltrb → 이미지 px box 중심 (cx, cy) (정수 반올림 전 float)."""
    l, t = cursor_to_image(box_ltrb[:2], OVERSAMPLE)
    r, b = cursor_to_image(box_ltrb[2:], OVERSAMPLE)
    return (l + r) / 2.0, (t + b) / 2.0


def cond_align_offset(box_ltrb, shape_hw):
    """align point(이미지 중심) - box 중심. cond.txt 만으로 결정 → crop 과 분리(decoupled).

    crop 을 어떻게 잡든 align point 의 기하는 안 변한다. 이 분리가 원본의 결함 — 내용검출
    inner-crop 의 off-center 가 offset 을 오염시키던 경로 — 를 통째로 없앤다.
    """
    h, w = shape_hw[:2]
    bcx, bcy = _cond_box_center(box_ltrb)
    return (int(round(w / 2.0 - bcx)), int(round(h / 2.0 - bcy)))


def cond_offset_norm(box_ltrb, shape_hw):
    """|offset| 를 이미지 *대각선* 으로 정규화(crop 무관 척도)."""
    h, w = shape_hw[:2]
    dx, dy = cond_align_offset(box_ltrb, shape_hw)
    diag = float(np.hypot(w, h)) or 1.0
    return float(np.hypot(dx, dy) / diag)


def check_cond_box(box_ltrb, shape_hw):
    """cond box 가 template 으로 쓸만한지 가드. 반환 (status, reason, offset_norm).

    status: 'ok' | 'warn' | 'skip'. inner = min(box변) - 2·CROP_INSET_PX(대칭 inset 후).
    skip 우선순위: degenerate → out_of_bounds → too_small → offset_too_far.
    """
    h, w = shape_hw[:2]
    x, y, bw, bh = _cond_box_to_xywh(box_ltrb)
    onorm = cond_offset_norm(box_ltrb, shape_hw)
    if bw <= 0 or bh <= 0:
        return "skip", "box:degenerate", onorm
    if x < 0 or y < 0 or x + bw > w or y + bh > h:
        return "skip", "box:out_of_bounds", onorm
    inner = min(bw, bh) - 2 * CROP_INSET_PX
    if inner < MIN_INNER_PX:
        return "skip", "box:too_small", onorm
    if onorm > OFFSET_SKIP:
        return "skip", "offset:too_far", onorm
    if onorm > OFFSET_WARN:
        return "warn", "offset:far", onorm
    if inner < WARN_INNER_PX:
        return "warn", "box:small", onorm
    return "ok", "ok", onorm


def cond_template_crop(gray, cond, *, inset=CROP_INSET_PX):
    """cond box stroke 를 inpaint 로 지운 뒤 box 내부를 *대칭* inset 해 template crop.

    대칭 inset → crop 중심 == box 중심 → cond_align_offset 과 정확히 일관.
    inset 후 너무 작아지면 inset 을 생략(작은 box 보호). 반환 (crop, (x0,y0,w,h)).

    **box stroke 만 지운다.** rcp cond 에 crosshair 가 있어도 그건 box 내부를 가로지르는
    *실제 내용* 이므로 inpaint 하면 매칭 신호가 깎인다 → crosshair_xy=None 으로 마스킹해
    box 테두리만 제거한다(msr 프레임의 crosshair 제거는 별개 — 거기선 distractor 라 지움).
    """
    box_only = CondInfo(scope=cond.scope, pixel=cond.pixel,
                        box_ltrb=cond.box_ltrb, crosshair_xy=None)
    cleaned = clean_image(gray, box_only)        # 튜닝된 1/1/2 로 box stroke 만 제거.
    x, y, bw, bh = _cond_box_to_xywh(cond.box_ltrb)
    h, w = gray.shape[:2]
    x0, y0 = max(0, x + inset), max(0, y + inset)
    x1, y1 = min(w, x + bw - inset), min(h, y + bh - inset)
    if x1 - x0 < MIN_INNER_PX or y1 - y0 < MIN_INNER_PX:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
    return cleaned[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def centered_area_crop(gray, area_ratio=CENTER_AREA_RATIO):
    """이미지 중심 기준 *면적 비율* crop (검증된 box-없음 fallback; offset 은 호출부에서 (0,0)).

    각 변 = sqrt(area_ratio) 비율(aspect 유지), 변 길이 하한 32 px. align point 가 이미지
    중심이라는 사전지식을 살려 매칭을 중심부에 집중시킨다. align_point_correction 의
    _centered_area_crop_bbox 와 동일 기하를 재사용한다(중복 방지; lazy import 로 로드 순서 무관).
    """
    from poc.workflow_3.vision.align_point_correction import _centered_area_crop_bbox

    x, y, cw, ch = _centered_area_crop_bbox(gray, area_ratio)
    return gray[y:y + ch, x:x + cw].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/vision/test_cond_template.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/vision/cond_template.py poc/workflow_3/vision/test_cond_template.py
git commit -m "feat(workflow_3): promote cond-box-crop geometry primitives to production

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: `AlignKeyTemplate.align_offset_xy` field + `build_template` param

**Files:**
- Modify: `poc/workflow_3/vision/align_key_matcher.py:66-78` (dataclass), `:206-226` (`build_template`)
- Test: `poc/workflow_3/vision/test_cond_template.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `poc/workflow_3/vision/test_cond_template.py` (add `build_template` to the imports block at the top, then add the two tests):

Add to the import block near the top of the file:

```python
from poc.workflow_3.vision.align_key_matcher import build_template
```

Append these test functions at the end of the file:

```python
def test_build_template_carries_align_offset():
    gray = np.full((64, 64), 120, dtype=np.uint8)
    tpl = build_template(gray, recipe_id="R", version="v0", key_type="om",
                         align_offset_xy=(5, -7))
    assert tpl.align_offset_xy == (5, -7)


def test_build_template_defaults_zero_offset():
    gray = np.full((64, 64), 120, dtype=np.uint8)
    tpl = build_template(gray, recipe_id="R", version="v0", key_type="om")
    assert tpl.align_offset_xy == (0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/vision/test_cond_template.py -q -k build_template`
Expected: FAIL — `TypeError: build_template() got an unexpected keyword argument 'align_offset_xy'`.

- [ ] **Step 3: Write minimal implementation**

In `poc/workflow_3/vision/align_key_matcher.py`, add the field to `AlignKeyTemplate`. Change:

```python
    nm_per_pixel: float | None
    key_type: str | None
    fetched_at: datetime
```

to:

```python
    nm_per_pixel: float | None
    key_type: str | None
    fetched_at: datetime
    align_offset_xy: tuple[int, int] = (0, 0)  # rcp px (image_center - box_center); reposition 시 best_scale 환산해 match 중심에 가산.
```

Then update `build_template` (lines 206–226). Change the signature:

```python
def build_template(
    raw_image: np.ndarray,
    *,
    recipe_id: str,
    version: str,
    nm_per_pixel: float | None = None,
    key_type: str | None = None,
) -> AlignKeyTemplate:
```

to:

```python
def build_template(
    raw_image: np.ndarray,
    *,
    recipe_id: str,
    version: str,
    nm_per_pixel: float | None = None,
    key_type: str | None = None,
    align_offset_xy: tuple[int, int] = (0, 0),
) -> AlignKeyTemplate:
```

And in its body, change the constructor:

```python
    return AlignKeyTemplate(
        recipe_id=recipe_id,
        version=version,
        raw_image=gray,
        edge_map=edges,
        distance_transform=dt,
        nm_per_pixel=nm_per_pixel,
        key_type=key_type,
        fetched_at=datetime.now(),
    )
```

to:

```python
    return AlignKeyTemplate(
        recipe_id=recipe_id,
        version=version,
        raw_image=gray,
        edge_map=edges,
        distance_transform=dt,
        nm_per_pixel=nm_per_pixel,
        key_type=key_type,
        fetched_at=datetime.now(),
        align_offset_xy=align_offset_xy,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/vision/test_cond_template.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Verify no matcher regressions**

The field has a default, so all existing `build_template(...)` callers stay valid. Confirm the matcher smoke test still passes:

Run: `uv run python poc/workflow_3/vision/test_align_key_match.py`
Expected: final line `10/10` (or the file's documented all-pass summary).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/vision/align_key_matcher.py poc/workflow_3/vision/test_cond_template.py
git commit -m "feat(workflow_3): AlignKeyTemplate carries align_offset_xy

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Re-point the lab eval to the promoted primitives (bit-parity, single source)

**Why:** The promoted functions must have exactly one definition. The lab keeps working by importing them from `cond_template`; its existing pure-synthetic test (`poc/workflow_2/test_golden_localization_eval_cond.py`, accesses `glec.cond_align_offset` etc.) then verifies the re-export wiring.

**Files:**
- Modify: `poc/workflow_2/golden_localization_eval_cond.py`

- [ ] **Step 1: Run the lab test now to capture the green baseline**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -q`
Expected: PASS (all green) — this is the contract Task 3 must preserve.

- [ ] **Step 2: Add the re-import**

In `poc/workflow_2/golden_localization_eval_cond.py`, add this import next to the other `poc.workflow_3.vision` imports (just after the existing `from poc.workflow_3.vision.clean_align_image import ...` line, ~line 58):

```python
from poc.workflow_3.vision.cond_template import (
    CROP_INSET_PX,
    MIN_INNER_PX,
    OFFSET_SKIP,
    OFFSET_WARN,
    WARN_INNER_PX,
    _cond_box_center,
    _cond_box_to_xywh,
    check_cond_box,
    cond_align_offset,
    cond_offset_norm,
    cond_template_crop,
)
```

- [ ] **Step 3: Delete the now-duplicated local definitions**

Remove these from `poc/workflow_2/golden_localization_eval_cond.py` (they now live in `cond_template`):

1. The five constant assignments in the `CROP_INSET_PX = 2 ... OFFSET_SKIP = 0.38` block (lines 130–134). Keep the surrounding explanatory comment lines (124–129) — only delete the five `NAME = value` assignment lines.
2. The function `def _cond_box_to_xywh(box_ltrb):` (lines 117–121).
3. The function `def _cond_box_center(box_ltrb):` (lines 255–259).
4. The function `def cond_align_offset(box_ltrb, shape_hw):` (lines 262–270).
5. The function `def cond_offset_norm(box_ltrb, shape_hw):` (lines 273–278).
6. The function `def check_cond_box(box_ltrb, shape_hw):` (lines 281–303).
7. The function `def cond_template_crop(gray, cond, *, inset=CROP_INSET_PX):` (lines 306–326).

Leave everything else (e.g. `_offset_diag_cond`, `_build_offset_templates_cond`, `displacement_bin`, the `DISP_BINS`/`RESCUE_MULT` constants) untouched — their internal calls to `_cond_box_to_xywh`, `check_cond_box`, `cond_template_crop`, `cond_align_offset`, `OFFSET_WARN`, etc. now resolve to the imported names.

- [ ] **Step 4: Verify the lab test still passes (re-export works)**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -q`
Expected: PASS (same count as Step 1). This proves `glec.cond_align_offset` / `glec.check_cond_box` / `glec.cond_template_crop` / `glec.OFFSET_SKIP` resolve to the production primitives.

- [ ] **Step 5: Verify the lab module still imports + runs to its data gate**

Run: `uv run python poc/workflow_2/golden_localization_eval_cond.py`
Expected: it imports cleanly (no `NameError`/`ImportError`) and prints `[ERROR] golden 데이터를 찾지 못했습니다: ...` then exits non-zero (no golden data on Mac — that is the expected no-data path, not an import failure).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/golden_localization_eval_cond.py
git commit -m "refactor(workflow_2): lab re-imports cond primitives from production cond_template

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: cond-aware template build + config rename (replace CV white-box detection)

**Why:** This swaps the uncalibrated `extract_annotation_box` (CV white-box detection, offset missing) for the verified cond path. Config field `crop_template_to_box: bool = False` becomes `cond_box_crop: bool = True` (default ON; OFF rolls back to whole-template).

**Files:**
- Modify: `poc/workflow_3/vision/align_fail_correct.py` (imports; `CorrectionConfig` field ~line 81; remove `extract_annotation_box` ~187-212; rewrite `_load_template` ~215-226; `build_templates_from_assets` ~229-245; `correct_align_fail_auto` build call ~509)
- Test: `poc/workflow_3/vision/test_align_fail_correct.py` (append `test_load_template_branches` + register in `main()`)

- [ ] **Step 1: Write the failing test**

In `poc/workflow_3/vision/test_align_fail_correct.py`, add these imports near the top import block:

```python
import tempfile
from pathlib import Path

import cv2

from poc.workflow_3.vision.align_fail_correct import _load_template
from poc.workflow_3.vision.cond_template import cond_align_offset
```

Add this test function (before `def main()`):

```python
def test_load_template_branches() -> bool:
    """_load_template 3분기: cond box → box-crop+offset, cond 없음 → center-crop+offset(0), flag off → whole."""
    gray = np.full((512, 512), 110, dtype=np.uint8)
    cv2.rectangle(gray, (150, 200), (250, 300), 255, 1)  # box px (150,200)-(250,300).

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        img = root / "IMAP0001.png"
        cv2.imwrite(str(img), gray)
        # cond sidecar: .<name>/cond.txt; box idx6..9 = cursor(px×10), crosshair idx4,5 = -1.
        cond_dir = root / ".IMAP0001.png"
        cond_dir.mkdir()
        (cond_dir / "cond.txt").write_text(
            "Scope\tOM\nPixel\t512,512\n!Cursor_info\t0,0,0,0,-1,-1,1500,2000,2500,3000\n",
            encoding="utf-8",
        )
        box_ltrb = (1500, 2000, 2500, 3000)
        exp_offset = cond_align_offset(box_ltrb, gray.shape)  # off-center → 비-(0,0).

        # 1) cond_box_crop=True + cond 존재 → box-crop + decoupled offset.
        t_box = _load_template(img, recipe_id="R", key_type="om", cond_box_crop=True)
        box_ok = (
            t_box.align_offset_xy == exp_offset
            and exp_offset != (0, 0)
            and t_box.raw_image.shape != (512, 512)
        )

        # 2) cond_box_crop=True + cond 없음(sidecar 미생성) → center-area crop + offset(0).
        img2 = root / "IMAP0002.png"
        cv2.imwrite(str(img2), gray)
        t_center = _load_template(img2, recipe_id="R", key_type="sem", cond_box_crop=True)
        center_ok = (
            t_center.align_offset_xy == (0, 0)
            and t_center.raw_image.shape != (512, 512)
        )

        # 3) cond_box_crop=False → whole-template(구 동작) + offset(0).
        t_whole = _load_template(img, recipe_id="R", key_type="om", cond_box_crop=False)
        whole_ok = (
            t_whole.align_offset_xy == (0, 0)
            and t_whole.raw_image.shape == (512, 512)
        )

    ok = box_ok and center_ok and whole_ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] load_template_branches: "
        f"box(off={t_box.align_offset_xy},shape={t_box.raw_image.shape}) "
        f"center(off={t_center.align_offset_xy},shape={t_center.raw_image.shape}) "
        f"whole(shape={t_whole.raw_image.shape})"
    )
    return ok
```

Register it in `main()` by adding `test_load_template_branches(),` to the `results = [...]` list (after `test_engineer_review_route(),`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/vision/test_align_fail_correct.py`
Expected: FAIL — `_load_template` currently takes `crop_to_box`, not `cond_box_crop`, so it raises `TypeError: _load_template() got an unexpected keyword argument 'cond_box_crop'` (and the run aborts before the summary).

- [ ] **Step 3: Update imports in `align_fail_correct.py`**

Add these imports after the existing `from poc.workflow_3.vision.align_fail_assets import ...` line (top of the file):

```python
from poc.workflow_3.vision.cond_file import load_cond
from poc.workflow_3.vision.cond_template import (
    CENTER_AREA_RATIO,
    centered_area_crop,
    check_cond_box,
    cond_align_offset,
    cond_template_crop,
)
```

- [ ] **Step 4: Rename the `CorrectionConfig` field**

Change (line ~81):

```python
    crop_template_to_box: bool = False  # 등록 이미지의 엔지니어 박스 내부로 template crop(미보정, 기본 off).
```

to:

```python
    cond_box_crop: bool = True  # cond.box_ltrb 기반 box-crop template(+decoupled offset). False → whole-template(구 동작) 롤백.
```

- [ ] **Step 5: Remove `extract_annotation_box` and rewrite `_load_template`**

Delete the entire `extract_annotation_box` function (the def + docstring + lazy import + body, lines ~187–212).

Replace the whole `_load_template` function (lines ~215–226) with:

```python
def _load_template(
    path: Path, *, recipe_id: str, key_type: str, cond_box_crop: bool
) -> AlignKeyTemplate:
    """등록 이미지를 cond-aware 하게 crop 해 AlignKeyTemplate 으로 만든다(offset 동봉).

    cond_box_crop=True(기본):
      - cond.box_ltrb 가 있고 check_cond_box 가 skip 이 아니면 → box 내부 crop
        (stroke inpaint + 대칭 inset) + offset = cond_align_offset(검증된 win 경로).
      - cond 부재/경계밖/너무작음/offset 과도(skip) → center-area crop + offset (0,0)(검증된 fallback).
    cond_box_crop=False: 전체 이미지(구 whole-template 동작 롤백) + offset (0,0).
    """
    gray = load_gray(path)
    if not cond_box_crop:
        crop, offset = gray, (0, 0)
    else:
        cond = load_cond(path)
        if (
            cond is not None
            and cond.box_ltrb is not None
            and check_cond_box(cond.box_ltrb, gray.shape)[0] != "skip"
        ):
            crop, _bbox = cond_template_crop(gray, cond)
            offset = cond_align_offset(cond.box_ltrb, gray.shape)
            print(f"[INFO] {key_type} template cond box-crop: offset={offset}")
        else:
            crop = centered_area_crop(gray, CENTER_AREA_RATIO)
            offset = (0, 0)
            print(f"[INFO] {key_type} template center-area crop (cond box 없음/skip)")
    return build_template(
        crop, recipe_id=recipe_id, version="v0", key_type=key_type,
        align_offset_xy=offset,
    )
```

- [ ] **Step 6: Update `build_templates_from_assets`**

Replace (lines ~229–245):

```python
def build_templates_from_assets(
    assets: AlignFailAssets, *, crop_to_box: bool = False
) -> dict[str, AlignKeyTemplate]:
    """recipe_om/recipe_sem → {'OM': ..., 'SEM': ...}. live_align_search 와 동일 계약.

    존재하는 등록 이미지만 담는다(부분 생성 허용). 둘 다 없으면 빈 dict.
    """
    templates: dict[str, AlignKeyTemplate] = {}
    if assets.recipe_om is not None:
        templates["OM"] = _load_template(
            assets.recipe_om, recipe_id=assets.recipe_id, key_type="om", crop_to_box=crop_to_box
        )
    if assets.recipe_sem is not None:
        templates["SEM"] = _load_template(
            assets.recipe_sem, recipe_id=assets.recipe_id, key_type="sem", crop_to_box=crop_to_box
        )
    return templates
```

with:

```python
def build_templates_from_assets(
    assets: AlignFailAssets, *, cond_box_crop: bool = True
) -> dict[str, AlignKeyTemplate]:
    """recipe_om/recipe_sem → {'OM': ..., 'SEM': ...}. live_align_search 와 동일 계약.

    존재하는 등록 이미지만 담는다(부분 생성 허용). 둘 다 없으면 빈 dict.
    cond_box_crop 은 _load_template 으로 전달된다(box-crop vs center vs whole).
    """
    templates: dict[str, AlignKeyTemplate] = {}
    if assets.recipe_om is not None:
        templates["OM"] = _load_template(
            assets.recipe_om, recipe_id=assets.recipe_id, key_type="om",
            cond_box_crop=cond_box_crop,
        )
    if assets.recipe_sem is not None:
        templates["SEM"] = _load_template(
            assets.recipe_sem, recipe_id=assets.recipe_id, key_type="sem",
            cond_box_crop=cond_box_crop,
        )
    return templates
```

- [ ] **Step 7: Update `correct_align_fail_auto` build call**

Change (line ~509):

```python
    templates = build_templates_from_assets(assets, crop_to_box=config.crop_template_to_box)
```

to:

```python
    templates = build_templates_from_assets(assets, cond_box_crop=config.cond_box_crop)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run python poc/workflow_3/vision/test_align_fail_correct.py`
Expected: the summary line shows all cases passed (the count is the prior total + 1), including `[PASS] load_template_branches: ...`.

- [ ] **Step 9: Confirm no dangling references to the removed symbols**

Run: `grep -rn "extract_annotation_box\|crop_template_to_box\|crop_to_box" poc/workflow_3/ | grep -v docs/`
Expected: no output (all production references removed; only spec/doc mentions remain elsewhere).

- [ ] **Step 10: Commit**

```bash
git add poc/workflow_3/vision/align_fail_correct.py poc/workflow_3/vision/test_align_fail_correct.py
git commit -m "feat(workflow_3): cond-aware template build replaces CV white-box detection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Apply the decoupled offset in `correct_align_fail` reposition

**Why:** The matched center (`best_xy`) is the box location; the true align point is `best_xy + offset × best_scale`. This is the headline behavior change (correct reposition target).

**Files:**
- Modify: `poc/workflow_3/vision/align_fail_correct.py` (reposition line ~382)
- Test: `poc/workflow_3/vision/test_align_fail_correct.py` (append `test_offset_applied_to_reposition` + register in `main()`)

- [ ] **Step 1: Write the failing test**

In `poc/workflow_3/vision/test_align_fail_correct.py`, add to the import block:

```python
from poc.workflow_3.vision.align_key_matcher import build_template
from poc.workflow_3.vision.live_align_search import clamp_to_fov
```

(`AlignKeyMatchResult` and `compute_align_key_score_ensemble` are already imported; `_FakeController`, `correct_align_fail`, `CorrectionConfig` are already in this file.)

Add this test function (before `def main()`):

```python
def test_offset_applied_to_reposition() -> bool:
    """reposition 타깃 == clamp(best_xy + round(offset × best_scale)). scale=1·scale≠1·offset0."""
    import poc.workflow_3.vision.align_fail_correct as afc

    frame = np.zeros((600, 800, 3), dtype=np.uint8)  # fw=800, fh=600.
    screen = np.zeros((600, 800), dtype=np.uint8)
    margin = CorrectionConfig().click_margin_ratio

    def _controlled(best_xy, best_scale):
        return AlignKeyMatchResult(
            score=0.9, chamfer_score=0.9, orb_inlier_ratio=0.0,
            best_xy=best_xy, best_scale=best_scale, decision="match",
            debug_overlay=np.zeros((4, 4, 3), dtype=np.uint8), distinctive=True,
        )

    def _run(offset, best_xy, best_scale):
        tpl = build_template(np.full((32, 32), 120, dtype=np.uint8),
                             recipe_id="R", version="v0", key_type="sem",
                             align_offset_xy=offset)
        fake = _FakeController(frame, screen, mode="SEM")
        orig = afc.compute_align_key_score_ensemble
        afc.compute_align_key_score_ensemble = lambda *a, **k: _controlled(best_xy, best_scale)
        try:
            correct_align_fail(fake, {"SEM": tpl},
                               ok_locator=lambda _s: (10, 10), dry_run=False)
        finally:
            afc.compute_align_key_score_ensemble = orig
        return fake.move_calls

    moves1 = _run((40, -30), (400, 300), 1.0)   # (400,300)+(40,-30) = (440,270).
    exp1 = clamp_to_fov(440, 270, 800, 600, margin)
    moves2 = _run((40, -30), (400, 300), 2.0)   # +round((40,-30)*2) = (480,240).
    exp2 = clamp_to_fov(480, 240, 800, 600, margin)
    moves0 = _run((0, 0), (400, 300), 1.0)      # offset0 → best_xy 그대로(회귀 가드).
    exp0 = clamp_to_fov(400, 300, 800, 600, margin)

    ok = moves1 == [exp1] and moves2 == [exp2] and moves0 == [exp0]
    print(
        f"[{'PASS' if ok else 'FAIL'}] offset_applied: "
        f"scale1={moves1}(exp{exp1}) scale2={moves2}(exp{exp2}) zero={moves0}(exp{exp0})"
    )
    return ok
```

Register it in `main()` by adding `test_offset_applied_to_reposition(),` to the `results = [...]` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/vision/test_align_fail_correct.py`
Expected: FAIL on the new case — `offset_applied` prints `scale1=[(400, 300)](exp(440, 270))` because the current code repositions to `best_xy` directly (offset not applied yet).

- [ ] **Step 3: Apply the offset in `correct_align_fail`**

In `poc/workflow_3/vision/align_fail_correct.py`, find the reposition line (~382):

```python
    # ---- PRIMARY: crosshair 를 best_xy 로 reposition. ----
    cx, cy = clamp_to_fov(result.best_xy[0], result.best_xy[1], fw, fh, config.click_margin_ratio)
```

Replace with:

```python
    # ---- PRIMARY: crosshair 를 align point 로 reposition. ----
    # template 이 들고 다니는 align offset(=rcp px image_center - box_center)을 best_scale 로
    # 환산해 match 중심에 더한다 → frame 의 진짜 align point. offset (0,0)이면 best_xy 그대로.
    ox, oy = template.align_offset_xy
    align_x = result.best_xy[0] + round(ox * result.best_scale)
    align_y = result.best_xy[1] + round(oy * result.best_scale)
    cx, cy = clamp_to_fov(align_x, align_y, fw, fh, config.click_margin_ratio)
```

(`template` is the routed template already bound at `template = route_template(templates, mode)`; the visibility gate and Tier 0 `second_ratio` routing read `result` and are unaffected.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python poc/workflow_3/vision/test_align_fail_correct.py`
Expected: all cases pass, including `[PASS] offset_applied: ...`. The pre-existing `test_primary_path` (which uses an offset-free demo template, `align_offset_xy=(0,0)`) still passes — offset (0,0) leaves `best_xy` unchanged.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/vision/align_fail_correct.py poc/workflow_3/vision/test_align_fail_correct.py
git commit -m "feat(workflow_3): apply decoupled align offset (offset*best_scale) at reposition

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Operational kill-switch — `ALIGN_FAIL_COND_BOX_CROP` env + cycle wiring

**Why:** cond-box-crop is default-ON in the production loop. A behavior-changing default needs an env override so the office can roll back to whole-template without a code edit.

**Files:**
- Modify: `poc/workflow_3/config.py` (`Workflow3Settings` field ~line 99; loader ~line 148)
- Modify: `poc/workflow_3/monitor/cycle.py` (`CorrectionConfig(...)` ~line 374)

- [ ] **Step 1: Add the setting field**

In `poc/workflow_3/config.py`, after the `reregister_second_ratio_threshold: float = 0.98` field (line ~99), add:

```python

    # --- cond box-crop template (Tier 1.1) ---
    # True(기본): cond.box_ltrb 로 box-crop template + decoupled offset(office 검증 rank1 +0.16~0.18).
    # False: whole-template(구 동작) 롤백 — env ALIGN_FAIL_COND_BOX_CROP=0.
    cond_box_crop: bool = True
```

- [ ] **Step 2: Wire the env override in the loader**

In `load_workflow3_settings()`, add to the `Workflow3Settings(...)` constructor call (next to `reregister_second_ratio_threshold=...`, line ~148):

```python
        cond_box_crop=env_flag("ALIGN_FAIL_COND_BOX_CROP", default=True),
```

- [ ] **Step 3: Pass the setting into `CorrectionConfig` in the cycle**

In `poc/workflow_3/monitor/cycle.py`, find the `CorrectionConfig(...)` call (~line 374):

```python
            config=CorrectionConfig(
                # 만성 모호 키 게이트(Tier 0.1) 활성화 — present 하나 second_ratio>tau 면
                # 자동 reposition+OK 대신 engineer_review 로 보류한다. notify 임계와 동일 값.
                reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
            ),
```

Replace with:

```python
            config=CorrectionConfig(
                # 만성 모호 키 게이트(Tier 0.1) 활성화 — present 하나 second_ratio>tau 면
                # 자동 reposition+OK 대신 engineer_review 로 보류한다. notify 임계와 동일 값.
                reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
                # cond box-crop template(Tier 1.1; env ALIGN_FAIL_COND_BOX_CROP 로 롤백 가능).
                cond_box_crop=settings.cond_box_crop,
            ),
```

- [ ] **Step 4: Verify the default + env override**

Run (default ON):
```bash
uv run python -c "from poc.workflow_3.config import load_workflow3_settings as L; assert L().cond_box_crop is True; print('default ON ok')"
```
Expected: `default ON ok`.

Run (env rollback):
```bash
ALIGN_FAIL_COND_BOX_CROP=0 uv run python -c "from poc.workflow_3.config import load_workflow3_settings as L; assert L().cond_box_crop is False; print('env OFF ok')"
```
Expected: `env OFF ok`.

- [ ] **Step 5: Verify the cycle module imports cleanly**

Run: `uv run python -c "import poc.workflow_3.monitor.cycle; print('cycle import ok')"`
Expected: `cycle import ok` (no `TypeError` from the new `CorrectionConfig` kwarg).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/monitor/cycle.py
git commit -m "feat(workflow_3): ALIGN_FAIL_COND_BOX_CROP env kill-switch + cycle wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Full-suite verification

**Why:** Confirm the whole change set is green before declaring done.

**Files:** none (verification only)

- [ ] **Step 1: Run the new + touched vision tests**

```bash
uv run pytest poc/workflow_3/vision/test_cond_template.py -q
uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -q
uv run python poc/workflow_3/vision/test_align_fail_correct.py
uv run python poc/workflow_3/vision/test_align_key_match.py
```
Expected: every command reports all-pass (pytest: `N passed`; script tests: their documented all-pass summary, e.g. `6/6`/`10/10` plus the two new correction cases).

- [ ] **Step 2: Run the correction demo (end-to-end primary path, dry-run)**

Run: `uv run python poc/workflow_3/vision/align_fail_correct.py`
Expected: prints `status=corrected path=primary ...` and exits 0 (the demo template has `align_offset_xy=(0,0)`, so the offset change is a no-op for it; this confirms no regression in the orchestration path).

- [ ] **Step 3: Final commit (if any uncommitted state remains)**

```bash
git status
# If clean, nothing to do. Otherwise stage and commit any stragglers.
```

---

## Out of scope / follow-up (non-blocking — do NOT implement here)

These come from spec §7 and are deliberately excluded:
- **paired-lift refinement** — recomputing center-vs-box on box-having frames only (tightens the estimate; the absolute box-arm numbers already justify porting).
- **gate-threshold recalibration for box templates** — the ensemble decision thresholds (0.6053/0.4727) were calibrated on non-box distributions (Tier 0.2 work). Localization is already box-verified, so this does not block.
- **veryfar displacement** — zero samples in the golden set; extreme-displacement effect unobserved (not a failure).
- **office e2e** (paused frame → box template → align point landing) — belongs to the real-equipment validation stage, not this Mac-side port.

---

## Self-review notes (completed by plan author)

- **Spec coverage:** §2.1 → Task 1; §2.2 → Task 2; §3 → Task 4; §4 → Task 5; §5 → Tasks 4 (field rename) + 6 (env override); §6 tests → Tasks 1/2/4/5; lab re-import (§2.1) → Task 3. §7 items explicitly deferred above.
- **Type consistency:** `align_offset_xy: tuple[int, int]` is identical in the dataclass field, `build_template` param, `_load_template`/`build_templates_from_assets` plumbing, and the `template.align_offset_xy` read in `correct_align_fail`. Config field `cond_box_crop: bool` is consistent across `CorrectionConfig`, `Workflow3Settings`, the loader env flag, and the cycle wiring. `check_cond_box(...)[0] != "skip"` matches the lab's `(status, reason, offset_norm)` return shape.
- **No placeholders:** every code step shows complete bodies; every command states expected output.
