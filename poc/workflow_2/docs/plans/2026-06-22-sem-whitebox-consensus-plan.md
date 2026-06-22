# SEM whitebox box-crop in consensus arm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a whitebox box-crop arm to the workflow_2 consensus recall eval and A/B it against the current center-crop arm, stratified OM vs SEM, to measure whether the engineer's whitebox recovers SEM `recall_miss`.

**Architecture:** Reuse the existing cond-box CV helpers (`cond_template_crop`/`cond_align_offset`, already production-ported) and the already-built-but-discarded `box` templates from `glec._build_offset_templates_cond`. Add (1) align-offset support to `_gt_in_topk` so a box template's match-center maps to the align point, (2) per-S-frame box crops in `_build_cond_by_recipe`, (3) a box arm in `_consensus_template_ab` with the fixed-denominator contract, (4) a `CONSENSUS_BOX_CROP` config knob + per-modality digest. Bench-only; production port gated on a positive SEM result.

**Tech Stack:** Python 3.10+, numpy, OpenCV (`cv2`), pytest. No new dependencies.

## Global Constraints

- Korean docstrings; `[INFO]/[ERROR]/[WARNING]` print-based logging (never the `logging` module).
- No `from __future__` imports. No `argparse`/CLI flags — config via `golden_eval_config.py` + env only.
- Absolute imports within `poc/`: `from poc.workflow_2.xxx import ...`. workflow_2 imports workflow_3, never the reverse.
- No em-dash (U+2014) inside `print()` strings (office console is cp949). Docstrings may use it.
- Commit directly to `main`, pathspec-only (concurrent sessions edit the same repo). Commit trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01ETb55kbRvcxc82tzqf3q6q`.
- Office data cannot come to Mac — all tests are synthetic/offline; real recall numbers come from an office run the user relays as `[DIGEST]`.
- Reuse, don't reimplement: box-crop/offset geometry must call `poc.workflow_3.align.cond_template` (bit-parity with production), never a local copy.

## File Structure

- Modify `poc/workflow_2/align_similarity.py` — `_gt_in_topk` (align-offset), `_consensus_template_ab` (box arm). Core matching + eval logic lives here.
- Modify `poc/workflow_2/golden_consensus_eval_cond.py` — `_build_cond_by_recipe` (box crops), `main` (config passthrough + digest).
- Modify `poc/workflow_2/golden_eval_config.example.py` — add `CONSENSUS_BOX_CROP`.
- Modify `poc/workflow_2/golden_eval_config_loader.py` — seed `CONSENSUS_BOX_CROP` into env.
- Create `poc/workflow_2/test_consensus_box_crop.py` — all tests for this feature.

---

### Task 1: `_gt_in_topk` align-offset support

**Files:**
- Modify: `poc/workflow_2/align_similarity.py` (`_gt_in_topk`, dist computation ~line 358)
- Test: `poc/workflow_2/test_consensus_box_crop.py`

**Interfaces:**
- Consumes: `compute_ensemble_candidates`/`compute_chamfer_candidates` candidates with `.xy` (template-center, frame px) and `.scale`; `AlignKeyTemplate.align_offset_xy: tuple[int,int]` (rcp px = image_center − box_center, engine.py:78).
- Produces: `_gt_in_topk` now maps each candidate to its **align point** = `candidate.xy + align_offset_xy × candidate.scale` before the truth-distance test. Behavior is **byte-identical when `align_offset_xy == (0,0)`** (all current callers pass center templates).

- [ ] **Step 1: Write the failing test**

```python
# poc/workflow_2/test_consensus_box_crop.py
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poc.workflow_2 import align_similarity as alsim
from poc.workflow_3.align.matching.engine import build_template


def _frame_with_marker(size=256, cx=128, cy=128, r=10):
    """단일 밝은 사각 마커가 있는 프레임 — 매칭이 유일하게 lock 되도록."""
    img = np.full((size, size), 30, np.uint8)
    img[cy - r:cy + r, cx - r:cx + r] = 220
    return img


def _marker_template(r=10):
    tpl_img = np.full((2 * r, 2 * r), 30, np.uint8)
    tpl_img[:] = 220
    return tpl_img


def test_gt_in_topk_applies_align_offset_to_reach_align_point():
    """offset 이 있는 box template: match 중심(box center)에 offset 을 더해야 truth(align point)와 일치."""
    # 마커(=box center)는 (128,128). align point(truth crosshair)는 offset 만큼 떨어진 (148,118).
    frame = _frame_with_marker(cx=128, cy=128, r=10)
    tpl = build_template(_marker_template(r=10), recipe_id="R", version="box", key_type="sem")
    tpl.align_offset_xy = (20, -10)   # align point = box center + (20,-10) = (148,118)
    truth_align_point = (148, 118)
    out = alsim._gt_in_topk(frame, truth_align_point, {"sem": tpl},
                            scales=(1.0,), topk=8)
    assert out is not None
    assert out["in_topk"] is True, "offset 적용 시 box-center match 가 align point 로 매핑돼 truth 히트해야 함"


def test_gt_in_topk_zero_offset_unchanged():
    """offset (0,0)(center template): align point == match center — 기존 동작 유지(회귀 가드)."""
    frame = _frame_with_marker(cx=128, cy=128, r=10)
    tpl = build_template(_marker_template(r=10), recipe_id="R", version="center", key_type="sem")
    # align_offset_xy 기본 (0,0). truth = marker center.
    out = alsim._gt_in_topk(frame, (128, 128), {"sem": tpl}, scales=(1.0,), topk=8)
    assert out is not None
    assert out["in_topk"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py -q`
Expected: `test_gt_in_topk_applies_align_offset_to_reach_align_point` FAILS (offset ignored → match center (128,128) is `GT_TOL_NORM`·short away from truth (148,118), `in_topk` False). The zero-offset test passes.

- [ ] **Step 3: Write minimal implementation**

In `align_similarity.py`, inside `_gt_in_topk`'s per-modality loop, read the template offset and apply it (scaled per candidate) in the distance computation. Replace the current dist line:

```python
        # 기존:
        # dists = [float(np.hypot(c.xy[0] - cxh, c.xy[1] - cyh)) / short for c in cands]
        ox, oy = getattr(tpl, "align_offset_xy", (0, 0)) or (0, 0)
        # box template: match 중심(box center) + offset×scale = align point. center tpl 은 offset (0,0) → 불변.
        dists = []
        for c in cands:
            sc = getattr(c, "scale", 1.0) or 1.0
            ax = c.xy[0] + ox * sc
            ay = c.xy[1] + oy * sc
            dists.append(float(np.hypot(ax - cxh, ay - cyh)) / short)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py -q`
Expected: both tests PASS.

- [ ] **Step 5: Run the existing suite to confirm no regression (offset=0 byte-identical)**

Run: `uv run pytest poc/workflow_2/ -q`
Expected: all pass (199 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/align_similarity.py poc/workflow_2/test_consensus_box_crop.py
git commit poc/workflow_2/align_similarity.py poc/workflow_2/test_consensus_box_crop.py -m "feat(workflow_2): _gt_in_topk applies align_offset (box-center -> align point)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ETb55kbRvcxc82tzqf3q6q"
```

---

### Task 2: Per-S-frame box crops in `_build_cond_by_recipe`

**Files:**
- Modify: `poc/workflow_2/golden_consensus_eval_cond.py` (`_build_cond_by_recipe`, `main` call site ~line 495-501)
- Test: `poc/workflow_2/test_consensus_box_crop.py`

**Interfaces:**
- Consumes: `glec._build_offset_templates_cond(assets) -> (center, box)` where `box[mod]` is `(box_tpl, offset)` or `None`; `_matched_crop(frame, center_xy, tw, th, scale)`; `clean_image(gray, cond)`; `_cond_crosshair_xy(cond)`.
- Produces: each `s_frames` entry gains `"crop_box"` (box-region crop sized to the box template, crosshair-removed) when a box template exists for that modality, else `None`. `entry` gains `"box_tpls": {mod: (box_tpl, offset) | None}`.

- [ ] **Step 1: Write the failing test**

```python
# add to poc/workflow_2/test_consensus_box_crop.py
from poc.workflow_2 import golden_consensus_eval_cond as gce


def test_box_crop_is_centered_on_box_region_offset_from_crosshair():
    """box crop = crosshair - offset 중심·box size. crosshair 중심 center crop 과 다른 영역."""
    # 합성: full frame 에 box marker 를 crosshair 에서 offset 만큼 떨어뜨려 둔다.
    size = 300
    gray = np.full((size, size), 30, np.uint8)
    crosshair = (200, 150)
    offset = (40, -20)               # offset = img_center - box_center → box center = crosshair - offset
    box_cx, box_cy = crosshair[0] - offset[0], crosshair[1] - offset[1]   # (160,170)
    gray[box_cy - 12:box_cy + 12, box_cx - 12:box_cx + 12] = 220          # box 영역 마커
    box_tpl = build_template(np.full((24, 24), 220, np.uint8),
                             recipe_id="R", version="box", key_type="sem")
    crop = gce._box_consensus_crop(gray, crosshair, offset, box_tpl)
    assert crop is not None
    assert crop.shape == (24, 24)
    # box 마커를 담았으면 평균이 밝다(잘못된 영역이면 어둡다).
    assert crop.mean() > 150, "box crop 이 box 영역(밝은 마커)을 담아야 함"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py::test_box_crop_is_centered_on_box_region_offset_from_crosshair -q`
Expected: FAIL with `AttributeError: module ... has no attribute '_box_consensus_crop'`.

- [ ] **Step 3: Write minimal implementation**

Add the helper near `_cond_consensus_crop` in `golden_consensus_eval_cond.py`:

```python
def _box_consensus_crop(gray, crosshair_xy, offset_xy, box_tpl):
    """box 영역(=crosshair - offset 중심) crop, box template 크기. consensus box arm 재료.

    offset = image_center - box_center 이고 S 프레임에선 align point(=image center) 가
    crosshair 에 오므로 box center = crosshair - offset. crosshair 는 distractor 라 호출부에서
    이미 정제된(clean) 프레임을 넘기거나, 여기선 raw 를 받아 box 영역만 자른다(box 영역엔
    보통 crosshair 가 없음). 없으면(OOB/너무작음) None.
    """
    bcx = crosshair_xy[0] - offset_xy[0]
    bcy = crosshair_xy[1] - offset_xy[1]
    bw = box_tpl.raw_image.shape[1]
    bh = box_tpl.raw_image.shape[0]
    return _matched_crop(gray, (bcx, bcy), bw, bh, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py::test_box_crop_is_centered_on_box_region_offset_from_crosshair -q`
Expected: PASS.

- [ ] **Step 5: Wire box crops + box_tpls into `_build_cond_by_recipe`**

Change the call site in `main` (~line 495) to keep `box`:

```python
        center_tpls, box_tpls = glec._build_offset_templates_cond(assets)
```
```python
        entry = _build_cond_by_recipe(assets, center_tpls, box_tpls)
```

Update `_build_cond_by_recipe` signature and body. Signature:

```python
def _build_cond_by_recipe(assets, center_tpls, box_tpls=None):
```

In `entry = {...}` add:

```python
        "box_tpls": {m: bt for m, bt in (box_tpls or {}).items()},   # {mod: (box_tpl, offset)|None}
```

After the existing `crop = _cond_consensus_crop(...)` / append block, compute the box crop for the frame (clean frame already removes crosshair+box; box region rarely holds the crosshair, so crop from the cleaned frame for parity with the center arm):

```python
        # box arm 재료: box template 이 있는 modality 만. clean 프레임에서 box 영역을 자른다.
        crop_box = None
        bt = (box_tpls or {}).get(mod)
        if bt is not None and xy is not None:
            _box_tpl, _off = bt
            cleaned = clean_image(gray, cond)
            crop_box = _box_consensus_crop(cleaned, xy, _off, _box_tpl)
        entry["s_frames"][-1]["crop_box"] = crop_box
```

(Place this immediately after `entry["s_frames"].append({...})` so `[-1]` is the frame just added. Co-register block below operates on `crop`; leave `crop_box` un-coregistered for now — Task 3 medians it directly.)

- [ ] **Step 6: Run the suite**

Run: `uv run pytest poc/workflow_2/ -q`
Expected: all pass (no behavior change to existing fields; new fields additive).

- [ ] **Step 7: Commit**

```bash
git commit poc/workflow_2/golden_consensus_eval_cond.py poc/workflow_2/test_consensus_box_crop.py -m "feat(workflow_2): box-crop S-frame material for consensus box arm

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ETb55kbRvcxc82tzqf3q6q"
```

---

### Task 3: Box arm in `_consensus_template_ab` (fixed-denominator A/B)

**Files:**
- Modify: `poc/workflow_2/align_similarity.py` (`_consensus_template_ab`)
- Test: `poc/workflow_2/test_consensus_box_crop.py`

**Interfaces:**
- Consumes: `by_recipe[rec]` entries with `s_frames[i]["crop_box"]` and `data["box_tpls"][mod] = (box_tpl, offset)|None` (Task 2); `_consensus(crops)`; `build_template(img, recipe_id, version, key_type, align_offset_xy=...)`; `_gt_in_topk` with align-offset (Task 1).
- Produces: when `box_crop=True`, return dict gains
  `res["box_crop_ab"]["per_modality"][mod][arm] = {"recall","rank1","n_eval","n_hit","n_no_candidate"}`
  for `arm in {"center","box"}`. Denominator `n_eval` = frames where the **center** arm produced a valid measurement (`gc is not None`); within it, box no-candidate/too-small/exception = miss (counts into `n_no_candidate`, not skipped). Off by default (`box_crop=False`) → return unchanged.

- [ ] **Step 1: Write the failing test**

```python
# add to poc/workflow_2/test_consensus_box_crop.py
from poc.workflow_3.align.matching.engine import AlignKeyTemplate


def _stack_crops(marker_val, size=40, n=4):
    crops = []
    for _ in range(n):
        c = np.full((size, size), 30, np.uint8)
        c[10:30, 10:30] = marker_val
        crops.append(c)
    return crops


def test_consensus_box_arm_reports_fixed_denominator_per_modality(monkeypatch, tmp_path):
    """box arm 이 켜지면 per-(modality, arm) recall + 고정 분모 카운트를 반환한다."""
    # _gt_in_topk 을 결정적 스텁으로: center 는 항상 hit, box 는 항상 no-candidate(None 후보 모사).
    calls = {"center": 0, "box": 0}

    def _fake_gt(gray, xy, tpls, *, scales=None, topk=None):
        (mod, tpl), = tpls.items()
        ver = getattr(tpl, "version", "")
        if "box" in ver:
            calls["box"] += 1
            return None              # box: 후보 0개 → miss(skip 아님)여야 함
        calls["center"] += 1
        return {"topk_rank": 1, "in_topk": True, "n_cand": 8,
                "best_cand_dist_norm": 0.0, "modality": mod, "cand_xys": [],
                "peak_ratio": 1.0, "cand_scores": [], "cand_ncc": None}

    monkeypatch.setattr(alsim, "_gt_in_topk", _fake_gt)

    # 한 recipe·SEM·4 S 프레임. center crop + box crop + box_tpls 제공.
    fm = [{"path": f"S{i}", "xy": (50, 50), "mod": "sem",
           "crop": _stack_crops(220)[0], "crop_box": _stack_crops(200)[0]} for i in range(4)]
    box_tpl = build_template(np.full((20, 20), 200, np.uint8),
                             recipe_id="R", version="sem_box", key_type="sem")
    by_recipe = {"R": {"s_frames": fm, "e_paths": [],
                       "rcp_tpls": {}, "box_tpls": {"sem": (box_tpl, (5, 5))},
                       "history_crops": {}}}

    res = alsim._consensus_template_ab(by_recipe, min_s=3, out_dir=str(tmp_path),
                                       box_crop=True,
                                       frame_loader=lambda f: np.full((100, 100), 30, np.uint8))
    ab = res["box_crop_ab"]["per_modality"]["sem"]
    assert ab["center"]["n_eval"] == ab["box"]["n_eval"], "두 arm 의 분모가 같아야(고정 분모)"
    assert ab["box"]["n_no_candidate"] == ab["box"]["n_eval"], "box 후보 0개는 miss 로 세야(분모 유지)"
    assert ab["box"]["recall"] == 0.0
    assert ab["center"]["recall"] == 1.0


def test_consensus_box_arm_off_by_default(tmp_path):
    """box_crop=False(기본) → box_crop_ab 없음(기존 동작 불변)."""
    fm = [{"path": f"S{i}", "xy": (50, 50), "mod": "sem",
           "crop": _stack_crops(220)[0]} for i in range(4)]
    by_recipe = {"R": {"s_frames": fm, "e_paths": [], "rcp_tpls": {}, "history_crops": {}}}
    res = alsim._consensus_template_ab(by_recipe, min_s=3, out_dir=str(tmp_path),
                                       frame_loader=lambda f: np.full((100, 100), 30, np.uint8))
    assert res is None or "box_crop_ab" not in (res or {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py -k box_arm -q`
Expected: `test_consensus_box_arm_reports_fixed_denominator_per_modality` FAILS (`_consensus_template_ab` has no `box_crop` kwarg → `TypeError`). `off_by_default` passes.

- [ ] **Step 3: Write minimal implementation**

Add `box_crop=False` to the `_consensus_template_ab` signature. Initialize a per-modality accumulator before the recipe loop:

```python
    # box arm A/B (whitebox box-crop vs center) — per (modality, arm) 고정 분모 카운트.
    box_ab = {}   # {mod: {arm: {"n_eval","n_hit","n_r1","n_no_cand"}}}

    def _box_cell(mod, arm):
        return box_ab.setdefault(mod, {}).setdefault(
            arm, {"n_eval": 0, "n_hit": 0, "n_r1": 0, "n_no_cand": 0})
```

Inside the existing per-frame loop, right after the center `gc = _gt_in_topk(...)` and its `if gc is None: continue` guard (so `gc is not None` defines the fixed denominator), add:

```python
            if box_crop:
                bt = (data.get("box_tpls") or {}).get(mod)
                cb = f.get("crop_box")
                # center arm cell (gc 유효 → 분모 +1).
                cc = _box_cell(mod, "center")
                cc["n_eval"] += 1
                if gc["in_topk"]:
                    cc["n_hit"] += 1
                if gc["topk_rank"] == 1:
                    cc["n_r1"] += 1
                # box arm cell — 같은 분모(+1). 재료 없거나 후보 0개면 miss.
                bx = _box_cell(mod, "box")
                bx["n_eval"] += 1
                gb = None
                if bt is not None and cb is not None:
                    others_box = [g.get("crop_box") for j, g in enumerate(fm)
                                  if j != i and g.get("crop_box") is not None] \
                        if not use_history else None
                    # history 경로: 모든 eval frame 공통 box consensus; LOO: held-out.
                    if use_history:
                        pool_box = [c for c in (
                            (data.get("history_crops_box") or {}).get(mod) or []) ]
                    else:
                        pool_box = others_box or []
                    if len(pool_box) >= 2:
                        box_tpl, off = bt
                        cons_box = build_template(_consensus(pool_box),
                                                  recipe_id=rec, version=f"s_consensus_box_{mod}",
                                                  key_type=mod, align_offset_xy=off)
                        gb = _gt_in_topk(gray, tuple(f["xy"]), {mod: cons_box})
                if gb is None:
                    bx["n_no_cand"] += 1
                else:
                    if gb["in_topk"]:
                        bx["n_hit"] += 1
                    if gb["topk_rank"] == 1:
                        bx["n_r1"] += 1
```

After the recipe loop, build the result section (guard `box_crop`):

```python
    if box_crop:
        per_mod = {}
        for mod, arms in box_ab.items():
            per_mod[mod] = {}
            for arm, c in arms.items():
                e = c["n_eval"] or 0
                per_mod[mod][arm] = {
                    "recall": round(c["n_hit"] / e, 4) if e else 0.0,
                    "rank1": round(c["n_r1"] / e, 4) if e else 0.0,
                    "n_eval": e, "n_hit": c["n_hit"], "n_no_candidate": c["n_no_cand"],
                }
        result["box_crop_ab"] = {"per_modality": per_mod}
```

(Use the function's existing return-dict variable name — locate the `return {...}`/`res = {...}` at the end of `_consensus_template_ab` and attach `box_crop_ab` to it before returning. If it returns a literal, assign it to a local first.)

Note: `history_crops_box` (history-pool box crops) may not exist yet; `(data.get("history_crops_box") or {})` yields `[]` → those frames count as `n_no_candidate` (honest miss) until a later task populates it. The LOO path (the common bench case for this experiment) is fully exercised.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py -k box_arm -q`
Expected: both PASS.

- [ ] **Step 5: Run the suite**

Run: `uv run pytest poc/workflow_2/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git commit poc/workflow_2/align_similarity.py poc/workflow_2/test_consensus_box_crop.py -m "feat(workflow_2): consensus box-crop arm (center vs whitebox, fixed denominator)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ETb55kbRvcxc82tzqf3q6q"
```

---

### Task 4: Config knob + per-modality digest

**Files:**
- Modify: `poc/workflow_2/golden_eval_config.example.py` (add `CONSENSUS_BOX_CROP`)
- Modify: `poc/workflow_2/golden_eval_config_loader.py` (seed into env)
- Modify: `poc/workflow_2/golden_consensus_eval_cond.py` (`main`: read flag, pass `box_crop`, print digest)
- Test: `poc/workflow_2/test_consensus_box_crop.py`

**Interfaces:**
- Consumes: `res["box_crop_ab"]["per_modality"]` (Task 3).
- Produces: pure formatter `gce._format_box_crop_digest(per_modality) -> list[str]` (one line per modality, `box−center` delta + counts). `main` reads `CONSENSUS_BOX_CROP` env (via loader) and passes `box_crop=` to `_consensus_template_ab`.

- [ ] **Step 1: Write the failing test**

```python
# add to poc/workflow_2/test_consensus_box_crop.py
def test_box_crop_digest_formats_per_modality_delta():
    per_mod = {
        "sem": {"center": {"recall": 0.71, "rank1": 0.71, "n_eval": 100, "n_hit": 71, "n_no_candidate": 0},
                "box": {"recall": 0.88, "rank1": 0.85, "n_eval": 100, "n_hit": 88, "n_no_candidate": 3}},
        "om": {"center": {"recall": 0.91, "rank1": 0.90, "n_eval": 50, "n_hit": 46, "n_no_candidate": 0},
               "box": {"recall": 0.90, "rank1": 0.89, "n_eval": 50, "n_hit": 45, "n_no_candidate": 1}},
    }
    lines = gce._format_box_crop_digest(per_mod)
    joined = "\n".join(lines)
    assert "sem" in joined.lower() and "om" in joined.lower()
    assert "+0.17" in joined or "+0.170" in joined, "SEM box-center delta(+0.17) 표기"
    assert "n_eval" in joined and "no_cand" in joined.replace("_candidate", "_cand")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py::test_box_crop_digest_formats_per_modality_delta -q`
Expected: FAIL — `_format_box_crop_digest` missing.

- [ ] **Step 3: Write minimal implementation**

Add to `golden_consensus_eval_cond.py`:

```python
def _format_box_crop_digest(per_modality):
    """per-modality center-vs-box recall digest 줄들. box-center delta + 고정분모 카운트.

    survivorship 자가점검: n_eval 동일 + n_no_candidate 노출(분모 축소 아님 확인).
    """
    lines = ["[DIGEST] box-crop A/B (consensus arm, per modality):"]
    for mod in sorted(per_modality):
        c = per_modality[mod].get("center", {})
        b = per_modality[mod].get("box", {})
        cr, br = c.get("recall", 0.0), b.get("recall", 0.0)
        delta = br - cr
        lines.append(
            f"  {mod}: center {cr:.3f} -> box {br:.3f} (delta {delta:+.3f}) "
            f"[n_eval {c.get('n_eval', 0)}/{b.get('n_eval', 0)}, "
            f"hit {c.get('n_hit', 0)}/{b.get('n_hit', 0)}, "
            f"box_no_cand {b.get('n_no_candidate', 0)}]")
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_consensus_box_crop.py::test_box_crop_digest_formats_per_modality_delta -q`
Expected: PASS.

- [ ] **Step 5: Add config knob + loader seeding + wiring**

In `golden_eval_config.example.py`, add near the other consensus knobs:

```python
# consensus arm whitebox box-crop A/B (center vs box, OM/SEM 층화). 1 이면 box arm 측정.
CONSENSUS_BOX_CROP = 0
```

In `golden_eval_config_loader.py` `seed_env()`, add (mirror the existing `setdefault`/`os.environ` pattern used for other constants):

```python
    _seed("CONSENSUS_BOX_CROP", getattr(_cfg, "CONSENSUS_BOX_CROP", 0))
```
(Use the file's existing `_seed`/setter helper name; if it sets `os.environ[...]` directly, follow that exact form.)

In `golden_consensus_eval_cond.py` near the other env reads (e.g. where `CLEAN_FRAME`/`COREGISTER` are read), add:

```python
CONSENSUS_BOX_CROP = os.getenv("CONSENSUS_BOX_CROP", "0") not in ("0", "", "false", "False")
```

Pass it into the call (~line 558):

```python
    res = _consensus_template_ab(
        by_recipe, min_s=CONSENSUS_MIN_S, out_dir=out_dir,
        combined_renderer=renderer, box_crop=CONSENSUS_BOX_CROP,
        frame_loader=_cleaned_frame_loader if CLEAN_FRAME else None)
```

After `res` is populated (and not None), print the digest:

```python
    if CONSENSUS_BOX_CROP and res and res.get("box_crop_ab"):
        for line in _format_box_crop_digest(res["box_crop_ab"]["per_modality"]):
            print(line)
```

- [ ] **Step 6: Run the suite + offline smoke**

Run: `uv run pytest poc/workflow_2/ -q`
Expected: all pass.

Run: `uv run python poc/workflow_2/golden_consensus_eval_cond.py`
Expected: exits cleanly with `[WARNING] ... no data` on Mac (no golden data) — confirms wiring imports/runs.

- [ ] **Step 7: Commit**

```bash
git commit poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config_loader.py poc/workflow_2/golden_consensus_eval_cond.py poc/workflow_2/test_consensus_box_crop.py -m "feat(workflow_2): CONSENSUS_BOX_CROP knob + per-modality box-crop digest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ETb55kbRvcxc82tzqf3q6q"
```

- [ ] **Step 8: Push**

```bash
git push origin main
```

---

## Office run (user-relayed)

```bash
# golden_eval_config.py 에서 CONSENSUS_BOX_CROP=1 설정 후:
uv run python poc/workflow_2/golden_consensus_eval_cond.py
```
Relay the `[DIGEST] box-crop A/B` lines. Decision: **SEM box recall ≫ center (OM ~flat)** → port box-crop into the production consensus path (`workflow_3/align` consensus templates/resolve). SEM flat / box ≤ center → pivot (re-registration quantification / lower-mag context).

## Notes on scope

- The comparative recall (box vs center) is the **office deliverable**, not a unit test — synthetic ensemble behavior is not a faithful proxy for periodic SEM keys (spec §6 risk). Unit tests pin **correctness** (offset mapping, fixed denominator, crop geometry); the office digest measures the **effect**.
- `history_crops_box` (box crops of the disjoint history pool) is left unpopulated in this plan; the history path counts as `n_no_candidate` until added. The LOO path is the primary bench case here and is fully implemented. If the office set is history-dominated, add a follow-up task mirroring `_crop_history_by_mod` for box crops.
