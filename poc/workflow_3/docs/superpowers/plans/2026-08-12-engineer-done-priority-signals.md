# Engineer-Done Priority Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Remote Monitoring window closure and fresh Assist quality the primary engineer-done signals, with three strictly increasing numerator readings used only after Assist is repeatedly unusable.

**Architecture:** `sem_monitor/assist_score.py` will expose a compact `AssistObservation` that distinguishes a readable panel from an unavailable one and fingerprints only the Measurement cells. `monitor/engineer_done_align_adjustment.py` will evaluate that observation before a separately sampled numerator state machine; readable red Assist evidence permanently disables the numerator fallback for that watch. The verified `assist_panel_crop_region` geometry stays unchanged, while Assist-specific locator artifacts use the run directory and service-accurate stage labels.

**Tech Stack:** Python >=3.10, dataclasses, Pillow, NumPy, OpenCV, existing `Workflow1VLMClient`, `uv`, pytest-compatible self-tests.

## Global Constraints

- Do not change `PANEL_LEFT_RATIO`, `PANEL_RIGHT_RATIO`, `PANEL_TOP_RATIO`, or `PANEL_BOTTOM_RATIO`; the current `assist_panel_crop_region` is office-verified.
- `Measurement` is required; `Addressing1` is optional and only vetoes when its visible score is red; `Addressing2` is not part of engineer-done.
- A readable red Assist result must never be overridden by numerator growth.
- Numerator fallback requires three successful OCR readings `n1 < n2 < n3`; unchanged frames do not reset the sequence, but equal/decreasing reads, OCR misses, and regrounding do.
- Use only `mai-ui` route slugs for current grounding defaults; do not reintroduce ui-venus service calls.
- Keep automation safe/offline by default and preserve the existing `engineer_watch_sec` cap.
- Invoke every Python program and test through `uv run`; do not add `argparse`, `sys.path` hacks, `__future__` imports, or broad formatting churn.

---

## File Structure

- `poc/workflow_3/sem_monitor/assist_score.py`: identify the two relevant columns, build/read their grid, produce `AssistObservation`, fingerprint Measurement cells, and save Assist evidence.
- `poc/workflow_3/sem_monitor/test_assist_score.py`: synthetic coverage for optional Addressing1, absent Addressing2, clipped Measurement headers, fingerprints, and observation status.
- `poc/workflow_3/vlm/ui_venus_mai_locator.py`: add an opt-in service-accurate artifact naming mode without changing legacy callers.
- `poc/workflow_3/vlm/test_locator_combo.py`: guard default legacy names and new Assist service-name mode.
- `poc/workflow_3/monitor/engineer_done_align_adjustment.py`: priority state machine, independent numerator sampling, calibration output, and run-directory wiring.
- `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`: detector state-machine and window-close behavior tests.
- `poc/workflow_3/config.py`: replace the obsolete delta threshold with explicit Assist-unusable and numerator-reading thresholds.
- `poc/workflow_3/README.md`: document the new completion priority and current `mai-ui` service default.

---

### Task 1: Make Measurement the Required Assist Column

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py:38-532`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Consumes: PaddleOCR spotting items in panel-pixel coordinates.
- Produces: `AssistLayout.grid`, with `columns` equal to `("Addressing1", "Measurement")` when Addressing1 is found or `("Measurement",)` when it is absent.
- Preserves: `row_verdict(cells: dict) -> str`, `read_row_states(image, layout) -> list[RowState]`, and the four panel crop ratios.

- [ ] **Step 1: Write failing tests for the two-column contract**

Add fixtures that omit Addressing2 and, separately, omit both Addressing columns:

```python
def _measurement_only_items():
    items = [_item("Measuremen", 210, 5, 260, 25)]
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("34", 220, top, 250, top + 18))
    return items


def test_grid_builds_without_addressing2():
    items = [item for item in _panel_items() if item["text"] != "Addressing2"]
    layout = build_score_grid(items, (300, 260))
    assert layout is not None
    assert layout.columns == ("Addressing1", "Measurement")


def test_grid_builds_with_measurement_only():
    layout = build_score_grid(_measurement_only_items(), (300, 260))
    assert layout is not None
    assert layout.columns == ("Measurement",)


def test_measurement_header_accepts_five_character_clip_only():
    assert asc._header_column_for("Measu") == "Measurement"
    assert asc._header_column_for("Meas") is None
```

Keep these as pytest tests; the final verification runs both the existing direct self-test and pytest collection.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_grid_builds_without_addressing2 \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_grid_builds_with_measurement_only \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_measurement_header_accepts_five_character_clip_only
```

Expected: the first two tests fail because `build_score_grid` currently requires all three headers. The clipped-header test already passes and is retained as a regression guard for the right-edge behavior.

- [ ] **Step 3: Implement the minimal active-column grid**

Use explicit column roles:

```python
ASSIST_COLUMNS = ("Addressing1", "Addressing2", "Measurement")
ASSIST_SCORE_COLUMNS = ("Addressing1", "Measurement")
ASSIST_REQUIRED_COLUMNS = ("Measurement",)
```

In `build_score_grid`:

```python
header_boxes = _match_header_boxes(items)
missing = [name for name in ASSIST_REQUIRED_COLUMNS if name not in header_boxes]
if missing:
    print(f"[WARNING] Assist 필수 헤더 누락({missing}) - 격자 생성 실패")
    return None

active_columns = tuple(
    name for name in ASSIST_SCORE_COLUMNS if name in header_boxes
)
number_x_ranges = {column: None for column in active_columns}
```

When only Measurement exists, do not assign every number on the panel to it merely because it is the nearest header. Accept a number only when its center lies inside the Measurement header span expanded by half the header width:

```python
def _number_column_for(cx, header_boxes, active_columns):
    if len(active_columns) > 1:
        return _assign_number_column(cx, {name: header_boxes[name] for name in active_columns})
    column = active_columns[0]
    box = header_boxes[column]
    width = max(1.0, float(box["right"]) - float(box["left"]))
    if float(box["left"]) - width * 0.5 <= cx <= float(box["right"]) + width * 0.5:
        return column
    return None
```

Build row boxes and `AssistLayout.columns` from `active_columns`, not all three legacy columns. Leave `row_verdict` unchanged because missing Addressing1 already defaults to blank and Measurement remains authoritative.

> **As-built (2026-08-13):** the shipped `_number_column_for` also guards the `len(active_columns) > 1`
> branch, which the snippet above left as plain nearest-header matching. Office capture showed
> Addressing2 scores landing inside the active columns' x geometry: nearest-header matching has no
> notion of "this number belongs to a column we are not scoring", so an Addressing2 digit was
> silently attributed to Addressing1 or Measurement and corrupted the verdict. The implementation
> rejects a number whose center falls inside an **inactive** header's expected span (same
> half-header-width tolerance) before running nearest-column matching, and clamps the per-column
> `number_x_ranges` away from inactive header spans. See `sem_monitor/assist_score.py:423-443` and
> `:541-552`. Tolerance for legitimate score/header bbox offsets inside active columns is unchanged.

- [ ] **Step 4: Run the whole Assist suite and verify GREEN**

Run:

```bash
uv run python poc/workflow_3/sem_monitor/test_assist_score.py
```

Expected: all cases pass. Update old tests that assert three layout columns so they assert the active columns actually present in their fixture; keep a three-header fixture test proving Addressing2 may still be recognized without affecting verdict logic.

- [ ] **Step 5: Commit the active-column change**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "Fix Assist score columns for engineer done"
```

---

### Task 2: Add Explicit Assist Observations and Freshness

**Files:**
- Modify: `poc/workflow_3/sem_monitor/assist_score.py:130-834`
- Modify: `poc/workflow_3/monitor/engineer_done_align_adjustment.py:367-450`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`
- Test: `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`

**Interfaces:**
- Produces: `AssistObservation(status: str, rows: list[RowState], panel_fingerprint: str | None, reason: str)`.
- Produces: `measurement_fingerprint(image, layout) -> str | None`.
- Produces: `_make_assist_fn(tool_window, settings, debug_dir=None) -> Callable[[Image.Image], AssistObservation]`.
- Consumes later: `EngineerDoneDetector(tool_window, settings, assist_fn=callable)` in Task 4.

- [ ] **Step 1: Write failing tests for observation status and Measurement-only freshness**

Add:

```python
def test_measurement_fingerprint_ignores_addressing_changes():
    layout = build_score_grid(_panel_items(), (300, 260))
    before = _synth_panel([("black", "black")] * 7)
    after = before.copy()
    ImageDraw.Draw(after).rectangle((15, 40, 55, 55), fill=(200, 20, 20))
    assert asc.measurement_fingerprint(before, layout) == asc.measurement_fingerprint(after, layout)


def test_measurement_fingerprint_changes_with_measurement_cells():
    layout = build_score_grid(_panel_items(), (300, 260))
    before = _synth_panel([("black", "black")] * 7)
    after = before.copy()
    ImageDraw.Draw(after).rectangle((215, 40, 255, 55), fill=(200, 20, 20))
    assert asc.measurement_fingerprint(before, layout) != asc.measurement_fingerprint(after, layout)
```

Add monitor-side closure tests using the existing `_RowsHarness` pattern:

```python
def test_assist_fn_distinguishes_unusable_from_pending():
    image = Image.new("RGB", (20, 20), (240, 240, 240))
    with _RowsFnHarness([], locate_ok=False):
        failed_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        failed = failed_fn(image)

    pending_rows = _rows_of(["pending"] * 7)
    with _RowsFnHarness([pending_rows], locate_ok=True):
        pending_fn = _make_assist_fn(object(), _settings(), debug_dir=None)
        pending = pending_fn(image)

    assert failed.status == "unusable"
    assert failed.reason == "layout_unavailable"
    assert pending.status == "unusable"
    assert pending.reason == "measurement_unreadable"
    assert [row.verdict for row in pending.rows] == ["pending"] * 7
```

Update `_RowsFnHarness._locate` to return a real two-column grid inside its
20x20 synthetic panel instead of the old empty box:

```python
grid = []
for row_idx in range(7):
    top = 1 + row_idx * 2
    grid.append([
        {"left": 1, "top": top, "right": 7, "bottom": top + 1},
        {"left": 12, "top": top, "right": 18, "bottom": top + 1},
    ])
layout = SimpleNamespace(
    grid=grid, columns=("Addressing1", "Measurement")
)
```

This makes `measurement_fingerprint` exercise production geometry.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_measurement_fingerprint_ignores_addressing_changes \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_measurement_fingerprint_changes_with_measurement_cells \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_assist_fn_distinguishes_unusable_from_pending
```

Expected: FAIL because `measurement_fingerprint`, `AssistObservation`, and `_make_assist_fn` do not exist.

- [ ] **Step 3: Implement the observation value and fingerprint**

In `assist_score.py`:

```python
import hashlib


@dataclass(frozen=True)
class AssistObservation:
    status: str
    rows: list[RowState] = field(default_factory=list)
    panel_fingerprint: str | None = None
    reason: str = ""


def measurement_fingerprint(image, layout) -> str | None:
    if image is None or layout is None or "Measurement" not in layout.columns:
        return None
    column_idx = layout.columns.index("Measurement")
    gray = np.array(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    chunks = []
    for row in layout.grid:
        box = row[column_idx]
        left = max(0, min(width, int(box["left"])))
        right = max(left, min(width, int(box["right"])))
        top = max(0, min(height, int(box["top"])))
        bottom = max(top, min(height, int(box["bottom"])))
        cell = gray[top:bottom, left:right]
        if cell.size:
            chunks.append((cell // 16).tobytes())
    if not chunks:
        return None
    return hashlib.sha256(b"".join(chunks)).hexdigest()
```

Quantizing to 16-level buckets suppresses small remote-render noise while preserving a changed score/thumbnail.

- [ ] **Step 4: Replace `_make_rows_fn` with an image-consuming Assist closure**

Rename it `_make_assist_fn`. Preserve its cached layout, throttle, blank-relocate cap, and overlay-on-change state. The closure signature becomes:

```python
def assist_fn(image):
    if state["layout"] is None and time.time() < state["next_locate_at"]:
        return AssistObservation(status="unusable", reason="locate_throttled")
    if state["layout"] is None:
        located = locate_assist_layout(
            tool_window, "", "", image, debug_dir=debug_dir
        )
        if located is None:
            state["next_locate_at"] = time.time() + max(
                settings.engineer_done_reground_sec, 0.0
            )
            return AssistObservation(status="unusable", reason="layout_unavailable")
        state["panel_box"], state["layout"] = located

    panel = crop_image(image, state["panel_box"])
    rows = read_row_states(panel, state["layout"])
    if not rows:
        return AssistObservation(status="unusable", reason="rows_empty")
    fingerprint = measurement_fingerprint(panel, state["layout"])
    if not any(row.verdict in {"ok", "fail"} for row in rows):
        return AssistObservation(
            status="unusable",
            rows=rows,
            panel_fingerprint=fingerprint,
            reason="measurement_unreadable",
        )
    return AssistObservation(
        status="usable",
        rows=rows,
        panel_fingerprint=fingerprint,
        reason="ok",
    )
```

Do not call `capture_window` inside this closure; Task 4 supplies the detector's one full-window capture per poll.

- [ ] **Step 5: Run both focused suites and verify GREEN**

Run:

```bash
uv run python poc/workflow_3/sem_monitor/test_assist_score.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
```

Expected: all tests pass after updating the old `_make_rows_fn` harness imports and calls to `_make_assist_fn(image)`.

- [ ] **Step 6: Commit the observation seam**

```bash
git add poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py \
  poc/workflow_3/monitor/engineer_done_align_adjustment.py \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
git commit -m "Add explicit Assist observations"
```

---

### Task 3: Unify Assist Artifacts with Service-Accurate Names

**Files:**
- Modify: `poc/workflow_3/vlm/ui_venus_mai_locator.py:330-865`
- Modify: `poc/workflow_3/sem_monitor/assist_score.py:692-783`
- Test: `poc/workflow_3/vlm/test_locator_combo.py`
- Test: `poc/workflow_3/sem_monitor/test_assist_score.py`

**Interfaces:**
- Produces: optional `artifact_naming: str = "legacy"` argument on `analyze_window_target`.
- `artifact_naming="service"` produces neutral artifact dict keys (`coarse_response`, `refine_response`, `result_json`, `overlay`, `zoom_overlay`) and filenames derived from the actual route slugs.
- Preserves: every existing caller's legacy filenames and artifact keys by default.

- [ ] **Step 1: Write failing tests for opt-in artifact naming and run-directory routing**

Add a locator helper test around a pure naming function:

```python
def test_service_artifact_names_use_actual_route_slugs():
    names = loc._artifact_names("assist_panel", "mai-ui", "mai-ui", mode="service")
    assert names["coarse_response"] == "assist_panel_coarse_mai_ui_response.txt"
    assert names["refine_response"] == "assist_panel_refine_mai_ui_response.txt"
    assert names["result_json"] == "assist_panel_locator_result.json"


def test_legacy_artifact_names_remain_default():
    names = loc._artifact_names("assist_panel", "mai-ui", "mai-ui", mode="legacy")
    assert names["coarse_response"] == "assist_panel_ui_venus_response.txt"
```

Extend the Assist locate stub to capture `debug_image_dir` and `artifact_naming`:

```python
assert call["debug_image_dir"] == debug_dir
assert call["artifact_naming"] == "service"
assert (debug_dir / "assist_panel_crop_region.jpg").exists()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/vlm/test_locator_combo.py::test_service_artifact_names_use_actual_route_slugs \
  poc/workflow_3/vlm/test_locator_combo.py::test_legacy_artifact_names_remain_default \
  poc/workflow_3/sem_monitor/test_assist_score.py::test_locate_uses_run_dir_and_saves_panel_crop
```

Expected: FAIL because `_artifact_names`, `artifact_naming`, and the crop-region artifact do not exist.

- [ ] **Step 3: Implement opt-in naming without altering legacy behavior**

Add:

```python
def _slug_token(value: str) -> str:
    return "_".join(part for part in re.split(r"[^a-zA-Z0-9]+", value or "") if part).lower()


def _artifact_names(prefix, coarse_service, refine_service, *, mode="legacy"):
    if mode != "service":
        return {
            "coarse_response": f"{prefix}_ui_venus_response.txt",
            "refine_response": f"{prefix}_mai_ui_response.txt",
            "result_json": f"{prefix}_ui_venus_mai_result.json",
            "overlay": f"{prefix}_ui_venus_mai_overlay.jpg",
            "zoom_overlay": f"{prefix}_ui_venus_mai_zoom_overlay.jpg",
        }
    coarse = _slug_token(coarse_service)
    refine = _slug_token(refine_service)
    return {
        "coarse_response": f"{prefix}_coarse_{coarse}_response.txt",
        "refine_response": f"{prefix}_refine_{refine}_response.txt",
        "result_json": f"{prefix}_locator_result.json",
        "overlay": f"{prefix}_locator_overlay.jpg",
        "zoom_overlay": f"{prefix}_locator_zoom_overlay.jpg",
    }
```

Thread `artifact_naming` through the success and both failure branches. In service mode use overlay labels `coarse_<slug>_bbox`, `<target>_crop_region`, and `refine_<slug>_point`; retain old labels in legacy mode.

- [ ] **Step 4: Route all Assist evidence to its run and save the verified crop**

In `locate_assist_layout`:

```python
artifact_dir = debug_dir if debug_dir is not None else DEBUG_ARTIFACT_DIR
result = analyze_window_target(
    window, window_title, backend, assist_panel_target(),
    debug_image_dir=artifact_dir,
    log_name=LOG_NAME,
    component_name=LOG_NAME,
    artifact_prefix="assist_panel",
    artifact_naming="service",
    image=image,
    timeout_sec=15.0,
)
```

Immediately after the unchanged `panel_box` crop:

```python
save_debug_jpeg(panel.convert("RGB"), artifact_dir / "assist_panel_crop_region.jpg")
```

- [ ] **Step 5: Run locator and Assist regression suites**

Run:

```bash
uv run python poc/workflow_3/vlm/test_locator_combo.py
uv run python poc/workflow_3/sem_monitor/test_assist_score.py
```

Expected: both suites pass; old non-Assist callers still use legacy artifact names.

- [ ] **Step 6: Commit artifact routing**

```bash
git add poc/workflow_3/vlm/ui_venus_mai_locator.py poc/workflow_3/vlm/test_locator_combo.py \
  poc/workflow_3/sem_monitor/assist_score.py poc/workflow_3/sem_monitor/test_assist_score.py
git commit -m "Clarify Assist locator debug artifacts"
```

---

### Task 4: Implement the Priority Detector and Three-Reading Fallback

**Files:**
- Modify: `poc/workflow_3/monitor/engineer_done_align_adjustment.py:1-590`
- Modify: `poc/workflow_3/config.py:84-102,276-286`
- Test: `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py`

**Interfaces:**
- Consumes: `assist_fn(image) -> AssistObservation` from Task 2.
- Produces: `NumeratorObservation(sampled: bool, value: int | None, reason: str)`.
- Adds test injection: `numerator_fn(image) -> NumeratorObservation`; production leaves it unset and uses `_observe_numerator(image)`.
- Produces settings: `engineer_done_assist_unusable_after: int = 3`, `engineer_done_numerator_increase_reads: int = 3`.
- Removes operational use of: `engineer_done_min_delta` and the old baseline/delta AND condition.

- [ ] **Step 1: Write failing detector tests for priority and freshness**

Use injected full-frame captures, `AssistObservation` sequences, and numerator observations. Add these helpers first:

```python
def _assist_ok(fingerprint):
    return asc.AssistObservation(
        status="usable", rows=_rows_all_ok(7),
        panel_fingerprint=fingerprint, reason="ok",
    )


def _assist_fail(fingerprint):
    rows = _rows_all_ok(7)
    rows[-1].cells["Measurement"] = "red"
    return asc.AssistObservation(
        status="usable", rows=rows,
        panel_fingerprint=fingerprint, reason="ok",
    )


def _assist_unusable():
    return asc.AssistObservation(
        status="unusable", rows=[], panel_fingerprint=None,
        reason="layout_unavailable",
    )


def _priority_detector(*, assist, numerators):
    observations = [
        NumeratorObservation(
            sampled=True,
            value=value,
            reason="read" if value is not None else "ocr_miss",
        )
        for value in numerators
    ] or [NumeratorObservation(sampled=False, reason="no_change")]
    return EngineerDoneDetector(
        None,
        _settings(
            engineer_done_ok_streak=6,
            engineer_done_assist_unusable_after=3,
            engineer_done_numerator_increase_reads=3,
        ),
        capture_fn=lambda: Image.new("RGB", (400, 200), (0, 0, 0)),
        assist_fn=assist,
        numerator_fn=_CountingFn(observations),
    )


def _run_numerator_sequence(values):
    assist = _CountingFn([_assist_unusable()] * len(values))
    detector = _priority_detector(assist=assist, numerators=values)
    return [detector() for _ in values][-1]
```

Then add:

```python
def test_assist_needs_fresh_change_after_watch_start():
    assist = _CountingFn([
        _assist_ok("same"),
        _assist_ok("same"),
        _assist_ok("changed"),
    ])
    detector = _priority_detector(assist=assist, numerators=[])
    assert [detector(), detector(), detector()] == [False, False, True]


def test_red_assist_permanently_blocks_numerator_fallback():
    assist = _CountingFn([
        _assist_fail("red"),
        _assist_unusable(),
        _assist_unusable(),
        _assist_unusable(),
    ])
    detector = _priority_detector(assist=assist, numerators=[10, 11, 12])
    assert [detector(), detector(), detector(), detector()] == [False] * 4


def test_numerator_fallback_requires_three_unusable_assist_observations():
    assist = _CountingFn([_assist_unusable(), _assist_unusable(), _assist_unusable()])
    detector = _priority_detector(assist=assist, numerators=[10, 11, 12])
    assert [detector(), detector(), detector()] == [False, False, True]


def test_builder_keeps_assist_when_numerator_clients_fail():
    import poc.workflow_3.monitor.engineer_done_align_adjustment as module

    saved = (module._make_ground_fn, module._make_ocr_fn, module._make_assist_fn)
    try:
        module._make_ground_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("numerator grounding unavailable")
        )
        module._make_ocr_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("numerator OCR unavailable")
        )
        module._make_assist_fn = lambda *args, **kwargs: (
            lambda image: _assist_ok("fresh")
        )
        detector = module.build_engineer_done_detector(object(), _settings())
    finally:
        module._make_ground_fn, module._make_ocr_fn, module._make_assist_fn = saved
    assert detector is not None
```

Add parameterized or looped cases proving these do not finish:

```python
for values in ([10, 10, 11], [10, 12, None], [10, 9, 10]):
    assert not _run_numerator_sequence(values)
```

Also assert that an Assist observation becoming usable resets `assist_unusable_streak` to zero. Keep the new assertion-style cases under pytest; preserve the existing boolean-style `main()` list for its legacy smoke coverage.

- [ ] **Step 2: Run focused state-machine tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_assist_needs_fresh_change_after_watch_start \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_red_assist_permanently_blocks_numerator_fallback \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_numerator_fallback_requires_three_unusable_assist_observations \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_invalid_numerator_sequences_do_not_finish \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_builder_keeps_assist_when_numerator_clients_fail
```

Expected: FAIL because the detector still requires delta AND Assist streak and has no priority state.

- [ ] **Step 3: Introduce explicit numerator samples**

Add:

```python
@dataclass(frozen=True)
class NumeratorObservation:
    sampled: bool
    value: int | None = None
    reason: str = ""
```

Refactor localization/cropping to accept the already captured image. `_observe_numerator(image)` returns:

- `NumeratorObservation(False, reason="no_change")` when the crop has not changed; do not reset sequence.
- `NumeratorObservation(True, n, "read")` on a successful OCR reading.
- `NumeratorObservation(True, None, "ocr_miss")` on a changed crop that OCR cannot read; reset sequence.

Do not use `_baseline_n` or compute delta.

Add `numerator_fn=None` to `EngineerDoneDetector.__init__`, store it as
`self._numerator_fn`, and select the test injection without changing production wiring:

```python
numerator = (
    self._numerator_fn(image)
    if self._numerator_fn is not None
    else self._observe_numerator(image)
)
```

- [ ] **Step 4: Implement sequence update and priority evaluation**

Initialize:

```python
self._assist_unusable_streak = 0
self._assist_baseline_fingerprint = None
self._assist_changed_since_start = False
self._assist_failure_seen = False
self._numerator_sequence: list[int] = []
```

Add:

```python
def _update_numerator_sequence(self, observation):
    if not observation.sampled:
        return
    n = observation.value
    if n is None:
        self._numerator_sequence.clear()
        return
    if self._numerator_sequence and n > self._numerator_sequence[-1]:
        self._numerator_sequence.append(n)
    else:
        self._numerator_sequence = [n]
    keep = max(1, self.s.engineer_done_numerator_increase_reads)
    self._numerator_sequence = self._numerator_sequence[-keep:]
```

At the start of `__call__`, capture once and evaluate Assist. Remove the old
`_roi_ratios is None` early return from `__call__`; localization refusal must
skip only the numerator observation, never Assist primary evaluation:

```python
image = self._capture_fn()
assist = self._assist_fn(image) if self._assist_fn is not None else AssistObservation(
    status="unusable", reason="assist_fn_missing"
)
if assist.status == "usable":
    self._assist_unusable_streak = 0
    verdicts = [row.verdict for row in assist.rows]
    if "fail" in verdicts:
        self._assist_failure_seen = True
    if self._assist_baseline_fingerprint is None:
        self._assist_baseline_fingerprint = assist.panel_fingerprint
    elif assist.panel_fingerprint != self._assist_baseline_fingerprint:
        self._assist_changed_since_start = True
    streak = ok_streak(assist.rows)
    if self._assist_changed_since_start and streak >= self.s.engineer_done_ok_streak:
        return True
else:
    self._assist_unusable_streak += 1

numerator = self._observe_numerator(image)
self._update_numerator_sequence(numerator)
fallback_open = (
    self._assist_unusable_streak >= self.s.engineer_done_assist_unusable_after
    and not self._assist_failure_seen
)
return fallback_open and len(self._numerator_sequence) >= self.s.engineer_done_numerator_increase_reads
```

When numerator ROI is invalidated or regrounded, clear `_numerator_sequence`.

Refactor `build_engineer_done_detector` so `_make_assist_fn` is always built
first. Catch numerator client creation separately:

```python
assist_fn = _make_assist_fn(tool_window, settings, debug_dir=debug_dir)
try:
    ground_fn = _make_ground_fn(settings, vlm_client=vlm_client)
    ocr_fn = _make_ocr_fn(settings)
except Exception as exc:
    print(f"[WARNING] numerator fallback client 생성 실패(Assist primary 유지): {exc}")
    ground_fn = None
    ocr_fn = None
return EngineerDoneDetector(
    tool_window,
    settings,
    ground_fn=ground_fn,
    ocr_fn=ocr_fn,
    assist_fn=assist_fn,
    debug_dir=debug_dir,
)
```

- [ ] **Step 5: Replace obsolete settings**

In `Workflow3Settings`:

```python
engineer_done_ok_streak: int = 6
engineer_done_assist_unusable_after: int = 3
engineer_done_numerator_increase_reads: int = 3
```

In `load_workflow3_settings`:

```python
engineer_done_assist_unusable_after=env_int(
    "ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER", 3
),
engineer_done_numerator_increase_reads=env_int(
    "ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS", 3
),
```

Remove `engineer_done_min_delta` and `ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA` references from executable code and current README. Do not edit the historical 2026-08-11 spec/plan.

> **As-built (2026-08-13):** replacing the fields was not enough on its own. These three settings are
> the only thing standing between "engineer finished" and "close the tool window", and every one of
> them fails **open** if misconfigured: `ok_streak=0` completes on the first poll,
> `numerator_increase_reads=0` completes without any counter growth. Env vars are typed by hand at
> the office, so a typo silently disarms the detector instead of erroring.
>
> Shipped alongside the field replacement:
>
> - `validate_engineer_done_priority_settings` (`config.py:48`), called from
>   `Workflow3Settings.__post_init__` (`config.py:127`) so an invalid combination cannot be
>   constructed at all.
> - A `_configuration_error` fail-closed path in the detector
>   (`engineer_done_align_adjustment.py:169-182`). With the frozen dataclass validating on
>   construction this branch is unreachable in production and only fires for hand-built test
>   doubles - kept deliberately, because the detector must never silently complete on a settings
>   object it did not construct.

- [ ] **Step 6: Run the complete detector suite and verify GREEN**

Run:

```bash
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
```

Expected: all tests pass. Replace old delta-focused tests with priority tests; retain grounding, OCR parsing, throttle, exception, and relocalization coverage where behavior remains relevant.

- [ ] **Step 7: Commit the priority detector**

```bash
git add poc/workflow_3/config.py poc/workflow_3/monitor/engineer_done_align_adjustment.py \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
git commit -m "Change engineer done to priority signals"
```

---

### Task 5: Make Manual Window Closure Explicit and Update Operator Docs

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py:571-603`
- Modify: `poc/workflow_3/monitor/test_engineer_done_align_adjustment.py:357-395`
- Modify: `poc/workflow_3/README.md:60-75,205-220`
- Modify: `poc/workflow_3/workflow_3_config.example.py:110-120`

**Interfaces:**
- Consumes: `RecordingSession.stop_reason`.
- Produces: an explicit engineer-completion log only for `window_gone`; other recording termination reasons remain accurately labeled.

- [ ] **Step 1: Write failing watch-reason tests**

Extend `_FakeRecording` with `stop_reason` and add:

```python
def test_watch_logs_manual_completion_only_for_window_gone():
    recording = _FakeRecording(alive_checks=0, stop_reason="window_gone")
    output = io.StringIO()
    with redirect_stdout(output):
        _engineer_watch(recording, 60.0, poll_sec=0.0)
    assert "엔지니어가 Remote Monitoring 창을 닫음" in output.getvalue()


def test_watch_does_not_call_max_sec_manual_completion():
    recording = _FakeRecording(alive_checks=0, stop_reason="max_sec")
    output = io.StringIO()
    with redirect_stdout(output):
        _engineer_watch(recording, 60.0, poll_sec=0.0)
    assert "엔지니어가 Remote Monitoring 창을 닫음" not in output.getvalue()
    assert "max_sec" in output.getvalue()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_watch_logs_manual_completion_only_for_window_gone \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py::test_watch_does_not_call_max_sec_manual_completion
```

Expected: FAIL because `_engineer_watch` currently logs only the generic end.

- [ ] **Step 3: Add reason-aware terminal logging**

After the loop:

```python
reason = getattr(recording, "stop_reason", "")
if reason == "window_gone":
    print("[INFO] 엔지니어가 Remote Monitoring 창을 닫음 - 명시적 완료로 watch 종료")
elif reason:
    print(f"[INFO] 녹화 세션 종료로 watch 종료: reason={reason}")
print("[INFO] engineer watch 종료")
```

Avoid a second state-changing window check; the recording thread owns this signal.

- [ ] **Step 4: Update current operator documentation**

Replace README's old `delta AND streak` explanation with this ordered table:

```markdown
1. 엔지니어가 Remote Monitoring 창의 X를 눌러 닫음 (`window_gone`) → 즉시 종료
2. Assist Measurement 최신 정상 6행 + watch 이후 새 화면 변화 → 종료
3. Assist 3회 연속 판독 불가 + numerator OCR 3회 연속 증가 → fallback 종료
4. 불확정 → `ALIGN_FAIL_ENGINEER_WATCH_SEC` cap까지 대기
```

Document:

- `ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER=3`
- `ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS=3`
- `ALIGN_FAIL_ENGINEER_DONE_OK_STREAK=6`
- `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE=mai-ui`

Remove the stale README default `ui-venus` and obsolete MIN_DELTA tuning advice. Update `workflow_3_config.example.py` comments without adding new module constants because the two thresholds are env-driven operational tuning.

- [ ] **Step 5: Run focused and static verification**

Run:

```bash
uv run python poc/workflow_3/sem_monitor/test_assist_score.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/vlm/test_locator_combo.py
git diff --check
```

Expected: every suite passes and `git diff --check` is silent.

- [ ] **Step 6: Commit docs and close-signal logging**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_engineer_done_align_adjustment.py \
  poc/workflow_3/README.md poc/workflow_3/workflow_3_config.example.py
git commit -m "Clarify engineer watch completion signals"
```

---

## Final Verification

- [ ] Run all focused tests:

```bash
uv run python poc/workflow_3/sem_monitor/test_assist_score.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/vlm/test_locator_combo.py
```

- [ ] Run pytest forms to catch collection/import differences:

```bash
uv run pytest -q \
  poc/workflow_3/sem_monitor/test_assist_score.py \
  poc/workflow_3/monitor/test_engineer_done_align_adjustment.py \
  poc/workflow_3/vlm/test_locator_combo.py
```

- [ ] Confirm no operational code or current docs retain the old rule:

```bash
rg -n "engineer_done_min_delta|ENGINEER_DONE_MIN_DELTA|두 조건을 모두" \
  poc/workflow_3 --glob '*.py' --glob 'README.md' --glob '*.example.py'
```

Expected: no matches outside historical specs/plans.

- [ ] Confirm the crop geometry is unchanged:

```bash
git diff b9d5c01 -- poc/workflow_3/sem_monitor/assist_score.py | \
  rg "PANEL_(LEFT|RIGHT|TOP|BOTTOM)_RATIO"
```

Expected: no changed ratio definitions.

- [ ] Confirm repository hygiene:

```bash
git status --short
git diff --check
```

- [ ] Office Windows calibration remains required before enabling `ALIGN_FAIL_ENGINEER_DONE_DETECT=1`. Record the run directory containing `assist_panel_crop_region.jpg`, locator overlay, OCR overlay, grid overlay, numerator readings, and final completion reason.
