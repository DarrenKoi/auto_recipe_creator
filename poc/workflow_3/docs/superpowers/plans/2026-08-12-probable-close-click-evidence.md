# Probable Close-Click Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a low-confidence `probable_close_click` only when cursor VLM misses the final top-right interaction immediately before a confirmed `window_gone` recording termination.

**Architecture:** A new pure recording-filter module will combine the existing recording manifest, last `ChangeEvent`, last cursor result, and a conservative diagonal-stroke check in the top-right change ROI. It will emit evidence in the offline interaction timeline and debug folder, but the live monitor and GUI automation layers will never consume it as a completion or replay instruction.

**Tech Stack:** Python >=3.10, dataclasses/dicts, JSON, Pillow, NumPy, OpenCV, existing recording-filter timeline and debug helpers, `uv`, pytest.

## Global Constraints

- `RecordingSession.stop_reason == "window_gone"` remains the only live proof that the Remote Monitoring window disappeared.
- `probable_close_click` is post-processing evidence only; it must not trigger window close, click replay, engineer-done, or teardown behavior.
- Require all three: `window_gone`, final top-right change, and cursor VLM missing/unavailable.
- The fixed close-button X alone is never sufficient because it is present in every frame.
- Accept a partially visible lower-half X candidate; fail closed when diagonal evidence is absent.
- Keep confidence low (`0.35`) and record the exact evidence string `window_gone + top_right_change + cursor_vlm_missing`.
- Invoke every Python program and test through `uv run`; add no CLI flags, new services, or live GUI actions.

---

## File Structure

- `poc/workflow_3/recording_filter/close_click_evidence.py`: pure manifest/change/cursor/CV inference and evidence overlay writer.
- `poc/workflow_3/recording_filter/test_close_click_evidence.py`: synthetic positive and negative cases.
- `poc/workflow_3/recording_filter/filter_recording.py`: invoke inference after cursor detection, merge the result into output, and count it in summary.
- `poc/workflow_3/recording_filter/timeline.py`: accept explicit inferred evidence events without treating them as clicks.
- `poc/workflow_3/recording_filter/test_filter_recording.py`: end-to-end artifact/timeline proof.
- `poc/workflow_3/README.md`: operator meaning and safety boundary.

As-built additions (2026-08-13), not anticipated when this plan was written:

- `poc/workflow_3/workflow_extract/grouping.py` + `extract_workflow.py`: the downstream consumer
  must also refuse to treat an inferred event as a performed action. Filtering only inside
  `recording_filter` was not enough - `workflow_extract` reads the timeline and would have rendered
  `probable_close_click` into a Korean procedure step as though the engineer had really clicked it.
  `_workflow_action_events` drops non-`replayable` events before grouping; `grouping.py` carries the
  matching R5 guard.
- `poc/workflow_3/recording_filter/filter_recording.py:_reset_close_click_evidence`: clears a stale
  `close_click_evidence/` directory at the start of a run. Without it, a re-run that infers nothing
  leaves the previous run's evidence images on disk, which reads as "this run found a close click".

---

### Task 1: Build a Conservative Close-Click Evidence Classifier

**Files:**
- Create: `poc/workflow_3/recording_filter/close_click_evidence.py`
- Create: `poc/workflow_3/recording_filter/test_close_click_evidence.py`

**Interfaces:**
- Consumes: `capture_dir: Path`, `change_events: list[ChangeEvent]`, `click_events: list[ClickEvent]`.
- Produces: `infer_probable_close_click(capture_dir, change_events, click_events) -> dict | None` in the common timeline schema.
  **As-built (2026-08-13):** the second parameter shipped as `candidate_events`, not `change_events`.
  `run_filter` passes the raw Stage 1 list rather than the gate survivors (see Task 2 Step 5), and
  `change_events` is the established name for the gate-surviving list everywhere else in this
  package - keeping it here would have named the parameter after the one input it must not receive.
  The function only ever uses `candidate_events[-1]`, and the cursor lookup fails closed when that
  exact rank has no cursor result, so the wider input cannot loosen a gate.
- Produces: `write_close_click_evidence(event: dict, change: ChangeEvent, out_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write a synthetic positive test**

Create two 600x400 grayscale frames. Draw a lower-half X candidate only in the current frame at the top-right using two diagonal strokes. Write a `recording_manifest.json` with `stop_reason: "window_gone"`, a matching final `ChangeEvent`, and a final `ClickEvent` with `cursor_source="vlm"`, `cursor_visible=False`, and no coordinates. Use this fixture helper:

```python
def _close_candidate_fixture(
    tmp_path, *, stop_reason="window_gone", cursor_visible=False,
    top_right=True, diagonal=True,
):
    rec = tmp_path / "run" / "recording"
    rec.mkdir(parents=True)
    prev = np.full((400, 600), 240, dtype=np.uint8)
    cv2.line(prev, (580, 10), (590, 20), 40, 2)
    cv2.line(prev, (590, 10), (580, 20), 40, 2)
    curr = prev.copy()
    origin_x, origin_y = (560, 12) if top_right else (280, 180)
    if diagonal:
        cv2.line(curr, (origin_x, origin_y), (origin_x + 18, origin_y + 18), 10, 3)
        cv2.line(curr, (origin_x + 18, origin_y), (origin_x, origin_y + 18), 10, 3)
    else:
        cv2.rectangle(curr, (origin_x, origin_y), (origin_x + 18, origin_y + 18), 10, -1)
    prev_path = rec / "tag_rcs_0000_00000000ms.jpg"
    curr_path = rec / "tag_rcs_0001_00000300ms.jpg"
    cv2.imwrite(str(prev_path), prev)
    cv2.imwrite(str(curr_path), curr)
    (rec / "recording_manifest.json").write_text(
        json.dumps({"stop_reason": stop_reason}), encoding="utf-8"
    )
    change = ChangeEvent(
        rank=0, frame_path=str(curr_path), prev_frame_path=str(prev_path),
        timestamp_sec=0.3, frame_index=1,
        change_bbox={
            "left": origin_x, "top": origin_y,
            "right": origin_x + 19, "bottom": origin_y + 19,
        },
        largest_blob_area_px=361, changed_pixels=361,
    )
    cursor_box = {
        "left": origin_x, "top": origin_y,
        "right": origin_x + 19, "bottom": origin_y + 19,
    }
    click = ClickEvent(
        change=change, is_click=cursor_visible,
        status="click" if cursor_visible else "no_click",
        cursor_visible=cursor_visible,
        cursor_kind="rcs_black_arrow" if cursor_visible else None,
        cursor_bbox=cursor_box if cursor_visible else None,
        cursor_xy=[origin_x + 9, origin_y + 9] if cursor_visible else None,
        click_window=None, changed_in_window_px=0,
        confidence=0.0, evidence="", cursor_source="vlm",
    )
    return rec, change, click
```

```python
def test_infers_probable_close_click_from_all_three_signals(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path, stop_reason="window_gone")
    event = infer_probable_close_click(rec, [change], [click])
    assert event is not None
    assert event["action"] == "probable_close_click"
    assert event["confidence"] == 0.35
    assert event["evidence"] == (
        "window_gone + top_right_change + cursor_vlm_missing"
    )
    assert event["replayable"] is False
```

- [ ] **Step 2: Write negative tests for every missing gate**

```python
def test_no_inference_without_window_gone(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path, stop_reason="max_sec")
    assert infer_probable_close_click(rec, [change], [click]) is None


def test_no_inference_when_cursor_was_found(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path, cursor_visible=True)
    assert infer_probable_close_click(rec, [change], [click]) is None


def test_no_inference_when_change_is_not_top_right(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path, top_right=False)
    assert infer_probable_close_click(rec, [change], [click]) is None


def test_no_inference_for_static_close_x_without_diagonal_change(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path, diagonal=False)
    assert infer_probable_close_click(rec, [change], [click]) is None


def test_no_inference_when_final_event_was_truncated_from_cursor_results(tmp_path):
    rec, change, _click = _close_candidate_fixture(tmp_path)
    assert infer_probable_close_click(rec, [change], []) is None
```

Each test changes one variable from the positive fixture and asserts `is None`.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
uv run pytest -q poc/workflow_3/recording_filter/test_close_click_evidence.py
```

Expected: collection fails because `close_click_evidence.py` does not exist.

- [ ] **Step 4: Implement manifest and terminal-event gates**

Create:

```python
EVIDENCE_TEXT = "window_gone + top_right_change + cursor_vlm_missing"
CLOSE_REGION_WIDTH_RATIO = 0.10
CLOSE_REGION_HEIGHT_RATIO = 0.10
PROBABLE_CLOSE_CONFIDENCE = 0.35


def _load_stop_reason(capture_dir: Path) -> str:
    path = Path(capture_dir) / "recording_manifest.json"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("stop_reason") or "")
    except (OSError, ValueError, TypeError):
        return ""


def _cursor_missing_for(change, click_events) -> bool:
    event = next((item for item in click_events if item.rank == change.rank), None)
    return bool(
        event is not None
        and event.cursor_source in {"vlm", "none"}
        and not event.cursor_visible
        and event.cursor_xy is None
        and event.status in {"no_click", "cursor_unavailable"}
    )
```

Use the final `change_events[-1]` only, require a matching cursor result for that exact rank, and return `None` when Stage 2a was truncated before it.

- [ ] **Step 5: Implement top-right geometry and partial-X CV check**

Read the final current frame to obtain its dimensions. Define the close area as the rightmost/topmost 10%, with minimum size 64x48 pixels:

```python
def _close_region(width, height):
    region_w = max(64, int(round(width * CLOSE_REGION_WIDTH_RATIO)))
    region_h = max(48, int(round(height * CLOSE_REGION_HEIGHT_RATIO)))
    return {"left": max(0, width - region_w), "top": 0, "right": width, "bottom": min(height, region_h)}
```

Require `change.change_bbox` to intersect this region. Build an absolute-difference mask from the previous/current frames, crop it to the intersection, and use `cv2.HoughLinesP`. A partial X candidate exists only when the detected line set contains at least one positive-slope and one negative-slope diagonal with absolute angle between 20 and 70 degrees and endpoints within 24 pixels of one another:

```python
def _has_diagonal_pair(mask) -> bool:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=8, minLineLength=6, maxLineGap=3)
    if lines is None:
        return False
    positive, negative = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        folded = ((angle + 90) % 180) - 90
        if 20 <= folded <= 70:
            positive.append((x1, y1, x2, y2))
        elif -70 <= folded <= -20:
            negative.append((x1, y1, x2, y2))
    return any(_line_endpoint_distance(a, b) <= 24 for a in positive for b in negative)
```

This accepts a lower-half X/V-like pair but rejects the fixture's fixed close-button X because that X exists in both frames and disappears from the frame difference.

- [ ] **Step 6: Return a non-replayable timeline event and evidence paths**

Return:

```python
{
    "t_sec": change.timestamp_sec,
    "seq": 0,
    "action": "probable_close_click",
    "coords": {"x": center_x, "y": center_y},
    "element": "Remote Monitoring close button",
    "element_source": "inferred",
    "target_kind": "ui_control",
    "region": "window_title_bar",
    "generation": 0,
    "occlusion": "unknown",
    "cursor_source": "inferred_after_vlm_miss",
    "text": None,
    "confidence": PROBABLE_CLOSE_CONFIDENCE,
    "frame": Path(change.frame_path).name,
    "source_frames": {
        "prev": Path(change.prev_frame_path).name,
        "curr": Path(change.frame_path).name,
    },
    "evidence": EVIDENCE_TEXT,
    "replayable": False,
    "candidate_box": candidate_box,
}
```

`write_close_click_evidence(event, change, out_dir)` opens `change.frame_path`, saves the current frame with `top_right_close_region`, `change_bbox`, and `candidate_box` overlays, and writes `probable_close_click.json` under `out_dir`.

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```bash
uv run pytest -q poc/workflow_3/recording_filter/test_close_click_evidence.py
```

Expected: all positive/negative tests pass.

- [ ] **Step 8: Commit the isolated classifier**

```bash
git add poc/workflow_3/recording_filter/close_click_evidence.py \
  poc/workflow_3/recording_filter/test_close_click_evidence.py
git commit -m "Add probable close click evidence"
```

---

### Task 2: Integrate Evidence into Recording-Filter Outputs

**Files:**
- Modify: `poc/workflow_3/recording_filter/filter_recording.py:240-390`
- Modify: `poc/workflow_3/recording_filter/timeline.py:30-90`
- Modify: `poc/workflow_3/recording_filter/test_filter_recording.py`

**Interfaces:**
- Consumes: `infer_probable_close_click(capture_dir, change_events, click_events) -> dict | None` and `write_close_click_evidence(event, change, out_dir) -> list[Path]` from Task 1.
- Changes: `build_timeline(click_events, typing_events=None, *, gate_info=None, labels=None, inferred_events=None) -> list[dict]`.
- Produces summary field: `probable_close_clicks: int` with value 0 or 1.

- [ ] **Step 1: Write a failing timeline test proving inferred evidence is distinct from click**

```python
def test_timeline_keeps_probable_close_distinct_and_non_replayable():
    inferred = {
        "t_sec": 9.0,
        "seq": 0,
        "action": "probable_close_click",
        "replayable": False,
        "confidence": 0.35,
    }
    timeline = build_timeline([], inferred_events=[inferred])
    assert [event["action"] for event in timeline] == ["probable_close_click"]
    assert timeline[0]["replayable"] is False
```

- [ ] **Step 2: Write a failing end-to-end filter test**

Create a recording fixture with the positive Task 1 frames and manifest. Use a fake VLM client returning `cursor_visible=false`. Disable unrelated stages:

```python
class _NoCursorClient:
    def chat_with_image_b64(self, **kwargs):
        return _FakeResponse(json.dumps({
            "cursor_visible": False,
            "cursor_kind": None,
            "cursor_bbox": None,
            "confidence": 0.0,
            "evidence": "cursor merged with title-bar edge",
        }))


rec, _change, _click = _close_candidate_fixture(tmp_path)
out_dir = rec.parent / "recording_filter"
settings = RecordingFilterSettings(
    vlm_request_delay_sec=0.0,
    min_change_area_px=20,
    region_gate_enabled=False,
    element_label_enabled=False,
    typing_detect_enabled=False,
)
assert run_filter(input_dir=rec, settings=settings, client=_NoCursorClient()) == "success"
timeline = json.loads((out_dir / "interaction_timeline.json").read_text())["events"]
assert [event["action"] for event in timeline] == ["probable_close_click"]
assert (out_dir / "close_click_evidence" / "probable_close_click.json").exists()
summary = json.loads((out_dir / "summary.json").read_text())
assert summary["probable_close_clicks"] == 1
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/recording_filter/test_filter_recording.py::test_timeline_keeps_probable_close_distinct_and_non_replayable \
  poc/workflow_3/recording_filter/test_filter_recording.py::test_run_filter_records_probable_close_click
```

Expected: FAIL because `build_timeline` has no `inferred_events` parameter and `run_filter` does not invoke the classifier.

- [ ] **Step 4: Extend timeline merging without changing click selection**

In `build_timeline`:

```python
for inferred in (inferred_events or []):
    event = dict(inferred)
    event["replayable"] = False
    events.append(event)
```

Keep the existing click condition exactly `ce.status == "click" and ce.is_click`; never convert probable evidence into `ClickEvent`.

- [ ] **Step 5: Invoke inference after Stage 2a and before timeline construction**

In `run_filter`:

```python
from poc.workflow_3.recording_filter.close_click_evidence import (
    infer_probable_close_click,
    write_close_click_evidence,
)

probable_close = infer_probable_close_click(capture_dir, stage1_events, click_events)
inferred_events = [probable_close] if probable_close is not None else []
if probable_close is not None:
    write_close_click_evidence(
        probable_close, stage1_events[-1], out_dir / "close_click_evidence"
    )

timeline = build_timeline(
    click_events,
    typing_events,
    gate_info=gate_info,
    labels=labels,
    inferred_events=inferred_events,
)
```

Add `"probable_close_clicks": len(inferred_events)` to `summary.json`. Because the timeline may now contain evidence without a true click, retain return status `success` when the timeline is non-empty.

> **As-built (2026-08-13):** inference reads `stage1_events` (the raw Stage 1 list), not the
> gate-surviving `change_events` shown in earlier drafts of this step. The real last event of a
> recording may have been dropped as `ambient`/occluded or cut by the Stage 2a cap; walking the
> survivor list instead would promote an older top-right candidate to "terminal". The classifier
> fails closed when the exact rank has no cursor result, so the wider input does not loosen any gate.
>
> This decoupling is why the `success` sentence above needed enforcing rather than just stating:
> "gate discarded everything" and "timeline is empty" are no longer the same condition, so
> `run_filter` must check `timeline` itself before returning `all_events_discarded`. Shipped in
> `filter_recording.py:514` with regression test
> `test_run_filter_reports_success_when_only_inferred_event_survives`.

- [ ] **Step 6: Run recording-filter regressions and verify GREEN**

Run:

```bash
uv run pytest -q \
  poc/workflow_3/recording_filter/test_close_click_evidence.py \
  poc/workflow_3/recording_filter/test_filter_recording.py \
  poc/workflow_3/recording_filter/test_click_detect.py \
  poc/workflow_3/recording_filter/test_cursor_prompt.py
```

Expected: all pass; existing true-click timelines remain unchanged.

- [ ] **Step 7: Commit pipeline integration**

```bash
git add poc/workflow_3/recording_filter/filter_recording.py \
  poc/workflow_3/recording_filter/timeline.py \
  poc/workflow_3/recording_filter/test_filter_recording.py
git commit -m "Record close click fallback evidence"
```

---

### Task 3: Document the Evidence-Only Safety Boundary

**Files:**
- Modify: `poc/workflow_3/README.md`
- Test: static repository checks

**Interfaces:**
- Documents: `probable_close_click` is inferred, low confidence, and non-replayable.

- [ ] **Step 1: Add the operator-facing explanation**

Under recording-filter outputs, add:

```markdown
- `probable_close_click`: `recording_manifest.json`이 `window_gone`이고, 마지막 변화가
  우상단이며, cursor VLM이 해당 프레임에서 커서를 못 찾은 경우에만 기록하는
  low-confidence 사후 증거입니다. 원격 화면 가장자리에서 아래쪽 절반만 보이는 X/커서를
  복원하기 위한 last resort이며 `replayable=false`입니다. 이 이벤트는 live 완료 판정,
  자동 클릭, workflow 재생에 사용하지 않습니다.
```

- [ ] **Step 2: Verify no live layer imports the inference module**

Run:

```bash
rg -n "close_click_evidence|probable_close_click" \
  poc/workflow_3/monitor poc/workflow_3/rcs poc/workflow_3/sem_monitor
```

Expected: no Python imports or action handling in live layers. A README mention is acceptable.

> **As-built (2026-08-13):** the live-layer sweep above is necessary but not sufficient. Extend it to
> the offline consumer, which must not render inferred evidence as a performed step:
>
> ```bash
> rg -n "replayable|probable_close_click" poc/workflow_3/workflow_extract
> ```
>
> Expected: `extract_workflow.py` and `grouping.py` both filter on `replayable is False` before an
> event can become a procedure action. "Evidence-only" is a property of the whole chain from
> timeline to Korean procedure, not of `recording_filter` alone.

- [ ] **Step 3: Run full focused verification**

Run:

```bash
uv run pytest -q poc/workflow_3/recording_filter/
git diff --check
```

Expected: all recording-filter tests pass and diff check is silent.

- [ ] **Step 4: Commit documentation**

```bash
git add poc/workflow_3/README.md
git commit -m "Document probable close click evidence"
```

---

## Final Verification

- [ ] Run the complete recording-filter suite:

```bash
uv run pytest -q poc/workflow_3/recording_filter/
```

- [ ] Confirm positive and negative gates directly:

```bash
uv run pytest -q poc/workflow_3/recording_filter/test_close_click_evidence.py -v
```

- [ ] Confirm the new action is never treated as a true click:

```bash
rg -n 'action.*probable_close_click|probable_close_click.*action|is_click.*probable' \
  poc/workflow_3 --glob '*.py'
```

Expected: construction/serialization/tests only; no `is_click=True`, GUI action, or replay branch.

- [ ] Confirm manifest gating remains exact:

```bash
rg -n 'stop_reason.*window_gone|window_gone.*stop_reason' \
  poc/workflow_3/recording_filter/close_click_evidence.py
```

Expected: classifier explicitly requires `window_gone`.

- [ ] Confirm repository hygiene:

```bash
git status --short
git diff --check
```

- [ ] Office validation is evidence-only: close Remote Monitoring with X, run the recording filter, and inspect `recording_filter/close_click_evidence/`. Failure to infer must not affect engineer-done or teardown; a successful inference must remain `confidence=0.35` and `replayable=false`.
