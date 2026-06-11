# recording_filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `poc/workflow_3/recording_filter/`, an offline on-demand tool that reduces a `recording/` frame folder to change-events (cv2) and extracts engineer mouse-clicks (VLM cursor + cv2 ROI), emitting reduced frames plus an `interaction_timeline.json`.

**Architecture:** Two-stage cheap→expensive cascade. Stage 1 (`frame_reduce`, pure cv2) prunes thousands of frames to dozens of "change events", each carrying a native-pixel `change_bbox`. Stage 2a (`click_detect`, VLM + cv2) runs only on survivors: one VLM cursor-locate call per frame, then a cv2 check of whether the change sits inside an ROI around the cursor. `timeline` merges click events into an ordered JSON timeline (with typing fields reserved for a deferred Stage 2b). `filter_recording` orchestrates and writes artifacts. Ported from `poc/workflow_2/filter_frames_by_change.py` + `poc/workflow_2/vlm_cursor_click_filter.py`.

**Tech Stack:** Python 3.10+, OpenCV (`cv2`), NumPy, Pillow, `uv`, pytest. VLM via `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient` (ui-venus proxy). No CLI args (project rule); config via `RecordingFilterSettings` + env.

**Spec:** `poc/workflow_3/docs/superpowers/specs/2026-06-11-recording-filter-design.md`

---

## Conventions (apply to every task)

- Korean docstrings; `[INFO]`/`[WARNING]`/`[ERROR]` print prefixes (never the `logging` module); no em-dash (U+2014) inside `print()`.
- Absolute imports: `from poc.workflow_3.recording_filter import ...`.
- Tests are pytest-style (`tmp_path`, plain `assert`), run with `uv run pytest <path> -v`.
- Commit after each task (directly to `main`, no branch). End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## File Structure

```
poc/workflow_3/recording_filter/
├─ __init__.py            # public exports
├─ settings.py            # RecordingFilterSettings + load_recording_filter_settings()   [Task 1]
├─ frame_reduce.py        # ChangeEvent + reduce_frames() + frame helpers                [Task 2]
├─ cursor_prompt.py       # cursor_system_prompt() / cursor_user_prompt()                [Task 3]
├─ click_detect.py        # ClickEvent + detect_clicks()                                 [Task 4]
├─ timeline.py            # build_timeline() + write_click_overlays()                     [Task 5]
├─ filter_recording.py    # run_filter() orchestrator + __main__ entry                    [Task 6]
├─ test_settings.py       [Task 1]
├─ test_frame_reduce.py   [Task 2]
├─ test_cursor_prompt.py  [Task 3]
├─ test_click_detect.py   [Task 4]
├─ test_timeline.py       [Task 5]
└─ test_filter_recording.py [Task 6]
```
`text_diff.py` (Stage 2b OCR-diff typing) is intentionally NOT in this plan — deferred.

---

## Task 1: Package scaffold + settings

**Files:**
- Create: `poc/workflow_3/recording_filter/__init__.py`
- Create: `poc/workflow_3/recording_filter/settings.py`
- Test: `poc/workflow_3/recording_filter/test_settings.py`

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_settings.py`:

```python
"""RecordingFilterSettings 기본값 + env override 테스트."""

from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)


def test_defaults_match_spec():
    s = RecordingFilterSettings()
    assert s.diff_threshold == 25
    assert s.resize_width == 1280
    assert s.min_change_area_px == 5000
    assert s.cursor_click_window_px == 200
    assert s.click_min_changed_px == 1500
    assert s.vlm_service == "ui-venus"
    assert s.vlm_request_delay_sec == 1.0
    assert s.max_vlm_calls == 0


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RECORDING_FILTER_MIN_CHANGE_AREA_PX", "9000")
    monkeypatch.setenv("RECORDING_FILTER_CLICK_WINDOW_PX", "120")
    monkeypatch.setenv("RECORDING_FILTER_VLM_REQUEST_DELAY_SEC", "0")
    monkeypatch.setenv("RECORDING_FILTER_MAX_VLM_CALLS", "5")
    s = load_recording_filter_settings()
    assert s.min_change_area_px == 9000
    assert s.cursor_click_window_px == 120
    assert s.vlm_request_delay_sec == 0.0
    assert s.max_vlm_calls == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.workflow_3.recording_filter'`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/__init__.py`:

```python
"""recording_filter — 녹화 프레임 필터 + 상호작용 타임라인 (오프라인 온디맨드).

RecordingSession(monitor/recording.py) 이 남긴 tool 창 녹화 프레임을
(1) cv2 변화 이벤트로 축소하고 (2) VLM 커서 탐지로 클릭을 추출해
interaction_timeline.json 으로 만든다. 자세한 설계는
docs/superpowers/specs/2026-06-11-recording-filter-design.md 참고.
"""

from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)

__all__ = [
    "RecordingFilterSettings",
    "load_recording_filter_settings",
]
```

Create `poc/workflow_3/recording_filter/settings.py`:

```python
"""recording_filter 실행 파라미터 — env 주도 dataclass (CLI 인자 없음)."""

from dataclasses import dataclass, field

from poc.workflow_3.util import env_float, env_int
from poc.workflow_3.vlm.flask_vlm import UI_VENUS_MODEL_NAME

# 기본 VLM service slug.
_DEFAULT_SERVICE = "ui-venus"


@dataclass
class RecordingFilterSettings:
    """필터 파이프라인 튜닝 파라미터 (bench 상수 이식)."""

    # ---- Stage 1: cv2 프레임 축소 ----
    diff_threshold: int = 25            # absdiff 이진화 임계
    resize_width: int = 1280            # diff 계산용 다운스케일 폭
    min_change_area_px: int = 5000      # 가장 큰 변화 blob 면적 임계(생존 조건)
    # ---- Stage 2a: 커서-기하 클릭 ----
    cursor_click_window_px: int = 200   # 커서 중심 정사각 ROI 한 변
    click_min_changed_px: int = 1500    # ROI 안 변화 픽셀 임계(클릭 조건)
    click_diff_threshold: int = 25      # native diff 마스크 임계
    # ---- VLM ----
    vlm_service: str = _DEFAULT_SERVICE
    vlm_model: str = field(default_factory=lambda: UI_VENUS_MODEL_NAME)
    vlm_request_delay_sec: float = 1.0  # 프록시 과부하 방지 간격
    max_vlm_calls: int = 0              # 0 = 생존 전체 처리(샘플링 없음)


def load_recording_filter_settings() -> RecordingFilterSettings:
    """env override 를 적용한 설정을 만든다."""
    return RecordingFilterSettings(
        diff_threshold=env_int("RECORDING_FILTER_DIFF_THRESHOLD", 25),
        resize_width=env_int("RECORDING_FILTER_RESIZE_WIDTH", 1280),
        min_change_area_px=env_int("RECORDING_FILTER_MIN_CHANGE_AREA_PX", 5000),
        cursor_click_window_px=env_int("RECORDING_FILTER_CLICK_WINDOW_PX", 200),
        click_min_changed_px=env_int("RECORDING_FILTER_CLICK_MIN_CHANGED_PX", 1500),
        click_diff_threshold=env_int("RECORDING_FILTER_CLICK_DIFF_THRESHOLD", 25),
        vlm_request_delay_sec=env_float("RECORDING_FILTER_VLM_REQUEST_DELAY_SEC", 1.0),
        max_vlm_calls=env_int("RECORDING_FILTER_MAX_VLM_CALLS", 0),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/recording_filter/test_settings.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/recording_filter/__init__.py poc/workflow_3/recording_filter/settings.py poc/workflow_3/recording_filter/test_settings.py
git commit -m "feat(recording_filter): package scaffold + RecordingFilterSettings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Stage 1 — frame_reduce (cv2 change events + native bbox)

**Files:**
- Create: `poc/workflow_3/recording_filter/frame_reduce.py`
- Test: `poc/workflow_3/recording_filter/test_frame_reduce.py`

**Note:** ports `poc/workflow_2/filter_frames_by_change.py`, adding `change_bbox` (largest-blob bounding box scaled to native pixels via `connectedComponentsWithStats` `CC_STAT_LEFT/TOP/WIDTH/HEIGHT`).

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_frame_reduce.py`:

```python
"""frame_reduce 합성 프레임 테스트 — 변화 blob 주입 → 생존 + native bbox."""

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.frame_reduce import (
    ChangeEvent,
    collect_frame_paths,
    reduce_frames,
)
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


def _write_frame(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, format="JPEG", quality=95)


def _base(h=400, w=600, value=30):
    return np.full((h, w), value, dtype=np.uint8)


def test_reduce_keeps_only_changed_frames(tmp_path):
    # 3장: f0 (base), f1 (큰 흰 사각형 = 변화), f2 (f1 과 동일 = 변화 없음)
    f0 = _base()
    f1 = _base()
    f1[100:200, 150:300] = 255            # 100x150 = 15000px 변화 blob
    f2 = f1.copy()
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)
    _write_frame(tmp_path / "rec_rcs_0002_00000600ms.jpg", f2)

    settings = RecordingFilterSettings(min_change_area_px=5000)
    events = reduce_frames(tmp_path, settings)

    # f0->f1 은 큰 변화로 생존, f1->f2 는 변화 없어 탈락.
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ChangeEvent)
    assert ev.rank == 0
    assert ev.frame_path.endswith("rec_rcs_0001_00000300ms.jpg")
    assert ev.timestamp_sec == 0.3


def test_change_bbox_is_native_pixels(tmp_path):
    f0 = _base()
    f1 = _base()
    f1[100:200, 150:300] = 255
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)

    # resize_width 가 native(600) 보다 크면 다운스케일 없음 -> bbox 가 native 와 정합.
    settings = RecordingFilterSettings(min_change_area_px=5000, resize_width=4000)
    events = reduce_frames(tmp_path, settings)
    bbox = events[0].change_bbox
    # dilate(5x5, 2회) 로 약간 팽창하므로 여유 두고 검증.
    assert 130 <= bbox["left"] <= 160
    assert 80 <= bbox["top"] <= 110
    assert 290 <= bbox["right"] <= 320
    assert 190 <= bbox["bottom"] <= 220


def test_below_threshold_dropped(tmp_path):
    f0 = _base()
    f1 = _base()
    f1[10:15, 10:15] = 255                # 25px << 5000 임계
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)
    events = reduce_frames(tmp_path, RecordingFilterSettings(min_change_area_px=5000))
    assert events == []


def test_collect_frame_paths_sorted(tmp_path):
    for name in ["rec_rcs_0002_x.jpg", "rec_rcs_0000_x.jpg", "rec_rcs_0001_x.jpg"]:
        _write_frame(tmp_path / name, _base())
    paths = collect_frame_paths(tmp_path)
    assert [p.name for p in paths] == [
        "rec_rcs_0000_x.jpg",
        "rec_rcs_0001_x.jpg",
        "rec_rcs_0002_x.jpg",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_frame_reduce.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'reduce_frames'`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/frame_reduce.py`:

```python
"""STAGE 1 — cv2 absdiff 변화 이벤트로 녹화 프레임을 축소한다.

poc/workflow_2/filter_frames_by_change.py 의 엔진을 이식하면서, 가장 큰 변화
blob 의 *위치* bbox(change_bbox)를 native 픽셀로 함께 산출한다(현재 bench 는
면적만 계산하고 위치를 버린다). 이 bbox 는 Stage 2a 의 ROI 참고치이자
향후 Stage 2b(OCR-diff) 의 crop 영역으로 재사용된다.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2

from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


@dataclass
class ChangeEvent:
    """Stage 1 생존 프레임 1건."""

    rank: int                 # 0부터, 생존 순서
    frame_path: str           # 현재(curr) 프레임 절대경로
    prev_frame_path: str      # 직전 프레임 절대경로
    timestamp_sec: float      # 파일명 <elapsed_ms> 복원
    frame_index: int          # 파일명 seq (없으면 -1)
    change_bbox: dict         # native px {left,top,right,bottom}
    largest_blob_area_px: int
    changed_pixels: int


def _parse_timestamp_sec(frame_path: Path) -> float:
    """파일명 끝의 <ms> 토큰에서 초 단위 타임스탬프를 파싱한다."""
    for part in reversed(frame_path.stem.split("_")):
        if part.endswith("ms") and part[:-2].isdigit():
            return round(int(part[:-2]) / 1000.0, 3)
    return 0.0


def _parse_frame_index(frame_path: Path) -> int:
    """파일명에서 seq 정수를 파싱한다(<tag>_rcs_<seq>_...). 실패 시 -1."""
    parts = frame_path.stem.split("_")
    for i, part in enumerate(parts):
        if part == "rcs" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """frames 디렉터리의 JPEG 를 파일명 정렬 순으로 반환한다."""
    return sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )


def _load_diff_gray(image_path: Path, resize_width: int):
    """grayscale 로 로드하고 resize_width 로 다운스케일한다.

    반환: (resized_gray, native_w, native_h) 또는 None(로드 실패).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    native_h, native_w = image.shape[:2]
    if resize_width > 0 and native_w > resize_width:
        new_h = max(1, int(round(native_h * (resize_width / native_w))))
        image = cv2.resize(image, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return image, native_w, native_h


def _largest_blob_stats(dilated):
    """dilate 된 이진 마스크에서 (면적, bbox) 를 반환한다. 변화 없으면 (0, zero-bbox)."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    best_label, best_area = -1, 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area, best_label = area, label
    if best_label < 0:
        return 0, {"left": 0, "top": 0, "right": 0, "bottom": 0}
    left = int(stats[best_label, cv2.CC_STAT_LEFT])
    top = int(stats[best_label, cv2.CC_STAT_TOP])
    width = int(stats[best_label, cv2.CC_STAT_WIDTH])
    height = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return best_area, {"left": left, "top": top, "right": left + width, "bottom": top + height}


def _compute_change(prev_gray, curr_gray, diff_threshold: int):
    """두 grayscale 프레임의 (changed_px, blob_area, blob_bbox(diff-space)) 를 구한다."""
    if prev_gray.shape != curr_gray.shape:
        target = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target, interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    changed_px = int(cv2.countNonZero(dilated))
    area, bbox = _largest_blob_stats(dilated)
    diff_h, diff_w = dilated.shape[:2]
    return changed_px, area, bbox, diff_w, diff_h


def _scale_bbox(bbox: dict, native_w: int, native_h: int, diff_w: int, diff_h: int) -> dict:
    """diff-space bbox 를 native 픽셀로 스케일한다."""
    sx = native_w / diff_w if diff_w else 1.0
    sy = native_h / diff_h if diff_h else 1.0
    return {
        "left": int(round(bbox["left"] * sx)),
        "top": int(round(bbox["top"] * sy)),
        "right": int(round(bbox["right"] * sx)),
        "bottom": int(round(bbox["bottom"] * sy)),
    }


def reduce_frames(frames_dir: Path, settings: RecordingFilterSettings) -> list[ChangeEvent]:
    """인접 프레임 변화로 ChangeEvent 목록을 만든다(변화 없는 프레임은 탈락)."""
    frames_dir = Path(frames_dir)
    frame_paths = collect_frame_paths(frames_dir)
    if len(frame_paths) < 2:
        print(f"[WARNING] 변화 비교에 최소 2장 필요: found={len(frame_paths)} in {frames_dir}")
        return []

    events: list[ChangeEvent] = []
    loaded = _load_diff_gray(frame_paths[0], settings.resize_width)
    if loaded is None:
        print(f"[WARNING] 첫 프레임 로드 실패: {frame_paths[0]}")
        return []
    prev_gray = loaded[0]
    rank = 0
    for prev_path, curr_path in zip(frame_paths[:-1], frame_paths[1:]):
        loaded = _load_diff_gray(curr_path, settings.resize_width)
        if loaded is None:
            print(f"[WARNING] 프레임 로드 실패: {curr_path}")
            prev_gray = None
            continue
        curr_gray, native_w, native_h = loaded
        if prev_gray is None:
            prev_gray = curr_gray
            continue

        changed_px, area, bbox_diff, diff_w, diff_h = _compute_change(
            prev_gray, curr_gray, settings.diff_threshold
        )
        if area >= settings.min_change_area_px:
            events.append(
                ChangeEvent(
                    rank=rank,
                    frame_path=str(curr_path.resolve()),
                    prev_frame_path=str(prev_path.resolve()),
                    timestamp_sec=_parse_timestamp_sec(curr_path),
                    frame_index=_parse_frame_index(curr_path),
                    change_bbox=_scale_bbox(bbox_diff, native_w, native_h, diff_w, diff_h),
                    largest_blob_area_px=area,
                    changed_pixels=changed_px,
                )
            )
            rank += 1
        prev_gray = curr_gray

    print(f"[INFO] Stage 1 완료: change_events={len(events)} / pairs={len(frame_paths) - 1}")
    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/recording_filter/test_frame_reduce.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/recording_filter/frame_reduce.py poc/workflow_3/recording_filter/test_frame_reduce.py
git commit -m "feat(recording_filter): Stage 1 frame_reduce with native change_bbox

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: cursor_prompt (ported, self-contained)

**Files:**
- Create: `poc/workflow_3/recording_filter/cursor_prompt.py`
- Test: `poc/workflow_3/recording_filter/test_cursor_prompt.py`

**Note:** verbatim port of `_cursor_system_prompt` / `_cursor_user_prompt` from `poc/workflow_2/vlm_cursor_click_filter.py` (renamed public). Keeps the 3-variant cursor description (DVR-X / RCS-black-arrow / RCS-white-arrow).

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_cursor_prompt.py`:

```python
"""cursor_prompt 가 3-변형 커서 + 출력 스키마 키를 담는지 가드한다."""

from poc.workflow_3.recording_filter.cursor_prompt import (
    cursor_system_prompt,
    cursor_user_prompt,
)


def test_system_prompt_mentions_three_cursor_variants():
    sys = cursor_system_prompt()
    assert "DVR" in sys
    assert "RCS" in sys
    assert "SEM Monitor" in sys


def test_user_prompt_declares_json_schema():
    user = cursor_user_prompt()
    for key in ("cursor_visible", "cursor_kind", "cursor_bbox", "confidence"):
        assert key in user
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_cursor_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/cursor_prompt.py`:

```python
"""CD-SEM 환경 마우스 커서 coarse 탐지 프롬프트 (vlm_cursor_click_filter 에서 이식).

DVR / RCS / RCS-on-SEM-Monitor 세 가지 커서 변형을 모두 인지하도록 설명한다.
"""


def cursor_system_prompt() -> str:
    """커서 coarse 탐지 시스템 프롬프트."""
    return (
        "You locate the mouse cursor inside a screenshot of CD-SEM tooling. "
        "The cursor can appear in one of three forms depending on which window the pointer is over:\n"
        "  1) DVR camera feed: a small black 'X' (crosshair) glyph, ~10-20 px on each side.\n"
        "  2) RCS application (default Windows pointer): a small black arrow with a thin white outline.\n"
        "  3) RCS SEM Monitor box (the dark live-SEM image area): the same arrow inverted to "
        "white with a thin black outline so it stays visible against the dark background.\n"
        "Return strict JSON only. Locate ONLY the mouse cursor; do not confuse it with similar-looking "
        "but static UI artifacts such as SEM crosshair reticles, alignment-key marks, toolbar icon glyphs, "
        "or measurement annotations. A real cursor sits on top of the underlying content, is small "
        "(typically 12-32 px on a side), and never has anti-aliased text or numbers attached to it. "
        "If no cursor is visible, say so."
    )


def cursor_user_prompt() -> str:
    """커서 coarse 탐지 사용자 프롬프트(JSON 스키마 지정)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "cursor_visible": true,\n'
        '  "cursor_kind": "dvr_x | rcs_black_arrow | rcs_white_arrow",\n'
        '  "coord_system": "relative_1000",\n'
        '  "cursor_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0,\n'
        '  "evidence": "short string describing the glyph and where it sits"\n'
        "}\n"
        "The bbox must tightly enclose the entire visible cursor glyph (X for dvr_x, "
        "the full arrow shape for rcs_black_arrow / rcs_white_arrow). "
        "Set cursor_kind to whichever of the three variants you actually see; if you cannot tell, "
        'set it to "unknown". '
        "If no cursor is visible, set cursor_visible=false, cursor_bbox=null, and cursor_kind=null."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/recording_filter/test_cursor_prompt.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/recording_filter/cursor_prompt.py poc/workflow_3/recording_filter/test_cursor_prompt.py
git commit -m "feat(recording_filter): port CD-SEM cursor-locate prompts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Stage 2a — click_detect (VLM cursor + cv2 ROI)

**Files:**
- Create: `poc/workflow_3/recording_filter/click_detect.py`
- Test: `poc/workflow_3/recording_filter/test_click_detect.py`

**Note:** ports the detection core of `poc/workflow_2/vlm_cursor_click_filter.py`. The VLM client is **injected** (`detect_clicks(..., client=...)`) so tests run offline. The client must expose `chat_with_image_b64(*, image_b64, system_message, user_text, image_mime, temperature) -> obj` where `obj.text` is the JSON string.

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_click_detect.py`:

```python
"""click_detect 테스트 — 가짜 VLM client + 합성 변화로 클릭 판정."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.click_detect import ClickEvent, detect_clicks
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """고정 커서 bbox(0-1000) 를 반환하거나, raise_exc 로 실패를 흉내낸다."""

    def __init__(self, cursor_bbox_1000=None, visible=True, raise_exc=False):
        self.cursor_bbox_1000 = cursor_bbox_1000 or {"left": 480, "top": 230, "right": 520, "bottom": 270}
        self.visible = visible
        self.raise_exc = raise_exc
        self.calls = 0

    def chat_with_image_b64(self, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError("vlm down")
        payload = {
            "cursor_visible": self.visible,
            "cursor_kind": "rcs_black_arrow" if self.visible else None,
            "cursor_bbox": self.cursor_bbox_1000 if self.visible else None,
            "confidence": 0.9,
            "evidence": "fake",
        }
        return _FakeResponse(json.dumps(payload))


def _make_pair(tmp_path, change_box):
    """change_box(L,T,R,B) 영역만 다른 prev/curr 프레임 쌍을 쓰고 ChangeEvent 를 만든다."""
    h, w = 400, 600
    prev = np.full((h, w), 30, dtype=np.uint8)
    curr = prev.copy()
    l, t, r, b = change_box
    curr[t:b, l:r] = 255
    prev_path = tmp_path / "rec_rcs_0000_00000000ms.jpg"
    curr_path = tmp_path / "rec_rcs_0001_00000300ms.jpg"
    Image.fromarray(prev, mode="L").save(prev_path, format="JPEG", quality=95)
    Image.fromarray(curr, mode="L").save(curr_path, format="JPEG", quality=95)
    return ChangeEvent(
        rank=0,
        frame_path=str(curr_path.resolve()),
        prev_frame_path=str(prev_path.resolve()),
        timestamp_sec=0.3,
        frame_index=1,
        change_bbox={"left": l, "top": t, "right": r, "bottom": b},
        largest_blob_area_px=(r - l) * (b - t),
        changed_pixels=(r - l) * (b - t),
    )


def _settings():
    return RecordingFilterSettings(vlm_request_delay_sec=0.0, click_min_changed_px=1500)


def test_change_near_cursor_is_click(tmp_path):
    # 커서 bbox 1000 중심 (500,250) -> px (300,125) 부근. 변화도 그 근처.
    ev = _make_pair(tmp_path, change_box=(250, 80, 360, 180))
    client = _FakeClient(cursor_bbox_1000={"left": 480, "top": 180, "right": 520, "bottom": 220})
    out = detect_clicks([ev], _settings(), client=client)
    assert len(out) == 1
    assert isinstance(out[0], ClickEvent)
    assert out[0].is_click is True
    assert out[0].status == "click"
    assert out[0].cursor_xy is not None


def test_change_far_from_cursor_is_no_click(tmp_path):
    # 변화는 좌상단, 커서는 우하단 -> ROI 안 변화 없음.
    ev = _make_pair(tmp_path, change_box=(0, 0, 110, 110))
    client = _FakeClient(cursor_bbox_1000={"left": 950, "top": 950, "right": 990, "bottom": 990})
    out = detect_clicks([ev], _settings(), client=client)
    assert out[0].is_click is False
    assert out[0].status == "no_click"


def test_cursor_not_visible_is_no_click(tmp_path):
    ev = _make_pair(tmp_path, change_box=(250, 80, 360, 180))
    client = _FakeClient(visible=False)
    out = detect_clicks([ev], _settings(), client=client)
    assert out[0].is_click is False
    assert out[0].cursor_visible is False


def test_vlm_exception_marks_cursor_unavailable_and_survives(tmp_path):
    ev = _make_pair(tmp_path, change_box=(250, 80, 360, 180))
    client = _FakeClient(raise_exc=True)
    out = detect_clicks([ev], _settings(), client=client)
    assert len(out) == 1
    assert out[0].status == "cursor_unavailable"
    assert out[0].is_click is False


def test_max_vlm_calls_truncates(tmp_path):
    ev = _make_pair(tmp_path, change_box=(250, 80, 360, 180))
    events = [ev, ev, ev]
    client = _FakeClient()
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, max_vlm_calls=2)
    out = detect_clicks(events, settings, client=client)
    assert len(out) == 2          # 캡에서 중단
    assert client.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_click_detect.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'detect_clicks'`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/click_detect.py`:

```python
"""STAGE 2a — VLM 커서 탐지 + cv2 ROI 변화로 마우스 클릭을 추출한다.

poc/workflow_2/vlm_cursor_click_filter.py 의 탐지 코어를 이식한다. 생존 프레임마다
VLM 커서 coarse bbox 를 1회 얻고, 커서 중심 정사각 ROI 안의 native 변화 픽셀이
임계 이상이면 클릭으로 본다. VLM client 는 주입(injection)이라 테스트는 오프라인.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from poc.workflow_3.recording_filter.cursor_prompt import (
    cursor_system_prompt,
    cursor_user_prompt,
)
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.util import encode_image_webp
from poc.workflow_3.util.json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    extract_json,
    normalize_bbox_1000,
)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


@dataclass
class ClickEvent:
    """Stage 2a 결과 1건 (ChangeEvent 확장)."""

    change: ChangeEvent
    is_click: bool
    status: str                 # click | no_click | cursor_unavailable
    cursor_visible: bool
    cursor_kind: str | None
    cursor_bbox: dict | None    # native px
    cursor_xy: list | None      # [x, y]
    click_window: dict | None
    changed_in_window_px: int
    confidence: float
    evidence: str

    # 편의 접근자 (timeline 에서 사용).
    @property
    def frame_path(self) -> str:
        return self.change.frame_path

    @property
    def prev_frame_path(self) -> str:
        return self.change.prev_frame_path

    @property
    def timestamp_sec(self) -> float:
        return self.change.timestamp_sec

    @property
    def rank(self) -> int:
        return self.change.rank


def _diff_mask(prev_path: Path, curr_path: Path, threshold: int):
    """native 해상도에서 prev/curr 변화 마스크(dilate 이진)를 만든다."""
    prev_gray = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr_gray = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
    if prev_gray is None or curr_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        target = (curr_gray.shape[1], curr_gray.shape[0])
        prev_gray = cv2.resize(prev_gray, target, interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(thresh, kernel, iterations=2)


def _window_around(cx: int, cy: int, side: int, img_w: int, img_h: int) -> dict:
    """커서 중심을 둘러싼 정사각 ROI bbox 를 만든다."""
    half = max(1, side // 2)
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(img_w, cx + half)
    bottom = min(img_h, cy + half)
    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
    return {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)}


def _count_changed_in_window(mask, window: dict) -> int:
    """변화 마스크 안에서 window 영역의 변화 픽셀 수를 센다."""
    crop = mask[window["top"]:window["bottom"], window["left"]:window["right"]]
    if crop.size == 0:
        return 0
    return int(cv2.countNonZero(crop))


def _locate_cursor(client, frame_path: Path):
    """프레임에서 커서 coarse bbox(native px) 를 탐지한다.

    반환: (parsed_dict, cursor_px_bbox|None, img_w, img_h).
    """
    image = Image.open(frame_path).convert("RGB")
    image_b64, width, height = encode_image_webp(image)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=cursor_system_prompt(),
        user_text=cursor_user_prompt(),
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get("cursor_visible") is not True:
        return parsed, None, width, height
    bbox_1000 = normalize_bbox_1000(parsed.get("cursor_bbox"))
    if bbox_1000 is None:
        return parsed, None, width, height
    return parsed, bbox_1000_to_pixels(bbox_1000, width, height), width, height


def _unavailable_event(change: ChangeEvent) -> ClickEvent:
    return ClickEvent(
        change=change, is_click=False, status="cursor_unavailable",
        cursor_visible=False, cursor_kind=None, cursor_bbox=None, cursor_xy=None,
        click_window=None, changed_in_window_px=0, confidence=0.0, evidence="",
    )


def detect_clicks(
    change_events: list[ChangeEvent],
    settings: RecordingFilterSettings,
    *,
    client,
) -> list[ClickEvent]:
    """생존 프레임마다 커서를 찾아 ROI 변화로 클릭을 판정한다."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")

    results: list[ClickEvent] = []
    calls = 0
    for change in change_events:
        if settings.max_vlm_calls and calls >= settings.max_vlm_calls:
            print(f"[WARNING] max_vlm_calls={settings.max_vlm_calls} 도달 -> 이후 생존 분류 중단")
            break

        try:
            parsed, cursor_px, width, height = _locate_cursor(client, Path(change.frame_path))
            calls += 1
        except Exception as exc:
            calls += 1
            print(f"[WARNING] 커서 탐지 실패(cursor_unavailable): {change.frame_path}: {exc}")
            results.append(_unavailable_event(change))
            _sleep(settings.vlm_request_delay_sec)
            continue

        if cursor_px is None:
            results.append(
                ClickEvent(
                    change=change, is_click=False, status="no_click",
                    cursor_visible=False, cursor_kind=parsed.get("cursor_kind"),
                    cursor_bbox=None, cursor_xy=None, click_window=None,
                    changed_in_window_px=0,
                    confidence=float(parsed.get("confidence") or 0.0),
                    evidence=str(parsed.get("evidence") or ""),
                )
            )
            _sleep(settings.vlm_request_delay_sec)
            continue

        center = bbox_center(cursor_px)
        mask = _diff_mask(
            Path(change.prev_frame_path), Path(change.frame_path), settings.click_diff_threshold
        )
        if mask is None:
            results.append(_unavailable_event(change))
            _sleep(settings.vlm_request_delay_sec)
            continue
        window = _window_around(center["x"], center["y"], settings.cursor_click_window_px, width, height)
        changed = _count_changed_in_window(mask, window)
        is_click = changed >= settings.click_min_changed_px
        results.append(
            ClickEvent(
                change=change, is_click=is_click,
                status="click" if is_click else "no_click",
                cursor_visible=True, cursor_kind=parsed.get("cursor_kind"),
                cursor_bbox=cursor_px, cursor_xy=[center["x"], center["y"]],
                click_window=window, changed_in_window_px=changed,
                confidence=float(parsed.get("confidence") or 0.0),
                evidence=str(parsed.get("evidence") or ""),
            )
        )
        _sleep(settings.vlm_request_delay_sec)

    n_click = sum(1 for r in results if r.is_click)
    print(f"[INFO] Stage 2a 완료: clicks={n_click} / processed={len(results)}")
    return results


def _sleep(delay_sec: float) -> None:
    """프록시 과부하 방지 대기(테스트는 0 으로 무효화)."""
    if delay_sec and delay_sec > 0:
        time.sleep(delay_sec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/recording_filter/test_click_detect.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/recording_filter/click_detect.py poc/workflow_3/recording_filter/test_click_detect.py
git commit -m "feat(recording_filter): Stage 2a click_detect (VLM cursor + cv2 ROI)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: timeline (ordered InteractionEvent JSON + overlays)

**Files:**
- Create: `poc/workflow_3/recording_filter/timeline.py`
- Test: `poc/workflow_3/recording_filter/test_timeline.py`

**Note:** `build_timeline(click_events, typing_events=None)` — `typing_events` reserved for the deferred Stage 2b. Only `is_click` events become `action="click"` timeline entries. `write_click_overlays` draws cursor bbox + ROI on click frames via `save_marked_bboxes`.

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_timeline.py`:

```python
"""timeline 테스트 — 시간순 정렬 + 스키마 + 오버레이 생성."""

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.click_detect import ClickEvent
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.timeline import build_timeline, write_click_overlays


def _click_event(tmp_path, rank, t_sec, is_click=True):
    name = f"rec_rcs_{rank:04d}_x.jpg"
    path = tmp_path / name
    Image.fromarray(np.full((400, 600), 30, dtype=np.uint8), mode="L").save(path, format="JPEG")
    change = ChangeEvent(
        rank=rank, frame_path=str(path.resolve()), prev_frame_path=str(path.resolve()),
        timestamp_sec=t_sec, frame_index=rank,
        change_bbox={"left": 0, "top": 0, "right": 10, "bottom": 10},
        largest_blob_area_px=100, changed_pixels=100,
    )
    return ClickEvent(
        change=change, is_click=is_click, status="click" if is_click else "no_click",
        cursor_visible=True, cursor_kind="rcs_black_arrow",
        cursor_bbox={"left": 290, "top": 120, "right": 310, "bottom": 140},
        cursor_xy=[300, 130], click_window={"left": 200, "top": 30, "right": 400, "bottom": 230},
        changed_in_window_px=9000, confidence=0.9, evidence="x",
    )


def test_timeline_sorted_by_time_with_seq(tmp_path):
    events = [
        _click_event(tmp_path, rank=0, t_sec=2.0),
        _click_event(tmp_path, rank=1, t_sec=0.5),
        _click_event(tmp_path, rank=2, t_sec=1.0),
    ]
    timeline = build_timeline(events)
    assert [e["t_sec"] for e in timeline] == [0.5, 1.0, 2.0]
    assert [e["seq"] for e in timeline] == [0, 1, 2]


def test_timeline_schema_and_click_only(tmp_path):
    events = [
        _click_event(tmp_path, rank=0, t_sec=0.5, is_click=True),
        _click_event(tmp_path, rank=1, t_sec=1.0, is_click=False),  # no_click 제외
    ]
    timeline = build_timeline(events)
    assert len(timeline) == 1
    e = timeline[0]
    for key in ("t_sec", "seq", "action", "coords", "element", "text", "confidence", "frame", "source_frames"):
        assert key in e
    assert e["action"] == "click"
    assert e["coords"] == {"x": 300, "y": 130}
    assert e["element"] is None and e["text"] is None


def test_write_click_overlays_creates_files(tmp_path):
    events = [_click_event(tmp_path, rank=0, t_sec=0.5, is_click=True)]
    out_dir = tmp_path / "click_events"
    paths = write_click_overlays(events, out_dir)
    assert len(paths) == 1
    assert paths[0].exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/timeline.py`:

```python
"""클릭 이벤트를 시간순 InteractionEvent 타임라인으로 병합하고 오버레이를 기록한다.

스키마는 미래 타이핑(Stage 2b)과 공용이다(element/text 필드 예약). build_timeline 은
typing_events 인자를 미리 받아 추가 시 재설계가 없도록 한다.
"""

from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def build_timeline(click_events, typing_events=None) -> list[dict]:
    """클릭(+미래 타이핑) 이벤트를 시간순 정렬된 dict 목록으로 만든다."""
    events: list[dict] = []
    for ce in click_events:
        if ce.status != "click" or not ce.is_click:
            continue
        coords = {"x": ce.cursor_xy[0], "y": ce.cursor_xy[1]} if ce.cursor_xy else None
        events.append(
            {
                "t_sec": ce.timestamp_sec,
                "seq": 0,
                "action": "click",
                "coords": coords,
                "element": None,           # 예약: 클릭 위 요소 라벨
                "text": None,              # 예약: 타이핑 텍스트
                "confidence": ce.confidence,
                "frame": Path(ce.frame_path).name,
                "source_frames": {
                    "prev": Path(ce.prev_frame_path).name,
                    "curr": Path(ce.frame_path).name,
                },
            }
        )
    for te in (typing_events or []):
        events.append(te)  # 이미 동일 스키마 dict 라고 가정(Stage 2b).

    events.sort(key=lambda e: e["t_sec"])
    for i, event in enumerate(events):
        event["seq"] = i
    return events


def write_click_overlays(click_events, out_dir: Path) -> list[Path]:
    """클릭 프레임에 커서 bbox + ROI 박스를 그려 별도 폴더에 저장한다."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow 가 필요합니다(PIL import 실패).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ce in click_events:
        if not ce.is_click or ce.cursor_bbox is None:
            continue
        image = Image.open(ce.frame_path).convert("RGB")
        elements = {
            "cursor": {
                "bbox": ce.cursor_bbox,
                "center": {"x": ce.cursor_xy[0], "y": ce.cursor_xy[1]} if ce.cursor_xy else None,
            },
            "roi": {"bbox": ce.click_window},
        }
        colors = {"cursor": "red", "roi": "yellow"}
        out_path = out_dir / f"{ce.rank:03d}_{Path(ce.frame_path).name}"
        save_marked_bboxes(image, elements, colors, out_path)
        written.append(out_path)
    print(f"[INFO] 클릭 오버레이 {len(written)} 장 기록: {out_dir}")
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_3/recording_filter/test_timeline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/recording_filter/timeline.py poc/workflow_3/recording_filter/test_timeline.py
git commit -m "feat(recording_filter): interaction timeline + click overlays

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: filter_recording orchestrator + entry point

**Files:**
- Create: `poc/workflow_3/recording_filter/filter_recording.py`
- Modify: `poc/workflow_3/recording_filter/__init__.py` (export the public API)
- Test: `poc/workflow_3/recording_filter/test_filter_recording.py`

**Note:** `run_filter(*, input_dir=None, settings=None, client=None) -> str`. Resolves input/output, runs Stage 1 + 2a, writes `change_events/`, `change_events.json`, `click_events/`, `interaction_timeline.json`, `summary.json`. Injecting `input_dir` + `client` keeps the e2e test offline. `__main__` calls `run_filter()` with no args (project rule).

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/recording_filter/test_filter_recording.py`:

```python
"""filter_recording e2e — 합성 녹화 폴더 + 가짜 client 로 산출물 검증."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.filter_recording import run_filter
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def chat_with_image_b64(self, **kwargs):
        payload = {
            "cursor_visible": True,
            "cursor_kind": "rcs_black_arrow",
            "cursor_bbox": {"left": 480, "top": 180, "right": 520, "bottom": 220},
            "confidence": 0.9,
            "evidence": "fake",
        }
        return _FakeResponse(json.dumps(payload))


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, format="JPEG", quality=95)


def _recording_dir(tmp_path):
    rec = tmp_path / "tag123" / "recording"
    base = np.full((400, 600), 30, dtype=np.uint8)
    f1 = base.copy()
    f1[80:180, 250:360] = 255              # 커서(px ~300,120) 근처 큰 변화 -> 클릭
    _write(rec / "tag_rcs_0000_00000000ms.jpg", base)
    _write(rec / "tag_rcs_0001_00000300ms.jpg", f1)
    _write(rec / "tag_rcs_0002_00000600ms.jpg", f1.copy())   # 변화 없음 -> 탈락
    return rec


def test_run_filter_produces_artifacts(tmp_path):
    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, min_change_area_px=5000)
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status == "success"

    out_dir = rec.parent / "recording_filter"
    assert (out_dir / "change_events.json").exists()
    assert (out_dir / "interaction_timeline.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "change_events").is_dir()

    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    assert len(timeline["events"]) == 1
    assert timeline["events"][0]["action"] == "click"


def test_run_filter_not_enough_frames(tmp_path):
    rec = tmp_path / "tag" / "recording"
    _write(rec / "tag_rcs_0000_00000000ms.jpg", np.full((400, 600), 30, dtype=np.uint8))
    status = run_filter(input_dir=rec, settings=RecordingFilterSettings(), client=_FakeClient())
    assert status == "not_enough_frames"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_3/recording_filter/test_filter_recording.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/recording_filter/filter_recording.py`:

```python
"""recording_filter 엔트리포인트 — 입력 해석 → Stage 1+2a → 산출물 기록.

CLI 인자 없음(프로젝트 규칙). 입력은 env/모듈상수/자동탐색으로, 산출은 입력의
형제 recording_filter/ 폴더에 쓴다.

실행:
    uv run python poc/workflow_3/recording_filter/filter_recording.py
"""

import os
import shutil
import time
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.debug_artifacts import save_debug_json
from poc.workflow_3.recording_filter.click_detect import detect_clicks
from poc.workflow_3.recording_filter.frame_reduce import collect_frame_paths, reduce_frames
from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)
from poc.workflow_3.recording_filter.timeline import build_timeline, write_click_overlays
from poc.workflow_3.util import format_elapsed_ms

# 분석할 recording/ 폴더를 직접 적어 쓸 수 있다(가장 우선). 비우면 env/자동탐색.
INPUT_DIR_OVERRIDE = r""


def _resolve_input_dir() -> Path | None:
    """분석할 recording/ 폴더를 결정한다(override -> env -> 자동탐색)."""
    override = (INPUT_DIR_OVERRIDE or "").strip()
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            print(f"[INFO] INPUT_DIR_OVERRIDE 사용: {path}")
            return path.resolve()
        print(f"[ERROR] INPUT_DIR_OVERRIDE 디렉터리를 찾지 못했습니다: {path}")
        return None

    env_path = os.getenv("RECORDING_FILTER_INPUT_DIR", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] RECORDING_FILTER_INPUT_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None

    # 등록(captured_img_from_rcs) + 미등록(_unregistered) 두 경로 형태 모두 탐색.
    candidates = sorted(
        [
            *ALIGN_IMAGES_DIR.glob("*/*/*/captured_img_from_rcs/*/recording"),
            *ALIGN_IMAGES_DIR.glob("*/_unregistered/*/recording"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0].resolve()
        print(f"[INFO] 최신 recording/ 자동 선택: {latest}")
        return latest
    print(f"[ERROR] 분석할 recording/ 폴더를 찾지 못했습니다(루트: {ALIGN_IMAGES_DIR}).")
    return None


def _resolve_frames_dir(capture_dir: Path) -> Path | None:
    """실제 JPEG 가 있는 디렉터리를 결정한다(capture_dir 직접 또는 frames/ 하위)."""
    if any(capture_dir.glob("*.jpg")) or any(capture_dir.glob("*.jpeg")):
        return capture_dir
    frames_dir = capture_dir / "frames"
    if frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
        return frames_dir
    print(f"[ERROR] JPEG 프레임이 없습니다: {capture_dir}")
    return None


def _resolve_output_dir(capture_dir: Path) -> Path:
    """산출 폴더를 결정한다(env override -> capture_dir 형제 recording_filter/)."""
    env_out = os.getenv("RECORDING_FILTER_OUTPUT_DIR", "").strip()
    if env_out:
        return Path(env_out).expanduser().resolve()
    return (capture_dir.parent / "recording_filter").resolve()


def _copy_change_events(change_events, out_dir: Path) -> None:
    """Stage 1 생존 프레임을 rank 접두로 복사한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev in change_events:
        src = Path(ev.frame_path)
        dst = out_dir / f"{ev.rank:03d}_{src.name}"
        shutil.copy2(src, dst)


def _change_events_payload(change_events) -> list[dict]:
    return [
        {
            "rank": ev.rank,
            "frame_path": ev.frame_path,
            "prev_frame_path": ev.prev_frame_path,
            "timestamp_sec": ev.timestamp_sec,
            "frame_index": ev.frame_index,
            "change_bbox": ev.change_bbox,
            "largest_blob_area_px": ev.largest_blob_area_px,
            "changed_pixels": ev.changed_pixels,
        }
        for ev in change_events
    ]


def run_filter(*, input_dir=None, settings: RecordingFilterSettings = None, client=None) -> str:
    """필터 파이프라인을 실행하고 상태 문자열을 반환한다."""
    started_at = time.time()
    settings = settings or load_recording_filter_settings()

    capture_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if capture_dir is None:
        return "input_not_found"
    frames_dir = _resolve_frames_dir(capture_dir)
    if frames_dir is None:
        return "frames_not_found"
    if len(collect_frame_paths(frames_dir)) < 2:
        print(f"[ERROR] 변화 비교에 최소 2장 필요: {frames_dir}")
        return "not_enough_frames"

    out_dir = _resolve_output_dir(capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 ----
    change_events = reduce_frames(frames_dir, settings)
    _copy_change_events(change_events, out_dir / "change_events")
    save_debug_json(
        out_dir / "change_events.json",
        {
            "capture_dir": str(capture_dir),
            "frames_dir": str(frames_dir),
            "min_change_area_px": settings.min_change_area_px,
            "diff_threshold": settings.diff_threshold,
            "resize_width": settings.resize_width,
            "events": _change_events_payload(change_events),
        },
    )

    # ---- Stage 2a ----
    if client is None:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.vlm_service, model_name=settings.vlm_model)
    click_events = detect_clicks(change_events, settings, client=client)
    write_click_overlays(
        [ce for ce in click_events if ce.is_click], out_dir / "click_events"
    )
    timeline = build_timeline(click_events)
    save_debug_json(
        out_dir / "interaction_timeline.json",
        {"capture_dir": str(capture_dir), "events": timeline},
    )

    truncated = len(change_events) - len(click_events)
    save_debug_json(
        out_dir / "summary.json",
        {
            "capture_dir": str(capture_dir),
            "output_dir": str(out_dir),
            "total_change_events": len(change_events),
            "processed_for_click": len(click_events),
            "clicks": sum(1 for ce in click_events if ce.is_click),
            "timeline_events": len(timeline),
            "vlm_calls": len(click_events),
            "truncated": truncated > 0,
            "skipped_due_to_cap": max(0, truncated),
            "max_vlm_calls": settings.max_vlm_calls,
            "elapsed": format_elapsed_ms(started_at),
        },
    )

    print(
        f"[INFO] 완료: change_events={len(change_events)}, clicks="
        f"{sum(1 for ce in click_events if ce.is_click)}, out={out_dir}, "
        f"elapsed={format_elapsed_ms(started_at)}"
    )
    return "success" if timeline else "no_clicks"


if __name__ == "__main__":
    result = run_filter()
    raise SystemExit(0 if result in {"success", "no_clicks"} else 1)
```

- [ ] **Step 4: Update `__init__.py` exports**

Replace the body of `poc/workflow_3/recording_filter/__init__.py` with:

```python
"""recording_filter — 녹화 프레임 필터 + 상호작용 타임라인 (오프라인 온디맨드).

RecordingSession(monitor/recording.py) 이 남긴 tool 창 녹화 프레임을
(1) cv2 변화 이벤트로 축소하고 (2) VLM 커서 탐지로 클릭을 추출해
interaction_timeline.json 으로 만든다. 자세한 설계는
docs/superpowers/specs/2026-06-11-recording-filter-design.md 참고.
"""

from poc.workflow_3.recording_filter.click_detect import ClickEvent, detect_clicks
from poc.workflow_3.recording_filter.filter_recording import run_filter
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent, reduce_frames
from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)
from poc.workflow_3.recording_filter.timeline import build_timeline

__all__ = [
    "RecordingFilterSettings",
    "load_recording_filter_settings",
    "ChangeEvent",
    "reduce_frames",
    "ClickEvent",
    "detect_clicks",
    "build_timeline",
    "run_filter",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_3/recording_filter/test_filter_recording.py -v`
Expected: PASS (2 passed)

Run the whole package: `uv run pytest poc/workflow_3/recording_filter/ -v`
Expected: PASS (all tests across all files)

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/recording_filter/filter_recording.py poc/workflow_3/recording_filter/__init__.py poc/workflow_3/recording_filter/test_filter_recording.py
git commit -m "feat(recording_filter): run_filter orchestrator + entry point

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Remove superseded workflow_2 originals

**Files:**
- Delete: `poc/workflow_2/filter_frames_by_change.py`
- Delete: `poc/workflow_2/vlm_cursor_click_filter.py`
- Possibly modify: any file importing them (resolve to workflow_3 or remove).

**Note:** The spec specifies "move", so the originals are deleted after the port. `workflow_1` frozen files are left untouched (the cursor prompt already lives self-contained in the workflow_2 file we ported).

- [ ] **Step 1: Find importers**

Run:
```bash
grep -rn "filter_frames_by_change\|vlm_cursor_click_filter" poc/ --include="*.py" | grep -v "poc/workflow_2/filter_frames_by_change.py\|poc/workflow_2/vlm_cursor_click_filter.py"
```
Expected: no hits outside the two files themselves. If any hit appears, update that import to `from poc.workflow_3.recording_filter import reduce_frames` (or remove the dead reference) before deleting.

- [ ] **Step 2: Delete the two files**

```bash
git rm poc/workflow_2/filter_frames_by_change.py poc/workflow_2/vlm_cursor_click_filter.py
```

- [ ] **Step 3: Verify nothing broke**

Run: `uv run pytest poc/workflow_3/recording_filter/ -v`
Expected: PASS (all tests)

Run (import smoke for the bench package, ensure no dangling import):
```bash
uv run python -c "import poc.workflow_2"
```
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(recording_filter): remove superseded workflow_2 filter drivers

이전 완료 - 엔진은 poc/workflow_3/recording_filter 로 이식됨. 벤치가 필요하면
from poc.workflow_3.recording_filter import reduce_frames 로 가져온다.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final Verification

- [ ] Run the full package test suite:
  `uv run pytest poc/workflow_3/recording_filter/ -v`
  Expected: all green.
- [ ] Confirm the entry point imports cleanly (no network needed for import):
  `uv run python -c "from poc.workflow_3.recording_filter import run_filter; print('ok')"`
  Expected: `ok`.
- [ ] (Office, real data) Point at a real recording and eyeball outputs:
  `RECORDING_FILTER_INPUT_DIR=<...>/captured_img_from_rcs/<tag>/recording uv run python poc/workflow_3/recording_filter/filter_recording.py`
  Then inspect `<tag>/recording_filter/`: `change_events/` count is much smaller than raw frames; `click_events/` overlays land the cursor box on real clicks; `interaction_timeline.json` events are time-ordered; `summary.json` `truncated=false`.

## Notes for the implementer

- **Why two diff computations:** Stage 1 diffs on a *downscaled* gray (speed over thousands of frames) and scales the blob bbox back to native; Stage 2a re-diffs at *native* resolution because the ROI is sized in native pixels around the VLM cursor. This intentional duplication keeps each stage's coordinate space correct.
- **No `SAFE_MODE` gate:** this tool emits zero mouse/keyboard output. It only reads frames and calls the VLM. It runs anywhere (Mac/dev) against copied recordings.
- **Deferred Stage 2b (typing):** add `text_diff.py` producing `TypingEvent` dicts (same timeline schema), crop to `ChangeEvent.change_bbox` before OCR (never OCR the full screenshot — known PaddleOCR-VL hallucination), then pass them to `build_timeline(click_events, typing_events=...)`. No rearchitecting required.
```
