# Workflow Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 수동 녹화 프레임에서 뽑은 `interaction_timeline.json` 을 의미 단위 step 으로 묶어 `workflow.json` + 한국어 절차서 `workflow.md` 로 만든다.

**Architecture:** `recording_filter` 에 두 가지를 더한다 — Stage 2a 가 사이드카 커서를 쓰면 VLM 콜을 건너뛰고(순수 가산 분기), Stage 2b 가 타이핑 구간을 찾아 OCR 로 값을 복원한다. 그 위에 신규 패키지 `workflow_extract` 가 타임라인을 읽어 greedy 단일 패스로 step 을 묶는다 — VLM 콜 0회, 순수 함수, 오프라인 테스트.

**Tech Stack:** Python 3.10+, opencv-python, Pillow, pytest, 기존 `Workflow1VLMClient` (paddleocr-vl-1.5 / mai-ui)

**Spec:** `poc/workflow_3/docs/superpowers/specs/2026-08-11-workflow-extraction-design.md`

## Global Constraints

- **Korean docstrings** — 모든 모듈/함수 docstring 은 한국어.
- **No `__future__` imports** — `from __future__ import annotations` 금지.
- **Print-based logging** — `[INFO]` / `[ERROR]` / `[WARNING]` 접두. `logging` 모듈 금지. **`print()` 문자열 안에 em-dash(U+2014) 금지** (오피스 콘솔 cp949).
- **Absolute imports** — `from poc.workflow_3.xxx import ...`.
- **No CLI arguments** — `argparse` 금지. 설정은 dataclass + env 오버라이드. 스크립트는 `uv run python <script>.py` 만으로 실행.
- **Image format** — 로컬 저장은 JPEG, VLM 전송은 WebP(quality=90).
- **Test style** — `recording_filter` 와 동일한 pytest 스타일(`test_*.py`, 평범한 `def test_*` 함수, 클래스 없음).
- **Commit** — main 에 직접 커밋. **pathspec 으로 내 파일만** (`git add -A` 금지 — 병렬 세션이 같은 repo 를 편집한다).
- **불변식** — 모든 타임라인 이벤트는 정확히 하나의 step `raw_events` 에 나타난다. 누락도 중복도 없다.

## File Structure

| 파일 | 책임 |
|------|------|
| `recording_filter/settings.py` (수정) | Stage 2b 파라미터 추가 |
| `recording_filter/click_detect.py` (수정) | 사이드카 커서 소스 분기 + `cursor_source` 필드 |
| `recording_filter/type_detect.py` (신규) | Stage 2b — 타이핑 구간 탐지 + OCR 값 복원 |
| `recording_filter/filter_recording.py` (수정) | Stage 2b 배선 + summary 필드 |
| `workflow_extract/settings.py` (신규) | `WorkflowExtractSettings` |
| `workflow_extract/steps.py` (신규) | `WorkflowStep` + step dict 생성 |
| `workflow_extract/grouping.py` (신규) | R1~R5 규칙 + greedy 패스 + 불변식 |
| `workflow_extract/render.py` (신규) | step → 한국어 markdown |
| `workflow_extract/extract_workflow.py` (신규) | 엔트리포인트 — 입력 3파일 로드, 산출 기록 |

---

### Task 1: Stage 2a 사이드카 커서 소스

**Files:**
- Modify: `poc/workflow_3/recording_filter/click_detect.py`
- Test: `poc/workflow_3/recording_filter/test_click_detect.py`

**Interfaces:**
- Consumes: `region_gate.nearest_meta(metas, t_sec) -> FrameMeta|None`, `region_gate.screen_point_to_frame(cursor_xy, rect, frame_wh) -> tuple|None` (반환 `(fx, fy)` 튜플), `region_gate.read_frame_size(frame_path) -> (w,h)|None`
- Produces: `resolve_sidecar_cursor(change, metas, frame_wh) -> list|None`, `detect_clicks(..., metas=None)`, `ClickEvent.cursor_source: str`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`test_click_detect.py` 끝에 추가:

```python
from poc.workflow_3.recording_filter.click_detect import resolve_sidecar_cursor
from poc.workflow_3.recording_filter.region_gate import FrameMeta


def _typing_meta(t_sec, cursor_xy):
    return FrameMeta(
        t_sec=t_sec, rect={"left": 0, "top": 0, "right": 1600, "bottom": 1000},
        occlusion="none", cursor_xy=cursor_xy, cursor_in_window=True,
    )


def test_resolve_sidecar_cursor_converts_screen_to_frame():
    """rect 1600x1000 / frame 800x500 이면 배율 0.5 가 적용돼야 한다."""
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_typing_meta(10.0, [400, 200])]
    assert resolve_sidecar_cursor(change, metas, (800, 500)) == [200, 100]


def test_resolve_sidecar_cursor_none_without_meta():
    """사이드카가 없으면 None (호출부가 VLM 경로로 폴백해야 한다)."""
    change = _change_event(rank=0, t_sec=10.0)
    assert resolve_sidecar_cursor(change, [], (800, 500)) is None


def test_resolve_sidecar_cursor_none_when_cursor_missing():
    """cursor_xy 가 None 이면 '커서 없음'이 아니라 '판정 불가'라 None 이다."""
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_typing_meta(10.0, None)]
    assert resolve_sidecar_cursor(change, metas, (800, 500)) is None
```

`_change_event` 헬퍼가 파일에 없으면 함께 추가:

```python
def _change_event(rank, t_sec):
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/cd_{rank}.jpg", prev_frame_path=f"/tmp/cd_prev_{rank}.jpg",
        timestamp_sec=t_sec, frame_index=rank,
        change_bbox={"left": 0, "top": 0, "right": 10, "bottom": 10},
        largest_blob_area_px=100, changed_pixels=100,
    )
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_click_detect.py -v -k sidecar`
Expected: FAIL — `ImportError: cannot import name 'resolve_sidecar_cursor'`

- [ ] **Step 3: 최소 구현**

`click_detect.py` 의 `_locate_cursor` 위에 추가:

```python
def resolve_sidecar_cursor(change, metas, frame_wh):
    """사이드카에서 이 프레임의 커서 프레임 좌표를 얻는다. 불가하면 None.

    수동 녹화 세션의 로컬 커서(GetCursorPos)는 엔지니어의 커서 그 자체라
    VLM 추정보다 정확하고 콜이 들지 않는다. 알람 녹화는 사이드카가 없어
    항상 None 이 나오고 호출부가 기존 VLM 경로로 폴백한다.

    좌표 변환은 region_gate.screen_point_to_frame 을 쓴다 - 단순 뺄셈은
    오피스 125/150% 배율에서 좌표계를 섞는다(2026-08-10 FINDING 2).
    """
    if not metas or not frame_wh:
        return None
    meta = nearest_meta(metas, change.timestamp_sec)
    if meta is None or meta.cursor_xy is None or meta.rect is None:
        return None
    point = screen_point_to_frame(meta.cursor_xy, meta.rect, frame_wh)
    if point is None:
        return None
    fx, fy = point   # screen_point_to_frame 은 (fx, fy) 튜플을 돌려준다(dict 아님).
    return [int(fx), int(fy)]
```

파일 상단 import 에 추가:

```python
from poc.workflow_3.recording_filter.region_gate import (
    nearest_meta,
    read_frame_size,
    screen_point_to_frame,
)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_click_detect.py -v -k sidecar`
Expected: PASS (3 passed)

- [ ] **Step 5: `cursor_source` 필드와 detect_clicks 분기 테스트를 쓴다**

```python
class _ExplodingClient:
    """호출되면 실패한다 - 사이드카 경로가 VLM 을 부르지 않음을 증명한다."""

    def chat_with_image_b64(self, **kwargs):
        raise AssertionError("사이드카 경로에서 VLM 을 부르면 안 된다")


def test_detect_clicks_uses_sidecar_without_vlm(tmp_path, monkeypatch):
    """사이드카가 있으면 VLM 콜 없이 cursor_source='sidecar' 로 판정한다."""
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_typing_meta(10.0, [400, 200])]
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect.read_frame_size",
        lambda path: (800, 500),
    )
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._diff_mask",
        lambda prev, curr, thr: None,
    )
    events = detect_clicks(
        [change], RecordingFilterSettings(vlm_request_delay_sec=0.0),
        client=_ExplodingClient(), metas=metas,
    )
    assert events[0].cursor_source == "sidecar"


def test_detect_clicks_falls_back_to_vlm_without_sidecar(monkeypatch):
    """사이드카가 없으면 오늘과 동일하게 VLM 경로를 탄다."""
    change = _change_event(rank=0, t_sec=10.0)
    calls = []

    def _fake_locate(client, frame_path):
        calls.append(frame_path)
        return {"cursor_visible": False}, None, 800, 500

    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._locate_cursor", _fake_locate
    )
    events = detect_clicks(
        [change], RecordingFilterSettings(vlm_request_delay_sec=0.0),
        client=object(), metas=None,
    )
    assert len(calls) == 1
    assert events[0].cursor_source == "vlm"
```

- [ ] **Step 6: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_click_detect.py -v -k "sidecar or fallback"`
Expected: FAIL — `TypeError: detect_clicks() got an unexpected keyword argument 'metas'`

- [ ] **Step 7: `ClickEvent` 에 필드 추가**

```python
    confidence: float
    evidence: str
    cursor_source: str = "none"   # sidecar | vlm | none
```

기본값이 있으므로 기존 생성부는 그대로 동작한다. VLM 경로의 두 생성부
(`cursor_px is None` 분기와 최종 분기)에는 `cursor_source="vlm"` 을 명시한다.

- [ ] **Step 8: `detect_clicks` 에 사이드카 분기를 넣는다**

`detect_clicks` 시그니처를 바꾸고:

```python
def detect_clicks(change_events, settings, *, client, metas=None) -> list[ClickEvent]:
```

루프 안 `try: parsed, cursor_px, ... = _locate_cursor(...)` **앞**에 삽입:

```python
        frame_wh = read_frame_size(change.frame_path) if metas else None
        sidecar_xy = resolve_sidecar_cursor(change, metas, frame_wh)
        if sidecar_xy is not None:
            results.append(
                _sidecar_event(change, sidecar_xy, frame_wh, settings)
            )
            continue
```

사이드카 경로는 VLM 을 부르지 않으므로 `calls` 를 올리지 않고
`_sleep(settings.vlm_request_delay_sec)` 도 하지 않는다(프록시 과부하 방지
간격은 네트워크 호출에만 필요하다).

`_sidecar_event` 를 `_unavailable_event` 옆에 추가:

```python
# 사이드카 좌표에는 bbox 가 없다. 오버레이(write_click_overlays)가 bbox 없는
# 이벤트를 건너뛰므로, 점 주위에 합성 bbox 를 만들어 클릭 오버레이가 계속
# 그려지게 한다 - 안 만들면 수동 세션의 오버레이가 통째로 사라진다.
SIDECAR_CURSOR_BBOX_PX = 32


def _sidecar_event(change, cursor_xy, frame_wh, settings) -> ClickEvent:
    """사이드카 커서로 ROI 변화를 세어 클릭을 판정한다(VLM 콜 없음)."""
    width, height = frame_wh
    mask = _diff_mask(
        Path(change.prev_frame_path), Path(change.frame_path), settings.click_diff_threshold
    )
    if mask is None:
        event = _unavailable_event(change)
        event.cursor_source = "sidecar"
        return event
    window = _window_around(
        cursor_xy[0], cursor_xy[1], settings.cursor_click_window_px, width, height
    )
    changed = _count_changed_in_window(mask, window)
    is_click = changed >= settings.click_min_changed_px
    return ClickEvent(
        change=change, is_click=is_click,
        status="click" if is_click else "no_click",
        cursor_visible=True, cursor_kind="sidecar",
        cursor_bbox=_window_around(
            cursor_xy[0], cursor_xy[1], SIDECAR_CURSOR_BBOX_PX, width, height
        ),
        cursor_xy=list(cursor_xy), click_window=window,
        changed_in_window_px=changed, confidence=1.0,
        evidence="sidecar cursor", cursor_source="sidecar",
    )
```

- [ ] **Step 9: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/ -v`
Expected: PASS — 기존 71개 + 신규 5개가 모두 통과. 기존 테스트가 하나라도 깨지면 사이드카 분기가 VLM 경로를 침범한 것이다.

- [ ] **Step 10: `filter_recording.py` 에서 metas 를 넘긴다**

`detect_clicks(change_events, settings, client=client)` 를 다음으로 바꾼다:

```python
    click_events = detect_clicks(change_events, settings, client=client, metas=metas)
```

`metas` 는 Stage 1.5 블록에서 이미 로드돼 있다(`metas = load_frame_meta(...)`).
`region_gate_enabled=0` 이어도 `metas` 는 그 위에서 로드되므로 항상 유효하다.

- [ ] **Step 11: 타임라인에 `cursor_source` 를 싣는다**

`timeline.py` 의 `build_timeline` 안 `events.append({...})` 에 추가:

```python
                "cursor_source": getattr(ce, "cursor_source", "none"),
```

- [ ] **Step 12: 커밋**

```bash
git add poc/workflow_3/recording_filter/click_detect.py \
        poc/workflow_3/recording_filter/test_click_detect.py \
        poc/workflow_3/recording_filter/filter_recording.py \
        poc/workflow_3/recording_filter/timeline.py
git commit -m "feat(recording_filter): Stage 2a 사이드카 커서 소스 (수동 세션 VLM 콜 제거)"
```

---

### Task 2: Stage 2b 설정 파라미터

**Files:**
- Modify: `poc/workflow_3/recording_filter/settings.py`
- Test: `poc/workflow_3/recording_filter/test_settings.py`

**Interfaces:**
- Produces: `RecordingFilterSettings.typing_*` 필드 7개

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_typing_settings_defaults():
    """Stage 2b 기본값 - 스펙 §5 와 일치해야 한다."""
    s = RecordingFilterSettings()
    assert s.typing_detect_enabled is True
    assert s.typing_cursor_still_px == 8
    assert s.typing_min_burst_events == 3
    assert s.typing_burst_idle_sec == 1.5
    assert s.typing_focus_max_sec == 2.0
    assert s.typing_ocr_service == "paddleocr-vl-1.5"


def test_typing_settings_env_override(monkeypatch):
    """env 로 임계값을 바꿀 수 있어야 한다(CLI 인자 없음 규칙)."""
    monkeypatch.setenv("RECORDING_FILTER_TYPING_MIN_BURST_EVENTS", "5")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_DETECT", "0")
    s = load_recording_filter_settings()
    assert s.typing_min_burst_events == 5
    assert s.typing_detect_enabled is False
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_settings.py -v -k typing`
Expected: FAIL — `AttributeError: 'RecordingFilterSettings' object has no attribute 'typing_detect_enabled'`

- [ ] **Step 3: 구현**

`settings.py` 의 dataclass 에 Stage 2c 블록 뒤로 추가:

```python
    # ---- Stage 2b: 타이핑 구간 ----
    typing_detect_enabled: bool = True
    typing_cursor_still_px: int = 8       # 이 이상 움직이면 구간이 끊긴다.
    typing_min_burst_events: int = 3      # 구간으로 인정할 최소 change event 수.
    typing_burst_idle_sec: float = 1.5    # 변화가 이 시간 없으면 구간 종료.
    typing_focus_max_sec: float = 2.0     # 구간 직전 이 시간 안의 클릭을 필드로 본다.
    typing_ocr_service: str = "paddleocr-vl-1.5"
```

`load_recording_filter_settings` 에 추가:

```python
        typing_detect_enabled=env_flag("RECORDING_FILTER_TYPING_DETECT", True),
        typing_cursor_still_px=env_int("RECORDING_FILTER_TYPING_CURSOR_STILL_PX", 8),
        typing_min_burst_events=env_int("RECORDING_FILTER_TYPING_MIN_BURST_EVENTS", 3),
        typing_burst_idle_sec=env_float("RECORDING_FILTER_TYPING_BURST_IDLE_SEC", 1.5),
        typing_focus_max_sec=env_float("RECORDING_FILTER_TYPING_FOCUS_MAX_SEC", 2.0),
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_settings.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/recording_filter/settings.py \
        poc/workflow_3/recording_filter/test_settings.py
git commit -m "feat(recording_filter): Stage 2b 타이핑 탐지 설정 파라미터"
```

---

### Task 3: Stage 2b 타이핑 구간 탐지 (OCR 없음)

**Files:**
- Create: `poc/workflow_3/recording_filter/type_detect.py`
- Test: `poc/workflow_3/recording_filter/test_type_detect.py`

**Interfaces:**
- Consumes: `ChangeEvent`, `FrameMeta`, `nearest_meta`, `RecordingFilterSettings.typing_*`
- Produces: `TypingBurst(ranks, start_t_sec, end_t_sec, roi, cursor_xy)`, `find_typing_bursts(change_events, metas, settings) -> list[TypingBurst]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`test_type_detect.py` 신규:

```python
"""Stage 2b 타이핑 구간 탐지 테스트 - 커서 정지 + 국소 반복 변화."""

from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.region_gate import FrameMeta
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.recording_filter.type_detect import find_typing_bursts

_RECT = {"left": 0, "top": 0, "right": 800, "bottom": 500}
_FIELD = {"left": 100, "top": 100, "right": 300, "bottom": 130}


def _ev(rank, t_sec, bbox=None):
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/t_{rank}.jpg", prev_frame_path=f"/tmp/t_prev_{rank}.jpg",
        timestamp_sec=t_sec, frame_index=rank, change_bbox=bbox or dict(_FIELD),
        largest_blob_area_px=500, changed_pixels=500,
    )


def _meta(t_sec, cursor_xy):
    return FrameMeta(
        t_sec=t_sec, rect=_RECT, occlusion="none",
        cursor_xy=cursor_xy, cursor_in_window=True,
    )


def test_finds_burst_when_cursor_still_and_change_localized():
    """커서가 멈춘 채 같은 영역이 4회 바뀌면 타이핑 구간 1개."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(4)]
    bursts = find_typing_bursts(events, metas, RecordingFilterSettings())
    assert len(bursts) == 1
    assert bursts[0].ranks == [0, 1, 2, 3]


def test_no_burst_when_cursor_moves():
    """커서가 움직이면 타이핑이 아니다(마우스 조작 중 화면 변화)."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    metas = [_meta(10.0 + i * 0.3, [200 + i * 50, 200]) for i in range(4)]
    assert find_typing_bursts(events, metas, RecordingFilterSettings()) == []


def test_no_burst_below_min_events():
    """2건짜리 변화는 구간으로 인정하지 않는다(기본 임계 3)."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(2)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(2)]
    assert find_typing_bursts(events, metas, RecordingFilterSettings()) == []


def test_idle_gap_splits_bursts():
    """변화가 idle 상한을 넘게 끊기면 별개 구간이 된다."""
    times = [10.0, 10.3, 10.6, 30.0, 30.3, 30.6]
    events = [_ev(i, t) for i, t in enumerate(times)]
    metas = [_meta(t, [200, 200]) for t in times]
    bursts = find_typing_bursts(events, metas, RecordingFilterSettings())
    assert [b.ranks for b in bursts] == [[0, 1, 2], [3, 4, 5]]


def test_no_burst_without_sidecar():
    """사이드카가 없으면 커서 정지를 알 수 없으므로 구간을 만들지 않는다."""
    events = [_ev(i, 10.0 + i * 0.3) for i in range(4)]
    assert find_typing_bursts(events, [], RecordingFilterSettings()) == []


def test_roi_is_union_of_change_boxes():
    """구간 ROI 는 구성 change_bbox 의 합집합이어야 한다(글자가 오른쪽으로 늘어난다)."""
    boxes = [
        {"left": 100, "top": 100, "right": 150, "bottom": 130},
        {"left": 140, "top": 100, "right": 200, "bottom": 130},
        {"left": 190, "top": 100, "right": 260, "bottom": 130},
    ]
    events = [_ev(i, 10.0 + i * 0.3, boxes[i]) for i in range(3)]
    metas = [_meta(10.0 + i * 0.3, [200, 200]) for i in range(3)]
    burst = find_typing_bursts(events, metas, RecordingFilterSettings())[0]
    assert burst.roi == {"left": 100, "top": 100, "right": 260, "bottom": 130}
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_type_detect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.recording_filter.type_detect'`

- [ ] **Step 3: 구현**

`type_detect.py` 신규:

```python
"""STAGE 2b - 타이핑 구간을 찾아 OCR 로 입력값을 복원한다.

타이핑은 마우스가 멈춘 채 픽셀이 계속 바뀌는 유일한 조작이다 - 클릭(커서 이동 후
1회 국소 변화)의 정확한 반대다. 입력을 후킹하지 않으므로 키는 기록되지 않고,
화면에 렌더된 글자만 OCR 로 복원한다.

캐럿(텍스트 커서) 깜빡임도 같은 신호를 내므로, 구간 시작/끝 OCR 텍스트가 같으면
타이핑이 아니라고 판정해 구간을 버린다.
"""

from dataclasses import dataclass, field
from pathlib import Path

from poc.workflow_3.recording_filter.region_gate import nearest_meta


@dataclass
class TypingBurst:
    """타이핑으로 보이는 연속 변화 구간 1건."""

    ranks: list = field(default_factory=list)   # 구성 ChangeEvent 의 rank 목록
    start_t_sec: float = 0.0
    end_t_sec: float = 0.0
    roi: dict = field(default_factory=dict)     # 구성 change_bbox 의 합집합
    cursor_xy: list = field(default_factory=list)   # 프레임 좌표가 아니라 화면 좌표
    frame_path: str = ""        # 구간 시작 프레임
    end_frame_path: str = ""    # 구간 종료 프레임


def _union_box(a, b):
    """두 bbox 의 합집합을 만든다. 한쪽이 비면 다른 쪽을 그대로 돌려준다."""
    if not a:
        return dict(b) if b else {}
    if not b:
        return dict(a)
    return {
        "left": min(int(a["left"]), int(b["left"])),
        "top": min(int(a["top"]), int(b["top"])),
        "right": max(int(a["right"]), int(b["right"])),
        "bottom": max(int(a["bottom"]), int(b["bottom"])),
    }


def _cursor_moved(prev_xy, curr_xy, still_px) -> bool:
    """두 화면 좌표 사이 이동이 still_px 를 넘는지 본다."""
    if prev_xy is None or curr_xy is None:
        return True
    dx = float(curr_xy[0]) - float(prev_xy[0])
    dy = float(curr_xy[1]) - float(prev_xy[1])
    return (dx * dx + dy * dy) > (float(still_px) ** 2)


def _flush(current, settings, out):
    """모아둔 구간이 최소 길이를 넘으면 결과에 넣는다."""
    if current is not None and len(current.ranks) >= settings.typing_min_burst_events:
        out.append(current)


def find_typing_bursts(change_events, metas, settings) -> list:
    """커서 정지 + 국소 반복 변화로 타이핑 구간을 찾는다(OCR 없음).

    사이드카가 없으면 커서 정지를 알 수 없으므로 빈 목록을 돌려준다 - 알람 녹화는
    이 단계를 통째로 건너뛴다.
    """
    if not metas or not change_events:
        return []

    bursts: list = []
    current = None
    prev_cursor = None
    prev_t = None

    for event in change_events:
        meta = nearest_meta(metas, event.timestamp_sec)
        cursor = meta.cursor_xy if meta is not None else None
        if cursor is None:
            _flush(current, settings, bursts)
            current, prev_cursor, prev_t = None, None, None
            continue

        idle_broken = (
            prev_t is not None
            and (event.timestamp_sec - prev_t) > settings.typing_burst_idle_sec
        )
        if current is None or idle_broken or _cursor_moved(
            prev_cursor, cursor, settings.typing_cursor_still_px
        ):
            _flush(current, settings, bursts)
            current = TypingBurst(
                ranks=[event.rank], start_t_sec=event.timestamp_sec,
                end_t_sec=event.timestamp_sec, roi=dict(event.change_bbox or {}),
                cursor_xy=list(cursor), frame_path=event.frame_path,
                end_frame_path=event.frame_path,
            )
        else:
            current.ranks.append(event.rank)
            current.end_t_sec = event.timestamp_sec
            current.roi = _union_box(current.roi, event.change_bbox)
            current.end_frame_path = event.frame_path

        prev_cursor, prev_t = cursor, event.timestamp_sec

    _flush(current, settings, bursts)
    return bursts
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_type_detect.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/recording_filter/type_detect.py \
        poc/workflow_3/recording_filter/test_type_detect.py
git commit -m "feat(recording_filter): Stage 2b 타이핑 구간 탐지 (커서 정지 + 국소 반복 변화)"
```

---

### Task 4: Stage 2b OCR 값 복원 + 포커스 연결

**Files:**
- Modify: `poc/workflow_3/recording_filter/type_detect.py`
- Test: `poc/workflow_3/recording_filter/test_type_detect.py`

**Interfaces:**
- Consumes: `TypingBurst`, `ClickEvent`, `vlm.prompts.prompt_ocr_assist.build_spotting_prompt`, `vlm.ocr_spotting.parse_spotting_items`
- Produces: `resolve_typing_events(bursts, click_events, settings, *, ocr_client, image_loader=None) -> list[dict]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
from poc.workflow_3.recording_filter.type_detect import resolve_typing_events


class _StubOCR:
    """구간 시작/끝에 서로 다른 텍스트를 돌려주는 OCR 스텁."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def chat_with_image_b64(self, **kwargs):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1

        class _R:
            pass

        r = _R()
        r.text = '[{"text": "%s", "box": [0, 0, 10, 10]}]' % text if text else "[]"
        return r


def _burst(ranks=(0, 1, 2), start=10.0, end=12.0):
    return TypingBurst(
        ranks=list(ranks), start_t_sec=start, end_t_sec=end,
        roi={"left": 100, "top": 100, "right": 300, "bottom": 130},
        cursor_xy=[200, 200], frame_path="/tmp/t_0.jpg", end_frame_path="/tmp/t_2.jpg",
    )


def _fake_loader(path):
    from PIL import Image

    return Image.new("RGB", (800, 500), "white")


def test_typing_event_recovers_value_from_ocr():
    """구간 끝 OCR 텍스트가 값이 된다."""
    ocr = _StubOCR(["", "MCD916_ALIGN_02"])
    events = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert len(events) == 1
    assert events[0]["action"] == "type_text"
    assert events[0]["text"] == "MCD916_ALIGN_02"
    assert events[0]["element_source"] == "ocr"


def test_caret_blink_rejected_when_text_unchanged():
    """시작/끝 텍스트가 같으면 캐럿 깜빡임이므로 이벤트를 만들지 않는다."""
    ocr = _StubOCR(["same", "same"])
    assert resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    ) == []


def test_focus_click_supplies_target_label():
    """구간 직전 클릭이 필드 이름을 준다.

    라벨은 ClickEvent 가 아니라 Stage 2c 의 labels dict(rank -> ElementLabel)에
    들어 있으므로 별도 인자로 넘긴다.
    """
    from poc.workflow_3.recording_filter.element_label import ElementLabel

    ocr = _StubOCR(["", "value"])
    click = _click_event_for_focus(t_sec=9.0)
    events = resolve_typing_events(
        [_burst()], [click], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
        labels={99: ElementLabel(text="Recipe Name", source="ocr", confidence=1.0)},
    )
    assert events[0]["element"] == "Recipe Name"


def test_target_none_when_no_focus_click():
    """Tab 포커스 등으로 직전 클릭이 없으면 target 은 null 이다(추측 금지)."""
    ocr = _StubOCR(["", "value"])
    events = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=ocr, image_loader=_fake_loader,
    )
    assert events[0]["element"] is None
    assert events[0]["text"] == "value"


def test_ocr_failure_yields_event_without_value():
    """OCR 이 던져도 구간은 남긴다(값만 비운다)."""

    class _Boom:
        def chat_with_image_b64(self, **kwargs):
            raise RuntimeError("ocr down")

    events = resolve_typing_events(
        [_burst()], [], RecordingFilterSettings(),
        ocr_client=_Boom(), image_loader=_fake_loader,
    )
    assert events[0]["text"] is None
    assert events[0]["element_source"] == "none"
```

`_click_event_for_focus` 헬퍼도 추가. **`ClickEvent` 에는 라벨 필드가 없다** — 라벨은
Stage 2c 가 `{rank: ElementLabel}` dict 로 따로 만들므로 `resolve_typing_events` 도
`labels` 를 별도 인자로 받는다:

```python
def _click_event_for_focus(t_sec):
    """포커스 클릭 역할의 ClickEvent 를 만든다(rank=99, 라벨은 labels dict 로 전달)."""
    from poc.workflow_3.recording_filter.click_detect import ClickEvent

    change = _ev(99, t_sec)
    return ClickEvent(
        change=change, is_click=True, status="click", cursor_visible=True,
        cursor_kind="sidecar", cursor_bbox=None, cursor_xy=[200, 110],
        click_window=None, changed_in_window_px=2000, confidence=1.0,
        evidence="", cursor_source="sidecar",
    )
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_type_detect.py -v -k "typing_event or caret or focus or target_none or ocr_failure"`
Expected: FAIL — `ImportError: cannot import name 'resolve_typing_events'`

- [ ] **Step 3: 구현**

`type_detect.py` 에 추가:

```python
def _ocr_text_in_box(image, box, ocr_client) -> str:
    """box 영역의 텍스트를 PaddleOCR Spotting 으로 읽어 한 문자열로 잇는다.

    element_label 의 _read_with_ocr 는 '클릭 지점 최근접 1건'을 고르지만, 여기서는
    필드 전체 내용이 필요하므로 모든 항목을 x 순으로 잇는다. 정렬 실패는 삼킨다 -
    캐럿 판별(before != after)에는 순서가 아니라 결정성만 있으면 된다.
    """
    from poc.workflow_3.util import encode_image_webp
    from poc.workflow_3.vlm.ocr_spotting import parse_spotting_items
    from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_spotting_prompt

    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    crop_b64, _w, _h = encode_image_webp(crop, quality=90)
    system_msg, user_text = build_spotting_prompt()
    response = ocr_client.chat_with_image_b64(
        image_b64=crop_b64, system_message=system_msg, user_text=user_text,
        image_mime="image/webp", temperature=0.0,
    )
    items = parse_spotting_items((response.text or "").strip())
    # parse_spotting_items 는 항상 {"text", "bbox"} 로 정규화해 돌려준다("box" 는 입력
    # 쪽에서만 허용되는 별칭이다). bbox 는 리스트가 아니라 left/top/right/bottom dict 다.
    try:
        items = sorted(items, key=lambda it: float(it["bbox"]["left"]))
    except Exception:
        pass
    return " ".join(str(it.get("text") or "").strip() for it in items).strip()


def _focus_label(burst, click_events, labels, settings):
    """구간 직전 focus_max_sec 안의 클릭에서 필드 라벨을 얻는다. 없으면 None.

    Tab/단축키로 포커스를 옮기면 직전 클릭이 없다. 그때는 라벨을 추측하지 않는다 -
    추측 라벨은 새 오차원이고, 값은 라벨 없이도 쓸모가 있다.
    """
    best = None
    for ce in click_events or []:
        if not ce.is_click:
            continue
        gap = burst.start_t_sec - ce.timestamp_sec
        if 0 <= gap <= settings.typing_focus_max_sec:
            if best is None or ce.timestamp_sec > best.timestamp_sec:
                best = ce
    if best is None:
        return None
    label = (labels or {}).get(best.rank)
    text = getattr(label, "text", "") if label is not None else ""
    return text or None


def _default_image_loader(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def resolve_typing_events(
    bursts, click_events, settings, *, ocr_client, image_loader=None, labels=None
) -> list:
    """구간별로 OCR 2회를 돌려 타임라인 스키마의 type_text 이벤트를 만든다.

    before == after 인 구간은 캐럿 깜빡임으로 보고 버린다. OCR 이 실패하면 값을
    비운 채로 이벤트는 남긴다 - '여기서 무언가를 입력했다'는 사실 자체가 절차의
    일부이기 때문이다.
    """
    loader = image_loader or _default_image_loader
    events = []
    for burst in bursts:
        before, after, source = "", "", "none"
        try:
            before = _ocr_text_in_box(loader(burst.frame_path), burst.roi, ocr_client)
            after = _ocr_text_in_box(loader(burst.end_frame_path), burst.roi, ocr_client)
            source = "ocr"
        except Exception as exc:
            print(f"[WARNING] 타이핑 구간 OCR 실패(값 없이 기록): {exc}")

        # 양쪽이 다 비어 있으면 "변화 없음"이 아니라 "OCR 이 아무것도 못 읽음"이다.
        # 이 둘을 같이 취급하면 ROI 가 어긋난 오독이 캐럿 깜빡임과 구별되지 않아
        # 실제 타이핑이 조용히 사라진다. §8 의 "OCR 실패도 구간은 발행" 규칙을 따른다.
        if source == "ocr" and not before and not after:
            print(
                f"[WARNING] 구간 양끝 OCR 이 모두 빈 텍스트입니다(값 없이 기록) "
                f"(t={burst.start_t_sec:.1f}s)"
            )
            source = "none"
            after = ""
        elif source == "ocr" and before == after:
            print(
                f"[INFO] 캐럿 깜빡임으로 판단해 구간을 버립니다 "
                f"(t={burst.start_t_sec:.1f}s, 텍스트 변화 없음)"
            )
            continue

        events.append({
            "t_sec": burst.start_t_sec,
            "seq": 0,
            "action": "type_text",
            "coords": {"x": int(burst.cursor_xy[0]), "y": int(burst.cursor_xy[1])}
            if burst.cursor_xy else None,
            "element": _focus_label(burst, click_events, labels, settings),
            "element_source": source,
            # 하드코딩하지 않는다 - 라벨 출처가 없으면 이 패키지의 규칙
            # (timeline.derive_target_kind)에 따라 "unknown" 이어야 한다. 그러지 않으면
            # OCR 이 실패한 타이핑이 "다른 장비로 이식 가능"하다고 잘못 주장한다.
            "target_kind": derive_target_kind("ui", source),
            "region": "ui",
            "generation": 0,
            "occlusion": "unknown",
            "text": after or None,
            "confidence": 1.0 if source == "ocr" else 0.0,
            "frame": Path(burst.end_frame_path).name,
            "source_frames": {"prev": burst.frame_path, "curr": burst.end_frame_path},
            "cursor_source": "sidecar",
            "t_sec_end": burst.end_t_sec,
        })
    return events
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_type_detect.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/recording_filter/type_detect.py \
        poc/workflow_3/recording_filter/test_type_detect.py
git commit -m "feat(recording_filter): Stage 2b OCR 값 복원 + 포커스 연결 + 캐럿 배제"
```

---

### Task 5: Stage 2b 를 filter_recording 에 배선

**Files:**
- Modify: `poc/workflow_3/recording_filter/filter_recording.py`
- Test: `poc/workflow_3/recording_filter/test_filter_recording.py`

**Interfaces:**
- Consumes: `find_typing_bursts`, `resolve_typing_events`
- Produces: `interaction_timeline.json` 에 `type_text` 이벤트 포함, `summary.json` 에 `typing_bursts` / `typing_events` / `vlm_calls_stage2b_ocr`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

기존 `_recording_dir(tmp_path)` / `_FakeClient` 헬퍼를 그대로 쓴다(파일 상단에 이미 있다):

```python
def test_summary_reports_typing_counts(tmp_path):
    """summary.json 이 Stage 2b 건수를 보고해야 한다(조용한 누락 금지).

    사이드카가 없는 합성 세션이라 구간은 0건이지만, 필드 자체는 존재해야 한다 -
    없으면 소비자가 '타이핑이 0건'과 '스테이지가 안 돌았다'를 구분할 수 없다.
    """
    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, min_change_area_px=5000)
    assert run_filter(input_dir=rec, settings=settings, client=_FakeClient()) == "success"

    out_dir = rec.parent / "recording_filter"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["typing_bursts"] == 0
    assert summary["typing_events"] == 0
    assert summary["vlm_calls_stage2b_ocr"] == 0


def test_typing_disabled_by_env_still_reports_zero(monkeypatch, tmp_path):
    """RECORDING_FILTER_TYPING_DETECT=0 이어도 필드는 남아야 한다."""
    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=5000, typing_detect_enabled=False
    )
    assert run_filter(input_dir=rec, settings=settings, client=_FakeClient()) == "success"
    summary = json.loads(
        (rec.parent / "recording_filter" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["typing_bursts"] == 0
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/test_filter_recording.py -v -k typing`
Expected: FAIL — `KeyError: 'typing_bursts'` 또는 `assert 'typing_bursts' in summary`

- [ ] **Step 3: 구현**

`filter_recording.py` 의 Stage 2c 블록 **뒤**, `build_timeline` **앞**에 삽입:

```python
    # ---- Stage 2b: 타이핑 구간 ----
    typing_events = []
    typing_bursts = []
    if settings.typing_detect_enabled:
        from poc.workflow_3.recording_filter.type_detect import (
            find_typing_bursts,
            resolve_typing_events,
        )

        typing_bursts = find_typing_bursts(change_events, metas, settings)
        if typing_bursts:
            from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

            typing_ocr = Workflow1VLMClient(settings.typing_ocr_service)
            typing_events = resolve_typing_events(
                typing_bursts, click_events, settings,
                ocr_client=typing_ocr, labels=labels,
            )
        print(
            f"[INFO] Stage 2b 완료: 구간 {len(typing_bursts)} 건 -> "
            f"타이핑 이벤트 {len(typing_events)} 건"
        )
```

`build_timeline` 호출을 바꾼다:

```python
    timeline = build_timeline(
        click_events, typing_events, gate_info=gate_info, labels=labels
    )
```

`summary.json` payload 에 추가:

```python
            "typing_bursts": len(typing_bursts),
            "typing_events": len(typing_events),
            "vlm_calls_stage2b_ocr": len(typing_bursts) * 2,
```

`vlm_calls_total_estimate` 도 갱신:

```python
            "vlm_calls_total_estimate": (
                region_map_calls + len(click_events) + label_calls + len(typing_bursts) * 2
            ),
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/recording_filter/ -v`
Expected: PASS — 전체 통과

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/recording_filter/filter_recording.py \
        poc/workflow_3/recording_filter/test_filter_recording.py
git commit -m "feat(recording_filter): Stage 2b 배선 + summary 타이핑 집계"
```

---

### Task 6: workflow_extract 설정 + step 스키마

**Files:**
- Create: `poc/workflow_3/workflow_extract/__init__.py`
- Create: `poc/workflow_3/workflow_extract/settings.py`
- Create: `poc/workflow_3/workflow_extract/steps.py`
- Test: `poc/workflow_3/workflow_extract/test_steps.py`

**Interfaces:**
- Produces: `WorkflowExtractSettings`, `load_workflow_extract_settings()`, `make_step(events, action, rule, **kw) -> dict`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
"""workflow_extract step 스키마 테스트."""

from poc.workflow_3.workflow_extract.settings import (
    WorkflowExtractSettings,
    load_workflow_extract_settings,
)
from poc.workflow_3.workflow_extract.steps import make_step


def _event(seq, action="click", **kw):
    base = {
        "seq": seq, "t_sec": 10.0 + seq, "action": action, "coords": {"x": 100, "y": 200},
        "element": "PM", "element_source": "ocr", "target_kind": "ui_control",
        "region": "ui", "generation": 0, "occlusion": "none", "text": None,
        "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }
    base.update(kw)
    return base


def test_make_step_carries_raw_events_and_rule():
    step = make_step([_event(3), _event(4)], action="select_from_dropdown", rule="R2")
    assert step["raw_events"] == [3, 4]
    assert step["grouping_rule"] == "R2"
    assert step["action"] == "select_from_dropdown"


def test_make_step_t_sec_is_start_end_pair():
    step = make_step([_event(3), _event(4)], action="click_repeat", rule="R4")
    assert step["t_sec"] == [13.0, 14.0]


def test_make_step_single_event_repeats_timestamp():
    step = make_step([_event(3)], action="click", rule="R5")
    assert step["t_sec"] == [13.0, 13.0]


def test_make_step_uses_t_sec_end_for_typing_burst():
    """타이핑 이벤트는 구간이므로 끝 시각을 잃으면 안 된다(Stage 2b 가 t_sec_end 를 싣는다)."""
    typing = _event(3, action="type_text", text="abc")
    typing["t_sec_end"] = 20.0
    step = make_step([typing], action="type_text", rule="R3")
    assert step["t_sec"] == [13.0, 20.0]


def test_make_step_defaults_are_explicit_nulls():
    """스키마 필드는 항상 존재해야 한다 - 소비자가 키 유무를 분기하면 안 된다."""
    step = make_step([_event(3)], action="click", rule="R5")
    for key in ("target", "target_kind", "value", "value_source",
                "coords_in_live_box", "intent", "count", "inferred"):
        assert key in step


def test_settings_defaults_match_spec():
    s = WorkflowExtractSettings()
    assert s.recenter_window_sec == 1.5
    assert s.recenter_min_ratio == 0.40
    assert s.dropdown_max_sec == 5.0
    assert s.focus_max_sec == 2.0
    assert s.repeat_window_sec == 6.0
    assert s.repeat_min_count == 3
    assert s.same_target_px == 24


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("WORKFLOW_EXTRACT_RECENTER_MIN_RATIO", "0.6")
    assert load_workflow_extract_settings().recenter_min_ratio == 0.6
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_steps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.workflow_extract'`

- [ ] **Step 3: 구현**

`__init__.py`:

```python
"""수동 녹화 타임라인을 의미 단위 workflow step 으로 묶는 오프라인 패키지.

recording_filter 산출(interaction_timeline.json)만 읽고 VLM 을 부르지 않는다 -
그룹핑 규칙은 튜닝 회차가 가장 많은 단계라 재실행이 공짜여야 한다.
"""
```

`settings.py`:

```python
"""workflow_extract 실행 파라미터 - env 주도 dataclass (CLI 인자 없음)."""

from dataclasses import dataclass

from poc.workflow_3.util import env_flag, env_float, env_int


@dataclass
class WorkflowExtractSettings:
    """그룹핑 규칙 임계값. 전부 첫 실측 후 조정 대상이다."""

    recenter_window_sec: float = 1.5    # R1: 클릭 직후 이 시간 안의 변화를 본다.
    recenter_min_ratio: float = 0.40    # R1: live_box 대비 변화 면적 비율 임계.
    dropdown_max_sec: float = 5.0       # R2: 열기 -> 고르기 최대 간격.
    focus_max_sec: float = 2.0          # R3: 포커스 클릭으로 흡수할 최대 간격.
    repeat_window_sec: float = 6.0      # R4: 반복 클릭으로 묶을 시간 창.
    repeat_min_count: int = 3           # R4: 반복으로 인정할 최소 횟수.
    same_target_px: int = 24            # R4: 라벨이 없을 때 동일 대상 판정 거리.
    thumbnails_enabled: bool = True     # step 별 표시 프레임 저장 on/off.


def load_workflow_extract_settings() -> WorkflowExtractSettings:
    """env override 를 적용한 설정을 만든다."""
    return WorkflowExtractSettings(
        recenter_window_sec=env_float("WORKFLOW_EXTRACT_RECENTER_WINDOW_SEC", 1.5),
        recenter_min_ratio=env_float("WORKFLOW_EXTRACT_RECENTER_MIN_RATIO", 0.40),
        dropdown_max_sec=env_float("WORKFLOW_EXTRACT_DROPDOWN_MAX_SEC", 5.0),
        focus_max_sec=env_float("WORKFLOW_EXTRACT_FOCUS_MAX_SEC", 2.0),
        repeat_window_sec=env_float("WORKFLOW_EXTRACT_REPEAT_WINDOW_SEC", 6.0),
        repeat_min_count=env_int("WORKFLOW_EXTRACT_REPEAT_MIN_COUNT", 3),
        same_target_px=env_int("WORKFLOW_EXTRACT_SAME_TARGET_PX", 24),
        thumbnails_enabled=env_flag("WORKFLOW_EXTRACT_THUMBNAILS", True),
    )
```

`steps.py`:

```python
"""workflow step dict 를 만드는 단일 지점.

스키마 필드는 값이 없어도 항상 존재한다 - 소비자(render, 미래 재생기)가 키 유무로
분기하기 시작하면 스키마가 암묵적으로 갈라지기 때문이다.
"""


def make_step(events, *, action, rule, target=None, target_kind=None, value=None,
              value_source="none", coords_in_live_box=None, intent=None,
              count=None, inferred=False) -> dict:
    """구성 이벤트 목록에서 step dict 하나를 만든다.

    events 는 시간순이라고 가정한다(그룹핑 패스가 순서대로 넘긴다). raw_events 는
    원본 타임라인의 seq 를 그대로 든다 - 이것이 그룹핑을 되돌릴 수 있게 하는 유일한
    연결고리다.
    """
    first, last = events[0], events[-1]
    if target_kind is None:
        target_kind = first.get("target_kind") or "unknown"
    # 타이핑 이벤트는 구간이라 시작 시각 하나로는 길이를 잃는다. Stage 2b 가 싣는
    # t_sec_end 를 우선 쓴다(클릭 이벤트에는 없으므로 t_sec 로 폴백).
    # `or` 가 아니라 `is not None` 인 이유: 정상값 0.0 이 falsy 라서 `or` 를 쓰면
    # 있는 값을 버리고 폴백해버린다("없음"과 "0"을 섞지 않는다).
    raw_end = last.get("t_sec_end")
    end_t = float(raw_end if raw_end is not None else last["t_sec"])
    return {
        "seq": 0,   # 그룹핑 패스가 마지막에 다시 매긴다.
        "action": action,
        "target": target,
        "target_kind": target_kind,
        "value": value,
        "value_source": value_source,
        "coords_in_live_box": coords_in_live_box,
        "t_sec": [float(first["t_sec"]), end_t],
        "generation": int(first.get("generation") or 0),
        "grouping_rule": rule,
        "inferred": bool(inferred),
        "intent": intent,
        "count": count,
        "raw_events": [int(e["seq"]) for e in events],
        "frame": first.get("frame"),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_steps.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/
git commit -m "feat(workflow_extract): 설정 + step 스키마 단일 생성점"
```

---

### Task 7: 그룹핑 R5 기본 경로 + 불변식

**Files:**
- Create: `poc/workflow_3/workflow_extract/grouping.py`
- Test: `poc/workflow_3/workflow_extract/test_grouping.py`

**Interfaces:**
- Consumes: `make_step`, `WorkflowExtractSettings`
- Produces: `GroupingContext(settings, live_boxes, changes, frame_wh)`, `group_events(events, ctx) -> list[dict]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
"""그룹핑 규칙 R1~R5 + 불변식 테스트 (순수 함수, VLM 없음)."""

from poc.workflow_3.workflow_extract.grouping import GroupingContext, group_events
from poc.workflow_3.workflow_extract.settings import WorkflowExtractSettings


def _event(seq, t_sec, action="click", region="ui", element="PM",
           target_kind="ui_control", coords=None, generation=0, text=None):
    return {
        "seq": seq, "t_sec": t_sec, "action": action,
        "coords": coords if coords is not None else {"x": 100, "y": 200},
        "element": element, "element_source": "ocr" if element else "none",
        "target_kind": target_kind, "region": region, "generation": generation,
        "occlusion": "none", "text": text, "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }


def _ctx(**kw):
    return GroupingContext(
        settings=kw.pop("settings", WorkflowExtractSettings()),
        live_boxes=kw.pop("live_boxes", {}),
        changes=kw.pop("changes", []),
        frame_wh=kw.pop("frame_wh", (1600, 1000)),
    )


def test_lone_clicks_become_r5_steps():
    events = [_event(0, 10.0), _event(1, 30.0, element="OK")]
    steps = group_events(events, _ctx())
    assert [s["grouping_rule"] for s in steps] == ["R5", "R5"]
    assert [s["action"] for s in steps] == ["click", "click"]


def test_steps_are_renumbered_sequentially():
    events = [_event(0, 10.0), _event(1, 30.0, element="OK")]
    steps = group_events(events, _ctx())
    assert [s["seq"] for s in steps] == [0, 1]


def test_invariant_every_event_used_exactly_once():
    """불변식: 모든 이벤트가 정확히 하나의 step raw_events 에 나타난다."""
    events = [_event(i, 10.0 + i * 20) for i in range(5)]
    steps = group_events(events, _ctx())
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == [0, 1, 2, 3, 4]


def test_empty_timeline_yields_no_steps():
    assert group_events([], _ctx()) == []


def test_events_sorted_by_time_before_grouping():
    """입력이 시간순이 아니어도 결과는 시간순이어야 한다."""
    events = [_event(0, 50.0), _event(1, 10.0, element="OK")]
    steps = group_events(events, _ctx())
    assert steps[0]["raw_events"] == [1]
    assert steps[1]["raw_events"] == [0]
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.workflow_extract.grouping'`

- [ ] **Step 3: 구현**

`grouping.py`:

```python
"""타임라인 이벤트를 의미 단위 step 으로 묶는다 - greedy 단일 패스.

좌->우로 한 번 훑으며 R1..R5 를 우선순위대로 시도하고, 먼저 맞는 규칙이 이벤트를
가져간다. 규칙이 틀려도 되돌릴 수 있도록 모든 이벤트는 정확히 하나의 step
raw_events 에 들어간다(패스 끝에서 assert 로 확인한다).
"""

from dataclasses import dataclass, field

from poc.workflow_3.workflow_extract.steps import make_step


@dataclass
class GroupingContext:
    """규칙이 참조하는 부수 입력. 없으면 해당 규칙이 degrade 한다."""

    settings: object
    live_boxes: dict = field(default_factory=dict)   # {generation: live_box}
    changes: list = field(default_factory=list)      # change_events.json 의 events
    frame_wh: tuple = None                           # (w, h). None 이면 R2 degrade.


def _rule_default(events, i, ctx):
    """R5 - 위 규칙에 안 걸린 이벤트를 1:1 step 으로 만든다."""
    event = events[i]
    return make_step(
        [event], action="click", rule="R5",
        target=event.get("element"), value=None,
    ), 1


_RULES = [_rule_default]


def group_events(events, ctx) -> list:
    """이벤트 목록을 step 목록으로 묶는다(시간순 정렬 후 greedy 단일 패스)."""
    ordered = sorted(events or [], key=lambda e: float(e["t_sec"]))
    steps = []
    i = 0
    while i < len(ordered):
        for rule in _RULES:
            result = rule(ordered, i, ctx)
            if result is not None:
                step, consumed = result
                steps.append(step)
                i += consumed
                break
        else:   # 모든 규칙이 None 을 돌려주는 일은 없어야 한다(R5 가 항상 잡는다).
            raise AssertionError(f"이벤트를 처리할 규칙이 없습니다: seq={ordered[i]['seq']}")

    for seq, step in enumerate(steps):
        step["seq"] = seq

    _assert_invariant(ordered, steps)
    return steps


def _assert_invariant(events, steps) -> None:
    """모든 이벤트가 정확히 한 번씩 raw_events 에 나타나는지 확인한다.

    이 불변식이 깨지면 잘못된 그룹핑을 되돌릴 수 없다 - 산출물을 내보내기 전에
    여기서 멈추는 편이 조용히 왜곡된 절차서를 내는 것보다 낫다.
    """
    expected = sorted(int(e["seq"]) for e in events)
    seen = sorted(r for step in steps for r in step["raw_events"])
    if seen != expected:
        raise AssertionError(
            f"그룹핑 불변식 위반: 입력 {len(expected)} 건, raw_events {len(seen)} 건. "
            f"누락={sorted(set(expected) - set(seen))}, 중복/과잉={sorted(set(seen) - set(expected))}"
        )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/grouping.py \
        poc/workflow_3/workflow_extract/test_grouping.py
git commit -m "feat(workflow_extract): greedy 그룹핑 패스 + R5 기본 경로 + 불변식 가드"
```

---

### Task 8: 규칙 R1 (FOV 이동 더블클릭)

**Files:**
- Modify: `poc/workflow_3/workflow_extract/grouping.py`
- Test: `poc/workflow_3/workflow_extract/test_grouping.py`

**Interfaces:**
- Consumes: `GroupingContext.live_boxes`, `GroupingContext.changes`
- Produces: `_rule_double_click(events, i, ctx)`, `box_overlap_ratio(bbox, live_box) -> float`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
_LIVE_BOX = {"left": 200, "top": 100, "right": 1000, "bottom": 700}


def _change(t_sec, bbox):
    return {"timestamp_sec": t_sec, "change_bbox": bbox}


def test_r1_fires_on_live_image_click_with_recenter_change():
    """라이브 박스 클릭 직후 박스 대부분이 다시 그려지면 FOV 이동 더블클릭."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    changes = [_change(10.4, dict(_LIVE_BOX))]
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=changes))
    assert steps[0]["action"] == "double_click"
    assert steps[0]["grouping_rule"] == "R1"
    assert steps[0]["intent"] == "fov_move"
    assert steps[0]["inferred"] is True


def test_r1_does_not_fire_on_small_local_change():
    """라이브 박스 안이라도 국소 변화면 단발 클릭(마커/선택)이다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    small = {"left": 300, "top": 200, "right": 340, "bottom": 240}
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, small)]))
    assert steps[0]["action"] == "click"
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_does_not_fire_outside_live_region():
    """UI 컨트롤 클릭은 뒤에 큰 변화가 와도 더블클릭이 아니다."""
    events = [_event(0, 10.0, region="ui")]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, dict(_LIVE_BOX))])
    )
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_ignores_change_outside_time_window():
    """1.5초 창 밖의 변화는 이 클릭의 결과로 보지 않는다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(20.0, dict(_LIVE_BOX))])
    )
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_degrades_without_live_box():
    """region_map.json 이 없으면 비율을 못 재므로 평범한 click 으로 degrade."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    steps = group_events(events, _ctx(live_boxes={}, changes=[_change(10.4, dict(_LIVE_BOX))]))
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_sets_normalized_coords_in_live_box():
    """live_image step 은 창 픽셀이 아니라 라이브 박스 내부 정규화 좌표를 든다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 600, "y": 400})]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, dict(_LIVE_BOX))])
    )
    assert steps[0]["coords_in_live_box"] == [0.5, 0.5]


def test_r5_live_image_click_also_gets_normalized_coords():
    """더블클릭이 아닌 라이브 박스 단발 클릭도 정규화 좌표를 들어야 한다.

    스펙 §6 은 'live_image step' 전체에 coords_in_live_box 를 요구한다. R1 만
    채우면 마커/선택 클릭이 창 픽셀만 든 채 남아, 소비자가 두 종류의 live_image
    step 을 서로 다르게 다뤄야 한다.
    """
    events = [_event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 600, "y": 400})]
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[]))
    assert steps[0]["grouping_rule"] == "R5"
    assert steps[0]["coords_in_live_box"] == [0.5, 0.5]
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v -k r1`
Expected: FAIL — 6개 중 대부분이 `assert 'R5' == 'R1'` 류로 실패

- [ ] **Step 3: 구현**

`grouping.py` 의 `_rule_default` **위**에 추가:

```python
def box_overlap_ratio(bbox, live_box) -> float:
    """bbox 와 live_box 의 교집합 면적을 live_box 면적으로 나눈 비율."""
    if not bbox or not live_box:
        return 0.0
    left = max(int(bbox["left"]), int(live_box["left"]))
    top = max(int(bbox["top"]), int(live_box["top"]))
    right = min(int(bbox["right"]), int(live_box["right"]))
    bottom = min(int(bbox["bottom"]), int(live_box["bottom"]))
    if right <= left or bottom <= top:
        return 0.0
    live_area = (int(live_box["right"]) - int(live_box["left"])) * (
        int(live_box["bottom"]) - int(live_box["top"])
    )
    if live_area <= 0:
        return 0.0
    return ((right - left) * (bottom - top)) / float(live_area)


def normalized_in_live_box(coords, live_box):
    """클릭 좌표를 라이브 박스 내부 0~1 좌표로 바꾼다. 불가하면 None.

    창 위치/크기에 독립적이고, '좌표가 아니라 영상 내용에 의존한다'는 것을
    스키마 자체로 드러낸다.
    """
    if not coords or not live_box:
        return None
    width = int(live_box["right"]) - int(live_box["left"])
    height = int(live_box["bottom"]) - int(live_box["top"])
    if width <= 0 or height <= 0:
        return None
    nx = (float(coords["x"]) - int(live_box["left"])) / width
    ny = (float(coords["y"]) - int(live_box["top"])) / height
    return [round(nx, 4), round(ny, 4)]


def _has_recenter_change(t_sec, live_box, ctx) -> bool:
    """클릭 직후 창 안에서 라이브 박스 대부분이 다시 그려졌는지 본다."""
    for change in ctx.changes or []:
        delta = float(change.get("timestamp_sec") or 0.0) - float(t_sec)
        if delta < 0 or delta > ctx.settings.recenter_window_sec:
            continue
        if box_overlap_ratio(change.get("change_bbox"), live_box) >= ctx.settings.recenter_min_ratio:
            return True
    return False


def _rule_double_click(events, i, ctx):
    """R1 - 라이브 박스 클릭 + recenter 시그니처 = FOV 이동 더블클릭(추론).

    프레임 주기(~3-4fps)로는 두 번의 누름을 시간으로 분리할 수 없다. 대신 결과를
    본다 - recenter 는 라이브 박스 전체를 다시 그리고, 단발 클릭은 국소 변화만
    남긴다. 관측이 아니라 추론이므로 inferred=True 를 남긴다.
    """
    event = events[i]
    if event.get("action") != "click" or event.get("region") != "live_image":
        return None
    live_box = (ctx.live_boxes or {}).get(int(event.get("generation") or 0))
    if not live_box:
        return None
    if not _has_recenter_change(event["t_sec"], live_box, ctx):
        return None
    return make_step(
        [event], action="double_click", rule="R1", intent="fov_move", inferred=True,
        target=event.get("element"), target_kind="live_image",
        coords_in_live_box=normalized_in_live_box(event.get("coords"), live_box),
    ), 1
```

`_rule_default` 도 갱신한다 — 라이브 박스 안의 단발 클릭 역시 `live_image` step 이므로
정규화 좌표를 들어야 한다(스펙 §6):

```python
def _rule_default(events, i, ctx):
    """R5 - 위 규칙에 안 걸린 이벤트를 1:1 step 으로 만든다."""
    event = events[i]
    live_box = (ctx.live_boxes or {}).get(int(event.get("generation") or 0))
    normalized = None
    if event.get("region") == "live_image" and live_box:
        normalized = normalized_in_live_box(event.get("coords"), live_box)
    return make_step(
        [event], action="click", rule="R5",
        target=event.get("element"), value=None,
        coords_in_live_box=normalized,
    ), 1
```

`_RULES` 를 갱신:

```python
_RULES = [_rule_double_click, _rule_default]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/grouping.py \
        poc/workflow_3/workflow_extract/test_grouping.py
git commit -m "feat(workflow_extract): R1 FOV 이동 더블클릭 추론 + 라이브 박스 정규화 좌표"
```

---

### Task 9: 규칙 R2 (드롭다운 선택)

**Files:**
- Modify: `poc/workflow_3/workflow_extract/grouping.py`
- Test: `poc/workflow_3/workflow_extract/test_grouping.py`

**Interfaces:**
- Consumes: `sem_monitor.pm_dropdown.dropdown_region_below(button_xy, frame_wh) -> (l,t,r,b)|None`
- Produces: `_rule_dropdown(events, i, ctx)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_r2_groups_open_and_select():
    """PM 클릭 -> 바로 아래 영역 클릭 = 드롭다운 선택 1 step."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    steps = group_events([open_ev, pick_ev], _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "select_from_dropdown"
    assert steps[0]["target"] == "PM"
    assert steps[0]["value"] == "210"
    assert steps[0]["value_source"] == "ocr"
    assert steps[0]["raw_events"] == [0, 1]


def test_r2_does_not_group_when_too_slow():
    """5초를 넘으면 별개 조작이다."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 30.0, element="210", coords={"x": 810, "y": 420})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2


def test_r2_does_not_group_click_outside_dropdown_region():
    """아래가 아니라 옆을 눌렀으면 드롭다운이 아니다."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="OK", coords={"x": 200, "y": 310})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2


def test_r2_degrades_without_frame_size():
    """frame_wh 를 모르면 드롭다운 기하를 계산할 수 없어 degrade."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    steps = group_events([open_ev, pick_ev], _ctx(frame_wh=None))
    assert [s["grouping_rule"] for s in steps] == ["R5", "R5"]


def test_r2_requires_ui_control_opener():
    """라이브 영상 위 클릭은 드롭다운 여는 동작이 아니다."""
    open_ev = _event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v -k r2`
Expected: FAIL — `assert 2 == 1` (아직 묶이지 않는다)

- [ ] **Step 3: 구현**

`grouping.py` 에 추가:

```python
def _point_in_region(coords, region) -> bool:
    """(l, t, r, b) 튜플 영역 안에 점이 있는지 본다."""
    if not coords or not region:
        return False
    left, top, right, bottom = region
    return left <= float(coords["x"]) <= right and top <= float(coords["y"]) <= bottom


def _rule_dropdown(events, i, ctx):
    """R2 - ui_control 클릭 직후 그 아래 영역 클릭 = 드롭다운 선택.

    기하는 sem_monitor.pm_dropdown.dropdown_region_below 를 그대로 쓴다. PM 드롭다운
    실행기가 이미 쓰는 함수라, 관측이 인식하는 드롭다운과 실행기가 수행할 수 있는
    드롭다운이 어긋날 수 없다. 다만 그 비율 상수는 PM 전용 보정이라 더 넓은
    드롭다운은 놓칠 수 있다(첫 실측 후 일반 비율셋 필요 여부를 판단한다).

    바로 다음 이벤트만 본다 - 사이에 다른 조작이 끼면 묶지 않는다. 비인접 이벤트를
    소비하면 사이 이벤트가 건너뛰어져 불변식이 깨진다.
    """
    from poc.workflow_3.sem_monitor.pm_dropdown import dropdown_region_below

    opener = events[i]
    if opener.get("action") != "click" or opener.get("target_kind") != "ui_control":
        return None
    if not opener.get("coords") or not ctx.frame_wh:
        return None
    if i + 1 >= len(events):
        return None
    picker = events[i + 1]
    if picker.get("action") != "click" or not picker.get("coords"):
        return None
    if float(picker["t_sec"]) - float(opener["t_sec"]) > ctx.settings.dropdown_max_sec:
        return None
    region = dropdown_region_below(
        {"x": int(opener["coords"]["x"]), "y": int(opener["coords"]["y"])}, ctx.frame_wh
    )
    if region is None or not _point_in_region(picker.get("coords"), region):
        return None
    return make_step(
        [opener, picker], action="select_from_dropdown", rule="R2",
        target=opener.get("element"), target_kind="ui_control",
        value=picker.get("element"),
        value_source=picker.get("element_source") or "none",
    ), 2
```

`_RULES` 를 갱신:

```python
_RULES = [_rule_double_click, _rule_dropdown, _rule_default]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/grouping.py \
        poc/workflow_3/workflow_extract/test_grouping.py
git commit -m "feat(workflow_extract): R2 드롭다운 선택 (실행기와 동일 기하 재사용)"
```

---

### Task 10: 규칙 R3 (타이핑 + 포커스 흡수) 와 R4 (반복 클릭)

**Files:**
- Modify: `poc/workflow_3/workflow_extract/grouping.py`
- Test: `poc/workflow_3/workflow_extract/test_grouping.py`

**Interfaces:**
- Produces: `_rule_type_text(events, i, ctx)`, `_rule_click_repeat(events, i, ctx)`, `same_target(a, b, settings) -> bool`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_r3_absorbs_focus_click():
    """필드 클릭 직후 타이핑이면 클릭은 포커스로 흡수돼 1 step 이 된다."""
    click = _event(0, 10.0, element="Recipe Name")
    typing = _event(1, 11.0, action="type_text", element="Recipe Name", text="MCD916")
    steps = group_events([click, typing], _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "type_text"
    assert steps[0]["value"] == "MCD916"
    assert steps[0]["raw_events"] == [0, 1]


def test_r3_standalone_typing_without_focus_click():
    """Tab 포커스면 직전 클릭이 없어도 type_text step 이 나온다."""
    typing = _event(0, 11.0, action="type_text", element=None, text="MCD916")
    steps = group_events([typing], _ctx())
    assert steps[0]["action"] == "type_text"
    assert steps[0]["target"] is None
    assert steps[0]["raw_events"] == [0]


def test_r3_does_not_absorb_distant_click():
    """포커스 창(2초)을 넘긴 클릭은 별개 조작이다."""
    click = _event(0, 10.0, element="Recipe Name")
    typing = _event(1, 20.0, action="type_text", element="Recipe Name", text="MCD916")
    assert len(group_events([click, typing], _ctx())) == 2


def test_r4_groups_repeated_clicks_on_same_label():
    """같은 라벨을 3회 이상 누르면 반복 1 step."""
    events = [_event(i, 10.0 + i, element="Zoom In") for i in range(3)]
    steps = group_events(events, _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "click_repeat"
    assert steps[0]["count"] == 3
    assert steps[0]["raw_events"] == [0, 1, 2]


def test_r4_needs_min_count():
    """2회는 반복으로 묶지 않는다."""
    events = [_event(i, 10.0 + i, element="Zoom In") for i in range(2)]
    assert len(group_events(events, _ctx())) == 2


def test_r4_matches_by_coords_when_label_missing():
    """라벨이 없으면 좌표 근접(24px)으로 동일 대상을 판정한다."""
    events = [
        _event(i, 10.0 + i, element=None, coords={"x": 100 + i * 5, "y": 200})
        for i in range(3)
    ]
    steps = group_events(events, _ctx())
    assert steps[0]["action"] == "click_repeat"


def test_r4_does_not_mix_label_and_coords():
    """한쪽만 라벨이 있으면 묶지 않는다 - 묶임이 OCR 운에 좌우되면 재현되지 않는다."""
    events = [
        _event(0, 10.0, element="Zoom In"),
        _event(1, 11.0, element=None),
        _event(2, 12.0, element="Zoom In"),
    ]
    assert len(group_events(events, _ctx())) == 3


def test_invariant_holds_across_all_rules():
    """R1~R5 가 섞여도 불변식이 유지된다."""
    events = [
        _event(0, 10.0, element="PM", coords={"x": 800, "y": 300}),
        _event(1, 11.0, element="210", coords={"x": 810, "y": 420}),
        _event(2, 20.0, action="type_text", element=None, text="abc"),
        _event(3, 40.0, element="Zoom In"),
        _event(4, 41.0, element="Zoom In"),
        _event(5, 42.0, element="Zoom In"),
    ]
    steps = group_events(events, _ctx())
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == [0, 1, 2, 3, 4, 5]
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v -k "r3 or r4 or all_rules"`
Expected: FAIL — type_text 이벤트가 R5 로 떨어져 `assert 'click' == 'type_text'`

- [ ] **Step 3: 구현**

`grouping.py` 에 추가:

```python
def same_target(a, b, settings) -> bool:
    """두 클릭이 같은 대상을 눌렀는지 본다.

    라벨이 둘 다 있으면 라벨로, 둘 다 없으면 좌표 근접으로 판정한다. 한쪽만 있는
    경우는 같다고 보지 않는다 - 같은 버튼을 두 번 눌렀는데 한 번만 OCR 이 성공한
    경우를 억지로 묶으면 묶임 여부가 OCR 운에 좌우되어 재현되지 않는다.
    """
    label_a = (a.get("element") or "").strip()
    label_b = (b.get("element") or "").strip()
    if label_a and label_b:
        return label_a == label_b
    if label_a or label_b:
        return False
    ca, cb = a.get("coords"), b.get("coords")
    if not ca or not cb:
        return False
    dx = float(ca["x"]) - float(cb["x"])
    dy = float(ca["y"]) - float(cb["y"])
    return (dx * dx + dy * dy) <= float(settings.same_target_px) ** 2


def _rule_type_text(events, i, ctx):
    """R3 - 타이핑 구간. 직전 필드 클릭이 있으면 포커스로 흡수한다."""
    event = events[i]
    if event.get("action") == "type_text":
        return make_step(
            [event], action="type_text", rule="R3", target=event.get("element"),
            target_kind="ui_control", value=event.get("text"),
            value_source=event.get("element_source") or "none",
        ), 1

    if event.get("action") != "click" or i + 1 >= len(events):
        return None
    typing = events[i + 1]
    if typing.get("action") != "type_text":
        return None
    if float(typing["t_sec"]) - float(event["t_sec"]) > ctx.settings.focus_max_sec:
        return None
    return make_step(
        [event, typing], action="type_text", rule="R3",
        target=typing.get("element") or event.get("element"),
        target_kind="ui_control", value=typing.get("text"),
        value_source=typing.get("element_source") or "none",
    ), 2


def _rule_click_repeat(events, i, ctx):
    """R4 - 같은 대상을 짧은 창 안에 여러 번 누른 것을 하나로 묶는다."""
    first = events[i]
    if first.get("action") != "click":
        return None
    group = [first]
    for j in range(i + 1, len(events)):
        nxt = events[j]
        if nxt.get("action") != "click":
            break
        if float(nxt["t_sec"]) - float(first["t_sec"]) > ctx.settings.repeat_window_sec:
            break
        if not same_target(first, nxt, ctx.settings):
            break
        group.append(nxt)
    if len(group) < ctx.settings.repeat_min_count:
        return None
    return make_step(
        group, action="click_repeat", rule="R4", target=first.get("element"),
        count=len(group),
    ), len(group)
```

`_RULES` 를 최종 우선순위로 갱신:

```python
_RULES = [
    _rule_double_click,   # R1
    _rule_dropdown,       # R2
    _rule_type_text,      # R3
    _rule_click_repeat,   # R4
    _rule_default,        # R5
]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_grouping.py -v`
Expected: PASS (24 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/grouping.py \
        poc/workflow_3/workflow_extract/test_grouping.py
git commit -m "feat(workflow_extract): R3 타이핑+포커스 흡수, R4 반복 클릭"
```

---

### Task 11: 한국어 절차서 렌더러

**Files:**
- Create: `poc/workflow_3/workflow_extract/render.py`
- Test: `poc/workflow_3/workflow_extract/test_render.py`

**Interfaces:**
- Consumes: step dict 목록
- Produces: `render_markdown(steps, session) -> str`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
"""절차서 렌더러 테스트 - step 목록에서 한국어 markdown 을 만든다."""

from poc.workflow_3.workflow_extract.render import render_markdown

_SESSION = {"eqp_id": "MCD916", "tag": "20260811_150000",
            "capture_dir": "/x/recording", "total_events": 4, "duration_sec": 120.0}


def _step(seq, action, **kw):
    base = {
        "seq": seq, "action": action, "target": "PM", "target_kind": "ui_control",
        "value": None, "value_source": "none", "coords_in_live_box": None,
        "t_sec": [10.0, 12.0], "generation": 0, "grouping_rule": "R5",
        "inferred": False, "intent": None, "count": None, "raw_events": [seq],
        "frame": f"f_{seq}.jpg",
    }
    base.update(kw)
    return base


def test_renders_numbered_steps():
    md = render_markdown([_step(0, "click"), _step(1, "click", target="OK")], _SESSION)
    assert "1." in md and "2." in md
    assert "PM" in md and "OK" in md


def test_renders_typed_value():
    md = render_markdown(
        [_step(0, "type_text", target="Recipe Name", value="MCD916_A", value_source="ocr")],
        _SESSION,
    )
    assert "MCD916_A" in md


def test_marks_inferred_double_click():
    """추론된 더블클릭은 문서에서 추론임이 드러나야 한다."""
    md = render_markdown(
        [_step(0, "double_click", inferred=True, intent="fov_move",
               target_kind="live_image", target=None)],
        _SESSION,
    )
    assert "추론" in md


def test_footer_lists_limitations():
    """과신을 막기 위해 한계가 문서에 남아야 한다."""
    md = render_markdown([_step(0, "click")], _SESSION)
    for token in ("키 입력", "드래그", "스크롤"):
        assert token in md


def test_coverage_table_reports_rule_distribution():
    """어떤 규칙이 몇 개를 만들었는지 보여야 오작동 규칙을 지목할 수 있다."""
    md = render_markdown(
        [_step(0, "click", grouping_rule="R5"), _step(1, "click", grouping_rule="R5")],
        _SESSION,
    )
    assert "R5" in md


def test_empty_steps_still_renders_header():
    md = render_markdown([], _SESSION)
    assert "MCD916" in md
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.workflow_extract.render'`

- [ ] **Step 3: 구현**

`render.py`:

```python
"""step 목록을 엔지니어가 읽을 한국어 절차서(markdown)로 만든다.

이 문서의 목적은 자동화가 아니라 판단 근거다 - 엔지니어에게 "이게 당신이 한
절차가 맞습니까"를 물을 수 있어야 한다. 그래서 한계를 푸터에 명시한다.
"""

_ACTION_LABEL = {
    "click": "클릭",
    "double_click": "더블클릭",
    "select_from_dropdown": "드롭다운 선택",
    "type_text": "값 입력",
    "click_repeat": "반복 클릭",
}

_LIMITATIONS = [
    "키 입력은 기록하지 않는다. 화면에 렌더된 값만 OCR 로 복원했다.",
    "Enter/Tab/단축키는 관측할 수 없다.",
    "드래그는 관측할 수 없다(버튼 눌림 상태를 폴링하지 않는다).",
    "스크롤/휠은 관측할 수 없다.",
    "더블클릭은 관측이 아니라 화면 변화로부터의 추론이다.",
    "라이브 영상 위 조작은 좌표가 아니라 내용에 의존하므로 재생하려면 CV 재해석이 필요하다.",
]


def _describe(step) -> str:
    """step 하나를 한 줄 한국어로 서술한다."""
    action = _ACTION_LABEL.get(step["action"], step["action"])
    target = step.get("target")
    value = step.get("value")
    parts = []
    if target:
        parts.append(f"**{target}**")
    elif step.get("target_kind") == "live_image":
        parts.append("**라이브 SEM 영상**")
    else:
        parts.append("**(라벨 없음)**")
    parts.append(action)
    if value:
        parts.append(f"-> `{value}`")
    if step.get("count"):
        parts.append(f"({step['count']}회)")
    if step.get("inferred"):
        parts.append("_(추론)_")
    return " ".join(parts)


def _coverage_rows(steps) -> list:
    """규칙별 step 수를 센다 - 오작동 규칙을 지목할 수 있어야 한다."""
    counts = {}
    for step in steps:
        rule = step.get("grouping_rule") or "?"
        counts[rule] = counts.get(rule, 0) + 1
    return sorted(counts.items())


def render_markdown(steps, session) -> str:
    """step 목록 + 세션 정보로 절차서 markdown 문자열을 만든다."""
    lines = [
        f"# 수동 조작 절차 - {session.get('eqp_id', '?')} ({session.get('tag', '?')})",
        "",
        f"- 녹화 경로: `{session.get('capture_dir', '?')}`",
        f"- 세션 길이: {float(session.get('duration_sec') or 0.0):.1f}s",
        f"- 원본 이벤트 {session.get('total_events', 0)} 건 -> step {len(steps)} 건",
        "",
        "## 절차",
        "",
    ]
    if not steps:
        lines.append("_추출된 step 이 없습니다._")
    for step in steps:
        start, end = step["t_sec"]
        lines.append(f"{step['seq'] + 1}. [{start:.1f}s] {_describe(step)}")
    lines.extend(["", "## 규칙별 분포", "", "| 규칙 | step 수 |", "|------|---------|"])
    for rule, count in _coverage_rows(steps):
        lines.append(f"| {rule} | {count} |")
    lines.extend(["", "## 한계", ""])
    for item in _LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_render.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/workflow_extract/render.py \
        poc/workflow_3/workflow_extract/test_render.py
git commit -m "feat(workflow_extract): 한국어 절차서 렌더러 (한계 푸터 포함)"
```

---

### Task 12: 엔트리포인트 + 오류 처리

**Files:**
- Create: `poc/workflow_3/workflow_extract/extract_workflow.py`
- Test: `poc/workflow_3/workflow_extract/test_extract_workflow.py`

**Interfaces:**
- Consumes: `group_events`, `GroupingContext`, `render_markdown`, `load_workflow_extract_settings`, `debug_artifacts.save_debug_json`
- Produces: `run_extract(input_dir=None, settings=None) -> str` (상태 문자열)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
"""엔트리포인트 테스트 - 입력 3파일 로드, degrade, 종료 상태."""

import json

from poc.workflow_3.workflow_extract.extract_workflow import run_extract


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _timeline_event(seq, t_sec, action="click", element="PM"):
    return {
        "seq": seq, "t_sec": t_sec, "action": action, "coords": {"x": 100, "y": 200},
        "element": element, "element_source": "ocr", "target_kind": "ui_control",
        "region": "ui", "generation": 0, "occlusion": "none", "text": None,
        "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }


def _session(tmp_path, events):
    out = tmp_path / "recording_filter"
    _write(out / "interaction_timeline.json",
           {"capture_dir": str(tmp_path / "recording"), "events": events})
    _write(out / "region_map.json", {"maps": [{"generation": 0, "live_box": None}]})
    _write(out / "change_events.json", {"events": []})
    return out


def test_missing_timeline_is_an_error(tmp_path):
    assert run_extract(input_dir=tmp_path) == "timeline_not_found"


def test_empty_timeline_is_not_success(tmp_path):
    """이벤트 0건은 조용한 성공이 아니다."""
    out = _session(tmp_path, [])
    assert run_extract(input_dir=out) == "no_events"


def test_writes_workflow_json_and_markdown(tmp_path):
    out = _session(tmp_path, [_timeline_event(0, 10.0), _timeline_event(1, 40.0, element="OK")])
    assert run_extract(input_dir=out) == "success"
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 2
    assert (out / "workflow.md").is_file()


def test_degrades_without_region_map(tmp_path):
    """region_map.json 이 없어도 실패하지 않고 R1 만 degrade 한다."""
    out = _session(tmp_path, [_timeline_event(0, 10.0)])
    (out / "region_map.json").unlink()
    assert run_extract(input_dir=out) == "success"


def test_workflow_json_records_settings(tmp_path):
    """임계값을 바꿔가며 재실행하므로 산출물이 자기 설정을 들고 있어야 한다."""
    out = _session(tmp_path, [_timeline_event(0, 10.0)])
    run_extract(input_dir=out)
    payload = json.loads((out / "workflow.json").read_text(encoding="utf-8"))
    assert "settings" in payload
    assert payload["settings"]["recenter_min_ratio"] == 0.40
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/test_extract_workflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_3.workflow_extract.extract_workflow'`

- [ ] **Step 3: 구현**

`extract_workflow.py`:

```python
"""workflow_extract 엔트리포인트 - 타임라인을 workflow.json + workflow.md 로.

CLI 인자 없음(프로젝트 규칙). 입력은 env/자동탐색으로, 산출은 입력 폴더에 쓴다.
VLM 을 부르지 않으므로 임계값을 바꿔가며 몇 번이든 재실행할 수 있다.

실행:
    uv run python poc/workflow_3/workflow_extract/extract_workflow.py
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.debug_artifacts import save_debug_json
from poc.workflow_3.workflow_extract.grouping import GroupingContext, group_events
from poc.workflow_3.workflow_extract.render import render_markdown
from poc.workflow_3.workflow_extract.settings import load_workflow_extract_settings


def _read_json(path):
    """JSON 을 읽는다. 없거나 깨졌으면 None (호출부가 degrade 를 결정한다)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_input_dir():
    """분석할 recording_filter/ 폴더를 결정한다(env -> 자동탐색)."""
    env_path = os.getenv("WORKFLOW_EXTRACT_INPUT_DIR", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] WORKFLOW_EXTRACT_INPUT_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None
    candidates = sorted(
        ALIGN_IMAGES_DIR.glob("*/_manual/*/recording_filter"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        latest = candidates[0].resolve()
        print(f"[INFO] 최신 recording_filter/ 자동 선택: {latest}")
        return latest
    print(f"[ERROR] 분석할 recording_filter/ 폴더를 찾지 못했습니다(루트: {ALIGN_IMAGES_DIR}).")
    return None


def _load_live_boxes(out_dir):
    """region_map.json 에서 {generation: live_box} 를 만든다. 없으면 빈 dict."""
    payload = _read_json(Path(out_dir) / "region_map.json")
    if not payload:
        print("[WARNING] region_map.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return {}
    boxes = {}
    for entry in payload.get("maps") or []:
        if entry.get("live_box"):
            boxes[int(entry.get("generation") or 0)] = entry["live_box"]
    return boxes


def _load_changes(out_dir):
    """change_events.json 의 events 를 돌려준다. 없으면 빈 목록."""
    payload = _read_json(Path(out_dir) / "change_events.json")
    if not payload:
        print("[WARNING] change_events.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return []
    return payload.get("events") or []


def _frame_size(capture_dir):
    """프레임 하나를 열어 (w, h) 를 얻는다. 실패하면 None (R2 degrade)."""
    from poc.workflow_3.recording_filter.region_gate import read_frame_size

    try:
        for frame in sorted(Path(capture_dir).glob("*.jpg")):
            size = read_frame_size(frame)
            if size:
                return size
    except Exception:
        pass
    print("[WARNING] 프레임 크기를 얻지 못했습니다 - R2(드롭다운)가 비활성화됩니다.")
    return None


def run_extract(*, input_dir=None, settings=None) -> str:
    """타임라인을 workflow.json + workflow.md 로 만든다. 상태 문자열 반환."""
    settings = settings or load_workflow_extract_settings()
    out_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if out_dir is None:
        return "input_not_found"

    timeline_payload = _read_json(out_dir / "interaction_timeline.json")
    if not timeline_payload:
        print(
            f"[ERROR] interaction_timeline.json 이 없습니다: {out_dir}\n"
            "        먼저 filter_recording.py 를 실행하세요."
        )
        return "timeline_not_found"

    events = timeline_payload.get("events") or []
    if not events:
        print("[ERROR] 타임라인에 이벤트가 0건입니다 - 추출할 절차가 없습니다.")
        return "no_events"

    capture_dir = timeline_payload.get("capture_dir") or ""
    ctx = GroupingContext(
        settings=settings,
        live_boxes=_load_live_boxes(out_dir),
        changes=_load_changes(out_dir),
        frame_wh=_frame_size(capture_dir) if capture_dir else None,
    )
    steps = group_events(events, ctx)

    duration = max(float(e["t_sec"]) for e in events) if events else 0.0
    session = {
        "eqp_id": Path(capture_dir).parts[-4] if len(Path(capture_dir).parts) >= 4 else "?",
        "tag": Path(capture_dir).parent.name if capture_dir else "?",
        "capture_dir": capture_dir,
        "total_events": len(events),
        "duration_sec": duration,
    }

    save_debug_json(
        out_dir / "workflow.json",
        {"session": session, "settings": asdict(settings), "steps": steps},
    )
    (out_dir / "workflow.md").write_text(
        render_markdown(steps, session), encoding="utf-8"
    )
    print(
        f"[INFO] 완료: 이벤트 {len(events)} 건 -> step {len(steps)} 건, out={out_dir}"
    )
    return "success"


if __name__ == "__main__":
    result = run_extract()
    raise SystemExit(0 if result == "success" else 1)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_3/workflow_extract/ -v`
Expected: PASS (전체 통과)

- [ ] **Step 5: 전체 테스트 스위트 확인**

Run: `uv run pytest poc/workflow_3/recording_filter poc/workflow_3/workflow_extract -v`
Expected: PASS — 기존 71개 + 신규 전부. 기존 테스트가 깨지면 Stage 2a/2b 가 기존 경로를 침범한 것이다.

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/workflow_extract/extract_workflow.py \
        poc/workflow_3/workflow_extract/test_extract_workflow.py
git commit -m "feat(workflow_extract): 엔트리포인트 + 입력 3파일 degrade + 종료 상태"
```

---

### Task 13: 문서 갱신

**Files:**
- Modify: `CLAUDE.md`
- Modify: `poc/workflow_3/README.md`

- [ ] **Step 1: `CLAUDE.md` 의 Repository Structure 에 한 줄 추가**

`poc/workflow_3/recording_filter/` 줄 아래에:

```
poc/workflow_3/workflow_extract/ # offline: interaction timeline -> 의미 단위 workflow.json + 한국어 절차서 (VLM 콜 0)
```

- [ ] **Step 2: `CLAUDE.md` 의 Testing 절에 실행 명령 추가**

```bash
# workflow_extract — 그룹핑/렌더 단위 테스트 (VLM 불필요, Mac 실행 가능)
uv run pytest poc/workflow_3/workflow_extract
```

- [ ] **Step 3: `CLAUDE.md` 의 Running Modules 절에 실행 명령 추가**

```bash
WORKFLOW_EXTRACT_INPUT_DIR=<recording_filter 경로> \
  uv run python poc/workflow_3/workflow_extract/extract_workflow.py   # timeline -> workflow.json + workflow.md
```

- [ ] **Step 4: 수동 녹화 절에 Stage 2a/2b 변화를 한 문단으로 반영**

`**엔지니어 수동 조작 녹화**` 항목의 사이드카 설명 뒤에 추가:

```
- **Stage 2a 사이드카 커서 + Stage 2b 타이핑** (2026-08-11) — 사이드카에 커서가 있으면
  Stage 2a 가 VLM 커서 탐지를 건너뛴다(`cursor_source` 필드로 구분; 알람 녹화는 사이드카가
  없어 기존 VLM 경로 그대로). Stage 2b 는 **커서 정지 + 국소 반복 변화**로 타이핑 구간을 찾아
  구간 시작/끝 OCR 2콜로 값을 복원하고, before == after 면 캐럿 깜빡임으로 보고 버린다.
  `MANUAL_RECORD_*` 가 아니라 `RECORDING_FILTER_TYPING_*` 네임스페이스다.
```

- [ ] **Step 5: 커밋**

```bash
git add CLAUDE.md poc/workflow_3/README.md
git commit -m "docs: workflow_extract 패키지 + Stage 2a/2b 변경 반영"
```

---

## 오피스 실행 순서 (구현 완료 후)

각 단계는 앞 단계가 정상일 때만 의미가 있다.

1. `RECORDING_FILTER_MAX_VLM_CALLS=300 uv run python poc/workflow_3/recording_filter/filter_recording.py`
2. **`region_map_gen0.jpg` 의 시안 박스가 실제 라이브 SEM 영역과 맞는지 확인.** 틀리면 여기서 멈춘다 — R1 과 게이트 판정 전부가 무효다.
3. `summary.json` 확인 — `gate_passed / total_change_events` (90%+ 제거가 정상), `typing_bursts` / `typing_events`, 그리고 타임라인의 `cursor_source` 분포(수동 세션인데 `vlm` 이 대부분이면 사이드카 조인이 깨진 것이다).
4. `uv run python poc/workflow_3/workflow_extract/extract_workflow.py`
5. **`workflow.md` 를 엔지니어와 함께 읽고 묻는다: "이게 당신이 한 절차가 맞습니까?"** 이번 작업의 산출물은 자동화가 아니라 이 질문에 대한 답이다.

## 첫 실측 후 조정 대상

전부 blind 로 정한 값이라 첫 세션 결과를 보고 조정한다.

| 값 | 기본 | 조정 신호 |
|----|------|-----------|
| `recenter_min_ratio` | 0.40 | 라이브 박스 단발 클릭이 더블클릭으로 잡히면 올린다 |
| `typing_min_burst_events` | 3 | 짧은 입력이 안 잡히면 내린다 |
| `repeat_min_count` | 3 | 반복이 과하게 묶이면 올린다 |
| R2 드롭다운 비율 | PM 전용 | PM 외 드롭다운을 놓치면 일반 비율셋을 만든다 |
