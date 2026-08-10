"""timeline 테스트 — 시간순 정렬 + 스키마 + 오버레이 생성."""

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.click_detect import ClickEvent
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.timeline import build_timeline, derive_target_kind, write_click_overlays


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


def test_derive_target_kind_ui_control():
    """ui 영역 + 라벨 있음 -> 이식 가능한 ui_control."""
    assert derive_target_kind("ui", "ocr") == "ui_control"
    assert derive_target_kind("ui", "vlm") == "ui_control"


def test_derive_target_kind_live_image():
    """라이브 영상 위 조작은 라벨 유무와 무관하게 live_image."""
    assert derive_target_kind("live_image", "ocr") == "live_image"
    assert derive_target_kind("live_image", "none") == "live_image"


def test_derive_target_kind_unknown():
    """라벨이 없으면 사람이 봐야 한다."""
    assert derive_target_kind("ui", "none") == "unknown"
    assert derive_target_kind("unknown", "none") == "unknown"


class _Change:
    def __init__(self, rank, t):
        self.rank = rank
        self.frame_path = f"/tmp/f{rank}.jpg"
        self.prev_frame_path = f"/tmp/f{rank - 1}.jpg"
        self.timestamp_sec = t


class _Click:
    def __init__(self, rank, t):
        self.change = _Change(rank, t)
        self.status = "click"
        self.is_click = True
        self.cursor_xy = [100, 200]
        self.confidence = 0.8

    @property
    def frame_path(self):
        return self.change.frame_path

    @property
    def prev_frame_path(self):
        return self.change.prev_frame_path

    @property
    def timestamp_sec(self):
        return self.change.timestamp_sec

    @property
    def rank(self):
        return self.change.rank


def test_timeline_carries_new_fields():
    """게이트/라벨 정보가 이벤트에 실린다."""
    from poc.workflow_3.recording_filter.element_label import ElementLabel

    clicks = [_Click(rank=0, t=1.0)]
    gate_info = {0: {"generation": 2, "region": "ui", "occlusion": "none"}}
    labels = {0: ElementLabel(text="Start", source="ocr", confidence=1.0)}

    events = build_timeline(clicks, gate_info=gate_info, labels=labels)

    assert len(events) == 1
    ev = events[0]
    assert ev["element"] == "Start"
    assert ev["element_source"] == "ocr"
    assert ev["target_kind"] == "ui_control"
    assert ev["region"] == "ui"
    assert ev["generation"] == 2
    assert ev["occlusion"] == "none"


def test_timeline_defaults_without_gate_or_labels():
    """게이트/라벨 정보가 없어도 기존 스키마로 동작한다(하위 호환)."""
    events = build_timeline([_Click(rank=0, t=1.0)])
    ev = events[0]
    assert ev["element"] is None
    assert ev["element_source"] == "none"
    assert ev["target_kind"] == "unknown"
    assert ev["region"] == "unknown"
    assert ev["generation"] == 0
    assert ev["occlusion"] == "unknown"
