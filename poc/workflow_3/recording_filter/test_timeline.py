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
