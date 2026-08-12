import json

import cv2
import numpy as np

from poc.workflow_3.recording_filter.click_detect import ClickEvent
from poc.workflow_3.recording_filter.close_click_evidence import (
    infer_probable_close_click,
    write_close_click_evidence,
)
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent


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


def test_writes_close_click_evidence_frame_and_json(tmp_path):
    rec, change, click = _close_candidate_fixture(tmp_path)
    event = infer_probable_close_click(rec, [change], [click])

    written = write_close_click_evidence(event, change, tmp_path / "evidence")

    assert {path.name for path in written} == {
        "probable_close_click.jpg", "probable_close_click.json",
    }
    assert all(path.is_file() for path in written)
    payload = json.loads((tmp_path / "evidence" / "probable_close_click.json").read_text())
    assert payload["action"] == "probable_close_click"
