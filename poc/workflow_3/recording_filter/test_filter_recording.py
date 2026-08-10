"""filter_recording e2e — 합성 녹화 폴더 + 가짜 client 로 산출물 검증."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.click_detect import ClickEvent
from poc.workflow_3.recording_filter.filter_recording import _label_click_events, run_filter
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.recording_filter.timeline import build_timeline


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


class _SpottingOcrClient:
    """OCR 대역 - 클릭 지점 crop 안에 텍스트 항목이 하나 있다고 답한다(성공 경로)."""

    def chat_with_image_b64(self, **kwargs):
        payload = json.dumps([
            {"text": "Start", "bbox": {"left": 100, "top": 100, "right": 200, "bottom": 130}},
        ])
        return _FakeResponse(payload)


class _NeverCalledClient:
    """OCR 이 성공하면 절대 불려서는 안 되는 VLM 대역(호출되면 즉시 실패)."""

    def chat_with_image_b64(self, **kwargs):
        raise AssertionError("OCR 이 성공했는데 VLM 폴백이 호출됐다")


def _click_event_with_frame(rank, frame_path):
    """지정한 frame_path 를 그대로 쓰는 ClickEvent 를 만든다(존재 여부는 호출부 책임)."""
    change = ChangeEvent(
        rank=rank, frame_path=frame_path, prev_frame_path=frame_path,
        timestamp_sec=float(rank), frame_index=rank,
        change_bbox={"left": 0, "top": 0, "right": 10, "bottom": 10},
        largest_blob_area_px=100, changed_pixels=100,
    )
    return ClickEvent(
        change=change, is_click=True, status="click",
        cursor_visible=True, cursor_kind="rcs_black_arrow",
        cursor_bbox={"left": 290, "top": 120, "right": 310, "bottom": 140},
        cursor_xy=[300, 130], click_window={"left": 200, "top": 30, "right": 400, "bottom": 230},
        changed_in_window_px=9000, confidence=0.9, evidence="x",
    )


def test_label_click_events_isolates_bad_frame_from_the_rest(tmp_path):
    """가운데 이벤트의 프레임이 없어도 앞뒤 두 이벤트는 정상 라벨링된다.

    수정 전에는 Image.open() 실패가 루프 밖으로 그대로 던져져 나머지 이벤트 처리와
    interaction_timeline.json/summary.json 기록 자체를 막았다. _label_click_events 는
    이벤트 하나의 실패를 격리해야 한다 - 이 테스트가 실패한다면 그 격리가 깨진 것이다.
    """
    good = np.full((400, 600), 30, dtype=np.uint8)

    def _write_frame(rank):
        path = tmp_path / f"frame_{rank}.jpg"
        Image.fromarray(good, mode="L").save(path, format="JPEG")
        return str(path.resolve())

    events = [
        _click_event_with_frame(0, _write_frame(0)),
        _click_event_with_frame(1, str(tmp_path / "missing_frame.jpg")),  # 존재하지 않음
        _click_event_with_frame(2, _write_frame(2)),
    ]

    settings = RecordingFilterSettings(element_crop_px=260)
    labels, label_errors = _label_click_events(
        events, settings, tmp_path / "element_crops",
        ocr_client=_SpottingOcrClient(), vlm_client=_NeverCalledClient(),
    )

    assert label_errors == 1, label_errors
    assert set(labels.keys()) == {0, 2}, labels.keys()
    assert labels[0].source == "ocr" and labels[0].text == "Start", labels[0]
    assert labels[2].source == "ocr" and labels[2].text == "Start", labels[2]

    # build_timeline 이 실패한 이벤트에 문서화된 기본값을 채우는지까지 사슬로 확인한다.
    timeline = build_timeline(events, labels=labels)
    assert len(timeline) == 3, timeline
    by_frame = {ev["frame"]: ev for ev in timeline}

    failed_frame = "missing_frame.jpg"
    assert failed_frame in by_frame, by_frame.keys()
    failed_ev = by_frame[failed_frame]
    assert failed_ev["element"] is None
    assert failed_ev["element_source"] == "none"
    assert failed_ev["target_kind"] == "unknown"

    for rank, frame_name in ((0, "frame_0.jpg"), (2, "frame_2.jpg")):
        ok_ev = by_frame[frame_name]
        assert ok_ev["element"] == "Start", (rank, ok_ev)
        assert ok_ev["element_source"] == "ocr", (rank, ok_ev)


def test_run_filter_survives_element_label_failure(tmp_path, monkeypatch):
    """Stage 2c 라벨링이 통째로 실패해도 run_filter 는 죽지 않고 산출물을 남긴다.

    _label_one_click 을 강제로 예외를 던지게 바꿔치기해, run_filter 전체가 그
    예외를 흡수하고 interaction_timeline.json/summary.json 을 정상 기록하는지 본다.
    수정 전 코드였다면 이 monkeypatch 만으로 run_filter 가 예외를 던져 이 테스트가
    실패했을 것이다.
    """
    import poc.workflow_3.recording_filter.filter_recording as filter_recording_mod

    def _always_fails(*_args, **_kwargs):
        raise RuntimeError("강제 실패(테스트)")

    monkeypatch.setattr(filter_recording_mod, "_label_one_click", _always_fails)

    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, min_change_area_px=5000)
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status == "success"

    out_dir = rec.parent / "recording_filter"
    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(timeline["events"]) == 1                # 실패해도 이벤트 자체는 남는다
    ev = timeline["events"][0]
    assert ev["element"] is None
    assert ev["element_source"] == "none"

    assert summary["element_label_errors"] == 1, summary
    assert summary["labeled"] == 0, summary
