"""filter_recording e2e — 합성 녹화 폴더 + 가짜 client 로 산출물 검증."""

import base64
import io
import json

import cv2
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


class _NoCursorClient:
    """닫기 후보에서 커서가 합쳐져 보이지 않는 Stage 2a 응답."""

    def chat_with_image_b64(self, **kwargs):
        return _FakeResponse(json.dumps({
            "cursor_visible": False,
            "cursor_kind": None,
            "cursor_bbox": None,
            "confidence": 0.0,
            "evidence": "cursor merged with title-bar edge",
        }))


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


def _close_candidate_recording(tmp_path):
    """Task 1의 세 gate를 만족하는 닫기 후보 녹화를 만든다."""
    rec = tmp_path / "close_candidate" / "recording"
    rec.mkdir(parents=True)
    prev = np.full((400, 600), 240, dtype=np.uint8)
    cv2.line(prev, (580, 10), (590, 20), 40, 2)
    cv2.line(prev, (590, 10), (580, 20), 40, 2)
    curr = prev.copy()
    cv2.line(curr, (560, 12), (578, 30), 10, 3)
    cv2.line(curr, (578, 12), (560, 30), 10, 3)
    cv2.imwrite(str(rec / "tag_rcs_0000_00000000ms.jpg"), prev)
    cv2.imwrite(str(rec / "tag_rcs_0001_00000300ms.jpg"), curr)
    (rec / "recording_manifest.json").write_text(
        json.dumps({"stop_reason": "window_gone"}), encoding="utf-8"
    )
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


def test_run_filter_records_probable_close_click(tmp_path):
    rec = _close_candidate_recording(tmp_path)
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
    assert timeline[0]["replayable"] is False
    assert (out_dir / "close_click_evidence" / "probable_close_click.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["probable_close_clicks"] == 1


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


# ---------------------------------------------------------------------------
# 2026-08-10 최종 리뷰 FINDING 1/2/3/6/7 - 게이트/가림/region/집계/복사량.
# ---------------------------------------------------------------------------

class _StubDetection:
    """detect_sem_box 대역 - 라이브 박스를 고정 좌표로 준다."""

    def __init__(self, bbox_px):
        self.detected = bbox_px is not None
        self.bbox_px = bbox_px


def _write_sidecar(rec, records):
    """frame_meta.jsonl 을 녹화 루트에 쓴다(FrameMetaWriter 와 같은 위치/형식)."""
    lines = [json.dumps(rec_item, ensure_ascii=False) for rec_item in records]
    (rec / "frame_meta.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stub_sem_box(monkeypatch, bbox_px):
    """region_gate 가 함수 안에서 import 하는 detect_sem_box 를 바꿔치기한다."""
    import poc.workflow_3.sem_monitor.sem_box_detect as sem_box_detect_mod

    monkeypatch.setattr(
        sem_box_detect_mod, "detect_sem_box",
        lambda _image, _client, **_kwargs: _StubDetection(bbox_px),
    )


def test_click_inside_live_box_is_reported_as_live_image(tmp_path, monkeypatch):
    """라이브 SEM 영상 안 클릭은 target_kind="live_image" 로 나와야 한다.

    두 가지를 한 번에 건다.
    - FINDING 3: region 을 verdict 에서 파생하면 살아남는 이벤트는 전부 candidate
      라 region 이 항상 "ui" 가 되고 live_image 분기가 죽는다.
    - FINDING 2: 사이드카 rect(480x320) 와 프레임(600x400) 이 1.25배 차이나므로,
      배율 보정이 없으면 커서 프레임좌표가 (240,160) 으로 계산돼 라이브 박스
      (left=280) 밖으로 오판된다.
    """
    rec = _recording_dir(tmp_path)
    _stub_sem_box(monkeypatch, {"left": 280, "top": 100, "right": 600, "bottom": 400})
    # rect 는 480x320(논리) / 프레임은 600x400(물리) -> 125% 배율.
    rect = {"left": 0, "top": 0, "right": 480, "bottom": 320}
    _write_sidecar(rec, [
        {"frame": f"seq_{i}", "t_sec": t, "window_rect": rect,
         "foreground_title": "Remote Monitoring System - MCD916",
         "occlusion": "none", "cursor_screen_xy": [240, 160], "cursor_in_window": True}
        for i, t in enumerate((0.0, 0.3, 0.6))
    ])

    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=5000, element_label_enabled=False,
    )
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status == "success", status

    out_dir = rec.parent / "recording_filter"
    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    assert len(timeline["events"]) == 1, timeline
    event = timeline["events"][0]
    assert event["region"] == "live_image", event
    assert event["target_kind"] == "live_image", event

    # FINDING 6 - VLM 호출 집계는 스테이지별로 분해되어 있어야 한다.
    # (2026-08-11 리뷰 I3) 이 세션은 사이드카가 있어 Stage 2a 가 VLM 을 부르지
    # 않는다 - 예전 집계(len(click_events))는 일어나지 않은 콜 1건을 청구했다.
    # 실제 콜은 Stage 1.5 영역 지도 1건뿐이다.
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "vlm_calls" not in summary, summary
    assert summary["vlm_calls_stage1_5_region_map"] == 1, summary
    assert summary["vlm_calls_stage2a_cursor"] == 0, summary
    assert summary["cursor_from_sidecar"] == 1, summary
    assert summary["vlm_calls_total_estimate"] == 1, summary


def test_run_filter_reports_when_occlusion_discards_everything(tmp_path, monkeypatch):
    """가림으로 이벤트가 전멸하면 성공 형태 상태를 돌려주면 안 된다(FINDING 1).

    수정 전에는 timeline 이 비어도 "no_clicks"(exit 0) 라, 사이드카 버그로 모든
    프레임이 "full" 로 찍힌 세션이 조용히 성공처럼 끝났다.
    """
    rec = _recording_dir(tmp_path)
    _stub_sem_box(monkeypatch, None)
    rect = {"left": 0, "top": 0, "right": 600, "bottom": 400}
    _write_sidecar(rec, [
        {"frame": f"seq_{i}", "t_sec": t, "window_rect": rect,
         "foreground_title": "Notepad", "occlusion": "full",
         "cursor_screen_xy": [10, 10], "cursor_in_window": False}
        for i, t in enumerate((0.0, 0.3, 0.6))
    ])

    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=5000, element_label_enabled=False,
    )
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status == "all_events_discarded", status

    out_dir = rec.parent / "recording_filter"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["occluded_events_excluded"] == 1, summary
    assert summary["total_change_events"] == 1, summary
    assert summary["gate_passed"] == 0, summary

    # 감사 추적(change_events.json)은 Stage 1 전체를 유지한다.
    change_events = json.loads((out_dir / "change_events.json").read_text(encoding="utf-8"))
    assert len(change_events["events"]) == 1, change_events
    # FINDING 7 - 게이트가 버린 프레임은 디스크로 복사하지 않는다.
    assert list((out_dir / "change_events").glob("*.jpg")) == []


def test_change_event_copies_only_gate_survivors(tmp_path, monkeypatch):
    """게이트를 통과한 프레임만 change_events/ 로 복사된다(FINDING 7).

    Stage 1 은 2건(ambient 1 + candidate 1)을 내고 게이트가 1건을 걷어낸다.
    복사가 게이트 앞에 있으면 2장이 복사돼 이 테스트가 실패한다.
    """
    rec = tmp_path / "tag_copy" / "recording"
    live_box = {"left": 300, "top": 200, "right": 600, "bottom": 400}
    base = np.full((400, 600), 30, dtype=np.uint8)
    in_live = base.copy()
    in_live[240:350, 340:500] = 255           # 라이브 박스 안에서만 변화 -> ambient
    in_ui = in_live.copy()
    in_ui[20:120, 20:180] = 255               # UI 영역 변화 -> candidate
    _write(rec / "tag_rcs_0000_00000000ms.jpg", base)
    _write(rec / "tag_rcs_0001_00000300ms.jpg", in_live)
    _write(rec / "tag_rcs_0002_00000600ms.jpg", in_live.copy())   # 변화 없음
    _write(rec / "tag_rcs_0003_00000900ms.jpg", in_ui)

    rect = {"left": 0, "top": 0, "right": 600, "bottom": 400}     # 100% 배율
    _write_sidecar(rec, [
        {"frame": f"seq_{i}", "t_sec": t, "window_rect": rect,
         "foreground_title": "x", "occlusion": "none",
         "cursor_screen_xy": [10, 10], "cursor_in_window": True}   # 커서는 라이브 박스 밖
        for i, t in enumerate((0.0, 0.3, 0.6, 0.9))
    ])
    _stub_sem_box(monkeypatch, live_box)

    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=5000, element_label_enabled=False,
    )
    # 클릭으로 이어질지는 이 테스트의 관심사가 아니다(게이트 통과분이 있으면 충분).
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status in {"success", "no_clicks"}, status

    out_dir = rec.parent / "recording_filter"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_change_events"] == 2, summary
    assert summary["ambient_events_dropped"] == 1, summary
    assert summary["gate_passed"] == 1, summary

    copied = list((out_dir / "change_events").glob("*.jpg"))
    assert len(copied) == 1, copied
    # 감사 추적은 Stage 1 전체(2건)를 유지한다.
    change_events = json.loads((out_dir / "change_events.json").read_text(encoding="utf-8"))
    assert len(change_events["events"]) == 2, change_events


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


class _CropAwareVlmClient:
    """Stage 2c/2b 가 내부에서 만드는 실제 Workflow1VLMClient 를 대체한다.

    filter_recording 은 Stage 2c/2b 의 OCR client 를 함수 안에서 직접
    `Workflow1VLMClient(service_slug)` 로 만들기 때문에(run_filter 의 `client`
    인자로는 주입되지 않는다), 이 클래스로 그 심볼 자체를 monkeypatch 한다.
    호출마다 다른 이미지를 받으므로, crop 크기/평균 밝기로 어느 스테이지의
    어느 콜인지 구분해 응답한다 - 몇 번째 호출인지에 의존하면 스테이지 순서가
    바뀌어도 우연히 통과할 수 있어 순서를 못 박지 못한다.
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def chat_with_image_b64(self, *, image_b64, **_kwargs):
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if (width, height) == (260, 260):
            # Stage 2c 의 요소 crop(settings.element_crop_px=260, 클램프 없음).
            payload = [
                {"text": "Start", "bbox": {"left": 100, "top": 100, "right": 160, "bottom": 130}},
            ]
        else:
            # Stage 2b 의 타이핑 ROI crop - 채움 값(밝기)으로 전/후를 가른다.
            mean = float(np.array(image.convert("L")).mean())
            text = "before123" if mean < 128 else "after456"
            payload = [{"text": text, "bbox": {"left": 0, "top": 0, "right": 20, "bottom": 20}}]
        return _FakeResponse(json.dumps(payload))


def _typing_session(tmp_path, *, typing_box, times, cursor_after_click):
    """클릭 1건 + 같은 자리에서 3회 변화하는 합성 세션을 만든다.

    typing_box 는 (top, bottom, left, right) native 픽셀. 채움 값이 90->150->210 으로
    바뀌므로 ROI OCR 대역이 전/후를 밝기로 구분할 수 있다.
    """
    rec = tmp_path / "recording"
    width, height = 600, 400
    base = np.full((height, width), 30, dtype=np.uint8)

    click = base.copy()
    click[80:180, 250:360] = 255                      # 클릭 변화(F0->F1)

    top, bottom, left, right = typing_box
    frames = [base, click]
    prev = click
    for fill in (90, 150, 210):
        nxt = prev.copy()
        nxt[top:bottom, left:right] = fill
        frames.append(nxt)
        prev = nxt

    for i, (frame, t_sec) in enumerate(zip(frames, times)):
        _write(rec / f"tag_rcs_{i:04d}_{int(t_sec * 1000):08d}ms.jpg", frame)

    rect = {"left": 0, "top": 0, "right": width, "bottom": height}   # 100% 배율
    cursors = [[300, 130], [300, 130]] + [list(cursor_after_click)] * 3
    _write_sidecar(rec, [
        {"frame": f"seq_{i}", "t_sec": t_sec, "window_rect": rect, "foreground_title": "x",
         "occlusion": "none", "cursor_screen_xy": cursor, "cursor_in_window": True}
        for i, (t_sec, cursor) in enumerate(zip(times, cursors))
    ])
    return rec


def test_typing_burst_rejected_when_change_is_far_from_focus_click(tmp_path, monkeypatch):
    """커서가 멈춘 채 **멀리 떨어진** 영역이 반복 변화하면 타이핑이 아니다(리뷰 C2).

    이 테스트는 예전에 정반대를 단언했다: 커서를 (500,50) 에 세워 두고 250px 떨어진
    영역을 3번 바꾼 뒤 `type_text(element="Start", text="after456")` 가 나오기를
    기대했다. 그것이 바로 최악의 실패 형태다 - 엔지니어가 **Start 를 누르고 지켜보는**
    동안 진행률 패널이 리페인트되면, 문서에 "Start 값 입력 -> 7 / 20" 이 value_source
    "ocr", confidence 1.0 으로 실린다. 하지도 않은 조작이 가장 확신에 찬 얼굴로.

    이제는 변화가 필드 기준점(포커스 클릭 좌표) 근처여야 하므로 구간이 생기지 않고,
    버린 사실을 경고로 남긴다.
    """
    rec = _typing_session(
        tmp_path, typing_box=(300, 370, 430, 530),
        times=(0.0, 0.3, 0.8, 1.0, 1.2), cursor_after_click=(500, 50),
    )
    monkeypatch.setattr(
        "poc.workflow_3.vlm.vlm_client.Workflow1VLMClient", _CropAwareVlmClient
    )
    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=3000, region_gate_enabled=False,
    )
    assert run_filter(input_dir=rec, settings=settings, client=_FakeClient()) == "success"

    out_dir = rec.parent / "recording_filter"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["typing_bursts"] == 0, summary
    assert summary["typing_events"] == 0, summary
    assert summary["vlm_calls_stage2b_ocr"] == 0, summary

    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    assert [ev["action"] for ev in timeline["events"] if ev["action"] == "type_text"] == []


def test_typing_burst_accepted_when_change_overlaps_focus_click(tmp_path, monkeypatch):
    """필드를 클릭한 자리에서 변화가 반복되면 타이핑 구간이 된다(C2 긍정 경로).

    동시에 두 가지를 더 못박는다.
    - Stage 2b 가 Stage 2c **뒤**에 돌아야만 성립하는 단언(element="Start"). 순서가
      바뀌면 이 시점의 labels 가 비어 있어 라벨이 붙지 않는다.
    - 리뷰 I4: 타이핑 구간 프레임들은 커서 ROI 변화 임계도 함께 넘겨 Stage 2a 가
      클릭으로도 판정한다. 억제하지 않으면 같은 구간이 "값 입력" 1건 +
      "반복 클릭 3회" 로 두 번 보고된다. 타임라인에는 진짜 필드 클릭 1건과
      type_text 1건만 남아야 한다.
    """
    rec = _typing_session(
        tmp_path, typing_box=(100, 170, 250, 360),
        # 클릭(0.3s) 과 타이핑 시작(2.0s) 사이를 idle 상한(1.5s) 보다 벌려, 클릭
        # 프레임이 구간에 흡수되지 않고 자체 이벤트로 남게 한다.
        times=(0.0, 0.3, 2.0, 2.3, 2.6), cursor_after_click=(300, 130),
    )
    monkeypatch.setattr(
        "poc.workflow_3.vlm.vlm_client.Workflow1VLMClient", _CropAwareVlmClient
    )
    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=3000, region_gate_enabled=False,
    )
    assert run_filter(input_dir=rec, settings=settings, client=_FakeClient()) == "success"

    out_dir = rec.parent / "recording_filter"
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["typing_bursts"] == 1, summary
    assert summary["typing_events"] == 1, summary
    assert summary["vlm_calls_stage2b_ocr"] == 2, summary
    assert summary["clicks_superseded_by_typing"] == 3, summary
    # 사이드카 세션이므로 Stage 2a 는 VLM 을 부르지 않는다(리뷰 I3).
    assert summary["vlm_calls_stage2a_cursor"] == 0, summary

    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    actions = [ev["action"] for ev in timeline["events"]]
    assert actions.count("type_text") == 1, timeline
    assert actions.count("click") == 1, timeline
    typing = next(ev for ev in timeline["events"] if ev["action"] == "type_text")
    assert typing["element"] == "Start", typing     # Stage 2c 라벨이 실려 있어야 한다.
    assert typing["text"] == "after456", typing
    # 타이핑 coords 는 클릭과 같은 프레임 좌표계여야 한다(화면 좌표 혼입 금지).
    assert typing["coords"] == {"x": 300, "y": 130}, typing


def test_run_filter_finds_sidecar_in_capture_root_when_frames_are_nested(tmp_path, monkeypatch):
    """frames/ 하위에 프레임이 있어도 사이드카는 녹화 루트에서 찾는다(FINDING 8)."""
    from poc.workflow_3.recording_filter.filter_recording import _resolve_meta_dir

    rec = tmp_path / "tag_nested" / "recording"
    frames = rec / "frames"
    base = np.full((400, 600), 30, dtype=np.uint8)
    _write(frames / "tag_rcs_0000_00000000ms.jpg", base)
    _write(frames / "tag_rcs_0001_00000300ms.jpg", base)
    _write_sidecar(rec, [
        {"frame": "seq_0", "t_sec": 0.0,
         "window_rect": {"left": 0, "top": 0, "right": 600, "bottom": 400},
         "foreground_title": "x", "occlusion": "none",
         "cursor_screen_xy": [10, 10], "cursor_in_window": True},
    ])

    assert _resolve_meta_dir(rec, frames) == rec


def test_module_call_cap_applies_without_env(monkeypatch):
    """env 없이 실행하면 모듈 상수 MAX_VLM_CALLS 가 상한이 된다(긴 env 한 줄 불필요)."""
    from poc.workflow_3.recording_filter import filter_recording as fr

    monkeypatch.delenv("RECORDING_FILTER_MAX_VLM_CALLS", raising=False)
    monkeypatch.setattr(fr, "MAX_VLM_CALLS", 300)
    assert fr._load_settings_with_call_cap().max_vlm_calls == 300


def test_env_call_cap_beats_module_constant(monkeypatch):
    """실제 shell env 는 항상 이긴다 - 한 번만 다르게 돌릴 방법이 남아야 한다."""
    from poc.workflow_3.recording_filter import filter_recording as fr

    monkeypatch.setenv("RECORDING_FILTER_MAX_VLM_CALLS", "50")
    monkeypatch.setattr(fr, "MAX_VLM_CALLS", 300)
    assert fr._load_settings_with_call_cap().max_vlm_calls == 50


def test_env_zero_can_restore_unlimited(monkeypatch):
    """env 로 0(무제한)을 명시하면 모듈 상수가 그걸 덮지 않는다."""
    from poc.workflow_3.recording_filter import filter_recording as fr

    monkeypatch.setenv("RECORDING_FILTER_MAX_VLM_CALLS", "0")
    monkeypatch.setattr(fr, "MAX_VLM_CALLS", 300)
    assert fr._load_settings_with_call_cap().max_vlm_calls == 0


def test_injected_settings_are_not_capped(tmp_path, monkeypatch):
    """settings 를 주입하면 모듈 상수를 적용하지 않는다(주입값이 곧 계약)."""
    from poc.workflow_3.recording_filter import filter_recording as fr

    monkeypatch.delenv("RECORDING_FILTER_MAX_VLM_CALLS", raising=False)
    monkeypatch.setattr(fr, "MAX_VLM_CALLS", 300)
    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, min_change_area_px=5000,
        element_label_enabled=False, max_vlm_calls=0,
    )
    run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    summary = json.loads(
        (rec.parent / "recording_filter" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["max_vlm_calls"] == 0, summary
