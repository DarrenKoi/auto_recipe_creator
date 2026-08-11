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


from poc.workflow_3.recording_filter.click_detect import resolve_sidecar_cursor
from poc.workflow_3.recording_filter.region_gate import FrameMeta


def _change_event(rank, t_sec):
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/cd_{rank}.jpg", prev_frame_path=f"/tmp/cd_prev_{rank}.jpg",
        timestamp_sec=t_sec, frame_index=rank,
        change_bbox={"left": 0, "top": 0, "right": 10, "bottom": 10},
        largest_blob_area_px=100, changed_pixels=100,
    )


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
