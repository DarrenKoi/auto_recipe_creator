"""click_detect 테스트 — 가짜 VLM client + 합성 변화로 클릭 판정."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.click_detect import (
    ClickEvent,
    detect_clicks,
    flag_static_cursor_detections,
)
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

    def _fake_locate(client, frame_path, mask_boxes=None):
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


def _offscreen_meta(t_sec, cursor_xy):
    """포인터가 창 밖인 사이드카 레코드(창 rect 자체는 정상)."""
    return FrameMeta(
        t_sec=t_sec, rect={"left": 0, "top": 0, "right": 1600, "bottom": 1000},
        occlusion="none", cursor_xy=cursor_xy, cursor_in_window=False,
    )


def test_resolve_sidecar_cursor_none_when_pointer_maps_outside_frame():
    """프레임 밖으로 매핑되는 포인터는 커서 관측이 아니다(2026-08-12).

    RCS Remote Monitoring 창은 장비 화면을 비추는 뷰라, 프레임에 그려진 커서와
    로컬 포인터(GetCursorPos)는 별개다. 실제 세션에서 포인터가 내내 창 밖이었는데
    이 함수가 화면 밖 좌표를 돌려주는 바람에 호출부가 VLM 을 건너뛰고 빈 ROI 만
    세어 클릭이 전멸했다.
    """
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_offscreen_meta(10.0, [4000, 3000])]     # rect(1600x1000) 밖.
    assert resolve_sidecar_cursor(change, metas, (800, 500)) is None


def test_resolve_sidecar_cursor_none_when_pointer_is_negative():
    """창 왼쪽/위로 벗어난 포인터도 마찬가지로 관측 실패다."""
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_offscreen_meta(10.0, [-500, -400])]
    assert resolve_sidecar_cursor(change, metas, (800, 500)) is None


def test_detect_clicks_falls_back_to_vlm_when_sidecar_cursor_is_offscreen(monkeypatch):
    """사이드카가 있어도 커서가 프레임 밖이면 VLM 경로로 되돌아간다.

    이것이 50% -> 0% 회귀의 수정 지점이다. 사이드카 도입 전에는 VLM 이 프레임에
    그려진 커서를 봤고(약 50% 성공), 도입 후에는 아예 호출되지 않았다.
    """
    change = _change_event(rank=0, t_sec=10.0)
    metas = [_offscreen_meta(10.0, [4000, 3000])]
    calls = []

    def _fake_locate(client, frame_path, mask_boxes=None):
        calls.append(frame_path)
        return {"cursor_visible": False}, None, 800, 500

    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect.read_frame_size",
        lambda path: (800, 500),
    )
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._locate_cursor", _fake_locate
    )
    events = detect_clicks(
        [change], RecordingFilterSettings(vlm_request_delay_sec=0.0),
        client=object(), metas=metas,
    )
    assert len(calls) == 1, "사이드카가 쓸모없으면 VLM 을 불러야 한다"
    assert events[0].cursor_source == "vlm"


def test_detect_clicks_still_prefers_usable_sidecar(monkeypatch):
    """포인터가 프레임 안이면 종전대로 사이드카를 쓰고 VLM 을 부르지 않는다(음성 대조군)."""
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


def _vlm_click_event(rank, xy, *, is_click=True):
    """VLM 경로로 커서를 잡은 ClickEvent 를 만든다(정적 오탐 필터 테스트용)."""
    return ClickEvent(
        change=_change_event(rank, t_sec=float(rank)),
        is_click=is_click, status="click" if is_click else "no_click",
        cursor_visible=True, cursor_kind="hand",
        cursor_bbox={"left": xy[0] - 8, "top": xy[1] - 8, "right": xy[0] + 8, "bottom": xy[1] + 8},
        cursor_xy=list(xy), click_window={"left": 0, "top": 0, "right": 10, "bottom": 10},
        changed_in_window_px=9999, confidence=0.9, evidence="", cursor_source="vlm",
    )


def test_static_cursor_cluster_is_rejected():
    """전 프레임 같은 자리에 잡힌 '커서'는 고정 UI 아이콘으로 보고 무효화한다.

    이 툴은 'Full Size' 버튼과 라이브 SEM 영상 사이에 손바닥 아이콘을 늘 그려 둔다.
    그 자리는 라이브 박스 테두리 변화로 ROI 임계를 넘기 쉬워, 그대로 두면 없던
    클릭이 대량으로 만들어진다.
    """
    events = [_vlm_click_event(i, (500, 400)) for i in range(12)]
    decoys = flag_static_cursor_detections(events, RecordingFilterSettings())
    assert len(decoys) == 1
    assert decoys[0]["count"] == 12
    assert all(not e.is_click for e in events)
    assert all(e.status == "cursor_static_decoy" for e in events)


def test_static_cursor_tolerates_small_jitter():
    """VLM bbox 는 프레임마다 몇 px 흔들린다 - 허용 오차 안이면 같은 자리로 본다."""
    events = [_vlm_click_event(i, (500 + (i % 3), 400 - (i % 2))) for i in range(12)]
    assert len(flag_static_cursor_detections(events, RecordingFilterSettings())) == 1


def test_moving_cursor_is_not_rejected():
    """실제로 움직인 커서는 건드리지 않는다(음성 대조군)."""
    events = [_vlm_click_event(i, (100 + i * 60, 200 + i * 40)) for i in range(12)]
    assert flag_static_cursor_detections(events, RecordingFilterSettings()) == []
    assert all(e.is_click for e in events)


def test_static_cursor_needs_majority_not_just_repeats():
    """한 자리에 몇 번 머무는 정상 조작(반복 클릭)은 무효화하지 않는다.

    10회 반복이라도 전체 탐지의 절반에 못 미치면 정적 아이콘으로 단정하지 않는다.
    """
    events = [_vlm_click_event(i, (500, 400)) for i in range(10)]
    events += [_vlm_click_event(100 + i, (100 + i * 50, 700)) for i in range(15)]
    assert flag_static_cursor_detections(events, RecordingFilterSettings()) == []


def test_static_cursor_reject_ignores_sidecar_events():
    """사이드카 좌표는 VLM 오탐이 아니다 - 정적이어도 이 필터의 대상이 아니다."""
    events = [_vlm_click_event(i, (500, 400)) for i in range(12)]
    for event in events:
        event.cursor_source = "sidecar"
    assert flag_static_cursor_detections(events, RecordingFilterSettings()) == []


def test_static_cursor_reject_can_be_disabled():
    """킬 스위치 - 정적 판정이 정상 조작을 먹는 세션에서 즉시 끌 수 있어야 한다."""
    events = [_vlm_click_event(i, (500, 400)) for i in range(12)]
    settings = RecordingFilterSettings(static_cursor_reject=False, vlm_request_delay_sec=0.0)
    detect_clicks([], settings, client=object(), metas=None)   # 배선 확인용(빈 입력).
    assert all(e.is_click for e in events)


def _vlm_event_at(rank, xy, t_sec):
    """지정 시각의 VLM 탐지 이벤트(정적 판정의 시간 폭 테스트용)."""
    event = _vlm_click_event(rank, xy)
    event.change.timestamp_sec = t_sec
    return event


def test_three_decoys_are_all_flagged_by_time_span():
    """오탐원이 셋으로 갈려 어느 것도 과반이 아니어도 전부 잡는다.

    (2026-08-12) 이 창의 고정 그래픽은 셋이다(손바닥 / 우상단 닫기 X / 라이브 박스
    좌상단 '>'). 폴백이 셋으로 나뉘면 각 무리는 33% 라, "과반" 기준만으로는 하나도
    못 잡는다. 정적 아이콘은 세션 내내 같은 자리에 나타난다는 점(시간 폭)으로 잡는다.
    """
    events = []
    for i in range(12):
        events.append(_vlm_event_at(i, (500, 400), t_sec=i * 30.0))          # 손바닥
        events.append(_vlm_event_at(100 + i, (1900, 20), t_sec=i * 30.0))    # 닫기 X
        events.append(_vlm_event_at(200 + i, (960, 300), t_sec=i * 30.0))    # '>' 마크
    decoys = flag_static_cursor_detections(events, RecordingFilterSettings())
    assert len(decoys) == 3, decoys
    assert all(e.status == "cursor_static_decoy" for e in events)


def test_short_burst_at_one_spot_is_not_a_decoy():
    """같은 버튼을 짧은 구간에 반복 클릭하는 정상 조작은 살린다(음성 대조군).

    시간 폭 기준을 넣으면서 이 케이스가 무너지지 않는지가 핵심이다 - 12번 눌러도
    몇 초 안에 몰려 있으면 정적 아이콘이 아니다.
    """
    events = [_vlm_event_at(i, (500, 400), t_sec=i * 0.4) for i in range(12)]
    events += [_vlm_event_at(100 + i, (100 + i * 50, 700), t_sec=100.0 + i) for i in range(15)]
    assert flag_static_cursor_detections(events, RecordingFilterSettings()) == []


class _MaskAwareClient:
    """가리기 전에는 오탐 자리를, 가린 뒤에는 진짜 커서를 돌려주는 대역."""

    def __init__(self):
        self.masked_calls = 0

    def locate(self, client, frame_path, mask_boxes=None):
        if mask_boxes:
            self.masked_calls += 1
            # 가려졌으니 창 우상단(닫기 버튼 옆)의 진짜 화살표를 찾아낸다.
            return ({"cursor_visible": True, "cursor_kind": "rcs_black_arrow",
                     "confidence": 0.8, "evidence": "arrow by the close button"},
                    {"left": 1880, "top": 8, "right": 1904, "bottom": 32}, 1920, 1080)
        return ({"cursor_visible": True, "cursor_kind": "hand", "confidence": 0.9,
                 "evidence": "palm"},
                {"left": 492, "top": 392, "right": 508, "bottom": 408}, 1920, 1080)


def test_masked_retry_recovers_the_real_cursor(monkeypatch):
    """오탐 자리를 가리고 다시 물으면 진짜 커서를 회수한다.

    (2026-08-12) 커서가 우상단 닫기 버튼 근처에 있을 때 모델이 손바닥/'>' 로
    되돌아갔다. 그 프레임을 그냥 버리면 **창 가장자리 조작만 골라서** 사라져
    무작위 손실보다 나쁘다(계통적 편향).
    """
    fake = _MaskAwareClient()
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._locate_cursor", fake.locate
    )
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._diff_mask",
        lambda prev, curr, thr: np.zeros((1080, 1920), dtype=np.uint8),
    )
    changes = [_change_event(i, t_sec=i * 30.0) for i in range(12)]
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0)
    events = detect_clicks(changes, settings, client=object(), metas=None)

    assert fake.masked_calls == 12, "무효화된 이벤트마다 가린 재질의가 있어야 한다"
    # 회수된 이벤트는 오탐 자리가 아니라 진짜 커서 좌표를 들고 있어야 한다.
    assert all(e.cursor_source == "vlm_masked" for e in events), [e.cursor_source for e in events]
    assert all(e.cursor_xy[0] > 1800 for e in events), [e.cursor_xy for e in events]


def test_masked_retry_respects_call_budget(monkeypatch):
    """재질의도 max_vlm_calls 예산 안에서만 한다(상한이 뚫리면 안 된다)."""
    fake = _MaskAwareClient()
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._locate_cursor", fake.locate
    )
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._diff_mask",
        lambda prev, curr, thr: None,
    )
    changes = [_change_event(i, t_sec=i * 30.0) for i in range(12)]
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, max_vlm_calls=12)
    detect_clicks(changes, settings, client=object(), metas=None)
    assert fake.masked_calls == 0, "1차 탐지로 예산을 다 썼으면 재질의는 없다"


def test_masked_retry_can_be_disabled(monkeypatch):
    """킬 스위치 - 재질의 콜을 원치 않으면 끌 수 있다."""
    fake = _MaskAwareClient()
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._locate_cursor", fake.locate
    )
    monkeypatch.setattr(
        "poc.workflow_3.recording_filter.click_detect._diff_mask",
        lambda prev, curr, thr: None,
    )
    changes = [_change_event(i, t_sec=i * 30.0) for i in range(12)]
    settings = RecordingFilterSettings(
        vlm_request_delay_sec=0.0, static_cursor_retry_masked=False
    )
    detect_clicks(changes, settings, client=object(), metas=None)
    assert fake.masked_calls == 0
