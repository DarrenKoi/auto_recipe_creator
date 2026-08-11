"""Stage 2b 타이핑 구간 탐지 테스트 - 커서 정지 + 국소 반복 변화."""

from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.region_gate import FrameMeta
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.recording_filter.type_detect import (
    TypingBurst,
    find_typing_bursts,
    resolve_typing_events,
)

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
