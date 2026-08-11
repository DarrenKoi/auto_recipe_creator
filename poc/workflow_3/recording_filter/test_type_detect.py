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
