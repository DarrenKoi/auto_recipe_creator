"""조작 미확인 화면 변화 - 클릭/타이핑으로 설명되지 않은 변화를 타임라인에 남긴다."""

from poc.workflow_3.recording_filter.click_detect import ClickEvent
from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings
from poc.workflow_3.recording_filter.unattributed_change import find_unattributed_changes

_BOX = {"left": 100, "top": 100, "right": 300, "bottom": 200}


def _change(rank, t_sec, bbox=None):
    return ChangeEvent(
        rank=rank, frame_path=f"/tmp/f_{rank}.jpg", prev_frame_path=f"/tmp/f_{rank}_prev.jpg",
        timestamp_sec=t_sec, frame_index=rank, change_bbox=bbox or dict(_BOX),
        largest_blob_area_px=9000, changed_pixels=9000,
    )


def _judged(change, *, status="no_click", cursor_xy=None, is_click=False):
    """Stage 2a 가 이 변화에 내린 판정 1건."""
    return ClickEvent(
        change=change, is_click=is_click, status=status,
        cursor_visible=cursor_xy is not None, cursor_kind=None, cursor_bbox=None,
        cursor_xy=cursor_xy, click_window=None, changed_in_window_px=0,
        confidence=0.0, evidence="", cursor_source="vlm",
    )


def _find(changes, judged, *, typing_ranks=(), gate_info=None, settings=None):
    return find_unattributed_changes(
        changes, judged, set(typing_ranks), gate_info or {},
        settings or RecordingFilterSettings(),
    )


def test_change_with_no_cursor_found_becomes_a_screen_change_event():
    change = _change(0, 12.3)
    gate_info = {0: {"generation": 2, "region": "ui", "occlusion": "none"}}

    events = _find([change], [_judged(change)], gate_info=gate_info)

    assert len(events) == 1
    event = events[0]
    assert event["action"] == "screen_change"
    assert event["t_sec"] == 12.3
    assert event["t_sec_end"] == 12.3
    assert event["coords"] is None            # 어디를 눌렀는지는 모른다.
    assert event["element"] is None
    assert event["reasons"] == ["cursor_not_found"]
    assert event["change_bbox"] == _BOX
    assert event["change_ranks"] == [0]
    assert event["frame"] == "f_0.jpg"
    assert event["source_frames"] == {"prev": "f_0_prev.jpg", "curr": "f_0.jpg"}
    assert (event["region"], event["generation"], event["occlusion"]) == ("ui", 2, "none")


def test_changes_owned_by_a_click_or_typing_are_not_reported():
    clicked, typed, missed = _change(0, 1.0), _change(1, 5.0), _change(2, 9.0)
    judged = [
        _judged(clicked, status="click", cursor_xy=[150, 150], is_click=True),
        _judged(typed),                 # 커서는 못 찾았지만 Stage 2b 가 타이핑으로 가져갔다.
        _judged(missed),
    ]

    events = _find([clicked, typed, missed], judged, typing_ranks={1})

    assert [e["change_ranks"] for e in events] == [[2]]


def test_change_never_judged_because_of_the_call_cap_is_not_reported():
    """콜 상한에 잘린 변화는 '판정 못 함'이지 '조작 미확인'이 아니다(summary 가 따로 센다)."""
    judged_one, capped = _change(0, 1.0), _change(1, 5.0)

    events = _find([judged_one, capped], [_judged(judged_one)])

    assert [e["change_ranks"] for e in events] == [[0]]


def test_reason_tells_a_missing_cursor_from_a_change_away_from_the_cursor():
    """커서를 찾았는데 그 주변이 안 바뀐 것도 '조작 없음'의 증거는 아니다.

    버튼을 눌렀는데 대화상자는 화면 가운데 열리는 경우가 그렇다. 고정 그래픽을
    커서로 오인한 판정(cursor_static_decoy)은 좌표가 있어도 커서를 찾은 것이 아니다.
    """
    away, decoy, unavailable = _change(0, 1.0), _change(1, 10.0), _change(2, 20.0)
    judged = [
        _judged(away, cursor_xy=[700, 400]),
        _judged(decoy, status="cursor_static_decoy", cursor_xy=[40, 40]),
        _judged(unavailable, status="cursor_unavailable"),
    ]

    events = _find([away, decoy, unavailable], judged)

    assert [e["reasons"] for e in events] == [
        ["cursor_elsewhere"], ["cursor_not_found"], ["cursor_not_found"],
    ]


def test_changes_close_in_time_merge_into_one_event():
    """대화상자 하나가 열리며 여러 프레임에 걸쳐 그려지는 것은 관측 하나다."""
    first = _change(0, 10.0, {"left": 100, "top": 100, "right": 300, "bottom": 200})
    second = _change(1, 10.4, {"left": 250, "top": 150, "right": 500, "bottom": 320})
    later = _change(2, 30.0)
    judged = [_judged(first), _judged(second, cursor_xy=[700, 400]), _judged(later)]

    events = _find(
        [first, second, later], judged,
        settings=RecordingFilterSettings(unattributed_merge_gap_sec=1.5),
    )

    assert [e["change_ranks"] for e in events] == [[0, 1], [2]]
    merged = events[0]
    assert (merged["t_sec"], merged["t_sec_end"]) == (10.0, 10.4)
    assert merged["change_bbox"] == {"left": 100, "top": 100, "right": 500, "bottom": 320}
    assert merged["reasons"] == ["cursor_elsewhere", "cursor_not_found"]
    # 전/후 프레임은 묶음 전체를 감싼다 - 엔지니어가 '무엇이 바뀌었나'를 한 쌍으로 본다.
    assert merged["frame"] == "f_0.jpg"
    assert merged["source_frames"] == {"prev": "f_0_prev.jpg", "curr": "f_1.jpg"}


def test_merge_does_not_cross_a_click():
    """사이에 클릭이 끼면 앞뒤 변화는 서로 다른 관측이다 - 묶으면 타임라인 순서가 뒤집힌다."""
    before, click, after = _change(0, 10.0), _change(1, 10.3), _change(2, 10.6)
    judged = [
        _judged(before),
        _judged(click, status="click", cursor_xy=[150, 150], is_click=True),
        _judged(after),
    ]

    events = _find([before, click, after], judged)

    assert [e["change_ranks"] for e in events] == [[0], [2]]


def test_merge_does_not_cross_a_window_resize():
    """generation 이 바뀌면 좌표계가 달라져 영역 합집합이 의미를 잃는다."""
    small, resized = _change(0, 10.0), _change(1, 10.3)
    gate_info = {0: {"generation": 0}, 1: {"generation": 1}}

    events = _find([small, resized], [_judged(small), _judged(resized)], gate_info=gate_info)

    assert [(e["change_ranks"], e["generation"]) for e in events] == [([0], 0), ([1], 1)]


def test_changes_that_arrive_out_of_time_order_are_not_merged():
    """음수 간격은 '가깝다'가 아니다 - 묶으면 시작이 끝보다 늦은 관측이 나온다."""
    late, early = _change(0, 500.0), _change(1, 50.0)

    events = _find([late, early], [_judged(late), _judged(early)])

    assert [(e["t_sec"], e["t_sec_end"]) for e in events] == [(500.0, 500.0), (50.0, 50.0)]
