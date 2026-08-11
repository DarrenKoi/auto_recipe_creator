"""그룹핑 규칙 R1~R5 + 불변식 테스트 (순수 함수, VLM 없음)."""

from poc.workflow_3.workflow_extract.grouping import GroupingContext, group_events
from poc.workflow_3.workflow_extract.settings import WorkflowExtractSettings


def _event(seq, t_sec, action="click", region="ui", element="PM",
           target_kind="ui_control", coords=None, generation=0, text=None):
    return {
        "seq": seq, "t_sec": t_sec, "action": action,
        "coords": coords if coords is not None else {"x": 100, "y": 200},
        "element": element, "element_source": "ocr" if element else "none",
        "target_kind": target_kind, "region": region, "generation": generation,
        "occlusion": "none", "text": text, "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }


def _ctx(**kw):
    return GroupingContext(
        settings=kw.pop("settings", WorkflowExtractSettings()),
        live_boxes=kw.pop("live_boxes", {}),
        changes=kw.pop("changes", []),
        frame_wh=kw.pop("frame_wh", (1600, 1000)),
    )


def test_lone_clicks_become_r5_steps():
    events = [_event(0, 10.0), _event(1, 30.0, element="OK")]
    steps = group_events(events, _ctx())
    assert [s["grouping_rule"] for s in steps] == ["R5", "R5"]
    assert [s["action"] for s in steps] == ["click", "click"]


def test_steps_are_renumbered_sequentially():
    events = [_event(0, 10.0), _event(1, 30.0, element="OK")]
    steps = group_events(events, _ctx())
    assert [s["seq"] for s in steps] == [0, 1]


def test_invariant_every_event_used_exactly_once():
    """불변식: 모든 이벤트가 정확히 하나의 step raw_events 에 나타난다."""
    events = [_event(i, 10.0 + i * 20) for i in range(5)]
    steps = group_events(events, _ctx())
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == [0, 1, 2, 3, 4]


def test_empty_timeline_yields_no_steps():
    assert group_events([], _ctx()) == []


def test_events_sorted_by_time_before_grouping():
    """입력이 시간순이 아니어도 결과는 시간순이어야 한다."""
    events = [_event(0, 50.0), _event(1, 10.0, element="OK")]
    steps = group_events(events, _ctx())
    assert steps[0]["raw_events"] == [1]
    assert steps[1]["raw_events"] == [0]
