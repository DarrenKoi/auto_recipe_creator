"""그룹핑 규칙 R1~R5 + 불변식 테스트 (순수 함수, VLM 없음)."""

import pytest

import poc.workflow_3.workflow_extract.grouping as grouping
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


def test_rule_returning_non_positive_consumed_raises_instead_of_looping(monkeypatch):
    """소비 개수 0(또는 음수)을 돌려주는 규칙은 무한루프 대신 즉시 예외를 내야 한다.

    R5 는 항상 1 을 돌려주므로 오늘은 무해하지만, 이 while 루프는 R1-R4 가
    plug-in 될 제네릭 엔진이다 - 여러 개를 소비하는 규칙이 버그로 0/음수를
    돌려주면 i 가 전진하지 않아 크래시도 로그도 없이 그냥 멈춘 것처럼 걸린다.
    """

    def _broken_rule(events, i, ctx):
        return {}, 0

    monkeypatch.setattr(grouping, "_RULES", [_broken_rule])
    with pytest.raises(AssertionError, match="소비"):
        group_events([_event(0, 10.0)], _ctx())


def test_assert_invariant_reports_duplicated_seq_distinctly():
    """중복(같은 seq 가 두 step 에 겹쳐 들어감)은 누락/미상 seq 와 구분되어 보고돼야 한다.

    이것이 바로 원안의 `set(seen) - set(expected)` 버그가 놓쳤던 케이스다:
    중복된 seq 도 expected 의 원소라 차집합에서는 사라진다.
    """
    events = [{"seq": i} for i in range(5)]
    steps = [
        {"raw_events": [0]},
        {"raw_events": [0, 1]},   # seq 0 중복
        {"raw_events": [2]},
        {"raw_events": [3]},
        {"raw_events": [4]},
    ]
    with pytest.raises(AssertionError) as excinfo:
        grouping._assert_invariant(events, steps)
    message = str(excinfo.value)
    assert "중복=[0]" in message
    assert "누락=[]" in message
    assert "입력에 없는 seq=[]" in message


def test_assert_invariant_reports_missing_seq():
    events = [{"seq": i} for i in range(5)]
    steps = [
        {"raw_events": [0]},
        {"raw_events": [1]},
        {"raw_events": [2]},
        {"raw_events": [3]},
        # seq 4 를 담은 step 이 없음
    ]
    with pytest.raises(AssertionError) as excinfo:
        grouping._assert_invariant(events, steps)
    message = str(excinfo.value)
    assert "누락=[4]" in message
    assert "중복=[]" in message
    assert "입력에 없는 seq=[]" in message


def test_assert_invariant_reports_seq_not_in_input():
    events = [{"seq": i} for i in range(3)]
    steps = [
        {"raw_events": [0]},
        {"raw_events": [1]},
        {"raw_events": [2, 99]},   # 99 는 입력에 없던 seq
    ]
    with pytest.raises(AssertionError) as excinfo:
        grouping._assert_invariant(events, steps)
    message = str(excinfo.value)
    assert "입력에 없는 seq=[99]" in message
    assert "중복=[]" in message
    assert "누락=[]" in message


def test_default_rule_is_last_in_rule_list():
    """R5(fallback) 가 마지막이어야 종료 보장이 성립한다 - 나중에 R1-R4 를
    추가할 때 실수로 R5 뒤에 붙이면 이 불변식이 조용히 깨진다."""
    assert grouping._RULES[-1] is grouping._rule_default
