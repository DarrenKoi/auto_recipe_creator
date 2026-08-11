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


_LIVE_BOX = {"left": 200, "top": 100, "right": 1000, "bottom": 700}


def _change(t_sec, bbox):
    return {"timestamp_sec": t_sec, "change_bbox": bbox}


def test_r1_fires_on_live_image_click_with_recenter_change():
    """라이브 박스 클릭 직후 박스 대부분이 다시 그려지면 FOV 이동 더블클릭."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    changes = [_change(10.4, dict(_LIVE_BOX))]
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=changes))
    assert steps[0]["action"] == "double_click"
    assert steps[0]["grouping_rule"] == "R1"
    assert steps[0]["intent"] == "fov_move"
    assert steps[0]["inferred"] is True


def test_r1_does_not_fire_on_small_local_change():
    """라이브 박스 안이라도 국소 변화면 단발 클릭(마커/선택)이다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    small = {"left": 300, "top": 200, "right": 340, "bottom": 240}
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, small)]))
    assert steps[0]["action"] == "click"
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_does_not_fire_outside_live_region():
    """UI 컨트롤 클릭은 뒤에 큰 변화가 와도 더블클릭이 아니다."""
    events = [_event(0, 10.0, region="ui")]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, dict(_LIVE_BOX))])
    )
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_ignores_change_outside_time_window():
    """1.5초 창 밖의 변화는 이 클릭의 결과로 보지 않는다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(20.0, dict(_LIVE_BOX))])
    )
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_degrades_without_live_box():
    """region_map.json 이 없으면 비율을 못 재므로 평범한 click 으로 degrade."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image", element=None)]
    steps = group_events(events, _ctx(live_boxes={}, changes=[_change(10.4, dict(_LIVE_BOX))]))
    assert steps[0]["grouping_rule"] == "R5"


def test_r1_sets_normalized_coords_in_live_box():
    """live_image step 은 창 픽셀이 아니라 라이브 박스 내부 정규화 좌표를 든다."""
    events = [_event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 600, "y": 400})]
    steps = group_events(
        events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[_change(10.4, dict(_LIVE_BOX))])
    )
    assert steps[0]["coords_in_live_box"] == [0.5, 0.5]


def test_r2_groups_open_and_select():
    """PM 클릭 -> 바로 아래 영역 클릭 = 드롭다운 선택 1 step."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    steps = group_events([open_ev, pick_ev], _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "select_from_dropdown"
    assert steps[0]["target"] == "PM"
    assert steps[0]["value"] == "210"
    assert steps[0]["value_source"] == "ocr"
    assert steps[0]["raw_events"] == [0, 1]


def test_r2_marks_output_as_inferred():
    """R2 는 기하 추론일 뿐 드롭다운이 실제로 열렸다는 증거가 없다 - R1 과 같은 이유로
    inferred=True 를 남겨야 절차서에서 관측과 구분된다(2026-08-12 리뷰)."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    steps = group_events([open_ev, pick_ev], _ctx())
    assert steps[0]["action"] == "select_from_dropdown"
    assert steps[0]["inferred"] is True


def test_r2_does_not_group_when_too_slow():
    """5초를 넘으면 별개 조작이다."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 30.0, element="210", coords={"x": 810, "y": 420})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2


def test_r2_does_not_group_click_outside_dropdown_region():
    """아래가 아니라 옆을 눌렀으면 드롭다운이 아니다.

    dy=100 은 수직 가드(_DROPDOWN_MIN_ROW_GAP_PX=12)와 행 높이(~24px) 를 한참
    넘어서므로 수직 가드는 이 케이스의 판정에 관여하지 않는다 - 실제로 막는 건
    x=200 이 드롭다운 폭 밖이라는 `_point_in_region` 의 가로 범위 검사다.
    """
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="OK", coords={"x": 200, "y": 400})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2


def test_r2_degrades_without_frame_size():
    """frame_wh 를 모르면 드롭다운 기하를 계산할 수 없어 degrade."""
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    steps = group_events([open_ev, pick_ev], _ctx(frame_wh=None))
    assert [s["grouping_rule"] for s in steps] == ["R5", "R5"]


def test_r2_requires_ui_control_opener():
    """라이브 영상 위 클릭은 드롭다운 여는 동작이 아니다."""
    open_ev = _event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 800, "y": 300})
    pick_ev = _event(1, 12.0, element="210", coords={"x": 810, "y": 420})
    assert len(group_events([open_ev, pick_ev], _ctx())) == 2


def test_r5_live_image_click_also_gets_normalized_coords():
    """더블클릭이 아닌 라이브 박스 단발 클릭도 정규화 좌표를 들어야 한다.

    스펙 §6 은 'live_image step' 전체에 coords_in_live_box 를 요구한다. R1 만
    채우면 마커/선택 클릭이 창 픽셀만 든 채 남아, 소비자가 두 종류의 live_image
    step 을 서로 다르게 다뤄야 한다.
    """
    events = [_event(0, 10.0, region="live_image", target_kind="live_image",
                     element=None, coords={"x": 600, "y": 400})]
    steps = group_events(events, _ctx(live_boxes={0: _LIVE_BOX}, changes=[]))
    assert steps[0]["grouping_rule"] == "R5"
    assert steps[0]["coords_in_live_box"] == [0.5, 0.5]


def test_r3_absorbs_focus_click():
    """필드 클릭 직후 타이핑이면 클릭은 포커스로 흡수돼 1 step 이 된다."""
    click = _event(0, 10.0, element="Recipe Name")
    typing = _event(1, 11.0, action="type_text", element="Recipe Name", text="MCD916")
    steps = group_events([click, typing], _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "type_text"
    assert steps[0]["value"] == "MCD916"
    assert steps[0]["raw_events"] == [0, 1]


def test_r3_standalone_typing_without_focus_click():
    """Tab 포커스면 직전 클릭이 없어도 type_text step 이 나온다."""
    typing = _event(0, 11.0, action="type_text", element=None, text="MCD916")
    steps = group_events([typing], _ctx())
    assert steps[0]["action"] == "type_text"
    assert steps[0]["target"] is None
    assert steps[0]["raw_events"] == [0]


def test_r3_does_not_absorb_distant_click():
    """포커스 창(2초)을 넘긴 클릭은 별개 조작이다."""
    click = _event(0, 10.0, element="Recipe Name")
    typing = _event(1, 20.0, action="type_text", element="Recipe Name", text="MCD916")
    assert len(group_events([click, typing], _ctx())) == 2


def test_r4_groups_repeated_clicks_on_same_label():
    """같은 라벨을 3회 이상 누르면 반복 1 step."""
    events = [_event(i, 10.0 + i, element="Zoom In") for i in range(3)]
    steps = group_events(events, _ctx())
    assert len(steps) == 1
    assert steps[0]["action"] == "click_repeat"
    assert steps[0]["count"] == 3
    assert steps[0]["raw_events"] == [0, 1, 2]


def test_r4_needs_min_count():
    """2회는 반복으로 묶지 않는다."""
    events = [_event(i, 10.0 + i, element="Zoom In") for i in range(2)]
    assert len(group_events(events, _ctx())) == 2


def test_r4_matches_by_coords_when_label_missing():
    """라벨이 없으면 좌표 근접(24px)으로 동일 대상을 판정한다."""
    events = [
        _event(i, 10.0 + i, element=None, coords={"x": 100 + i * 5, "y": 200})
        for i in range(3)
    ]
    steps = group_events(events, _ctx())
    assert steps[0]["action"] == "click_repeat"


def test_r4_does_not_mix_label_and_coords():
    """한쪽만 라벨이 있으면 묶지 않는다 - 묶임이 OCR 운에 좌우되면 재현되지 않는다."""
    events = [
        _event(0, 10.0, element="Zoom In"),
        _event(1, 11.0, element=None),
        _event(2, 12.0, element="Zoom In"),
    ]
    assert len(group_events(events, _ctx())) == 3


def test_same_target_treats_empty_string_element_as_no_label():
    """빈 문자열 라벨(element="")도 None 과 동일하게 '라벨 없음'으로 취급돼야 한다.

    OCR 이 라벨을 못 읽으면 파이프라인 어딘가에서 None 대신 "" 을 돌려줄 수 있다 -
    두 표현이 같은 판정 경로를 타지 않으면 어느 쪽이 오느냐에 따라 그룹핑 결과가
    갈리는데, 그건 same_target 이 정확히 막으려던 "OCR 운" 문제의 재발이다.
    """
    a = _event(0, 10.0, element="", coords={"x": 100, "y": 200})
    b = _event(1, 11.0, element=None, coords={"x": 105, "y": 200})
    assert grouping.same_target(a, b, WorkflowExtractSettings()) is True


def test_r3_preserves_empty_string_typed_value():
    """타이핑 후 지운 값(text="")은 실제 이벤트이지 '값 없음'이 아니다.

    `event.get("text") or None` 같은 폴백으로 "단순화"하면 지운 값과 애초에 값이
    없던 경우가 똑같이 None 이 되어 조용히 구분이 사라진다.
    """
    click = _event(0, 10.0, element="Recipe Name")
    typing = _event(1, 11.0, action="type_text", element="Recipe Name", text="")
    steps = group_events([click, typing], _ctx())
    assert steps[0]["value"] == ""


def test_r2_groups_first_row_pick_just_below_opener():
    """드롭다운 첫 행처럼 오프너 바로 아래(15~20px)인 피커도 여전히 R2 로 묶여야 한다.

    가드가 반경(radial) 검사였다면 이 구간(첫 행 높이 근처)이 오검(false reject)돼
    진짜 드롭다운 선택이 disconnected 한 R5 클릭 2개로 깨졌을 것이다 - 방향성
    (수직 간격) 검사여야 이 경계를 살릴 수 있다.
    """
    open_ev = _event(0, 10.0, element="PM", coords={"x": 800, "y": 300})
    pick_ev = _event(1, 11.0, element="210", coords={"x": 805, "y": 317})
    steps = group_events([open_ev, pick_ev], _ctx())
    assert steps[0]["action"] == "select_from_dropdown"


def test_invariant_holds_across_all_rules():
    """R1~R5 가 섞여도 불변식이 유지된다."""
    events = [
        _event(0, 10.0, element="PM", coords={"x": 800, "y": 300}),
        _event(1, 11.0, element="210", coords={"x": 810, "y": 420}),
        _event(2, 20.0, action="type_text", element=None, text="abc"),
        _event(3, 40.0, element="Zoom In"),
        _event(4, 41.0, element="Zoom In"),
        _event(5, 42.0, element="Zoom In"),
    ]
    steps = group_events(events, _ctx())
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 2026-08-11 최종 리뷰 C3/I2 - R2 오발화, R3 target_kind 승계, 규칙 상호작용.
# ---------------------------------------------------------------------------

def test_r2_does_not_steal_repeat_clicks_with_human_jitter():
    """같은 버튼을 3회 누를 때의 지터(18px)는 드롭다운이 아니라 R4 반복 클릭이다.

    (리뷰 C3 Case A) dy>=12 가드는 dy 가 거의 0 인 경우만 막았다. 사람이 같은
    버튼을 다시 누르면 보통 10~20px 어긋나므로 두 번째 클릭이 곧바로
    "드롭다운에서 자기 오프너와 같은 라벨을 선택" 이 되고, 세 번째 클릭만 R5 로
    남았다. 드롭다운이 자기 오프너의 라벨을 고르는 일은 없다.
    """
    events = [
        _event(0, 10.0, element="Zoom In", coords={"x": 800, "y": 300}),
        _event(1, 11.0, element="Zoom In", coords={"x": 803, "y": 318}),
        _event(2, 12.0, element="Zoom In", coords={"x": 800, "y": 301}),
    ]
    steps = group_events(events, _ctx())
    assert [s["grouping_rule"] for s in steps] == ["R4"], steps
    assert steps[0]["count"] == 3


def test_r2_does_not_steal_repeat_clicks_without_labels():
    """라벨이 없어도(OCR 실패) 좌표 근접 판정으로 같은 대상이면 R2 가 아니다.

    dy=12 는 수직 가드(_DROPDOWN_MIN_ROW_GAP_PX)를 정확히 통과하는 값이라, 좌표
    기반 same_target 가드가 없으면 첫 두 클릭이 R2 에 선점된다.
    """
    events = [
        _event(i, 10.0 + i, element=None, coords={"x": 800, "y": 300 + i * 12})
        for i in range(3)
    ]
    steps = group_events(events, _ctx())
    assert [s["grouping_rule"] for s in steps] == ["R4"], steps


def test_r2_does_not_group_click_on_lower_control():
    """세로로 쌓인 폼에서 아래 컨트롤을 누른 것은 드롭다운 선택이 아니다.

    (리뷰 C3 Case B) crop 기하의 세로 띠는 프레임 높이의 0.45 배(1080 에서 486px)라
    폼 절반이 후보였다. 두 진짜 클릭이 한 step 으로 뭉치면 값이 조작되고 두 번째
    클릭은 문서에서 사라진다 - 되돌릴 수는 있어도(raw_events) 읽는 사람은 속는다.
    트리거 상한(_DROPDOWN_MAX_DROP_PX)을 넘으면 각자 R5 로 남아야 한다.
    """
    open_ev = _event(0, 10.0, element="Recipe", coords={"x": 600, "y": 300})
    pick_ev = _event(1, 12.0, element="Load", coords={"x": 650, "y": 440})
    steps = group_events([open_ev, pick_ev], _ctx())
    assert [s["grouping_rule"] for s in steps] == ["R5", "R5"], steps


def test_no_dropdown_step_ever_has_target_equal_to_value():
    """어떤 조합에서도 target == value 인 select_from_dropdown 은 나오면 안 된다.

    이것은 R2 의 의미 자체에서 나오는 불변식이다 - 오프너 라벨과 선택값이 같으면
    그건 드롭다운 선택이 아니라 같은 것을 두 번 누른 것이다.
    """
    events = [
        _event(0, 10.0, element="Zoom In", coords={"x": 800, "y": 300}),
        _event(1, 10.6, element="Zoom In", coords={"x": 802, "y": 316}),
        _event(2, 11.2, element="Zoom In", coords={"x": 799, "y": 302}),
        _event(3, 20.0, element="PM", coords={"x": 800, "y": 300}),
        _event(4, 21.0, element="210", coords={"x": 810, "y": 360}),
        _event(5, 40.0, element="Recipe", coords={"x": 600, "y": 300}),
        _event(6, 41.0, element="Load", coords={"x": 650, "y": 460}),
    ]
    steps = group_events(events, _ctx())
    dropdowns = [s for s in steps if s["action"] == "select_from_dropdown"]
    assert dropdowns, steps          # 진짜 드롭다운 1건은 여전히 잡혀야 한다
    for step in steps:
        if step["action"] == "select_from_dropdown":
            assert step["target"] != step["value"], step
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == list(range(7))


def test_r3_inherits_unknown_target_kind_from_typing_event():
    """OCR 이 실패한 타이핑은 이식 가능성을 알 수 없으므로 unknown 을 유지해야 한다.

    (리뷰 I2) 하드코딩된 ui_control 은 Task 4 가 Stage 2b 에서 파생하도록 고친 값을
    다시 덮어써, 라벨을 읽은 적 없는 step 이 "다른 장비에서 라벨로 다시 찾을 수
    있다"고 주장하게 만들었다.
    """
    typing = _event(0, 11.0, action="type_text", element=None, text=None,
                    target_kind="unknown")
    typing["element_source"] = "none"
    steps = group_events([typing], _ctx())
    assert steps[0]["target_kind"] == "unknown", steps[0]
    assert steps[0]["value_source"] == "none"


def test_r3_absorb_branch_inherits_target_kind_from_typing_not_click():
    """흡수 갈래에서도 클릭(events[0])이 아니라 타이핑 이벤트의 값을 승계한다."""
    click = _event(0, 10.0, element="Recipe Name")          # target_kind=ui_control
    typing = _event(1, 11.0, action="type_text", element=None, text=None,
                    target_kind="unknown")
    typing["element_source"] = "none"
    steps = group_events([click, typing], _ctx())
    assert len(steps) == 1
    assert steps[0]["target_kind"] == "unknown", steps[0]


def test_group_events_consumes_a_real_build_timeline_payload():
    """실제 build_timeline 산출물을 그대로 그룹핑할 수 있어야 한다(모듈 간 계약).

    R1/R2 테스트가 손으로 쓴 dict 만 먹였기 때문에, 생산자와 소비자의 스키마가
    어긋났는지 아무 테스트도 몰랐다(같은 종류의 사고가 C1 이었다).
    """
    from poc.workflow_3.recording_filter.click_detect import ClickEvent
    from poc.workflow_3.recording_filter.frame_reduce import ChangeEvent
    from poc.workflow_3.recording_filter.timeline import build_timeline

    def _click(rank, t_sec, xy):
        change = ChangeEvent(
            rank=rank, frame_path=f"/tmp/f_{rank}.jpg", prev_frame_path=f"/tmp/p_{rank}.jpg",
            timestamp_sec=t_sec, frame_index=rank,
            change_bbox={"left": 0, "top": 0, "right": 10, "bottom": 10},
            largest_blob_area_px=9000, changed_pixels=9000,
        )
        return ClickEvent(
            change=change, is_click=True, status="click", cursor_visible=True,
            cursor_kind="sidecar", cursor_bbox=None, cursor_xy=list(xy),
            click_window=None, changed_in_window_px=9000, confidence=1.0,
            evidence="", cursor_source="sidecar",
        )

    gate_info = {
        0: {"generation": 0, "region": "ui", "occlusion": "none", "verdict": "candidate"},
        1: {"generation": 0, "region": "ui", "occlusion": "none", "verdict": "candidate"},
    }
    timeline = build_timeline(
        [_click(0, 10.0, (800, 300)), _click(1, 11.0, (810, 360))], gate_info=gate_info
    )
    steps = group_events(timeline, _ctx())
    seen = [r for s in steps for r in s["raw_events"]]
    assert sorted(seen) == [0, 1], steps
