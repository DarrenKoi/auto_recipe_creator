"""workflow_extract step 스키마 테스트."""

from poc.workflow_3.workflow_extract.settings import (
    WorkflowExtractSettings,
    load_workflow_extract_settings,
)
from poc.workflow_3.workflow_extract.steps import make_step


def _event(seq, action="click", **kw):
    base = {
        "seq": seq, "t_sec": 10.0 + seq, "action": action, "coords": {"x": 100, "y": 200},
        "element": "PM", "element_source": "ocr", "target_kind": "ui_control",
        "region": "ui", "generation": 0, "occlusion": "none", "text": None,
        "confidence": 1.0, "frame": f"f_{seq}.jpg",
    }
    base.update(kw)
    return base


def test_make_step_carries_raw_events_and_rule():
    step = make_step([_event(3), _event(4)], action="select_from_dropdown", rule="R2")
    assert step["raw_events"] == [3, 4]
    assert step["grouping_rule"] == "R2"
    assert step["action"] == "select_from_dropdown"


def test_make_step_t_sec_is_start_end_pair():
    step = make_step([_event(3), _event(4)], action="click_repeat", rule="R4")
    assert step["t_sec"] == [13.0, 14.0]


def test_make_step_single_event_repeats_timestamp():
    step = make_step([_event(3)], action="click", rule="R5")
    assert step["t_sec"] == [13.0, 13.0]


def test_make_step_uses_t_sec_end_for_typing_burst():
    """타이핑 이벤트는 구간이므로 끝 시각을 잃으면 안 된다(Stage 2b 가 t_sec_end 를 싣는다)."""
    typing = _event(3, action="type_text", text="abc")
    typing["t_sec_end"] = 20.0
    step = make_step([typing], action="type_text", rule="R3")
    assert step["t_sec"] == [13.0, 20.0]


def test_make_step_t_sec_end_zero_is_not_discarded():
    """t_sec_end == 0.0 은 falsy 지만 유효한 값이다 - `or` 폴백이면 t_sec 로 잘못 덮인다.

    start(t_sec)=13.0, t_sec_end=0.0 로 두 결과가 확실히 갈리게 만든다: `or` 폴백이면
    [13.0, 13.0] 이 나오고(버그), `is not None` 판정이면 [13.0, 0.0] 이 나온다(정답).
    """
    typing = _event(3, action="type_text", text="abc")
    typing["t_sec_end"] = 0.0
    step = make_step([typing], action="type_text", rule="R3")
    assert step["t_sec"] == [13.0, 0.0]


def test_make_step_defaults_are_explicit_nulls():
    """스키마 필드는 항상 존재해야 한다 - 소비자가 키 유무를 분기하면 안 된다."""
    step = make_step([_event(3)], action="click", rule="R5")
    for key in ("target", "target_kind", "value", "value_source",
                "coords_in_live_box", "intent", "count", "inferred"):
        assert key in step


def test_settings_defaults_match_spec():
    s = WorkflowExtractSettings()
    assert s.recenter_window_sec == 1.5
    assert s.recenter_min_ratio == 0.40
    assert s.dropdown_max_sec == 5.0
    assert s.focus_max_sec == 2.0
    assert s.repeat_window_sec == 6.0
    assert s.repeat_min_count == 3
    assert s.same_target_px == 24
    assert s.thumbnails_enabled is True


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("WORKFLOW_EXTRACT_RECENTER_MIN_RATIO", "0.6")
    assert load_workflow_extract_settings().recenter_min_ratio == 0.6


def test_settings_env_override_thumbnails_enabled(monkeypatch):
    monkeypatch.setenv("WORKFLOW_EXTRACT_THUMBNAILS", "0")
    assert load_workflow_extract_settings().thumbnails_enabled is False
