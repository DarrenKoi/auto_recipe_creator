"""RecordingFilterSettings 기본값 + env override 테스트."""

from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)


def test_defaults_match_spec():
    s = RecordingFilterSettings()
    assert s.diff_threshold == 25
    assert s.resize_width == 1280
    assert s.min_change_area_px == 5000
    assert s.cursor_click_window_px == 200
    assert s.click_min_changed_px == 1500
    assert s.vlm_service == "mai-ui"
    assert s.vlm_request_delay_sec == 1.0
    assert s.max_vlm_calls == 0


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RECORDING_FILTER_MIN_CHANGE_AREA_PX", "9000")
    monkeypatch.setenv("RECORDING_FILTER_CLICK_WINDOW_PX", "120")
    monkeypatch.setenv("RECORDING_FILTER_VLM_REQUEST_DELAY_SEC", "0")
    monkeypatch.setenv("RECORDING_FILTER_MAX_VLM_CALLS", "5")
    s = load_recording_filter_settings()
    assert s.min_change_area_px == 9000
    assert s.cursor_click_window_px == 120
    assert s.vlm_request_delay_sec == 0.0
    assert s.max_vlm_calls == 5
