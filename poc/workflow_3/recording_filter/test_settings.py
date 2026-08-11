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


def test_typing_settings_defaults():
    """Stage 2b 기본값 - 스펙 §5 와 일치해야 한다."""
    s = RecordingFilterSettings()
    assert s.typing_detect_enabled is True
    assert s.typing_cursor_still_px == 8
    assert s.typing_min_burst_events == 3
    assert s.typing_burst_idle_sec == 1.5
    assert s.typing_focus_max_sec == 2.0
    assert s.typing_ocr_service == "paddleocr-vl-1.5"


def test_typing_settings_env_override(monkeypatch):
    """env 로 임계값을 바꿀 수 있어야 한다(CLI 인자 없음 규칙)."""
    monkeypatch.setenv("RECORDING_FILTER_TYPING_MIN_BURST_EVENTS", "5")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_DETECT", "0")
    s = load_recording_filter_settings()
    assert s.typing_min_burst_events == 5
    assert s.typing_detect_enabled is False
