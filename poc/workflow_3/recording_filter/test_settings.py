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
    """env 로 임계값을 바꿀 수 있어야 한다(CLI 인자 없음 규칙).

    (2026-08-11 리뷰 E2) 5개 필드 중 2개만 덮여 있었다. 오피스는 첫 실행 뒤 이
    knob 들을 눈으로 못 보고(=코드 없이) 튜닝하며, 국소성 가드(C2) 이후에는 이들이
    타이핑 탐지의 주요 레버다 - env 이름에 오타가 있으면 조용히 기본값이 유지되고
    엔지니어는 "임계값을 바꿨는데 아무 변화가 없다"만 보게 된다.
    """
    monkeypatch.setenv("RECORDING_FILTER_TYPING_MIN_BURST_EVENTS", "5")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_DETECT", "0")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_CURSOR_STILL_PX", "16")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_BURST_IDLE_SEC", "2.5")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_FOCUS_MAX_SEC", "3.5")
    s = load_recording_filter_settings()
    assert s.typing_min_burst_events == 5
    assert s.typing_detect_enabled is False
    assert s.typing_cursor_still_px == 16
    assert s.typing_burst_idle_sec == 2.5
    assert s.typing_focus_max_sec == 3.5


def test_typing_locality_settings_defaults_and_env_override(monkeypatch):
    """국소성 가드(리뷰 C2)의 두 임계값도 기본값 + env 를 고정한다."""
    s = RecordingFilterSettings()
    assert s.typing_roi_max_px == 200
    assert s.typing_roi_max_area_px == 40000

    monkeypatch.setenv("RECORDING_FILTER_TYPING_ROI_MAX_PX", "120")
    monkeypatch.setenv("RECORDING_FILTER_TYPING_ROI_MAX_AREA_PX", "20000")
    loaded = load_recording_filter_settings()
    assert loaded.typing_roi_max_px == 120
    assert loaded.typing_roi_max_area_px == 20000
