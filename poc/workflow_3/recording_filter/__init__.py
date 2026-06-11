"""recording_filter — 녹화 프레임 필터 + 상호작용 타임라인 (오프라인 온디맨드).

RecordingSession(monitor/recording.py) 이 남긴 tool 창 녹화 프레임을
(1) cv2 변화 이벤트로 축소하고 (2) VLM 커서 탐지로 클릭을 추출해
interaction_timeline.json 으로 만든다. 자세한 설계는
docs/superpowers/specs/2026-06-11-recording-filter-design.md 참고.
"""

from poc.workflow_3.recording_filter.settings import (
    RecordingFilterSettings,
    load_recording_filter_settings,
)

__all__ = [
    "RecordingFilterSettings",
    "load_recording_filter_settings",
]
