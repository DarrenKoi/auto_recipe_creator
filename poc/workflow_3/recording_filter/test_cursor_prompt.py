"""cursor_prompt 가 3-변형 커서 + 출력 스키마 키를 담는지 가드한다."""

from poc.workflow_3.recording_filter.cursor_prompt import (
    cursor_system_prompt,
    cursor_user_prompt,
)


def test_system_prompt_mentions_three_cursor_variants():
    sys = cursor_system_prompt()
    assert "DVR" in sys
    assert "RCS" in sys
    assert "SEM Monitor" in sys


def test_user_prompt_declares_json_schema():
    user = cursor_user_prompt()
    for key in ("cursor_visible", "cursor_kind", "cursor_bbox", "confidence"):
        assert key in user
