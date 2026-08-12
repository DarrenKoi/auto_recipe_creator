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


def test_system_prompt_covers_hand_cursor():
    """손 모양 커서를 명시한다 - 클릭 순간에 가장 흔한 글리프다(2026-08-12).

    예전 프롬프트는 커서를 세 형태로 못박아, 버튼 위에서 손 모양으로 바뀐 프레임을
    모델이 커서 아님으로 흘렸다(오피스 실측 탐지율 약 50%).
    """
    sys = cursor_system_prompt()
    assert "hand" in sys.lower()
    assert "index finger" in sys.lower()


def test_user_prompt_allows_unlisted_cursor_shapes():
    """못 보던 글리프도 좌표를 돌려주게 한다 - 침묵으로 버리면 이벤트를 잃는다."""
    user = cursor_user_prompt()
    assert "hand" in user
    assert "other" in user


def test_system_prompt_size_hint_admits_larger_glyphs():
    """손/대기 커서는 화살표보다 크다 - 32px 상한이 그것들을 배제했다."""
    assert "12-48" in cursor_system_prompt()
