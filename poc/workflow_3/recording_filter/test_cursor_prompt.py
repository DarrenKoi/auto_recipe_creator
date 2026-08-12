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


def test_system_prompt_names_all_three_static_decoys():
    """이 창의 고정 그래픽 셋을 위치와 함께 배제한다(2026-08-12 사용자 확인).

    A) 'Full Size' 버튼과 라이브 SEM 영상 사이의 손바닥 아이콘
    B) 우상단 타이틀바의 닫기 'X' 버튼
    C) 라이브 SEM 박스 좌상단 안쪽의 '>' 마크

    셋 다 모든 프레임의 같은 자리에 있어, 모델이 진짜 커서를 못 찾을 때 되돌아가는
    자리다 - 한 번 물면 세션 내내 일관되게 틀린 트랙이 나온다.
    """
    sys = cursor_system_prompt().lower()
    assert "full size" in sys
    assert "open-palm" in sys or "open palm" in sys
    assert "close button" in sys
    assert "top-right" in sys
    assert "'>'" in sys or "chevron" in sys
    assert "top-left" in sys


def test_system_prompt_handles_pointer_over_the_close_button():
    """커서가 닫기 X 위에 있을 때 화살표를 보고하게 한다(가장 잦은 실패 케이스).

    이 순간 모델은 화살표와 X 를 분리하지 못하고 손바닥/'>' 로 되돌아갔다.
    """
    sys = cursor_system_prompt().lower()
    assert "arrow" in sys
    assert "cursor_visible=false" in sys


def test_system_prompt_distinguishes_pointing_hand_from_palm():
    """손가락 하나 편 커서 vs 손바닥 아이콘을 형태로 구분하게 한다."""
    sys = cursor_system_prompt().lower()
    assert "index finger" in sys
    assert "open-palm" in sys or "open palm" in sys
