"""수동 녹화 런처 단위 테스트 - RCS/Windows 없이 Mac 에서 돈다.

`uv run python poc/workflow_3/monitor/test_manual_record.py` 로 직접 실행.
"""

from poc.workflow_3.monitor.manual_record import (
    manual_recording_dir,
    parse_eqp_from_title,
    sanitize_eqp_for_path,
)


def test_parse_eqp_from_plain_title():
    """표준 제목에서 EQP 를 뽑는다."""
    assert parse_eqp_from_title("Remote Monitoring System - MCD916") == "MCD916"
    print("[OK] test_parse_eqp_from_plain_title")


def test_parse_eqp_strips_surrounding_whitespace():
    """접두어 뒤 공백은 제거된다."""
    assert parse_eqp_from_title("Remote Monitoring System -   MCD916  ") == "MCD916"
    print("[OK] test_parse_eqp_strips_surrounding_whitespace")


def test_parse_eqp_keeps_trailing_tokens():
    """EQP 뒤에 부가 정보가 붙어도 통째로 보존한다(정규화는 sanitize 담당)."""
    assert parse_eqp_from_title("Remote Monitoring System - MCD916 (Online)") == "MCD916 (Online)"
    print("[OK] test_parse_eqp_keeps_trailing_tokens")


def test_parse_eqp_returns_empty_for_prefix_only():
    """접두어만 있고 EQP 가 없으면 빈 문자열."""
    assert parse_eqp_from_title("Remote Monitoring System -") == ""
    assert parse_eqp_from_title("Remote Monitoring System - ") == ""
    print("[OK] test_parse_eqp_returns_empty_for_prefix_only")


def test_parse_eqp_returns_empty_for_other_window():
    """다른 창 제목이면 빈 문자열."""
    assert parse_eqp_from_title("RCS - Main") == ""
    assert parse_eqp_from_title("") == ""
    print("[OK] test_parse_eqp_returns_empty_for_other_window")


def test_parse_eqp_is_case_insensitive_on_prefix():
    """접두어 대소문자는 무시한다(창 제목 표기 흔들림 대비)."""
    assert parse_eqp_from_title("REMOTE MONITORING SYSTEM - MCD916") == "MCD916"
    print("[OK] test_parse_eqp_is_case_insensitive_on_prefix")


def test_sanitize_replaces_path_hostile_chars():
    """폴더명에 못 쓰는 문자는 밑줄로 바꾼다."""
    assert sanitize_eqp_for_path("MCD916 (Online)") == "MCD916_Online"
    assert sanitize_eqp_for_path("A/B:C*D") == "A_B_C_D"
    print("[OK] test_sanitize_replaces_path_hostile_chars")


def test_sanitize_falls_back_for_empty():
    """빈 입력은 unknown_eqp 로 떨어진다 - 프레임을 잃지 않기 위해서."""
    assert sanitize_eqp_for_path("") == "unknown_eqp"
    assert sanitize_eqp_for_path("   ") == "unknown_eqp"
    assert sanitize_eqp_for_path("///") == "unknown_eqp"
    print("[OK] test_sanitize_falls_back_for_empty")


def test_manual_recording_dir_shape():
    """경로는 <root>/<eqp>/_manual/<tag>/recording 형태다."""
    from poc.workflow_3 import ALIGN_IMAGES_DIR

    path = manual_recording_dir("MCD916", "20260810_140000")
    assert path == ALIGN_IMAGES_DIR / "MCD916" / "_manual" / "20260810_140000" / "recording", path
    print("[OK] test_manual_recording_dir_shape")


if __name__ == "__main__":
    test_parse_eqp_from_plain_title()
    test_parse_eqp_strips_surrounding_whitespace()
    test_parse_eqp_keeps_trailing_tokens()
    test_parse_eqp_returns_empty_for_prefix_only()
    test_parse_eqp_returns_empty_for_other_window()
    test_parse_eqp_is_case_insensitive_on_prefix()
    test_sanitize_replaces_path_hostile_chars()
    test_sanitize_falls_back_for_empty()
    test_manual_recording_dir_shape()
    print("\n[OK] manual_record 파싱/경로 테스트 통과")
