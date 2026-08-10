"""수동 녹화 런처 단위 테스트 - RCS/Windows 없이 Mac 에서 돈다.

`uv run python poc/workflow_3/monitor/test_manual_record.py` 로 직접 실행.
"""

import json

from poc.workflow_3.monitor.manual_record import (
    manual_recording_dir,
    parse_eqp_from_title,
    sanitize_eqp_for_path,
)
from poc.workflow_3.monitor.frame_meta import (
    FrameMetaWriter,
    build_meta_record,
    classify_occlusion,
    probe_points,
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


def test_sanitize_rejects_dot_only_path_escape():
    """"..", ".", "..." 처럼 온통 점으로만 된 결과는 상위 디렉터리 탈출로 이어지므로 폴백한다."""
    assert sanitize_eqp_for_path("..") == "unknown_eqp"
    assert sanitize_eqp_for_path(".") == "unknown_eqp"
    assert sanitize_eqp_for_path("...") == "unknown_eqp"
    print("[OK] test_sanitize_rejects_dot_only_path_escape")


def test_sanitize_keeps_unicode_word_chars():
    """한글 등 유니코드 단어 문자는 깎이지 않는다 - 서로 다른 EQP 명이 충돌하면 안 된다."""
    assert sanitize_eqp_for_path("장비1") == "장비1"
    print("[OK] test_sanitize_keeps_unicode_word_chars")


def test_sanitize_strips_trailing_dot():
    """Windows 가 잘못 처리하는 끝 "." 은 잘라낸다."""
    assert sanitize_eqp_for_path("MCD916.") == "MCD916"
    print("[OK] test_sanitize_strips_trailing_dot")


def test_manual_recording_dir_shape():
    """경로는 <root>/<eqp>/_manual/<tag>/recording 형태다."""
    from poc.workflow_3 import ALIGN_IMAGES_DIR

    path = manual_recording_dir("MCD916", "20260810_140000")
    assert path == ALIGN_IMAGES_DIR / "MCD916" / "_manual" / "20260810_140000" / "recording", path
    print("[OK] test_manual_recording_dir_shape")


def test_classify_occlusion_none_when_all_hits_are_ours():
    """5점 모두 우리 창이면 가림 없음."""
    assert classify_occlusion([10, 10, 10, 10, 10], {10}) == "none"
    print("[OK] test_classify_occlusion_none_when_all_hits_are_ours")


def test_classify_occlusion_full_when_no_hit_is_ours():
    """한 점도 우리 창이 아니면 완전히 가려진 것."""
    assert classify_occlusion([99, 99, 99, 99, 99], {10}) == "full"
    print("[OK] test_classify_occlusion_full_when_no_hit_is_ours")


def test_classify_occlusion_partial_when_mixed():
    """일부만 우리 창이면 부분 가림 - 포커스를 안 뺏은 겹침이 여기 잡힌다."""
    assert classify_occlusion([10, 99, 10, 10, 99], {10}) == "partial"
    print("[OK] test_classify_occlusion_partial_when_mixed")


def test_classify_occlusion_accepts_child_handles():
    """자식 컨트롤 핸들도 우리 창으로 친다(our_handles 에 여러 개 허용)."""
    assert classify_occlusion([10, 11, 12, 10, 11], {10, 11, 12}) == "none"
    print("[OK] test_classify_occlusion_accepts_child_handles")


def test_classify_occlusion_unknown_when_no_hits():
    """조회 자체가 실패해 표본이 없으면 판정하지 않는다."""
    assert classify_occlusion([], {10}) == "unknown"
    print("[OK] test_classify_occlusion_unknown_when_no_hits")


def test_probe_points_are_inside_rect():
    """5개 표본점은 모두 rect 내부다(경계 포함 안 함)."""
    rect = {"left": 100, "top": 50, "right": 500, "bottom": 250}
    points = probe_points(rect)
    assert len(points) == 5, points
    for x, y in points:
        assert rect["left"] < x < rect["right"], (x, rect)
        assert rect["top"] < y < rect["bottom"], (y, rect)
    # 중앙점이 포함된다.
    assert (300, 150) in points, points
    print("[OK] test_probe_points_are_inside_rect")


def test_probe_points_handles_tiny_rect():
    """1픽셀 창에서도 터지지 않는다(클램프)."""
    points = probe_points({"left": 0, "top": 0, "right": 1, "bottom": 1})
    assert len(points) == 5, points
    print("[OK] test_probe_points_handles_tiny_rect")


def test_meta_writer_appends_one_json_per_line(tmp_path=None):
    """프레임당 정확히 1줄 JSON 이 append 된다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        writer = FrameMetaWriter(out_dir)
        writer.append(build_meta_record(
            frame_name="rec_rcs_0000_00000000ms.jpg", t_sec=0.0,
            rect={"left": 0, "top": 0, "right": 100, "bottom": 100},
            foreground_title="Remote Monitoring System - MCD916",
            occlusion="none", cursor_xy=(10, 20),
        ))
        writer.append(build_meta_record(
            frame_name="rec_rcs_0001_00000200ms.jpg", t_sec=0.2,
            rect={"left": 0, "top": 0, "right": 100, "bottom": 100},
            foreground_title="Notepad", occlusion="partial", cursor_xy=None,
        ))
        writer.close()

        lines = (out_dir / "frame_meta.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2, lines
        first = json.loads(lines[0])
        assert first["frame"] == "rec_rcs_0000_00000000ms.jpg"
        assert first["occlusion"] == "none"
        assert first["cursor_screen_xy"] == [10, 20]
        assert first["cursor_in_window"] is True
        second = json.loads(lines[1])
        assert second["cursor_screen_xy"] is None
        assert second["cursor_in_window"] is False
    print("[OK] test_meta_writer_appends_one_json_per_line")


def test_meta_writer_survives_write_failure():
    """기록 실패는 경고만 하고 삼킨다 - 프레임 손실보다 나쁜 건 없다."""
    from pathlib import Path

    writer = FrameMetaWriter(Path("/nonexistent-root/definitely/not/writable"))
    writer.append({"frame": "x.jpg"})   # 예외가 밖으로 나오면 안 된다.
    writer.close()
    print("[OK] test_meta_writer_survives_write_failure")


if __name__ == "__main__":
    test_parse_eqp_from_plain_title()
    test_parse_eqp_strips_surrounding_whitespace()
    test_parse_eqp_keeps_trailing_tokens()
    test_parse_eqp_returns_empty_for_prefix_only()
    test_parse_eqp_returns_empty_for_other_window()
    test_parse_eqp_is_case_insensitive_on_prefix()
    test_sanitize_replaces_path_hostile_chars()
    test_sanitize_falls_back_for_empty()
    test_sanitize_rejects_dot_only_path_escape()
    test_sanitize_keeps_unicode_word_chars()
    test_sanitize_strips_trailing_dot()
    test_manual_recording_dir_shape()
    test_classify_occlusion_none_when_all_hits_are_ours()
    test_classify_occlusion_full_when_no_hit_is_ours()
    test_classify_occlusion_partial_when_mixed()
    test_classify_occlusion_accepts_child_handles()
    test_classify_occlusion_unknown_when_no_hits()
    test_probe_points_are_inside_rect()
    test_probe_points_handles_tiny_rect()
    test_meta_writer_appends_one_json_per_line()
    test_meta_writer_survives_write_failure()
    print("\n[OK] manual_record 파싱/경로 테스트 통과")
