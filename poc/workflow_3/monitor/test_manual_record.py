"""수동 녹화 런처 단위 테스트 - RCS/Windows 없이 Mac 에서 돈다.

`uv run python poc/workflow_3/monitor/test_manual_record.py` 로 직접 실행.
"""

import json

from poc.workflow_3.monitor.manual_record import (
    ManualRecordSettings,
    budget_stop_reason,
    dir_size_mb,
    manual_recording_dir,
    parse_eqp_from_title,
    pick_window_row,
    resolve_capture_handles,
    sanitize_eqp_for_path,
    _watch_until_stop,
)
from poc.workflow_3.monitor.frame_meta import (
    FrameMetaWriter,
    build_meta_record,
    classify_occlusion,
    normalize_hits_to_root,
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


# ---------------------------------------------------------------------------
# 2026-08-10 최종 리뷰 FINDING 1 - WindowFromPoint 는 자식 컨트롤 핸들을 준다.
# root 로 정규화하지 않으면 MFC 계열인 RCS 창은 항상 "full" 로 찍혀 분석에서
# 모든 프레임이 폐기된다. NULL(0) 결과도 "남의 창" 이 아니라 "정보 없음" 이다.
# ---------------------------------------------------------------------------

def test_classify_occlusion_unknown_when_all_hits_are_zero():
    """WindowFromPoint 가 전부 NULL(0) 이면 판정하지 않는다(full 아님)."""
    assert classify_occlusion([0, 0, 0, 0, 0], {10}) == "unknown"
    print("[OK] test_classify_occlusion_unknown_when_all_hits_are_zero")


def test_classify_occlusion_ignores_zero_hits_among_valid_ones():
    """0 은 표본에서 빠지고 나머지로 판정한다."""
    assert classify_occlusion([0, 10, 10, 0, 10], {10}) == "none"
    print("[OK] test_classify_occlusion_ignores_zero_hits_among_valid_ones")


def test_normalize_hits_to_root_maps_child_handles_to_our_root():
    """자식 컨트롤 핸들을 root 로 올리면 우리 창으로 인식된다(가림 없음)."""
    child_hits = [201, 202, 203, 204, 205]      # 원시 WindowFromPoint 결과
    normalized = normalize_hits_to_root(child_hits, lambda _handle: 10)
    assert normalized == [10, 10, 10, 10, 10], normalized
    assert classify_occlusion(normalized, {10}) == "none"
    print("[OK] test_normalize_hits_to_root_maps_child_handles_to_our_root")


def test_normalize_hits_to_root_detects_foreign_window():
    """남의 창 위 표본은 root 로 올려도 우리 창이 아니다(진짜 가림은 여전히 잡힌다)."""
    normalized = normalize_hits_to_root([301, 302], lambda _handle: 99)
    assert classify_occlusion(normalized, {10}) == "full"
    print("[OK] test_normalize_hits_to_root_detects_foreign_window")


def test_normalize_hits_to_root_keeps_zero_as_no_information():
    """0/None 입력은 root 조회 없이 None(정보 없음)으로 남는다."""
    normalized = normalize_hits_to_root([0, None, 0], lambda _handle: 10)
    assert normalized == [None, None, None], normalized
    assert classify_occlusion(normalized, {10}) == "unknown"
    print("[OK] test_normalize_hits_to_root_keeps_zero_as_no_information")


def test_normalize_hits_to_root_treats_resolver_failure_as_unknown():
    """root 조회가 실패(예외/0)하면 원시 핸들로 되돌리지 않고 판정 불가로 둔다."""
    def _boom(_handle):
        raise OSError("GetAncestor 실패")

    assert normalize_hits_to_root([201, 202], _boom) == [None, None]
    assert normalize_hits_to_root([201, 202], lambda _handle: 0) == [None, None]
    print("[OK] test_normalize_hits_to_root_treats_resolver_failure_as_unknown")


# ---------------------------------------------------------------------------
# 2026-08-10 최종 리뷰 FINDING 5 - 가림 판정 핸들은 실제 캡처하는 창에서 뽑는다.
# ---------------------------------------------------------------------------

def test_resolve_capture_handles_uses_resolved_window_handle():
    """고른 창과 캡처 창이 같으면 그 핸들 하나를 쓴다(경고 없음)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handles = resolve_capture_handles(object(), 10, extract_fn=lambda _w: 10)
    assert handles == {10}, handles
    assert "[WARNING]" not in buf.getvalue(), buf.getvalue()
    print("[OK] test_resolve_capture_handles_uses_resolved_window_handle")


def test_resolve_capture_handles_warns_and_prefers_capture_window_on_mismatch():
    """선택 핸들과 캡처 창 핸들이 다르면 경고하고 캡처 창 기준으로 판정한다."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handles = resolve_capture_handles(
            object(), 10, "Remote Monitoring System - MCD917",
            "Remote Monitoring System - MCD916", extract_fn=lambda _w: 77,
        )
    assert handles == {77}, handles
    output = buf.getvalue()
    assert "[WARNING]" in output, output
    assert "MCD916" in output and "MCD917" in output, output
    print("[OK] test_resolve_capture_handles_warns_and_prefers_capture_window_on_mismatch")


def test_resolve_capture_handles_falls_back_to_picked_handle():
    """캡처 창 핸들을 못 얻으면 고른 핸들로 폴백한다(경고 후 계속 진행)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handles = resolve_capture_handles(object(), 10, extract_fn=lambda _w: None)
    assert handles == {10}, handles
    assert "[WARNING]" in buf.getvalue(), buf.getvalue()

    def _boom(_window):
        raise RuntimeError("추출 실패")

    handles = resolve_capture_handles(object(), 10, extract_fn=_boom)
    assert handles == {10}, handles
    print("[OK] test_resolve_capture_handles_falls_back_to_picked_handle")


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


def test_meta_writer_disables_after_write_failure():
    """쓰기 시점 실패도 이후 append 를 영구히 막는다 - 유효한 값도 써지면 안 된다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        writer = FrameMetaWriter(out_dir)
        writer.append({"frame": "a.jpg", "bad": object()})  # json.dumps 가 터진다.
        writer.append({"frame": "b.jpg"})  # 유효해도 이후는 전부 막혀야 한다.
        writer.close()

        meta_path = out_dir / "frame_meta.jsonl"
        content = meta_path.read_text(encoding="utf-8") if meta_path.exists() else ""
        lines = [line for line in content.strip().split("\n") if line]
        assert lines == [], lines
    print("[OK] test_meta_writer_disables_after_write_failure")


def test_meta_writer_warns_only_once_on_repeated_failures():
    """지속 장애에서도 경고는 최초 1회만 찍는다 - 콘솔 도배 방지."""
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        writer = FrameMetaWriter(out_dir)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            writer.append({"frame": "a.jpg", "bad": object()})
            writer.append({"frame": "b.jpg", "bad": object()})
            writer.append({"frame": "c.jpg", "bad": object()})
        writer.close()
        assert buf.getvalue().count("[WARNING]") == 1, buf.getvalue()
    print("[OK] test_meta_writer_warns_only_once_on_repeated_failures")


def test_meta_writer_write_failure_raises_nothing():
    """실패하는 append 가 연속으로 호출돼도 예외가 밖으로 새지 않는다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        writer = FrameMetaWriter(out_dir)
        writer.append({"frame": "a.jpg", "bad": object()})
        writer.append({"frame": "b.jpg", "bad": object()})
        writer.append({"frame": "c.jpg", "bad": object()})
        writer.close()
    print("[OK] test_meta_writer_write_failure_raises_nothing")


def test_budget_ok_when_under_all_limits():
    """상한 아래면 빈 문자열(계속 진행)."""
    s = ManualRecordSettings(max_frames=4000, max_disk_mb=2000)
    assert budget_stop_reason(100, 10.0, s) == ""
    print("[OK] test_budget_ok_when_under_all_limits")


def test_budget_stops_on_frame_limit():
    """프레임 상한 도달 시 frame_budget."""
    s = ManualRecordSettings(max_frames=100, max_disk_mb=2000)
    assert budget_stop_reason(100, 10.0, s) == "frame_budget"
    assert budget_stop_reason(101, 10.0, s) == "frame_budget"
    print("[OK] test_budget_stops_on_frame_limit")


def test_budget_stops_on_disk_limit():
    """디스크 상한 도달 시 disk_budget."""
    s = ManualRecordSettings(max_frames=4000, max_disk_mb=50)
    assert budget_stop_reason(10, 50.0, s) == "disk_budget"
    print("[OK] test_budget_stops_on_disk_limit")


def test_budget_frame_limit_wins_when_both_exceeded():
    """둘 다 넘으면 프레임을 먼저 보고한다(사유가 하나여야 manifest 가 명확)."""
    s = ManualRecordSettings(max_frames=10, max_disk_mb=10)
    assert budget_stop_reason(999, 999.0, s) == "frame_budget"
    print("[OK] test_budget_frame_limit_wins_when_both_exceeded")


def test_budget_zero_means_unlimited():
    """0 은 무제한 - max_sec 과 같은 규약."""
    s = ManualRecordSettings(max_frames=0, max_disk_mb=0)
    assert budget_stop_reason(10 ** 9, 10.0 ** 9, s) == ""
    print("[OK] test_budget_zero_means_unlimited")


def test_dir_size_mb_counts_files():
    """폴더 용량을 MB 로 센다."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.jpg").write_bytes(b"x" * (1024 * 1024))
        (root / "b.jpg").write_bytes(b"x" * (1024 * 512))
        assert 1.4 < dir_size_mb(root) < 1.6, dir_size_mb(root)
    print("[OK] test_dir_size_mb_counts_files")


def test_pick_window_row_single_match():
    """모니터링 창이 하나면 그대로 채택한다."""
    rows = [("Remote Monitoring System - MCD916", 10)]
    assert pick_window_row(rows, "") == rows[0]
    print("[OK] test_pick_window_row_single_match")


def test_pick_window_row_none_when_empty():
    """하나도 없으면 None."""
    assert pick_window_row([], "") is None
    print("[OK] test_pick_window_row_none_when_empty")


def test_pick_window_row_requires_eqp_when_ambiguous():
    """여러 개인데 EQP 지정이 없으면 None - 임의 선택하지 않는다."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "") is None
    print("[OK] test_pick_window_row_requires_eqp_when_ambiguous")


def test_pick_window_row_disambiguates_by_eqp():
    """EQP 를 주면 그 창을 고른다(대소문자 무시)."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "mcd917") == rows[1]
    assert pick_window_row(rows, "MCD918") is None
    print("[OK] test_pick_window_row_disambiguates_by_eqp")


def test_pick_window_row_ambiguous_prefix_returns_none():
    """부분 문자열이 여러 창에 매칭되면 첫 번째를 임의 채택하지 않는다(FINDING 1)."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "MCD91") is None
    print("[OK] test_pick_window_row_ambiguous_prefix_returns_none")


def test_pick_window_row_exact_enough_substring_still_resolves():
    """정확히 한 창에만 매칭되는 부분 문자열은 여전히 그 창을 고른다."""
    rows = [
        ("Remote Monitoring System - MCD916", 10),
        ("Remote Monitoring System - MCD917", 11),
    ]
    assert pick_window_row(rows, "MCD916") == rows[0]
    print("[OK] test_pick_window_row_exact_enough_substring_still_resolves")


def test_watch_until_stop_returns_watch_error_and_lets_caller_teardown():
    """감시 루프 중 예기치 못한 예외가 나도 사유를 반환해 teardown 을 이어가게 한다(FINDING 2).

    is_alive() 가 두 번째 호출에서 터지는 가짜 세션으로, 예외가 밖으로 새지
    않고 "watch_error" 로 잡히는지, 그리고 그 반환값으로 호출부(main 이
    실제로 하는 것과 동일하게) session.stop() 을 계속 부를 수 있는지 확인한다.
    """
    from pathlib import Path

    class _FakeSession:
        def __init__(self):
            self.calls = 0
            self.frames = []
            self.stopped_with = None

        def is_alive(self):
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("boom")
            return True

        def stop(self, reason):
            self.stopped_with = reason
            return self.frames

    fake = _FakeSession()
    settings = ManualRecordSettings(watch_interval_sec=0.0)
    reason = _watch_until_stop(fake, Path("/tmp"), settings)
    assert reason == "watch_error", reason

    # main() 은 반환된 사유로 무조건 session.stop() 을 부른다 - 그 계약을
    # 여기서 재현해 teardown 이 실제로 도달함을 확인한다.
    fake.stop(reason)
    assert fake.stopped_with == "watch_error"
    print("[OK] test_watch_until_stop_returns_watch_error_and_lets_caller_teardown")


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
    test_classify_occlusion_unknown_when_all_hits_are_zero()
    test_classify_occlusion_ignores_zero_hits_among_valid_ones()
    test_normalize_hits_to_root_maps_child_handles_to_our_root()
    test_normalize_hits_to_root_detects_foreign_window()
    test_normalize_hits_to_root_keeps_zero_as_no_information()
    test_normalize_hits_to_root_treats_resolver_failure_as_unknown()
    test_resolve_capture_handles_uses_resolved_window_handle()
    test_resolve_capture_handles_warns_and_prefers_capture_window_on_mismatch()
    test_resolve_capture_handles_falls_back_to_picked_handle()
    test_probe_points_are_inside_rect()
    test_probe_points_handles_tiny_rect()
    test_meta_writer_appends_one_json_per_line()
    test_meta_writer_survives_write_failure()
    test_meta_writer_disables_after_write_failure()
    test_meta_writer_warns_only_once_on_repeated_failures()
    test_meta_writer_write_failure_raises_nothing()
    test_budget_ok_when_under_all_limits()
    test_budget_stops_on_frame_limit()
    test_budget_stops_on_disk_limit()
    test_budget_frame_limit_wins_when_both_exceeded()
    test_budget_zero_means_unlimited()
    test_dir_size_mb_counts_files()
    test_pick_window_row_single_match()
    test_pick_window_row_none_when_empty()
    test_pick_window_row_requires_eqp_when_ambiguous()
    test_pick_window_row_disambiguates_by_eqp()
    test_pick_window_row_ambiguous_prefix_returns_none()
    test_pick_window_row_exact_enough_substring_still_resolves()
    test_watch_until_stop_returns_watch_error_and_lets_caller_teardown()
    print("\n[OK] manual_record 파싱/경로 테스트 통과")
