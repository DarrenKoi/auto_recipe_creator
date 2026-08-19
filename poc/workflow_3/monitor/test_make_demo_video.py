"""make_demo_video 의 prelude 접합 / letterbox 합성 단위 테스트.

오피스 장비 없이 Mac 에서 돈다 - 검사 대상이 화면 캡처가 아니라 **시간축 접합과
캔버스 계산**이라, 프레임은 cv2 로 만든 단색 이미지로 충분하다.

  uv run pytest poc/workflow_3/monitor/test_make_demo_video.py
"""

import json

import cv2
import numpy as np

from poc.workflow_3.monitor.make_demo_video import (
    canvas_size,
    fit_into_canvas,
    merge_prelude,
    parse_segments,
    read_started_epoch,
    resolve_segments,
    scan_frames,
    trim_frames,
)


def _write_frame(directory, tag, seq, elapsed_ms, size=(80, 60)):
    """RecordingSession 파일명 규약에 맞는 더미 프레임 1장."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}_rcs_{seq:04d}_{elapsed_ms:08d}ms.jpg"
    image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def _write_manifest(directory, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recording_manifest.json").write_text(
        json.dumps(fields, ensure_ascii=False), encoding="utf-8"
    )


# ------------------------------------------------------------------
# 시간축 접합.
# ------------------------------------------------------------------


def test_merge_prelude_shifts_main_by_epoch_gap(tmp_path):
    """본 녹화는 두 manifest 의 절대 시작 시각 차이만큼 뒤로 밀린다."""
    main_dir = tmp_path / "recording"
    pre_dir = main_dir / "prelude"
    _write_frame(pre_dir, "T_pre", 0, 0)
    _write_frame(pre_dir, "T_pre", 1, 4_000)
    _write_frame(main_dir, "T", 0, 0)
    _write_frame(main_dir, "T", 1, 2_000)
    _write_manifest(pre_dir, started_epoch=1000.0)
    _write_manifest(main_dir, started_epoch=1030.0)  # 접속에 30초 걸렸다.

    merged, manifest_dir = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [0.0, 4.0, 30.0, 32.0]
    # 로그 패널 기준점은 **가장 이른** 세션이어야 한다.
    assert manifest_dir == pre_dir
    print("[OK] test_merge_prelude_shifts_main_by_epoch_gap")


def test_merge_prelude_falls_back_to_legacy_started_at(tmp_path):
    """started_epoch 이 없는 구버전 manifest 는 started_at 문자열로 접합한다."""
    main_dir = tmp_path / "recording"
    pre_dir = main_dir / "prelude"
    _write_frame(pre_dir, "T_pre", 0, 0)
    _write_frame(main_dir, "T", 0, 0)
    _write_manifest(pre_dir, started_at="2026-08-19T10:00:00")
    _write_manifest(main_dir, started_at="2026-08-19T10:00:25")

    merged, _ = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [0.0, 25.0]
    print("[OK] test_merge_prelude_falls_back_to_legacy_started_at")


def test_merge_prelude_approximates_when_manifest_missing(tmp_path):
    """기준점이 없으면 시간을 지어내지 않고 prelude 끝 +1s 로 근사한다."""
    main_dir = tmp_path / "recording"
    pre_dir = main_dir / "prelude"
    _write_frame(pre_dir, "T_pre", 0, 0)
    _write_frame(pre_dir, "T_pre", 1, 6_000)
    _write_frame(main_dir, "T", 0, 0)

    merged, _ = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [0.0, 6.0, 7.0]
    print("[OK] test_merge_prelude_approximates_when_manifest_missing")


def test_merge_prelude_rejects_negative_offset(tmp_path):
    """본 녹화가 prelude 보다 이르다고 나오면(시계 역행) 근사로 떨어진다.

    음수 오프셋을 그대로 쓰면 본 녹화가 prelude **앞**으로 가 순서가 뒤집힌다.
    """
    main_dir = tmp_path / "recording"
    pre_dir = main_dir / "prelude"
    _write_frame(pre_dir, "T_pre", 0, 0)
    _write_frame(pre_dir, "T_pre", 1, 3_000)
    _write_frame(main_dir, "T", 0, 0)
    _write_manifest(pre_dir, started_epoch=2000.0)
    _write_manifest(main_dir, started_epoch=1990.0)

    merged, _ = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [0.0, 3.0, 4.0]
    print("[OK] test_merge_prelude_rejects_negative_offset")


def test_merge_prelude_skipped_when_disabled(tmp_path, monkeypatch):
    """DEMO_VIDEO_PRELUDE=0 이면 prelude 가 있어도 본 녹화만 쓴다."""
    main_dir = tmp_path / "recording"
    pre_dir = main_dir / "prelude"
    _write_frame(pre_dir, "T_pre", 0, 0)
    _write_frame(main_dir, "T", 0, 1_000)
    monkeypatch.setenv("DEMO_VIDEO_PRELUDE", "0")

    merged, manifest_dir = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [1.0]
    assert manifest_dir == main_dir
    print("[OK] test_merge_prelude_skipped_when_disabled")


def test_merge_prelude_noop_without_prelude_dir(tmp_path):
    """prelude 폴더가 없는 기존 녹화는 그대로 통과한다 (하위호환)."""
    main_dir = tmp_path / "recording"
    _write_frame(main_dir, "T", 0, 500)

    merged, manifest_dir = merge_prelude(main_dir, scan_frames(main_dir))

    assert [round(t, 1) for t, _ in merged] == [0.5]
    assert manifest_dir == main_dir
    print("[OK] test_merge_prelude_noop_without_prelude_dir")


def test_read_started_epoch_prefers_float_field(tmp_path):
    """started_epoch 이 있으면 초 단위 문자열보다 그쪽을 쓴다(소수점 보존)."""
    _write_manifest(tmp_path, started_at="2026-08-19T10:00:00", started_epoch=12.75)
    assert read_started_epoch(tmp_path) == 12.75
    assert read_started_epoch(tmp_path / "없음") is None
    print("[OK] test_read_started_epoch_prefers_float_field")


# ------------------------------------------------------------------
# 편집 (남길 구간).
# ------------------------------------------------------------------


def test_parse_segments_basic_and_open_ends():
    """"12-45" / "120-" / "-45" 세 형식을 모두 읽는다."""
    assert parse_segments("12-45", 0.0, 300.0) == [(12.0, 45.0)]
    assert parse_segments("120-", 0.0, 300.0) == [(120.0, 300.0)]
    assert parse_segments("-45", 0.0, 300.0) == [(0.0, 45.0)]
    # 콜론 구분자와 공백/여러 구간.
    assert parse_segments(" 0:30 , 120:260 ", 0.0, 300.0) == [(0.0, 30.0), (120.0, 260.0)]
    print("[OK] test_parse_segments_basic_and_open_ends")


def test_parse_segments_negative_counts_from_end():
    """음수는 끝에서부터 센다 - 영상 길이를 모르는 채로 뒷부분을 자를 때 쓴다."""
    assert parse_segments("0--30", 0.0, 300.0) == [(0.0, 270.0)]
    assert parse_segments("-60--10", 0.0, 300.0) == [(240.0, 290.0)]
    print("[OK] test_parse_segments_negative_counts_from_end")


def test_parse_segments_rejects_garbage_loudly():
    """못 읽은 조각은 조용히 버리지 않는다 (오타로 엉뚱한 영상이 나가면 최악)."""
    assert parse_segments("abc, 45-12, 10-20", 0.0, 300.0) == [(10.0, 20.0)]
    print("[OK] test_parse_segments_rejects_garbage_loudly")


def test_resolve_segments_prefers_explicit_list():
    """SEGMENTS 가 있으면 START/END 는 무시된다 (필터 규칙은 한 벌이어야 한다)."""
    frames = [(0.0, None), (300.0, None)]
    assert resolve_segments(frames, "10-20", 100.0, 200.0) == [(10.0, 20.0)]
    print("[OK] test_resolve_segments_prefers_explicit_list")


def test_resolve_segments_from_start_end_env():
    """START/END 는 구간 하나의 축약형. END 음수는 끝에서 N초 자르기."""
    frames = [(0.0, None), (300.0, None)]
    assert resolve_segments(frames, "", 40.0, 0.0) == [(40.0, 300.0)]
    assert resolve_segments(frames, "", 40.0, -20.0) == [(40.0, 280.0)]
    assert resolve_segments(frames, "", 0.0, 0.0) == []  # 편집 없음.
    print("[OK] test_resolve_segments_from_start_end_env")


def test_trim_frames_keeps_original_times():
    """잘라낸 뒤에도 프레임 시각은 원본 그대로다.

    구간을 앞으로 당겨 붙이면 overlay 시각과 로그 패널이 원본과 어긋난다. 사라진
    간격은 build_timeline 이 압축하며 화면에 표시하는 것이 이 모듈의 규약이다.
    """
    frames = [(t, None) for t in (0.0, 10.0, 20.0, 30.0, 40.0)]
    kept = trim_frames(frames, [(0.0, 10.0), (30.0, 100.0)])
    assert [t for t, _ in kept] == [0.0, 10.0, 30.0, 40.0]
    assert trim_frames(frames, []) is frames
    print("[OK] test_trim_frames_keeps_original_times")


# ------------------------------------------------------------------
# 캔버스 / letterbox.
# ------------------------------------------------------------------


def test_canvas_size_contains_both_sources():
    """캔버스는 두 소스의 각 축 최대치를 담는다 (짝수 보정 포함)."""
    screen = np.zeros((1080, 1921, 3), dtype=np.uint8)   # 홀수 폭 - 짝수로 내려가야.
    tool = np.zeros((900, 1200, 3), dtype=np.uint8)

    assert canvas_size([screen, tool], 0) == (1920, 1080)
    # 가로 상한이 걸리면 높이는 비례 축소된다.
    assert canvas_size([screen, tool], 960) == (960, 540)
    print("[OK] test_canvas_size_contains_both_sources")


def test_fit_into_canvas_preserves_aspect_and_pads():
    """비율을 지킨 채 중앙에 놓고 남는 자리는 검게 둔다."""
    image = np.full((900, 1200, 3), 200, dtype=np.uint8)  # 4:3
    out = fit_into_canvas(image, (1920, 1080))

    assert out.shape == (1080, 1920, 3)
    # 4:3 을 높이에 맞추면 1440 폭 -> 좌우 240px 씩 검은 띠.
    assert int(out[540, 10].max()) == 0
    assert int(out[540, 1910].max()) == 0
    assert int(out[540, 960].min()) > 0
    print("[OK] test_fit_into_canvas_preserves_aspect_and_pads")


def test_fit_into_canvas_passthrough_when_same_size():
    """이미 캔버스 크기면 리사이즈 없이 그대로 돌려준다."""
    image = np.full((60, 80, 3), 7, dtype=np.uint8)
    out = fit_into_canvas(image, (80, 60))
    assert out is image
    print("[OK] test_fit_into_canvas_passthrough_when_same_size")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    class _Monkey:
        """pytest 없이 직접 실행할 때 쓰는 최소 monkeypatch 대역."""

        def __init__(self):
            self._saved = []

        def setenv(self, name, value):
            import os

            self._saved.append((name, os.environ.get(name)))
            os.environ[name] = value

        def undo(self):
            import os

            for name, old in reversed(self._saved):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old

    def _run(func):
        import inspect

        params = inspect.signature(func).parameters
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = {}
            if "tmp_path" in params:
                kwargs["tmp_path"] = Path(tmp)
            monkey = _Monkey()
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = monkey
            try:
                func(**kwargs)
            finally:
                monkey.undo()

    for _name, _func in sorted(dict(globals()).items()):
        if _name.startswith("test_") and callable(_func):
            _run(_func)
    print("[INFO] make_demo_video 테스트 완료")
