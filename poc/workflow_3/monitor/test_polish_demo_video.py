"""시연 영상 녹화/마무리 - 순수 함수 + 합성 영상 왕복 (Mac 실행, 실장비 불필요)."""

import json

import imageio_ffmpeg
import numpy as np
import pytest

from poc.workflow_3.monitor import demonstration_rcs_control as demo
from poc.workflow_3.monitor import polish_demo_video as pdv
from poc.workflow_3.monitor.demo_record_rcs import stage_subtitles

STAGES = [
    {"stage": "login", "detail": "", "start": 1.0, "end": 9.0},
    {"stage": "view_tab", "detail": "", "start": 9.5, "end": 14.0},
    {"stage": "visit", "detail": "MCD019", "start": 16.0, "end": 30.0},
    {"stage": "visit", "detail": "MCDC10", "start": 32.0, "end": 40.0},
]


def test_clip_range_by_stage_detail_and_explicit_override():
    assert pdv.clip_range({"stage": ["login", "view_tab"]}, STAGES, 50.0, 0.5) == (0.5, 14.5)
    assert pdv.clip_range({"stage": "visit", "detail": "MCDC10"}, STAGES, 50.0, 0.0) == (32.0, 40.0)
    assert pdv.clip_range({"stage": "visit"}, STAGES, 50.0, 0.0) == (16.0, 40.0)
    assert pdv.clip_range({"start": 3.0, "end": -5.0}, STAGES, 50.0, 0.0) == (3.0, 45.0)
    # 시작/끝을 둘 다 적으면 없는 stage 여도 그 구간을 쓴다
    assert pdv.clip_range({"stage": "in_tool", "start": 1.0, "end": 2.0}, STAGES, 50.0, 0.0) == (1.0, 2.0)
    with pytest.raises(ValueError):
        pdv.clip_range({"stage": "in_tool"}, STAGES, 50.0, 0.0)


def test_camera_follows_latest_active_click_else_home():
    clicks = [(2.0, 100, 50), (2.5, 300, 60)]
    home = (960, 540)
    assert pdv.camera_target(0.0, clicks, home, 1.3, 0.6, 1.5) == (1.0, 960, 540)
    assert pdv.camera_target(1.6, clicks, home, 1.3, 0.6, 1.5) == (1.3, 100, 50)
    assert pdv.camera_target(2.2, clicks, home, 1.3, 0.6, 1.5) == (1.3, 300, 60)  # 다음 클릭 lead
    assert pdv.camera_target(5.0, clicks, home, 1.3, 0.6, 1.5) == (1.0, 960, 540)
    assert pdv.camera_target(2.0, clicks, home, 1.0, 0.6, 1.5) == (1.0, 960, 540)  # 확대 끔


def test_step_camera_converges_smoothly():
    cam = (1.0, 0.0, 0.0)
    for _ in range(90):
        cam = pdv.step_camera(cam, (1.3, 100.0, 50.0), 1 / 30, 0.35)
    assert cam == pytest.approx((1.3, 100.0, 50.0), abs=0.2)


def test_base_rect_widens_crop_to_output_aspect_inside_frame():
    x0, y0, w, h = pdv.base_rect((0.4, 0.4, 0.6, 0.6), 1920, 1080, 16 / 9)
    assert w / h == pytest.approx(16 / 9)
    assert 0 <= x0 and x0 + w <= 1920 and 0 <= y0 and y0 + h <= 1080
    assert pdv.base_rect(None, 1920, 1080, 16 / 9) == (0.0, 0.0, 1920, 1080)


def test_view_rect_stays_inside_base_at_edges():
    base = (0.0, 0.0, 1920.0, 1080.0)
    x0, y0, w, h = pdv.view_rect((1.5, 5.0, 5.0), base)  # 좌상단 구석 클릭
    assert (x0, y0) == (0.0, 0.0) and w == pytest.approx(1280.0)


def test_pick_monitor_prefers_origin_monitor_over_first_enumerated():
    from poc.workflow_3.monitor.screen_video import ScreenVideoRecorder

    monitors = [{}, {"left": -1920, "top": 0}, {"left": 0, "top": 0}]
    assert ScreenVideoRecorder.pick_monitor(monitors, None) is monitors[2]
    assert ScreenVideoRecorder.pick_monitor(monitors, 1) is monitors[1]
    assert ScreenVideoRecorder.pick_monitor(monitors, 9) is monitors[2]


def test_fade_level_edges():
    assert pdv.fade_level(0.0, 0.0, 4.0, 0.5) == 0.0
    assert pdv.fade_level(2.0, 0.0, 4.0, 0.5) == 1.0
    assert pdv.fade_level(3.75, 0.0, 4.0, 0.5) == pytest.approx(0.5)


def test_stage_subtitles_skip_empty_and_fill_tool():
    texts = {"view_tab": "View Tab 모니터링", "visit": "{tool} 접속", "login": ""}
    subs = stage_subtitles(STAGES, texts)
    assert [s["text"] for s in subs] == ["View Tab 모니터링", "MCD019 접속", "MCDC10 접속"]
    assert subs[0]["start"] == 9.5 and subs[0]["end"] == 14.0


def test_staged_hook_reports_start_end_with_tool_even_on_error():
    calls = []

    def boom(tool_id):
        raise RuntimeError("x")

    wrapped = demo._staged(demo.STAGE_VISIT, boom, lambda *a: calls.append(a))
    with pytest.raises(RuntimeError):
        wrapped("MCD019")
    assert calls == [("visit", "start", "MCD019"), ("visit", "end", "MCD019")]
    assert demo._staged(demo.STAGE_LOGIN, boom, None) is boom
    assert demo._staged(demo.STAGE_IN_TOOL, None, print) is None


def _synthetic_clip(clip_dir, subtitles, clicks=(0.5,)):
    """1초 움직임 + 2초 정지(30fps, 320x180) 합성 녹화."""
    clip_dir.mkdir()
    writer = imageio_ffmpeg.write_frames(str(clip_dir / "raw.mp4"), (320, 180), fps=30,
                                         macro_block_size=1, ffmpeg_log_level="error")
    writer.send(None)
    for i in range(90):
        frame = np.full((180, 320, 3), 40, np.uint8)
        x = min(i, 30) * 8
        frame[60:100, x:x + 40] = 220
        writer.send(frame)
    writer.close()
    (clip_dir / "events.json").write_text(json.dumps({
        "fps": 30, "monitor": {"left": 0, "top": 0, "width": 320, "height": 180},
        "events": [{"t": t, "kind": "click", "x": 120, "y": 80} for t in clicks],
        "cursor": [[150, 90]] * 90,
    }), encoding="utf-8")
    (clip_dir / "subtitles.json").write_text(json.dumps(subtitles, ensure_ascii=False),
                                             encoding="utf-8")


@pytest.mark.parametrize("subtitles, kept_all", [
    ([], False),
    ([{"start": 1.5, "end": 2.6, "text": "정지 구간 설명"}], True),
])
def test_render_cuts_idle_unless_subtitled(tmp_path, monkeypatch, subtitles, kept_all):
    monkeypatch.setattr(pdv, "PREVIEW_WIDTH", 320)
    _synthetic_clip(tmp_path / "clip", subtitles)
    out = pdv.main([{"card": "배경", "body": "설명", "sec": 1.0},
                    {"clip": str(tmp_path / "clip"), "zoom": 1.0}], output=str(tmp_path / "final.mp4"))
    frames, _ = imageio_ffmpeg.count_frames_and_secs(out)
    clip_frames = frames - 30
    if kept_all:
        assert clip_frames >= 85
    else:
        assert 60 <= clip_frames <= 80  # 멈춘 2초 중 1초 + 끝 페이드만 남는다


def _fake_screen(monkeypatch, fail_after=None, delays=None, listener_calls=None):
    """mss/pynput 대역: 64x36 화면. 기본은 10fps 로 느린 캡처.

    fail_after 번째 캡처부터 예외. delays = 캡처 n 번째의 지연(초) 목록(끝나면 0.03).
    n 번째 캡처 화면은 밝기 n*40. listener_calls = 훅 콜백을 흉내 낼 (종류, 인자) 목록.
    """
    import sys
    import time
    import types

    calls = {"n": 0}

    class _Shot:
        width, height = 64, 36

        def __init__(self, value):
            self.bgra = bytes([min(255, value)]) * (64 * 36 * 4)

    class _Sct:
        monitors = [{}, {"left": 0, "top": 0, "width": 64, "height": 36}]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, mon):
            calls["n"] += 1
            if fail_after is not None and calls["n"] > fail_after:
                raise OSError("capture failed")
            if delays is None:
                time.sleep(0.1)
            else:
                time.sleep(delays[calls["n"] - 1] if calls["n"] <= len(delays) else 0.03)
            return _Shot((calls["n"] - 1) * 40)

    class _Listener:
        def __init__(self, **kwargs):
            self.callbacks = kwargs

        def start(self):
            for name, args in listener_calls or ():
                if name in self.callbacks:
                    self.callbacks[name](*args)

        def is_alive(self):
            return False

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    mouse = types.SimpleNamespace(Listener=_Listener,
                                  Controller=lambda: types.SimpleNamespace(position=(5, 6)))
    keyboard = types.SimpleNamespace(Listener=_Listener)
    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=_Sct))
    monkeypatch.setitem(sys.modules, "pynput", types.SimpleNamespace(mouse=mouse, keyboard=keyboard))
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)


def test_recorder_keeps_wall_clock_even_when_capture_is_slow(tmp_path, monkeypatch):
    """캡처가 느려도 영상 프레임 수 = 경과시간 x fps, 커서는 프레임마다 하나."""
    import time

    from poc.workflow_3.monitor.screen_video import ScreenVideoRecorder

    _fake_screen(monkeypatch)
    rec = ScreenVideoRecorder(tmp_path, fps=30).start()
    time.sleep(1.0)
    info = rec.stop()
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(tmp_path / "raw.mp4"))
    assert frames == rec.written == len(info["cursor"])
    assert 27 <= rec.written <= 36          # ~1초 x 30fps (복제로 채움)
    assert rec.captured < rec.written        # 실제 캡처는 더 적다
    assert info["cursor"][0] == [5, 6]
    assert info["error"] == ""


def test_recorder_reports_mid_recording_failure(tmp_path, monkeypatch):
    """녹화 도중 캡처가 죽으면 조용히 성공하지 않고 error 로 남긴다(events.json 은 보존)."""
    import time

    from poc.workflow_3.monitor.screen_video import ScreenVideoRecorder

    _fake_screen(monkeypatch, fail_after=3)
    rec = ScreenVideoRecorder(tmp_path, fps=30).start()
    time.sleep(0.6)
    info = rec.stop()
    assert "capture failed" in info["error"]
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))["error"]


def test_render_keeps_repeated_clicks_on_a_static_screen(tmp_path, monkeypatch):
    """화면이 안 바뀌어도 클릭(링)이 있는 순간은 정지 구간으로 잘리지 않는다."""
    monkeypatch.setattr(pdv, "PREVIEW_WIDTH", 320)
    counts = []
    for name, clicks in (("plain", (0.5,)), ("clicked", (0.5, 2.1, 2.4))):
        _synthetic_clip(tmp_path / name, [], clicks)
        out = pdv.main([{"clip": str(tmp_path / name), "zoom": 1.0}],
                       output=str(tmp_path / f"{name}.mp4"))
        counts.append(imageio_ffmpeg.count_frames_and_secs(out)[0])
    assert counts[1] >= counts[0] + 12  # 정지 2.0~2.5s 중 링이 있는 구간이 살아남는다


def test_recorder_places_late_capture_at_acquisition_time_and_stops_on_time(tmp_path, monkeypatch):
    """0.5초 걸린 두 번째 캡처는 0.5초부터 나온다(과거로 당기지 않는다). 영상 길이 = 정지 시각."""
    import time

    from poc.workflow_3.monitor.screen_video import ScreenVideoRecorder

    _fake_screen(monkeypatch, delays=[0.0, 0.5])
    rec = ScreenVideoRecorder(tmp_path, fps=30).start()
    time.sleep(1.0)
    info = rec.stop()
    reader = imageio_ffmpeg.read_frames(str(tmp_path / "raw.mp4"))
    next(reader)
    means = [np.frombuffer(raw, np.uint8).mean() for raw in reader]
    first_second = next(i for i, m in enumerate(means) if m > 20)  # 두 번째 화면(밝기 40)
    assert first_second >= 13                  # ~0.5초 x 30fps (이전 구현은 1~2)
    assert abs(info["duration"] - 1.0) < 0.15  # 인코딩 시간이 영상을 늘리지 않는다


def test_recorder_counts_only_human_input(tmp_path, monkeypatch):
    """injected(자동화 입력)는 세지 않고 사람 입력만 센다. 판별 불가는 따로."""
    from poc.workflow_3.monitor.screen_video import ScreenVideoRecorder

    calls = [
        ("on_click", (10, 10, "left", True, True)),    # 자동화 클릭
        ("on_click", (10, 10, "left", True, False)),   # 사람 클릭
        ("on_move", (11, 11, False)),                  # 사람 이동
        ("on_press", ("a", True)),                     # 자동화 키
        ("on_press", ("b",)),                          # 판별 불가(구 pynput)
    ]
    _fake_screen(monkeypatch, listener_calls=calls)
    info = ScreenVideoRecorder(tmp_path, fps=30).start().stop()
    assert info["human_input"] == {"move": 1, "click": 1, "scroll": 0, "key": 0, "unknown": 1}
    assert [e["injected"] for e in info["events"]] == [True, False]
