"""시연 영상 녹화/마무리 - 순수 함수 + 합성 영상 왕복 (Mac 실행, 실장비 불필요)."""

import json
import time

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


BASE = (0.0, 0.0, 1920.0, 1080.0)


def _shots(clicks, zoom=1.3):
    return pdv.plan_shots(clicks, BASE, zoom, gap=10.0, margin=0.12, lead=0.6, hold=1.5)


def test_login_clicks_become_one_shot():
    """로그인 입력칸 네 번 클릭(몇 초 간격, 가까운 자리) = 확대 한 번, 중심은 bbox 중심."""
    login = [(2.0, 900, 480), (6.0, 900, 520), (11.0, 900, 560), (15.0, 1000, 620)]
    assert _shots(login) == [(1.4, 16.5, 950.0, 550.0)]


def test_shots_split_on_far_click_long_gap_and_zoom_off():
    far = [(2.0, 100, 50), (3.0, 1800, 1000)]            # 한 확대 화면에 안 들어감
    assert len(_shots(far)) == 2
    late = [(2.0, 900, 500), (30.0, 900, 500)]           # gap 초과
    assert len(_shots(late)) == 2
    assert _shots(far, zoom=1.0) == []


def test_camera_holds_still_through_a_shot_and_returns_home():
    shots = _shots([(2.0, 900, 480), (6.0, 1000, 620)])
    home = (960, 540)
    assert pdv.camera_target(0.0, shots, home, 1.3) == (1.0, 960, 540)
    assert pdv.camera_target(3.0, shots, home, 1.3) == pdv.camera_target(5.9, shots, home, 1.3)
    assert pdv.camera_target(3.0, shots, home, 1.3) == (1.3, 950.0, 550.0)
    assert pdv.camera_target(9.0, shots, home, 1.3) == (1.0, 960, 540)


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


def _fake_screen(monkeypatch, fail_after=None, delays=None):
    """mss/pynput 대역: 64x36 화면. 기본은 10fps 로 느린 캡처.

    fail_after 번째 캡처부터 예외. delays = 캡처 n 번째의 지연(초) 목록(끝나면 0.03).
    n 번째 캡처 화면은 밝기 n*40.
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
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def stop(self):
            pass

        def join(self, timeout=None):
            pass

    mouse = types.SimpleNamespace(Listener=_Listener,
                                  Controller=lambda: types.SimpleNamespace(position=(5, 6)))
    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=_Sct))
    monkeypatch.setitem(sys.modules, "pynput", types.SimpleNamespace(mouse=mouse))
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)


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



def test_note_keeps_idle_frames_and_boxes_leave_when_screen_changes(tmp_path, monkeypatch):
    """판독 패널이 떠 있는 정지 구간은 자르지 않는다 + 판독 영역이 바뀌면 박스를 거둔다."""
    monkeypatch.setattr(pdv, "PREVIEW_WIDTH", 320)
    _synthetic_clip(tmp_path / "clip", [])
    (tmp_path / "clip" / "notes.json").write_text(json.dumps([
        {"t": 1.4, "title": "MCD019 화면 판독", "lines": ["PM 210 → OM 모드"],
         "boxes": [{"left": 200, "top": 120, "right": 300, "bottom": 170}]},
    ], ensure_ascii=False), encoding="utf-8")
    out = pdv.main([{"clip": str(tmp_path / "clip"), "zoom": 1.0}],
                   output=str(tmp_path / "final.mp4"))
    frames, _ = imageio_ffmpeg.count_frames_and_secs(out)
    assert frames >= 85  # 1.4s~2.9s 패널 -> 멈춘 구간이 남는다

    frame = np.zeros((100, 100, 3), np.uint8)
    note, state = {"t": 1.0, "rects": [(10, 10, 40, 40)]}, {}
    assert pdv.visible_note_rects(frame, note, state) == [(10, 10, 40, 40)]
    frame[10:40, 10:40] = 255  # tool 창이 덮었다
    assert pdv.visible_note_rects(frame, note, state) == []
    frame[:] = 0  # 되돌아와도 그 note 동안은 다시 안 그린다
    assert pdv.visible_note_rects(frame, note, state) == []


def test_note_focus_skips_boxes_wider_than_the_zoomed_view():
    notes = [{"t": 1.0, "rects": [(100, 10, 140, 30), (1700, 10, 1900, 30)]},  # 점유: 행 양 끝
             {"t": 2.0, "rects": [(500, 300, 900, 600)]}]
    assert pdv.note_focus_points(notes, (0, 0, 1920, 1080), 1.3, 0.12) == [(2.0, 700.0, 450.0)]


def test_note_panel_wraps_long_lines_and_embeds_evidence(tmp_path):
    from PIL import Image

    Image.new("RGB", (640, 480), (90, 90, 90)).save(tmp_path / "evidence.jpg")
    plain = pdv.note_patch("보정 결과", ("가" * 60,), 1920)
    with_image = pdv.note_patch("보정 결과", ("가" * 60,), 1920, str(tmp_path / "evidence.jpg"))
    assert plain.shape[1] < 1920 * 0.6
    assert with_image.shape[0] > plain.shape[0] + 1920 * 0.24 * 0.7


def test_alarm_journal_becomes_video_stages(tmp_path):
    from types import SimpleNamespace

    from poc.workflow_3.monitor import demo_record_alarm as dra

    run_dir = tmp_path / "take" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    base = time.mktime((2026, 9, 29, 10, 0, 10, 0, 0, -1))
    for step, end, ms in (("ensure_rcs_ready", 12, 1500), ("connect_tool", 20, 6000)):
        (run_dir / f"step_{step}.json").write_text(json.dumps({
            "step_id": step, "elapsed_ms": ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(base + end - 10))}))
    cycle = SimpleNamespace(started_at=base + 0.4, finished_at=base + 40,
                            correction_started_at=base + 22.0, correction_finished_at=base + 31.5)
    stages = {s["stage"]: s for s in dra.journal_stages(
        run_dir, cycle, lambda e: e - (base - 5), alarm_start=0.0, tool="MCD019")}

    assert stages["alarm"] == {"stage": "alarm", "detail": "MCD019", "start": 0.0, "end": 5.5}
    assert (stages["connect_tool"]["start"], stages["connect_tool"]["end"]) == (9.0, 15.0)
    assert (stages["correction"]["start"], stages["correction"]["end"]) == (27.0, 36.5)
    assert stages["teardown"]["end"] == 45.0
    assert dra.outcome_lines("corrected")[-1] == "OK 까지 자동 완료"
    assert "엔지니어" in dra.outcome_lines("fallback_exhausted")[-1]
    subs = dra.stage_subtitles(list(stages.values()), dra.STAGE_SUBTITLES)
    assert any("MCD019" in s["text"] for s in subs)


def test_note_box_reference_is_the_screen_at_t_ref_not_the_late_note_frame():
    """점유 note 는 더블클릭 뒤에 온다 - 그때 이미 tool 창이 덮었으면 박스를 그리지 않는다."""
    listing, tool = np.zeros((100, 100, 3), np.uint8), np.full((100, 100, 3), 200, np.uint8)
    note, state = {"t": 5.0, "t_ref": 2.0, "rects": [(10, 10, 40, 40)]}, {}
    pdv.capture_note_refs(listing, 2.0, [note], state)   # List 가 보이던 시각
    assert pdv.visible_note_rects(tool, note, state) == []


def test_reading_focus_wins_over_a_nearby_click_shot():
    note_shots = [(4.4, 8.5, 850.0, 500.0)]
    click_shots = [(5.4, 7.0, 1800.0, 500.0)]
    home = (960.0, 540.0)
    assert pdv.camera_goal(6.0, note_shots, click_shots, home, 1.3) == (1.3, 850.0, 500.0)
    assert pdv.camera_goal(9.0, note_shots, click_shots, home, 1.3) == (1.0, *home)


def test_rehearsal_outcome_never_claims_a_click():
    from poc.workflow_3.monitor import demo_record_alarm as dra

    assert "리허설" in dra.outcome_lines("corrected", rehearsal=True)[-1]
    assert dra.outcome_lines("corrected")[-1] == "OK 까지 자동 완료"


def test_reference_before_the_cut_start_is_still_used(tmp_path, monkeypatch):
    """t_ref 가 컷 시작 전이어도 그 화면을 기준으로 잡는다 + 컷 시작에 걸친 note 는 남는다."""
    monkeypatch.setattr(pdv, "PREVIEW_WIDTH", 320)
    _synthetic_clip(tmp_path / "clip", [])
    (tmp_path / "clip" / "notes.json").write_text(json.dumps([
        # 0.1s 에는 흰 상자가 x=0.8 부근, 그 뒤 옮겨가 1.5s 이후 그 자리는 배경 -> 박스 안 그림
        {"t": 1.2, "t_ref": 0.1, "title": "점유", "lines": ["x"],
         "boxes": [{"left": 0, "top": 60, "right": 40, "bottom": 100}]},
    ]), encoding="utf-8")
    drawn, panels = [], []
    monkeypatch.setattr(pdv, "draw_note_boxes", lambda f, rects, *a: drawn.append(rects))
    monkeypatch.setattr(pdv, "draw_note_panel", lambda *a: panels.append(1))
    pdv.main([{"clip": str(tmp_path / "clip"), "zoom": 1.0, "start": 1.5, "end": 3.0}],
             output=str(tmp_path / "final.mp4"))
    assert panels and not drawn


def test_two_line_subtitle_renders():
    """여러 줄 자막은 Pillow 가 실수 bbox 를 줘서 Image.new 가 깨졌다(View 탭 두 줄 자막)."""
    patch = pdv.subtitle_patch("첫 줄 설명입니다.\n둘째 줄 설명입니다.", 1920)
    assert patch.shape[0] > pdv.subtitle_patch("한 줄", 1920).shape[0] * 1.5


def test_missing_clip_dropped_with_its_intro_card(tmp_path, monkeypatch):
    monkeypatch.setattr(pdv, "DEMO_ROOT", tmp_path)
    (tmp_path / "rcs_1").mkdir()
    (tmp_path / "rcs_1" / pdv.VIDEO_NAME).write_bytes(b"")
    seq = [{"card": "A"}, {"clip": "rcs_"}, {"card": "B"}, {"clip": "alarm_"}]
    assert pdv.drop_missing_clips(seq) == seq[:2]
    with pytest.raises(FileNotFoundError):
        pdv.drop_missing_clips([{"card": "B"}, {"clip": "alarm_"}])


def test_final_video_then_event_recordings_with_subtitles(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(pdv, "DEMO_ROOT", tmp_path)
    monkeypatch.setattr(pdv, "EVENTS_DIR", tmp_path / "events")
    monkeypatch.setattr(pdv, "OUT_SIZE", (64, 36))
    monkeypatch.setattr(pdv, "PREVIEW_WIDTH", 0)
    monkeypatch.setattr(pdv, "RECORDING_MIN_SEC", 2.0)
    # 완성본 10프레임
    w = imageio_ffmpeg.write_frames(str(tmp_path / "final_1.mp4"), (64, 36), fps=pdv.FPS,
                                    macro_block_size=1, ffmpeg_log_level="error")
    w.send(None)
    for _ in range(10):
        w.send(np.zeros((36, 64, 3), np.uint8))
    w.close()
    # 이벤트 녹화: attempt 폴더 아래, 종횡비 다른 프레임 3장(0s, 0.5s, 10s -> 정지 압축)
    rec = tmp_path / "events" / "MCD026-1" / "attempt_1" / "recording"
    rec.mkdir(parents=True)
    for i, ms in enumerate((0, 500, 10000)):
        cv2.imwrite(str(rec / f"frame_{i:04d}_{ms:08d}ms.jpg"), np.full((50, 40, 3), 200, np.uint8))

    out = tmp_path / "full.mp4"
    pdv.main([{"video": "final_"}, {"recording": "MCD026-1", "subtitle": "자막"}], str(out))
    count = sum(1 for _ in list(imageio_ffmpeg.read_frames(str(out)))[1:])
    # 원본 0.5 + 압축 1.0 + tail 1.5 = 3.0s (>= MIN 2.0)
    assert count == 10 + round(3.0 * pdv.FPS)
    # 출력 파일 자신은 다음 video 항목 후보에서 빠진다
    assert pdv.resolve_video("f", exclude=tmp_path / "full.mp4").name == "final_1.mp4"


def test_memo_notice_starts_at_each_screen_reading_note():
    notes = [{"t": 5.0, "title": "MCD019 접속 전 확인"}, {"t": 12.0, "title": "MCD019 화면 판독"}]
    assert pdv.memo_notice_subtitles(notes, "안내", 6.0) == [(12.0, 18.0, "안내")]
    assert pdv.subtitle_patch(pdv.MEMO_NOTICE, 1920).shape[1] < 1920  # 두 줄로 화면 폭 안


def test_image_item_is_a_still_with_subtitle_and_dropped_when_missing(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setattr(pdv, "DEMO_ROOT", tmp_path)
    cv2.imwrite(str(tmp_path / "cube_alarm.jpeg"), np.full((30, 50, 3), 128, np.uint8))
    sent = []
    writer = type("W", (), {"send": lambda self, f: sent.append(f.shape)})()
    item = {"image": "cube_alarm.jpeg", "sec": 1.0, "subtitle": "자막"}
    assert pdv.write_image(writer, item, (64, 36), 30) == 30
    assert set(sent) == {(36, 64, 3)}
    seq = [{"clip": "/nonexistent"}, {"image": "missing.jpeg"}, item]
    monkeypatch.setattr(pdv, "resolve_clip_dir", lambda name: tmp_path)
    assert pdv.drop_missing_clips(seq) == [seq[0], item]
    assert pdv.subtitle_patch(pdv.SEQUENCE[-1]["subtitle"], 1920).shape[1] < 1920


def test_reword_expands_cv_even_before_korean_particle():
    assert pdv.reword("Align Key를 CV로 찾아") == "Align Key를 Computer Vision으로 찾아"
    assert pdv.reword("CV 패턴 매칭, CVD 는 그대로") == "Computer Vision 패턴 매칭, CVD 는 그대로"
    assert pdv.reword("보정까지 사람 없이") == "보정까지 엔지니어 없이"


def test_later_subtitle_wins_overlap_and_output_never_overwrites_its_source(tmp_path, monkeypatch):
    subs = [(0.0, 20.0, "기존 자막"), (5.0, 10.0, "안내")]
    assert pdv.active_subtitle(7.0, subs)[2] == "안내"
    assert pdv.active_subtitle(12.0, subs)[2] == "기존 자막"
    assert pdv.active_subtitle(25.0, subs) is None
    monkeypatch.setattr(pdv, "DEMO_ROOT", tmp_path)
    src = tmp_path / "final_1.mp4"
    src.write_bytes(b"original")
    with pytest.raises(ValueError):
        pdv.main([{"video": str(src)}], str(src))
    assert src.read_bytes() == b"original"


def test_rehearsal_line_is_dropped_from_the_video_panel():
    from poc.workflow_3.monitor import demo_record_alarm as dra

    lines = [line for line in map(pdv.reword, dra.outcome_lines("corrected", rehearsal=True)) if line]
    assert lines == ["Align Key 위치 찾음"]
    assert pdv.reword("Align Key 위치 확정") == "Align Key 위치 찾음"  # 이미 녹화된 notes.json
