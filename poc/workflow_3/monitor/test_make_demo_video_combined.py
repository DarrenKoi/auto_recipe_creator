"""make_demo_video_combined 의 회차 수집/번호/합성 단위 테스트.

오피스 장비 없이 Mac 에서 돈다 - 검사 대상이 화면 캡처가 아니라 **회차 정렬,
번호 매기기, 시간축 리셋, 공통 캔버스**라 단색 더미 프레임으로 충분하다.

  uv run pytest poc/workflow_3/monitor/test_make_demo_video_combined.py
"""

import json

import cv2
import numpy as np

from poc.workflow_3.monitor.make_demo_video_combined import (
    Trial,
    find_trial_dirs,
    group_by_recipe,
    load_trials,
    make_title_card,
    ordinal,
    probe_canvas,
    render_combined,
    resolve_labels,
    resolve_output_path,
    trial_start_time,
)


def _make_trial_dir(base, tag, *, frames=2, epoch=None, size=(80, 60), step_ms=1_000):
    """`captured_img_from_rcs/<tag>/recording` 한 벌을 만든다."""
    recording = base / tag / "recording"
    recording.mkdir(parents=True, exist_ok=True)
    for seq in range(frames):
        path = recording / f"{tag}_rcs_{seq:04d}_{seq * step_ms:08d}ms.jpg"
        cv2.imwrite(str(path), np.full((size[1], size[0], 3), 128, dtype=np.uint8))
    if epoch is not None:
        (recording / "recording_manifest.json").write_text(
            json.dumps({"started_epoch": epoch}), encoding="utf-8"
        )
    return recording


# ------------------------------------------------------------------
# 회차 라벨.
# ------------------------------------------------------------------


def test_ordinal_handles_teen_exception():
    """11~13 은 1/2/3 으로 끝나도 'th' 다 ('11st' 가 나가면 안 된다)."""
    assert [ordinal(n) for n in (1, 2, 3, 4)] == ["1st", "2nd", "3rd", "4th"]
    assert [ordinal(n) for n in (11, 12, 13)] == ["11th", "12th", "13th"]
    assert [ordinal(n) for n in (21, 22, 23, 101)] == ["21st", "22nd", "23rd", "101st"]
    print("[OK] test_ordinal_handles_teen_exception")


def test_resolve_labels_defaults_to_ordinal_trials():
    assert resolve_labels(3) == ["1st Trial", "2nd Trial", "3rd Trial"]
    print("[OK] test_resolve_labels_defaults_to_ordinal_trials")


def test_resolve_labels_custom_env_fills_gaps(monkeypatch):
    """지정한 만큼만 덮고, 모자라거나 빈 칸은 기본 서수로 되돌아간다."""
    monkeypatch.setenv("DEMO_COMBINED_LABELS", "Baseline, ,After tuning")
    assert resolve_labels(4) == ["Baseline", "2nd Trial", "After tuning", "4th Trial"]
    print("[OK] test_resolve_labels_custom_env_fills_gaps")


# ------------------------------------------------------------------
# 회차 수집/정렬.
# ------------------------------------------------------------------


def test_find_trial_dirs_sorts_by_start_time_not_name(tmp_path):
    """폴더 이름 정렬이 아니라 manifest 시작 시각으로 줄 세운다."""
    _make_trial_dir(tmp_path, "zzz_second", epoch=2_000.0)
    _make_trial_dir(tmp_path, "aaa_first", epoch=1_000.0)

    found = find_trial_dirs(tmp_path)

    assert [path.parent.name for path in found] == ["aaa_first", "zzz_second"]
    print("[OK] test_find_trial_dirs_sorts_by_start_time_not_name")


def test_find_trial_dirs_ignores_prelude_subfolder(tmp_path):
    """prelude 는 회차가 아니라 그 회차의 앞부분 - 별도 회차로 잡히면 안 된다."""
    recording = _make_trial_dir(tmp_path, "260824_101530", epoch=1_000.0)
    (recording / "prelude").mkdir()

    assert find_trial_dirs(tmp_path) == [recording]
    print("[OK] test_find_trial_dirs_ignores_prelude_subfolder")


def test_trial_start_time_falls_back_to_folder_tag(tmp_path):
    """manifest 가 없으면 폴더 tag(%y%m%d_%H%M%S)로 시각을 복원한다."""
    later = _make_trial_dir(tmp_path, "260824_113000")
    earlier = _make_trial_dir(tmp_path, "260824_101530")

    assert trial_start_time(earlier) < trial_start_time(later)
    print("[OK] test_trial_start_time_falls_back_to_folder_tag")


def test_group_by_recipe_keeps_recipes_apart(tmp_path):
    """자동 탐색이 서로 다른 장비/레시피를 한 영상에 섞지 않게 하는 안전장치."""
    recipe_a = tmp_path / "MCD513" / "CLS" / "RCP_A" / "captured_img_from_rcs"
    recipe_b = tmp_path / "MCD513" / "CLS" / "RCP_B" / "captured_img_from_rcs"
    _make_trial_dir(recipe_a, "260824_101530", epoch=1_000.0)
    _make_trial_dir(recipe_a, "260824_104500", epoch=2_000.0)
    _make_trial_dir(recipe_b, "260824_110000", epoch=3_000.0)

    groups = group_by_recipe(find_trial_dirs(tmp_path))

    assert {key.parent.name: len(value) for key, value in groups.items()} == {
        "RCP_A": 2, "RCP_B": 1
    }
    print("[OK] test_group_by_recipe_keeps_recipes_apart")


# ------------------------------------------------------------------
# 회차 적재.
# ------------------------------------------------------------------


def test_load_trials_restarts_each_timeline_at_zero(tmp_path):
    """회차는 각자의 t=0 을 갖는다 - 사이의 수십 분 공백을 영상에 담지 않는다."""
    first = _make_trial_dir(tmp_path, "260824_101530", frames=3, epoch=1_000.0)
    second = _make_trial_dir(tmp_path, "260824_113000", frames=2, epoch=6_000.0)

    trials = load_trials([first, second])

    assert [trial.label for trial in trials] == ["1st Trial", "2nd Trial"]
    assert [round(t, 1) for t, _ in trials[0].frames] == [0.0, 1.0, 2.0]
    assert [round(t, 1) for t, _ in trials[1].frames] == [0.0, 1.0]
    print("[OK] test_load_trials_restarts_each_timeline_at_zero")


def test_load_trials_prepends_prelude_within_a_trial(tmp_path):
    """접속 구간은 그 회차 **안에서** 앞에 붙는다 (별도 회차가 아니다)."""
    recording = _make_trial_dir(tmp_path, "260824_101530", frames=2, epoch=1_030.0)
    prelude = recording / "prelude"
    prelude.mkdir()
    for seq in range(2):
        cv2.imwrite(
            str(prelude / f"pre_rcs_{seq:04d}_{seq * 4_000:08d}ms.jpg"),
            np.full((60, 80, 3), 10, dtype=np.uint8),
        )
    (prelude / "recording_manifest.json").write_text(
        json.dumps({"started_epoch": 1_000.0}), encoding="utf-8"
    )

    trials = load_trials([recording])

    assert len(trials) == 1
    # prelude 0s/4s + 접속 30초 뒤 본 녹화 0s/1s -> 한 회차의 0/4/30/31.
    assert [round(t, 1) for t, _ in trials[0].frames] == [0.0, 4.0, 30.0, 31.0]
    print("[OK] test_load_trials_prepends_prelude_within_a_trial")


def test_load_trials_drops_empty_dir_before_numbering(tmp_path):
    """빈 폴더는 번호를 먹지 않는다 - 뒤 회차가 '3rd' 로 밀리면 자막이 거짓이 된다."""
    first = _make_trial_dir(tmp_path, "260824_101530", frames=2, epoch=1_000.0)
    empty = tmp_path / "260824_104500" / "recording"
    empty.mkdir(parents=True)
    third = _make_trial_dir(tmp_path, "260824_113000", frames=2, epoch=3_000.0)

    trials = load_trials([first, empty, third])

    assert [trial.label for trial in trials] == ["1st Trial", "2nd Trial"]
    assert [trial.tag for trial in trials] == ["260824_101530", "260824_113000"]
    print("[OK] test_load_trials_drops_empty_dir_before_numbering")


def test_load_trials_applies_per_trial_segments(monkeypatch, tmp_path):
    """DEMO_COMBINED_SEGMENTS_<n> 은 그 회차에만 걸리고 공용 설정을 이긴다."""
    first = _make_trial_dir(tmp_path, "260824_101530", frames=5, epoch=1_000.0)
    second = _make_trial_dir(tmp_path, "260824_113000", frames=5, epoch=3_000.0)
    monkeypatch.setenv("DEMO_COMBINED_SEGMENTS", "0-1")       # 공용: 앞 2장
    monkeypatch.setenv("DEMO_COMBINED_SEGMENTS_2", "2-4")     # 2회차만: 뒤 3장

    trials = load_trials([first, second])

    assert len(trials[0].frames) == 2
    assert len(trials[1].frames) == 3
    # 편집 뒤에도 회차 시작은 0 으로 다시 맞춘다.
    assert round(trials[1].frames[0][0], 1) == 0.0
    print("[OK] test_load_trials_applies_per_trial_segments")


# ------------------------------------------------------------------
# 합성.
# ------------------------------------------------------------------


def test_make_title_card_matches_canvas_and_draws_text():
    """타이틀 카드는 캔버스와 정확히 같은 크기여야 한다 (writer 가 거부한다)."""
    card = make_title_card((320, 240), "1st Trial", ["trial 1/3", "2026-08-24 10:15:30"])

    assert card.shape == (240, 320, 3)
    assert card.dtype == np.uint8
    assert card.max() > 0  # 검은 화면만 나가지 않았다.
    print("[OK] test_make_title_card_matches_canvas_and_draws_text")


def test_make_title_card_shrinks_long_label_into_narrow_canvas():
    """좁은 캔버스에서도 글자가 화면 밖으로 나가지 않는다."""
    card = make_title_card((160, 120), "A Very Long Trial Label Indeed", [])

    # 좌우 가장자리 2px 는 비어 있어야 한다(글자가 잘려 나가지 않았다는 증거).
    assert card[:, :2].max() == 0 and card[:, -2:].max() == 0
    print("[OK] test_make_title_card_shrinks_long_label_into_narrow_canvas")


def test_probe_canvas_covers_every_trial_size(tmp_path):
    """회차마다 창 크기가 달라도 한쪽이 잘리지 않게 각 축의 최대치를 잡는다."""
    small = _make_trial_dir(tmp_path, "260824_101530", size=(80, 60), epoch=1_000.0)
    wide = _make_trial_dir(tmp_path, "260824_113000", size=(120, 40), epoch=3_000.0)

    size = probe_canvas(load_trials([small, wide]), max_width=0)

    assert size == (120, 60)
    print("[OK] test_probe_canvas_covers_every_trial_size")


def test_resolve_output_path_uses_common_parent(tmp_path):
    """회차가 같은 captured_img_from_rcs 아래면 그 폴더에 결과를 놓는다."""
    first = _make_trial_dir(tmp_path, "260824_101530", epoch=1_000.0)
    second = _make_trial_dir(tmp_path, "260824_113000", epoch=3_000.0)

    out = resolve_output_path(load_trials([first, second]))

    assert out == tmp_path / "demo_combined.mp4"
    print("[OK] test_resolve_output_path_uses_common_parent")


def test_render_combined_writes_one_file_for_all_trials(tmp_path):
    """두 회차 + 타이틀 카드가 하나의 파일로 인코딩된다 (end-to-end)."""
    first = _make_trial_dir(tmp_path, "260824_101530", frames=3, epoch=1_000.0)
    second = _make_trial_dir(tmp_path, "260824_113000", frames=3, epoch=3_000.0)
    out_path = tmp_path / "demo_combined.mp4"

    result = render_combined(
        load_trials([first, second]),
        out_path,
        fps=10.0, speed=1.0, max_hold_sec=1.0, tail_hold_sec=0.5,
        max_width=0, overlay=True, title_sec=1.0,
    )

    assert result == "success"
    written = out_path if out_path.exists() else out_path.with_suffix(".avi")
    assert written.exists() and written.stat().st_size > 0
    print(f"[OK] test_render_combined_writes_one_file_for_all_trials -> {written.name}")


def test_trial_span_sec_is_zero_for_single_frame():
    """프레임 1장짜리 회차도 자막 계산에서 죽지 않는다."""
    assert Trial("1st Trial", "t", tmp := __import__("pathlib").Path("."), []).span_sec == 0.0
    assert Trial("1st Trial", "t", tmp, [(5.0, tmp)]).span_sec == 0.0
    print("[OK] test_trial_span_sec_is_zero_for_single_frame")
