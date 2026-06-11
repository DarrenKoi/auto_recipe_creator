"""frame_reduce 합성 프레임 테스트 — 변화 blob 주입 → 생존 + native bbox."""

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.frame_reduce import (
    ChangeEvent,
    collect_frame_paths,
    reduce_frames,
)
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


def _write_frame(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, format="JPEG", quality=95)


def _base(h=400, w=600, value=30):
    return np.full((h, w), value, dtype=np.uint8)


def test_reduce_keeps_only_changed_frames(tmp_path):
    # 3장: f0 (base), f1 (큰 흰 사각형 = 변화), f2 (f1 과 동일 = 변화 없음)
    f0 = _base()
    f1 = _base()
    f1[100:200, 150:300] = 255            # 100x150 = 15000px 변화 blob
    f2 = f1.copy()
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)
    _write_frame(tmp_path / "rec_rcs_0002_00000600ms.jpg", f2)

    settings = RecordingFilterSettings(min_change_area_px=5000)
    events = reduce_frames(tmp_path, settings)

    # f0->f1 은 큰 변화로 생존, f1->f2 는 변화 없어 탈락.
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ChangeEvent)
    assert ev.rank == 0
    assert ev.frame_path.endswith("rec_rcs_0001_00000300ms.jpg")
    assert ev.timestamp_sec == 0.3


def test_change_bbox_is_native_pixels(tmp_path):
    f0 = _base()
    f1 = _base()
    f1[100:200, 150:300] = 255
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)

    # resize_width 가 native(600) 보다 크면 다운스케일 없음 -> bbox 가 native 와 정합.
    settings = RecordingFilterSettings(min_change_area_px=5000, resize_width=4000)
    events = reduce_frames(tmp_path, settings)
    bbox = events[0].change_bbox
    # dilate(5x5, 2회) 로 약간 팽창하므로 여유 두고 검증.
    assert 130 <= bbox["left"] <= 160
    assert 80 <= bbox["top"] <= 110
    assert 290 <= bbox["right"] <= 320
    assert 190 <= bbox["bottom"] <= 220


def test_below_threshold_dropped(tmp_path):
    f0 = _base()
    f1 = _base()
    f1[10:15, 10:15] = 255                # 25px << 5000 임계
    _write_frame(tmp_path / "rec_rcs_0000_00000000ms.jpg", f0)
    _write_frame(tmp_path / "rec_rcs_0001_00000300ms.jpg", f1)
    events = reduce_frames(tmp_path, RecordingFilterSettings(min_change_area_px=5000))
    assert events == []


def test_collect_frame_paths_sorted(tmp_path):
    for name in ["rec_rcs_0002_x.jpg", "rec_rcs_0000_x.jpg", "rec_rcs_0001_x.jpg"]:
        _write_frame(tmp_path / name, _base())
    paths = collect_frame_paths(tmp_path)
    assert [p.name for p in paths] == [
        "rec_rcs_0000_x.jpg",
        "rec_rcs_0001_x.jpg",
        "rec_rcs_0002_x.jpg",
    ]
