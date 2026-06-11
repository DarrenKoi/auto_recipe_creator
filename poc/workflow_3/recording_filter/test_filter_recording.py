"""filter_recording e2e — 합성 녹화 폴더 + 가짜 client 로 산출물 검증."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.filter_recording import run_filter
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def chat_with_image_b64(self, **kwargs):
        payload = {
            "cursor_visible": True,
            "cursor_kind": "rcs_black_arrow",
            "cursor_bbox": {"left": 480, "top": 180, "right": 520, "bottom": 220},
            "confidence": 0.9,
            "evidence": "fake",
        }
        return _FakeResponse(json.dumps(payload))


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, format="JPEG", quality=95)


def _recording_dir(tmp_path):
    rec = tmp_path / "tag123" / "recording"
    base = np.full((400, 600), 30, dtype=np.uint8)
    f1 = base.copy()
    f1[80:180, 250:360] = 255              # 커서(px ~300,120) 근처 큰 변화 -> 클릭
    _write(rec / "tag_rcs_0000_00000000ms.jpg", base)
    _write(rec / "tag_rcs_0001_00000300ms.jpg", f1)
    _write(rec / "tag_rcs_0002_00000600ms.jpg", f1.copy())   # 변화 없음 -> 탈락
    return rec


def test_run_filter_produces_artifacts(tmp_path):
    rec = _recording_dir(tmp_path)
    settings = RecordingFilterSettings(vlm_request_delay_sec=0.0, min_change_area_px=5000)
    status = run_filter(input_dir=rec, settings=settings, client=_FakeClient())
    assert status == "success"

    out_dir = rec.parent / "recording_filter"
    assert (out_dir / "change_events.json").exists()
    assert (out_dir / "interaction_timeline.json").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "change_events").is_dir()

    timeline = json.loads((out_dir / "interaction_timeline.json").read_text(encoding="utf-8"))
    assert len(timeline["events"]) == 1
    assert timeline["events"][0]["action"] == "click"


def test_run_filter_not_enough_frames(tmp_path):
    rec = tmp_path / "tag" / "recording"
    _write(rec / "tag_rcs_0000_00000000ms.jpg", np.full((400, 600), 30, dtype=np.uint8))
    status = run_filter(input_dir=rec, settings=RecordingFilterSettings(), client=_FakeClient())
    assert status == "not_enough_frames"
