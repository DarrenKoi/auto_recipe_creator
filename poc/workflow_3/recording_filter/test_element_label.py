"""Stage 2c 요소 라벨링 - crop 기하 / OCR 우선 / VLM 폴백 테스트(클라이언트 주입)."""

import json

import numpy as np
from PIL import Image

from poc.workflow_3.recording_filter.element_label import (
    ElementLabel,
    crop_box_around,
    label_element,
    pick_nearest_item,
)
from poc.workflow_3.recording_filter.settings import RecordingFilterSettings


class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """호출 횟수를 세는 가짜 VLM/OCR 클라이언트."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat_with_image_b64(self, **_kwargs):
        self.calls += 1
        return _Resp(self.text)


class _BoomClient:
    def __init__(self):
        self.calls = 0

    def chat_with_image_b64(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("service down")


def _image(w=1200, h=800):
    return Image.fromarray(np.full((h, w, 3), 200, dtype=np.uint8), mode="RGB")


def test_crop_box_is_centered_and_clamped():
    box = crop_box_around(600, 400, 200, 1200, 800)
    assert box == {"left": 500, "top": 300, "right": 700, "bottom": 500}, box
    # 좌상단 모서리에서도 이미지 밖으로 안 나간다.
    edge = crop_box_around(10, 10, 200, 1200, 800)
    assert edge["left"] == 0 and edge["top"] == 0, edge


def test_pick_nearest_item_uses_click_point():
    items = [
        {"text": "Cancel", "box": {"left": 0, "top": 0, "right": 40, "bottom": 20}},
        {"text": "Start", "box": {"left": 100, "top": 100, "right": 160, "bottom": 130}},
    ]
    # crop 원점이 (500, 300) 이고 클릭이 (630, 415) 면 crop 좌표로 (130, 115) -> Start.
    picked = pick_nearest_item(items, (630, 415), (500, 300))
    assert picked["text"] == "Start", picked


def test_pick_nearest_item_empty():
    assert pick_nearest_item([], (10, 10), (0, 0)) is None


def test_label_uses_ocr_and_skips_vlm():
    """OCR 이 텍스트를 주면 VLM 은 호출되지 않는다 - 비용 설계의 핵심."""
    ocr_text = json.dumps([
        {"text": "Start Measurement",
         "box": {"left": 100, "top": 100, "right": 220, "bottom": 130}},
    ])
    ocr = _FakeClient(ocr_text)
    vlm = _FakeClient('{"element": "should not be used"}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert isinstance(label, ElementLabel)
    assert label.text == "Start Measurement", label
    assert label.source == "ocr", label
    assert ocr.calls == 1 and vlm.calls == 0, (ocr.calls, vlm.calls)


def test_label_falls_back_to_vlm_when_ocr_empty():
    """OCR 이 빈 결과면 VLM 이 서술한다(아이콘 버튼/라이브 영상)."""
    ocr = _FakeClient("[]")
    vlm = _FakeClient('{"element": "zoom icon button", "confidence": 0.6}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert label.text == "zoom icon button", label
    assert label.source == "vlm", label
    assert ocr.calls == 1 and vlm.calls == 1, (ocr.calls, vlm.calls)


def test_label_falls_back_to_vlm_when_ocr_raises():
    """OCR 이 던져도 VLM 폴백으로 이어진다."""
    ocr = _BoomClient()
    vlm = _FakeClient('{"element": "OK button"}')
    settings = RecordingFilterSettings(element_crop_px=260)

    label = label_element(_image(), (630, 415), settings, ocr_client=ocr, vlm_client=vlm)

    assert label.text == "OK button" and label.source == "vlm", label


def test_label_none_when_both_fail():
    """둘 다 실패하면 source=none - 이벤트 자체는 남아야 한다."""
    settings = RecordingFilterSettings(element_crop_px=260)
    label = label_element(
        _image(), (630, 415), settings, ocr_client=_BoomClient(), vlm_client=_BoomClient(),
    )
    assert label.text == "" and label.source == "none", label


def test_label_none_when_clients_missing():
    """클라이언트가 없으면 호출 없이 none."""
    settings = RecordingFilterSettings(element_crop_px=260)
    label = label_element(_image(), (10, 10), settings, ocr_client=None, vlm_client=None)
    assert label.source == "none", label
