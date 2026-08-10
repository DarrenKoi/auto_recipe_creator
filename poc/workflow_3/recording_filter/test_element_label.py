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


def _assert_valid_box(box, width, height):
    assert 0 <= box["left"] < box["right"] <= width, box
    assert 0 <= box["top"] < box["bottom"] <= height, box


def test_crop_box_is_centered_and_clamped():
    box = crop_box_around(600, 400, 200, 1200, 800)
    assert box == {"left": 500, "top": 300, "right": 700, "bottom": 500}, box
    # 좌상단 모서리에서도 이미지 밖으로 안 나간다.
    edge = crop_box_around(10, 10, 200, 1200, 800)
    assert edge["left"] == 0 and edge["top"] == 0, edge


def test_crop_box_far_beyond_right_bottom_stays_inside_frame():
    """오른쪽/아래로 프레임을 한참 벗어난 점도 유효한 박스를 준다(회귀: right<=left)."""
    box = crop_box_around(5000, 5000, 260, 1200, 800)
    _assert_valid_box(box, 1200, 800)


def test_crop_box_far_negative_stays_inside_frame():
    """왼쪽/위로 한참 벗어난(음수) 점도 유효한 박스를 준다."""
    box = crop_box_around(-5000, -5000, 260, 1200, 800)
    _assert_valid_box(box, 1200, 800)


def test_crop_box_four_corners_stay_inside_frame():
    width, height = 1200, 800
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        box = crop_box_around(x, y, 260, width, height)
        _assert_valid_box(box, width, height)


def test_crop_box_side_larger_than_image_clamps_to_full_frame():
    box = crop_box_around(600, 400, 5000, 1200, 800)
    _assert_valid_box(box, 1200, 800)
    assert box == {"left": 0, "top": 0, "right": 1200, "bottom": 800}, box


def test_crop_box_side_of_one_still_valid():
    box = crop_box_around(600, 400, 1, 1200, 800)
    _assert_valid_box(box, 1200, 800)


def test_pick_nearest_item_uses_click_point():
    items = [
        {"text": "Cancel", "bbox": {"left": 0, "top": 0, "right": 40, "bottom": 20}},
        {"text": "Start", "bbox": {"left": 100, "top": 100, "right": 160, "bottom": 130}},
    ]
    # crop 원점이 (500, 300) 이고 클릭이 (630, 415) 면 crop 좌표로 (130, 115) -> Start.
    picked = pick_nearest_item(items, (630, 415), (500, 300))
    assert picked["text"] == "Start", picked


def test_pick_nearest_item_empty():
    assert pick_nearest_item([], (10, 10), (0, 0)) is None


def test_label_uses_ocr_and_skips_vlm():
    """OCR 이 텍스트를 주면 VLM 은 호출되지 않는다 - 비용 설계의 핵심.

    회귀 가드: bbox 키가 `bbox` 대신 `box` 등으로 잘못되면 pick_nearest_item 이
    모든 OCR 항목을 못 찾아 매번 VLM 으로 새 나간다 - 그런데 그 상태로도 다른
    단정문(반환 타입/텍스트 존재)은 통과할 수 있어 vlm.calls 를 직접 세는 이
    단정문만이 그 결함을 잡는다.
    """
    ocr_text = json.dumps([
        {"text": "Start Measurement",
         "bbox": {"left": 100, "top": 100, "right": 220, "bottom": 130}},
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


def test_label_far_out_of_bounds_click_does_not_raise():
    """프레임을 한참 벗어난 클릭 지점도 던지지 않고 none 을 준다(Task 6 의 업스트림

    커서 좌표는 프레임 밖으로 살짝 벗어나는 게 흔한 일이라, crop_box_around 가
    유효한 박스를 못 만들면 image.crop() 에서 ValueError 로 전체 파이프라인이 죽는다.
    """
    settings = RecordingFilterSettings(element_crop_px=260)
    label = label_element(
        _image(), (5000, 5000), settings, ocr_client=None, vlm_client=None,
    )
    assert label.source == "none", label
