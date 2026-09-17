"""OK 버튼 locator 의 확인 게이트 - 다른 창의 OK 를 누르지 않는다(2026-09-15 오피스 오클릭)."""

import numpy as np
from types import SimpleNamespace

from poc.workflow_3.align import ok_button as ob


def _vlm(*texts):
    it = iter(texts)
    return SimpleNamespace(chat_with_image_b64=lambda **kw: SimpleNamespace(text=next(it)))


def _ocr(*texts):
    it = iter(texts)
    return SimpleNamespace(chat_with_image_path=lambda **kw: SimpleNamespace(text=next(it)))


DIALOG = ('{"dialog_visible": true, "coord_system": "pixel", '
          '"dialog_bbox": {"left": 400, "top": 300, "right": 800, "bottom": 600}}')
OK = ('{"ok_button_visible": true, "coord_system": "pixel", '
      '"ok_button_bbox": {"left": 240, "top": 240, "right": 340, "bottom": 280}}')
NO_DIALOG = '{"dialog_visible": false, "dialog_bbox": null}'
WAIT_INPUT = "Wait Input\nClick [OK] button after setting cross cursor to alignment mark."


def _locate(vlm, ocr, tmp_path, policy="lenient"):
    return ob.locate_ok_button(frame_bgr=np.zeros((600, 800), np.uint8), client=vlm,
                               ocr_client=ocr, confirm_policy=policy, debug_image_dir=tmp_path)


def test_classify_text_mismatch_is_not_unreadable():
    assert ob.classify_text(True, ["Align", "Fail"], ("align",)) == "confirmed"
    assert ob.classify_text(True, ["Recipe", "Saved"], ("align",)) == "mismatch"
    assert ob.classify_text(True, ["[Table]"], ("align",)) == "unreadable"
    assert ob.classify_text(False, [], ("align",)) == "unreadable"
    assert ob.classify_text(True, ["OK"], ("ok",), ("cancel",)) == "confirmed"
    assert ob.classify_text(True, ["Cancel"], ("ok",), ("cancel",)) == "mismatch"
    assert ob.classify_text(True, ["OK", "Cancel"], ("ok",), ("cancel",)) == "mismatch"


def test_wrong_dialog_is_rejected_even_when_lenient(tmp_path):
    assert _locate(_vlm(DIALOG, OK), _ocr("Recipe saved successfully", "OK"), tmp_path) is None


def test_only_wait_input_popup_is_accepted(tmp_path):
    """OK 는 여러 팝업에 있다 - 'Wait Input' 제목과 align 문구가 둘 다 읽혀야 한다."""
    assert ob.classify_dialog_text(True, WAIT_INPUT.split()) == "confirmed"
    assert ob.classify_dialog_text(True, ["WaitInput", "alignment", "mark"]) == "confirmed"
    assert ob.classify_dialog_text(True, ["Align", "Fail", "Continue?"]) == "mismatch"
    assert ob.classify_dialog_text(True, ["Wait", "Input", "Insert", "cassette"]) == "mismatch"
    assert _locate(_vlm(DIALOG, OK), _ocr("Align Fail\nContinue?", "OK"), tmp_path) is None


def test_unreadable_dialog_is_rejected_even_when_lenient(tmp_path):
    assert _locate(_vlm(DIALOG, OK), _ocr("", "OK"), tmp_path, "lenient") is None


def test_forbidden_label_is_rejected_in_every_policy(tmp_path):
    for policy in ("lenient", "strict", "off"):
        for label in ("Cancel", "Reject", "OK Retry"):
            assert _locate(_vlm(DIALOG, OK), _ocr(WAIT_INPUT, label), tmp_path, policy) is None


def test_unreadable_label_passes_lenient_but_not_strict(tmp_path):
    assert _locate(_vlm(DIALOG, OK), _ocr(WAIT_INPUT, ""), tmp_path, "lenient") == (690, 560)
    assert _locate(_vlm(DIALOG, OK), _ocr(WAIT_INPUT, ""), tmp_path, "strict") is None


def test_confirmed_dialog_and_label_click_inside_dialog(tmp_path):
    assert _locate(_vlm(DIALOG, OK), _ocr(WAIT_INPUT, "OK"), tmp_path, "strict") == (690, 560)


def test_no_dialog_means_no_click_without_ocr_calls(tmp_path):
    def boom(**kw):
        raise AssertionError("OCR must not run without a dialog")
    assert _locate(_vlm(NO_DIALOG), SimpleNamespace(chat_with_image_path=boom), tmp_path) is None


def test_bad_policy_string_falls_back_to_strict(monkeypatch):
    monkeypatch.setenv("ALIGN_OK_CONFIRM", "yolo")
    assert ob.load_ok_confirm_policy() == "strict"
    monkeypatch.setenv("ALIGN_OK_CONFIRM", "lenient")
    assert ob.load_ok_confirm_policy() == "lenient"
