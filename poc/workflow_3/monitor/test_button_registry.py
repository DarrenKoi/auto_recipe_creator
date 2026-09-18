"""button_registry 단위 테스트 - VLM/OCR/실장비 없이 Mac 에서 돈다."""

import pytest

from poc.workflow_3.monitor.button_registry import ButtonSpec, resolve

FM = ButtonSpec(key="file_manager", label="File Manager", required=(("file", "manag"),))
OK_A = ButtonSpec(key="ok_file_manager", label="OK", window="File Manager", required=(("ok",),))
OK_B = ButtonSpec(key="ok_optics", label="OK", window="Optics", required=(("ok",),))
SPECS = (FM, OK_A, OK_B)


def test_resolve_by_key_label_and_window_label():
    assert resolve("file_manager", SPECS) is FM
    assert resolve("file manager", SPECS) is FM
    assert resolve("Optics/OK", SPECS) is OK_B


def test_resolve_rejects_label_shared_by_several_windows():
    with pytest.raises(ValueError) as exc:
        resolve("OK", SPECS)
    assert "ok_file_manager" in str(exc.value) and "ok_optics" in str(exc.value)


def test_resolve_unknown_lists_registered_keys():
    with pytest.raises(ValueError) as exc:
        resolve("Exit", SPECS)
    assert "file_manager" in str(exc.value)


# --- find_button: 등록 영역 crop 먼저, 실패(미검출/라벨 불일치) 시 전체 화면 ---------

from PIL import Image  # noqa: E402

from poc.workflow_3.monitor.button_registry import find_button  # noqa: E402

AMP = ButtonSpec(key="amp", label="AMP", required=(("amp",),),
                 hint="just below the File Manager button", center=(0.80, 0.95))


def _fake(locate_answers, token_answers):
    """locate 는 호출 순서대로 답하고, 받은 이미지 크기/설명을 기록한다."""
    calls = []

    def locate(image, target):
        calls.append((image.size, target.description))
        return locate_answers.pop(0)

    def read_tokens(image, point, key):
        calls.append(("ocr", image.size, point))
        return token_answers.pop(0)

    return calls, locate, read_tokens


def test_region_crop_hit_returns_full_image_point_without_hint():
    image = Image.new("RGB", (1000, 800))
    calls, locate, read = _fake([{"x": 10, "y": 20}], [["AMP"]])
    point, reason, source = find_button(image, AMP, locate_fn=locate, read_tokens_fn=read)
    # 탐색 box: 중심 (800, 760), 반폭 250 / 반높이 64 -> left 550, top 696
    assert point == {"x": 560, "y": 716}
    assert (reason, source) == ("ok", "region")
    crop_size, description = calls[0]
    assert crop_size == (450, 104)  # 오른쪽/아래는 이미지 경계에서 잘린다
    assert "AMP" in description and "below the File Manager" not in description
    assert calls[1] == ("ocr", (1000, 800), {"x": 560, "y": 716})  # 확인은 전체 이미지


def test_region_label_mismatch_falls_back_to_full_screen_with_hint():
    image = Image.new("RGB", (1000, 800))
    calls, locate, read = _fake([{"x": 10, "y": 20}, {"x": 790, "y": 770}],
                                [["File", "Manager"], ["AMP"]])
    point, reason, source = find_button(image, AMP, locate_fn=locate, read_tokens_fn=read)
    assert point == {"x": 790, "y": 770}
    assert (reason, source) == ("ok", "full")
    assert calls[2][0] == (1000, 800) and "below the File Manager" in calls[2][1]


def test_both_fail_returns_last_reason_and_no_point():
    image = Image.new("RGB", (1000, 800))
    _, locate, read = _fake([None, None], [])
    assert find_button(image, AMP, locate_fn=locate, read_tokens_fn=read) == (
        None, "not_located", "full")


def test_popup_window_label_goes_into_prompt():
    image = Image.new("RGB", (1000, 800))
    calls, locate, read = _fake([None], [])
    find_button(image, OK_A, locate_fn=locate, read_tokens_fn=read)
    assert "'File Manager' window" in calls[0][1]


# --- inventory: 등록 버튼마다 좁은 라벨 영역을 읽어 표로 (클릭 승인에는 안 쓴다) ------

from poc.workflow_3.monitor.button_registry import inventory  # noqa: E402

ROT = ButtonSpec(key="rotation", label="Rot.", required=(("rot",),))  # 위치 미등록


def test_inventory_statuses():
    image = Image.new("RGB", (1000, 800))
    FM_AT = ButtonSpec(key="file_manager", label="File Manager",
                       required=(("file", "manag"),), center=(0.80, 0.90))
    boxes = []

    def read(img, box, key):
        boxes.append((key, box))
        if key == "file_manager":
            return ["File", "Manager"]
        raise RuntimeError("ocr down")

    rows = inventory(image, (FM_AT, AMP, ROT), read_fn=read)
    assert [(r["key"], r["status"]) for r in rows] == [
        ("file_manager", "label_seen"), ("amp", "read_error"), ("rotation", "no_region")]
    # 라벨 영역은 탐색 영역이 아니라 confirm_half(버튼 한 개) 크기다
    assert boxes[0][1] == {"left": 700, "top": 708, "right": 900, "bottom": 732}


def test_inventory_not_seen_when_other_text_read():
    image = Image.new("RGB", (1000, 800))
    rows = inventory(image, (AMP,), read_fn=lambda img, box, key: ["SECS", "Terminal"])
    assert rows[0]["status"] == "not_seen" and rows[0]["tokens"] == ["SECS", "Terminal"]


def test_spec_without_required_is_rejected():
    # 빈 required 는 _confirm_point 가 strict 에서도 확인을 건너뛴다 - 등록 단계에서 막는다
    with pytest.raises(ValueError):
        ButtonSpec(key="x", label="X", required=())


# --- poll_until: 클릭 후 효과가 처음 확인된 시각(원격 지연 관찰용) -------------------

from poc.workflow_3.monitor.button_registry import poll_until  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 100.0

    def now(self):
        return self.t

    def sleep(self, sec):
        self.t += sec


def test_poll_until_reports_first_seen_time_and_checks():
    clock, answers = _Clock(), [False, False, True]

    def check():
        clock.t += 1.5  # VLM+OCR 한 번의 비용도 경과시간에 들어간다
        return answers.pop(0)

    assert poll_until(check, timeout_sec=20, interval_sec=0.5, clock=clock.now,
                      sleep=clock.sleep) == (True, 5.5, 3)


def test_poll_until_stops_at_timeout_without_extra_check():
    clock = _Clock()

    def check():
        clock.t += 2.0
        return False

    found, elapsed, checks = poll_until(check, timeout_sec=5, interval_sec=1.0,
                                        clock=clock.now, sleep=clock.sleep)
    assert (found, checks) == (False, 2) and elapsed == 5.0


def test_whole_word_label_rejects_substring_hit():
    # 'amp' 부분 일치는 'Sample'/'Clamp' 도 통과시킨다(Codex 리뷰 P1) - 짧은 라벨은 단어 전체
    image = Image.new("RGB", (1000, 800))
    amp = ButtonSpec(key="amp", label="AMP", required=(("amp",),), whole_word=True)
    _, locate, read = _fake([{"x": 1, "y": 1}], [["Sample", "Clamp"]])
    assert find_button(image, amp, locate_fn=locate, read_tokens_fn=read)[0] is None
    _, locate, read = _fake([{"x": 1, "y": 1}], [["[AMP]"]])
    assert find_button(image, amp, locate_fn=locate, read_tokens_fn=read)[0] == {"x": 1, "y": 1}
    rows = inventory(image, (ButtonSpec(key="amp", label="AMP", required=(("amp",),),
                                        whole_word=True, center=(0.5, 0.5)),),
                     read_fn=lambda img, box, key: ["Sample"])
    assert rows[0]["status"] == "not_seen"
