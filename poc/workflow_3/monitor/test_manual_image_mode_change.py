"""manual_image_mode_change 판정/흐름 테스트 (VLM/실장비 불필요).

핵심은 OM 과 OM-D 를 **단어 전체**로 가르는 것이다 - 부분 일치면 OM-D 화면을 OM 으로 읽고
목록에서 OM 행을 눌러도 통과한다.
"""

from poc.workflow_3.monitor.manual_image_mode_change import (
    RESULT_ALREADY,
    RESULT_ARROW_NOT_LOCATED,
    RESULT_CHANGED,
    RESULT_ITEM_NOT_CONFIRMED,
    RESULT_ITEM_NOT_LOCATED,
    RESULT_MODE_UNREADABLE,
    RESULT_REHEARSAL,
    RESULT_UNVERIFIED,
    change_image_mode,
    item_description,
    list_box,
    read_mode,
)


class FakeImage:
    def __init__(self, width=1000, height=800):
        self.width, self.height = width, height

    def crop(self, box):
        left, top, right, bottom = box
        return FakeImage(right - left, bottom - top)


class Harness:
    """화살표/항목 좌표와 OCR 결과를 대본으로 주고, 클릭 순서를 기록한다."""

    def __init__(self, *, value_reads, item_tokens=("OM-D",), arrow=(700, 300), item=(40, 90)):
        self.value_reads = list(value_reads)     # 값 판독 대본(마지막 값이 반복된다)
        self.item_tokens = list(item_tokens)
        self.arrow, self.item = arrow, item
        self.clicks, self.escapes, self.slept = [], 0, []

    # --- 협력자 ---
    def capture(self, window):
        return FakeImage()

    def locate(self, image, target):
        if target.key == "image_mode_arrow":
            return None if self.arrow is None else {"x": self.arrow[0], "y": self.arrow[1]}
        return None if self.item is None else {"x": self.item[0], "y": self.item[1]}

    def read(self, image, box, label):
        if label == "image_mode_item":
            return list(self.item_tokens)
        return list(self.value_reads.pop(0) if len(self.value_reads) > 1 else self.value_reads[0])

    def click(self, window, image, point, key):
        self.clicks.append((key, point["x"], point["y"]))

    def sleep(self, sec):
        self.slept.append(sec)

    def escape(self):
        self.escapes += 1

    def run(self, target="OM-D", **kw):
        clock = iter(range(0, 200))
        return change_image_mode(
            object(), target,
            capture_fn=self.capture, locate_fn=self.locate, read_fn=self.read,
            click_fn=self.click, sleep_fn=self.sleep, escape_fn=self.escape,
            clock=lambda: next(clock), settle_sec=0.0,
            verify_timeout_sec=3.0, verify_poll_sec=1.0, **kw,
        )


# --- read_mode: OM vs OM-D ---

def test_om_d_is_not_read_as_om():
    assert read_mode(["OM-D"]) == "OM-D"
    assert read_mode(["OM"]) == "OM"


def test_hyphen_dropped_by_ocr_still_reads_om_d():
    assert read_mode(["OMD"]) == "OM-D"


def test_two_modes_in_one_crop_is_not_a_confirmation():
    # 확인 crop 이 OM 행과 OM-D 행을 함께 삼킨 경우 - 확정하지 않는다.
    assert read_mode(["OM", "OM-D"]) is None


def test_unknown_text_reads_nothing():
    assert read_mode(["Optics...", "ABC"]) is None


def test_item_description_names_siblings_for_the_vlm():
    text = item_description("OM-D")
    assert "'OM-D'" in text and "'OM'" in text and "'SEM'" in text


def test_list_box_opens_mostly_downward_from_the_arrow():
    box = list_box({"x": 700, "y": 300}, 1000, 800)
    assert box["top"] < 300 < box["bottom"]
    assert box["bottom"] - 300 > 300 - box["top"]   # 아래로 더 넓다
    assert box["left"] < 700 and box["right"] <= 1000


# --- 흐름 ---

def test_changes_mode_and_verifies_the_new_value():
    h = Harness(value_reads=[["OM"], ["OM-D"]])
    out = h.run()
    assert out["result"] == RESULT_CHANGED and out["current"] == "OM"
    assert [c[0] for c in h.clicks] == ["image_mode_arrow", "image_mode_item"]
    # 항목 좌표는 crop 원점만큼 되돌려진다(crop 좌표 그대로 누르면 엉뚱한 곳이다).
    assert h.clicks[1][1] > 40 and h.clicks[1][2] > 90
    assert h.escapes == 0


def test_already_in_target_mode_does_not_open_the_dropdown():
    h = Harness(value_reads=[["OM-D"]])
    out = h.run()
    assert out["result"] == RESULT_ALREADY and h.clicks == []


def test_unreadable_mode_never_clicks():
    h = Harness(value_reads=[["Optics..."]])
    out = h.run()
    assert out["result"] == RESULT_MODE_UNREADABLE and h.clicks == []


def test_arrow_not_located_never_clicks():
    h = Harness(value_reads=[["OM"]], arrow=None)
    out = h.run()
    assert out["result"] == RESULT_ARROW_NOT_LOCATED and h.clicks == []


def test_rehearsal_stops_before_opening_the_dropdown():
    h = Harness(value_reads=[["OM"]])
    out = h.run(action_enabled=False)
    assert out["result"] == RESULT_REHEARSAL and h.clicks == []


def test_wrong_row_label_is_not_clicked_and_the_list_is_closed():
    # VLM 이 OM 행을 짚었다 - 목표는 OM-D 다. 누르지 않고 목록을 닫는다.
    h = Harness(value_reads=[["OM"]], item_tokens=["OM"])
    out = h.run()
    assert out["result"] == RESULT_ITEM_NOT_CONFIRMED
    assert [c[0] for c in h.clicks] == ["image_mode_arrow"]
    assert h.escapes == 1


def test_missing_row_closes_the_list():
    h = Harness(value_reads=[["OM"]], item=None)
    out = h.run()
    assert out["result"] == RESULT_ITEM_NOT_LOCATED and h.escapes == 1


def test_value_that_never_changes_is_unverified_without_a_second_click():
    h = Harness(value_reads=[["OM"]])
    out = h.run()
    assert out["result"] == RESULT_UNVERIFIED
    assert [c[0] for c in h.clicks] == ["image_mode_arrow", "image_mode_item"]
