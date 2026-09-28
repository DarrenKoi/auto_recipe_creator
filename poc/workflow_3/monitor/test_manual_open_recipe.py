"""manual_open_recipe 행 찾기: 맨 위로 -> 쪽 넘김 -> 전체 일치 확인 / 목록 끝 종료 / 제목 오인 거부."""

from PIL import Image

from poc.workflow_3.monitor import manual_open_recipe as mor

HEADER = {"x": 500, "y": 100}
ROW0_Y, ROW_H, VISIBLE = 300, 24, 5


class FakeList:
    """VISIBLE 행만 보이는 목록. VLM 대역은 대상이 안 보이면 첫 행을 찍는다(환각 흉내)."""

    def __init__(self, names, offset, header_tokens=("Class",)):
        self.names, self.offset, self.header_tokens = names, offset, list(header_tokens)
        self.scrolls = []

    def capture(self, _window):
        return Image.new("L", (1000, 1000), color=(self.offset * 7) % 256)

    def locate(self, _image, target):
        if target.key.endswith("_header"):
            return dict(HEADER)
        visible = self.names[self.offset:self.offset + VISIBLE]
        want = target.description.split("'")[1]
        i = visible.index(want) if want in visible else 0
        return {"x": 480, "y": ROW0_Y + i * ROW_H}

    def read(self, _image, box, label):
        if label.endswith("_header"):
            return self.header_tokens
        row = ((box["top"] + box["bottom"]) // 2 - ROW0_Y + ROW_H // 2) // ROW_H
        return self.names[self.offset + row].split("_")  # OCR 이 '_' 에서 쪼갠 경우

    def scroll(self, _window, _image, _point, dy):
        self.scrolls.append(dy)
        self.offset = max(0, min(len(self.names) - VISIBLE, self.offset - dy))
        return True


def _find(fake, name):
    return mor.find_row_in_column(
        None, "Class", name, capture_fn=fake.capture, locate_fn=fake.locate,
        read_fn=fake.read, scroll_fn=fake.scroll, sleep_fn=lambda _s: None,
    )


NAMES = [f"RJ1BXXX_CG{6290 + i}" for i in range(30)] + ["RJ1BXXX_CG6300A"]


def test_scrolls_to_top_then_pages_down_to_exact_row():
    fake = FakeList(NAMES, offset=20)  # 대상(인덱스 10)이 현재 화면보다 위에 있다
    _image, point, result = _find(fake, "RJ1BXXX_CG6300")
    assert result == mor.RESULT_FOUND
    assert NAMES[fake.offset + (point["y"] - ROW0_Y) // ROW_H] == "RJ1BXXX_CG6300"
    assert fake.scrolls[0] > 0 and fake.scrolls[-1] < 0


def test_missing_name_stops_at_list_end():
    fake = FakeList(NAMES, offset=0)
    _image, point, result = _find(fake, "RJ1BXXX_CG9999")
    assert point is None and result == mor.RESULT_ROW_NOT_FOUND
    assert fake.offset == len(NAMES) - VISIBLE


def test_exact_match_and_title_rejection():
    assert mor.name_matches(["RJ1BXXX", "CG6300"], "RJ1BXXX_CG6300")
    assert not mor.name_matches(["RJ1BXXX_CG6300A"], "RJ1BXXX_CG6300")
    fake = FakeList(NAMES, offset=0, header_tokens=("File", "Manager(", "Class,", "IDW,"))
    assert _find(fake, "RJ1BXXX_CG6300")[2] == mor.RESULT_HEADER_NOT_CONFIRMED
