"""abort_button locator 파싱/스케일 self-test (라이브 VLM 불필요).

가짜 client 가 캔드 JSON 을 돌려주게 해 bbox -> screen 좌표 변환 경로만 검증한다.

    uv run python poc/workflow_3e/test_abort_button.py
"""

import numpy as np

from poc.workflow_3e.abort_button import locate_abort_button, locate_abort_confirm


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, text):
        self._text = text

    def chat_with_image_b64(self, **kwargs):
        return _FakeResp(self._text)


def test_visible_center_relative_1000():
    """relative_1000 bbox 중심이 프레임 픽셀로 환산된다(1000x500 프레임)."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient(
        '{"abort_button_visible": true, "coord_system": "relative_1000", '
        '"abort_button_bbox": {"left": 400, "top": 800, "right": 600, "bottom": 900}, '
        '"confidence": 0.9}'
    )
    xy = locate_abort_button(frame_bgr=frame, client=client)
    # center rel (500, 850) -> px (500, 425)
    ok = xy is not None and abs(xy[0] - 500) <= 2 and abs(xy[1] - 425) <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] visible_center_relative_1000: xy={xy}")
    return ok


def test_not_visible_returns_none():
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient('{"abort_button_visible": false, "abort_button_bbox": null}')
    xy = locate_abort_button(frame_bgr=frame, client=client)
    ok = xy is None
    print(f"[{'PASS' if ok else 'FAIL'}] not_visible_returns_none: xy={xy}")
    return ok


def test_confirm_locator_shares_schema():
    """확인 다이얼로그 locator 도 같은 스키마 키로 파싱된다."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient(
        '{"abort_button_visible": true, "coord_system": "relative_1000", '
        '"abort_button_bbox": {"left": 100, "top": 100, "right": 300, "bottom": 200}}'
    )
    xy = locate_abort_confirm(frame_bgr=frame, client=client)
    # center rel (200, 150) -> px (200, 75)
    ok = xy is not None and abs(xy[0] - 200) <= 2 and abs(xy[1] - 75) <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] confirm_locator_shares_schema: xy={xy}")
    return ok


def main():
    print("[INFO] abort_button self-test 시작")
    results = [
        test_visible_center_relative_1000(),
        test_not_visible_returns_none(),
        test_confirm_locator_shares_schema(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
