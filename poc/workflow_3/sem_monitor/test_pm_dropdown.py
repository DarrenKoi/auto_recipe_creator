"""pm_dropdown 순수 함수 테스트 (TDD).

VLM/OCR 호출부(클릭/캡처)는 Mac 에서 못 돌리므로 좌표·문자열 순수 함수만 검증한다.
실행: uv run python poc/workflow_3/sem_monitor/test_pm_dropdown.py
"""

from poc.workflow_3.sem_monitor.pm_dropdown import (
    crop_region_from_bbox,
    nearest_option,
    row_target_description,
)


def test_crop_region_pads_and_clamps():
    """coarse bbox 를 패딩하되 frame 경계 안으로 clamp 한다."""
    bbox = {"left": 100, "top": 50, "right": 200, "bottom": 250}
    region = crop_region_from_bbox(bbox, (1000, 800), pad_x_ratio=0.5, pad_y_ratio=0.2)
    assert region is not None
    l, t, r, b = region
    # bbox 바깥으로 패딩됨.
    assert l < 100 and r > 200 and t < 50 and b > 250
    # frame 안으로 clamp.
    assert 0 <= l and 0 <= t and r <= 1000 and b <= 800
    print("[OK] test_crop_region_pads_and_clamps")


def test_crop_region_none_on_bad_input():
    """bbox/frame 누락 또는 degenerate 면 None (클릭 좌표 안전)."""
    assert crop_region_from_bbox(None, (1000, 800)) is None
    assert crop_region_from_bbox({"left": 1, "top": 1, "right": 2, "bottom": 2}, None) is None
    # degenerate: pad=0 이면 폭/높이 4 미만 → None.
    tiny = {"left": 10, "top": 10, "right": 11, "bottom": 11}
    assert crop_region_from_bbox(tiny, (1000, 800), pad_x_ratio=0.0, pad_y_ratio=0.0) is None
    print("[OK] test_crop_region_none_on_bad_input")


def test_row_target_description_embeds_value():
    """mai-ui 가 그라운딩할 행 설명에 그 행의 배율 텍스트가 들어가야 한다."""
    desc = row_target_description(210.0, "210")
    assert "210" in desc
    # 드롭다운 행이라는 맥락도 줘서 라이브 이미지 숫자와 헷갈리지 않게.
    assert "dropdown" in desc.lower()
    print("[OK] test_row_target_description_embeds_value")


def test_nearest_option_picks_closest_value():
    """목표 배율값과 가장 가까운 옵션을 고른다. 빈 목록이면 None."""
    options = [
        {"value": 104.0, "text": "104"},
        {"value": 210.0, "text": "210"},
        {"value": 500.0, "text": "500"},
    ]
    assert nearest_option(options, 210.0)["text"] == "210"
    assert nearest_option(options, 220.0)["text"] == "210"   # 가장 가까운 것.
    assert nearest_option([], 210.0) is None
    print("[OK] test_nearest_option_picks_closest_value")


if __name__ == "__main__":
    test_crop_region_pads_and_clamps()
    test_crop_region_none_on_bad_input()
    test_row_target_description_embeds_value()
    test_nearest_option_picks_closest_value()
    print("\n=== pm_dropdown: 4/4 통과 ===")
