"""Assist Window 판독 self-test (VLM/OCR/실이미지 불필요).

합성 패널 이미지만 쓰므로 Mac 에서 그대로 돈다.

    uv run python poc/workflow_3/sem_monitor/test_assist_score.py
"""

from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from poc.workflow_3.sem_monitor import assist_score as asc
from poc.workflow_3.sem_monitor.assist_score import assist_panel_target


DIGIT_PANEL_SIZE = (300, 280)
DIGIT_ROW_PITCH = 30
DIGIT_FIRST_TOP = 45
DIGIT_COLUMN_LEFTS = {"Addressing1": 20, "Measurement": 220}


def _digit_font():
    """PIL 기본 비트맵 폰트. size 인자를 지원하지 않는 구버전도 그대로 돈다."""
    try:
        return ImageFont.load_default(size=15)
    except TypeError:
        return ImageFont.load_default()


def _draw_rect_glyph(draw, left, top, *, width=12, height=18, stroke=2, bar=3,
                     fill=(20, 20, 20)):
    """실측 숫자와 같은 잉크 밀도(tight bbox 대비 약 0.59)를 갖는 글자 모양을 그린다.

    PIL 기본 폰트의 글자 metric 은 Pillow 버전마다 흔들리므로, 밀도를 **설계값으로**
    고정해야 하는 테스트에서는 사각 링 + 가운데 바(= '8' 모양)로 그린다. 리뷰어가 실제
    폰트(Arial/Helvetica/Verdana, 11~16px)에서 잰 0.44~0.74(중앙값 0.58) 한가운데를
    노린 값이라, 옛 임계 0.55 + 무패딩 조합이면 이 글자는 unknown 으로 끊긴다.
    """
    draw.rectangle([left, top, left + width - 1, top + height - 1], fill=fill)
    draw.rectangle(
        [left + stroke, top + stroke, left + width - 1 - stroke, top + height - 1 - stroke],
        fill=(240, 240, 240),
    )
    mid = top + (height - bar) // 2
    draw.rectangle([left + stroke, mid, left + width - 1 - stroke, mid + bar - 1], fill=fill)


def test_panel_target_uses_proven_button_geometry():
    """패널 로케이트도 오피스에서 검증된 2단계 로케이터 기하를 따른다.

    bench_tool_window_reader 가 같은 tool 창에서 acc=1.000 을 낸 설정이다. 여기서 임의로
    다른 값을 쓰면 '입증된 설정' 이라는 근거가 사라진다.
    """
    from poc.workflow_3.rcs import bench_tool_window_reader as bench

    mine = assist_panel_target()
    theirs = bench._button_target("Queue")
    ok = (
        mine.min_crop_width == theirs.min_crop_width
        and mine.vertical_pad_min_px == theirs.vertical_pad_min_px
        and "Assist" in mine.description
    )
    print(f"[{'PASS' if ok else 'FAIL'}] panel_target_uses_proven_button_geometry")
    return ok


def _swap_asc(state, name, value):
    """assist_score 모듈 속성을 교체하고 원복용으로 저장한다."""
    state[name] = getattr(asc, name)
    setattr(asc, name, value)


def _restore_asc(state):
    for name, orig in state.items():
        setattr(asc, name, orig)


# locate_assist_layout 의 기본 통합 테스트용 이미지 크기/점 - _panel_items() 의 좌표가
# (max x=260, max y=238) 이므로, 크롭된 패널이 그보다 확실히 커야 정규화 후 항목이
# 패널 밖으로 안 잘린다(정규화가 patch 밖 항목을 버리는 동작을 이 테스트에서 우연히
# 건드리지 않기 위함 - 그건 별도 테스트가 전담한다).
_LOCATE_IMAGE_SIZE = (1000, 1000)
_LOCATE_POINT = {"x": 500, "y": 500}


def _expected_panel_size(image_size, point):
    """locate_assist_layout 의 panel_box 계산과 같은 식으로 패널 crop 크기를 구한다."""
    width, height = image_size
    left = max(0, int(point["x"] - width * asc.PANEL_LEFT_RATIO))
    right = min(width, int(point["x"] + width * asc.PANEL_RIGHT_RATIO))
    top = max(0, int(point["y"] - height * asc.PANEL_TOP_RATIO))
    bottom = min(height, int(point["y"] + height * asc.PANEL_BOTTOM_RATIO))
    return right - left, bottom - top


# ------------------------------------------------------------------
# item-free CV counting (read_assist_state) - 격자/OCR 없이 패널 픽셀만 본다.
# ------------------------------------------------------------------


def _render_count_panel(row_colors, *, header=True, size=DIGIT_PANEL_SIZE):
    """행별 잉크 색으로 합성 Assist 패널을 그린다.

    row_colors 는 위에서 아래 순서이며 각 원소는 "black"|"red"|None(빈 행)이다.
    헤더는 기본으로 그린다 - 실제 Assist 표에는 열 헤더가 항상 있고, 헤더도 검정
    잉크라 띠로 잡히므로, 헤더 없는 패널을 픽스처로 쓰면 실제와 다른 조건에서
    행 수를 검증하게 된다(초기 슬라이스에서 실제로 어긋났다).
    """
    image = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(image)
    if header:
        draw.text((10, 5), "Addressing1  Addressing2  Measurement",
                  fill=(20, 20, 20), font=_digit_font())
    for row_idx, color in enumerate(row_colors):
        if color is None:
            continue
        fill = (20, 20, 20) if color == "black" else (200, 20, 20)
        top = DIGIT_FIRST_TOP + row_idx * DIGIT_ROW_PITCH
        for left in DIGIT_COLUMN_LEFTS.values():
            _draw_rect_glyph(draw, left, top, fill=fill)
            _draw_rect_glyph(draw, left + 12, top, fill=fill)
    return image


def test_count_state_counts_black_ink_bands():
    """헤더 없는 패널에서 검정 숫자 띠 개수가 그대로 정상 행 수가 된다."""
    state = asc.read_assist_state(_render_count_panel(["black"] * 3))
    ok = (
        state.status == "usable"
        and state.ok_row_count == 3
        and state.has_red is False
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_counts_black_ink_bands: "
        f"status={state.status}, rows={state.ok_row_count}, red={state.has_red}"
    )
    return ok


def test_count_state_flags_red_and_excludes_it_from_rows():
    """붉은 행이 하나라도 있으면 has_red 이고, 그 행은 정상 행 수에 안 들어간다.

    사용자 결정(2026-08-13): 표가 새 측정마다 초기화되므로 표에 보이는 red 는 이번
    사이클의 실패다. 보수적으로 done 을 막는다 - 놓쳐도 engineer_watch_sec cap 이
    안전망이고, 잘못 닫는 쪽이 훨씬 비싸다.
    """
    state = asc.read_assist_state(_render_count_panel(["black", "red", "black"]))
    ok = state.has_red is True and state.ok_row_count == 2
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_flags_red_and_excludes_it_from_rows: "
        f"rows={state.ok_row_count}, red={state.has_red}"
    )
    return ok


def test_count_state_does_not_count_header_as_a_row():
    """헤더 텍스트도 검정 잉크라 띠로 잡힌다 - 행으로 세면 항상 1을 부풀린다."""
    state = asc.read_assist_state(_render_count_panel(["black"] * 3))
    ok = state.ok_row_count == 3
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_does_not_count_header_as_a_row: "
        f"rows={state.ok_row_count} (기대 3)"
    )
    return ok


def test_count_state_empty_table_is_zero_rows():
    """새 측정 시작 직후 표는 헤더만 남고 비어 있다 - 정상 행 0."""
    state = asc.read_assist_state(_render_count_panel([]))
    ok = state.ok_row_count == 0 and state.has_red is False
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_empty_table_is_zero_rows: "
        f"rows={state.ok_row_count}"
    )
    return ok


def test_count_state_ignores_dense_dark_block():
    """그레이스케일 SEM 썸네일 같은 밀집 어두운 덩어리는 행이 아니다.

    구 구현의 `INK_DENSE_MAX_RATIO` 가드(셀 단위)와 같은 목적이다: 썸네일을 검정
    글자로 오인해 행 수를 부풀리면 측정이 없는데도 done 이 뜬다.
    """
    image = _render_count_panel(["black"] * 3)
    draw = ImageDraw.Draw(image)
    # 숫자 띠보다 아래, 글자 높이의 몇 배가 되는 어두운 블록.
    block_top = DIGIT_FIRST_TOP + 3 * DIGIT_ROW_PITCH + 10
    draw.rectangle([20, block_top, 280, block_top + 90], fill=(60, 60, 60))
    state = asc.read_assist_state(image)
    ok = state.ok_row_count == 3
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_ignores_dense_dark_block: "
        f"rows={state.ok_row_count} (기대 3)"
    )
    return ok


def _locate_panel_with(state, *, point, image_size=_LOCATE_IMAGE_SIZE, raises=False):
    """analyze_window_target 만 스텁하고 locate_assist_panel 을 1회 호출한다.

    OCR 스텁이 필요 없다는 것 자체가 이 시임의 요점이다 - 패널 위치 확보 경로에서
    PaddleOCR 왕복이 사라졌다.
    """
    def fake_locator(*a, **k):
        if raises:
            raise RuntimeError("locator boom")
        return SimpleNamespace(point=point)

    _swap_asc(state, "analyze_window_target", fake_locator)
    return asc.locate_assist_panel(
        None, "", "", Image.new("RGB", image_size, (240, 240, 240))
    )


def test_locate_panel_returns_box_without_ocr():
    state = {}
    try:
        box = _locate_panel_with(state, point=_LOCATE_POINT)
    finally:
        _restore_asc(state)
    expected_w, expected_h = _expected_panel_size(_LOCATE_IMAGE_SIZE, _LOCATE_POINT)
    ok = (
        box is not None
        and box["right"] - box["left"] == expected_w
        and box["bottom"] - box["top"] == expected_h
    )
    print(f"[{'PASS' if ok else 'FAIL'}] locate_panel_returns_box_without_ocr: box={box}")
    return ok


def test_locate_panel_none_when_vlm_refuses():
    """VLM 이 패널을 못 찾으면([-1,-1] -> point None) 박스가 없다 - cap 대기."""
    state = {}
    try:
        box = _locate_panel_with(state, point=None)
    finally:
        _restore_asc(state)
    ok = box is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_panel_none_when_vlm_refuses")
    return ok


def main():
    print("[INFO] assist_score self-test 시작")
    results = [
        test_count_state_counts_black_ink_bands(),
        test_count_state_flags_red_and_excludes_it_from_rows(),
        test_count_state_does_not_count_header_as_a_row(),
        test_count_state_empty_table_is_zero_rows(),
        test_count_state_ignores_dense_dark_block(),
        test_locate_panel_returns_box_without_ocr(),
        test_locate_panel_none_when_vlm_refuses(),
        test_panel_target_uses_proven_button_geometry(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
