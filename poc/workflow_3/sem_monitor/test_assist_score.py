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
# 숫자를 그릴 x 위치. 판독기는 열을 구분하지 않으므로(한 행에 검정 숫자가 하나라도
# 있으면 측정 1회) 열 이름은 더 이상 의미가 없다 - 위치만 남긴다.
DIGIT_COLUMN_LEFTS = (20, 220)


def _digit_font():
    """PIL 기본 비트맵 폰트. size 인자를 지원하지 않는 구버전도 그대로 돈다."""
    try:
        return ImageFont.load_default(size=15)
    except TypeError:
        return ImageFont.load_default()


def _draw_rect_glyph(draw, left, top, *, width=12, height=18, fill=(20, 20, 20)):
    """숫자 한 글자 자리를 채우는 잉크 사각형을 그린다.

    현재 판독기는 잉크 밀도를 보지 않는다 - 픽셀이 어두운지, 띠가 얼마나 두꺼운지만
    본다. 그래서 예전의 링+바('8' 모양) 구성은 필요 없다. 폰트 metric 에 기대지 않고
    높이를 설계값으로 고정하는 것만이 이 헬퍼의 목적이다(띠 높이 판정이 그것에 의존).
    """
    draw.rectangle([left, top, left + width - 1, top + height - 1], fill=fill)


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


# locate_assist_panel 통합 테스트용 이미지 크기/점. 크기 자체에 의미는 없고, 패널
# 비율(PANEL_*_RATIO)을 적용한 crop 이 이미지 경계에 걸려 잘리지 않을 만큼 크면 된다.
_LOCATE_IMAGE_SIZE = (1000, 1000)
_LOCATE_POINT = {"x": 500, "y": 500}


def _expected_panel_size(image_size, point):
    """locate_assist_panel 의 panel_box 계산과 같은 식으로 패널 crop 크기를 구한다."""
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
        for left in DIGIT_COLUMN_LEFTS:
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


def test_count_state_drops_exactly_one_leading_band_as_header():
    """헤더 텍스트도 검정 잉크라 띠로 잡힌다 - 맨 위 띠 하나는 항상 버린다.

    같은 3행을 헤더 유/무로 그려 **차이**를 본다. 헤더가 있으면 4띠에서 1을 버려 3,
    없으면 3띠에서 1을 버려 2다. 헤더 있는 경우만 단독으로 확인하면 정상 계수
    테스트와 완전히 같은 단언이 되어(둘 다 rows==3) HEADER_BAND_COUNT 가 0 이 되어도
    한쪽은 통과한다 - 실제로 그렇게 중복돼 있었다.

    후자(2)는 버그가 아니라 문서화된 한계다: crop 이 헤더를 잘라내면 데이터 행 하나를
    잃지만, 과소 계수 -> done 지연은 안전한 방향이다.
    """
    with_header = asc.read_assist_state(_render_count_panel(["black"] * 3, header=True))
    without_header = asc.read_assist_state(_render_count_panel(["black"] * 3, header=False))
    ok = (
        with_header.ok_row_count == 3
        and without_header.ok_row_count == 2
        and with_header.ok_row_count - without_header.ok_row_count
        == asc.HEADER_BAND_COUNT
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] count_state_drops_exactly_one_leading_band_as_header: "
        f"header={with_header.ok_row_count} (기대 3), "
        f"no_header={without_header.ok_row_count} (기대 2)"
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
        test_count_state_drops_exactly_one_leading_band_as_header(),
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
