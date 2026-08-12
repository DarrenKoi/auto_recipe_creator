"""Assist Window score 판독 self-test (VLM/실이미지 불필요).

합성 이미지와 합성 OCR 항목만 쓰므로 Mac 에서 그대로 돈다.

    uv run python poc/workflow_3/sem_monitor/test_assist_score.py
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from poc.workflow_3.sem_monitor import assist_score as asc
from poc.workflow_3.sem_monitor.assist_score import (
    AssistLayout,
    RowState,
    assist_panel_target,
    build_score_grid,
    classify_ink,
    normalize_spotting_items_to_panel,
    ok_streak,
    read_row_states,
    row_verdict,
    save_assist_overlay,
)


def _cell(rgb=(240, 240, 240), *, ink=None, ink_px=40):
    """배경 40x20 셀에 잉크 픽셀을 ink_px 개 찍어 돌려준다."""
    cell = np.full((20, 40, 3), rgb, dtype=np.uint8)
    if ink is not None:
        flat = cell.reshape(-1, 3)
        flat[:ink_px] = ink
    return cell


def test_black_ink():
    ok = classify_ink(_cell(ink=(20, 20, 20))) == "black"
    print(f"[{'PASS' if ok else 'FAIL'}] black_ink")
    return ok


def test_red_ink():
    ok = classify_ink(_cell(ink=(200, 20, 20))) == "red"
    print(f"[{'PASS' if ok else 'FAIL'}] red_ink")
    return ok


def test_blank_cell():
    """잉크가 없으면 blank (측정 진행 중인 행)."""
    ok = classify_ink(_cell()) == "blank"
    print(f"[{'PASS' if ok else 'FAIL'}] blank_cell")
    return ok


def test_blank_when_ink_below_min_pixels():
    """안티에일리어싱 몇 픽셀은 잉크로 치지 않는다."""
    ok = classify_ink(_cell(ink=(20, 20, 20), ink_px=3)) == "blank"
    print(f"[{'PASS' if ok else 'FAIL'}] blank_when_ink_below_min_pixels")
    return ok


def test_mixed_ink_is_unknown():
    """빨강 비율이 흑도 적도 아닌 구간(0.10~0.30)이면 판정 불가 -> unknown.

    streak 을 끊어야 하는 상태다. 애매함이 done 판정으로 새면 엔지니어 작업 중에 창이 닫힌다.
    """
    cell = _cell(ink=(20, 20, 20), ink_px=40)
    flat = cell.reshape(-1, 3)
    flat[:8] = (200, 20, 20)   # 잉크 40px 중 8px 만 빨강 -> 비율 0.2
    ok = classify_ink(cell) == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] mixed_ink_is_unknown")
    return ok


def test_dense_dark_cell_is_unknown_not_black():
    """(C2(b)) 셀 대부분이 어두운 픽셀로 덮이면 - 예: 헤더 x 범위가 그레이스케일 SEM
    썸네일과 겹친 경우 - 숫자가 아니라 unknown 이어야 한다.

    수정 전에는 chroma/dominance 로만 흑/적을 갈랐으므로 무채색(그레이) 썸네일이
    black 으로 오판정돼 streak 이 허위로 쌓였다. 이 테스트는 그 실패 시나리오를
    재현한다: 셀 전체(20x40=800px)를 무채색 어두운 픽셀로 채우면(ink 비율 100%)
    black 이 아니라 unknown 이 나와야 streak 이 끊긴다.
    """
    dense_cell = _cell(ink=(30, 30, 30), ink_px=800)  # 셀 전체를 잉크로 채움(밀집)
    ok = classify_ink(dense_cell) == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] dense_dark_cell_is_unknown_not_black")
    return ok


def test_normal_digit_density_still_classifies():
    """밀집 가드가 정상 숫자(성긴 잉크)까지 unknown 으로 끊어버리면 안 된다."""
    ok = classify_ink(_cell(ink=(20, 20, 20), ink_px=40)) == "black"  # 40/800 = 5%
    print(f"[{'PASS' if ok else 'FAIL'}] normal_digit_density_still_classifies")
    return ok


def _cells(addr1="black", addr2="blank", meas="black"):
    return {"Addressing1": addr1, "Addressing2": addr2, "Measurement": meas}


def test_verdict_ok_without_addressing2():
    """Addressing2 는 대개 비어 있다. 없어도 정상 판정이어야 한다."""
    ok = row_verdict(_cells()) == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_ok_without_addressing2")
    return ok


def test_verdict_fail_on_red_measurement():
    ok = row_verdict(_cells(meas="red")) == "fail"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_fail_on_red_measurement")
    return ok


def test_verdict_fail_on_red_addressing1():
    """Addressing1 이 빨강이어도 그 측정은 실패다."""
    ok = row_verdict(_cells(addr1="red")) == "fail"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_fail_on_red_addressing1")
    return ok


def test_verdict_pending_when_measurement_blank():
    ok = row_verdict(_cells(meas="blank")) == "pending"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_pending_when_measurement_blank")
    return ok


def test_verdict_ok_when_only_measurement_present():
    """Addressing1 이 없는 레시피도 Measurement 로 완료를 판정한다.

    없는 칸을 '진행 중' 으로 읽으면 그 레시피는 영영 done 이 되지 않는다.
    """
    ok = row_verdict(_cells(addr1="blank")) == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_ok_when_only_measurement_present")
    return ok


def test_verdict_unknown_beats_ok():
    ok = row_verdict(_cells(meas="unknown")) == "unknown"
    print(f"[{'PASS' if ok else 'FAIL'}] verdict_unknown_beats_ok")
    return ok


def _rows(verdicts):
    """verdict 문자열 목록을 RowState 목록으로 (index 0 = 가장 오래된 행)."""
    mapping = {
        "ok": _cells(),
        "fail": _cells(meas="red"),
        "pending": _cells(meas="blank"),
        "unknown": _cells(meas="unknown"),
    }
    return [RowState(cells=dict(mapping[v])) for v in verdicts]


def test_streak_counts_from_newest():
    ok = ok_streak(_rows(["fail", "ok", "ok", "ok"])) == 3
    print(f"[{'PASS' if ok else 'FAIL'}] streak_counts_from_newest")
    return ok


def test_streak_skips_trailing_pending():
    """최신 행이 측정 진행 중(빈칸)이어도 그 앞의 연속 정상은 살아 있어야 한다."""
    ok = ok_streak(_rows(["ok", "ok", "ok", "pending"])) == 3
    print(f"[{'PASS' if ok else 'FAIL'}] streak_skips_trailing_pending")
    return ok


def test_streak_broken_by_fail_and_unknown():
    ok = (
        ok_streak(_rows(["ok", "ok", "fail", "ok"])) == 1
        and ok_streak(_rows(["ok", "ok", "unknown", "ok"])) == 1
    )
    print(f"[{'PASS' if ok else 'FAIL'}] streak_broken_by_fail_and_unknown")
    return ok


def test_streak_all_ok_is_full_length():
    ok = ok_streak(_rows(["ok"] * 7)) == 7
    print(f"[{'PASS' if ok else 'FAIL'}] streak_all_ok_is_full_length")
    return ok


def test_streak_empty_rows_is_zero():
    ok = ok_streak([]) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] streak_empty_rows_is_zero")
    return ok


def test_streak_top_anchor_when_newest_row_at_top():
    """(I9) ASSIST_NEWEST_ROW_AT="top" 이면 ok_streak 도 앞(index 0)에서부터 걸어야 한다.

    같은 rows 목록을 "bottom" 관례로 읽으면(뒤에서부터) fail 을 먼저 만나 streak=0
    이지만, "top" 관례로 읽으면(앞에서부터) 선행 pending 을 건너뛴 뒤 ok 3개를 센다 -
    둘 중 하나만 뒤집히면 이 결과가 정반대로 나온다.
    """
    rows = _rows(["pending", "ok", "ok", "ok", "fail"])
    state = {}
    try:
        _swap_asc(state, "ASSIST_NEWEST_ROW_AT", "top")
        streak = ok_streak(rows)
    finally:
        _restore_asc(state)
    ok = streak == 3
    print(f"[{'PASS' if ok else 'FAIL'}] streak_top_anchor_when_newest_row_at_top: {streak}")
    return ok


def _item(text, left, top, right, bottom):
    return {"text": text, "bbox": {"left": left, "top": top, "right": right, "bottom": bottom}}


def _panel_items():
    """헤더 3개 + 숫자 4행(부분만 채워진 표)을 흉내낸 OCR 결과.

    7행 슬롯의 top 은 40,70,...,220 (pitch 30). 표가 부분만 찼다면 채워진 행은 **아래쪽**
    이므로 최신 4행(top=130,160,190,220)에만 숫자를 둔다. 열: 10-60 / 110-160 / 210-260.
    """
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
    ]
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("12", 20, top, 50, top + 18))
        items.append(_item("34", 220, top, 250, top + 18))
    return items


def _measurement_only_items():
    items = [_item("Measuremen", 210, 5, 260, 25)]
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("34", 220, top, 250, top + 18))
        # 다른 패널 영역의 숫자는 Measurement 셀 범위에 영향을 주면 안 된다.
        items.append(_item("99", 20, top, 50, top + 18))
    return items


def test_grid_builds_without_addressing2():
    items = [item for item in _panel_items() if item["text"] != "Addressing2"]
    layout = build_score_grid(items, (300, 260))
    assert layout is not None
    assert layout.columns == ("Addressing1", "Measurement")


def test_grid_builds_with_measurement_only():
    layout = build_score_grid(_measurement_only_items(), (300, 260))
    assert layout is not None
    assert layout.columns == ("Measurement",)
    assert layout.grid[0][0]["left"] == 210
    assert layout.grid[0][0]["right"] == 260


def test_addressing2_score_does_not_change_active_grid_or_verdict():
    items = _panel_items()
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("77", 120, top, 150, top + 18))

    layout = build_score_grid(items, (300, 260))
    assert layout is not None
    assert layout.columns == ("Addressing1", "Measurement")
    assert layout.grid[0][0]["left"] == 10
    assert layout.grid[0][0]["right"] == 60
    assert layout.grid[0][1]["left"] == 210
    assert layout.grid[0][1]["right"] == 260

    image = _synth_panel([("black", "black")] * 7)
    draw = ImageDraw.Draw(image)
    draw.rectangle([120, 220, 150, 237], fill=(200, 20, 20))
    rows = read_row_states(image, layout)
    assert rows[-1].verdict == "ok"


def test_measurement_header_accepts_five_character_clip_only():
    assert asc._header_column_for("Measu") == "Measurement"
    assert asc._header_column_for("Meas") is None


def _top_heavy_items():
    """숫자가 표 맨 위 2행에만 있는 OCR 결과.

    7행으로 외삽하면 위쪽 행들의 좌표가 패널 밖(음수)으로 나간다. build_score_grid 는
    일부러 clamp 하지 않으므로, 그 음수 박스를 read_row_states 가 제대로 잘라내는지
    확인하는 데 쓴다.
    """
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
    ]
    for idx in range(2):
        top = 40 + idx * 30
        items.append(_item("12", 20, top, 50, top + 18))
        items.append(_item("34", 220, top, 250, top + 18))
    return items


def _synth_panel(row_specs):
    """행별 (addr1, meas) 색 지정으로 합성 패널 이미지를 만든다.

    row_specs 는 길이 7. 각 원소는 ("black"|"red"|None, "black"|"red"|None).
    None 은 빈칸(잉크 없음).
    """
    image = Image.new("RGB", (300, 260), (240, 240, 240))
    pixels = image.load()
    ink = {"black": (20, 20, 20), "red": (200, 20, 20)}
    for row_idx, (addr1, meas) in enumerate(row_specs):
        top = 40 + row_idx * 30
        for column_left, state in ((20, addr1), (220, meas)):
            if state is None:
                continue
            for dx in range(20):
                for dy in range(10):
                    pixels[column_left + dx, top + dy] = ink[state]
    return image


def _layout_for_synth():
    return build_score_grid(_panel_items(), (300, 260))


def _scale_bbox(bbox, sx, sy):
    return {
        "left": bbox["left"] * sx, "top": bbox["top"] * sy,
        "right": bbox["right"] * sx, "bottom": bbox["bottom"] * sy,
    }


def _items_in_norm1000(panel_size):
    """`_panel_items()` 를 0-1000 정규화 좌표로 표현한 OCR 응답을 흉내낸다.

    panel_size 픽셀 위치 p 에 대해 정규화 좌표는 p * 1000/dim 이다 - 실제 모델이
    crop 픽셀 대신 0-1000 좌표를 돌려주면 이렇게 나온다.
    """
    pw, ph = panel_size
    sx, sy = 1000.0 / pw, 1000.0 / ph
    return [
        {"text": item["text"], "bbox": _scale_bbox(item["bbox"], sx, sy)}
        for item in _panel_items()
    ]


def test_resolve_coord_space_detects_frac01():
    items = [_item("Addressing1", 0.05, 0.02, 0.2, 0.1)]
    sx, sy, space = asc._resolve_item_coord_space(items, (300, 260))
    ok = space == "frac01" and abs(sx - 300) < 1e-6 and abs(sy - 260) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] resolve_coord_space_detects_frac01")
    return ok


def test_resolve_coord_space_detects_crop_px():
    sx, sy, space = asc._resolve_item_coord_space(_panel_items(), (300, 260))
    ok = space == "crop_px" and sx == 1.0 and sy == 1.0
    print(f"[{'PASS' if ok else 'FAIL'}] resolve_coord_space_detects_crop_px")
    return ok


def test_resolve_coord_space_detects_norm1000():
    items = _items_in_norm1000((300, 260))
    sx, sy, space = asc._resolve_item_coord_space(items, (300, 260))
    ok = space == "norm1000" and abs(sx - 300 / 1000.0) < 1e-6 and abs(sy - 260 / 1000.0) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] resolve_coord_space_detects_norm1000")
    return ok


def test_normalize_scales_norm1000_items_back_into_panel_px():
    """(C1) 0-1000 좌표 항목이 정규화 후 원래 crop 픽셀 위치로 되돌아와야 한다.

    이게 없으면(수정 전) 0-1000 bbox 가 crop 픽셀인 것처럼 그대로 build_score_grid 에
    들어간다 - 텍스트 매칭은 좌표 무관이라 헤더는 여전히 잡히고 격자도 "성공"하지만,
    셀은 실제 숫자가 아니라 엉뚱한(대개 훨씬 작은 좌표 범위 밖 또는 다른 영역) 자리를
    가리킨다.
    """
    items_1000 = _items_in_norm1000((300, 260))
    normalized = normalize_spotting_items_to_panel(items_1000, (300, 260))
    header = next(it for it in normalized if it["text"] == "Addressing1")
    original = next(it for it in _panel_items() if it["text"] == "Addressing1")
    bb, ob = header["bbox"], original["bbox"]
    ok = (
        len(normalized) == len(items_1000)
        and abs(bb["left"] - ob["left"]) <= 1
        and abs(bb["right"] - ob["right"]) <= 1
        and abs(bb["top"] - ob["top"]) <= 1
        and abs(bb["bottom"] - ob["bottom"]) <= 1
    )
    print(f"[{'PASS' if ok else 'FAIL'}] normalize_scales_norm1000_items_back_into_panel_px: "
          f"{bb} vs {ob}")
    return ok


def test_normalize_drops_items_outside_panel_bounds():
    """패널 밖으로 매핑된 항목은 버려야 한다 - 좌표계 오판정/OCR 오검출 방지.

    Ghost 의 bbox(right=310, bottom=290)는 crop_px 판정 임계(<= max(300,260)*1.05=315)
    안에 들어 좌표계 판정 자체는 안 바뀌지만(panel(300,260)을 살짝 벗어남), 패널 경계를
    넘으므로 버려져야 한다.
    """
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Ghost", 295, 235, 310, 290),  # panel(300,260) 살짝 밖
    ]
    normalized = normalize_spotting_items_to_panel(items, (300, 260))
    ok = len(normalized) == 1 and normalized[0]["text"] == "Addressing1"
    print(f"[{'PASS' if ok else 'FAIL'}] normalize_drops_items_outside_panel_bounds")
    return ok


def test_normalize_empty_items_returns_empty():
    ok = normalize_spotting_items_to_panel([], (300, 260)) == []
    print(f"[{'PASS' if ok else 'FAIL'}] normalize_empty_items_returns_empty")
    return ok


def test_build_score_grid_with_norm1000_items_lands_on_real_cells():
    """(C1) 실패 시나리오 재현: 0-1000 좌표 OCR 응답이 정규화 없이 들어가면 셀이
    회색조 썸네일 위에 놓일 수 있다. locate_assist_layout 과 동일하게 정규화를 거친
    뒤 build_score_grid 에 넣으면 _panel_items() 와 (반올림 오차 이내로) 같은 격자가
    나와야 한다.
    """
    items_1000 = _items_in_norm1000((300, 260))
    normalized = normalize_spotting_items_to_panel(items_1000, (300, 260))
    layout = build_score_grid(normalized, (300, 260))
    baseline = build_score_grid(_panel_items(), (300, 260))
    if layout is None or baseline is None:
        print("[FAIL] build_score_grid_with_norm1000_items_lands_on_real_cells: layout None")
        return False
    ok = all(
        abs(layout.grid[r][c][k] - baseline.grid[r][c][k]) <= 1
        for r in range(len(layout.grid))
        for c in range(len(layout.grid[r]))
        for k in ("left", "right", "top", "bottom")
    )
    print(f"[{'PASS' if ok else 'FAIL'}] build_score_grid_with_norm1000_items_lands_on_real_cells")
    return ok


def test_grid_top_anchor_when_newest_row_at_top():
    """(I9) ASSIST_NEWEST_ROW_AT="top" 이면 build_score_grid 도 anchoring 방향을 뒤집어야
    한다 - 최신 데이터가 있는 맨 위 띠를 index 0 에 맞추고 아래로(빈 슬롯 방향) 채운다.
    """
    state = {}
    try:
        _swap_asc(state, "ASSIST_NEWEST_ROW_AT", "top")
        layout = build_score_grid(_top_heavy_items(), (300, 260))
    finally:
        _restore_asc(state)
    if layout is None:
        print("[FAIL] grid_top_anchor_when_newest_row_at_top: layout None")
        return False
    tops = [row[0]["top"] for row in layout.grid]
    # 맨 위(index 0)가 실제 숫자 띠(top=40 부근)에 고정되고, 아래로 pitch(30)씩 증가해야
    # 한다 - bottom 기본값이면 index 0 은 음수(외삽된 빈 슬롯)가 된다.
    ok = tops[0] >= 0 and all(tops[i + 1] - tops[i] == 30 for i in range(len(tops) - 1))
    print(f"[{'PASS' if ok else 'FAIL'}] grid_top_anchor_when_newest_row_at_top: {tops}")
    return ok


def test_read_rows_marks_black_and_red():
    specs = [("black", "black")] * 6 + [("black", "red")]
    rows = read_row_states(_synth_panel(specs), _layout_for_synth())
    ok = len(rows) == 7 and rows[-1].verdict == "fail" and rows[0].verdict == "ok"
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_marks_black_and_red: "
          f"{[r.verdict for r in rows]}")
    return ok


def test_read_rows_blank_is_pending():
    specs = [("black", "black")] * 6 + [("black", None)]
    rows = read_row_states(_synth_panel(specs), _layout_for_synth())
    ok = rows[-1].verdict == "pending" and ok_streak(rows) == 6
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_blank_is_pending: streak={ok_streak(rows)}")
    return ok


def test_read_rows_returns_empty_without_layout():
    ok = read_row_states(_synth_panel([("black", "black")] * 7), None) == []
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_returns_empty_without_layout")
    return ok


def test_read_rows_clamps_boxes_outside_panel():
    """패널 밖으로 나간 행은 빈칸(pending)이 되어야 한다 - 다른 영역을 읽으면 안 된다.

    numpy 는 음수 인덱스를 뒤에서부터 세므로, clamp 를 빼면 예외 없이 **이미지 아래쪽**을
    조용히 샘플링한다. 거기에 잉크가 있으면 없는 측정을 정상으로 읽는다.
    """
    layout = build_score_grid(_top_heavy_items(), (300, 260))
    if layout is None:
        print("[FAIL] read_rows_clamps_boxes_outside_panel: layout None")
        return False
    if layout.grid[0][0]["top"] >= 0:
        print("[FAIL] read_rows_clamps_boxes_outside_panel: 픽스처가 음수 좌표를 못 만듦")
        return False
    rows = read_row_states(_synth_panel([("black", "black")] * 7), layout)
    ok = (
        len(rows) == 7
        and rows[0].verdict == "pending"   # 패널 밖 -> 빈칸
        and rows[6].verdict == "ok"        # 패널 안 행은 그대로 읽힌다
    )
    print(f"[{'PASS' if ok else 'FAIL'}] read_rows_clamps_boxes_outside_panel: "
          f"{[r.verdict for r in rows]}")
    return ok


def test_grid_has_full_rows_and_columns():
    layout = build_score_grid(_panel_items(), (300, 260))
    ok = (
        layout is not None
        and len(layout.grid) == 7
        and all(len(row) == 2 for row in layout.grid)
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_has_full_rows_and_columns")
    return ok


def test_grid_extrapolates_missing_rows_by_pitch():
    """표가 부분만 차 있어도 pitch 로 7행을 채운다(행 간격 30px)."""
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_extrapolates_missing_rows_by_pitch: layout None")
        return False
    tops = [row[0]["top"] for row in layout.grid]
    diffs = {tops[i + 1] - tops[i] for i in range(len(tops) - 1)}
    ok = diffs == {30}
    print(f"[{'PASS' if ok else 'FAIL'}] grid_extrapolates_missing_rows_by_pitch: {sorted(diffs)}")
    return ok


def test_grid_columns_follow_headers():
    """열 판별은 헤더 텍스트로 하지만, x 범위는 그 열에 배정된 숫자의 합집합을 쓴다.

    _panel_items() 의 숫자는 Addressing1 열에 20-50, Measurement 열에 220-250 이다
    (각각 헤더 범위 10-60 / 210-260 안에 들어간다). Addressing2 는 인식되더라도
    score grid 의 활성 열에는 포함하지 않는다.

    (F1) 어느 쪽이든 좌우로 폭의 CELL_PAD_X_RATIO(0.35)만큼 넓혀야 한다 - 글자가 셀을
    꽉 채우면 classify_ink 의 밀집 가드가 정상 숫자를 썸네일로 오인한다. 그래서 기대값은
    20-50 -> 10-60, 220-250 -> 210-260 이다.
    """
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_columns_follow_headers: layout None")
        return False
    first = layout.grid[0]
    ok = (
        layout.columns == ("Addressing1", "Measurement")
        and first[0]["left"] == 10 and first[0]["right"] == 60    # 숫자 합집합 20-50 + pad
        and first[1]["left"] == 210 and first[1]["right"] == 260  # 숫자 합집합 220-250 + pad
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_columns_follow_headers")
    return ok


def test_grid_columns_fallback_to_header_when_no_numbers():
    """세 헤더가 모두 인식되어도 Addressing2 는 score grid 에서 제외한다.

    Addressing2 는 대개 비어 있으므로 active columns 에 포함하지 않는다. Measurement 는
    여전히 필수 열이자 authoritative score 열이다.
    """
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_columns_fallback_to_header_when_no_numbers: layout None")
        return False
    ok = (
        "Addressing2" in asc._match_header_boxes(_panel_items())
        and layout.columns == ("Addressing1", "Measurement")
        and len(layout.grid[0]) == 2
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_columns_fallback_to_header_when_no_numbers")
    return ok


def test_grid_columns_ignore_header_order_in_items():
    """헤더의 x 위치가 뒤섞여도 활성 열은 텍스트로 잡아야 한다.

    여기서는 Measurement 헤더를 Addressing2 보다 왼쪽에 둔다. 헤더를 x 로 정렬해
    ASSIST_SCORE_COLUMNS 에 순서대로 배정하는 구현이면 Measurement 열을 잘못 잡아
    이 테스트가 깨진다. 실제 tool 에서 Addressing2 는 대개 비어 있어 위치 추정이
    Measurement 를 잘못 고를 수 있다 - 그걸 막는 게 텍스트 매칭의 존재 이유다.
    """
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Measurement", 110, 5, 160, 25),
        _item("Addressing2", 210, 5, 260, 25),
    ]
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("12", 20, top, 50, top + 18))
        items.append(_item("34", 220, top, 250, top + 18))

    layout = build_score_grid(items, (300, 260))
    if layout is None:
        print("[FAIL] grid_columns_ignore_header_order_in_items: layout None")
        return False
    first = layout.grid[0]
    # grid 열 순서는 활성 열 고정: [Addressing1, Measurement].
    # 헤더 순서가 뒤섞여도(Measurement 가 Addressing2 보다 왼쪽) 숫자는 자신이 겹치는
    # 헤더 x 범위로 배정된다: "12"(20-50)는 Addressing1(10-60) 아래, "34"(220-250)는
    # Addressing2(210-260) 아래이므로 무시된다. Measurement(110-160)는 숫자가 없어
    # 헤더 폴백을 쓴다. (F1) 각 범위에 좌우 패딩(폭의 0.35)이 붙는다.
    ok = (
        layout.columns == ("Addressing1", "Measurement")
        and first[0]["left"] == 10 and first[0]["right"] == 60
        and first[1]["left"] == 92 and first[1]["right"] == 178
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_columns_ignore_header_order_in_items")
    return ok


DIGIT_PANEL_SIZE = (300, 280)
DIGIT_ROW_PITCH = 30
DIGIT_FIRST_TOP = 45
DIGIT_COLUMN_LEFTS = {"Addressing1": 20, "Measurement": 220}


def _digit_headers():
    """렌더 패널용 헤더 항목 3개(숫자 띠보다 위에 있어야 한다)."""
    return [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
    ]


def _ink_mask(rgb):
    """classify_ink 와 같은 기준(채널 평균 < INK_MEAN_MAX)으로 잉크 마스크를 만든다."""
    arr = np.array(rgb).astype(np.int16)
    return arr[:, :, :3].mean(axis=2) < asc.INK_MEAN_MAX


def _tight_bbox(image, region):
    """region(left, top, right, bottom) 안에서 실제 잉크의 tight bbox 를 찾는다."""
    left, top, right, bottom = region
    mask = _ink_mask(image.crop((left, top, right, bottom)))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (
        left + int(xs.min()), top + int(ys.min()),
        left + int(xs.max()) + 1, top + int(ys.max()) + 1,
    )


def _cell_ink_ratio(image, box):
    """격자 셀 하나의 잉크 비율(= classify_ink 의 밀집 가드가 보는 값)."""
    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    mask = _ink_mask(crop)
    return float(mask.sum()) / float(mask.size)


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


def _render_digit_panel(*, use_font, rows=7, ink=(20, 20, 20)):
    """실제 글자(또는 글자 모양)를 그린 합성 Assist 패널과 그 OCR 항목을 만든다.

    OCR 항목의 bbox 는 **실제로 그려진 잉크의 tight bbox** 다 - PaddleOCR 이 돌려주는
    것과 같은 성격의 좌표라, build_score_grid 의 패딩과 classify_ink 의 밀집 가드가
    실제 배선 그대로 상호작용한다.
    """
    image = Image.new("RGB", DIGIT_PANEL_SIZE, (240, 240, 240))
    draw = ImageDraw.Draw(image)
    font = _digit_font() if use_font else None

    items = _digit_headers()
    tight_ratios = []
    for row_idx in range(rows):
        top = DIGIT_FIRST_TOP + row_idx * DIGIT_ROW_PITCH
        for column, left in DIGIT_COLUMN_LEFTS.items():
            if use_font:
                draw.text((left, top), "18", fill=ink, font=font)
            else:
                # 두 자리 숫자처럼 붙여 그린다(합쳐진 tight bbox 가 OCR bbox 에 해당).
                _draw_rect_glyph(draw, left, top, fill=ink)
                _draw_rect_glyph(draw, left + 12, top, fill=ink)
            box = _tight_bbox(image, (left - 4, top - 6, left + 44, top + DIGIT_ROW_PITCH - 4))
            if box is None:
                continue
            items.append(_item("18", box[0], box[1], box[2], box[3]))
            mask = _ink_mask(image.crop(box))
            tight_ratios.append(float(mask.sum()) / float(mask.size))
    return image, items, tight_ratios


def _digit_panel_streak(use_font):
    """렌더 패널을 실제 build_score_grid + read_row_states 로 통과시킨다."""
    image, items, tight_ratios = _render_digit_panel(use_font=use_font)
    layout = build_score_grid(items, DIGIT_PANEL_SIZE)
    if layout is None:
        return None, None, None, tight_ratios, image
    rows = read_row_states(image, layout)
    ratios = [_cell_ink_ratio(image, layout.grid[r][c]) for r in range(len(layout.grid))
              for c in (0, 1)]
    return layout, rows, ratios, tight_ratios, image


def test_rendered_digits_read_ok_end_to_end():
    """(F3) 진짜 글자를 그린 패널을 실제 격자 생성 + 판독 경로로 통과시킨다.

    기존 밀도 테스트는 800px 셀에 잉크 40px(5%) 짜리 합성 블록만 써서 경계 근처를 전혀
    건드리지 않았다 - 그래서 "셀을 글자 bbox 에 딱 맞춤(F1 이전)" + "밀집 임계 0.55(F2
    이전)" 의 충돌(정상 숫자가 전부 unknown 이 되어 streak 이 영영 0)을 못 잡았다.
    이 테스트는 패딩과 밀집 가드를 **함께** 지나가므로 그 조합이 깨지면 바로 빨개진다.
    """
    layout, rows, ratios, tight_ratios, _img = _digit_panel_streak(use_font=True)
    if layout is None:
        print("[FAIL] rendered_digits_read_ok_end_to_end: layout None")
        return False
    verdicts = [r.verdict for r in rows]
    streak = ok_streak(rows)
    max_ratio = max(ratios)
    ok = (
        verdicts == ["ok"] * 7
        and streak == 7
        and max_ratio < asc.INK_DENSE_MAX_RATIO
    )
    print(f"[{'PASS' if ok else 'FAIL'}] rendered_digits_read_ok_end_to_end: streak={streak} "
          f"tight_ink={min(tight_ratios):.3f}~{max(tight_ratios):.3f} "
          f"cell_ink_max={max_ratio:.3f} (< {asc.INK_DENSE_MAX_RATIO}) {verdicts}")
    return ok


def test_realistic_glyph_density_survives_density_guard():
    """(F3) 실제 숫자와 같은 밀도(tight bbox 대비 0.4~0.7)의 글자 모양도 ok 로 읽혀야 한다.

    PIL 폰트 metric 에 의존하지 않는 결정적 버전이다 - 글자의 tight bbox 밀도를 여기서
    직접 재서 "현실적인 밀도를 시험하고 있음" 자체를 단언한다(5% 블록이면 이 단언이
    먼저 깨진다).
    """
    layout, rows, ratios, tight_ratios, _img = _digit_panel_streak(use_font=False)
    if layout is None:
        print("[FAIL] realistic_glyph_density_survives_density_guard: layout None")
        return False
    streak = ok_streak(rows)
    tight = max(tight_ratios)
    ok = (
        0.40 <= min(tight_ratios) and tight <= 0.70          # 진짜 숫자다운 밀도인가
        and [r.verdict for r in rows] == ["ok"] * 7
        and streak == 7
        and max(ratios) < asc.INK_DENSE_MAX_RATIO
    )
    print(f"[{'PASS' if ok else 'FAIL'}] realistic_glyph_density_survives_density_guard: "
          f"streak={streak} tight_ink={tight:.3f} cell_ink_max={max(ratios):.3f}")
    return ok


def test_thumbnail_dense_cells_read_unknown_end_to_end():
    """(F3) 임계를 0.85 로 올려도 썸네일(셀의 95%+ 가 어두움)은 여전히 unknown 이어야 한다.

    가드가 변별력을 잃지 않았다는 증거다 - 여기서 black 이 나오면 없는 측정을 정상으로
    읽어 streak 이 허위로 쌓이고, 엔지니어 작업 중에 tool 창이 닫힌다.
    """
    image, items, _tight = _render_digit_panel(use_font=False)
    layout = build_score_grid(items, DIGIT_PANEL_SIZE)
    if layout is None:
        print("[FAIL] thumbnail_dense_cells_read_unknown_end_to_end: layout None")
        return False
    draw = ImageDraw.Draw(image)
    for row_boxes in layout.grid:
        for col_idx in (0, 1):
            box = row_boxes[col_idx]
            draw.rectangle([box["left"], box["top"], box["right"] - 1, box["bottom"] - 1],
                           fill=(60, 60, 60))
            # 셀의 아주 일부만 배경으로 남겨 밀도를 1.0 이 아닌 0.95 안팎으로 만든다.
            draw.rectangle([box["left"], box["top"], box["right"] - 1, box["top"]],
                           fill=(240, 240, 240))
    rows = read_row_states(image, layout)
    ratios = [_cell_ink_ratio(image, layout.grid[r][c]) for r in range(7) for c in (0, 1)]
    streak = ok_streak(rows)
    ok = (
        all(r.verdict == "unknown" for r in rows)
        and streak == 0
        and 0.90 <= min(ratios) <= 1.0
    )
    print(f"[{'PASS' if ok else 'FAIL'}] thumbnail_dense_cells_read_unknown_end_to_end: "
          f"streak={streak} cell_ink={min(ratios):.3f}~{max(ratios):.3f}")
    return ok


def test_padded_cell_height_stays_within_row_pitch():
    """(F1) 세로 패딩이 이웃 행을 물면 안 된다.

    기존 픽스처는 pitch 30 / 글자 높이 18 이라, 패딩을 무작정 35%씩 더하면 30.6 이 되어
    위아래 행의 숫자를 함께 담는다. pitch 안으로 잘려 눈에 보이는 틈이 남아야 한다.
    """
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] padded_cell_height_stays_within_row_pitch: layout None")
        return False
    heights = {row[0]["bottom"] - row[0]["top"] for row in layout.grid}
    tops = [row[0]["top"] for row in layout.grid]
    pitch = tops[1] - tops[0]
    height = max(heights)
    ok = len(heights) == 1 and height < pitch and height > 18  # 원래 글자 높이보단 커야
    print(f"[{'PASS' if ok else 'FAIL'}] padded_cell_height_stays_within_row_pitch: "
          f"height={height} pitch={pitch}")
    return ok


def test_resolve_coord_space_norm1000_on_large_panel():
    """(F4) 패널 crop 이 1000px 을 넘어도 0-1000 좌표를 crop 픽셀로 오판정하면 안 된다.

    큰 모니터(2560 폭)에서 패널 crop 은 대략 1126x518 이라 크기 의존 임계는 1182 가
    된다 - 진짜 0-1000 좌표가 그 아래라 crop 픽셀로 읽히고, 셀이 썸네일 위에 놓여 모든
    행이 black 으로 읽히는(streak 이 7 에 고정되는) 원래 버그가 되살아난다.
    """
    panel = (1126, 518)
    items = [
        _item("Addressing1", 20, 40, 200, 90),
        _item("Addressing2", 400, 40, 580, 90),
        _item("Measurement", 780, 40, 1000, 90),
        _item("18", 60, 200, 140, 260),
    ]
    sx, sy, space = asc._resolve_item_coord_space(items, panel)
    ok = (
        space == "norm1000"
        and abs(sx - 1126 / 1000.0) < 1e-6
        and abs(sy - 518 / 1000.0) < 1e-6
    )
    print(f"[{'PASS' if ok else 'FAIL'}] resolve_coord_space_norm1000_on_large_panel: {space}")
    return ok


def test_grid_none_without_headers():
    """헤더를 못 읽으면 어느 열이 무엇인지 알 수 없으므로 격자를 만들지 않는다."""
    items = [_item("12", 20, 40, 50, 58), _item("34", 220, 40, 250, 58)]
    ok = build_score_grid(items, (300, 260)) is None
    print(f"[{'PASS' if ok else 'FAIL'}] grid_none_without_headers")
    return ok


def test_grid_none_with_single_number_row():
    """행이 하나면 pitch 를 알 수 없다. 추정하지 않고 실패시킨다."""
    items = [
        _item("Addressing1", 10, 5, 60, 25),
        _item("Addressing2", 110, 5, 160, 25),
        _item("Measurement", 210, 5, 260, 25),
        _item("12", 20, 40, 50, 58),
    ]
    ok = build_score_grid(items, (300, 260)) is None
    print(f"[{'PASS' if ok else 'FAIL'}] grid_none_with_single_number_row")
    return ok


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


def _stub_ocr_client(items):
    """parse_spotting_items 가 items 를 돌려주도록 VLM 왕복을 통째로 대체한다."""
    class _Client:
        def __init__(self, *a, **k):
            pass

        def chat_with_image_b64(self, **kwargs):
            return SimpleNamespace(text="stub")

    return _Client


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


def _locate_with(state, *, point, items=None, image_size=_LOCATE_IMAGE_SIZE,
                  ocr_raises=False, locator_raises=False, debug_dir=None):
    """스텁을 걸고 locate_assist_layout 을 1회 호출한다."""
    def fake_locator(*a, **k):
        if locator_raises:
            raise RuntimeError("locator boom")
        return SimpleNamespace(point=point)

    def fake_parse(_raw):
        if ocr_raises:
            raise RuntimeError("ocr boom")
        return items if items is not None else _panel_items()

    _swap_asc(state, "analyze_window_target", fake_locator)
    _swap_asc(state, "Workflow1VLMClient", _stub_ocr_client(items))
    _swap_asc(state, "parse_spotting_items", fake_parse)
    return asc.locate_assist_layout(
        None, "", "", Image.new("RGB", image_size, (240, 240, 240)), debug_dir=debug_dir
    )


def test_locate_returns_layout_on_happy_path():
    state = {}
    try:
        result = _locate_with(state, point=_LOCATE_POINT)
    finally:
        _restore_asc(state)
    ok = result is not None and len(result[1].grid) == 7
    print(f"[{'PASS' if ok else 'FAIL'}] locate_returns_layout_on_happy_path")
    return ok


def test_locate_with_norm1000_items_still_produces_grid():
    """(C1) 실제 배선 경로(locate_assist_layout) 를 통째로 거쳐도, OCR 이 0-1000
    정규화 좌표를 돌려주면 여전히 정상 격자가 나와야 한다(수정 전에는 좌표계를 안
    맞춰 잘못된 자리를 가리키는 격자를 - 혹은 항목이 패널 밖으로 튀어 아예 격자를 -
    만들었을 수 있다).
    """
    panel_size = _expected_panel_size(_LOCATE_IMAGE_SIZE, _LOCATE_POINT)
    items_1000 = _items_in_norm1000(panel_size)
    state = {}
    try:
        result = _locate_with(state, point=_LOCATE_POINT, items=items_1000)
    finally:
        _restore_asc(state)
    baseline = build_score_grid(_panel_items(), panel_size)
    ok = (
        result is not None
        and baseline is not None
        and len(result[1].grid) == 7
        and all(
            abs(result[1].grid[r][c][k] - baseline.grid[r][c][k]) <= 1
            for r in range(7)
            for c in range(len(baseline.grid[r]))
            for k in ("left", "right", "top", "bottom")
        )
    )
    print(f"[{'PASS' if ok else 'FAIL'}] locate_with_norm1000_items_still_produces_grid")
    return ok


def test_locate_none_when_too_few_items_survive_normalization():
    """(C1) 정규화 후(패널 밖 제거) 항목이 부족하면 확신 없는 격자를 만들지 않고 None."""
    # 패널(대략 440x360) 훨씬 밖으로 나가는 좌표 - crop_px 로 해석되면 전부 버려진다.
    far_outside = [
        _item("Addressing1", 5000, 5000, 5060, 5020),
        _item("Addressing2", 5100, 5000, 5160, 5020),
        _item("Measurement", 5200, 5000, 5260, 5020),
        _item("12", 5000, 5100, 5030, 5118),
    ]
    state = {}
    try:
        result = _locate_with(state, point=_LOCATE_POINT, items=far_outside)
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_too_few_items_survive_normalization")
    return ok


def test_locate_none_when_point_outside_image():
    """grounding 점이 창 밖이면 crop 박스가 뒤집힌다. 예외 없이 None 이어야 한다.

    analyze_window_target 은 crop 오프셋을 더할 때 전체 이미지 경계로 다시 clamp 하지
    않으므로 이런 점이 실제로 나올 수 있다. 여기서 예외가 나면 폴링 루프가 죽는다.
    """
    state = {}
    try:
        result = _locate_with(state, point={"x": -400, "y": 130})
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_point_outside_image")
    return ok


def test_locate_none_when_locator_raises():
    state = {}
    try:
        result = _locate_with(state, point={"x": 150, "y": 130}, locator_raises=True)
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_locator_raises")
    return ok


def test_locate_none_when_no_point():
    state = {}
    try:
        result = _locate_with(state, point=None)
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_no_point")
    return ok


def test_locate_none_when_ocr_raises():
    state = {}
    try:
        result = _locate_with(state, point={"x": 150, "y": 130}, ocr_raises=True)
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_ocr_raises")
    return ok


def test_locate_none_when_grid_cannot_be_built():
    """헤더가 없어 격자를 못 만들면 None. 나쁜 격자를 캐시하는 것이 최악이다."""
    state = {}
    try:
        result = _locate_with(state, point={"x": 150, "y": 130},
                              items=[_item("12", 20, 40, 50, 58)])
    finally:
        _restore_asc(state)
    ok = result is None
    print(f"[{'PASS' if ok else 'FAIL'}] locate_none_when_grid_cannot_be_built")
    return ok


def test_locate_failure_leaves_evidence_on_disk():
    """격자 확보 실패 시 crop 이미지 + OCR 항목 덤프가 남아야 한다.

    오피스는 콘솔 한 줄과 디스크 산출물로만 디버깅한다(2026-08-12 실측: 실패했는데
    폴더가 비어 있어 원인을 좁힐 수 없었다). 조용한 실패로 되돌아가는 회귀를 막는다.
    """
    state = {}
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        try:
            # 항목 수는 충분하지만(too_few_items 를 통과) 헤더가 없어 격자 단계에서 실패.
            numbers = [_item("12", 20, 40 + i * 30, 50, 58 + i * 30) for i in range(5)]
            result = _locate_with(
                state, point={"x": 150, "y": 130}, items=numbers, debug_dir=debug_dir,
            )
        finally:
            _restore_asc(state)
        jpgs = list(debug_dir.glob("assist_ocr_read_fail_*.jpg"))
        jsons = list(debug_dir.glob("assist_ocr_read_fail_*.json"))
        payload = json.loads(jsons[0].read_text(encoding="utf-8")) if jsons else {}
        ok = (
            result is None
            and len(jpgs) == 1 and jpgs[0].stat().st_size > 0
            and len(jsons) == 1
            and payload.get("reason") == "grid_build"
            and payload.get("item_count") == len(numbers)
            # OCR 이 실제로 읽은 텍스트가 남아야 원인 규명이 된다.
            and payload.get("items", [{}])[0].get("text") == "12"
        )
    print(
        f"[{'PASS' if ok else 'FAIL'}] locate_failure_leaves_evidence: "
        f"jpg={len(jpgs)} json={len(jsons)} reason={payload.get('reason')}"
    )
    return ok


def test_score_text_accepts_decimal_and_signed():
    """소수점/부호 score 를 숫자로 인정해야 한다.

    구 구현(all(isdigit))은 "0.85" 를 버려 숫자 항목이 0개가 됐고, 그 결과 세 열의
    score 를 하나도 못 읽었다(2026-08-12 오피스). 값은 쓰지 않고 위치만 쓰므로
    관대해도 위험이 없다.
    """
    accepted = ["12", "0.85", "-3.2", "+7", "1,234", " 42 "]
    rejected = ["", ".", "-", "Addressing1", "1.2.3", "N/A"]
    ok = all(asc._is_score_text(t) for t in accepted) and not any(
        asc._is_score_text(t) for t in rejected
    )
    bad = [t for t in accepted if not asc._is_score_text(t)] + [
        t for t in rejected if asc._is_score_text(t)
    ]
    print(f"[{'PASS' if ok else 'FAIL'}] score_text_accepts_decimal_and_signed: 오분류={bad}")
    return ok


def test_header_match_survives_ocr_variants():
    """공백/구두점/잘림 변형도 헤더로 인식해야 한다 (완전 일치는 실전에서 무너진다)."""
    cases = {
        "Addressing1": "Addressing1", "Addressing 1": "Addressing1",
        "Measurement:": "Measurement", "Measuremen": "Measurement",
        "Addressing2": "Addressing2",
    }
    ok = all(asc._header_column_for(text) == column for text, column in cases.items())
    # 무관한 텍스트를 헤더로 오인하면 격자가 엉뚱한 x 범위로 선다.
    ok = ok and asc._header_column_for("Point") is None and asc._header_column_for("12") is None
    got = {t: asc._header_column_for(t) for t in cases}
    print(f"[{'PASS' if ok else 'FAIL'}] header_match_survives_ocr_variants: {got}")
    return ok


def test_split_addressing_headers_assigned_by_x_order():
    """OCR 이 "Addressing"+"1" 로 쪼개 읽으면 x 순서로 1/2 를 배정한다.

    쪼개진 "Addressing" 은 그 자체로 1인지 2인지 알 수 없다. 표의 열 순서가 고정이므로
    왼쪽이 Addressing1 이다. 이 배정이 없으면 헤더 3개를 못 채워 격자가 통째로 실패한다.
    """
    items = [
        _item("Addressing", 110, 5, 155, 25),   # 오른쪽 -> Addressing2
        _item("Addressing", 10, 5, 55, 25),     # 왼쪽   -> Addressing1
        _item("Measurement", 210, 5, 260, 25),
    ]
    boxes = asc._match_header_boxes(items)
    ok = (
        set(boxes) == set(asc.ASSIST_COLUMNS)
        and boxes["Addressing1"]["left"] == 10
        and boxes["Addressing2"]["left"] == 110
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] split_addressing_headers_by_x: "
        f"{ {k: v.get('left') for k, v in boxes.items()} }"
    )
    return ok


def test_grid_builds_with_decimal_scores_and_split_headers():
    """실전형 OCR 출력(쪼개진 헤더 + 소수점 score)으로도 격자가 서야 한다.

    두 수정이 함께 동작해야 의미가 있다 - 하나만 고치면 여전히 격자가 안 선다.
    """
    items = [
        _item("Addressing", 10, 5, 55, 25),
        _item("Addressing", 110, 5, 155, 25),
        _item("Measurement", 210, 5, 260, 25),
    ]
    for idx in range(4):
        top = 130 + idx * 30
        items.append(_item("0.85", 20, top, 50, top + 18))
        items.append(_item("-1.20", 220, top, 250, top + 18))
    layout = build_score_grid(items, (300, 260))
    ok = layout is not None and len(layout.grid) == asc.ASSIST_ROWS
    print(
        f"[{'PASS' if ok else 'FAIL'}] grid_with_decimal_and_split_headers: "
        f"rows={len(layout.grid) if layout else None}"
    )
    return ok


def test_locate_success_also_saves_ocr_overlay():
    """성공해도 OCR 판독 오버레이가 남아야 한다.

    헤더 3개만 맞으면 격자는 서므로, 숫자를 엉뚱하게 읽어도 '성공' 으로 보인다.
    판독 품질은 성공 여부와 별개로 눈으로 검증해야 한다는 요구(2026-08-12).
    role 이 함께 기록되어 header/score 해석까지 JSON 으로 재구성 가능해야 한다.
    """
    state = {}
    with tempfile.TemporaryDirectory() as tmp:
        debug_dir = Path(tmp)
        try:
            result = _locate_with(state, point=_LOCATE_POINT, debug_dir=debug_dir)
        finally:
            _restore_asc(state)
        jpgs = list(debug_dir.glob("assist_ocr_read_ok.jpg"))
        jsons = list(debug_dir.glob("assist_ocr_read_ok.json"))
        payload = json.loads(jsons[0].read_text(encoding="utf-8")) if jsons else {}
        roles = {item.get("role") for item in payload.get("items", [])}
        ok = (
            result is not None                      # 성공 경로임을 확인.
            and len(jpgs) == 1 and jpgs[0].stat().st_size > 0
            and payload.get("reason") == "ok"
            and "header" in roles and "score" in roles
        )
    print(
        f"[{'PASS' if ok else 'FAIL'}] locate_success_saves_ocr_overlay: "
        f"jpg={len(jpgs)} roles={sorted(roles)}"
    )
    return ok


def test_ocr_overlay_marks_roles_distinctly():
    """오버레이가 header/score/other 를 서로 다른 색으로 그린다 (한눈에 구분 가능).

    색이 같으면 '무엇으로 해석됐는지' 를 그림에서 읽을 수 없어 오버레이의 목적이 사라진다.
    """
    colors = {asc._ocr_item_role(t) for t in ("Addressing1", "123", "Point No.")}
    distinct = len({asc._OCR_ROLE_COLORS[r] for r in colors}) == len(colors)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "ocr.jpg"
        asc.save_ocr_items_overlay(
            Image.new("RGB", (200, 100), (240, 240, 240)),
            [_item("Addressing1", 5, 5, 60, 20), _item("12", 5, 40, 30, 55)],
            out,
        )
        written = out.exists() and out.stat().st_size > 0
    ok = distinct and written and colors == {"header", "score", "other"}
    print(f"[{'PASS' if ok else 'FAIL'}] ocr_overlay_marks_roles_distinctly: roles={sorted(colors)}")
    return ok


def test_locate_failure_evidence_survives_missing_debug_dir():
    """debug_dir 미지정이어도 증거는 기본 폴더로 떨어진다(호출부가 안 넘겨도 잃지 않는다)."""
    state = {}
    with tempfile.TemporaryDirectory() as tmp:
        fallback = Path(tmp) / "assist_score"
        _swap_asc(state, "DEBUG_ARTIFACT_DIR", fallback)
        try:
            result = _locate_with(
                state, point={"x": 150, "y": 130},
                items=[_item("12", 20, 40, 50, 58)],
            )
        finally:
            _restore_asc(state)
        ok = result is None and len(list(fallback.glob("assist_ocr_read_fail_*.json"))) == 1
    print(f"[{'PASS' if ok else 'FAIL'}] locate_failure_evidence_default_dir")
    return ok


def test_overlay_writes_a_file():
    """오버레이는 오피스가 행 방향/열 매핑/색 임계를 한 장으로 검증하는 수단이다."""
    layout = _layout_for_synth()
    image = _synth_panel([("black", "black")] * 7)
    rows = read_row_states(image, layout)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "overlay.jpg"
        save_assist_overlay(image, layout, rows, out)
        ok = out.exists() and out.stat().st_size > 0
    print(f"[{'PASS' if ok else 'FAIL'}] overlay_writes_a_file")
    return ok


def main():
    print("[INFO] assist_score self-test 시작")
    results = [
        test_black_ink(),
        test_red_ink(),
        test_blank_cell(),
        test_blank_when_ink_below_min_pixels(),
        test_mixed_ink_is_unknown(),
        test_dense_dark_cell_is_unknown_not_black(),
        test_normal_digit_density_still_classifies(),
        test_verdict_ok_without_addressing2(),
        test_verdict_fail_on_red_measurement(),
        test_verdict_fail_on_red_addressing1(),
        test_verdict_pending_when_measurement_blank(),
        test_verdict_ok_when_only_measurement_present(),
        test_verdict_unknown_beats_ok(),
        test_streak_counts_from_newest(),
        test_streak_skips_trailing_pending(),
        test_streak_broken_by_fail_and_unknown(),
        test_streak_all_ok_is_full_length(),
        test_streak_empty_rows_is_zero(),
        test_streak_top_anchor_when_newest_row_at_top(),
        test_grid_has_full_rows_and_columns(),
        test_grid_extrapolates_missing_rows_by_pitch(),
        test_grid_columns_follow_headers(),
        test_grid_columns_fallback_to_header_when_no_numbers(),
        test_grid_columns_ignore_header_order_in_items(),
        test_grid_none_without_headers(),
        test_grid_none_with_single_number_row(),
        test_grid_top_anchor_when_newest_row_at_top(),
        test_padded_cell_height_stays_within_row_pitch(),
        test_rendered_digits_read_ok_end_to_end(),
        test_realistic_glyph_density_survives_density_guard(),
        test_thumbnail_dense_cells_read_unknown_end_to_end(),
        test_resolve_coord_space_norm1000_on_large_panel(),
        test_resolve_coord_space_detects_frac01(),
        test_resolve_coord_space_detects_crop_px(),
        test_resolve_coord_space_detects_norm1000(),
        test_normalize_scales_norm1000_items_back_into_panel_px(),
        test_normalize_drops_items_outside_panel_bounds(),
        test_normalize_empty_items_returns_empty(),
        test_build_score_grid_with_norm1000_items_lands_on_real_cells(),
        test_read_rows_marks_black_and_red(),
        test_read_rows_blank_is_pending(),
        test_read_rows_returns_empty_without_layout(),
        test_read_rows_clamps_boxes_outside_panel(),
        test_panel_target_uses_proven_button_geometry(),
        test_locate_returns_layout_on_happy_path(),
        test_locate_with_norm1000_items_still_produces_grid(),
        test_locate_none_when_too_few_items_survive_normalization(),
        test_locate_none_when_point_outside_image(),
        test_locate_none_when_locator_raises(),
        test_locate_none_when_no_point(),
        test_locate_none_when_ocr_raises(),
        test_locate_none_when_grid_cannot_be_built(),
        test_locate_failure_leaves_evidence_on_disk(),
        test_score_text_accepts_decimal_and_signed(),
        test_header_match_survives_ocr_variants(),
        test_split_addressing_headers_assigned_by_x_order(),
        test_grid_builds_with_decimal_scores_and_split_headers(),
        test_locate_success_also_saves_ocr_overlay(),
        test_ocr_overlay_marks_roles_distinctly(),
        test_locate_failure_evidence_survives_missing_debug_dir(),
        test_overlay_writes_a_file(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
