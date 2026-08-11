"""Assist Window score 판독 self-test (VLM/실이미지 불필요).

합성 이미지와 합성 OCR 항목만 쓰므로 Mac 에서 그대로 돈다.

    uv run python poc/workflow_3/sem_monitor/test_assist_score.py
"""

import numpy as np

from poc.workflow_3.sem_monitor.assist_score import (
    AssistLayout,
    RowState,
    build_score_grid,
    classify_ink,
    ok_streak,
    row_verdict,
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


def test_grid_has_full_rows_and_columns():
    layout = build_score_grid(_panel_items(), (300, 260))
    ok = (
        layout is not None
        and len(layout.grid) == 7
        and all(len(row) == 3 for row in layout.grid)
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
    layout = build_score_grid(_panel_items(), (300, 260))
    if layout is None:
        print("[FAIL] grid_columns_follow_headers: layout None")
        return False
    first = layout.grid[0]
    ok = (
        layout.columns == ("Addressing1", "Addressing2", "Measurement")
        and first[0]["left"] == 10 and first[0]["right"] == 60
        and first[2]["left"] == 210 and first[2]["right"] == 260
    )
    print(f"[{'PASS' if ok else 'FAIL'}] grid_columns_follow_headers")
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


def main():
    print("[INFO] assist_score self-test 시작")
    results = [
        test_black_ink(),
        test_red_ink(),
        test_blank_cell(),
        test_blank_when_ink_below_min_pixels(),
        test_mixed_ink_is_unknown(),
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
        test_grid_has_full_rows_and_columns(),
        test_grid_extrapolates_missing_rows_by_pitch(),
        test_grid_columns_follow_headers(),
        test_grid_none_without_headers(),
        test_grid_none_with_single_number_row(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
