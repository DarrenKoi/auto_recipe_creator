"""Assist Window score 판독 self-test (VLM/실이미지 불필요).

합성 이미지와 합성 OCR 항목만 쓰므로 Mac 에서 그대로 돈다.

    uv run python poc/workflow_3/sem_monitor/test_assist_score.py
"""

import numpy as np

from poc.workflow_3.sem_monitor.assist_score import classify_ink


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


def main():
    print("[INFO] assist_score self-test 시작")
    results = [
        test_black_ink(),
        test_red_ink(),
        test_blank_cell(),
        test_blank_when_ink_below_min_pixels(),
        test_mixed_ink_is_unknown(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
