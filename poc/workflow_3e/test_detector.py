"""filter_measurement_fail self-test — raw 알람에서 측정 실패 ALID 만 분리.

    uv run python poc/workflow_3e/test_detector.py
"""

import pandas as pd

from poc.workflow_3e.detector import filter_measurement_fail


def _rows():
    return pd.DataFrame(
        [
            {"EQP_ID": "EQP1", "ALID": "9006"},   # align fail
            {"EQP_ID": "EQP2", "ALID": "9012"},   # measurement fail (예시 ALID)
            {"EQP_ID": "EQP3", "ALID": "9999"},   # 무관
        ]
    )


def test_splits_by_alid():
    out = filter_measurement_fail(_rows(), "9012")
    ok = len(out) == 1 and out.iloc[0]["EQP_ID"] == "EQP2"
    print(f"[{'PASS' if ok else 'FAIL'}] splits_by_alid: n={len(out)}")
    return ok


def test_empty_alid_zero():
    out = filter_measurement_fail(_rows(), "")
    ok = out is not None and len(out) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] empty_alid_zero: n={len(out)}")
    return ok


def test_none_rows():
    out = filter_measurement_fail(None, "9012")
    ok = out is not None and len(out) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] none_rows: n={len(out)}")
    return ok


def test_no_match_zero():
    out = filter_measurement_fail(_rows(), "1234")
    ok = len(out) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] no_match_zero: n={len(out)}")
    return ok


def main():
    print("[INFO] detector self-test 시작")
    results = [test_splits_by_alid(), test_empty_alid_zero(), test_none_rows(), test_no_match_zero()]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
