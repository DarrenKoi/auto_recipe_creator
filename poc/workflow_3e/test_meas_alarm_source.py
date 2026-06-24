"""measurement_fail_rows self-test — 전용 provider 우선 / ALID 필터 폴백.

    uv run python poc/workflow_3e/test_meas_alarm_source.py
"""

import pandas as pd

from poc.workflow_3e import meas_alarm_source as mas


def _raw():
    return pd.DataFrame(
        [
            {"EQP_ID": "EQP1", "ALID": "9006"},   # align
            {"EQP_ID": "EQP2", "ALID": "9100"},   # measurement (예시 ALID)
        ]
    )


def test_fallback_filters_by_alid():
    """provider 부재면 raw 피드를 ALID 로 거른다."""
    orig = mas._PROVIDER
    mas._PROVIDER = None
    try:
        out = mas.measurement_fail_rows(_raw(), "9100")
    finally:
        mas._PROVIDER = orig
    ok = len(out) == 1 and out.iloc[0]["EQP_ID"] == "EQP2"
    print(f"[{'PASS' if ok else 'FAIL'}] fallback_filters_by_alid: n={len(out)}")
    return ok


def test_provider_preferred():
    """provider 가 있으면 raw 와 무관하게 그 결과를 쓴다(ALID 필터 안 탐)."""
    orig = mas._PROVIDER
    provided = pd.DataFrame([{"EQP_ID": "EQPX", "ALID": "9100", "UTC9": ""}])
    mas._PROVIDER = lambda: provided
    try:
        out = mas.measurement_fail_rows(_raw(), "")  # alid 빈값이어도 provider 우선
    finally:
        mas._PROVIDER = orig
    ok = len(out) == 1 and out.iloc[0]["EQP_ID"] == "EQPX"
    print(f"[{'PASS' if ok else 'FAIL'}] provider_preferred: eqp={out.iloc[0]['EQP_ID'] if len(out) else '-'}")
    return ok


def test_provider_exception_falls_back():
    """provider 예외면 ALID 필터로 폴백(루프가 죽지 않게)."""
    orig = mas._PROVIDER

    def _boom():
        raise RuntimeError("MES timeout")

    mas._PROVIDER = _boom
    try:
        out = mas.measurement_fail_rows(_raw(), "9100")
    finally:
        mas._PROVIDER = orig
    ok = len(out) == 1 and out.iloc[0]["EQP_ID"] == "EQP2"
    print(f"[{'PASS' if ok else 'FAIL'}] provider_exception_falls_back: n={len(out)}")
    return ok


def test_template_importable_and_schema():
    """견본 모듈이 dev PC 에서 import 되고 표준 스키마 row 를 만든다."""
    from poc.workflow_3e.temp_office_meas_many_fails import (
        ALARM_COLUMNS,
        build_measurement_fail_alarm_row,
        get_measurement_fail_alarms,
    )

    row = build_measurement_fail_alarm_row(
        eqp_id="EQP1", recipe_id="CLS/RCP", utc9="2026-06-25 10:00:00",
        fail_count=20, total_points=100,
    )
    empty = get_measurement_fail_alarms()
    ok = (
        all(k in row for k in ("EQP_ID", "ALID", "UTC9", "RECIPE_ID", "ALARM_NAME"))
        and "20/100" in row["ALARM_NAME"]
        and list(empty.columns) == ALARM_COLUMNS and len(empty) == 0
    )
    print(f"[{'PASS' if ok else 'FAIL'}] template_importable_and_schema: "
          f"label={row['ALARM_NAME']!r} cols_ok={list(empty.columns) == ALARM_COLUMNS}")
    return ok


def main():
    print("[INFO] meas_alarm_source self-test 시작")
    results = [
        test_fallback_filters_by_alid(),
        test_provider_preferred(),
        test_provider_exception_falls_back(),
        test_template_importable_and_schema(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
