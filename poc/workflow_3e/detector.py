"""측정 실패 임계 알람 필터 — 원본 알람 DataFrame 에서 MEAS_FAIL_ALID 만 남긴다.

workflow_3/monitor/alarm_source.py 를 건드리지 않으려고, 공급자의 raw alarms(=
source.poll())를 받아 여기서 직접 ALID 로 거른다(align 필터와 완전히 독립).

검출은 무상태다 — '연속 N회 실패' 스트릭 누적은 MES 가 셈하고, 임계 도달 시 한 건의
임계 알람을 쏜다. 여기서는 그 알람 row 만 골라내면 된다.
"""

import pandas as pd


def filter_measurement_fail(rows, alid: str):
    """알람 rows 에서 측정 실패 임계 ALID 만 남긴다.

    alid 가 빈 문자열(미설정)이거나 ALID 컬럼이 없으면 0건(잡 자가 비활성). rows 가
    None 이면 빈 DataFrame.
    """
    alid = (alid or "").strip()
    if rows is None:
        return pd.DataFrame()
    if hasattr(rows, "empty") and rows.empty:
        return rows
    if not alid or not hasattr(rows, "columns") or "ALID" not in rows.columns:
        return rows.iloc[0:0] if hasattr(rows, "iloc") else pd.DataFrame()
    mask = rows["ALID"].astype(str).str.strip() == alid
    return rows[mask].reset_index(drop=True)


__all__ = ["filter_measurement_fail"]
