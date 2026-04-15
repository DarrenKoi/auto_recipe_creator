"""CD-SEM Align Fail 알람 데이터 소스 (시뮬레이션).

실제 MES/알람 DB 연동 전 개발·테스트용 fake 데이터를 반환한다.
추후 실제 DB 쿼리로 교체 예정.

사용법:
  from poc.workflow_1.office_align_fail_alarm import get_cdsem_alarms, filter_align_fail
"""

import os
from datetime import datetime, timedelta

import pandas as pd

from poc.workflow_1.util import env_flag

ALID_ALIGN_FAIL = "9006"

# 기본 fake EQP_ID — workflow_select_tool.py 의 DEFAULT_TARGET_TOOL_NAME 과 동일
_DEFAULT_FAIL_EQP_ID = "6MCD2201"


def get_cdsem_alarms() -> pd.DataFrame:
    """최근 2분간 CD-SEM 알람 DataFrame 을 반환한다.

    컬럼:
    EQP_ID, ALID, AL_TEXT, LOT_ID, RECIPE_ID, OPERATION_DESC,
    ALARM_MODELNAME, EQ_STAT, SERVER_IP, UTC9

    환경변수 ``ALIGN_FAIL_SIMULATE`` 가 ``false`` 이면
    ALID=9006 행을 포함하지 않는다 (정상 상태 시뮬레이션).
    """
    now = datetime.now()

    rows = [
        {
            "EQP_ID": "6MCD2301",
            "ALID": "1001",
            "AL_TEXT": "Wafer Transfer Delay",
            "LOT_ID": "LOT-2301-A",
            "RECIPE_ID": "RECIPE-01",
            "OPERATION_DESC": "CD-SEM Measure",
            "ALARM_MODELNAME": "VeritySEM",
            "EQ_STAT": "ACTIVE",
            "SERVER_IP": "10.10.10.11",
            "UTC9": (now - timedelta(seconds=90)).strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "EQP_ID": "6MCD2302",
            "ALID": "2003",
            "AL_TEXT": "Stage Temperature Warning",
            "LOT_ID": "LOT-2302-B",
            "RECIPE_ID": "RECIPE-02",
            "OPERATION_DESC": "CD-SEM Align",
            "ALARM_MODELNAME": "VeritySEM",
            "EQ_STAT": "ACTIVE",
            "SERVER_IP": "10.10.10.12",
            "UTC9": (now - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    if env_flag("ALIGN_FAIL_SIMULATE", default=True):
        fail_eqp = os.getenv("ALIGN_FAIL_EQP_ID", _DEFAULT_FAIL_EQP_ID).strip()
        rows.append(
            {
                "EQP_ID": fail_eqp,
                "ALID": ALID_ALIGN_FAIL,
                "AL_TEXT": "Align Fail",
                "LOT_ID": "LOT-ALIGN-FAIL",
                "RECIPE_ID": "RECIPE-ALIGN",
                "OPERATION_DESC": "CD-SEM Align",
                "ALARM_MODELNAME": "VeritySEM",
                "EQ_STAT": "ACTIVE",
                "SERVER_IP": "10.10.10.21",
                "UTC9": (now - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return pd.DataFrame(rows)


def filter_align_fail(df: pd.DataFrame) -> pd.DataFrame:
    """ALID=9006 (Align Fail) 행만 필터링한다."""
    return df[df["ALID"] == ALID_ALIGN_FAIL].copy()


if __name__ == "__main__":
    alarms = get_cdsem_alarms()
    print("[INFO] 전체 알람:")
    print(alarms.to_string(index=False))
    print()

    fails = filter_align_fail(alarms)
    if fails.empty:
        print("[INFO] Align Fail 없음")
    else:
        print(f"[WARNING] Align Fail 감지 ({len(fails)}건):")
        print(fails.to_string(index=False))
