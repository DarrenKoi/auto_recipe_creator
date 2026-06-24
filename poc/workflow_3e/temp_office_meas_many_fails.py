"""[오피스 견본] 측정 연속 실패 알람 provider — workflow_3e 측정 abort 잡의 입력원.

이 파일은 git 추적용 **견본(temp_)** 이다. 오피스 PC 에서 `office_meas_many_fails.py` 로
복사해 MES 연결 부분만 채운다(실제 office_* 파일은 gitignore `**/office_*` 대상이라
git 에 올라가지 않는다). workflow_3e 는 정위치 `poc.workflow_3e.office_meas_many_fails`
에서 이 모듈을 로드한다.

역할: '한 recipe 가 한 tool 에서 측정 중 N회 연속 실패' 임계에 도달한 tool 들을 찾아
workflow_3 align-fail 과 **동일한 스키마**의 알람 rows(pandas DataFrame)로 돌려준다.
연속 실패 카운팅(스트릭)은 여기(MES 쪽)에서 끝낸다 — workflow_3e 는 무상태로 이 결과만
소비한다(align-fail 과 대칭: 검출은 office, 처리는 workflow).

오피스가 구현할 함수는 **딱 하나**: `get_measurement_fail_alarms()`.
나머지(`build_measurement_fail_alarm_row`)는 스키마를 보장해주는 헬퍼다.

    # workflow_3e 연결(자동): poc/workflow_3e/meas_alarm_source.py 가 매 poll 마다
    #   get_measurement_fail_alarms() 를 호출한다. 이 모듈이 없으면 공유 CD-SEM 피드를
    #   MEAS_FAIL_ALID(env) 로 거르는 폴백 경로로 동작한다.
"""

import pandas as pd

# --- 오피스 튜닝 상수 ---
# 측정 연속 실패 알람 ALID — 오피스에서 실제 값으로 확정. align fail(9006)과 달라야 한다.
# workflow_3e env MEAS_FAIL_ALID 와 같은 값을 쓰면 (이 provider 가 없을 때의) ALID-필터
# 폴백 경로도 함께 동작한다.
MEAS_FAIL_ALID = "9100"
# abort 를 띄울 연속 실패 임계(예: 100점 recipe 에서 20점 연속 실패면 신뢰 상실로 중단).
MEAS_FAIL_THRESHOLD = 20

# 알람 row 표준 컬럼 — workflow_3 align-fail 과 동일(monitor 의 _collapse_rows_by_tool 가
# 읽는 키들). 컬럼명/대문자는 이 순서·철자 그대로 유지해야 그대로 소비된다.
ALARM_COLUMNS = [
    "EQP_ID",          # (필수) 장비 ID — 접속 대상 + EQP_ID edge-trigger dedup 기준.
    "ALID",            # (필수) 알람 ID — 측정 실패 식별자(MEAS_FAIL_ALID). align 9006 과 구분.
    "UTC9",            # (필수) 알람 시각 "%Y-%m-%d %H:%M:%S" — 최근성 윈도우 필터 + 캡처 폴더 태그.
    "RECIPE_ID",       # (권장) "<class>/<recipe>" — 캡처/자산 경로. 없으면 _unregistered 로 처리.
    "ALARM_NAME",      # (권장) 사람이 읽는 라벨 — 연속 실패 수를 여기 담으면 알람 로그 + cube 알림에 노출.
    "OPERATION_DESC",  # (권장) 공정/스텝 설명.
    "LOT_TYPE_CD",     # (권장) lot 종류 코드.
    "TIMESTAMP",       # (선택) 알람 시각 대체(없으면 UTC9 사용).
]


def build_measurement_fail_alarm_row(
    *,
    eqp_id: str,
    recipe_id: str,
    utc9: str,
    fail_count: int,
    total_points: int = 0,
    operation_desc: str = "",
    lot_type_cd: str = "",
    alid: str = MEAS_FAIL_ALID,
) -> dict:
    """측정 실패 정보 1건을 표준 알람 row(dict)로 만든다(스키마 보장 헬퍼).

    입력:
      eqp_id        : 장비 ID(필수)
      recipe_id     : "<class>/<recipe>"(권장 — 캡처/자산 경로용)
      utc9          : 알람 시각 "%Y-%m-%d %H:%M:%S"(필수 — 최근성 윈도우 통과해야 처리됨)
      fail_count    : 연속 실패 점 수(필수 — 라벨/info 표기)
      total_points  : recipe 총 측정점 수(선택 — 라벨에 "20/100" 으로 표기)
      operation_desc / lot_type_cd : 추가 컨텍스트(선택)
      alid          : 알람 ID(기본 MEAS_FAIL_ALID)

    출력: ALARM_COLUMNS 키를 갖는 dict. 연속 실패 수/총점은 ALARM_NAME 에 함께 실어,
    별도 수치 컬럼을 아직 소비하지 않는 현재 파이프라인에서도 엔지니어가 알람 로그와
    cube 알림에서 바로 보게 한다.
    """
    label = f"Measurement Consecutive Fail ({fail_count}"
    label += f"/{total_points})" if total_points else ")"
    return {
        "EQP_ID": eqp_id,
        "ALID": alid,
        "UTC9": utc9,
        "RECIPE_ID": recipe_id,
        "ALARM_NAME": label,
        "OPERATION_DESC": operation_desc,
        "LOT_TYPE_CD": lot_type_cd,
        "TIMESTAMP": utc9,
    }


def get_measurement_fail_alarms() -> "pd.DataFrame":
    """[오피스 구현] 지금 '연속 N회 측정 실패' 임계를 넘긴 tool 들의 표준 알람 rows.

    출력: ALARM_COLUMNS 스키마의 pandas DataFrame(없으면 빈 DataFrame). 입력 없음 —
    내부에서 MES 를 조회한다. 매 poll(주기 ALIGN_FAIL_POLL_SEC) 마다 호출된다.

    **무상태로 호출돼도 되게** 설계됐다: 같은 임계 상태가 지속되는 동안 같은 EQP_ID row 를
    계속 돌려줘도 된다 — workflow_3e 가 EQP_ID edge-trigger 로 1회만 abort 하고, 임계가
    풀려 row 가 사라지면 그 EQP_ID 를 재처리 가능 상태로 되돌린다(align-fail 과 동일).

    구현 가이드(오피스):
      1. 지금 측정 중인 (tool, recipe, lot) 들을 MES 에서 조회.
      2. 각 run 의 최근 측정점 결과로 '연속 실패 수'를 계산(스트릭은 여기서 마무리).
      3. fail_count >= MEAS_FAIL_THRESHOLD 인 run 만 build_measurement_fail_alarm_row 로 변환.
      4. UTC9 는 '지금' 검출 시각으로(최근성 윈도우 통과). pd.DataFrame(rows, columns=ALARM_COLUMNS).

    주의:
      * RECIPE_ID 는 "<class>/<recipe>" 형태로 채울 것(workflow_3 경로 규약).
      * 이미 abort 한 run 을 또 돌려줘도 안전하다(edge-trigger 가 막음). 단 같은 알람을
        '영구히' 돌려주면 그 tool 은 임계 해제 전까지 재abort 가 안 된다(의도된 동작).
    """
    # --- 개발 PC 견본: 빈 결과. 오피스에서 아래 예시처럼 MES 조회로 교체한다. ---
    rows: list[dict] = []
    # 예시(주석 — 오피스 구현 시 풀어서 사용):
    # now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    # for run in _query_running_measurements():            # 오피스: MES 측정 run 목록
    #     n = _consecutive_fail_count(run)                 # 오피스: 최근 점들의 연속 실패 수
    #     if n >= MEAS_FAIL_THRESHOLD:
    #         rows.append(build_measurement_fail_alarm_row(
    #             eqp_id=run.eqp_id, recipe_id=run.recipe_id, utc9=now,
    #             fail_count=n, total_points=run.total_points,
    #             operation_desc=run.operation_desc, lot_type_cd=run.lot_type_cd,
    #         ))
    return pd.DataFrame(rows, columns=ALARM_COLUMNS)


# ------------------------------------------------------------------
# (선택) 측정 실패 전용 cube 메시지.
# ------------------------------------------------------------------
# 기본적으로 workflow_3e.notify 는 align 과 같은 office_rich_notify.send_cube_align_fail_info
# 어댑터로 abort 결과를 보낸다(ALARM_NAME 에 담은 연속 실패 수가 cube 요약에 함께 나간다).
# 측정 실패용 별도 cube 양식을 원하면 office_rich_notify 에 아래 시그니처를 추가하고
# workflow_3e.notify 를 그쪽으로 바꾸면 된다(필수 아님):
#
#   def send_cube_meas_fail_info(eqp_id, recipe_id, *, summary=""):
#       """측정 연속 실패 cube rich notification 발송."""
#       ...


__all__ = [
    "MEAS_FAIL_ALID",
    "MEAS_FAIL_THRESHOLD",
    "ALARM_COLUMNS",
    "build_measurement_fail_alarm_row",
    "get_measurement_fail_alarms",
]
