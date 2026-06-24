"""측정 실패 알람 소스 — 전용 office provider 우선, 없으면 공유 피드 ALID 필터 폴백.

해석 순서:
  1. `poc.workflow_3e.office_meas_many_fails.get_measurement_fail_alarms()` (정위치 provider)
     — 있으면 매 poll 마다 호출해 측정 실패 rows 를 받는다(스트릭은 MES 가 계산).
  2. 없으면 raw CD-SEM 피드(source.poll())를 MEAS_FAIL_ALID 로 거른다 — 측정 실패 알람이
     공유 알람 피드에 별도 ALID 로 들어오는 환경 대비.

office 모듈은 `temp_office_meas_many_fails.py` 견본을 복사해 구현한다(gitignore 대상).
"""

from poc.workflow_3.monitor.integration_loader import load_office_integration
from poc.workflow_3e.detector import filter_measurement_fail


def _load_meas_provider():
    """office_meas_many_fails.get_measurement_fail_alarms 를 정위치에서 찾는다. 없으면 None."""
    integration = load_office_integration(
        "office_meas_many_fails",
        "poc.workflow_3e.office_meas_many_fails",
        required_attrs=("get_measurement_fail_alarms",),
    )
    if not integration.available:
        return None
    return integration.attrs["get_measurement_fail_alarms"]


_PROVIDER = _load_meas_provider()
MEAS_PROVIDER_AVAILABLE = _PROVIDER is not None


def measurement_fail_rows(raw_alarms, alid: str):
    """측정 실패 알람 rows 를 얻는다(전용 provider 우선, ALID 필터 폴백).

    provider 가 있으면 raw_alarms 와 무관하게 그 결과를 쓰고(이미 측정 실패만 반환),
    예외/부재면 raw_alarms 를 MEAS_FAIL_ALID 로 거른다. 둘 다 같은 표준 스키마를 돌려준다.
    """
    if _PROVIDER is not None:
        try:
            return _PROVIDER()
        except Exception as exc:
            print(f"[WARNING] measurement-fail provider 예외 - ALID 필터 폴백: {exc}")
    return filter_measurement_fail(raw_alarms, alid)


__all__ = ["measurement_fail_rows", "MEAS_PROVIDER_AVAILABLE"]
