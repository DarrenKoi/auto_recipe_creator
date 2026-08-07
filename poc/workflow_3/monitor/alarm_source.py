"""align fail 알람 공급자 — office MES 모듈 또는 replay CSV fixture.

office 모듈(`office_align_fail_alarm`)은 gitignore(`**/office_*`) 대상이라 오피스
PC 에만 존재한다. 해석 순서:

  1. `poc.workflow_3.monitor.office_align_fail_alarm`  (정위치)
  2. replay CSV (`ALIGN_FAIL_REPLAY_CSV`)               (개발 PC dry-run 용)
  3. 비활성 (경고만)

본 모듈은 "알람 rows 를 어떻게 얻는가" 만 책임지고, 윈도우 필터/edge-trigger 는
호출부(`align_fail_monitor`)가 담당한다.

replay 사용법 (실알람을 기다리지 않고 사이클을 1회 강제 - tool 창 안쪽 VLM 경로
검증에 쓴다). `replay_fixture.example.csv` 를 복사해 EQP_ID/RECIPE_ID 만 실제 값으로
바꾼 뒤:

    $env:ALIGN_FAIL_ALARM_SOURCE = "replay"
    $env:ALIGN_FAIL_REPLAY_CSV = "poc/workflow_3/monitor/replay_fixture.csv"
    uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py

컬럼은 EQP_ID / RECIPE_ID / ALID / UTC9 (+ALARM_NAME/OPERATION_DESC/LOT_TYPE_CD 선택).
ALID 는 9006 이어야 필터를 통과하고, UTC9 는 poll 시 현재 시각으로 재기록되므로
값 자체는 자리표시자여도 된다. rows 는 **첫 poll 에 한 번만** 방출된다.
"""

import os

import pandas as pd

from poc.workflow_3.monitor.integration_loader import load_office_integration


class AlarmSource:
    """poll() 로 알람 DataFrame 을 돌려주는 공급자."""

    def __init__(self, kind: str, poll_fn, filter_fn, *, available: bool = True):
        self.kind = kind
        self.available = available
        self._poll_fn = poll_fn
        self._filter_fn = filter_fn

    def poll(self):
        """전체 CD-SEM 알람 rows 를 반환한다 (DataFrame 또는 None)."""
        if not self.available:
            return None
        return self._poll_fn()

    def filter_align_fail(self, rows):
        """알람 rows 에서 align fail(ALID=9006) 만 남긴다."""
        if rows is None:
            return None
        return self._filter_fn(rows)


def _load_office_module():
    """office_align_fail_alarm 을 정위치에서 찾는다. 없으면 None."""
    integration = load_office_integration(
        "office_align_fail_alarm",
        "poc.workflow_3.monitor.office_align_fail_alarm",
        required_attrs=("get_cdsem_alarms", "filter_align_fail"),
    )
    return integration.module if integration.available else None


def _replay_filter_align_fail(rows: "pd.DataFrame") -> "pd.DataFrame":
    """replay rows 에서 ALID=9006 만 남긴다 (ALID 컬럼이 없으면 전체 통과)."""
    if rows is None or rows.empty or "ALID" not in rows.columns:
        return rows
    mask = rows["ALID"].astype(str).str.strip() == "9006"
    return rows[mask].reset_index(drop=True)


class _ReplaySource:
    """CSV fixture 를 1회 방출하는 개발용 소스.

    첫 poll 에 CSV rows(UTC9 는 현재 시각으로 재기록 — 윈도우 필터 통과용)를 주고,
    이후에는 빈 결과를 줘 edge-trigger 해제 경로까지 자연스럽게 exercise 한다.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._emitted = False

    def poll(self):
        if self._emitted:
            return pd.DataFrame()
        self._emitted = True
        rows = pd.read_csv(self.csv_path, dtype=str).fillna("")
        rows["UTC9"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] replay 알람 방출: {self.csv_path} ({len(rows)} rows)")
        return rows


def load_alarm_source(kind: str = "office") -> AlarmSource:
    """설정된 종류의 AlarmSource 를 만든다. office 모듈이 없으면 replay→비활성 폴백."""
    if kind == "replay":
        csv_path = os.environ.get("ALIGN_FAIL_REPLAY_CSV", "").strip()
        if csv_path and os.path.isfile(csv_path):
            replay = _ReplaySource(csv_path)
            return AlarmSource("replay", replay.poll, _replay_filter_align_fail)
        print(f"[WARNING] ALIGN_FAIL_REPLAY_CSV 가 없거나 파일이 아님: {csv_path!r} - 알람 비활성")
        return AlarmSource("disabled", lambda: None, lambda rows: rows, available=False)

    module = _load_office_module()
    if module is not None:
        return AlarmSource("office", module.get_cdsem_alarms, module.filter_align_fail)

    print(
        "[WARNING] office_align_fail_alarm 모듈을 찾지 못함 — 알람 폴링 비활성. "
        "개발 PC 에서는 ALIGN_FAIL_ALARM_SOURCE=replay + ALIGN_FAIL_REPLAY_CSV 를 쓰세요."
    )
    return AlarmSource("disabled", lambda: None, lambda rows: rows, available=False)


__all__ = ["AlarmSource", "load_alarm_source"]
