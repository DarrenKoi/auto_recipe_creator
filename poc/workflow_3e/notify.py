"""측정 abort 결과 cube 알림 — workflow_3 notify 의 office 어댑터를 재사용한다.

notify_correction_outcome 과 같은 _SEND_CUBE_FN(office_rich_notify 위임)을 그대로 쓰되,
메시지만 abort 잡용으로 바꾼다. 어댑터 부재(개발 PC)면 텍스트 로그만 남긴다.
"""

import inspect
import threading

import poc.workflow_3.monitor.notify as wf3_notify
from poc.workflow_3.logger import log_work2_event

LOG_COMPONENT = "measurement_abort_notify"

_ABORT_SUMMARY = {
    "aborted": "측정을 자동 중단했습니다(연속 측정 실패 임계 도달).",
    "abort_dry_run": "측정 중단 대상 감지(notify-only) - 엔지니어 확인 필요.",
    "abort_button_not_found": "Stop/Abort 버튼 미검출 - 엔지니어 직접 중단 필요.",
    "abort_error": "측정 중단 시도 중 오류 - 엔지니어 직접 확인 필요.",
}


def notify_abort_outcome(
    eqp_id: str,
    recipe_id: str,
    outcome,
    *,
    capture_path: str = "",
    detail: str = "",
    enabled: bool = True,
) -> None:
    """측정 abort 결과를 cube rich notification 으로 비차단 발송한다.

    outcome 은 status 문자열('aborted'|'abort_dry_run'|'abort_button_not_found'|
    'abort_error') 또는 None(rcs 비활성). detail 은 알람 정보(예: 연속 실패 수가 담긴
    ALARM_NAME)로, 요약 앞에 붙어 엔지니어에게 전달된다. enabled=False/어댑터 부재면
    텍스트 로그만.
    """
    status = outcome or "unknown"
    summary = _ABORT_SUMMARY.get(status, "측정 실패 abort 잡 - 엔지니어 확인 필요.")
    if detail:
        summary = f"{detail} - {summary}"
    if capture_path:
        summary = f"{summary} (capture={capture_path})"

    log_work2_event(
        component=LOG_COMPONENT, message="abort_notify", level="warning",
        eqp_id=eqp_id, recipe_id=recipe_id, status=str(status), summary=summary,
    )
    if not enabled or not wf3_notify.RICH_NOTIFY_AVAILABLE:
        print(f"[INFO] abort cube 알림 비활성 - 요약 로그만: EQP_ID={eqp_id} | {summary}")
        return

    def _run():
        try:
            fn = wf3_notify._SEND_CUBE_FN
            params = inspect.signature(fn).parameters
            if "summary" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                fn(eqp_id, recipe_id, summary=summary)
            else:
                fn(eqp_id, recipe_id)
        except Exception as exc:
            print(f"[WARNING] abort cube notify 예외: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    print(f"[INFO] abort cube 알림 발송(비차단): EQP_ID={eqp_id} | {summary}")


__all__ = ["notify_abort_outcome"]
