"""모니터 기동 시 RCS 세션 준비 — 실행 -> 로그인 -> List 탭까지 미리 끝낸다.

`align_fail_monitor` 는 알람이 오기 전까지 RCS 를 건드리지 않았다. 그래서 RCS 가
안 떠 있으면 **첫 알람이 그 비용을 통째로 낸다** — 부팅 + 로그인 + List 탭이 알람
처리 시간에 붙고, 장비는 그동안 멈춰 있다. 이 모듈은 그 준비를 루프 진입 **전** 1회로
당겨, 문제가 있으면 알람이 오기 전에 콘솔에서 드러나게 한다.

준비 순서(사용자 지정, 2026-08-19): `open_rcs -> workflow_login -> List 탭 -> 알람 대기`.

**List 탭이 이 모듈의 존재 이유 중 절반이다.** `connect_to_tool` 은 "현재 List 탭에서"
tool 을 찾는다고 가정하는데, 복구 로그인은 `target_tool_name=""` 로 부르므로
`workflow_login.build_login_workflow_steps` 의 List 클릭 step 이 아예 안 붙는다 —
`click_list_tab`/`verify_list_tab_opened`/`open_target_tool` 이 전부 `if
normalized_tool_name:` 한 블록에 묶여 있어서, "tool 을 열지 않는다" 는 계약이 List
탭까지 같이 꺼 버린다. 그 공백을 여기서 명시적으로 메운다.

실행/로그인 자체는 `rcs_recovery.recover_rcs_session` 을 그대로 쓴다(중복 실행 가드와
"복구 로그인은 tool 에 접속하지 않는다" 계약이 이미 거기 있다). 이 모듈이 더하는 것은
① 이미 로그인돼 있으면 건드리지 않기 ② List 탭 열기 ③ 실패해도 죽지 않기 다.

**best-effort 다.** 준비가 실패해도 모니터는 뜬다 — 알람이 오면 사이클의
`ensure_rcs_ready` step 이 같은 복구를 다시 시도하므로, 준비 실패가 곧 감시 중단이
되어서는 안 된다. 그래서 모든 실패는 예외가 아니라 status 로 나간다.
"""

from dataclasses import dataclass

from poc.workflow_3.rcs.view_list_tab_rcs import EXIT_SUCCESS as LIST_TAB_SUCCESS

# 준비 결과 status. 알람 사이클의 failure_class 와 섞이지 않게 접두를 따로 둔다 —
# 준비 단계에서 난 문제와 알람 처리 중 난 문제는 대응이 다르다.
STATUS_READY = "ready"
STATUS_NO_WINDOW = "rcs_preflight_no_window"
STATUS_LIST_TAB_FAILED = "rcs_preflight_list_tab_failed"


@dataclass
class PreflightOutcome:
    """준비 결과. 창을 실어 호출부가 다시 찾지 않게 한다."""

    status: str = STATUS_READY
    window: object = None
    title: str = ""
    backend: str = ""
    launched: bool = False
    error: str = ""

    @property
    def ready(self) -> bool:
        """알람을 바로 받을 수 있는 상태인가."""
        return self.status == STATUS_READY


def ensure_rcs_session_ready(
    settings,
    *,
    find_window_fn,
    recover_fn,
    open_list_fn,
) -> PreflightOutcome:
    """RCS 를 로그인 + List 탭 상태까지 올려둔다. 실패해도 예외를 올리지 않는다.

    협력자:
      find_window_fn(**kwargs) -> (window, title, backend)   메인 창 탐색
      recover_fn()             -> RecoveryOutcome            실행 + 로그인
      open_list_fn(window, title, backend) -> exit_code(str) List 탭 클릭
    """
    window, title, backend = find_window_fn(
        timeout_sec=settings.connect_window_timeout_sec
    )
    launched = False

    if window is None:
        if not settings.rcs_recovery_enabled:
            # 같은 스위치가 두 경로(preflight/알람 시 복구)를 함께 꺼야 한다. 껐다고
            # 생각한 사람이 모니터 기동만으로 RCS 가 뜨는 것을 보면 안 된다.
            print(
                "[WARNING] RCS 메인 창 없음 - 복구 비활성(ALIGN_FAIL_RCS_RECOVERY=0)이라 "
                "준비를 건너뜁니다. 알람 전에 RCS 를 직접 로그인해 두세요."
            )
            return PreflightOutcome(status=STATUS_NO_WINDOW)

        print("[INFO] RCS 메인 창 없음 - 기동 준비로 실행+로그인 시도")
        recovery = recover_fn()
        if recovery.status != "recovered":
            print(
                f"[WARNING] RCS 준비 실패: status={recovery.status} error={recovery.error} "
                "- 모니터는 계속 뜹니다(알람 시 복구를 다시 시도)."
            )
            return PreflightOutcome(
                status=recovery.status, launched=recovery.launched, error=recovery.error,
            )
        window, title, backend = recovery.window, recovery.title, recovery.backend
        launched = recovery.launched
    else:
        print(f"[INFO] RCS 이미 로그인됨 - 재실행/재로그인 생략: title={title!r}")

    # List 탭은 여기서 반드시 연다. connect_to_tool 이 '현재 List 탭' 을 전제하는데,
    # 복구 로그인 경로(target_tool_name="")에는 List 클릭 step 이 없다 - 모듈 docstring 참고.
    try:
        exit_code = open_list_fn(window, title, backend)
    except Exception as exc:
        print(f"[WARNING] List 탭 열기 예외(준비만 미완, 감시는 계속): {exc}")
        return PreflightOutcome(
            status=STATUS_LIST_TAB_FAILED, window=window, title=title,
            backend=backend, launched=launched, error=f"{type(exc).__name__}: {exc}",
        )

    if exit_code != LIST_TAB_SUCCESS:
        # 창까지 버리지 않는다 - List 가 이미 열려 있어 클릭이 불필요했을 수도 있고,
        # 다음 알람의 connect 가 성공할 수도 있다.
        print(f"[WARNING] List 탭 열기 실패: exit_code={exit_code} (감시는 계속)")
        return PreflightOutcome(
            status=STATUS_LIST_TAB_FAILED, window=window, title=title,
            backend=backend, launched=launched, error=f"list_tab_exit={exit_code}",
        )

    print(f"[INFO] RCS 준비 완료: 로그인 + List 탭 (relaunched={launched}) - 알람 대기 시작")
    return PreflightOutcome(
        status=STATUS_READY, window=window, title=title,
        backend=backend, launched=launched,
    )


__all__ = [
    "STATUS_LIST_TAB_FAILED",
    "STATUS_NO_WINDOW",
    "STATUS_READY",
    "PreflightOutcome",
    "ensure_rcs_session_ready",
]
