"""RCS 부재 시 재실행+재로그인 복구 - 협력자 주입형 모듈.

`cycle._exec_ensure_rcs_ready` 가 쓰던 복구 분기를 떼어냈다. 실장비 없이 판정을
시험하기 위해 프로세스 조회/실행/로그인/창 대기를 전부 주입받는다(share_request.py,
row_occupant.read_occupancy 와 같은 규약).

이 모듈은 **tool 에 접속하지 않는다**. 복구의 끝은 RCS 메인 창이며, 어느 tool 로
들어갈지는 알람이 정한다(cycle 의 connect_tool 단계).
"""

from dataclasses import dataclass

from poc.workflow_3.rcs.open_rcs import RCS_EXE


# 복구 결과 status. 호출부(cycle) 가 failure_class 로 그대로 옮겨 적는다.
RECOVERED = "recovered"
STATUS_LAUNCH_ERROR = "rcs_recovery_launch_error"
STATUS_LOGIN_ERROR = "rcs_recovery_error"
STATUS_WINDOW_NOT_FOUND = "rcs_recovery_no_window"


@dataclass
class RecoveryOutcome:
    """복구 시도 결과. 창을 함께 실어 호출부의 재탐색을 없앤다."""

    status: str = RECOVERED
    window: object = None
    title: str = ""
    backend: str = ""
    launched: bool = False
    error: str = ""


def recover_rcs_session(
    settings,
    *,
    find_processes_fn,
    launch_fn,
    login_fn,
    wait_window_fn,
) -> RecoveryOutcome:
    """RCS 를 (필요하면) 실행하고 로그인해 메인 창을 확보한다."""
    launched = False
    running = find_processes_fn(RCS_EXE)
    if running is None:
        # 모름 - 실행을 보류한다. 빈 리스트("안 돌고 있음")와 반드시 구분해야 한다.
        # 이 분기가 곧 "RcsMainHD.exe 를 안 띄운다" 이므로, 왜 조회가 안 됐는지까지
        # 짚어 준다(대개 psutil 미설치 -> `uv sync` 한 번으로 끝난다).
        print(
            "[WARNING] RCS 프로세스 조회 불가 - 중복 실행 방지를 위해 **재실행 보류**, "
            "로그인만 시도합니다. psutil 이 설치돼 있는지 확인하세요(uv sync)."
        )
    elif not running:
        # exe 경로 기본값은 특정 사용자 경로라 PC 가 다르면 여기서 깨진다. 어떤 경로로
        # 띄우려 했는지 찍어야 RCS_EXE_PATH 오설정이 로그만으로 드러난다.
        print(f"[INFO] RCS 프로세스 없음 - 실행 시도: {RCS_EXE}")
        try:
            launch_fn(RCS_EXE)
        except Exception as exc:
            # 예외로 튀면 preflight 는 '준비 예외' 한 줄로, 알람 사이클은 status="error"
            # 로 뭉뚱그려져 "왜 RCS 가 안 떴는지" 가 manifest 에서 사라진다.
            print(f"[ERROR] RCS 실행 실패: {RCS_EXE} error={exc}")
            return RecoveryOutcome(
                status=STATUS_LAUNCH_ERROR,
                error=f"{type(exc).__name__}: {exc} (exe={RCS_EXE})",
            )
        launched = True
    else:
        print(f"[INFO] RCS 프로세스가 이미 있음(count={len(running)}) - 재실행 없이 로그인만 시도")
    # target_tool_name="" 는 계약이다 - 기본값에 맡기면 로그인 워크플로가 env 가
    # 지목한 tool 을 열어 버린다(test_login_never_opens_a_tool 참고).
    try:
        login_fn(settings, target_tool_name="")
    except Exception as exc:
        return RecoveryOutcome(
            status=STATUS_LOGIN_ERROR,
            launched=launched,
            error=f"{type(exc).__name__}: {exc}",
        )

    timeout_sec = settings.rcs_recovery_window_timeout_sec
    window, title, backend = wait_window_fn(timeout_sec=timeout_sec)
    if window is None:
        return RecoveryOutcome(
            status=STATUS_WINDOW_NOT_FOUND,
            launched=launched,
            error=f"로그인 후 {timeout_sec:.0f}s 안에 RCS 메인 창 미출현",
        )
    return RecoveryOutcome(
        status=RECOVERED,
        window=window,
        title=title,
        backend=backend,
        launched=launched,
    )
