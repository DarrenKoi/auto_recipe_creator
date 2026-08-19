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

# 기존 프로세스 판정. "없음" 과 "모름" 과 "좀비" 는 대응이 전부 다르다.
PROCESS_UNKNOWN = "unknown"    # 조회 자체가 불가(psutil 부재 등) - 실행 보류.
PROCESS_NONE = "none"          # 확실히 없음 - 실행한다.
PROCESS_WINDOWED = "windowed"  # 창을 가진 정상 프로세스 - 건드리지 않는다.
PROCESS_STALE = "stale"        # 프로세스는 있는데 창이 하나도 없음 - 좀비 가능성.


def classify_existing_processes(running, list_windows_fn) -> str:
    """기존 RCS 프로세스를 4-상태로 가른다.

    2026-08-19 오피스에서 **작업 표시줄에 없고 PID 만 남은 RCS** 가 있었다. 중복 실행
    가드가 그것을 "살아 있음" 으로 세는 바람에 재실행이 막혔고, 로그인만 시도하다 창이
    안 떠 `rcs_recovery_no_window` 로 끝났다 - 가드가 원래 막으려던 것(중복 실행)과
    정반대 상황에서 발목을 잡았다. 그래서 창 보유 여부까지 본다.

    `list_windows_fn(pid) -> list` 는 그 PID 가 가진 top-level 창 목록이다. 주어지지
    않거나 예외가 나면 **`windowed`(살아 있음)로 본다** - 프로세스 종료는 되돌릴 수
    없으므로, 모를 때는 죽이지 않는 쪽이 맞다. 창을 가진 프로세스가 **하나라도** 있으면
    좀비로 보지 않는다(그 하나가 엔지니어가 쓰는 세션일 수 있다).
    """
    if running is None:
        return PROCESS_UNKNOWN
    if not running:
        return PROCESS_NONE
    if list_windows_fn is None:
        return PROCESS_WINDOWED

    for proc in running:
        pid = proc.get("pid") if isinstance(proc, dict) else None
        if pid is None or pid < 0:
            return PROCESS_WINDOWED  # PID 를 모르면 판단하지 않는다.
        try:
            if list_windows_fn(pid):
                return PROCESS_WINDOWED
        except Exception as exc:
            print(f"[WARNING] PID={pid} 창 조회 실패(살아 있는 것으로 간주): {exc}")
            return PROCESS_WINDOWED
    return PROCESS_STALE


def _terminate_stale(running, terminate_fn) -> None:
    """창 없는 프로세스를 종료한다. 실패는 삼킨다 - 어차피 실행을 이어서 시도한다."""
    for proc in running:
        pid = proc.get("pid") if isinstance(proc, dict) else None
        if pid is None or pid < 0:
            continue
        try:
            terminate_fn(pid)
            print(f"[INFO] 창 없는 RCS 프로세스 종료: PID={pid}")
        except Exception as exc:
            print(f"[WARNING] PID={pid} 종료 실패(실행은 계속 시도): {exc}")


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
    list_windows_fn=None,
    terminate_fn=None,
) -> RecoveryOutcome:
    """RCS 를 (필요하면) 실행하고 로그인해 메인 창을 확보한다.

    `list_windows_fn(pid)` / `terminate_fn(pid)` 는 **창 없는 좀비 프로세스** 대응용
    이며 둘 다 선택이다. 주지 않으면 종전 동작(프로세스가 있으면 재실행하지 않음)
    그대로다 - `classify_existing_processes` 참고.
    """
    launched = False
    running = find_processes_fn(RCS_EXE)
    verdict = classify_existing_processes(running, list_windows_fn)

    if verdict == PROCESS_UNKNOWN:
        # 모름 - 실행을 보류한다. 빈 리스트("안 돌고 있음")와 반드시 구분해야 한다.
        # 이 분기가 곧 "RcsMainHD.exe 를 안 띄운다" 이므로, 왜 조회가 안 됐는지까지
        # 짚어 준다(대개 psutil 미설치 -> `uv sync` 한 번으로 끝난다).
        print(
            "[WARNING] RCS 프로세스 조회 불가 - 중복 실행 방지를 위해 **재실행 보류**, "
            "로그인만 시도합니다. psutil 이 설치돼 있는지 확인하세요(uv sync)."
        )
    elif verdict in (PROCESS_NONE, PROCESS_STALE):
        if verdict == PROCESS_STALE:
            kill_enabled = getattr(settings, "rcs_kill_stale_enabled", False)
            print(
                f"[WARNING] RCS 프로세스는 있으나(count={len(running)}) 창이 하나도 "
                "없습니다 - 작업 표시줄에 안 보이는 좀비 프로세스일 수 있습니다."
            )
            if not (kill_enabled and terminate_fn is not None):
                # 종료는 되돌릴 수 없어 기본값은 보고만 한다. 그래도 무엇을 해야 하는지
                # 는 짚어 준다 - 이 한 줄이 없으면 오피스에서 원인을 못 찾는다.
                print(
                    "[WARNING] 재실행 보류(중복 실행 가드) - 로그인만 시도합니다. "
                    "이 상태가 반복되면 작업 관리자에서 해당 PID 를 종료하거나 "
                    "ALIGN_FAIL_RCS_KILL_STALE=1 로 자동 종료를 켜세요."
                )
                return _login_and_wait(settings, login_fn, wait_window_fn, launched)
            _terminate_stale(running, terminate_fn)

        # exe 경로 기본값은 특정 사용자 경로라 PC 가 다르면 여기서 깨진다. 어떤 경로로
        # 띄우려 했는지 찍어야 RCS_EXE_PATH 오설정이 로그만으로 드러난다.
        print(f"[INFO] RCS 실행 시도: {RCS_EXE}")
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
    return _login_and_wait(settings, login_fn, wait_window_fn, launched)


def _login_and_wait(settings, login_fn, wait_window_fn, launched: bool) -> RecoveryOutcome:
    """로그인 후 메인 창을 기다린다. 실행 여부와 무관한 뒷단이라 따로 뽑았다."""
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
