"""사이클 teardown 을 단계별로 독립 보호해 실행하는 헬퍼.

teardown 의 계약은 "가능한 만큼 정리한다" 이다. 한 단계가 실패했다고 뒤 단계
(특히 사용자 입력 차단 해제, tool 창 닫기)를 건너뛰면 장비와 엔지니어가 잠긴 채
남는다. 그래서 각 단계를 개별 try 로 감싸고 무조건 다음으로 넘어간다.

**순서 규약**: 모든 teardown 목록의 **첫 단계는 사용자 입력 차단 해제**여야 한다.
뒤 단계가 전부 실패해도 엔지니어의 마우스/키보드는 풀려 있어야 하기 때문이다.
이 규약은 test_teardown.py 가 세 사이클 모두에 대해 검사한다.
"""


def run_teardown(steps, *, label=""):
    """teardown 단계를 순서대로 실행하되, 각 단계를 독립 보호한다.

    steps: (이름, 인자없는 callable) 목록. 이름은 로그/반환값 식별자다.
    label: 로그에 붙일 사이클 식별 문자열(예: "align_fail_cycle EQP1").
    반환: 실패한 (이름, "예외타입: 메시지") 목록 — 성공만이면 빈 목록.
    이 함수 자체는 어떤 경우에도 예외를 올리지 않는다.
    """
    failures = []
    suffix = f" [{label}]" if label else ""
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            failures.append((name, detail))
            print(f"[WARNING] teardown 단계 실패({name}){suffix}: {detail}")
    return failures


def close_tool_checked(close_fn, eqp_id, *, attempts=2):
    """close_fn(eqp_id) 로 tool 창을 닫고, `close_failed` 면 다시 시도한 뒤 그래도 실패면 던진다.

    세 사이클(alarm / check-only / 3e abort)의 close_tool 단계가 전부 이것을 쓴다 -
    한 곳만 결과를 확인하면 나머지는 창이 열린 채 '깨끗한 teardown' 으로 보고한다.
    `tool_window_not_found`(엔지니어가 이미 닫음)는 실패가 아니다. close_fn 을 인자로
    받는 이유는 각 사이클 모듈의 `close_tool` 이름이 테스트 치환 지점이기 때문이다.
    """
    for attempt in range(1, attempts + 1):
        closed = close_fn(eqp_id)
        if getattr(closed, "exit_code", "") != "close_failed":
            return closed
        print(f"[WARNING] tool 창 닫기 실패({attempt}/{attempts}): EQP_ID={eqp_id}")
    raise RuntimeError(f"tool 창 닫기 실패({attempts}회 시도): EQP_ID={eqp_id}")


__all__ = ["close_tool_checked", "run_teardown"]
