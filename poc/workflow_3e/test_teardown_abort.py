"""workflow_3e abort 사이클의 teardown 순서 불변식 테스트.

workflow_3 는 workflow_3e 를 절대 import 하면 안 된다(one-way 계층). 이 테스트는
`_abort_teardown_steps`(poc.workflow_3e.abort_cycle) 를 검증하려면 workflow_3e 를
import 해야 하므로, 이 파일을 workflow_3e 쪽에 둔다 - poc/workflow_3/monitor/에 두면
그 파일이 workflow_3e 를 import 하게 되어 계층을 거꾸로 만든다.

형제 사이클(alarm/check-only)의 teardown 순서 테스트는
`poc/workflow_3/monitor/test_teardown.py` 에 있다.

`uv run python poc/workflow_3e/test_teardown_abort.py` 로 직접 실행.
"""


def test_abort_cycle_teardown_unblocks_input_first():
    """workflow_3e abort 사이클도 같은 순서 규약을 따른다(F4 - 복제된 형태 방지)."""
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3e.abort_cycle import _abort_teardown_steps

    settings = load_workflow3_settings()
    steps = _abort_teardown_steps("EQP1", {}, settings, input_blocked=True)
    names = [n for n, _ in steps]
    assert names == ["input_unblock", "close_tool", "close_alert"], names
    print("[OK] test_abort_cycle_teardown_unblocks_input_first")


if __name__ == "__main__":
    test_abort_cycle_teardown_unblocks_input_first()
    print("\n[OK] abort 사이클 teardown 테스트 통과")
