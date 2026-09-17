"""WorkflowRunner 저널 위치 테스트 - 이벤트 폴더 안에서 돌면 저널도 그 안에 쌓인다.

`uv run pytest poc/workflow_3/runner/test_workflow_runner.py`
"""

from pathlib import Path

from poc.workflow_3.runner.workflow_config import WorkflowSettings
from poc.workflow_3.runner.workflow_runner import WorkflowRunner
from poc.workflow_3.runner.workflow_types import WorkflowStep
from poc.workflow_3.util.event_dir import event_scope


def _run_one_step(context, name="align_fail_cycle_MCD019"):
    runner = WorkflowRunner(WorkflowSettings(safe_mode=True), workflow_name=name)
    step = WorkflowStep(step_id="s1", step_type="action", target_description="noop")
    return runner.run(
        [step], context,
        lambda s, ctx: runner._build_result(
            step=s, status="success", started_at=0.0, safe_mode=True
        ),
    )


def test_journal_lands_under_the_active_event(tmp_path):
    take = tmp_path / "MCD019-260917_134600"

    with event_scope(take):
        run = _run_one_step({})

    run_dir = Path(run.run_dir)
    assert run_dir.parent == take / "runs"
    assert (run_dir / "step_s1.json").is_file()
    assert (run_dir / "run_state.json").is_file()


def test_nested_runs_in_one_event_get_separate_journals(tmp_path):
    """사이클 안에서 복구 로그인이 자기 runner 를 돌려도 저널이 섞이지 않는다."""
    take = tmp_path / "MCD019-260917_134600"

    with event_scope(take):
        outer = _run_one_step({})
        inner = _run_one_step({}, name="rcs_login")

    assert outer.run_dir != inner.run_dir
    assert sorted(p.name for p in (take / "runs").iterdir()) == sorted(
        [Path(outer.run_dir).name, Path(inner.run_dir).name]
    )
