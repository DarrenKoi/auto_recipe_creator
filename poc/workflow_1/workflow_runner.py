"""워크플로 순차 실행기."""

import json
import time
from pathlib import Path

from poc.workflow_1 import LOG_DIR
from poc.workflow_3.logger import log_work2_event
from poc.workflow_3.util import make_timestamp_tag
from poc.workflow_1.workflow_config import WorkflowSettings
from poc.workflow_1.workflow_types import (
    ConditionGroup,
    ConditionGroupType,
    ConditionType,
    StepCondition,
    StepResult,
    WorkflowRun,
    WorkflowStep,
)


class ConditionChecker:
    """StepCondition / ConditionGroup 평가기."""

    def __init__(self, context: dict | None = None):
        self.context = context or {}
        self._handlers = {
            ConditionType.ALWAYS: self._check_always,
            ConditionType.WINDOW_VISIBLE: self._check_window_visible,
            ConditionType.WINDOW_FOUND: self._check_window_found,
            ConditionType.WINDOW_APPEARED: self._check_window_appeared,
            ConditionType.DIALOG_DISAPPEARED: self._check_dialog_disappeared,
            ConditionType.PROCESS_ALIVE: self._check_process_alive,
            ConditionType.FIELD_READY_FOR_INPUT: self._check_field_ready_for_input,
            ConditionType.TEXT_APPEARED: self._check_text_appeared,
            ConditionType.TEXT_ALREADY_PRESENT: self._check_text_appeared,
            ConditionType.MASKED_TEXT_PRESENT: self._check_masked_text_present,
        }

    def bind_context(self, context: dict) -> None:
        """새 context 로 평가 대상을 교체한다."""
        self.context = context

    def check_condition(self, condition: StepCondition) -> bool:
        """단일 조건을 평가한다."""
        handler = self._handlers.get(condition.condition_type)
        if handler is None:
            raise ValueError(f"미지원 조건 타입: {condition.condition_type}")
        return handler(condition)

    def check_group(self, group: ConditionGroup | None) -> bool:
        """조건 그룹을 평가한다."""
        if group is None:
            return True

        conditions = group.conditions or []
        if not conditions:
            return True

        if group.group_type == ConditionGroupType.ALL:
            return all(self.check_condition(condition) for condition in conditions)
        return any(self.check_condition(condition) for condition in conditions)

    def _match_title_fragment(self, title: str, fragment: str | None) -> bool:
        """부분 제목 일치 여부를 확인한다."""
        if not fragment:
            return bool(title)
        return fragment.lower() in title.lower()

    def _check_always(self, _condition: StepCondition) -> bool:
        return True

    def _check_window_visible(self, condition: StepCondition) -> bool:
        window = self.context.get("login_window")
        title = str(self.context.get("window_title") or "")
        return window is not None and self._match_title_fragment(title, condition.title_fragment)

    def _check_window_found(self, condition: StepCondition) -> bool:
        return self._check_window_visible(condition)

    def _check_window_appeared(self, condition: StepCondition) -> bool:
        if condition.title_prefix:
            prefix = condition.title_prefix.lower()
            candidates = (
                (
                    self.context.get("post_login_window"),
                    str(self.context.get("post_login_title") or ""),
                ),
                (
                    self.context.get("rcs_main_window"),
                    str(self.context.get("rcs_main_title") or ""),
                ),
            )
            return any(
                window is not None and title.lower().startswith(prefix)
                for window, title in candidates
            )

        return (
            self.context.get("post_login_window") is not None
            or self.context.get("rcs_main_window") is not None
        )

    def _check_dialog_disappeared(self, condition: StepCondition) -> bool:
        if self.context.get("login_window_visible") is False:
            return True
        title = str(self.context.get("window_title") or "")
        if condition.title_fragment:
            return condition.title_fragment.lower() not in title.lower()
        return self.context.get("login_window") is None

    def _check_process_alive(self, condition: StepCondition) -> bool:
        if condition.exe_name:
            running_exe_name = str(self.context.get("process_exe_name") or "")
            if running_exe_name:
                return running_exe_name.lower() == condition.exe_name.lower()
        return bool(self.context.get("process_alive"))

    def _check_field_ready_for_input(self, condition: StepCondition) -> bool:
        target_key = condition.target_key or ""
        return bool(target_key) and self.context.get("focused_target_key") == target_key

    def _check_text_appeared(self, condition: StepCondition) -> bool:
        target_key = condition.target_key or self.context.get("active_target_key")
        if not target_key:
            return False

        typed_values = self.context.get("typed_values", {})
        typed_value = str(typed_values.get(target_key, "") or "")
        if not typed_value:
            return False

        expected_text = condition.expected_text
        if expected_text is None:
            return True
        return typed_value == expected_text

    def _check_masked_text_present(self, condition: StepCondition) -> bool:
        target_key = condition.target_key or self.context.get("active_target_key")
        if not target_key:
            return False
        typed_values = self.context.get("typed_values", {})
        return bool(str(typed_values.get(target_key, "") or ""))


class WorkflowRunner:
    """순차 워크플로 실행기."""

    def __init__(
        self,
        settings: WorkflowSettings,
        *,
        workflow_name: str,
        log_name: str = "workflow_runner",
        component_name: str = "workflow_runner",
    ):
        self.settings = settings
        self.workflow_name = workflow_name
        self.log_name = log_name
        self.component_name = component_name

    def run(self, steps: list[WorkflowStep], context: dict, executor) -> WorkflowRun:
        """step 목록을 순서대로 실행한다."""
        started_at = time.time()
        run_id = make_timestamp_tag(started_at)
        run_dir = LOG_DIR / "workflow_runs" / f"{run_id}_{self.workflow_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        run = WorkflowRun(
            run_id=run_id,
            workflow_name=self.workflow_name,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
            retry_budget_remaining=self.settings.total_retry_budget,
            settings_snapshot=self.settings.to_snapshot(),
            run_dir=str(run_dir),
        )
        context["run_dir"] = run_dir

        checker = ConditionChecker(context)
        step_statuses: dict[str, str] = {}

        log_work2_event(
            component=self.component_name,
            message="workflow_started",
            log_name=self.log_name,
            workflow_name=self.workflow_name,
            run_id=run.run_id,
            safe_mode=self.settings.safe_mode,
        )
        self._write_run_state(run)

        for step_index, step in enumerate(steps):
            run.current_step_index = step_index
            context["current_step_index"] = step_index
            checker.bind_context(context)

            print(
                f"[INFO] step 시작: index={step_index}, step_id={step.step_id}, "
                f"type={step.step_type}, target={step.target_key or '-'}"
            )

            dependency_failure = self._check_dependencies(step, step_statuses)
            if dependency_failure is not None:
                result = dependency_failure
            elif step.skip_if is not None and checker.check_group(step.skip_if):
                result = self._build_result(
                    step=step,
                    status="skipped",
                    started_at=time.time(),
                    safe_mode=self.settings.safe_mode,
                )
            elif not checker.check_group(step.preconditions):
                result = self._build_result(
                    step=step,
                    status="failed",
                    failure_class="precondition_lost",
                    error_message="step preconditions not met",
                    started_at=time.time(),
                    safe_mode=self.settings.safe_mode,
                )
            else:
                step_started_at = time.time()
                result = executor(step, context)
                if result.step_id != step.step_id:
                    result.step_id = step.step_id
                if result.attempt_count <= 0:
                    result.attempt_count = 1
                checker.bind_context(context)
                if result.status == "success" and not checker.check_group(step.success_criteria):
                    result.status = "failed"
                    result.failure_class = result.failure_class or "verify_failed"
                    result.error_message = result.error_message or "success_criteria_not_met"
                    result.elapsed_ms = max(
                        result.elapsed_ms,
                        (time.time() - step_started_at) * 1000,
                    )

            run.step_results.append(result)
            step_statuses[step.step_id] = result.status
            self._write_step_result(run_dir, result)
            self._write_run_state(run)

            if result.status not in {"success", "skipped"}:
                run.status = "aborted" if result.status == "failed" else result.status
                break
        else:
            run.status = "completed"

        run.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self._write_run_state(run)
        log_work2_event(
            component=self.component_name,
            message="workflow_finished",
            log_name=self.log_name,
            workflow_name=self.workflow_name,
            run_id=run.run_id,
            status=run.status,
            step_count=len(run.step_results),
            run_dir=run.run_dir,
        )
        return run

    def _check_dependencies(
        self,
        step: WorkflowStep,
        step_statuses: dict[str, str],
    ) -> StepResult | None:
        """명시적 depends_on 을 검사한다."""
        if not step.depends_on:
            return None

        missing = [
            dep_step_id
            for dep_step_id in step.depends_on
            if step_statuses.get(dep_step_id) not in {"success", "skipped"}
        ]
        if not missing:
            return None

        return self._build_result(
            step=step,
            status="failed",
            failure_class="dependency_failed",
            error_message=f"unsatisfied dependencies: {', '.join(missing)}",
            started_at=time.time(),
            safe_mode=self.settings.safe_mode,
        )

    def _build_result(
        self,
        *,
        step: WorkflowStep,
        status: str,
        started_at: float,
        safe_mode: bool,
        failure_class: str | None = None,
        error_message: str | None = None,
    ) -> StepResult:
        """기본 StepResult 를 생성한다."""
        return StepResult(
            step_id=step.step_id,
            status=status,
            failure_class=failure_class,
            attempt_count=1,
            strategy_used="phase1_direct",
            vlm_service_used="",
            detected_point=None,
            detected_bbox=None,
            screen_point=None,
            verification_result=None,
            before_screenshot=None,
            after_screenshot=None,
            error_message=error_message,
            elapsed_ms=(time.time() - started_at) * 1000,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            safe_mode=safe_mode,
        )

    def _write_run_state(self, run: WorkflowRun) -> None:
        """run_state.json 을 저장한다."""
        run_dir = Path(run.run_dir or "")
        if not run_dir:
            return
        state_path = run_dir / "run_state.json"
        state_path.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_step_result(self, run_dir: Path, result: StepResult) -> None:
        """step 단위 결과 JSON 을 저장한다."""
        step_path = run_dir / f"step_{result.step_id}.json"
        step_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["ConditionChecker", "WorkflowRunner"]
