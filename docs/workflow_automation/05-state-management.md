# 워크플로 메모리 / 상태 관리

## 5.1 Per-Step 결과 기록

각 step 실행 결과를 구조화된 데이터로 기록합니다:

```python
@dataclass
class StepResult:
    """단일 step의 실행 결과."""
    step_id: str
    status: str                    # "success" | "failed" | "skipped" | "escalated"
    failure_class: str | None      # "detect_failed" | "verify_failed" | ...
    attempt_count: int
    strategy_used: str             # 마지막 성공/실패 시 사용한 전략
    vlm_service_used: str          # 마지막 시도에서 사용한 VLM 서비스
    detected_point: dict | None    # VLM이 탐지한 이미지 좌표
    screen_point: dict | None      # 실제 클릭한 스크린 좌표
    verification_result: dict | None  # 후행 검증 결과
    before_screenshot: str | None  # 파일 경로
    after_screenshot: str | None   # 파일 경로
    error_message: str | None
    elapsed_ms: float
    timestamp: str
    window_title_before: str | None = None
    window_title_after: str | None = None
    safe_mode: bool = True
    artifact_redacted: bool = False    # 민감정보 마스킹 여부
    needs_manual_check: bool = False   # non-idempotent halt 시 True
    manual_check_reason: str | None = None
```

## 5.2 Workflow Run 상태 파일

`open_rcs_state.json`과 같은 패턴으로 워크플로 전체 상태를 저장합니다:

```
poc/work2/logs/workflow_runs/
  └── 20260328_143022_login/
      ├── run_state.json          # 워크플로 전체 상태
      ├── step_ensure_rcs.json    # step별 결과
      ├── step_click_userid.json
      ├── before_click_userid.jpeg
      ├── after_click_userid.jpeg
      └── ...
```

`run_state.json` 구조:

```json
{
  "run_id": "20260328_143022",
  "workflow_name": "rcs_login",
  "status": "completed",
  "started_at": "2026-03-28T14:30:22",
  "finished_at": "2026-03-28T14:31:45",
  "current_step_index": 6,
  "total_retries_used": 2,
  "retry_budget_remaining": 8,
  "interrupts_encountered": [],
  "step_results": [
    {"step_id": "ensure_rcs", "status": "success", "attempt_count": 1},
    {"step_id": "click_userid_input", "status": "success", "attempt_count": 2, "strategy_used": "jitter"},
    "..."
  ]
}
```

민감정보 처리 규칙:

- password step 은 plaintext 기대값을 저장하지 않습니다
- OCR raw text 는 필요 시 redact 버전과 원본을 분리하고, 기본 분석 경로는 redact 버전을 사용합니다
- screenshot artifact 는 password field 주변을 blur/mask 한 버전을 별도 저장할 수 있습니다

## 5.3 Checkpoint / Resume

워크플로가 중단되었을 때 이어서 진행합니다:

```python
def resume(self, run_state_path: str) -> WorkflowRun:
    """중단된 워크플로를 이어서 실행한다."""
    run = WorkflowRun.load(run_state_path)

    # 완료된 step 건너뛰기
    remaining_steps = run.get_remaining_steps()

    # 중요: 윈도우 상태는 보존되지 않을 수 있음 → 재탐색
    self.window = self._find_window()

    for step in remaining_steps:
        # non-idempotent halt로 중단된 경우 resume_option 확인
        if step.step_id == run.halted_at_step:
            resume_option = run.resume_option  # "skip_this_step" | "retry_from_detect" | "abort"
            if resume_option == "abort":
                run.abort("manual_abort")
                break
            elif resume_option == "skip_this_step":
                run.record(StepResult(step_id=step.step_id, status="skipped"))
                continue
            # "retry_from_detect" → 정상 실행 루프 진입

        result = self._execute_with_retry(step, run)
        run.record(result)
        run.save()  # 매 step 후 저장 (중간 실패 대비)
        ...
```

주의사항:
- resume 시 **윈도우를 다시 찾아야** 합니다 (시간이 지나면 상태가 바뀜)
- 완료된 step의 결과는 유지하되, "이전 step 결과가 여전히 유효한가?"는 검증이 필요할 수 있음
- resume 직후에는 최근 성공 step 자체보다 **다음 step의 precondition**을 다시 검증하는 방식이 안전합니다
- `idempotent=False` step 이 마지막 성공 step 이었다면 단순 skip 보다 수동 확인 절차가 필요할 수 있습니다
- v1에서는 이 기능을 넣지 않고, 상태 저장 포맷만 먼저 준비하는 편이 현실적입니다

## 5.4 실패 이력 기반 적응 (v2)

v1에서는 구현하지 않지만, 구조적으로 가능한 발전:

```python
# 과거 워크플로 실행에서 특정 step + 특정 VLM 조합의 실패율이 높으면
# 다음 워크플로에서 해당 step에 다른 VLM을 우선 시도

failure_history = load_failure_stats("click_userid_input")
# → {"ui-venus": {"attempts": 5, "failures": 3}, "mai-ui": {"attempts": 2, "failures": 0}}
# → mai-ui를 먼저 시도
```

이것은 `StepResult`에 `vlm_service_used`를 기록하는 것만으로 데이터가 축적됩니다.
