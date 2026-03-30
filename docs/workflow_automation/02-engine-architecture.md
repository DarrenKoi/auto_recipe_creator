# 워크플로 엔진 아키텍처

## 2.1 Step 조건 타입 시스템

조건(preconditions / success_criteria / skip_if)을 raw dict로 관리하면 오타나 누락 필드가 런타임까지 발견되지 않습니다.
타입 안전을 위해 **원자 조건 + 조건 그룹(enum + dataclass)** 구조를 사용합니다:

```python
from enum import Enum

class ConditionType(Enum):
    """step 조건의 종류."""
    ALWAYS = "always"
    WINDOW_VISIBLE = "window_visible"
    WINDOW_FOUND = "window_found"
    WINDOW_APPEARED = "window_appeared"
    DIALOG_DISAPPEARED = "dialog_disappeared"
    PROCESS_ALIVE = "process_alive"
    FIELD_READY_FOR_INPUT = "field_ready_for_input"
    TEXT_APPEARED = "text_appeared"
    TEXT_ALREADY_PRESENT = "text_already_present"
    MASKED_TEXT_PRESENT = "masked_text_present"


@dataclass
class StepCondition:
    """단일 원자 조건을 정의하는 타입 안전 구조체."""
    condition_type: ConditionType
    title_fragment: str | None = None   # WINDOW_VISIBLE, WINDOW_FOUND, DIALOG_DISAPPEARED
    title_prefix: str | None = None     # WINDOW_APPEARED
    exe_name: str | None = None         # PROCESS_ALIVE
    expected_text: str | None = None    # TEXT_APPEARED, TEXT_ALREADY_PRESENT
    verify_method: str | None = None    # "ocr" | "uia_or_ocr" | "uia_only" | "masked" | "window_title"
    target_key: str | None = None       # FIELD_READY_FOR_INPUT 등에서 대상 지정


class ConditionGroupType(Enum):
    ALL = "all"
    ANY = "any"


@dataclass
class ConditionGroup:
    """여러 원자 조건을 AND / OR 로 묶는 구조."""
    group_type: ConditionGroupType = ConditionGroupType.ALL
    conditions: list[StepCondition] | None = None
```

조건 평가는 `ConditionChecker` 클래스에 집중합니다:

```python
class ConditionChecker:
    """StepCondition / ConditionGroup 을 실제 GUI 상태와 대조하여 평가한다."""

    def check_condition(self, condition: StepCondition) -> bool:
        handler = self._handlers.get(condition.condition_type)
        if handler is None:
            raise ValueError(f"미지원 조건 타입: {condition.condition_type}")
        return handler(condition)

    def check_group(self, group: ConditionGroup) -> bool:
        conditions = group.conditions or []
        if not conditions:
            return True
        results = [self.check_condition(condition) for condition in conditions]
        if group.group_type == ConditionGroupType.ALL:
            return all(results)
        return any(results)

    _handlers: dict  # ConditionType → Callable[[StepCondition], bool]
```

이 구조의 이점:
- **IDE 자동완성**: condition_type에 올 수 있는 값을 즉시 확인 가능
- **누락 필드 조기 발견**: dataclass 초기화 시 필수 필드 검증
- **조건 폭발 방지**: `CREDENTIALS_READY` 같은 step 전용 복합 enum 을 늘리지 않아도 됨
- **재사용성**: 동일 원자 조건을 여러 step에서 조합해 재사용 가능

## 2.2 Step 정의

각 step은 하나의 "관찰 → 판단 → 행동 → 검증" 사이클입니다:

```python
@dataclass
class WorkflowStep:
    """워크플로의 단일 실행 단위."""
    step_id: str                        # 예: "click_userid_input"
    step_type: str                      # "click" | "type" | "double_click" | "scroll" | "verify_only"
    target_description: str             # VLM 프롬프트에 전달할 타겟 설명
    preconditions: ConditionGroup       # 실행 전에 만족해야 하는 조건 묶음
    success_criteria: ConditionGroup    # 실행 후 기대하는 상태 변화 묶음
    skip_if: ConditionGroup | None = None  # 이미 만족되면 step 생략
    safety_tier: int = 2                # 0-3 (doc 04 참조)
    max_retries: int = 3               # 이 step의 최대 재시도 횟수
    retry_profile: str = "default_text_field"  # step 특성별 재시도 프로필
    depends_on: list[str] | None = None # 선행 step_id 목록 (None = 이전 step 성공 필요)
    idempotent: bool = True            # resume / 재실행 시 안전 여부
    timeout_sec: float = 30.0          # 이 step 전체 타임아웃
    detect_timeout_sec: float = 15.0   # VLM 탐지 단계 타임아웃
    act_timeout_sec: float = 5.0       # pynput 액션 단계 타임아웃
    verify_timeout_sec: float = 10.0   # 후행 검증 단계 타임아웃
    reserved_retry_budget: int = 0     # 워크플로 전체 예산에서 이 step에 예약된 최소 재시도 횟수
```

`preconditions` / `success_criteria` 예시:

```python
# 클릭 전 login dialog 가 실제로 보이는지 확인
ConditionGroup(
    conditions=[
        StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In"),
    ],
)

# click 후 dialog 닫힘 확인
ConditionGroup(
    conditions=[
        StepCondition(condition_type=ConditionType.DIALOG_DISAPPEARED, title_fragment="Log In"),
    ],
)

# login 버튼 클릭 전 자격 증명 준비 완료 확인
ConditionGroup(
    group_type=ConditionGroupType.ALL,
    conditions=[
        StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In"),
        StepCondition(condition_type=ConditionType.TEXT_APPEARED, expected_text="2067928", verify_method="ocr"),
        StepCondition(condition_type=ConditionType.MASKED_TEXT_PRESENT, target_key="password_input", verify_method="masked"),
    ],
)

# 새 화면 전환 확인
ConditionGroup(
    conditions=[
        StepCondition(condition_type=ConditionType.WINDOW_APPEARED, title_prefix="Remote Control System"),
    ],
)
```

왜 이렇게 나누는가:

- `depends_on`은 "이전 step이 성공했는가"만 표현합니다
- 실제 GUI 자동화에서는 "지금도 그 성공 상태가 유효한가"를 별도로 확인해야 합니다
- v1에서도 `preconditions` / `skip_if`가 있어야 이미 완료된 step을 안전하게 건너뛸 수 있습니다

## 2.3 Phase별 타임아웃 설계

step 전체에 단일 `timeout_sec`만 두면 VLM 호출이 느려질 때 검증 시간이 부족해집니다.
**phase별 독립 타임아웃**으로 이 문제를 방지합니다:

```
step_timeout_sec = 30.0  (전체 안전장치)
  ├── detect_timeout_sec = 15.0  (VLM 좌표 탐지)
  ├── act_timeout_sec = 5.0      (pynput 클릭/타이핑)
  └── verify_timeout_sec = 10.0  (후행 검증)
```

규칙:
- 각 phase는 자체 타임아웃으로 독립 관리됩니다
- phase 타임아웃 초과 시 해당 phase만 실패 처리하고, failure_class를 기록합니다
- `step_timeout_sec`는 모든 phase + settle 대기를 포함한 최종 안전장치입니다
- VLM 호출 타임아웃은 `detect_timeout_sec`보다 짧아야 합니다 (예: VLM 호출 10초, detect phase 15초)

```python
# phase 타임아웃 적용 예시
def _detect_with_timeout(self, step: WorkflowStep, strategy: str) -> dict | None:
    """VLM 탐지를 phase 타임아웃 내에서 수행한다."""
    try:
        return self._detect(step, strategy, timeout=step.detect_timeout_sec)
    except TimeoutError:
        return None  # failure_class = "detect_timeout"
```

## 2.4 상태 머신

각 step은 다음 상태를 거칩니다:

```
pending → env_check → detecting → acting → settling → verifying → success
              ↓           ↓          ↓                    ↓
           retrying ← retrying ← retrying ←           failed
                                                         ↓
                                                     escalated
```

- `env_check`: 예상치 못한 foreground 윈도우 검사 (Section 2.5 참조)
- `detecting`: VLM으로 타겟 좌표를 찾는 중
- `acting`: pynput으로 클릭/타이핑 수행 중
- `settling`: 액션 후 화면 안정화 대기 (poll-until-stable)
- `verifying`: 후행 검증 (VLM 또는 OCR로 결과 확인)
- `retrying`: 실패 후 다른 전략으로 재시도
- `escalated`: 재시도 예산 소진, 사람에게 에스컬레이션

워크플로 전체 상태:

```
not_started → running → completed
                 ↓
              paused (state 파일 저장, 수동 확인 대기)
                 ↓
              aborted (재시도 예산 소진 또는 안전 규칙 위반)
```

v1에서는 `paused` 상태를 저장하되, 자동 `resume()`까지는 구현 범위에 넣지 않습니다.

추가로 step 결과는 단순 success/fail 외에 **실패 분류(reason class)** 를 가져야 합니다:

- `detect_failed`: 타겟 좌표를 찾지 못함
- `detect_timeout`: VLM 응답이 phase 타임아웃을 초과함
- `act_failed`: 클릭/타이핑 입력 자체가 실패
- `verify_failed`: 액션은 했지만 기대 결과가 확인되지 않음
- `verify_timeout`: 검증 phase 타임아웃 초과
- `window_unstable`: transition frame, occlusion, foreground 흔들림
- `unexpected_foreground`: 예상치 못한 윈도우가 전면에 출현 (Section 2.5)
- `unsafe_to_retry`: safety tier 또는 safe zone 규칙 위반
- `halt_non_idempotent`: idempotent=False step의 검증 실패로 자동 재실행 차단

이 분류가 있어야 재시도가 "고정 ladder"가 아니라 "실패 유형별 routing"으로 바뀝니다.

## 2.5 예상치 못한 Foreground 윈도우 처리

GUI 자동화에서 가장 흔한 실패 원인 중 하나는 **예상하지 못한 윈도우가 전면에 나타나는 것**입니다:
에러 팝업, Windows 업데이트 알림, 보안 경고, 또는 RCS 자체의 모달 dialog 등.

이 문제를 `window_unstable`과 구분하여 `unexpected_foreground`로 별도 분류합니다.

```python
@dataclass
class ForegroundCheckResult:
    """foreground 윈도우 검사 결과."""
    is_expected: bool           # 기대한 윈도우가 foreground인가
    actual_title: str           # 실제 foreground 윈도우 타이틀
    expected_title: str         # 기대한 윈도우 타이틀
    is_known_interrupt: bool    # 알려진 interrupt 패턴인가 (팝업, 알림 등)
```

처리 전략:

```
매 step 의 env_check phase:
  1. foreground 윈도우 타이틀 확인
  2. 기대한 윈도우와 일치? → 정상 진행
  3. 불일치?
     a. 알려진 interrupt 패턴 (에러 팝업, 확인 dialog 등)
        → 자동 닫기 시도 (Escape, 확인 버튼 클릭)
        → 재확인 후 정상 진행 또는 재시도
     b. 알 수 없는 윈도우
        → screenshot 저장 + failure_class="unexpected_foreground"
        → 기대 윈도우 foreground 복구 시도 (SetForegroundWindow)
        → 복구 실패 시 escalation
```

알려진 interrupt 패턴 등록:

```python
KNOWN_INTERRUPT_PATTERNS = [
    {"title_contains": "Error", "action": "click_ok"},
    {"title_contains": "경고", "action": "click_ok"},
    {"title_contains": "Windows Update", "action": "dismiss"},
    {"title_contains": "Security", "action": "dismiss"},
]
```

규칙:
- env_check는 **모든 step의 detect phase 전에** 실행됩니다
- interrupt 자동 처리는 `safety_tier <= 1`인 동작(OK 클릭, Escape)만 허용합니다
- 자동 처리 후에도 기대 윈도우가 foreground가 아니면 escalation합니다
- 모든 interrupt 발생은 로그에 기록합니다

## 2.6 RCS 로그인 워크플로 예시

`action_login.py`의 현재 흐름을 step으로 분해하면:

```python
LOGIN_WORKFLOW = [
    WorkflowStep(
        step_id="ensure_rcs",
        step_type="verify_only",
        target_description="RCS 프로세스 실행 확인",
        preconditions=ConditionGroup(conditions=[StepCondition(condition_type=ConditionType.ALWAYS)]),
        success_criteria=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.PROCESS_ALIVE, exe_name="RcsMainHD.exe")]
        ),
        safety_tier=0,
        max_retries=1,
    ),
    WorkflowStep(
        step_id="find_login_window",
        step_type="verify_only",
        target_description="로그인 창 탐색",
        preconditions=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.PROCESS_ALIVE, exe_name="RcsMainHD.exe")]
        ),
        success_criteria=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.WINDOW_FOUND, title_fragment="Log In")]
        ),
        safety_tier=0,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="click_userid_input",
        step_type="double_click",
        target_description="the editable text field next to the 'User ID' label",
        preconditions=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In")]
        ),
        success_criteria=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, verify_method="uia_or_ocr")]
        ),
        retry_profile="text_field_click",
        safety_tier=2,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="type_userid",
        step_type="type",
        target_description="User ID 입력 필드에 텍스트 입력",
        preconditions=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, target_key="userid_input")]
        ),
        success_criteria=ConditionGroup(
            conditions=[
                StepCondition(
                    condition_type=ConditionType.TEXT_APPEARED,
                    expected_text="2067928",
                    verify_method="ocr",
                ),
            ],
        ),
        skip_if=ConditionGroup(
            conditions=[
                StepCondition(
                    condition_type=ConditionType.TEXT_ALREADY_PRESENT,
                    expected_text="2067928",
                    verify_method="ocr",
                ),
            ],
        ),
        retry_profile="typed_text",
        safety_tier=2,
        max_retries=2,
        depends_on=["click_userid_input"],
    ),
    WorkflowStep(
        step_id="click_password_input",
        step_type="double_click",
        target_description="the editable text field next to the 'Password' label",
        preconditions=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In")]
        ),
        success_criteria=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, verify_method="uia_only")]
        ),
        retry_profile="password_field_click",
        safety_tier=2,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="type_password",
        step_type="type",
        target_description="Password 입력 필드에 텍스트 입력",
        preconditions=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, target_key="password_input")]
        ),
        success_criteria=ConditionGroup(
            conditions=[
                StepCondition(
                    condition_type=ConditionType.MASKED_TEXT_PRESENT,
                    target_key="password_input",
                    verify_method="masked",
                ),
            ],
        ),
        retry_profile="password_type",
        idempotent=False,
        safety_tier=2,
        max_retries=2,
        depends_on=["click_password_input"],
    ),
    WorkflowStep(
        step_id="click_login_button",
        step_type="click",
        target_description="the 'Log In' button at the bottom of the dialog",
        preconditions=ConditionGroup(
            group_type=ConditionGroupType.ALL,
            conditions=[
                StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In"),
                StepCondition(condition_type=ConditionType.TEXT_APPEARED, expected_text="2067928", verify_method="ocr"),
                StepCondition(
                    condition_type=ConditionType.MASKED_TEXT_PRESENT,
                    target_key="password_input",
                    verify_method="masked",
                ),
            ],
        ),
        success_criteria=ConditionGroup(
            conditions=[StepCondition(condition_type=ConditionType.WINDOW_APPEARED, title_prefix="Remote Control System")]
        ),
        retry_profile="dialog_submit",
        safety_tier=2,
        max_retries=2,
        reserved_retry_budget=2,
    ),
]
```

## 2.7 WorkflowRunner

순차 실행기의 핵심 루프:

```python
class WorkflowRunner:
    """워크플로 step을 순서대로 실행하고 검증하는 엔진."""

    def run(self, workflow: list[WorkflowStep]) -> WorkflowRun:
        run = WorkflowRun(workflow_name=..., steps=workflow)

        for step in workflow:
            # 선행 조건 확인
            if not self._check_dependencies(step, run):
                run.abort(f"dependency_failed: {step.depends_on}")
                break

            # skip_if 확인
            if step.skip_if and self._condition_checker.check_group(step.skip_if):
                run.record(StepResult(step_id=step.step_id, status="skipped"))
                continue

            # 재시도 예산 확인
            if not self._has_retry_budget(step, run):
                run.record(StepResult(step_id=step.step_id, status="escalated",
                                      failure_class="workflow_budget_exhausted"))
                run.pause()
                break

            # 실행 + 재시도 루프
            result = self._execute_with_retry(step, run)
            run.record(result)

            if result.status == "escalated":
                run.pause()  # checkpoint 저장
                break

        return run

    def _execute_with_retry(self, step: WorkflowStep, run: WorkflowRun) -> StepResult:
        for attempt in range(step.max_retries + 1):
            # 워크플로 전체 예산 확인
            if not self._has_retry_budget(step, run) and attempt > 0:
                break

            strategy = self._pick_strategy(step, attempt)

            # 0. 예상치 못한 foreground 윈도우 검사
            fg_check = self._check_foreground(step)
            if not fg_check.is_expected:
                recovered = self._handle_unexpected_foreground(fg_check, step)
                if not recovered:
                    return StepResult(
                        step_id=step.step_id, status="escalated",
                        failure_class="unexpected_foreground",
                    )
                # 복구 성공 시 이번 attempt를 처음부터 재시작
                continue

            # 1. 창 안정성 확인
            stability = self._check_window_stability(step)
            if not stability.ok:
                self._handle_unstable_window(step, stability)
                run.increment_retry_count()
                continue

            # 2. 캡처
            before_image = capture_window(self.window)

            # 3. 탐지 (VLM) — phase 타임아웃 적용
            point = self._detect_with_timeout(step, strategy)
            if point is None:
                run.increment_retry_count()
                continue  # 다음 재시도

            # 4. 액션 (pynput) — non-idempotent step 재실행 방어
            if not step.idempotent and attempt > 0:
                return StepResult(
                    step_id=step.step_id, status="escalated",
                    failure_class="halt_non_idempotent",
                    attempt_count=attempt + 1,
                    strategy_used=strategy,
                    needs_manual_check=True,
                    manual_check_reason=(
                        f"step '{step.step_id}'은 idempotent=False이며, "
                        f"이전 attempt의 액션이 이미 수행되었을 수 있습니다. "
                        f"수동으로 현재 상태를 확인한 후 resume 해주세요."
                    ),
                )
            self._act(step, point)

            # 5. 화면 안정화 대기 (poll-until-stable)
            settled = self._wait_until_stable(step)
            if not settled:
                run.increment_retry_count()
                continue

            # 6. 검증 — phase 타임아웃 적용
            verified = self._verify_with_timeout(step, before_image)
            if verified:
                return StepResult(
                    step_id=step.step_id,
                    status="success",
                    failure_class=None,
                    attempt_count=attempt + 1,
                    strategy_used=strategy,
                )

            # 검증 실패는 기본적으로 HALT / escalate 대상이다.
            # 단, 같은 액션을 다시 수행하지 않는 verify-only 재확인만 제한적으로 허용한다.
            if not step.idempotent:
                return StepResult(
                    step_id=step.step_id,
                    status="escalated",
                    failure_class="halt_non_idempotent",
                    attempt_count=attempt + 1,
                    strategy_used=strategy,
                )
            if self._can_retry_after_verify_failure(step, attempt):
                run.increment_retry_count()
                continue
            return StepResult(
                step_id=step.step_id,
                status="escalated",
                failure_class="verify_failed",
                attempt_count=attempt + 1,
                strategy_used=strategy,
            )

        return StepResult(
            step_id=step.step_id,
            status="escalated",
            failure_class="retry_budget_exhausted",
            attempt_count=step.max_retries + 1,
            strategy_used=strategy,
        )
```

기존 모듈과의 연동:

- 설정: `flask_vlm.SHARED_PIPELINE_SETTINGS`에서 서비스 slug, 기본 모델 등 로드
- VLM 호출: `Work2VLMClient(service_slug=...)` 그대로 사용
- 좌표 파싱: `util/json_utils.py`의 `extract_json()`, `parse_coords()` 재사용
- 디버그 저장: `util/debug_image_utils.py` 재사용
- 로깅: `logger.py`의 `log_work2_event()` 확장
