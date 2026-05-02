# 워크플로 엔진과 재시도 전략

이 문서는 `poc/work2/`의 GUI 자동화를 단일 스크립트 실행에서 **순차 워크플로 엔진**으로 발전시키는 설계를 다룹니다.
핵심 질문은 세 가지입니다:

1. 순서대로 클릭/입력을 진행하면서 각 단계의 성공 여부를 어떻게 판단하는가?
2. 실패했을 때 변형을 주며 재시도하는 것이 가능한가?
3. 진행 상태를 "기억"하며 이어서 진행할 수 있는가?

결론부터 말하면: **모두 구현 가능하고, 기존 인프라를 상당 부분 재사용할 수 있습니다.**

## 1. 현재 상태 분석

### 1.1 기존 구조의 한계

현재 `poc/work2/`는 독립적인 스크립트를 수동으로 순서대로 실행합니다:

```
open_rcs.py → action_login.py → select_tool.py → ...
```

스크립트 간 상태 공유는 `open_rcs_state.json`의 PID 파일 하나뿐입니다.

누락된 것:

- **상태 머신**: step 간 전이 조건이 코드에 암시적으로만 존재
- **후행 검증**: 클릭 후 "성공했는가?"를 VLM으로 확인하지 않음
- **재시도 오케스트레이션**: VLM 탐지 실패 시 다른 전략으로 재시도하는 구조 없음
- **실패 이력**: 어떤 step이 어떤 이유로 실패했는지 기록하지 않음

### 1.2 `action_login.py` — 워크플로의 원형

`action_login.py`는 이미 순차 워크플로의 원형을 보여줍니다:

```python
# 1. 로그인 창 탐색
login_window, window_title, backend = find_login_window()

# 2. 스크린샷 1회 캡처
shared_image = capture_window(login_window)

# 3. 타겟 순서대로 탐지 + 클릭
for target_key in ACTION_TARGETS:  # ["userid_input", "password_input", "login_button"]
    result = analyze_login_target(login_window, ..., target_config, image=shared_image)
    screen_point = image_point_to_screen(login_window, result.point)
    _click_at_screen(screen_point, target_key)

# 4. 로그인 성공 확인 — 메인 창 대기
rcs_window, rcs_title, _ = wait_for_rcs_main_window()
```

여기서 **빠져 있는 것**:

- step 2에서 VLM 탐지 실패 시 → 다른 모델로 재시도 없음
- step 3에서 클릭 후 → "이 필드에 커서가 들어갔는가?" 검증 없음
- step 4에서 실패 시 → 처음부터 다시 시도하는 구조 없음

이 gap을 채우는 것이 워크플로 엔진의 역할입니다.

## 2. 워크플로 엔진 아키텍처

### 2.1 Step 조건 타입 시스템

조건(precondition / success_condition / skip_condition)을 raw dict로 관리하면 오타나 누락 필드가 런타임까지 발견되지 않습니다.
타입 안전을 위해 **enum + dataclass 기반 조건 타입**을 사용합니다:

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
    PASSWORD_ENTERED = "password_entered"
    CREDENTIALS_READY = "credentials_ready"


@dataclass
class StepCondition:
    """step의 실행/성공/생략 조건을 정의하는 타입 안전 구조체."""
    condition_type: ConditionType
    title_fragment: str | None = None   # WINDOW_VISIBLE, WINDOW_FOUND, DIALOG_DISAPPEARED
    title_prefix: str | None = None     # WINDOW_APPEARED
    exe_name: str | None = None         # PROCESS_ALIVE
    expected_text: str | None = None    # TEXT_APPEARED, TEXT_ALREADY_PRESENT
    verify_method: str | None = None    # "ocr" | "uia_or_ocr" | "uia_only" | "masked"
    target_key: str | None = None       # FIELD_READY_FOR_INPUT 등에서 대상 지정
```

조건 평가는 `ConditionChecker` 클래스에 집중합니다:

```python
class ConditionChecker:
    """StepCondition을 실제 GUI 상태와 대조하여 평가한다."""

    def check(self, condition: StepCondition) -> bool:
        handler = self._handlers.get(condition.condition_type)
        if handler is None:
            raise ValueError(f"미지원 조건 타입: {condition.condition_type}")
        return handler(condition)

    _handlers: dict  # ConditionType → Callable[[StepCondition], bool]
```

이 구조의 이점:
- **IDE 자동완성**: condition_type에 올 수 있는 값을 즉시 확인 가능
- **누락 필드 조기 발견**: dataclass 초기화 시 필수 필드 검증
- **새 조건 타입 추가**: Enum에 추가 → handler 구현 → 끝. 문자열 매칭 분기문이 흩어지지 않음

### 2.2 Step 정의

각 step은 하나의 "관찰 → 판단 → 행동 → 검증" 사이클입니다:

```python
@dataclass
class WorkflowStep:
    """워크플로의 단일 실행 단위."""
    step_id: str                        # 예: "click_userid_input"
    step_type: str                      # "click" | "type" | "double_click" | "scroll" | "verify_only"
    target_description: str             # VLM 프롬프트에 전달할 타겟 설명
    precondition: StepCondition         # 실행 전에 만족해야 하는 조건
    success_condition: StepCondition    # 실행 후 기대하는 상태 변화
    skip_condition: StepCondition | None = None  # 이미 만족되면 step 생략
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

`precondition` / `success_condition` 예시:

```python
# 클릭 전 login dialog 가 실제로 보이는지 확인
StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In")

# click 후 dialog 닫힘 확인
StepCondition(condition_type=ConditionType.DIALOG_DISAPPEARED, title_fragment="Log In")

# type 후 필드에 텍스트 입력 확인
StepCondition(condition_type=ConditionType.TEXT_APPEARED, expected_text="2067928", verify_method="ocr")

# 새 화면 전환 확인
StepCondition(condition_type=ConditionType.WINDOW_APPEARED, title_prefix="Remote Control System")
```

왜 이렇게 나누는가:

- `depends_on`은 "이전 step이 성공했는가"만 표현합니다
- 실제 GUI 자동화에서는 "지금도 그 성공 상태가 유효한가"를 별도로 확인해야 합니다
- resume 시에도 `precondition` / `skip_condition`이 있어야 이미 완료된 step을 안전하게 건너뛸 수 있습니다

### 2.3 Phase별 타임아웃 설계

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

### 2.4 상태 머신

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
              paused (checkpoint 저장) → resumed → running
                 ↓
              aborted (재시도 예산 소진)
```

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

### 2.5 예상치 못한 Foreground 윈도우 처리

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

### 2.6 RCS 로그인 워크플로 예시

`action_login.py`의 현재 흐름을 step으로 분해하면:

```python
LOGIN_WORKFLOW = [
    WorkflowStep(
        step_id="ensure_rcs",
        step_type="verify_only",
        target_description="RCS 프로세스 실행 확인",
        precondition=StepCondition(condition_type=ConditionType.ALWAYS),
        success_condition=StepCondition(condition_type=ConditionType.PROCESS_ALIVE, exe_name="RcsMainHD.exe"),
        safety_tier=0,
        max_retries=1,
    ),
    WorkflowStep(
        step_id="find_login_window",
        step_type="verify_only",
        target_description="로그인 창 탐색",
        precondition=StepCondition(condition_type=ConditionType.PROCESS_ALIVE, exe_name="RcsMainHD.exe"),
        success_condition=StepCondition(condition_type=ConditionType.WINDOW_FOUND, title_fragment="Log In"),
        safety_tier=0,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="click_userid_input",
        step_type="double_click",
        target_description="the editable text field next to the 'User ID' label",
        precondition=StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In"),
        success_condition=StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, verify_method="uia_or_ocr"),
        retry_profile="text_field_click",
        safety_tier=2,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="type_userid",
        step_type="type",
        target_description="User ID 입력 필드에 텍스트 입력",
        precondition=StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, target_key="userid_input"),
        success_condition=StepCondition(condition_type=ConditionType.TEXT_APPEARED, expected_text="2067928", verify_method="ocr"),
        skip_condition=StepCondition(condition_type=ConditionType.TEXT_ALREADY_PRESENT, expected_text="2067928", verify_method="ocr"),
        retry_profile="typed_text",
        safety_tier=2,
        max_retries=2,
        depends_on=["click_userid_input"],
    ),
    WorkflowStep(
        step_id="click_password_input",
        step_type="double_click",
        target_description="the editable text field next to the 'Password' label",
        precondition=StepCondition(condition_type=ConditionType.WINDOW_VISIBLE, title_fragment="Log In"),
        success_condition=StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, verify_method="uia_only"),
        retry_profile="password_field_click",
        safety_tier=2,
        max_retries=3,
    ),
    WorkflowStep(
        step_id="type_password",
        step_type="type",
        target_description="Password 입력 필드에 텍스트 입력",
        precondition=StepCondition(condition_type=ConditionType.FIELD_READY_FOR_INPUT, target_key="password_input"),
        success_condition=StepCondition(condition_type=ConditionType.PASSWORD_ENTERED, verify_method="masked"),
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
        precondition=StepCondition(condition_type=ConditionType.CREDENTIALS_READY),
        success_condition=StepCondition(condition_type=ConditionType.WINDOW_APPEARED, title_prefix="Remote Control System"),
        retry_profile="dialog_submit",
        safety_tier=2,
        max_retries=2,
        reserved_retry_budget=2,
    ),
]
```

### 2.7 WorkflowRunner

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

            # skip_condition 확인
            if step.skip_condition and self._condition_checker.check(step.skip_condition):
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
            run.increment_retry_count()

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

## 3. VLM 기반 후행 검증 (Post-Action Verification)

### 3.1 현재 유일한 검증 패턴

현재 `action_login.py`의 유일한 후행 검증:

```python
# 4. 로그인 성공 확인 — 메인 RCS 창 대기
rcs_window, rcs_title, _ = wait_for_rcs_main_window()
login_verified = rcs_window is not None
```

이것은 **pywinauto 윈도우 타이틀 체크**입니다.
VLM을 사용한 시각적 검증은 없습니다.

### 3.2 Before/After 비교 전략

핵심 아이디어: **액션 전후 스크린샷을 VLM에 보내서 기대한 변화가 일어났는지 확인한다.**

```
[before 캡처] → [액션 수행] → [poll-until-stable] → [after 캡처] → [VLM 검증]
```

중요 규칙:

- after 캡처는 반드시 **액션 수행 후 새로 찍어야** 합니다 (stale 방지, doc 04 참조)
- before/after 둘 다 VLM에 전송하되, 질문은 **구체적**이어야 합니다
- "무엇이 바뀌었나?"가 아니라 "이 특정 변화가 일어났는가?"로 질문합니다
- before/after 비교 전에 frame 이 stable 한지 먼저 확인합니다
- unstable frame 이면 verify 실패가 아니라 `window_unstable`로 분류하고 recapture 먼저 시도합니다

#### Poll-Until-Stable 안정화 대기

고정 sleep (`POST_ACTION_SETTLE`) 대신 **poll-until-stable** 방식을 사용합니다.
RCS 같은 레거시 Windows 앱은 렌더링 속도가 가변적이므로 고정 sleep은 너무 짧거나(불안정) 너무 길어(느림) 문제가 됩니다.

```python
def _wait_until_stable(
    self,
    step: WorkflowStep,
    max_wait_sec: float = 3.0,
    poll_interval_sec: float = 0.3,
    similarity_threshold: float = 0.98,
) -> bool:
    """연속 캡처 2장의 유사도가 threshold를 넘을 때까지 대기한다.

    Returns:
        True: 화면이 안정됨
        False: max_wait_sec 내에 안정화되지 않음
    """
    prev_hash = self._capture_and_hash(self.window)
    elapsed = 0.0

    while elapsed < max_wait_sec:
        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec

        curr_hash = self._capture_and_hash(self.window)
        similarity = self._compare_hashes(prev_hash, curr_hash)

        if similarity >= similarity_threshold:
            return True
        prev_hash = curr_hash

    return False
```

구현 옵션 (가벼운 순서):
1. **pHash (perceptual hash)**: `imagehash` 라이브러리, 해밍 거리 비교. 가장 빠름
2. **Pixel histogram 비교**: `PIL.Image.histogram()` 차이. 전역 변화 감지에 적합
3. **SSIM**: `skimage.metrics.structural_similarity`. 정확하지만 느림. v1에서는 불필요

v1 권장: pHash 사용. 설치 부담 없고(`imagehash`), 충분히 정확합니다.

규칙:
- `max_wait_sec`는 step별로 조정 가능합니다 (scroll step은 길게, click step은 짧게)
- 안정화 실패 시 `window_unstable`로 분류하고, 재캡처 후 재시도합니다
- 최소 1회는 반드시 대기합니다 (poll_interval_sec 만큼)

### 3.3 Step Type별 검증 프롬프트

| step_type | 검증 질문 | 검증 수단 |
|-----------|----------|----------|
| `click` (버튼) | "이전에 보이던 dialog가 사라졌는가?" 또는 "버튼이 pressed 상태인가?" | VLM before/after |
| `click` (필드) | "입력 준비 상태가 되었는가?" | UIA/pywinauto 우선, 보조로 OCR/VLM |
| `type` | "입력 필드에 기대한 텍스트가 보이는가?" | OCR (가장 확실) |
| `double_click` | "새로운 화면이 열렸는가?" | VLM + 윈도우 타이틀 체크 |
| `scroll` | "리스트 내용이 변경되었는가?" | VLM before/after |

주의:

- `field_focused`를 VLM 단독으로 판정하는 것은 권장하지 않습니다
- Windows UIA 정보가 있으면 먼저 사용하고, 없을 때만 OCR/VLM을 보조 증거로 씁니다
- password field 는 plaintext OCR 검증 대신 "입력 마스크 개수 증가", "dialog 상태 변화", "submit 후 성공 여부"를 더 신뢰합니다

새 프롬프트 빌더가 필요합니다:

```python
# poc/work2/prompts/prompt_action_verify.py

def build_action_verification_prompt(
    step_type: str,
    target_description: str,
    success_condition: StepCondition,
) -> tuple[str, str]:
    """액션 후행 검증 프롬프트를 생성한다.

    Returns:
        (system_message, user_message) 튜플
    """
    # VLM 응답 형식: {"verified": true/false, "confidence": 0.0-1.0, "reason": "..."}
```

### 3.4 검증 판정 규칙

```
confidence >= 0.8 AND verified == true   → 성공
confidence >= 0.6 AND verified == true   → 약한 성공, 로그 경고, 1회 재검증 가능
verified == false OR confidence < 0.6    → 실패, 재시도 진입
타임아웃 (응답 없음)                       → 실패 (failure_class = "verify_timeout")
```

보강 규칙:

- `type` step 에서 입력 후 검증이 약하면 먼저 `verify_only` 재캡처를 1회 수행합니다
- 입력이 실제로 반영되었는지 불확실할 때는 곧바로 같은 문자열을 다시 타이핑하지 않습니다
- `idempotent=False` step 은 검증 실패 시 기본값이 재실행이 아니라 `halt_non_idempotent` 입니다 (Section 4.8 참조)

### 3.5 VLM 실현 가능성 분석

**VLM이 잘 판단하는 것:**

- dialog 출현/소멸 (큰 시각적 변화)
- 새 윈도우 출현 (타이틀 바 + 콘텐츠 변화)
- 버튼/탭의 selected vs unselected 상태
- 입력 필드의 텍스트 존재 여부 (특히 OCR sidecar 병행 시)
- 에러 팝업, 모달 dialog
- 스크롤 후 리스트 내용 변화

**VLM이 어려워하는 것:**

- 커서 깜빡임 (focus 여부 판단)
- 미세한 hover 효과
- 비시각적 상태 변화 (네트워크 상태, 백엔드 데이터)
- 애니메이션 중간 프레임
- 밀도 높은 engineering UI의 작은 체크박스 상태

**완화 전략:**

- poll-until-stable 대기 후 캡처 (고정 sleep 대신)
- OCR cross-check로 텍스트 검증 보강
- 반복 캡처로 일시적 상태 필터링
- structured JSON 응답 + confidence 강제로 hallucination 감소
- 검증이 어려운 step은 `verify_method: "window_title"` 또는 `"skip"`으로 대체 가능

**기존 precedent:**
`poc/work2/prompts/prompt_screen_analysis.py`의 `build_measurement_judgment_prompt()`가 이미 VLM 기반 성공/실패 판정을 수행하고 있습니다. 같은 패턴을 워크플로 검증에 적용 가능합니다.

### 3.6 Hybrid 검증 — 비용 대비 효과

모든 step에 VLM 검증을 걸면 비용과 시간이 과합니다. Hybrid 접근:

| safety_tier | 검증 수단 | 이유 |
|-------------|----------|------|
| Tier 0 | pywinauto 윈도우 타이틀 체크 | VLM 불필요, 빠름 |
| Tier 1 | OCR only (텍스트 필드) | 정확하고 빠름 |
| Tier 2 | VLM before/after 비교 | UI 상태 변화 판단 필요 |
| Tier 3 | VLM + OCR + 사람 확인 | 장비 영향 있는 액션 |

실무 규칙:

- 가능한 경우 "가장 싼 검증"을 먼저 씁니다
- window title / UIA / OCR 로 충분한 step 에는 VLM을 붙이지 않습니다
- VLM 검증은 "시각적 상태 변화가 핵심인 step"에만 집중합니다

## 4. 재시도 전략 (Retry with Variation)

### 4.1 Escalation Ladder

실패 시 단계적으로 전략을 바꿔가며 재시도한다는 방향은 맞습니다.
다만 **모든 실패에 동일한 ladder**를 적용하면 불필요한 액션이 늘어납니다.

권장 방식은 "고정 ladder"보다 **실패 유형별 retry routing** 입니다:

| failure_class | 첫 대응 | 다음 대응 |
|--------------|---------|----------|
| `window_unstable` | recapture only | foreground / overlap recovery |
| `unexpected_foreground` | interrupt 자동 처리 | foreground 복구 → escalation |
| `detect_failed` | same detect retry | crop-retry / model fallback / OCR anchor |
| `detect_timeout` | 같은 모델 1회 재시도 | 더 빠른 모델로 fallback |
| `verify_failed` | verify-only recapture | limited jitter or alternative verification |
| `verify_timeout` | 검증 방식 변경 (VLM → OCR) | 검증 생략 + 경고 로그 |
| `act_failed` | window refocus | same action 1회만 재시도 |
| `halt_non_idempotent` | 즉시 halt | 사람 에스컬레이션 |
| `unsafe_to_retry` | 즉시 halt | 사람 에스컬레이션 |

즉, ladder 는 "전부 순서대로"가 아니라 **해당 failure_class 에서 허용된 subset**만 사용해야 합니다.

예시:

```
detect_failed       → recapture → crop-retry → model fallback → OCR anchor → escalate
detect_timeout      → same model retry → faster model → escalate
verify_failed       → recapture+verify_only → alternative verifier → limited jitter → escalate
verify_timeout      → switch verify method → skip verify with warning → escalate
window_unstable     → recapture → foreground → doc 07 recovery → escalate
unexpected_foreground → auto-dismiss → foreground recovery → escalate
halt_non_idempotent → 즉시 escalate (자동 재실행 금지)
```

그래도 설명 편의를 위해 기본 ladder 를 아래처럼 둘 수 있습니다:

```
Level 0: recapture 또는 동일 재시도
   ↓ 실패
Level 1: verify-only recapture / foreground recovery
   ↓ 실패
Level 2: 좌표 jitter 또는 Crop-retry zoom
   ↓ 실패
Level 3: 대체 VLM 모델
   ↓ 실패
Level 4: OCR cross-validation
   ↓ 실패
Level 5: 사람 에스컬레이션
```

### 4.2 Level 0: 동일 재시도

- 같은 파라미터로 재캡처 + 재탐지
- 일시적인 화면 상태 변화(transition frame)로 인한 실패를 커버
- poll-until-stable 대기 후 재캡처이므로 exponential backoff 불필요
- 단, `verify_failed`에서는 "같은 액션 재수행"보다 `verify_only` 재캡처를 먼저 시도합니다

### 4.3 Level 1: 좌표 Jitter

VLM이 반환한 좌표가 약간 빗나간 경우를 커버합니다.

```python
def jitter_point(
    point: dict,
    offset: int = 5,
    element_bbox: dict | None = None,
    safe_zone: dict | None = None,
) -> list[dict]:
    """원점 주변 십자 패턴으로 후보 좌표를 생성한다.

    Args:
        point: VLM이 반환한 원래 좌표 {"x": int, "y": int}
        offset: jitter 거리 (px)
        element_bbox: VLM이 반환한 요소의 bounding box {"x1", "y1", "x2", "y2"} (있으면)
        safe_zone: doc 04의 safe zone 범위 {"x1", "y1", "x2", "y2"}

    Returns:
        safe zone과 element bbox 내에 있는 후보 좌표 목록
    """
    x, y = point["x"], point["y"]
    candidates = [
        {"x": x + offset, "y": y},   # 오른쪽
        {"x": x - offset, "y": y},   # 왼쪽
        {"x": x, "y": y + offset},   # 아래
        {"x": x, "y": y - offset},   # 위
    ]

    valid = []
    for c in candidates:
        # element bbox가 있으면 요소 영역 내인지 확인
        if element_bbox and not _point_in_bbox(c, element_bbox):
            continue
        # safe zone 범위 내인지 확인
        if safe_zone and not _point_in_bbox(c, safe_zone):
            continue
        valid.append(c)

    return valid
```

규칙:
- jitter 후보는 **doc 04의 safe zone 범위 내**에 있어야 합니다
- VLM이 bounding box를 반환했다면 **element bbox 내**로도 제한합니다
- bbox 없이 safe zone만으로 제한하면 작은 체크박스 등에서 요소 밖을 클릭할 수 있습니다
- 십자(cross) 패턴이 random보다 체계적입니다
- offset 기본값: 5px (RCS login dialog 기준으로 충분)
- jitter 는 `detect_failed` 또는 `verify_failed` 중 "오차가 작은 클릭성 control"에만 적용합니다
- `type_password` 같은 비가역 step 에는 기본적으로 jitter 재실행을 적용하지 않습니다

### 4.4 Level 2: Crop-Retry Zoom

기존 `ui_venus_mai_locator.py`의 2단계 파이프라인을 재시도 전략으로 활용합니다.

현재 구조:

```
full-screen VLM (ui-venus) → coarse bbox → crop → zoomed VLM (mai-ui) → refined point
```

재시도 시 변형:
- **padding ratio 확대**: `left_pad_ratio`, `right_pad_ratio`, `vertical_pad_ratio`를 1.5배 늘려서 더 넓은 영역을 crop
- **crop 중심 이동**: OCR이 찾은 텍스트 앵커 위치 기준으로 crop 영역 재설정
- **최소 crop 크기 확대**: `min_crop_width`, `min_crop_height`를 2배로

이 전략은 "타겟이 있는 것은 맞는데 정확한 위치를 못 찾았을 때" 가장 효과적입니다.

### 4.5 Level 3: 대체 VLM 모델 Fallback

기존 인프라가 이미 이것을 지원합니다:

```python
# 모델 교체는 service_slug 하나만 바꾸면 됨
client_venus = Work2VLMClient(service_slug="ui-venus")
client_mai = Work2VLMClient(service_slug="mai-ui")
```

fallback 순서 예시 (설정 가능):

```python
# flask_vlm.SHARED_PIPELINE_SETTINGS 에 추가
"workflow_service_fallback_order": ["ui-venus", "mai-ui", "kimi-k2.5"],
```

각 모델의 특성:
- `ui-venus`: 단일 요소 grounding에 최적화, `[x,y]` 0-1000 좌표
- `mai-ui`: crop + zoom 후 정밀 grounding에 강함
- `kimi-k2.5`: 범용 VLM, direct 연결

같은 스크린샷에서 한 모델이 실패해도 다른 모델이 성공하는 경우가 `login_benchmark.py`에서 이미 관측되었습니다.

### 4.6 Level 4: OCR Cross-Validation

VLM grounding 전에 OCR로 텍스트 앵커를 먼저 확보합니다.

```
OCR 전체 화면 → 타겟 텍스트 위치 발견 → 해당 영역만 crop → VLM grounding
```

이미 구현된 패턴:
- `select_tool.py`의 OCR pre-check
- `ocr_login_check.py`의 PaddleOCR-VL 텍스트 추출

효과:
- VLM의 탐색 범위를 좁혀서 정확도 향상
- 텍스트 기반 앵커는 VLM 좌표보다 안정적

운영 규칙:

- text-labeled control 은 OCR anchor 를 model fallback 보다 먼저 쓸 수도 있습니다
- icon-only control 은 OCR 단계가 무의미할 수 있으므로 retry profile 에서 비활성화할 수 있습니다

### 4.7 Level 5: 사람 에스컬레이션

모든 자동 재시도가 소진되면:

```python
print("[ESCALATION] step=click_userid_input")
print("[ESCALATION] 시도 횟수: 4, 사용 전략: ladder_full")
print("[ESCALATION] 증거 경로: poc/work2/logs/workflow_runs/20260328_143022/")
print("[ESCALATION] 워크플로를 일시 중지합니다. 수동 확인 후 resume 가능합니다.")
```

에스컬레이션 시 저장하는 증거:
- 모든 시도의 before/after 스크린샷
- 각 시도의 VLM raw 응답
- 탐지된 좌표와 실제 클릭 좌표
- 재시도 전략별 결과

### 4.8 Non-Idempotent Step 처리

`idempotent=False` step (예: `type_password`, 장비 제어 명령)은 검증 실패 시 자동 재실행이 위험합니다.

**문제 시나리오:**
1. password 입력 → 클릭은 성공 → 타이핑 수행 → 검증 실패 (OCR이 마스킹된 텍스트를 읽지 못함)
2. 자동 재실행 시 → 기존 입력에 추가로 타이핑 → password가 이중 입력됨

**처리 규칙:**

```python
# idempotent=False step의 재시도 정책
if not step.idempotent:
    # 1. detect phase 실패 → 재시도 허용 (아직 액션을 수행하지 않았으므로)
    # 2. act phase 이후 verify 실패 → 즉시 HALT
    # 3. HALT 시 구체적인 상황 정보를 제공
    pass
```

HALT 시 제공하는 정보:

```python
@dataclass
class NonIdempotentHalt:
    """idempotent=False step의 HALT 정보."""
    step_id: str
    last_action_performed: str       # "typed 8 characters" 등
    verify_result: dict | None       # 검증 시도 결과
    before_screenshot: str           # HALT 시점 스크린샷 경로
    suggested_manual_checks: list[str]  # ["필드에 텍스트가 이미 입력되어 있는지 확인",
                                        #  "입력된 텍스트가 올바른지 확인"]
    resume_options: list[str]        # ["skip_this_step", "retry_from_detect", "abort"]
```

resume 시 선택지:
- `skip_this_step`: 수동 확인 후 이미 성공한 것으로 처리
- `retry_from_detect`: 필드를 초기화(Ctrl+A → Delete)한 후 처음부터 재시도
- `abort`: 워크플로 중단

### 4.9 재시도 예산

무한 재시도를 방지합니다:

```python
# Per-step 제한
max_retries: int = 3  # WorkflowStep 기본값

# Per-workflow 전체 제한
WORKFLOW_TOTAL_RETRY_BUDGET = 10

# 예산 소진 시 → 워크플로 중단, 에스컬레이션
```

#### 예산 상호작용 규칙

per-step 예산과 workflow 전체 예산의 관계를 명확히 정의합니다:

```python
def _has_retry_budget(self, step: WorkflowStep, run: WorkflowRun) -> bool:
    """이 step이 재시도 예산을 사용할 수 있는지 확인한다."""
    remaining_workflow = WORKFLOW_TOTAL_RETRY_BUDGET - run.total_retries_used

    # 예약된 예산이 있는 미래 step들의 예약량 합산
    reserved_for_future = sum(
        s.reserved_retry_budget
        for s in run.get_remaining_steps()
        if s.step_id != step.step_id
    )

    # 이 step이 사용할 수 있는 예산 = 남은 예산 - 미래 예약분
    available_for_this_step = remaining_workflow - reserved_for_future

    return available_for_this_step > 0
```

설계 근거:
- 앞쪽 step이 예산을 모두 소진하면 뒤쪽 critical step(예: `click_login_button`)이 재시도할 수 없는 문제를 방지합니다
- `reserved_retry_budget`로 critical step에 최소 재시도 횟수를 보장합니다
- 예약되지 않은 잔여 예산은 선착순으로 사용합니다

추가 권장:

- `idempotent=False` step은 `max_retries`를 낮게 설정합니다 (0 또는 1)
- password 입력, destructive submit, 장비 영향 step 은 별도 예산으로 제한합니다

## 5. 워크플로 메모리 / 상태 관리

### 5.1 Per-Step 결과 기록

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

### 5.2 Workflow Run 상태 파일

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

### 5.3 Checkpoint / Resume

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

### 5.4 실패 이력 기반 적응 (v2)

v1에서는 구현하지 않지만, 구조적으로 가능한 발전:

```python
# 과거 워크플로 실행에서 특정 step + 특정 VLM 조합의 실패율이 높으면
# 다음 워크플로에서 해당 step에 다른 VLM을 우선 시도

failure_history = load_failure_stats("click_userid_input")
# → {"ui-venus": {"attempts": 5, "failures": 3}, "mai-ui": {"attempts": 2, "failures": 0}}
# → mai-ui를 먼저 시도
```

이것은 `StepResult`에 `vlm_service_used`를 기록하는 것만으로 데이터가 축적됩니다.

## 6. 구현 청사진

### 6.1 새 모듈

| 모듈 | 역할 |
|------|------|
| `poc/work2/workflow_types.py` | `WorkflowStep`, `StepCondition`, `ConditionType`, `StepResult`, `WorkflowRun` dataclass |
| `poc/work2/workflow_runner.py` | 순차 실행기, 재시도 루프, checkpoint, foreground 검사 |
| `poc/work2/workflow_verify.py` | 후행 검증 로직 (VLM/OCR/UIA/윈도우 타이틀), poll-until-stable |
| `poc/work2/workflow_retry.py` | 재시도 전략 (jitter, crop-zoom, model fallback, failure routing) |
| `poc/work2/workflow_conditions.py` | `ConditionChecker` — 조건 타입 평가 로직 |
| `poc/work2/prompts/prompt_action_verify.py` | 검증용 VLM 프롬프트 빌더 |
| `poc/work2/workflow_login.py` | RCS 로그인 워크플로 정의 (action_login.py의 워크플로 버전) |

v1에서는 모듈을 더 줄여도 됩니다:

- `workflow_types.py` (`StepCondition`, `ConditionType` 포함)
- `workflow_runner.py` (`ConditionChecker` 내장)
- `workflow_login.py`

즉, 검증/재시도 로직도 처음에는 `workflow_runner.py` 내부 private helper 로 시작해도 괜찮습니다.
실제 login workflow 1개가 안정된 뒤에 모듈 분리를 해도 늦지 않습니다.

### 6.2 기존 모듈 연동

변경이 **불필요한** 모듈 (그대로 사용):
- `vlm_client.py` — `Work2VLMClient`가 이미 임의 `service_slug` 지원
- `util/json_utils.py` — `extract_json()`, `parse_coords()` 그대로 사용
- `util/image_utils.py` — `capture_window()`, `encode_image_webp()` 그대로 사용
- `util/debug_image_utils.py` — 디버그 아티팩트 저장 그대로 사용
- `ui_venus_mai_locator.py` — 2단계 파이프라인 호출 그대로 사용

설정 추가가 필요한 모듈:
- `flask_vlm.py` — `SHARED_PIPELINE_SETTINGS`에 워크플로 관련 기본값 추가

```python
# flask_vlm.py SHARED_PIPELINE_SETTINGS 에 추가
"workflow_verify_service": "paddleocr-vl-1.5",      # 검증용 OCR 서비스
"workflow_service_fallback": ["ui-venus", "mai-ui"],  # 모델 fallback 순서
"workflow_total_retry_budget": 10,                    # 전체 재시도 예산
"workflow_settle_max_wait_sec": 3.0,                  # poll-until-stable 최대 대기
"workflow_settle_similarity_threshold": 0.98,          # 안정화 판정 임계값
```

### 6.3 단계별 구현 순서

```
Phase 1: dataclass + 골격
   WorkflowStep, StepCondition, ConditionType, StepResult, WorkflowRun dataclass 정의
   ConditionChecker 기본 구현
   WorkflowRunner 골격 (순차 실행만, 검증/재시도 없음)
   action_login.py 로직을 workflow_login.py로 매핑

Phase 2: 안정성 게이트 + 후행 검증
   foreground 검사 (unexpected_foreground 감지 + 알려진 interrupt 자동 처리)
   poll-until-stable 구현 (pHash 기반)
   prompt_action_verify.py 프롬프트 빌더
   workflow_verify.py 검증 로직
   click/type step에 UIA/OCR/VLM hybrid 검증 연결

Phase 3: failure-aware 재시도
   workflow_retry.py 전략 구현
   failure_class 분류 + retry routing
   jitter (element bbox 경계 검사 포함) + model fallback 통합
   재시도 예산 관리 (reserved_retry_budget 포함)
   non-idempotent step HALT 처리

Phase 4: crop-retry zoom + OCR cross-validation
   ui_venus_mai_locator.py 2단계 파이프라인을 재시도 전략으로 연결
   OCR pre-check 통합

Phase 5: 상태 저장 + checkpoint/resume
   WorkflowRun JSON 저장/로드
   resume 로직 (non-idempotent halt resume_option 포함)

Phase 6: 실패 이력 기반 적응 (v2)
   실패 통계 집계
   VLM 서비스 자동 선택
```

## 7. Safety Rules

- 검증 실패의 기본 동작은 **중단(HALT)**이지, 예산 초과 재시도가 아닙니다
- `idempotent=False` step은 act phase 이후 검증 실패 시 **즉시 HALT**합니다 (자동 재실행 금지)
- jitter 좌표는 doc 04의 safe zone **및** element bounding box 범위 내에 있어야 합니다
- 모델 fallback은 safety tier 검사를 우회하지 않습니다
- 사람 에스컬레이션이 최종 fallback이며, 무한 재시도는 없습니다
- 모든 재시도는 **전체 증거 trail**을 남깁니다 (before/after 스크린샷, VLM 응답, 좌표)
- Tier 3 step은 액션 **전과 후** 모두 검증이 필요합니다
- `SAFE_MODE` 토글은 워크플로 레벨에서도 존중됩니다
- password / credential step 의 artifact 와 로그는 기본적으로 redact 합니다
- unstable / occluded frame 은 "모델 실패"가 아니라 "입력 화면 품질 실패"로 취급합니다
- 매 step 전 **foreground 윈도우 검사**를 수행하여 예상치 못한 윈도우를 감지합니다
- 알려진 interrupt(에러 팝업, 시스템 알림)는 자동 처리하되, `safety_tier <= 1` 동작만 허용합니다

## 8. v1 범위

**v1에 포함:**
- `WorkflowStep` / `StepCondition` / `ConditionType` / `StepResult` / `WorkflowRun` dataclass
- `ConditionChecker` 기반 타입 안전 조건 평가
- 순차 실행기 (dependency 체크 포함)
- precondition / success_condition / skip_condition 기반 step 계약
- phase별 타임아웃 (detect / act / verify 독립 관리)
- foreground 윈도우 검사 + 알려진 interrupt 자동 처리
- poll-until-stable 안정화 대기 (pHash 기반)
- hybrid 후행 검증 (window title / UIA / OCR 우선, 필요한 곳만 VLM)
- failure-aware 재시도: recapture, jitter (bbox 경계 포함), model fallback
- non-idempotent step HALT 처리 (act 이후 자동 재실행 차단)
- per-step + per-workflow 재시도 예산 (reserved_retry_budget 포함)
- per-step 결과 JSON 로깅
- RCS 로그인 워크플로 구현

**v1에 미포함:**
- Checkpoint / Resume (상태 저장 포맷만 준비)
- 실패 이력 기반 서비스 자동 선택
- 병렬 step 실행
- VLM 기반 동적 워크플로 생성
- doc 06 align-fail 모니터링 통합
- doc 07 occlusion recovery 통합
- password artifact masking 자동화 고도화
