# 재시도 전략 (Retry with Variation)

## 4.1 Escalation Ladder

실패 시 단계적으로 전략을 바꿔가며 재시도한다는 방향은 맞습니다.
다만 **모든 실패에 동일한 ladder**를 적용하면 불필요한 액션이 늘어납니다.

권장 방식은 "고정 ladder"보다 **실패 유형별 retry routing** 입니다:

| failure_class | 첫 대응 | 다음 대응 |
|--------------|---------|----------|
| `window_unstable` | recapture only | foreground / overlap recovery |
| `unexpected_foreground` | interrupt 자동 처리 | foreground 복구 → escalation |
| `detect_failed` | same detect retry | crop-retry / model fallback / OCR anchor |
| `detect_timeout` | 같은 모델 1회 재시도 | 더 빠른 모델로 fallback |
| `verify_failed` | verify-only recapture | alternative verification → escalation |
| `verify_timeout` | 검증 방식 변경 (VLM → OCR) | verify-only recapture → escalation |
| `act_failed` | window refocus | same action 1회만 재시도 |
| `halt_non_idempotent` | 즉시 halt | 사람 에스컬레이션 |
| `unsafe_to_retry` | 즉시 halt | 사람 에스컬레이션 |

즉, ladder 는 "전부 순서대로"가 아니라 **해당 failure_class 에서 허용된 subset**만 사용해야 합니다.

예시:

```
detect_failed       → recapture → crop-retry → model fallback → OCR anchor → escalate
detect_timeout      → same model retry → faster model → escalate
verify_failed       → recapture+verify_only → alternative verifier → escalate
verify_timeout      → switch verify method → verify_only recapture → escalate
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

## 4.2 Level 0: 동일 재시도

- 같은 파라미터로 재캡처 + 재탐지
- 일시적인 화면 상태 변화(transition frame)로 인한 실패를 커버
- poll-until-stable 대기 후 재캡처이므로 exponential backoff 불필요
- 단, `verify_failed`에서는 "같은 액션 재수행"보다 `verify_only` 재캡처를 먼저 시도합니다

## 4.3 Level 1: 좌표 Jitter

VLM이 반환한 좌표가 약간 빗나간 경우를 커버합니다.

전제:

```python
@dataclass
class DetectionResult:
    point: dict | None
    element_bbox: dict | None
    service_slug: str
    raw_response_path: str | None
```

`jitter`는 `DetectionResult.element_bbox`가 있을 때만 활성화합니다.
bbox가 없으면 같은 액션을 더 위험하게 반복하는 셈이므로 `crop-retry` 또는 `model fallback`으로 바로 넘어갑니다.

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
- **element bbox 가 없으면 jitter 자체를 수행하지 않습니다**
- 십자(cross) 패턴이 random보다 체계적입니다
- offset 기본값: 5px (RCS login dialog 기준으로 충분)
- jitter 는 `detect_failed` 또는 `verify_failed` 중 "오차가 작은 클릭성 control"에만 적용합니다
- `type_password` 같은 비가역 step 에는 기본적으로 jitter 재실행을 적용하지 않습니다

## 4.4 Level 2: Crop-Retry Zoom

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

## 4.5 Level 3: 대체 VLM 모델 Fallback

기존 인프라가 이미 이것을 지원합니다:

```python
# 모델 교체는 service_slug 하나만 바꾸면 됨
client_venus = Work2VLMClient(service_slug="ui-venus")
client_mai = Work2VLMClient(service_slug="mai-ui")
```

fallback 순서 예시 (설정 가능):

```python
# WorkflowSettings 예시
service_fallback_order = ["ui-venus", "mai-ui", "kimi-k2.5"]
```

각 모델의 특성:
- `ui-venus`: 단일 요소 grounding에 최적화, `[x,y]` 0-1000 좌표
- `mai-ui`: crop + zoom 후 정밀 grounding에 강함
- `kimi-k2.5`: 범용 VLM, direct 연결

같은 스크린샷에서 한 모델이 실패해도 다른 모델이 성공하는 경우가 `login_benchmark.py`에서 이미 관측되었습니다.

## 4.6 Level 4: OCR Cross-Validation

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

## 4.7 Level 5: 사람 에스컬레이션

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

## 4.8 Non-Idempotent Step 처리

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

## 4.9 재시도 예산

무한 재시도를 방지합니다:

```python
# Per-step 제한
max_retries: int = 3  # WorkflowStep 기본값

# Per-workflow 전체 제한
WORKFLOW_TOTAL_RETRY_BUDGET = 10

# 예산 소진 시 → 워크플로 중단, 에스컬레이션
```

### 예산 상호작용 규칙

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
