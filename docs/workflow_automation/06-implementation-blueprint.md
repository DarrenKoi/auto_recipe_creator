# 구현 청사진

## 6.1 새 모듈

| 모듈 | 역할 |
|------|------|
| `poc/work2/workflow_types.py` | `WorkflowStep`, `StepCondition`, `ConditionType`, `StepResult`, `WorkflowRun` dataclass |
| `poc/work2/workflow_runner.py` | 순차 실행기, 재시도 루프, checkpoint, foreground 검사 |
| `poc/work2/workflow_verify.py` | 후행 검증 로직 (VLM/OCR/UIA/윈도우 타이틀), poll-until-stable |
| `poc/work2/workflow_retry.py` | 재시도 전략 (jitter, crop-zoom, model fallback, failure routing) |
| `poc/work2/workflow_conditions.py` | `ConditionChecker` — 조건 타입 평가 로직 |
| `poc/work2/workflow_config.py` | `WorkflowSettings` dataclass, retry/settle/verification 기본값 |
| `poc/work2/prompts/prompt_action_verify.py` | 검증용 VLM 프롬프트 빌더 |
| `poc/work2/workflow_login.py` | RCS 로그인 워크플로 정의 (action_login.py의 워크플로 버전) |

v1에서는 모듈을 더 줄여도 됩니다:

- `workflow_types.py` (`StepCondition`, `ConditionType` 포함)
- `workflow_runner.py` (`ConditionChecker` 내장)
- `workflow_login.py`

즉, 검증/재시도 로직도 처음에는 `workflow_runner.py` 내부 private helper 로 시작해도 괜찮습니다.
실제 login workflow 1개가 안정된 뒤에 모듈 분리를 해도 늦지 않습니다.

## 6.2 기존 모듈 연동

변경이 **불필요한** 모듈 (그대로 사용):
- `vlm_client.py` — `Work2VLMClient`가 이미 임의 `service_slug` 지원
- `util/json_utils.py` — `extract_json()`, `parse_coords()` 그대로 사용
- `util/image_utils.py` — `capture_window()`, `encode_image_webp()` 그대로 사용
- `util/debug_image_utils.py` — 디버그 아티팩트 저장 그대로 사용
- `ui_venus_mai_locator.py` — 2단계 파이프라인 호출 그대로 사용

설정 추가가 필요한 모듈:
- `flask_vlm.py` — 서비스 slug / 모델명 source of truth 로 유지
- `workflow_config.py` — 워크플로 운영 기본값을 별도 dataclass 로 관리

```python
@dataclass(frozen=True)
class WorkflowSettings:
    verify_service: str = "paddleocr-vl-1.5"
    service_fallback_order: tuple[str, ...] = ("ui-venus", "mai-ui")
    total_retry_budget: int = 10
    settle_max_wait_sec: float = 3.0
    settle_similarity_threshold: float = 0.98
    allow_optional_imagehash: bool = False
```

## 6.3 단계별 구현 순서

```
Phase 1: dataclass + 골격
   WorkflowStep, StepCondition, ConditionType, StepResult, WorkflowRun dataclass 정의
   ConditionGroup, ConditionChecker 기본 구현
   WorkflowRunner 골격 (순차 실행만, 검증/재시도 없음)
   action_login.py 로직을 workflow_login.py로 매핑
   완료 기준: action_login.py 흐름이 step 목록으로만 재현되고, 추가 액션 없이 dry-run trace 출력 가능

Phase 2: 안정성 게이트 + 후행 검증
   foreground 검사 (unexpected_foreground 감지 + 알려진 interrupt 자동 처리)
   poll-until-stable 구현 (histogram 기본, imagehash 선택)
   prompt_action_verify.py 프롬프트 빌더
   workflow_verify.py 검증 로직
   click/type step에 UIA/OCR/VLM hybrid 검증 연결
   완료 기준: office Windows 에서 로그인 dialog 캡처 후 verify-only step 이 성공/실패를 일관되게 기록

Phase 3: failure-aware 재시도
   workflow_retry.py 전략 구현
   failure_class 분류 + retry routing
   jitter (element bbox 있을 때만) + model fallback 통합
   재시도 예산 관리 (reserved_retry_budget 포함)
   non-idempotent step HALT 처리
   완료 기준: detect_failed / verify_failed / halt_non_idempotent 케이스가 서로 다른 routing 을 타는 로그 확인

Phase 4: crop-retry zoom + OCR cross-validation
   ui_venus_mai_locator.py 2단계 파이프라인을 재시도 전략으로 연결
   OCR pre-check 통합
   완료 기준: text-labeled control 에서 OCR anchor 유무에 따라 retry 경로가 달라짐을 증거로 확인

Phase 5: 상태 저장 + pause metadata
   WorkflowRun JSON 저장/로드
   paused 상태 / resume_options 저장
   완료 기준: halt 시 run_state.json 만으로 마지막 step, 증거 경로, 수동 선택지가 복원 가능

Phase 6: 실패 이력 기반 적응 (v2)
   실패 통계 집계
   VLM 서비스 자동 선택
```

권장 검증 게이트:
- 각 phase 종료 시 macOS 개발 환경 문법 검증 + office Windows 실기 확인을 분리해 기록합니다
- 최소 증거: 실행 명령, 결과 로그, before/after artifact 경로, failure_class 샘플
