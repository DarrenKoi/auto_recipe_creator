# 코어 타입과 워크플로 러너

이 문서는 `workflow_1`의 "기초 공사" 파일들을 설명한다.

- `__init__.py`
- `workflow_config.py`
- `workflow_types.py`
- `debug_artifacts.py`
- `logger.py`
- `workflow_runner.py`

## 1. `__init__.py`: 패키지 공용 경로 선언

이 파일의 코드 블록은 거의 전부 경로 상수 선언이다.

| 코드 블록 | 의미 |
| --- | --- |
| `WORKFLOW_1_DIR = ...` | 패키지 루트 경로를 기준점으로 삼는다 |
| `DEBUG_IMAGE_DIR = ...` | VLM 입력/오버레이 이미지를 저장할 기본 위치다 |
| `LOG_DIR = ...` | 로그 파일과 run state JSON이 저장될 기본 위치다 |
| `__all__ = [...]` | 어떤 이름을 외부에 공용 API처럼 노출할지 정리한다 |

이 블록이 중요한 이유는, 이후 모든 모듈이 디버그 이미지와 로그 저장 위치를 동일하게 해석하게 만들기 때문이다.

## 2. `workflow_config.py`: 런타임 설정 묶음

### `WorkflowSettings`

`WorkflowSettings`는 "로그인 워크플로가 어떤 방식으로 행동할지"를 한 덩어리로 표현한다.

대표 필드의 의미:

| 필드 | 의미 |
| --- | --- |
| `safe_mode` | 기본적으로 위험한 자동 클릭/입력을 막는 안전 스위치 |
| `action_enabled` | 실제 클릭과 타이핑을 할지 여부 |
| `typing_enabled` | 로그인 step 중 입력 step 자체를 넣을지 여부 |
| `service_fallback_order` | 향후 서비스 fallback이 필요할 때 쓸 우선순위 목록 |
| `pre_click_settle_sec`, `post_click_settle_sec` | 클릭 전후 화면 안정화 대기 시간 |
| `char_type_delay_sec` | 문자 하나씩 입력할 때 간격 |
| `login_verify_timeout_sec` | 로그인 후 후속 창 등장 검증 최대 대기 시간 |

### `load_workflow_settings()`

이 함수의 블록 의미는 다음과 같다.

1. `SAFE_MODE`를 읽는다.
2. `ACTION_LOGIN_ACTION_ENABLED` 기본값을 `not safe_mode`로 둔다.
3. 나머지 숫자형/불리언 설정을 환경변수 또는 기본값으로 채운다.

즉, 이 함수는 "실행 옵션 해석기"다.  
코드 곳곳에서 환경변수를 직접 읽지 않게 해준다.

## 3. `workflow_types.py`: 워크플로의 공용 언어

이 파일은 러너와 실행기 사이의 계약서 역할을 한다.

### `_serialize_value()`

이 헬퍼는 dataclass, enum, `Path`를 JSON으로 저장 가능한 일반 값으로 바꾼다.

의미:

- `StepResult`나 `WorkflowRun`을 그대로 `json.dumps()`할 수 없으니
- 저장 직전 직렬화 가능한 구조로 정리한다.

### `ConditionType`

각 enum 값은 step의 조건 종류를 뜻한다.

| 값 | 의미 |
| --- | --- |
| `ALWAYS` | 항상 참 |
| `WINDOW_VISIBLE` | 로그인 창이 현재 보이는가 |
| `WINDOW_APPEARED` | 로그인 이후 다른 창이 나타났는가 |
| `PROCESS_ALIVE` | RCS 프로세스가 살아 있는가 |
| `TEXT_APPEARED` | 특정 필드에 텍스트가 입력되었다고 볼 수 있는가 |
| `MASKED_TEXT_PRESENT` | 비밀번호 필드에 무언가 입력되었다고 볼 수 있는가 |

### `StepCondition` / `ConditionGroup`

- `StepCondition`
  원자 조건 하나를 표현한다.
- `ConditionGroup`
  여러 조건을 `ALL` 또는 `ANY`로 묶는다.

의미:

- precondition, success criteria, skip condition을 raw dict 대신 타입 구조로 다루게 해준다.
- 러너는 구체적인 로그인 로직을 몰라도 condition 객체만 보면 공통 처리할 수 있다.

### `WorkflowStep`

`WorkflowStep`은 "러너가 실행할 단위 작업"이다.

핵심 필드:

| 필드 | 의미 |
| --- | --- |
| `step_id` | 결과 파일명과 의존성 판정에 쓰는 고유 ID |
| `step_type` | `observe`, `type`, `click`, `verify_only` 같은 실행 분기 키 |
| `target_key` | 어떤 GUI 요소를 다룰지 나타내는 식별자 |
| `preconditions` | step 실행 전에 만족해야 할 조건 |
| `success_criteria` | executor가 끝난 뒤 만족해야 하는 조건 |
| `depends_on` | 선행 step 성공 여부 |
| `input_text` | type step일 때 실제로 넣을 문자열 |
| `redact_input_text` | 비밀번호처럼 artifact를 숨겨야 하는지 |

### `StepResult`

이 타입은 각 step의 결과 보고서다.

중요한 이유:

- status만 남기는 것이 아니라
- 어떤 좌표를 탐지했고
- 어떤 스크린샷을 저장했고
- 왜 실패했는지까지 함께 기록한다.

### `WorkflowRun`

전체 워크플로 실행 메타데이터다.

주요 역할:

- 어떤 workflow를 언제 시작했고
- 현재 몇 번째 step까지 갔고
- step 결과 목록이 무엇인지
- 이번 run의 저장 디렉터리가 어디인지

를 한 JSON으로 묶는다.

## 4. `debug_artifacts.py`: 이미지/텍스트/JSON 저장 계층

이 파일의 함수들은 모두 "디버그 증거물 저장기"라고 보면 된다.

### 경로 계산 블록

- `debug_image_path()`
  `poc.work2.debug_image_path`를 재사용해서 timestamp와 model name이 포함된 경로를 만든다.

의미:

- 파일명 충돌을 줄이고
- 어떤 모델 조합이 생성한 artifact인지 추적하기 쉽게 만든다.

### 원본 저장 블록

- `save_debug_jpeg()`
  원본 캡처 이미지를 JPEG로 저장한다.
- `save_debug_webp()`
  실제 VLM에 넣은 것과 동일한 WebP 버전을 저장한다.
- `save_debug_text()`
  프롬프트 응답 텍스트를 저장한다.
- `save_debug_json()`
  결과 payload를 저장한다.

### 시각화 블록

- `save_marked_bboxes()`
  bbox와 중심점을 이미지에 그려 저장한다.

이 함수의 세부 블록 의미:

1. 폰트를 준비한다.
2. 요소별 bbox를 이미지 범위 안으로 clamp한다.
3. 사각형과 중심점 crosshair를 그린다.
4. 라벨 텍스트 박스를 겹치지 않게 배치한다.
5. 최종 이미지를 저장한다.

즉, 이 함수는 "VLM이 어디를 봤다고 판단했는지"를 사람이 빠르게 검증하게 해주는 시각 증거 생성기다.

## 5. `logger.py`: 파일 기반 구조화 로그

이 파일은 repo 전반의 print 스타일과 별개로, `workflow_1` 전용 파일 로그를 남긴다.

### `_get_logger()`

이 블록은 `log_name`별 singleton logger를 만든다.

주요 의미:

- 같은 이름으로 여러 번 호출해도 핸들러를 중복 등록하지 않는다.
- 로그 파일은 rotating file handler로 관리된다.
- 실제 로그 파일은 `poc/workflow_1/logs/<name>.log`에 쌓인다.

### `_format_tokens()` / `_format_fields()`

- 토큰 사용량과 추가 필드를 문자열로 평탄화한다.
- 목적:
  로그를 grep하기 쉽게 만든다.

### `log_vlm_call()`

이 함수는 VLM 요청 결과를 기록한다.

성공 시:

- 서비스명
- 모델명
- 지연 시간
- 토큰 사용량
- endpoint
- 응답 본문

실패 시:

- 위 정보와 함께 error 메시지

를 기록한다.

### `log_work2_event()`

이 함수는 일반 이벤트 로그를 기록한다.

이름은 `work2`가 들어가 있지만 실제로는 `workflow_1`에서도 재사용한다.
즉, 이 함수는 "VLM 호출 외의 상태 변화"를 남기는 범용 이벤트 로거다.

## 6. `workflow_runner.py`: 순차 실행 엔진

이 파일은 `workflow_1`의 제어 중심이다.

### `ConditionChecker`

`ConditionChecker`의 의미는 "현재 context를 읽어 step 조건을 평가하는 작은 룰 엔진"이다.

#### `_handlers` 블록

`ConditionType` enum 값마다 검사 함수를 연결한다.

의미:

- if/elif 체인을 늘리는 대신
- 타입별 조건 검사 로직을 명확히 분리한다.

#### `check_condition()` / `check_group()`

- `check_condition()`
  단일 조건을 검사한다.
- `check_group()`
  여러 조건을 `ALL` 또는 `ANY`로 묶어 판정한다.

이 블록은 러너가 step precondition, skip, success criteria를 모두 같은 방식으로 다루게 한다.

#### 구체적인 condition 검사 메서드들

| 메서드 | 의미 |
| --- | --- |
| `_check_window_visible()` | 로그인 창 핸들과 제목이 기대값에 맞는지 |
| `_check_window_appeared()` | 로그인 후 새 창이 뜬 것으로 볼 수 있는지 |
| `_check_process_alive()` | context에 기록된 프로세스 정보가 살아 있는지 |
| `_check_field_ready_for_input()` | 특정 필드가 현재 포커스된 것으로 간주되는지 |
| `_check_text_appeared()` | `typed_values` 캐시에 기대 텍스트가 들어 있는지 |
| `_check_masked_text_present()` | 비밀번호 필드에 무언가 입력된 것으로 간주되는지 |

주의할 점:

- 현재 구현은 실제 OCR/UIA 재확인보다 `context` 캐시에 의존한다.
- 즉, "검증 엔진"이지만 아직은 실제 GUI 재판독보다 내부 상태 기반 판정 비중이 높다.

### `WorkflowRunner.run()`

이 메서드는 다음 순서로 동작한다.

```text
run_id 생성
-> run_dir 생성
-> run_state.json 초기 저장
-> 각 step 순회
   -> dependency 검사
   -> skip 조건 검사
   -> precondition 검사
   -> executor 호출
   -> success criteria 재검사
   -> step 결과 JSON 저장
-> 전체 상태 completed 또는 aborted 저장
```

각 코드 블록의 의미:

| 코드 블록 | 의미 |
| --- | --- |
| run 초기화 | 이번 실행의 메타데이터와 저장 폴더를 만든다 |
| `checker = ConditionChecker(context)` | context 기반 조건 평가기를 준비한다 |
| dependency 검사 | 선행 step 실패면 현재 step을 바로 실패 처리한다 |
| skip 검사 | 이미 만족된 step이면 실제 실행 없이 건너뛴다 |
| precondition 검사 | 지금 실행 가능한 상태가 아니면 실패 처리한다 |
| `executor(step, context)` | 실제 비즈니스 동작을 외부 함수에 위임한다 |
| success criteria 재검사 | executor가 성공을 반환해도 조건이 안 맞으면 최종 실패 처리한다 |
| 결과 파일 저장 | `step_<step_id>.json`, `run_state.json`을 계속 갱신한다 |

### `_check_dependencies()`

이 함수는 현재 step의 `depends_on` 목록이 모두 `success`인지 검사한다.

의미:

- GUI 자동화는 순서 의존성이 강하므로
- 조건 검사와 별개로 명시적인 선행 성공 관계를 강제한다.

### `_build_result()`

실행 전 실패나 skip 같은 공통 상황에서 쓰는 기본 `StepResult` 생성기다.

의미:

- executor를 호출하지 못한 경우에도 결과 형식을 동일하게 유지한다.

### `_write_run_state()` / `_write_step_result()`

이 블록은 디버깅에서 매우 중요하다.

- `run_state.json`
  전체 워크플로의 현재 상태 스냅샷
- `step_<id>.json`
  각 step의 독립 결과 보고서

즉, 콘솔 로그를 놓쳐도 파일만 보면 어느 단계에서 왜 멈췄는지 복원할 수 있다.
