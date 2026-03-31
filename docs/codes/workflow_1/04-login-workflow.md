# 로그인 워크플로 코드 해설

이 문서는 `poc/workflow_1/workflow_login.py`를 중심으로,
로그인 자동화가 step 기반 워크플로로 어떻게 구성되는지 설명한다.

## 1. 이 파일의 역할

`workflow_login.py`는 세 가지 책임을 가진다.

1. 로그인 step 목록 정의
2. step 하나를 실제로 수행하는 executor 구현
3. `WorkflowRunner`를 호출해 전체 실행

즉, 이 파일은 `workflow_1`의 실제 시나리오 정의서이자 엔트리포인트다.

## 2. 상단 import와 상수 블록의 의미

초반 import 블록은 성격상 세 묶음으로 나뉜다.

| 묶음 | 의미 |
| --- | --- |
| `login_rcs_common`, `poc.work2.util` | 실제 창 탐색, 캡처, 클릭 같은 외부 GUI 작업 |
| `workflow_config`, `workflow_runner`, `workflow_types` | 워크플로 엔진 공용 계층 |
| `pynput.keyboard` | 실제 타이핑 수행용 선택 의존성 |

상수 블록 의미:

- `DEFAULT_CREDENTIAL_USER_ID`, `DEFAULT_CREDENTIAL_PASSWORD`
  환경변수가 없을 때 쓸 기본 로그인 값
- `EXIT_SUCCESS`, `EXIT_WORKFLOW_ABORTED`
  엔트리포인트 종료 코드 문자열

## 3. `_load_login_targets()`: 지연 import 블록

이 함수는 `login_rcs_ui_venus_mai`를 함수 안에서 import한다.

의미:

- 타겟 정의와 탐지 함수를 필요할 때만 불러온다.
- import 순환을 느슨하게 만들고 초기 import 비용을 줄인다.

반환값:

- 탐지 성공 코드
- 타겟 정의 딕셔너리
- 탐지 함수

## 4. `load_login_credentials()`: 입력값 해석 블록

이 함수는 환경변수에서

- `ACTION_LOGIN_USER_ID`
- `ACTION_LOGIN_PASSWORD`

를 읽어 `userid_input`, `password_input` 딕셔너리로 반환한다.

의미:

- 이후 step 정의는 "어떤 텍스트를 넣을 것인가"만 보면 되고
- 환경변수 이름을 몰라도 된다.

## 5. `build_login_workflow_steps()`: step 정의 블록

이 함수는 로그인 워크플로의 구조를 선언한다.

## 5.1 공통 조건 정의

`visible_login_window`는 로그인 창이 현재 보이는지를 나타내는 조건 그룹이다.

의미:

- 여러 step에서 반복되는 precondition을 재사용한다.

## 5.2 `ensure_login_window`

이 첫 step은 `observe` 타입이다.

목적:

- 실제 입력이나 클릭 전에
- 로그인 창이 존재하는지 먼저 확인한다.

이 step의 의미:

- 워크플로의 시작점
- 이후 모든 step의 기반 상태 확인

## 5.3 `type_userid`

조건:

- `typing_enabled`가 켜져 있어야 함
- `userid_input` 자격증명이 비어 있지 않아야 함

성공 기준:

- `TEXT_APPEARED`
- 기대 텍스트는 실제 user id 문자열

의미:

- 사용자 ID 필드에 특정 문자열이 들어갔다고 판단되는 step이다.
- 다만 현재 판정은 실제 OCR이 아니라 `typed_values` context를 많이 활용한다.

## 5.4 `type_password`

비슷하지만 차이가 있다.

- 성공 조건은 `MASKED_TEXT_PRESENT`
- `idempotent=False`
- `redact_input_text=True`

이 블록의 의미:

- 비밀번호는 값을 그대로 artifact에 남기면 안 된다.
- 동일 step을 함부로 재실행하면 예상치 못한 입력 누적이 생길 수 있으므로 idempotent를 낮게 둔다.

## 5.5 `click_login_button`

이 step은 입력이 끝난 뒤 로그인 버튼을 클릭한다.

success criteria가 `ALWAYS`인 이유:

- 클릭 자체는 수행 즉시 참/거짓을 화면에서 확정하기 어렵다.
- 실제 성공 판단은 다음 verify step으로 넘긴다.

즉, 이 step은 "행동"만 담당한다.

## 5.6 `verify_updater_window`

이 step은 `verify_only` 타입이다.

성공 조건:

- `WINDOW_APPEARED`
- 제목 prefix는 `RCS Updater (V.` 이어야 함

의미:

- 로그인 버튼 클릭 자체가 아니라
- 로그인 후 상태 변화가 일어났는지를 최종 성공 기준으로 삼는다.

## 6. `context` 갱신 헬퍼 블록

### `_ensure_login_window_context()`

이 함수는 `find_login_window()`를 호출해 현재 로그인 창 상태를 다시 읽고,
결과를 `context`에 반영한다.

업데이트되는 값:

- `login_window`
- `window_title`
- `backend`
- `login_window_visible`
- `process_alive`

의미:

- 워크플로는 오래 실행될 수 있으니
- 이전 step이 저장한 창 핸들을 맹신하지 않고 현재 상태를 다시 반영한다.

### `_capture_login_window()`

이 함수는 실제 실행 step 전에

1. 로그인 창 재탐색
2. activate
3. foreground
4. screenshot capture

를 수행한다.

의미:

- 탐지 step은 항상 최신 캡처를 기준으로 돌아야 한다.
- 창이 뒤로 가 있거나 가려진 상태에서 찍은 이미지를 쓰지 않게 한다.

### `_maybe_save_capture()`

run 디렉터리가 있을 때만 JPEG를 저장한다.

의미:

- 실행 증거물 저장을 공통 처리한다.
- 비밀번호 step 같은 경우 `allow_save=False`로 호출해 민감 정보 노출을 줄일 수 있다.

## 7. `_clear_and_type()`: 실제 타이핑 블록

이 함수는 선택된 입력 필드에 대해

1. backspace 1회
2. 문자별 `keyboard.type()`
3. 문자 간 지연

을 수행한다.

중요한 분기:

- `action_enabled`가 꺼져 있거나
- `pynput`이 없으면

실제 입력 대신 dry-run 로그만 출력하고 `True`를 반환한다.

이 블록의 의미:

- safe mode에서도 워크플로 구조 자체는 계속 검증할 수 있게 한다.

## 8. `_build_base_result()`: 결과 객체 공통 생성 블록

이 함수는 `StepResult` 생성 boilerplate를 모아 둔 것이다.

주요 의미:

- step 실행 함수가 실패/성공 분기마다 같은 필드를 반복 작성하지 않게 한다.
- `artifact_redacted` 기본값을 `step.redact_input_text`에서 가져와 민감 step 처리 규칙을 일관되게 만든다.

## 9. `execute_login_step()`: 실제 step 실행기

이 함수가 `workflow_login.py`의 핵심이다.

step type별로 분기하는 executor라고 보면 된다.

## 9.1 `observe` 분기

동작:

1. 로그인 창 재탐색
2. 없으면 실패
3. 있으면 성공

의미:

- 워크플로 시작 직후 "붙을 대상 창이 실제로 존재하는가"를 확인하는 단계다.

## 9.2 `verify_only` 분기

동작:

1. dry-run이면 `skipped`
2. 아니면 일정 시간 기다림
3. updater 창이 뜰 때까지 polling
4. 있으면 after screenshot 저장
5. 없으면 `verify_failed`

이 블록의 의미:

- 행동과 검증을 분리한다.
- 로그인 버튼 클릭 결과를 후행 UI 변화로 판정한다.

## 9.3 공통 캡처와 사전 실패 처리

`observe`와 `verify_only`가 아닌 step은 먼저 아래 흐름을 거친다.

```text
로그인 창 찾기
-> foreground/capture
-> before screenshot 저장
-> target 설정 조회
-> VLM 탐지 실행
-> image 좌표를 screen 좌표로 변환
```

여기서 실패 가능한 지점:

- 로그인 창 없음
- capture 실패
- target key 정의 없음
- VLM 탐지 실패
- image -> screen 좌표 변환 실패

이 블록의 의미:

- 실제 클릭/입력 전에 필요한 준비와 실패 처리를 한 번에 묶는다.

## 9.4 `type` 분기

타이핑 step의 실제 동작은 아래 순서다.

```text
필드 1회 클릭
-> 잠시 대기
-> 같은 필드 2회 클릭
-> 잠시 대기
-> backspace 후 문자 입력
-> context["typed_values"] 갱신
-> context["focused_target_key"] 갱신
```

왜 단일 클릭 후 더블클릭을 하나:

- 포커스를 먼저 확실히 가져오고
- 기존 텍스트 선택 상태를 더 강하게 유도하려는 의도다.

`typed_values`를 저장하는 이유:

- 후속 success criteria에서 `TEXT_APPEARED`, `MASKED_TEXT_PRESENT`를 판정할 때 쓴다.

## 9.5 `click` 분기

클릭 step은 단순하다.

1. pre-click settle 대기
2. 1회 클릭
3. post-click settle 대기
4. `login_button`이면 `login_submit_attempted=True`

의미:

- 실제 성공 여부는 바로 확정하지 않고
- 최소한의 클릭 액션과 상태 표시만 남긴다.

## 9.6 after screenshot 및 결과 반환

민감 step이 아니고 캡처 가능하면 after screenshot을 저장한 뒤,
`_build_base_result()`로 최종 `StepResult`를 만든다.

의미:

- before/after 증거를 남겨 UI 변화가 있었는지 나중에 비교 가능하게 한다.

## 10. `run_login_workflow()`: 오케스트레이션 블록

이 함수는 아래 순서로 동작한다.

```text
설정 로드
-> 자격증명 로드
-> step 목록 생성
-> 기본 context 생성
-> WorkflowRunner 생성
-> run()
```

기본 context에 들어가는 `typed_values`, `process_exe_name`은
condition 검사와 결과 해석에 필요하다.

의미:

- 이 함수는 로그인 시나리오를 러너에 연결하는 어댑터다.

## 11. `main()`: 실행 엔트리포인트

`main()`은 `run_login_workflow()` 결과를 보고

- `completed`면 `success`
- 아니면 `workflow_aborted`

를 반환한다.

의미:

- 셸에서 실행했을 때는 단순한 exit code 인터페이스를 유지하면서
- 내부적으로는 상세 `WorkflowRun` 객체를 다룬다.

## 12. 이 파일의 설계 의도와 한계

### 설계 의도

- 로그인 자동화를 step 기반으로 쪼갠다.
- 각 step 결과를 JSON으로 남긴다.
- safe mode에서도 구조 검증이 가능해야 한다.
- 좌표 탐지는 `ui-venus + mai-ui` 파이프라인으로 공통화한다.

### 현재 한계

- retry budget은 설정만 있고 실제 재시도 전략은 거의 없다.
- 텍스트 입력 검증이 실제 화면 재확인보다 `context["typed_values"]`에 많이 의존한다.
- verify step은 updater 창 등장 하나에 집중되어 있다.

즉, `workflow_login.py`는 완성된 범용 워크플로 엔진보다는
"로그인 시나리오를 엔진 형태로 분해한 첫 번째 구현"으로 이해하는 것이 맞다.
