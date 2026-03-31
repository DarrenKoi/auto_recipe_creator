# `workflow_1` 구조와 파일 맵

## 1. 패키지 한눈에 보기

```text
poc/workflow_1/
├── __init__.py
├── debug_artifacts.py
├── logger.py
├── login_rcs_common.py
├── login_rcs_ui_venus_mai.py
├── open_rcs.py
├── ui_venus_mai_locator.py
├── workflow_config.py
├── workflow_login.py
├── workflow_runner.py
└── workflow_types.py
```

이 패키지는 크게 4계층으로 나뉜다.

| 계층 | 파일 | 역할 |
| --- | --- | --- |
| 패키지/경로 | `__init__.py` | 로그 디렉터리, 디버그 이미지 디렉터리 같은 공용 경로 제공 |
| 지원 유틸 | `debug_artifacts.py`, `logger.py`, `workflow_config.py`, `workflow_types.py` | 저장 형식, 로그, 설정, 데이터 모델 담당 |
| 외부 GUI/VLM 연결 | `open_rcs.py`, `login_rcs_common.py`, `login_rcs_ui_venus_mai.py`, `ui_venus_mai_locator.py` | RCS 실행, 창 탐색, VLM 기반 좌표 탐지 |
| 워크플로 오케스트레이션 | `workflow_runner.py`, `workflow_login.py` | step 실행 순서와 성공/실패 판단 |

## 2. 파일별 역할

### `__init__.py`

- `WORKFLOW_1_DIR`, `DEBUG_IMAGE_DIR`, `LOG_DIR`를 정의한다.
- 의미:
  이 파일은 기능 코드보다 "패키지의 공용 기준 경로"를 한 곳에 모아 두는 역할이 더 크다.
- 효과:
  각 모듈이 상대경로를 계산하지 않고 같은 기준 디렉터리를 공유할 수 있다.

### `workflow_config.py`

- `WorkflowSettings` dataclass를 정의한다.
- 환경변수와 기본값을 합쳐 `load_workflow_settings()`로 설정 객체를 만든다.
- 의미:
  step 동작 간격, dry-run 여부, typing 허용 여부, 로그인 후 검증 폴링 시간을 하드코딩 대신 한 곳에 모은다.

### `workflow_types.py`

- 워크플로 엔진이 주고받는 표준 타입을 모아 둔다.
- `ConditionType`, `ConditionGroupType`, `StepCondition`, `ConditionGroup`, `WorkflowStep`, `StepResult`, `WorkflowRun`가 있다.
- 의미:
  "워크플로 정의"와 "워크플로 실행 결과"의 형태를 고정해 러너와 개별 step 코드가 느슨하게 연결되게 한다.

### `workflow_runner.py`

- `ConditionChecker`
  현재 `context`를 보고 condition이 만족되는지 평가한다.
- `WorkflowRunner`
  step 목록을 순서대로 실행하고 step 결과 JSON과 run 상태 JSON을 저장한다.
- 의미:
  실제 GUI 동작은 하지 않고, step 순서 관리와 성공 판단만 담당한다.

### `open_rcs.py`

- `RcsMainHD.exe` 실행 전용 스크립트다.
- 이미 같은 프로세스가 있으면 재실행하지 않고 기존 PID를 상태 파일에 남긴다.
- 의미:
  로그인 자동화가 시작되기 전에 "RCS 프로세스가 떠 있는가?"를 보장하는 bootstrap 역할이다.

### `login_rcs_common.py`

- `open_rcs.py`가 남긴 상태 파일을 읽는다.
- PID가 살아 있는지 확인한다.
- 로그인 창, 메인 창, updater 창을 찾는다.
- 의미:
  이후 모듈들이 "창 찾기"를 매번 직접 구현하지 않도록 공용 탐색 계층을 만든다.

### `login_rcs_ui_venus_mai.py`

- 로그인 창 안의 `userid_input`, `password_input`, `login_button` 같은 타겟 정의를 모아 둔다.
- 실제 탐지는 `ui_venus_mai_locator.analyze_window_target()`에 위임한다.
- 의미:
  "로그인 화면에서 어떤 타겟을 찾을 것인가"와 "어떻게 찾을 것인가"를 분리한다.

### `ui_venus_mai_locator.py`

- 이 패키지의 핵심 VLM 좌표 탐지 모듈이다.
- 전체 창 이미지에서 `ui-venus`로 대략적인 bbox를 찾고, 그 주변 crop을 `mai-ui`로 다시 분석해 정밀 클릭 좌표를 뽑는다.
- 의미:
  전체 화면에서 바로 point를 찾는 것보다 더 안정적인 2단계 탐지 파이프라인이다.

### `workflow_login.py`

- 로그인 전용 step 목록을 만든다.
- 각 step을 실제로 수행하는 executor를 구현한다.
- 러너를 호출해 전체 로그인 워크플로를 실행한다.
- 의미:
  이 파일이 `workflow_1`의 실제 엔트리이자 비즈니스 로직이다.

## 3. 의존성 방향

```text
workflow_login.py
  -> workflow_runner.py
  -> workflow_types.py
  -> workflow_config.py
  -> login_rcs_common.py
  -> login_rcs_ui_venus_mai.py
       -> ui_venus_mai_locator.py
            -> debug_artifacts.py
            -> logger.py
            -> poc.work2.util
            -> poc.work2.vlm_client
```

방향이 중요한 이유:

- `workflow_runner.py`는 `workflow_login.py`를 몰라야 재사용 가능하다.
- 타겟 탐지 로직은 `workflow_login.py` 안에 직접 박지 않고 별도 모듈로 빼서 다른 워크플로에도 재사용할 수 있게 한다.
- `poc.work2`는 여전히 공용 인프라 계층이고, `workflow_1`은 그 위의 시나리오 계층이다.

## 4. 실행 중 공유되는 주요 `context`

`workflow_1`은 class 기반 상태 저장소를 두지 않고, `dict context`를 중심으로 상태를 공유한다.

| key | 언제 채워지나 | 의미 |
| --- | --- | --- |
| `run_dir` | 러너 시작 시 | 이번 실행의 결과 JSON/JPEG 저장 폴더 |
| `login_window` | 로그인 창 탐색 후 | 현재 로그인 창 핸들 |
| `window_title` | 로그인 창 탐색 후 | 현재 로그인 창 제목 |
| `backend` | 로그인 창 탐색 후 | `uia` 또는 `win32` 탐색 백엔드 |
| `process_alive` | 로그인 창 탐색 후 | 프로세스가 살아 있다고 판단했는지 |
| `typed_values` | type step 후 | 입력했다고 간주하는 값 캐시 |
| `focused_target_key` | type step 후 | 마지막으로 포커스 줬다고 간주하는 필드 |
| `active_target_key` | 탐지 직전 | 현재 step이 대상으로 삼는 GUI 요소 |
| `post_login_window` | 검증 step 후 | 로그인 후 발견한 updater 창 |
| `post_login_title` | 검증 step 후 | updater 창 제목 |

핵심 포인트:

- 성공 판정의 일부가 실제 OCR이 아니라 `context["typed_values"]` 기반이다.
- 즉 Phase 1에서는 "실제 화면을 완전히 다시 읽어 검증"하기보다 "내가 방금 입력한 값을 상태로 기억"하는 방식이 섞여 있다.

## 5. 현재 패키지의 성격

`workflow_1`은 완성형 범용 엔진이라기보다, 로그인 자동화를 step 기반 구조로 재편하는 첫 번째 실험 버전에 가깝다.

현재 강점:

- step 단위 결과 JSON 저장
- 조건 타입과 run 메타데이터 분리
- safe mode / dry-run 지원
- `ui-venus + mai-ui` 2단계 탐지 파이프라인 재사용 가능

현재 한계:

- retry 예산은 설정에 있지만 실제 재시도 오케스트레이션은 거의 없다
- 성공 검증이 화면 재인식보다 context 기반 추론에 더 의존한다
- 로그인 시나리오 외 일반화는 아직 제한적이다
