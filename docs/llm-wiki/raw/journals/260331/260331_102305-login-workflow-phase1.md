### 1. 진행 사항
- `docs/workflow_automation/README.md`, `docs/workflow_automation/02-engine-architecture.md`, `docs/workflow_automation/06-implementation-blueprint.md`, `docs/workflow_automation/07-safety-rules-and-scope.md`를 기준으로 `poc/work2/action_login.py` 개선 방향을 정리했다.
- `poc/work2/action_login.py`를 직접 액션 스크립트에서 워크플로 entrypoint로 전환하고, 실제 로그인 단계 실행은 `poc/work2/workflow_login.py`로 분리했다.
- `poc/work2/workflow_types.py`, `poc/work2/workflow_config.py`, `poc/work2/workflow_runner.py`를 추가해 `WorkflowStep`, `StepResult`, `WorkflowRun`, `ConditionType`, `ConditionGroup`, `WorkflowSettings` 기반의 phase 1 골격을 구현했다.
- `poc/work2/workflow_login.py`에서 `ensure_login_window` -> `type_userid` -> `type_password` -> `click_login_button` -> `verify_main_window` 순서의 로그인 워크플로를 정의했다.
- `poc/work2/util/__init__.py`에 optional helper 기본값을 추가해 macOS 개발 환경에서 `poc/work2/action_login.py` import 시 즉시 `ImportError`가 나지 않도록 정리했다.
- `uv run python -c "import poc.work2.workflow_login, poc.work2.action_login; print('import_ok')"`로 import 가능 여부를 확인했다.
- `uv run python -m compileall poc/work2/action_login.py poc/work2/workflow_login.py poc/work2/workflow_runner.py poc/work2/workflow_types.py poc/work2/workflow_config.py`로 문법 검증을 수행했다.
- `uv run python poc/work2/action_login.py`를 macOS에서 실행해 `window_utils unavailable - 로그인 창 탐색 불가`로 안전하게 중단되는지 확인했고, `poc/work2/logs/workflow_runs/260331_102048_rcs_login/run_state.json`과 step 결과 JSON이 생성됨을 확인했다.

### 2. 수정 내용
- 변경 파일: `poc/work2/action_login.py`
  기존 로그인 액션 구현을 제거하고 `poc.work2.workflow_login.main()`을 호출하는 thin entrypoint로 단순화했다.
- 신규 파일: `poc/work2/workflow_types.py`
  `ConditionType`, `ConditionGroupType`, `StepCondition`, `ConditionGroup`, `WorkflowStep`, `StepResult`, `WorkflowRun` dataclass 및 enum을 추가했다.
- 신규 파일: `poc/work2/workflow_config.py`
  `WorkflowSettings`와 `load_workflow_settings()`를 추가해 `SAFE_MODE`, action enable, typing enable, settle 시간, verify timeout을 환경변수 기반으로 읽도록 구성했다.
- 신규 파일: `poc/work2/workflow_runner.py`
  `ConditionChecker`와 `WorkflowRunner`를 추가해 step dependency 검사, precondition/success_criteria 평가, step 결과 JSON 저장, `run_state.json` 저장을 구현했다.
- 신규 파일: `poc/work2/workflow_login.py`
  로그인 전용 step 정의와 step executor를 추가했고, `find_login_window()`, `analyze_login_target()`, `click_at_screen()`, `wait_for_rcs_main_window()`를 단계별로 연결했다.
- 변경 파일: `poc/work2/util/__init__.py`
  `activate_window`, `capture_window`, `image_point_to_screen`, `click_at_screen` 등 optional import 대상에 `None` 기본값을 부여해 non-Windows 환경 import 경로를 안정화했다.
- 검증 결과:
  `import_ok` 확인 완료.
  `compileall` 성공.
  macOS 실행에서는 실제 Windows 의존 기능이 없어 `ensure_login_window` 단계에서 안전하게 abort 되었고, workflow run artifact 저장은 정상 동작했다.

### 3. 다음 단계
- `poc/work2/workflow_login.py`의 `verify_main_window` 이전 단계에 대해 docs의 phase 2 계획대로 post-action verification을 추가한다.
- `type_userid`, `type_password`, `click_login_button`에 대해 typed-context 기반 임시 검증 대신 OCR/UIA/window-title/VLM hybrid 검증 경로를 구체화한다.
- `workflow_runner.py`에 foreground 검사, `unexpected_foreground` 분류, `window_unstable` 대응, poll-until-stable 로직을 추가한다.
- 사무실 Windows 환경에서 `uv run python poc/work2/action_login.py`를 실제로 실행해 로그인 dialog 탐지, 입력, 메인 창 전환까지 end-to-end 검증한다.

### 4. 메모리 업데이트
- 변경 없음
