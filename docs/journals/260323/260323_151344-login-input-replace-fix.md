## 1. 진행 사항
- `poc/work2/action_login.py`를 점검해 로그인 성공 확인 경로가 `poc.work2.login_rcs_common.wait_for_rcs_main_window()`를 사용하고 있음을 확인했다.
- `poc/work2/login_rcs_common.py`의 `wait_for_rcs_main_window(timeout_sec=15.0, poll_interval_sec=2.0)`를 확인해 메인 RCS 창 타이틀 폴링 간격이 2초 기준으로 동작하는 상태를 점검했다.
- `poc/work2/action_login.py`의 입력 필드 처리 흐름을 검토해 `userid_input`, `password_input`에서 기존 텍스트 위에 append 되는 문제가 포커스/선택 안정성 부족과 연관되어 있음을 정리했다.
- `poc/work2/action_login.py`에서 타이핑 대상 입력창을 단일 클릭 대신 더블클릭한 뒤 `Ctrl+A -> Delete -> typing` 순서로 덮어쓰기하도록 흐름을 정리했다.
- `uv run python -m py_compile poc/work2/action_login.py`로 문법 검증을 수행했고, 사용자 확인 기준으로 수정 후 로그인 동작이 정상 동작하는 상태를 확인했다.

## 2. 수정 내용
- 수정 파일: `poc/work2/action_login.py`
  입력 필드 타겟(`userid_input`, `password_input`)에 대해 `_click_at_screen(..., click_count=2)`를 사용하도록 바꿨고, `PRE_TYPE_DOUBLE_CLICK_SETTLE_SEC` 대기 후 `_clear_and_type()`를 호출하도록 정리했다.
- 확인 파일: `poc/work2/login_rcs_common.py`
  `wait_for_rcs_main_window()`의 기본 `poll_interval_sec`가 `2.0`으로 유지되는 것을 확인해 로그인 성공 판정 타이틀 체크가 2초 주기로 동작하는 상태를 점검했다.
- 신규 파일: `docs/journals/260323/260323_151344-login-input-replace-fix.md`
  이번 세션의 로그인 입력창 덮어쓰기 안정화 작업과 확인 내용을 기록하는 저널 파일을 추가했다.

## 3. 다음 단계
- 사무실 Windows 환경에서 `uv run python poc/work2/open_rcs.py` 후 `uv run python poc/work2/action_login.py`를 다시 실행해 `userid_input`, `password_input` 모두에서 더블클릭 기반 덮어쓰기가 일관되게 유지되는지 확인한다.
- 특정 계정/환경에서 더블클릭 선택이 불안정하면 `PRE_TYPE_DOUBLE_CLICK_SETTLE_SEC` 또는 클릭 간격 보정이 추가로 필요한지 점검한다.

## 4. 메모리 업데이트
- 변경 없음
