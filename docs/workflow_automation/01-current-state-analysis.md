# 현재 상태 분석

## 1.1 기존 구조의 한계

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

## 1.2 `action_login.py` — 워크플로의 원형

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
