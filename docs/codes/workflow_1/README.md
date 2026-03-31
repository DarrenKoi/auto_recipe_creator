# `poc/workflow_1` Code Guide

이 디렉터리는 `poc/workflow_1/` 코드를 이해하기 쉽게 풀어쓴 문서 모음이다.

`workflow_1`은 RCS 로그인 자동화를 "한 번에 끝나는 단일 스크립트"가 아니라,
조건 기반 step을 순차 실행하는 작은 워크플로 엔진으로 분리해 놓은 패키지다.
공용 VLM 호출, 이미지 처리, 윈도우 제어 유틸은 대부분 `poc/work2/`에서 재사용하고,
`poc/workflow_1/`은 그 위에 얇은 오케스트레이션 계층을 얹는다.

## 먼저 이해할 핵심

1. `open_rcs.py`
   RCS 프로세스를 띄우고 PID 상태 파일을 남긴다.
2. `login_rcs_common.py`
   로그인 창, 메인 창, updater 창을 찾는 공용 창 탐색 계층이다.
3. `ui_venus_mai_locator.py`
   `ui-venus`로 큰 위치를 찾고 `mai-ui`로 세밀한 클릭 좌표를 정제한다.
4. `workflow_types.py` + `workflow_runner.py`
   step, condition, run result 같은 워크플로 공용 타입과 실행기를 정의한다.
5. `workflow_login.py`
   실제 로그인 step 목록과 step 실행 로직을 조합한다.

## 문서 구성

| 문서 | 목적 |
| --- | --- |
| [01-structure-and-file-map.md](01-structure-and-file-map.md) | 패키지 구조, 파일별 역할, 데이터 흐름 |
| [02-core-types-and-runner.md](02-core-types-and-runner.md) | 설정/타입/로깅/아티팩트/워크플로 러너 설명 |
| [03-window-and-target-detection.md](03-window-and-target-detection.md) | RCS 실행, 창 탐색, `ui-venus + mai-ui` 탐지 파이프라인 설명 |
| [04-login-workflow.md](04-login-workflow.md) | 로그인 step 정의와 실제 step 실행 코드 설명 |

## 전체 실행 흐름

```text
open_rcs.py
  -> open_rcs_state.json 기록
  -> login_rcs_common.find_login_window()
  -> workflow_login.build_login_workflow_steps()
  -> workflow_runner.run()
  -> execute_login_step()
  -> login_rcs_ui_venus_mai.analyze_login_target()
  -> ui_venus_mai_locator.analyze_window_target()
  -> click/type
  -> updater 창 검증
```

## 이 패키지를 읽는 순서

`workflow_1`을 처음 보는 경우 아래 순서가 가장 이해하기 쉽다.

1. [01-structure-and-file-map.md](01-structure-and-file-map.md)
2. [04-login-workflow.md](04-login-workflow.md)
3. [03-window-and-target-detection.md](03-window-and-target-detection.md)
4. [02-core-types-and-runner.md](02-core-types-and-runner.md)

## 이 문서 세트의 관점

- 코드를 "무엇을 하는가"보다 "왜 이 블록이 필요한가" 중심으로 설명한다.
- 함수 본문을 그대로 복붙하지 않고, 블록 단위의 의도를 해설한다.
- `workflow_1`이 아직 Phase 1 수준의 얇은 엔진이라는 점도 함께 정리한다.
