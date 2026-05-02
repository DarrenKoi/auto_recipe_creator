---
tags: [component, workflow, automation, safety]
level: intermediate
last_updated: 2026-05-02
status: needs-review
owner: 대영
sources: [
  raw/journals/260331/260331_102305-login-workflow-phase1.md,
  raw/journals/260316/20260316-work2-stepwise-rebuild-breakdown.md
]
---

# Workflow Runner

> GUI automation을 `observe -> decide -> act -> verify` 단계로 나누고, 각 step의 조건과 결과 artifact를 기록하는 실행 골격이다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

## 왜 존재하는가? (Why)

- 로그인 자동화는 창 탐색, 타겟 탐지, typing, click, post-action verification을 분리해야 실패 지점을 재현할 수 있다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- 이전 rebuild 계획은 한 파일에 여러 동작을 다시 합치지 않고 단계별 script와 step 책임을 명확히 분리하는 방향을 제안했다. (source: raw/journals/260316/20260316-work2-stepwise-rebuild-breakdown.md)

## 무엇인가? (What)

### 책임 범위

- `WorkflowStep`, `StepResult`, `WorkflowRun`, `ConditionType`, `ConditionGroup`, `WorkflowSettings` 같은 dataclass/enum 기반 구조로 workflow phase 1 골격을 표현한다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- runner는 step dependency, precondition, success criteria 평가, step result JSON 저장, `run_state.json` 저장을 담당한다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- Windows-only dependency가 없는 macOS 개발 환경에서는 window utility import가 없어도 안전하게 abort되도록 optional helper 기본값을 둔 흐름이 기록되어 있다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

### 핵심 진입점

- `workflow_types.py` — workflow step/result/run model. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- `workflow_config.py` — `SAFE_MODE`, action enable, typing enable, timing 설정. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- `workflow_runner.py` — condition check와 step 실행 상태 저장. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- `workflow_login.py` — `ensure_login_window -> type_userid -> type_password -> click_login_button -> verify_main_window` 흐름. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

> Conflict: `raw/journals/260331/260331_102305-login-workflow-phase1.md`는 work2 아래 workflow file 추가를 기록하지만, 현재 ingest 확인 시 `poc/work2/action_login.py:13`은 `poc.work2.workflow_login`을 import하고 `poc/work2/workflow_login.py`는 없으며 `poc/workflow_1/workflow_login.py:1`에 구현이 있었다. package boundary와 entrypoint 정상 동작 여부를 확인해야 한다.

### 의존성

- 내부: [rcs-login-automation.md](./rcs-login-automation.md), [rcs-tool-selection.md](./rcs-tool-selection.md), [gui-coordinate-and-window-focus.md](../concepts/gui-coordinate-and-window-focus.md)
- 외부: Windows GUI automation libraries, VLM target detection, OCR verification. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

### 데이터 모델 / 인터페이스

```text
WorkflowStep
  -> preconditions
  -> executor
  -> StepResult
  -> success_criteria
WorkflowRun
  -> run_state.json
  -> per-step JSON artifacts
```

## 어떻게 쓰는가? (How)

### 호출 예시

```powershell
uv run python poc/work2/action_login.py
```

저널 기준으로 이 entrypoint는 login workflow를 호출하는 thin wrapper로 전환되었다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

### 자주 쓰는 패턴

- 실제 액션은 `SAFE_MODE`와 action/typing enable 값을 확인한 뒤 수행한다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- 각 step은 before/after screenshot과 verification result를 남겨 실패 원인을 나중에 재검토할 수 있게 한다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- 후속 단계는 post-action verification, foreground 검사, unexpected foreground 분류, window stable polling을 추가하는 방향으로 남아 있다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)

### 안티패턴

- login, list tab, tool selection, verification을 검증 없이 하나의 opaque action script로 다시 합치지 않는다. (source: raw/journals/260316/20260316-work2-stepwise-rebuild-breakdown.md)

## 참고 자료 (References)

- 원본 메모: [260331_102305-login-workflow-phase1.md](../../raw/journals/260331/260331_102305-login-workflow-phase1.md)
- 원본 메모: [20260316-work2-stepwise-rebuild-breakdown.md](../../raw/journals/260316/20260316-work2-stepwise-rebuild-breakdown.md)
