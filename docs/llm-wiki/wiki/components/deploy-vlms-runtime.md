---
tags: [component, deploy-vlms, runtime, operations]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260315/20260315-vlm-process-persistence.md,
  raw/journals/260316/20260316-vlm-background-terminal-runbook.md,
  raw/journals/260316/20260316-host-ram-enginecore-investigation.md,
  raw/journals/260318/260318_143312_ui-tars-1token-fix.md,
  raw/journals/260407/260407_081348-model-option-decision-report.md
]
---

# deploy_vlms Runtime

> VLM 인스턴스를 시작, 확인, 중지하고 runtime logs/PID와 host memory 이슈를 추적하는 운영 컴포넌트다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)

## 왜 존재하는가? (Why)

- VLM server process가 terminal/session lifecycle에 묶이면 터미널 종료 시 serving이 중단될 수 있으므로, background 실행과 PID/log 위치를 명확히 해야 한다. (source: raw/journals/260315/20260315-vlm-process-persistence.md)
- `EngineCore_DP0 died unexpectedly` 같은 런타임 실패는 GPU VRAM만이 아니라 host RAM 또는 runtime compatibility 문제일 수 있어 운영 진단 절차가 필요하다. (source: raw/journals/260316/20260316-host-ram-enginecore-investigation.md)

## 무엇인가? (What)

### 책임 범위

- VLM instance start script, stop script, health check script의 운영 흐름을 다룬다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- runtime log는 `deploy_vlms/runtime/logs/{instance}.log`, PID는 `deploy_vlms/runtime/pids/{instance}.pid`에 남기는 구조로 정리되었다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- host RAM 부족 가능성은 `free -h`, `dmesg -T`, model log를 함께 보고 분리 진단한다. (source: raw/journals/260316/20260316-host-ram-enginecore-investigation.md)
- UI-TARS는 vLLM chat template 충돌로 1 token 즉시 EOS가 발생한 기록이 있어, 명시적 chat template 설정도 runtime 진단 항목에 포함한다. (source: raw/journals/260318/260318_143312_ui-tars-1token-fix.md)

### 핵심 진입점

- `deploy_vlms/scripts/start_model.py` — model instance 시작 wrapper. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- `deploy_vlms/scripts/stop_model.py` — model process 종료. (source: raw/journals/260315/20260315-vlm-process-persistence.md)
- `deploy_vlms/scripts/check_vlm.py` — serving endpoint 확인. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- `deploy_vlms/config/chat_templates/ui-tars.jinja` — UI-TARS vLLM serving에 명시적으로 제공한 chat template. (source: raw/journals/260318/260318_143312_ui-tars-1token-fix.md)

### 의존성

- 내부: [work2-vlm-routing.md](./work2-vlm-routing.md)
- 외부: H200 GPU server, vLLM runtime, shell/session manager. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

### 데이터 모델 / 인터페이스

```text
instance name -> config/models/*.env -> runtime/pids/*.pid
instance name -> runtime/logs/*.log -> health/model readiness
```

## 어떻게 쓰는가? (How)

### 호출 예시

```powershell
uv run python deploy_vlms/scripts/start_ui_venus.py
uv run python deploy_vlms/scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
uv run python deploy_vlms/scripts/stop_model.py ui-venus
```

이 명령 묶음은 저널에서 start, check, stop의 기준 흐름으로 정리되었다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)

### 자주 쓰는 패턴

- 터미널 종료와 무관하게 유지해야 하는 serving은 background 실행 여부와 PID/log 생성을 함께 확인한다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- model crash가 발생하면 GPU VRAM, host RAM, kernel OOM, model runtime mismatch를 분리해서 본다. (source: raw/journals/260316/20260316-host-ram-enginecore-investigation.md)
- OpenAI-compatible `/v1/chat/completions`에서는 1 token만 나오지만 raw `/v1/completions`는 정상인 경우, proxy보다 vLLM chat template path를 먼저 의심한다. (source: raw/journals/260318/260318_143312_ui-tars-1token-fix.md)

### 안티패턴

- `EngineCore_DP0 died unexpectedly`를 GPU VRAM 부족으로만 단정하지 않는다. (source: raw/journals/260316/20260316-host-ram-enginecore-investigation.md)
- background process가 살아 있는지 확인하지 않고 Flask proxy 또는 automation script 문제로 바로 넘어가지 않는다. (source: raw/journals/260316/20260316-vlm-background-terminal-runbook.md)

## 참고 자료 (References)

- 원본 메모: [20260315-vlm-process-persistence.md](../../raw/journals/260315/20260315-vlm-process-persistence.md)
- 원본 메모: [20260316-vlm-background-terminal-runbook.md](../../raw/journals/260316/20260316-vlm-background-terminal-runbook.md)
- 원본 메모: [20260316-host-ram-enginecore-investigation.md](../../raw/journals/260316/20260316-host-ram-enginecore-investigation.md)
- 원본 메모: [260318_143312_ui-tars-1token-fix.md](../../raw/journals/260318/260318_143312_ui-tars-1token-fix.md)
