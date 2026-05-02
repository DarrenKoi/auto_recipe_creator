---
tags: [component, rcs, login, automation, work2]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260316/20260316-login-rcs-coordinate-contract.md,
  raw/journals/260316/20260316-work2-rcs-window-focus.md,
  raw/journals/260316/20260316-work2-rebuild-boundary-update.md,
  raw/journals/260316/20260316-work2-window-search-correction.md,
  raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md,
  raw/journals/260323/260323_140321-login-rcs-experiment-update.md,
  raw/journals/260323/260323_151344-login-input-replace-fix.md
]
---

# RCS Login Automation

> RCS 실행, 로그인 창 탐색, 로그인 화면 좌표 grounding, 입력/검증을 단계별로 다루는 자동화 컴포넌트다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)

## 왜 존재하는가? (Why)

- RCS 로그인은 창 탐색, foreground 활성화, screenshot capture, target coordinate detection, 입력/클릭, main window 검증이 모두 맞아야 하므로 단일 거대 스크립트보다 단계별 실험이 필요했다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- 로그인 입력 필드에는 기존 텍스트 위에 append 되는 문제가 있었고, double click 후 `Ctrl+A`/delete/typing 계열의 덮어쓰기 흐름으로 안정화했다. (source: raw/journals/260323/260323_151344-login-input-replace-fix.md)

## 무엇인가? (What)

### 책임 범위

- `open_rcs.py`는 RCS executable을 열고 이미 열린 창을 정상 상태로 처리하는 baseline launch 단계다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- rebuild boundary 정리 후 work2 설명은 `connection_check.py`, `open_rcs.py`, `login_rcs.py` 중심으로 축소되었고, 오래된 combined login automation script는 제거 대상으로 기록되었다. (source: raw/journals/260316/20260316-work2-rebuild-boundary-update.md)
- `login_rcs.py`와 `login_rcs_common.py`는 상태 파일 PID 우선 탐색과 desktop scan fallback으로 로그인 창을 찾는 흐름을 제공한다. (source: raw/journals/260323/260323_140321-login-rcs-experiment-update.md)
- login target detection은 `UI-Venus`, `MAI-UI`, `UI-TARS`, `PaddleOCR-VL-1.5`, `GOT-OCR` 등 여러 실험 축으로 분리되었다. (source: raw/journals/260323/260323_140321-login-rcs-experiment-update.md)
- 좌표 응답 계약은 `coord_system="relative_1000"` 기반 클릭 좌표로 정리되었다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- UI-Venus login detection은 공식 단일 요소 prompt로 전환되었고, 요소당 1회 VLM 호출과 `[x, y]` parser를 사용하는 방향으로 단순화되었다. (source: raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)

### 핵심 진입점

- `poc/work2/open_rcs.py` — RCS 실행과 이미 열린 창 처리. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- `poc/work2/login_rcs.py` — 로그인 창 benchmark/detection entrypoint. (source: raw/journals/260323/260323_140321-login-rcs-experiment-update.md)
- `poc/work2/action_login.py` — 로그인 액션 entrypoint로 쓰인 흐름. (source: raw/journals/260323/260323_151344-login-input-replace-fix.md)

### 의존성

- 내부: [work2-vlm-routing.md](./work2-vlm-routing.md), [gui-coordinate-and-window-focus.md](../concepts/gui-coordinate-and-window-focus.md), [workflow-runner.md](./workflow-runner.md)
- 외부: Windows RCS GUI, Win32 foreground APIs, `pywinauto`, `pynput`. (source: raw/journals/260316/20260316-work2-window-search-correction.md)

### 데이터 모델 / 인터페이스

```text
open_rcs -> open_rcs_state.json -> login_rcs_common.find_login_window
login target -> relative_1000 image coordinate -> screen coordinate -> click/type
login submit -> wait_for_rcs_main_window -> verification
```

## 어떻게 쓰는가? (How)

### 호출 예시

```powershell
uv run python poc/work2/open_rcs.py
uv run python poc/work2/login_rcs.py
uv run python poc/work2/action_login.py
```

이 순서는 RCS launch, login dialog analysis, login action을 단계적으로 검증하는 흐름으로 저널에 반복해서 남아 있다. (source: raw/journals/260323/260323_140321-login-rcs-experiment-update.md)

### 자주 쓰는 패턴

- capture 직전 로그인 창을 foreground로 다시 올리고 실제 foreground handle 일치 여부까지 확인한다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- 입력 필드는 단일 클릭보다 double click 후 지우기와 typing을 조합해 append 문제를 줄인다. (source: raw/journals/260323/260323_151344-login-input-replace-fix.md)
- OCR-only, grounding-only, two-stage targeting script를 분리해 어떤 모델 조합이 안정적인지 비교한다. (source: raw/journals/260323/260323_140321-login-rcs-experiment-update.md)
- navigation mode는 모델이 OS를 직접 조작하는 방식이 아니라 action proposal을 local executor가 1 step씩 실행하고 post-action verification을 붙이는 구조로 제한한다. (source: raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)

### 안티패턴

- PID 검증 없이 desktop 전체 scan 결과만 믿고 다른 RCS instance를 선택하지 않는다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- image coordinate를 screen coordinate로 변환하지 않은 상태에서 바로 클릭하지 않는다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)

## 참고 자료 (References)

- 원본 메모: [20260316-login-rcs-coordinate-contract.md](../../raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- 원본 메모: [20260316-work2-rcs-window-focus.md](../../raw/journals/260316/20260316-work2-rcs-window-focus.md)
- 원본 메모: [20260316-work2-rebuild-boundary-update.md](../../raw/journals/260316/20260316-work2-rebuild-boundary-update.md)
- 원본 메모: [260318_163432_ui-venus-official-grounding-overhaul.md](../../raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)
- 원본 메모: [260323_140321-login-rcs-experiment-update.md](../../raw/journals/260323/260323_140321-login-rcs-experiment-update.md)
- 원본 메모: [260323_151344-login-input-replace-fix.md](../../raw/journals/260323/260323_151344-login-input-replace-fix.md)
