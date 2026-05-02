---
tags: [overview, project]
level: beginner
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260312/20260312-vlm-proxy-pipeline.md,
  raw/journals/260316/20260316-work2-window-search-correction.md,
  raw/journals/260331/260331_102305-login-workflow-phase1.md,
  raw/journals/260407/260407_081348-model-option-decision-report.md,
  raw/journals/260415/260415_081739-ocr-tool-list-research.md
]
---

# Auto Recipe Creator Overview

> CD-SEM/VeritySEM recipe setup을 위해 VLM 기반 화면 해석, OCR, GUI automation, stepwise workflow를 결합하는 자동화 실험 프로젝트다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)

## 왜 존재하는가? (Why)

- 수동 RCS GUI 조작을 줄이기 위해 로그인, Tool List 탐색, Tool ID 선택, 후속 검증을 자동화 가능한 작은 단계로 쪼갠다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- 화면 분석은 `UI-Venus`, `MAI-UI`, `UI-TARS`, `PaddleOCR-VL-1.5`, `GOT-OCR` 같은 모델을 역할별로 나누어 쓰는 방향으로 발전했다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- OCR 품질 문제는 prompt를 복잡하게 늘리기보다 crop, task keyword, fallback 모델을 조합해 해결하는 방향으로 정리되었다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)

## 무엇인가? (What)

### 핵심 컴포넌트

- [work2-vlm-routing.md](./components/work2-vlm-routing.md) — Flask proxy 기반 서비스 slug, endpoint, 모델 기본값을 정리하는 VLM/OCR route 계층.
- [rcs-login-automation.md](./components/rcs-login-automation.md) — RCS 실행, 로그인 창 탐색, 로그인 입력/검증 실험 흐름.
- [rcs-tool-selection.md](./components/rcs-tool-selection.md) — Tool List OCR, green box 탐지, target tool 선택 흐름.
- [workflow-runner.md](./components/workflow-runner.md) — step dependency, precondition, success criteria, artifact 저장을 담당하는 워크플로 골격.
- [deploy-vlms-runtime.md](./components/deploy-vlms-runtime.md) — VLM 인스턴스 시작, 로그, PID, RAM 이슈 진단 흐름.

### 기술 스택

- Backend: Flask proxy, OpenAI-compatible VLM route, file-based VLM request logging. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- Automation: `pywinauto`, `pynput`, Win32 foreground/window enumeration, JPEG debug artifact. (source: raw/journals/260316/20260316-work2-window-search-correction.md)
- VLM/OCR: `UI-Venus`, `MAI-UI`, `UI-TARS`, `PaddleOCR-VL-1.5`, `GOT-OCR`. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- Retrieval experiments: `CLIP`, `FAISS`, `MongoDB`, and candidate `DINOv2`/`Qdrant`. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

### 외부 의존성

- 사무실 Windows RCS 환경은 실제 GUI 동작 검증에 필요하다. (source: raw/journals/260331/260331_102305-login-workflow-phase1.md)
- H200 2장 서버는 VLM/OCR 모델 운영과 신규 모델 후보 검증의 기준 인프라로 다뤄졌다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

## 어떻게 시작하는가? (How)

### 로컬 개발 환경

- VLM/OCR route 확인은 [work2-vlm-routing.md](./components/work2-vlm-routing.md)의 `connection_check.py` 흐름을 먼저 따른다.
- RCS 로그인 실험은 [rcs-login-automation.md](./components/rcs-login-automation.md)의 `open_rcs.py` → `login_rcs.py` 또는 workflow entrypoint 순서를 따른다.

### 배포

- VLM runtime 운영은 [deploy-vlms-runtime.md](./components/deploy-vlms-runtime.md)를 기준으로 시작, 로그, PID, 종료 흐름을 확인한다.

### 자주 하는 작업

- 새 GUI 단계 추가: [workflow-runner.md](./components/workflow-runner.md)의 step 구조에 observe, action, verification 책임을 분리한다.
- Tool ID 선택 재현: [rcs-tool-selection.md](./components/rcs-tool-selection.md)에서 OCR crop과 target visibility 확인 흐름을 따른다.
- OCR 누락 대응: [ocr-task-keyword-strategy.md](./concepts/ocr-task-keyword-strategy.md)에서 `OCR:` keyword, crop, token/context 한도를 함께 점검한다.

## 주요 의사결정 (Decisions)

- 아직 `wiki/decisions/`에 합성 ADR은 없다.
- 모델 후보의 우선순위 판단은 현재 [model-and-retrieval-options.md](./concepts/model-and-retrieval-options.md)에 concept 페이지로 정리했다.

## 참고 자료 (References)

- [index.md](./index.md) — 전체 위키 목차
- [raw/journals/260312/20260312-vlm-proxy-pipeline.md](../raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- [raw/journals/260331/260331_102305-login-workflow-phase1.md](../raw/journals/260331/260331_102305-login-workflow-phase1.md)
- [raw/journals/260407/260407_081348-model-option-decision-report.md](../raw/journals/260407/260407_081348-model-option-decision-report.md)
- [raw/journals/260415/260415_081739-ocr-tool-list-research.md](../raw/journals/260415/260415_081739-ocr-tool-list-research.md)
