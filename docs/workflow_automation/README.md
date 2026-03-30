# 워크플로 엔진과 재시도 전략

이 문서는 `poc/work2/`의 GUI 자동화를 단일 스크립트 실행에서 **순차 워크플로 엔진**으로 발전시키는 설계를 다룹니다.
핵심 질문은 세 가지입니다:

1. 순서대로 클릭/입력을 진행하면서 각 단계의 성공 여부를 어떻게 판단하는가?
2. 실패했을 때 변형을 주며 재시도하는 것이 가능한가?
3. 진행 상태를 "기억"하며 이어서 진행할 수 있는가?

결론부터 말하면: **모두 구현 가능하고, 기존 인프라를 상당 부분 재사용할 수 있습니다.**

## 문서 구조

| 문서 | 내용 |
|------|------|
| [01-current-state-analysis.md](01-current-state-analysis.md) | 현재 상태 분석 — 기존 구조의 한계와 워크플로 원형 |
| [02-engine-architecture.md](02-engine-architecture.md) | 워크플로 엔진 아키텍처 — Step 조건 타입, 상태 머신, WorkflowRunner |
| [03-post-action-verification.md](03-post-action-verification.md) | VLM 기반 후행 검증 — Before/After 비교, poll-until-stable, Hybrid 검증 |
| [04-retry-strategy.md](04-retry-strategy.md) | 재시도 전략 — failure-aware routing, jitter, crop-retry, model fallback, 예산 관리 |
| [05-state-management.md](05-state-management.md) | 워크플로 메모리 / 상태 관리 — StepResult, Workflow Run, Checkpoint/Resume |
| [06-implementation-blueprint.md](06-implementation-blueprint.md) | 구현 청사진 — 새 모듈, 기존 모듈 연동, 단계별 구현 순서 |
| [07-safety-rules-and-scope.md](07-safety-rules-and-scope.md) | Safety Rules + v1 범위 |

## 관련 문서

- [docs/gui_control/04-dynamic-screen-safety.md](../gui_control/04-dynamic-screen-safety.md) — safe zone, safety tier 정의
- [docs/gui_control/06-view-tab-embedded-align-fail-monitoring.md](../gui_control/06-view-tab-embedded-align-fail-monitoring.md) — align-fail 모니터링
- [docs/gui_control/07-window-overlap-and-hidden-title-strategy.md](../gui_control/07-window-overlap-and-hidden-title-strategy.md) — 윈도우 겹침/숨김 처리
