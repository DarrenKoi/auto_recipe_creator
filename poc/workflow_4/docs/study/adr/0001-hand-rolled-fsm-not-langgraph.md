---
status: accepted
---

# 워크플로 엔진은 hand-rolled FSM + mermaid snapshot 이다 (외부 라이브러리 없음)

## 결정

workflow_4 의 상태 머신 워크플로 프레임워크는 **외부 workflow 라이브러리를 쓰지
않고 hand-rolled FSM**으로 구현한다. 시각화는 자체 `render_mermaid()` 문자열 생성과
`workflow_graph.md` overwrite 스냅샷으로 한다. 신규 의존성은 **0개**다.

- `validate()` 로 그래프 일관성(미지 target, default_next 누락, terminal 미도달)을
  실행 전에 검사한다.
- retry / fallback / escalate / global budget 은 **도메인 로직**으로서 프레임워크
  코드 안에 두고 유계(bounded) 루프로 구현한다.
- 상태는 `RunState.to_json_dict()` 명시적 직렬화로 `run_state.json` 에 남기고,
  live view 는 같은 디렉터리의 `workflow_graph.md`(현재 노드 `active` 강조)로 본다.

## 맥락 / 이유

- workflow_1/2/3 의 실행 흐름을 보면 "다음 단계 이동, 실패 시 재시도, 실패 유형별
  fallback, 예산 소진 시 escalate" 패턴이 반복된다(docs/workflow_automation/02+05).
- 이 패턴을 상태 머신으로 일반화할 때, 도메인 요구(실패 분류 라우팅, retry 예산,
  pause/abort, 디버그 산출물)가 라이브러리가 주는 것보다 **훨씬 작고 단순**하다.
- 오프라인/안전 동작과 "live graph 를 보면서 실행 중 상태 확인"이 핵심 요구인데,
  그건 40줄짜리 mermaid 문자열 생성기로 충분하다.
- 프로젝트 규칙상 신규 의존성을 추가하지 않고(`pyproject.toml` 불변), Windows 없는
  macOS 개발 PC에서도 pytest 로 전부 검증되어야 한다.

## 고려한 대안

| 대안 | 검토 결과 |
|---|---|
| **LangGraph** | LLM 런타임 + 클라우드/SaaS 추적 중심. 우리는 결정적(static) 그래프 + 로컬 산출물이 필요하며, 노드 함수 시그니처도 맞지 않는다. 과함. |
| **transitions** | 상태/이벤트 정의는 깔끔하지만 retry budget, failure_class 라우팅, persist 계약이 다 우리가 만들어야 한다 — 저장된 만큼만 얻는 셈. |
| **python-statemachine** | 비슷. 콜백 훅과 상태 열거 중심이라 "실패 분류 → fallback/retry/escalate" 트리 구조를 표현하기 어렵다. |
| **pydantic-graph** | pydantic 기반 선언 그래프로 유사하나, **v2 에서 persistence 를 제거**했다. 우리의 핵심 요구(전이마다 state 파일 + live graph)가 빠져 있다. |
| **Burr** | checkpoint/UI 가 잘 갖춰져 있어 **라이브 checkpoint UI 가 필요해지면** 다음 업그레이드 경로로 유력하다. 지금은 무거움. |
| **hand-rolled FSM** | **선택.** 0 deps, 예산/라우팅이 도메인 로직으로 명시, mermaid/ascii snapshot 으로 offline viz 충분. |

## 결과 (Consequences)

- 프레임워크는 상태 저장·복원(`RunState.from_json_dict`)과 그래프 시각화를 자체
  구현하며, 외부 라이브러리 학습/버전 마찰이 없다.
- retry 예산, fallback 재방문 상한, `MAX_TRANSITIONS` 안전장치가 엔진에 명시되어
  무한 루프가 구조적으로 불가능하다.
- live checkpoint(중단점 UI + 자동 resume) 같은 기능이 필요해지면 Burr 로 이전하는
  것을 재검토한다. 그때는 `RunState.to_json_dict()` 가 마이그레이션 재료다.
- 2026-08-28 현재 데모/테스트는 `uv run pytest poc/workflow_4/` 로 전부 통과한다.