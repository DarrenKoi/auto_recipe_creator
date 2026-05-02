# Dynamic Screen Safety

이 문서는 dynamic screen 관련 안전 규칙을 현재 `poc/work2` 상태에 맞게 정리합니다.
중요한 점은, 현재 mainline은 아직 high-risk action automation이 아니라는 것입니다.

## 1. Dynamic Screen이 다른 이유

정적인 dialog는 latency를 어느 정도 견딜 수 있습니다.
하지만 live SEM 화면, probe monitor, 실시간 status panel은 그렇지 않습니다.

주요 위험:

- 모델 추론 중 screenshot이 바뀜
- transition 중간 상태를 잘못 읽음
- stale 좌표 기반 click이 실제 장비 상태에 영향을 줄 수 있음

따라서 dynamic screen은 정적 dialog와 다른 safety class로 다뤄야 합니다.

## 2. 현재 repo 기준 Risk 모델

### Tier 0: Launch / Read-Only Analysis

- `open_rcs.py` 같은 launch-only 흐름
- `login_rcs.py` 같은 read-only screenshot 분석
- physical action 없음

현재 `poc/work2` mainline은 사실상 여기에 머물러 있습니다.

### Tier 1: Read-Only Monitoring

- 주기적인 screenshot 분석
- 상태 및 이상 징후 탐지
- measurement 결과 판정
- physical action 없음

### Tier 2: Low-Risk UI Recovery

- 정적인 dialog button 클릭
- danger zone 밖의 dismiss/retry 성격 action

### Tier 3: High-Risk Equipment-Affecting Actions

- probe monitor target 지정
- double-click 기반 위치 조정
- recipe 값 변경 후 실제 영향이 큰 action

Tier 2 이상은 opt-in 이어야 하고, Tier 3는 가장 강한 guard rail을 둬야 합니다.

## 3. 영역 기반 안정성

전체 화면이 stable한지 묻지 말고, 관련 영역이 stable한지 물어야 합니다.

일반적인 zone class:

- `static`: menu bar, toolbar, side panel
- `dynamic`: log, status, changing data
- `danger`: probe/SEM/live target 영역

정적인 UI click 검증에서는:

- static zone 위주로 안정성을 판단합니다.
- dynamic 또는 danger zone 변화는 별도로 취급합니다.

probe 검증에서는:

- probe/SEM 영역만 따로 보고 판단합니다.

## 4. 안정성 규칙

기본 원칙:

- capture 직후 오래 기다린 action은 금지
- 모델 응답 후에는 verification capture를 다시 수행
- 관련 영역 변화가 크면 예측 좌표를 폐기

실무 기본값 예시:

- `0.3s` 간격 poll
- 최소 `2`회 연속 stable check
- `diff_ratio < 0.02`면 stable 후보
- 약 `10s` timeout 후 중단

이 수치는 사무실 Windows 장비에서 최종 조정해야 합니다.

## 5. 안전한 실행 패턴

권장 패턴:

`health check -> capture -> stabilize -> analyze -> verify -> act`

피해야 하는 패턴:

`capture -> analyze -> long delay -> click`

특히 dynamic screen에서는 verification capture가 선택사항이 아닙니다.

## 6. Double-Click Guard Rail

high-risk double-click action은 다음 조건을 모두 통과해야 합니다:

1. 기본 mainline이 아니라 opt-in action 경로일 것
2. target이 허용 zone 안에 있을 것
3. stale verification을 통과했을 것
4. action builder가 `click_count=2`를 명시적으로 허용할 것

하나라도 실패하면 차단하고 evidence를 저장합니다.

현재 mainline 문서 기준으로는 이런 action을 기본 제공한다고 가정하지 않습니다.

## 7. Probe/SEM 화면용 프롬프팅 차이

probe/SEM 계열 프롬프트는 다음을 더 명확히 해야 합니다:

- live screen이라는 사실
- current position과 target position의 구분
- 허용 영역과 금지 영역 경계
- monitor 밖 좌표 거부 규칙

OCR은 여전히 보조 수단입니다.
핵심은 움직이는 화면 위에서의 grounding과 verification입니다.

## 8. 운영 가이드

- 먼저 Tier 0과 Tier 1을 안정화합니다.
- `connection_check -> open_rcs -> login_rcs` 순서로 perception baseline을 확인한 뒤 action automation을 논의합니다.
- dynamic, static, danger zone overlay를 artifact로 남깁니다.
- high-risk action은 human approval 없이는 기본 경로에 넣지 않습니다.
- zone 비율과 안정성 threshold는 macOS가 아니라 사무실 Windows 장비에서 튜닝합니다.
