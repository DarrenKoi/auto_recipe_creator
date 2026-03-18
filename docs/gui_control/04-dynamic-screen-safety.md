# Dynamic Screen Safety

이 문서는 dynamic screen 관련 연구를 SEM/probe-monitor automation용 운영 정책으로 압축한 문서입니다.

## 1. Dynamic Screen이 다른 이유

정적인 dialog는 어느 정도 latency를 견딜 수 있습니다. 하지만 probe-monitor와 live SEM 화면은 그렇지 않습니다.

주요 위험:

- 모델 inference 중 screenshot이 바뀜
- state transition이 animation 중간 상태로 캡처됨
- 잘못된 double-click이 실제 probe 이동을 유발할 수 있음

따라서 automation 정책은 이런 화면을 별도의 safety class로 다뤄야 합니다.

## 2. 단계별 Risk 모델

### Tier 1: Read-Only Monitoring

- 주기적인 screenshot 분석
- 상태 및 이상 징후 탐지
- 물리적 action 없음

### Tier 2: Low-Risk Recovery

- danger region 밖의 정적인 UI 버튼 클릭
- 복구 가능한 UI 상태 재시도 또는 dismiss

### Tier 3: High-Risk Probe Actions

- SEM/probe 영역 target 지정
- double-click 기반 재위치 조정
- 실제 장비 상태에 영향을 줄 수 있는 action

Tier 3는 가장 강한 guard rail이 필요하며, opt-in 상태로 유지되어야 합니다.

## 3. 영역 기반 안정성

전체 화면이 안정적인지를 묻지 말고, 관련 영역이 안정적인지를 물어야 합니다.

일반적인 zone class:

- `static`: menu bar, toolbar, side panel
- `dynamic`: log, status bar, 변화하지만 위험하지는 않은 데이터
- `danger`: probe monitor 또는 기타 물리적 영향이 있는 영역

정적인 UI click의 경우:

- SEM image 영역은 무시한다
- menu/panel 영역을 기준으로 안정성을 평가한다

probe 검증의 경우:

- probe-monitor 영역에만 집중한다

## 4. 안정성 규칙

실무 기본값:

- `0.3s`마다 poll
- 최소 `2`회 연속 stable check 필요
- `diff_ratio < 0.02`이면 stable로 간주
- 약 `10s` timeout 후 중단

모델 응답 이후의 verification 규칙:

- 관련 영역이 약 `10%` 이상 바뀌었으면 예측 좌표는 stale로 간주해야 합니다
- 클릭하지 말고 다시 캡처해서 재평가해야 합니다

## 5. 안전한 실행 패턴

사용할 패턴:

`capture -> stabilize -> analyze -> verify -> act`

사용하면 안 되는 패턴:

`capture -> analyze -> click after a long delay`

dynamic screen에서는 verification capture가 선택사항이 아닙니다.

## 6. Double-Click Guard Rail

high-risk double-click action은 다음 세 조건을 모두 통과해야 합니다:

1. `SAFE_MODE=false`
2. target이 허용 zone 안에 있거나 명시적인 danger override가 존재함
3. action builder가 `click_count=2`를 명시적으로 허용함

하나라도 실패하면 차단된 action을 기록하고 evidence를 보존합니다.

## 7. Probe 화면용 프롬프팅 차이

probe-monitor 프롬프트는 다음을 포함해야 합니다:

- 이미지가 live SEM/probe frame임을 명시
- 현재 probe 위치와 target 위치를 분리해서 설명
- monitor boundary를 명확히 제시
- monitor 영역 밖 좌표는 거부

OCR hint는 보조 수단으로만 사용합니다. 핵심 작업은 여전히 움직이는 화면 위에서의 visual target grounding입니다.

## 8. 운영 가이드

- Tier 2 또는 Tier 3를 켜기 전에 먼저 Tier 1 monitoring부터 시작합니다.
- probe를 움직이거나 high-risk recipe state를 바꿀 수 있는 action에는 human approval을 유지합니다.
- static, dynamic, danger zone을 표시한 region overlay를 저장합니다.
- zone 비율 튜닝은 macOS가 아니라 사무실 Windows 장비에서 수행합니다.
