# View 탭 내 Embedded Align Fail Monitoring

이 문서는 RCS `View` 탭에서 각 tool 의 작은 모니터링 box 안에 나타나는 `align fail` 알람을 어떻게 감지할지에 대한 구현/테스트 계획을 정리합니다.

중요한 전제는 다음과 같습니다:

- 알람은 RCS 자체 popup 이 아닙니다.
- 알람은 각 tool 의 live preview 안에서 보입니다.
- `View` 탭은 5열 matrix 형태이며, tool 수가 많으면 scroll 이 필요합니다.
- 각 tool preview 는 2~3초 간격으로 바뀌므로 단순 frame diff 만으로는 false positive 가 많이 납니다.

즉, 이 문제는 "RCS popup 탐지"가 아니라 "작은 remote preview 안의 embedded alarm 탐지"로 정의해야 합니다.

## 1. 핵심 전략

메인 전략은 전체 `View` 화면을 계속 VLM 에 보내는 방식이 아닙니다.

권장 구조:

1. 전체 `View` 화면은 layout 파악과 visible tile 위치 추출에만 사용합니다.
2. 각 tool 을 tile 단위로 분리합니다.
3. 실제 감지는 tile 의 `preview` 영역 기준으로 수행합니다.
4. 모든 tile 을 매 cycle VLM 에 보내지 않습니다.
5. 먼저 로컬 heuristic 로 suspicious tile 을 추립니다.
6. suspicious tile 만 OCR 로 확인합니다.
7. OCR 이 애매한 경우에만 VLM fallback 을 사용합니다.

이 방식을 쓰는 이유:

- full image 에서는 `align fail` text 가 너무 작게 보일 수 있습니다.
- 20개 이상 tool 을 full-screen VLM 으로 계속 감시하면 비용과 latency 가 과도합니다.
- normal SEM preview 변화와 실제 alarm emergence 를 분리하려면 tile 기준 시간축 비교가 필요합니다.

## 2. v1 목표

첫 버전 목표는 다음입니다:

- `View` 탭 전체를 read-only 로 순회한다.
- visible tile 들의 `tool_id` 를 안정적으로 읽는다.
- 각 tile preview 안의 `align fail` 알람 후보를 감지한다.
- false positive 를 줄이기 위해 repeated hit 기반으로 confirm 한다.
- confirmed 결과를 `tool_id` 와 evidence artifact 와 함께 저장한다.

v1 범위에서 하지 않는 것:

- tool open 자동 실행
- alarm dismiss/recovery action
- recipe 값 변경
- high-risk click automation

즉, v1 은 Tier 1 read-only monitoring 으로 유지합니다.

## 3. 현재 repo 기준 연결점

현재 repo 에서 바로 재사용할 수 있는 기반:

- `poc/work2/view_list_tab_rcs.py`
  - `View` 탭 진입 baseline
- `poc/work2/scan_tools_from_view.py`
  - green box 기반 visible tool 스캔 baseline
  - scroll 순회 baseline
  - debug artifact 저장 baseline
- `poc/work2/util/`
  - capture, crop, debug image, mouse/scroll helper
- `poc/work2/vlm_client.py`
  - OCR/VLM 공통 호출 계약
- `docs/gui_control/04-dynamic-screen-safety.md`
  - dynamic screen 안전 규칙

다만 이번 문제는 `scan_tools_from_view.py` 와 다르게, green title box 아래를 더블클릭하는 것이 목적이 아니라 tile preview 내부의 alarm 을 읽는 것이 목적입니다.

## 4. 제안 엔트리포인트

새 read-only 스크립트:

- `poc/work2/monitor_view_align_fail.py`

역할:

1. 메인 RCS 창 탐색
2. `View` 탭 열기
3. 현재 page 의 tile/header 추출
4. tile 별 preview monitoring
5. suspicious tile 에 대해서만 OCR/VLM 확인
6. scroll 하며 전체 tool set 순회
7. alert summary 및 debug artifact 저장

## 5. tile 분해 방식

각 visible tool tile 에 대해 최소 아래 정보를 유지합니다:

- `tool_id`
- `header_bbox`
- `tile_bbox`
- `preview_bbox`
- `page_index`
- `row_index`
- `col_index`

논리 영역은 다음 두 개로 나눕니다:

- `header_zone`
  - tool 식별용
  - 비교적 static
- `preview_zone`
  - 실제 live SEM/tool image 가 들어오는 영역
  - alarm 감시 대상

필요하면 추가 zone 도 둘 수 있습니다:

- `alarm_candidate_zone`
  - preview 전체 또는 중앙/하단 일부
  - popup/overlay 가 자주 생기는 위치를 좁힐 때 사용

## 6. header / tile 검출 전략

초기 단계에서는 기존 `scan_tools_from_view.py` 구조를 재사용하되 다음을 확장합니다:

- green title box 뿐 아니라 white title box 도 검출
- detected header box 를 reading order 로 정렬
- header OCR 로 `tool_id` 확보
- matrix 규칙을 사용해 tile bbox 와 preview bbox 를 계산

실무적으로는 다음 순서가 적절합니다:

1. content 영역 crop
2. title/header box 검출
3. 각 header box OCR
4. `tool_id` 정규화
5. header 와 spacing 을 기준으로 tile bbox 추론
6. tile 내부에서 preview bbox 계산

중요:

- full page VLM 으로 `align fail` 자체를 읽으려 하지 않습니다.
- full page 는 tile layout 을 안정적으로 자르는 용도로만 씁니다.

## 7. per-tool 상태 관리

모니터링은 page 단위가 아니라 `tool_id` 단위 상태 저장이 핵심입니다.

각 tool 별로 유지할 상태 예시:

- 최근 preview crop 2~4개
- 최근 suspicion score
- 최근 OCR text
- 최근 VLM result
- consecutive positive hit 수
- last_seen timestamp
- last_alerted timestamp
- current page index

이 상태가 필요한 이유:

- scroll 후 다시 같은 tool 이 보일 때 이력을 이어받아야 합니다.
- normal preview 변화와 persistent alarm 을 시간축으로 구분해야 합니다.
- repeated alert spam 을 막아야 합니다.

## 8. detection pipeline

권장 감지 순서는 다음과 같습니다:

`capture -> split tiles -> cheap candidate detection -> OCR confirm -> VLM fallback -> repeated-hit confirm -> alert`

### 8.1 capture / settle

현재 page 에 대해 먼저 짧은 settle check 를 둡니다.

예:

- `0.3s` 간격으로 2회 capture
- header/layout 이 크게 바뀌지 않았을 때만 본 cycle 분석 수행

이 단계의 목적은 scroll 직후 transition 중간 화면이나 focus 흔들림을 피하는 것입니다.

### 8.2 cheap candidate detection

preview 는 원래 계속 바뀌므로, 단순 absdiff 총량만 보면 noise 가 큽니다.

그래서 후보 감지는 아래 신호를 조합합니다:

- frame-to-frame diff ratio
- 새로 생긴 rectangular block
- 새로 생긴 text-like region
- preview 일부를 가리는 modal-like overlay
- warning/error tone 성격의 색상 patch
- 2회 이상 반복되는 persistence

핵심은:

- 일반적인 measurement/live preview motion 은 무시
- "갑자기 나타난 글자/box/overlay" 에만 점수를 주기

### 8.3 OCR confirm

suspicious tile 에 대해서만 OCR 을 수행합니다.

1차 타겟 키워드:

- `ALIGN FAIL`
- `ALIGNFAIL`
- `ALIGNMENT FAIL`

정규화 규칙:

- 대문자화
- 공백/특수문자 제거
- 부분 OCR 손상에 대비한 variant 매칭

추가 keyword 는 `.env` 로 확장 가능하게 둡니다.

### 8.4 VLM fallback

OCR 이 약하거나 text 가 일부만 보일 때만 tile crop 을 VLM 에 보냅니다.

질문은 좁게 유지합니다:

- 이 tool preview 내부에 alarm/message box 가 보이는가
- 그 메시지가 `align fail` 계열인가
- visible text 는 무엇인가
- confidence 는 얼마인가

strict JSON 예시:

```json
{
  "has_alarm": true,
  "is_align_fail": true,
  "alarm_text": "ALIGN FAIL",
  "confidence": 0.92,
  "reason": "A persistent dialog-like overlay with visible align fail text is present inside the tool preview."
}
```

## 9. alert confirm 규칙

false positive 를 줄이기 위해 즉시 alert 하지 않습니다.

기본 규칙:

- 같은 `tool_id` 에서 2회 연속 positive hit 시 alert
- 또는 strong OCR hit 1회 + 다음 cycle repeated suspicious frame 1회

이 규칙을 두는 이유:

- live preview 잡음으로 인한 one-frame anomaly 차단
- scroll/refresh 순간의 transient artifact 차단

추가 guard:

- 이미 alert 된 ongoing alarm 은 cooldown 동안 중복 report 억제
- 상태가 clear 되면 alert state reset

## 10. scroll monitoring 전략

tool 이 20개 이상이면 한 page 만 보면 안 됩니다.

권장 순서:

1. 현재 visible page 분석
2. page 당 N cycle monitoring
3. 한 page 아래로 scroll
4. 다음 visible page 분석
5. bottom 도달 시 top 으로 복귀
6. 다시 반복

이때 중요한 점:

- dedupe 는 scroll index 가 아니라 `tool_id` 기준으로 합니다.
- page 간 위치가 바뀌어도 같은 tool 은 같은 상태 객체를 사용합니다.
- bottom detection 은 page signature 또는 visible `tool_id` 반복으로 판단할 수 있습니다.

## 11. artifact 표준

각 cycle 또는 alert 마다 최소 아래 산출물을 남깁니다:

- full page JPEG
- content crop JPEG
- tile/header overlay JPEG
- suspicious tile JPEG
- suspicious tile WebP payload
- OCR raw text
- OCR parsed JSON
- VLM raw response
- VLM parsed JSON
- per-tool state summary JSON
- confirmed alert summary JSON

alert summary 에는 최소 아래 정보가 있어야 합니다:

- `tool_id`
- `page_index`
- `tile_bbox`
- `preview_bbox`
- `first_seen_at`
- `last_seen_at`
- `consecutive_hits`
- `ocr_text`
- `vlm_alarm_text`
- `confidence`
- `evidence_paths`

## 12. 왜 full-screen VLM continuous monitoring 이 메인 전략이 아닌가

질문의 핵심 중 하나는:

- tile 을 쪼개서 계속 VLM 에 보내야 하는가
- 아니면 전체 이미지를 읽고 `align fail` alarm box 존재 여부를 비교할 수 있는가

현재 권장 답은 다음과 같습니다:

- 전체 이미지는 layout 파악용으로만 사용
- 실제 알람 판독은 tile 기준으로 수행
- VLM continuous full-screen monitoring 은 비권장

이유:

- tile 내부 text 가 너무 작아 full-screen 에서 놓칠 가능성이 큼
- 20개 이상 tool 에서 full-screen semantic compare 는 noisy 함
- live SEM 변화가 많아 semantic drift 가 큼
- 비용/latency 대비 신호 품질이 좋지 않음

정리하면:

- `full image`: 어디에 어떤 tile 이 보이는지 파악
- `tile crop`: 실제 alarm 판정
- `OCR`: 1차 text authority
- `VLM`: fallback semantic confirmer

## 13. 단계별 구현 순서

### Phase 1. Layout Baseline

- `View` 탭 진입 자동화 재사용
- green/white header box 검출 안정화
- visible tile bbox / preview bbox 추출
- `tool_id` OCR 안정화

목표:

- 현재 page 에서 visible tool 목록을 신뢰 가능하게 얻기

### Phase 2. Preview Candidate Detector

- preview crop 기준 frame diff 실험
- rectangular overlay emergence heuristic 추가
- text-like region emergence heuristic 추가
- ordinary SEM 변화와 alarm-like 변화 분리

목표:

- suspicious tile 만 선별 가능하게 만들기

### Phase 3. OCR Confirm

- suspicious tile OCR
- `align fail` keyword normalization
- repeated hit state machine 추가

목표:

- text 기반 confirmed detection 확보

### Phase 4. VLM Fallback

- OCR ambiguous tile 에만 VLM 사용
- strict JSON parsing
- low-confidence reject

목표:

- OCR 실패 case 보완

### Phase 5. Full Scroll Monitor

- page 순회
- bottom detection
- revisit dedupe
- cooldown / re-alert policy

목표:

- 20개 이상 tool 에서 장시간 monitoring 가능

## 14. 테스트 계획

### 14.1 Gold dataset 수집

사무실 Windows 환경에서 다음을 수집합니다:

- 정상 no-alarm View screenshot / 짧은 clip
- tool preview 내부에 `align fail` 이 보이는 positive 예시
- 상단/중단/하단 scroll page 예시

### 14.2 Tile extraction 테스트

검증 항목:

- green/white header box 검출 성공률
- header reading order 정렬 정확도
- `tool_id` OCR 복원률
- tile bbox / preview bbox 계산 정확도

### 14.3 No-alarm 안정성 테스트

정상 live preview 변화만 있는 sequence 에 대해:

- suspicion score 가 누적되지 않는지
- repeated false alert 가 없는지 확인

### 14.4 Embedded alarm candidate 테스트

positive sequence 에 대해:

- preview 내부 box/text emergence 가 감지되는지
- ordinary SEM 변화와 구분되는지
- 최소 2-frame persistence 조건이 먹는지 확인

### 14.5 OCR confirm 테스트

검증 항목:

- `ALIGN FAIL` 정규화 매칭
- spacing/noise 변형 대응
- unrelated text false hit 방지

### 14.6 VLM fallback 테스트

unit test 에서는 mock 응답으로 검증합니다:

- OCR weak case 에만 호출되는지
- malformed JSON reject 되는지
- low-confidence 응답이 alert 로 승격되지 않는지

### 14.7 Per-tool state 테스트

검증 항목:

- scroll 후 재등장한 tool 이 같은 state 로 이어지는지
- ongoing alert 에 대해 spam 이 발생하지 않는지
- clear 후 재발생 시 새 alert 로 전환되는지

### 14.8 Scroll coverage 테스트

검증 항목:

- 20개 이상 tool 에서 전체 순회 가능 여부
- bottom detection 동작
- top restart 동작
- page signature 반복 처리 정확도

### 14.9 Read-only safety 테스트

기본 조건:

- `SAFE_MODE=true`
- tool open 없음
- dismiss/retry click 없음
- monitor + artifact 저장만 수행

### 14.10 Acceptance 기준

v1 acceptance:

- no-alarm dataset 에서 false alert 가 없어야 함
- visible `align fail` positive 는 올바른 `tool_id` 와 함께 report 되어야 함
- miss / false positive 분석에 필요한 artifact 가 남아야 함
- scroll 을 포함한 전체 View 순회가 지속적으로 동작해야 함

## 15. 기본 가정과 기본값

- alarm 은 RCS popup 이 아니라 tool preview 내부에 나타납니다.
- v1 은 View-only, read-only 로 유지합니다.
- primary confirm path 는 OCR 입니다.
- VLM 은 fallback 입니다.
- polling cadence 는 preview 갱신 속도를 고려해 page 당 대략 1~2초에서 시작합니다.
- 기본 alert confirm 규칙은 같은 tool 에서 2회 연속 positive detection 입니다.

## 16. 실무 결론

질문에 대한 직접적인 결론은 다음입니다:

- 전체 이미지를 계속 VLM 에 보내며 `align fail` box 존재 여부를 비교하는 방식은 메인 전략으로 권장하지 않습니다.
- 각 tool preview 를 tile 단위로 분리해 모니터링하는 방식이 맞습니다.
- 하지만 모든 tile 을 매번 VLM 에 보내는 것은 비효율적이므로,
  - 먼저 local candidate detector 로 suspicious tile 을 좁히고,
  - OCR 로 1차 확인하고,
  - 정말 애매한 경우에만 VLM 을 fallback 으로 붙이는 구조가 가장 현실적입니다.

즉, 권장 아키텍처는 다음 한 줄로 요약됩니다:

`full-page layout detection -> per-tile preview monitoring -> OCR confirm -> VLM fallback -> repeated-hit alert`
