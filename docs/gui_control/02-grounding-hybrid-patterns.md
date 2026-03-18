# Grounding 및 Hybrid Automation 패턴

이 문서는 기존 automation 전략, Microsoft vision-tool 조사, hybrid pattern 메모, 그리고 더 넓은 engineering GUI automation 연구를 하나의 운영 가이드로 합친 문서입니다.

## 1. 기본 루프

이 저장소의 기본 루프는 다음과 같습니다:

1. 현재 screenshot을 observe한다
2. 필요한 최소 도구 조합으로 다음 action을 decide한다
3. mouse 또는 keyboard control로 act한다
4. 새로운 screenshot으로 결과를 verify한다

이보다 낙관적인 접근은 RCS 계열 소프트웨어에서 쉽게 깨집니다.

## 2. 도구별 역할 분리

| Tool | 가장 적합한 역할 | 이렇게 쓰는 것은 피할 것 |
| ------ | ------------------ | -------------------------- |
| `UI-Venus` | full-screen GUI grounding과 semantic target 선택 | exact text authority |
| `UI-TARS` | 대체 primary model, multi-step action reasoning | OCR 대체품 |
| `MAI-UI` | 작은 crop에 대한 zoom-in grounding sidecar | 항상 켜두는 full-screen primary |
| `PaddleOCR-VL-1.5` | text, spotting, layout, table reading | primary click planner |
| `GOT-OCR-2.0-hf` | 어려운 crop용 OCR fallback | 범용 GUI agent |
| `OmniParser V2` | SoM 스타일 parser, interactable box, icon caption | standalone workflow planner |
| `pywinauto` / `uiautomation` | 접근 가능한 widget에 대한 object control | custom-rendered UI parser |
| `pynput` | 최종 action 실행 | state 이해 |

## 3. 권장 Hybrid 패턴

### 3.1 Accessible-Control First

target이 일반적인 editable field이거나 신뢰할 수 있는 dialog control일 때 사용합니다.

흐름:

1. object lookup을 먼저 시도한다
2. object lookup이 실패하거나 control이 ambiguous할 때만 vision을 사용한다
3. action 후에는 여전히 verify한다

### 3.2 Full-Screen Grounding -> Crop Retry -> OCR Refinement

control이 보이기는 하지만 작거나, 혼잡하거나, 텍스트 의존성이 있을 때 사용합니다.

흐름:

1. 전체 screenshot에 `UI-Venus` 또는 `UI-TARS`를 적용한다
2. 예측된 영역 주변을 crop한다
3. 보통 `MAI-UI` 또는 동일한 primary model로 crop에 대해 grounding을 다시 수행한다
4. 텍스트 정밀도가 중요하면 crop에 OCR을 적용한다
5. 증거를 병합하고, verification 후에만 클릭한다

이 패턴은 밀도 높은 engineering screen에서 가장 중요합니다.

### 3.3 OmniParser / SoM Sidecar

UI에 작은 interactable 영역이 많거나 icon-only control이 있는 경우 사용합니다.

흐름:

1. OmniParser를 실행한다
2. 파싱된 box를 compact hint 또는 SoM overlay로 변환한다
3. primary VLM이 어떤 표시된 element가 중요한지 결정하게 한다
4. 선택된 영역을 실제 pixel 좌표로 되돌려 실행한다

실무 메모:

- OmniParser의 장점은 모든 모델을 대체하는 데 있지 않고, box와 interactability 정보를 추가해준다는 점에 있습니다.
- YOLO detection component는 AGPL-3.0 이슈가 있으므로 production 배포 전 검토가 필요합니다.

### 3.4 Try-Catch Fallback Chain

합리적인 escalation 순서는 다음과 같습니다:

1. object API
2. structured parser 또는 crop retry
3. full VLM reasoning
4. confidence가 계속 낮으면 human review

화면이 완전히 inaccessible하다고 이미 확인된 경우가 아니라면 이 순서를 뒤집지 않습니다.

## 4. 상태 관리

워크플로우가 알려져 있다면, 이 저장소는 open-ended ReAct loop보다 산업형 패턴을 우선해야 합니다.

### 4.1 알려진 워크플로우에는 State Machine 우선

RCS recipe 작업은 unrestricted desktop agent 문제라기보다 다음에 더 가깝습니다:

- `login_screen`
- `main_menu`
- `recipe_editor`
- `parameter_dialog`

state machine 기반 실행은 더 저렴하고, 검증하기 쉽고, 안전하게 만들기도 쉽습니다.

### 4.2 안정적인 절차에는 Plan-Then-Execute 사용

recipe flow가 알려져 있다면 stage 순서를 한 번 계획해두고, VLM은 주로 다음 역할에 사용합니다:

- state recognition
- target grounding
- anomaly detection
- post-action verification

### 4.3 History는 작고 유용하게 유지

유용한 history:

- 최근 screenshot
- 현재 state label
- 마지막 action
- 해결되지 않은 ambiguity 메모

유용하지 않은 history:

- 긴 raw OCR dump
- 이전 turn들의 전체 prompt transcript
- 반복되는 정책 문구 사본

## 5. 신뢰성 패턴

### 5.1 좌표만이 아니라 Layout도 캐시

화면 layout이 안정적이라면 다음을 캐시합니다:

- screen hash
- target region
- target label
- 마지막 성공 방법

의미 있는 layout 변경이 생기면 다시 검증합니다.

### 5.2 의사결정과 실행 분리

모델이 mouse를 직접 "소유"하게 하면 안 됩니다. 모델은 후보 action을 만들어야 하고, 그 action은 다음 기준으로 점검 가능해야 합니다:

- 현재 state
- 허용된 action type
- danger zone
- 최근 화면 변화

### 5.3 최종 승인에는 Human-In-The-Loop

action이 recipe 값 변경, measurement 실행, probe 이동에 가까워질수록 수동 확인 요구사항도 더 강해져야 합니다.

## 6. 이 저장소를 위한 실무 가이드

- `poc/work2/flask_vlm.py`와 `poc/work2/vlm_client.py`를 클라이언트 계약면으로 사용합니다.
- automation 로직을 디버깅하기 전 `poc/work2/connection_check.py`를 먼저 사용합니다.
- 서비스 간 정면 비교에는 `poc/work2/login_benchmark.py`를 사용합니다.
- OCR hint는 보조 컨텍스트로 취급하고, pixel 기반 grounding의 대체재로 취급하지 않습니다.
- source JPEG, 전송한 WebP, raw response, overlay image, call log를 포함한 debug artifact를 중요 자산으로 유지합니다.
