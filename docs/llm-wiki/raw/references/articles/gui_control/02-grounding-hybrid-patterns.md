# Grounding 및 Hybrid Automation 패턴

이 문서는 현재 `poc/work2` mainline에서 실제로 쓰는 패턴과, 다음 단계로 붙일 hybrid 확장 패턴을 함께 정리합니다.
핵심은 "지금 구현된 경로"와 "도입 후보"를 섞지 않는 것입니다.

## 1. 현재 기본 루프

현재 저장소에서 권장하는 기본 루프는 다음과 같습니다:

1. `connection_check.py`로 서비스 상태를 확인한다
2. `open_rcs.py`로 대상 프로그램을 띄운다
3. `login_rcs.py`로 로그인 창을 observe한다
4. 여러 GUI 서비스를 같은 이미지로 benchmark한다
5. overlay, raw response, parsed JSON을 verify한다
6. 실제 action이 필요하면 그 다음 단계에서만 act한다

즉, 현재 mainline은 "perception 계약을 먼저 안정화"하는 구조입니다.

## 2. 도구별 역할 분리

| Tool | 현재 가장 적합한 역할 | 이렇게 쓰는 것은 피할 것 |
| ------ | -------------------- | ------------------------ |
| `ui-venus` | 현재 기본 GUI grounding 서비스 | exact text authority |
| `ui-tars` | GUI grounding 비교군, action-style prompt 실험 | OCR 대체품 |
| `kimi-k2.5` | direct 회사 GUI 비교군 | latency 검증 없이 바로 실행 주체로 쓰기 |
| `qwen3-vl-30b-instruct` | direct 회사 GUI 비교군 | OCR 대체품 |
| `mai-ui` | crop-retry용 zoom-in sidecar 후보 | 항상 켜두는 full-screen 기본 모델 |
| `paddleocr-vl-1.5` | 기본 OCR, text parsing, spotting 확인 | primary click planner |
| `got-ocr` | 어려운 crop 또는 box 지정 OCR fallback | 범용 GUI agent |
| `pywinauto` / `uiautomation` | 접근 가능한 widget 탐색 | custom-rendered UI parser |
| `pynput` | 최종 mouse/keyboard 실행 | state 이해 |

## 3. 현재 권장 Hybrid 패턴

### 3.1 Mainline: Multi-Model Benchmark First

현재 로그인 창 분석의 기본 패턴입니다.

흐름:

1. `connection_check.py`로 health를 확인한다
2. `open_rcs.py`로 RCS를 실행한다
3. `login_rcs.py`가 로그인 창을 캡처한다
4. `login_benchmark.py`가 primary GUI 서비스들에 같은 screenshot을 보낸다
5. 결과 overlay와 raw JSON을 비교해 어떤 서비스가 더 안정적인지 본다

이 패턴의 장점:

- prompt drift와 serving 문제를 분리할 수 있음
- direct 모델과 proxy 모델을 같은 artifact 계약으로 비교할 수 있음
- action을 붙이기 전 grounding 품질을 먼저 올릴 수 있음

현재 서비스 선택은 `RCS_LOGIN_SERVICE_SLUGS` 환경변수 또는 `primary_gui` role 기본값을 따릅니다.

### 3.2 OCR Sidecar Validation

OCR은 현재 mainline에서 click planner가 아니라 evidence sidecar입니다.

흐름:

1. `login_rcs.py` 또는 기존 캡처 이미지로 screenshot을 확보한다
2. `ocr_login_check.py`를 실행한다
3. `PaddleOCR-VL-1.5`의 `OCR:` 와 `Spotting:`를 비교한다
4. 필요하면 `GOT-OCR` box OCR로 특정 영역을 재확인한다
5. 텍스트 evidence만 가져오고, 최종 click target 결정은 GUI grounding에 남긴다

실무 메모:

- `OCR:`는 텍스트 파싱 확인용에 가깝습니다.
- `Spotting:`는 위치 힌트가 나올 수 있지만 click grounding 계약은 아닙니다.
- `GOT-OCR`는 특정 crop 판독 fallback 으로 해석해야 합니다.

### 3.3 Full-Screen Grounding -> Crop Retry -> OCR Refinement

이 패턴은 여전히 유효하지만, 현재 repo mainline에 통합된 기본 경로는 아닙니다.

도입 시 권장 흐름:

1. full-screen GUI grounding으로 첫 후보를 얻는다
2. 후보 영역 주변을 crop한다
3. crop에서 `mai-ui` 또는 같은 GUI 서비스로 재시도한다
4. text disambiguation이 필요하면 OCR을 추가한다
5. evidence를 병합하고 strategy를 JSON에 기록한다

이 패턴은 다음 상황에서 특히 중요합니다:

- 작은 toolbar button
- 밀집된 parameter grid
- label은 OCR로 읽히지만 clickable area는 별도인 화면

### 3.4 Accessible-Control First

다음 경우에는 object-based 접근을 먼저 시도할 가치가 있습니다:

- 표준 edit control
- 명확한 dialog button
- UIA 정보가 안정적으로 보이는 화면

하지만 현재 `poc/work2` mainline은 object-first 구조가 아니라 screenshot-first 구조입니다.
즉, object API는 "먼저 써야 하는 기본 경로"가 아니라 "확실히 먹히는 화면에서의 cheap shortcut"에 가깝습니다.

### 3.5 Human Review Escalation

합리적인 escalation 순서는 다음과 같습니다:

1. 서비스 연결 상태 확인
2. full-screen benchmark
3. OCR sidecar
4. crop retry 또는 object fallback
5. still ambiguous 이면 human review

고위험 화면에서 이 순서를 건너뛰면 debug보다 사고가 먼저 납니다.

## 4. 상태 관리

### 4.1 현재 repo에 맞는 상태 단위

현재 문서 기준으로 중요한 상태 단위는 다음과 같습니다:

- `login_dialog`
- `main_tabs`
- `screen_analysis`
- `measurement_judgment`

이 naming은 `poc/work2/prompts/`의 빌더 구조와도 맞춰 두는 편이 낫습니다.

### 4.2 History는 작고 유용하게 유지

유용한 history:

- 최근 screenshot
- 현재 service slug
- 마지막 성공 artifact
- unresolved ambiguity 메모

유용하지 않은 history:

- 긴 raw OCR dump 전체
- 모든 이전 prompt transcript
- 모델별 장황한 설명문 복사본

### 4.3 모델이 마우스를 직접 소유하면 안 된다

모델은 candidate action 또는 candidate coordinate를 내야 합니다.
실행 레이어는 따로 남겨 두어야 다음을 검증할 수 있습니다:

- 현재 state
- 허용된 action type
- stale screenshot 여부
- danger zone 여부

## 5. 신뢰성 패턴

### 5.1 Artifact를 first-class로 유지

현재 repo에서 중요하게 남겨야 하는 것:

- source JPEG
- 전송 WebP
- raw response text
- parsed JSON
- overlay image
- log file

### 5.2 동일 screenshot 비교 계약을 유지

서비스를 바꾸더라도 가능하면 같은 screenshot, 같은 target set, 같은 응답 schema를 유지합니다.
그래야 prompt 문제인지 model 문제인지 비교가 됩니다.

### 5.3 Direct와 Proxy를 같은 운용 규칙으로 본다

연결 방식은 달라도 다음은 통일합니다:

- service slug
- debug artifact 형식
- benchmark summary 방식
- failure reporting 방식

## 6. 이 저장소를 위한 실무 가이드

- `poc/work2/flask_vlm.py`를 서비스 registry의 유일한 source of truth로 사용합니다.
- `poc/work2/vlm_client.py`를 공통 호출 계약으로 사용합니다.
- automation 로직을 디버깅하기 전 `poc/work2/connection_check.py`를 먼저 사용합니다.
- 로그인 창 비교에는 `poc/work2/login_rcs.py`와 `poc/work2/login_benchmark.py`를 우선 사용합니다.
- OCR 해석은 `poc/work2/ocr_login_check.py`에서 분리 검증합니다.
- `login_rcs_ui_venus.py`, `login_rcs_ui_tars.py`, `*_rev2.py`는 실험 자산으로 읽고, mainline 문서 기준점으로 삼지는 않습니다.
