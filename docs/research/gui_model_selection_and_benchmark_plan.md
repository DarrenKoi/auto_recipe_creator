# GUI 모델 선택 및 벤치마크 계획 (2026-03-17)

## 목적

이 문서는 현재 저장소 기준으로 아래 질문에 빠르게 답하기 위한 canonical 운영 문서다.

1. GUI 모델 비교는 어떤 순서로 시작해야 하는가
2. `MAI-UI`, `PaddleOCR-VL`, `GOT-OCR`는 언제 sidecar로 붙여야 하는가
3. 어떤 지표로 실험 결과를 해석해야 하는가

모델별 강점과 한계를 길게 분석한 deep dive는 [`deploy_vlms_model_roles_and_pipeline_research.md`](./deploy_vlms_model_roles_and_pipeline_research.md) 를 본다.

## 현재 기준 빠른 결론

- 외부 baseline 이 필요하면 `Kimi-K2.5`를 둔다.
- self-hosted GUI 주력 비교축은 `UI-Venus` vs `UI-TARS`다.
- small target / crowded crop 재탐색은 `MAI-UI`가 맡는다.
- 기본 OCR sidecar는 `PaddleOCR-VL-1.5`다.
- formatting 민감한 hard OCR fallback은 `GOT-OCR-2.0-hf`다.
- 5개 모델을 매 step 항상 호출하는 구조는 피한다.

## 로컬 설정 스냅샷

현재 저장소 안의 설정 파일 기준으로는 아래 5개 서비스가 모두 등록되어 있다.

| 역할 | 서비스 | 비고 |
|------|--------|------|
| 외부 baseline | `Kimi-K2.5` | 회사 API 경로를 통한 기준선 |
| GUI primary A | `ui-venus` | 기본 full-screen grounding 후보 |
| GUI primary B | `ui-tars` | 비교용 full-screen grounding 후보 |
| zoom-in sidecar | `mai-ui` | 작은 crop 재탐색 |
| OCR default | `paddleocr-vl-1.5` | dense text / table / panel OCR |
| OCR fallback | `got-ocr` | hard crop / formatting 민감 OCR |

중요한 점은 "설정에 등록되어 있다"와 "실제 런타임이 살아 있다"는 다르다는 점이다. 실제 상태는 `uv run python poc/work2/connection_check.py` 로 확인하는 편이 맞다.

## 권장 비교 순서

### 1단계: primary GUI 모델 비교

가장 먼저 비교할 축은 아래다.

1. `Kimi-K2.5` full-screen pass
2. `UI-Venus` full-screen pass
3. `UI-TARS` full-screen pass

이 단계에서는 sidecar를 붙이지 않는다. 그래야 coarse grounding 자체의 차이를 분리해서 볼 수 있다.

### 2단계: zoom-in sidecar 추가

1단계에서 고른 self-hosted 승자 모델에 `MAI-UI`만 추가한다.

이 단계는 아래 failure type이 실제로 줄어드는지 보는 단계다.

- 작은 버튼
- 좁은 탭
- 아이콘이 과밀한 툴바
- label 옆의 작은 input

### 3단계: OCR sidecar 추가

2단계 승자 구성에 `PaddleOCR-VL-1.5`를 붙인다.

이 단계는 아래와 같은 text-heavy 케이스를 본다.

- dense parameter panel
- 작은 숫자
- list / table / grid
- exact string verification

### 4단계: hard OCR fallback 추가

필요한 failure case에만 `GOT-OCR-2.0-hf`를 붙인다.

항상-on으로 두기보다 아래 같은 경우만 fallback으로 쓴다.

- formatting 보존이 중요함
- 작은 숫자 또는 코드가 흐림
- `PaddleOCR-VL-1.5` 결과가 애매함

## 측정 항목

실험 표는 아래 항목으로 통일하는 편이 좋다.

- `element hit rate`
- `click-point drift(px)`
- `retry count`
- `step completion rate`
- `small-text OCR recall`
- `latency`
- `sidecar escalation rate`

가능하면 같은 스크린샷 세트와 같은 task set으로 비교한다.

## 운영 규칙

### primary와 sidecar 역할을 섞지 않는다

- `UI-Venus`와 `UI-TARS`는 full-screen primary 슬롯 경쟁자다.
- `MAI-UI`는 primary 비교 대상이라기보다 crop retry 전용 sidecar에 가깝다.
- `PaddleOCR-VL-1.5`와 `GOT-OCR-2.0-hf`는 exact text authority에 가깝다.

### escalation은 조건부로만 건다

아래 정도의 규칙이면 충분하다.

1. 기본은 GUI primary 1회 호출
2. small target 또는 crowded crop이면 `MAI-UI`
3. text-heavy panel이면 `PaddleOCR-VL-1.5`
4. OCR 결과가 부족하면 `GOT-OCR`

### 런타임 검증을 문서보다 우선한다

이 폴더 안의 오래된 메모에는 당시 시점의 활성 상태가 섞여 있다. 현재 판단은 prose가 아니라 아래를 기준으로 맞춘다.

- `flask_api/vlm_serve/config.py`
- `poc/work2/flask_vlm.py`
- `poc/work2/connection_check.py`

## 기본 운영 스택

현재 저장소에 가장 자연스러운 기본형은 아래다.

1. `UI-Venus` 또는 `UI-TARS`로 full-screen grounding
2. 필요 시 `MAI-UI`로 crop 재탐색
3. exact text가 필요하면 `PaddleOCR-VL-1.5`
4. hard OCR만 `GOT-OCR-2.0-hf`

외부 baseline이 필요할 때만 `Kimi-K2.5`를 같은 평가 세트에 추가한다.

## 이 문서 다음에 볼 것

- 모델별 근거와 역할 분리: [`deploy_vlms_model_roles_and_pipeline_research.md`](./deploy_vlms_model_roles_and_pipeline_research.md)
- `UI-Venus`와 OCR 조합 상세: [`paddleocr_vl_ui_venus_pipeline_research.md`](./paddleocr_vl_ui_venus_pipeline_research.md)
- OmniParser 대체안 검토: [`omniparser_v2_integration_research.md`](./omniparser_v2_integration_research.md)
