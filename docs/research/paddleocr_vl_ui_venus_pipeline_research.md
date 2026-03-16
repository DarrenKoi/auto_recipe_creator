# PaddleOCR-VL-1.5 + UI-Venus 파이프라인 조사 메모 (2026-03-17)

## 목적

이 문서는 이 저장소, 특히 `poc/work2` 기준에서 `PaddleOCR-VL-1.5`와 `UI-Venus`를 어떻게 조합해야 이미지와 스크린샷에서 정보를 더 안정적으로 추출할 수 있는지 정리한 조사 메모다.

핵심 질문은 아래 3개다.

1. `PaddleOCR-VL-1.5`는 무엇에 강한가?
2. `UI-Venus`는 무엇에 강한가?
3. 둘 다 사용할 수 있을 때 가장 실용적인 파이프라인은 무엇인가?

이 문서는 `2026-03-17` 기준으로 작성했으며, 공식 저장소, 공식 모델 카드, 공식 문서, 공식 기술 보고서만 근거로 사용했다.

## 요약

- **정보 추출 중심(extraction-heavy)** 작업에서는 `PaddleOCR-VL-1.5 -> UI-Venus` 순서가 기본값으로 더 적합하다.
- **타깃 찾기 중심(grounding-heavy)** 작업에서는 `UI-Venus -> PaddleOCR-VL-1.5` 순서가 기본값으로 더 적합하다.
- `PaddleOCR-VL-1.5`는 정확한 텍스트, 레이아웃, spotting, 구조화된 시각-텍스트 파싱에 더 강하다.
- `UI-Venus`는 스크린샷 기반 UI 이해, 타깃 grounding, 그리고 어떤 control 또는 영역이 중요한지 고르는 데 더 강하다.
- 핵심 패턴은 "OCR 결과를 전부 그대로 VLM에 던지는 것"이 아니다. 핵심 패턴은 "먼저 OCR anchor를 만들고, 그다음 UI-Venus가 어떤 anchor가 중요한지 고르게 하는 것"이다.
- PaddleOCR 전처리는 항상 켜는 옵션이 아니라 조건부 옵션으로 보는 편이 맞다. 깨끗한 native screenshot보다 카메라 촬영, 원격 화면 사진, 왜곡, 회전 입력에서 훨씬 더 중요하다.
- 좌표가 중요하면 `OCR:` 하나만 쓰는 것보다 `Spotting:` 또는 좌표를 반환하는 PaddleOCR 흐름이 더 나은 경우가 많다.

## PaddleOCR-VL-1.5로 가능한 것

공식 모델 카드 기준 `PaddleOCR-VL-1.5`는 document parsing, text spotting, complex element understanding에 초점을 둔 OCR VLM이다.

공개된 강점은 아래와 같다.

- `109`개 언어 지원
- 텍스트, 표, 수식, 차트, 인장, spotting 지원
- irregular-shaped localization 지원
- skew, warping, screen photography, illumination variation 같은 조건에 대한 강건성 강조
- page-level parsing과 element-level recognition 모두 지원

즉, 이 모델은 단순한 문자 판독기라기보다, 콘텐츠가 어디에 있고 무엇이며 어떻게 구성되어 있는지까지 구조적으로 읽는 파서에 가깝다.

실무적으로 중요한 해석은 아래와 같다.

1. `OCR:`는 넓은 범위의 텍스트 추출에 유용하다.
2. `Spotting:`은 텍스트 위치와 인식이 모두 중요할 때 더 유용하다.
3. `Table Recognition:`은 리포트형, 그리드형, 파라미터 테이블형 화면에서 의미가 크다.
4. 이 모델은 완전히 깨끗한 screenshot보다 더 어려운 캡처 조건까지 전제로 설명되고 있다.

## 더 넓은 PaddleOCR 스택의 장점

여기서는 `PaddleOCR-VL-1.5` 모델 카드만이 아니라 PaddleOCR 공식 문서도 중요하다.

더 넓은 PaddleOCR 스택은 아래 기능을 제공한다.

- 텍스트 내용과 좌표를 함께 반환하는 OCR pipeline
- 최신 PP-OCR 계열의 single-character coordinates 지원
- 선택적 document orientation classification
- 선택적 text image unwarping
- 선택적 text-line orientation classification

즉, 전체 설계 공간은 단순히 "`UI-Venus` + OCR 모델 1회 호출"보다 넓다.

실용적인 분리는 아래처럼 볼 수 있다.

- 빠른 좌표+텍스트 1차 패스: 일반 PaddleOCR 또는 PP-OCR
- 더 풍부한 파싱과 어려운 케이스: `PaddleOCR-VL-1.5`

현재 이 저장소는 proxy 경로를 통해 `paddleocr-vl-1.5`를 노출하고 있다. 나중에 latency가 중요해지면 더 가벼운 local PaddleOCR pass를 추가할 수 있다.

## PaddleOCR-VL-1.5의 한계

태스크 프레이밍을 보면 이 모델의 한계도 비교적 명확하다.

1. 이 모델의 중심은 OCR, parsing, spotting이다.
2. 공식 태스크 예시는 `OCR:`, `Spotting:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:` 같은 task keyword 기반이다.
3. 모델 카드는 더 빠르고 더 완전한 page-level parsing을 위해 공식 PaddleOCR method를 따르라고 분명히 안내한다.

실무적으로 읽으면 아래와 같다.

- `PaddleOCR-VL-1.5`는 instruction 기반 GUI target selection의 primary planner로는 최적이 아니다.
- 이 모델은 텍스트 근거와 레이아웃 근거를 공급하는 extraction engine 또는 OCR sidecar로 쓰는 편이 더 맞다.

이 결론은 공식 task framing과 예시를 바탕으로 한 해석이지, 직접 인용문은 아니다.

## UI-Venus로 가능한 것

공식 `UI-Venus` 자료는 이 모델을 screenshot-driven GUI agent로 소개한다.

공개된 강점은 아래와 같다.

- screenshot 기반 GUI grounding
- screenshot 기반 mobile, desktop, web navigation
- `ScreenSpot-Pro`, `ScreenSpot-v2`, `OSWorld-G`, `AndroidWorld` 같은 벤치마크에서 강한 성능
- visual-only reasoning 강조
- `+ZoomIn` variant 결과를 공개하고 있어 crop 기반 retry 패턴과의 궁합이 좋음

이 저장소 관점에서 해석하면 아래와 같다.

1. 어떤 UI object가 instruction과 맞는지 고르는 데 강하다.
2. 단순 텍스트가 아니라 icon, tab, panel, menu, button, control을 UI 요소로 해석할 수 있다.
3. 이미 `poc/work2`가 채택한 screenshot-centric 흐름과 잘 맞는다.

## UI-Venus의 한계

공식 프레이밍은 grounding과 navigation 중심이지 dense OCR authority 중심이 아니다.

실무적으로는 아래 한계가 나온다.

- 작은 숫자, 긴 코드, dense parameter table, multiline text block, report-like panel처럼 exact text fidelity가 중요한 경우는 보통 `PaddleOCR-VL-1.5` 쪽이 더 적합하다.
- `UI-Venus`는 정확한 문자열의 최종 authority라기보다, 무엇이 중요하고 어디를 봐야 하는지 결정하는 데 더 적합하다.

이 부분은 공식 자료의 benchmark focus와 task framing을 바탕으로 한 해석이다.

## 권장 역할 분리

가장 깔끔한 역할 분리는 아래와 같다.

| 역할 | 더 적합한 모델 | 이유 |
|------|----------------|------|
| 정확한 텍스트 추출 | `PaddleOCR-VL-1.5` | OCR, spotting, parsing 특화 |
| 좌표 포함 텍스트 anchor 생성 | `PaddleOCR-VL-1.5` 또는 일반 PaddleOCR | text polygon, box, spotting, coordinate output |
| 표, 수식, 차트, 구조적 영역 읽기 | `PaddleOCR-VL-1.5` | 공개된 태스크와 직접 정렬됨 |
| instruction 기반 UI target selection | `UI-Venus` | grounding, navigation 특화 |
| icon, tab, button, menu, panel 해석 | `UI-Venus` | screenshot-only GUI reasoning |
| ambiguous crop 해석 | `UI-Venus` + OCR retry | 의미 해석과 텍스트 근거 결합 |

## 정보 추출 중심 작업에서 가장 좋은 파이프라인

목표가 이미지에서 정보를 잘 추출하는 것이라면, 아래 흐름이 기본형으로 가장 적합하다.

### 권장 흐름

1. **입력 triage**
   이미지가 clean screenshot인지, photo 또는 distorted capture인지 먼저 구분한다.
2. **조건부 PaddleOCR 전처리**
   입력 품질이 필요로 할 때만 orientation correction, unwarping, text-line orientation을 켠다.
3. **OCR 1차 패스**
   빠른 좌표가 충분하면 일반 PaddleOCR를 사용한다.
   콘텐츠가 dense하고 구조적이거나 다국어이거나 시각적으로 어려우면 `PaddleOCR-VL-1.5`를 사용한다.
4. **OCR 정규화**
   raw OCR output을 `text`, `bbox` 또는 `polygon`, `score`, `reading_order`, `block_type` 같은 필드를 가진 compact JSON으로 바꾼다.
5. **UI-Venus semantic pass**
   screenshot과 compact OCR anchor를 함께 주고, 어떤 영역 또는 값이 relevant한지 식별하게 한다.
6. **crop 기반 escalation**
   low-confidence 영역이나 dense 영역은 zoomed crop으로 두 모델을 다시 호출한다.
7. **merge**
   exact string은 OCR 결과를 우선한다.
   semantic selection과 field-role interpretation은 `UI-Venus`를 우선한다.
8. **verify**
   둘이 충돌하면 unresolved 상태로 두거나 second pass를 수행한다.

### 왜 이 순서가 좋은가

- 텍스트 추출 실패는 대개 "텍스트를 제대로 못 읽었다"는 문제인데, 이 부분은 OCR 모델이 더 잘 다룬다.
- UI 추출 실패는 대개 "텍스트는 읽었지만 잘못된 field를 골랐다"는 문제인데, 이 부분은 grounding 모델이 더 잘 다룬다.
- 따라서 정보 추출 중심 작업에서는 OCR이 evidence를 만들고, 그다음 `UI-Venus`가 의미를 정리하는 흐름이 안정적이다.

## 타깃 찾기 중심 작업에서 가장 좋은 파이프라인

목표가 버튼, 탭, 필드, click target을 찾는 것이라면 순서를 반대로 두는 편이 맞다.

### 권장 흐름

1. `UI-Venus` full-screen pass
2. low-confidence 또는 text-dependent case에만 OCR 호출
3. 가능성이 높은 label 또는 target cluster 주변으로 crop 생성
4. OCR text와 `UI-Venus` grounding을 합쳐 최종 target 결정
5. action 후 다음 screenshot을 다시 확인

### 대표 사례

- "Save 버튼 클릭"
- "recipe name input 찾기"
- "View 또는 List tab 선택"
- "작은 text label 옆 input box 찾기"

이런 문제는 extraction보다 grounding 비중이 크므로 `UI-Venus`가 먼저 가는 편이 맞다.

## `OCR:`와 `Spotting:`의 선택

현재 `poc/work2/prompts/ocr_assist.py`는 `OCR:`만 사용한다.

이건 무난한 기본값이지만 항상 최선은 아니다.

권장 분기는 아래와 같다.

- 넓은 범위의 텍스트를 읽고 싶다: `OCR:`
- 텍스트와 위치를 같이 읽고 싶다: `Spotting:`
- 표 구조를 읽고 싶다: `Table Recognition:`
- 차트 내용을 읽고 싶다: `Chart Recognition:`

즉, 전체 아키텍처를 바꾸지 않더라도 task-keyword branching을 추가하면 OCR 단계의 품질을 높일 수 있다.

## 전처리를 언제 켜야 하는가

공식 PaddleOCR 전처리 문서는 아래를 다룬다.

- orientation classification
- geometric distortion correction 또는 unwarping

일반 OCR 흐름에서는 text-line orientation classification도 지원한다.

실무 규칙은 아래 정도가 적당하다.

- native screenshot: 전처리 보통 `off`
- monitor photo, phone capture, remote-screen photo, warped image: 전처리 보통 `on`
- 회전 이슈가 잦다: orientation `on`
- perspective 또는 bending distortion이 있다: unwarping `on`

전처리는 항상 켜는 기본값이 아니라 조건부 단계로 두는 편이 맞다.

## 이 저장소에 맞는 운영 모드

### Option A: 현재 구조를 크게 안 흔드는 방식

1. 기본 GUI analysis model은 `UI-Venus`
2. 텍스트와 구조 sidecar는 `PaddleOCR-VL-1.5`
3. extraction task에서는 OCR 결과를 먼저 구조화
4. raw OCR dump 대신 compact OCR hint를 `UI-Venus`에 전달
5. low-confidence 영역에만 crop retry 수행

이 방식은 현재 `poc/work2/flask_vlm.py` 구조와 잘 맞는다.

### Option B: extraction latency와 precision을 더 올리는 방식

1. 빠른 일반 PaddleOCR pass 추가
2. dense, complex, multilingual, ambiguous region에만 `PaddleOCR-VL-1.5`로 escalation
3. 최종 semantic resolver는 `UI-Venus`

이 방식은 extraction 품질이 핵심이 될 때 장기적으로 가장 좋아 보인다. 다만 더 가벼운 PaddleOCR 경로는 아직 이 저장소에 없다.

## 하지 말아야 할 것

1. `PaddleOCR-VL-1.5`를 primary click-grounding model로 강제하지 않는다.
2. `UI-Venus`를 exact extracted text의 authority로 강제하지 않는다.
3. 전체 raw OCR text를 구조 없이 prompt에 그대로 넣지 않는다.
4. 어려운 케이스에서 single full-screen pass만 믿지 말고 crop-based retry를 사용한다.
5. clean screenshot에 document preprocessing을 기본값으로 전부 켜지 않는다.

## 최종 권고

현재 저장소 기준 가장 강한 실무 권고는 아래와 같다.

- 작업이 **정보 추출**이면 `PaddleOCR-VL-1.5`로 시작한 뒤 `UI-Venus`를 사용한다.
- 작업이 **UI target 찾기 또는 action grounding**이면 `UI-Venus`로 시작한 뒤 `PaddleOCR-VL-1.5`를 사용한다.
- `PaddleOCR-VL-1.5`는 text와 structure evidence engine으로 둔다.
- `UI-Venus`는 UI meaning과 grounding engine으로 둔다.
- 둘 사이 연결 포맷은 raw plain text가 아니라 compact OCR JSON hint로 둔다.
- small text, dense panel, low-confidence region에는 crop-based escalation을 사용한다.

가장 좋은 패턴은 "하나를 메인 모델로 두고 다른 하나를 대충 붙이는 것"이 아니다. 작업 종류에 따라 호출 순서가 바뀌는 dual pipeline이 가장 좋다.

## 바로 이어서 할 작업

1. `poc/work2/prompts/ocr_assist.py`에서 `OCR:` 하드코딩 대신 task-keyword branching 추가
2. `text`, `coords`, `score`, `block_type`를 반환하는 OCR normalization helper 추가
3. `UI-Venus` prompt에 넣을 compact OCR-hint format 표준화
4. low-confidence crop-retry 규칙 추가
5. screenshot-vs-photo 입력 유형에 따른 conditional preprocessing 추가

## Sources

- PaddleOCR GitHub: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR OCR pipeline docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>
- PaddleOCR document preprocessing docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/doc_preprocessor.html>
- PaddleOCR text line orientation docs: <https://www.paddleocr.ai/main/en/version3.x/module_usage/textline_orientation_classification.html>
- PaddleOCR-VL-1.5 model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>
- PaddleOCR-VL-1.5 technical report: <https://huggingface.co/papers/2601.21957>
- vLLM PaddleOCR-VL recipe: <https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- UI-Venus GitHub: <https://github.com/inclusionAI/UI-Venus>
- UI-Venus-1.5-8B model card: <https://huggingface.co/inclusionAI/UI-Venus-1.5-8B>
- UI-Venus-1.5 technical report: <https://huggingface.co/papers/2602.09082>
