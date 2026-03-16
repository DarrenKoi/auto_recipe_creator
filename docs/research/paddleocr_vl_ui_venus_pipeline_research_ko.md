# PaddleOCR-VL-1.5 + UI-Venus 파이프라인 조사 메모 (2026-03-17)

## 목적

이 문서는 현재 저장소의 `poc/work2` 방향을 기준으로, `PaddleOCR-VL-1.5`와 `UI-Venus`를 어떻게 조합해야 이미지에서 정보를 가장 잘 뽑을 수 있는지 정리한 조사 메모다.

이번 메모의 질문은 3개다.

1. `PaddleOCR-VL-1.5`로 무엇이 가능한가?
2. `UI-Venus`로 무엇이 가능한가?
3. "이미지 정보 추출" 기준 최적 파이프라인은 무엇인가?

조사 기준일은 `2026-03-17`이며, 공식 저장소, 공식 모델 카드, 공식 문서, 공식 기술 보고서만 근거로 사용했다.

## 결론 요약

- **text-heavy / extraction-heavy 이미지**에서는 `PaddleOCR-VL-1.5 -> UI-Venus` 순서가 가장 합리적이다.
- **grounding-heavy / action-heavy 화면**에서는 `UI-Venus -> PaddleOCR-VL-1.5` 순서가 더 낫다.
- `PaddleOCR-VL-1.5`는 **정확한 텍스트/레이아웃/spotting** 쪽이 강하고, `UI-Venus`는 **스크린샷 기반 UI 의미 해석과 타깃 grounding** 쪽이 강하다.
- 둘을 같이 쓸 때 핵심은 "`OCR 결과 전체를 그대로 VLM에 던지기`"가 아니라, **OCR로 region/text anchor를 만들고 UI-Venus가 그중 어떤 정보가 중요한지 고르게 하는 것**이다.
- clean screenshot 이면 PaddleOCR의 문서 전처리(`orientation`, `unwarping`)를 항상 켤 필요는 없다. 반대로 **카메라 촬영, 원격 화면 사진, 왜곡된 캡처**에는 전처리가 꽤 중요하다.
- 좌표가 필요한 extraction 에서는 `PaddleOCR-VL-1.5`의 `OCR:`만 고정으로 쓰기보다, **`Spotting:` 또는 일반 PaddleOCR 좌표 출력**을 같이 고려하는 편이 낫다.

## 1. PaddleOCR-VL-1.5로 가능한 것

공식 모델 카드 기준 `PaddleOCR-VL-1.5`는 `2026-01-29` 공개된 `0.9B` OCR VLM이고, 핵심 포지션은 **document parsing + text spotting + 복잡한 요소 인식**이다.

공식 기준 강점:

- `109`개 언어 지원
- `text`, `table`, `formula`, `chart`, `seal`, `spotting` 지원
- `irregular-shaped localization` 지원
- `skew`, `warping`, `screen-photography`, `illumination` 같은 실제 왜곡 조건에 대한 강건성 강조
- `page-level document parsing`과 `element-level recognition` 모두 지원

즉, 이 모델은 단순히 "문자를 읽는다" 수준이 아니라, **어디에 어떤 text block / table / formula / chart가 있는지 구조적으로 읽는 모델**에 가깝다.

실무적으로 특히 중요한 포인트는 아래다.

1. `OCR:`는 전체 텍스트 추출용으로 쓸 수 있다.
2. `Spotting:`은 **text-line localization + recognition** 성격이므로, 나중에 `UI-Venus`에 넘길 bbox anchor를 만들 때 더 적합하다.
3. `Table Recognition:`은 report/grid/table 성격 화면에서 별도 가치가 있다.
4. `screen-photography` 강건성을 공식적으로 강조하므로, 깨끗한 native screenshot 뿐 아니라 원격 촬영/왜곡 이미지에도 의미가 있다.

## 2. PaddleOCR 계열의 추가 장점

공식 PaddleOCR 문서 기준, 꼭 `PaddleOCR-VL-1.5`만 쓸 필요는 없다.

- PaddleOCR는 일반 OCR pipeline 에서 `text position coordinates`와 `content`를 바로 반환한다.
- 최신 문서 기준 PP-OCR 계열은 `single-character coordinates`까지 지원한다.
- 기본 OCR pipeline 은 `document orientation`, `text image unwarping`, `text line orientation`를 조합할 수 있다.

즉, 파이프라인을 아래처럼 나누는 선택지도 있다.

- 빠른 좌표/텍스트 초벌: `PP-OCRv5` 또는 일반 PaddleOCR
- 복잡한 구조/spotting/표/왜곡 대응: `PaddleOCR-VL-1.5`

이건 현재 저장소의 `paddleocr-vl-1.5` proxy 만으로 운영할 수도 있고, 나중에 local PaddleOCR python path 를 추가해서 더 가볍게 분기할 수도 있다는 뜻이다.

## 3. PaddleOCR-VL-1.5의 한계

공식 문서 범위와 태스크 정의를 기준으로 보면 한계도 분명하다.

1. 이 모델의 강점은 **OCR / parsing / spotting**이지, GUI action grounding 자체는 아니다.
2. 공식 Hugging Face 예시는 `OCR:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:`, `Spotting:` 같은 task keyword 중심이다.
3. 공식 모델 카드도 `transformers` 예시보다 **official method 가 더 빠르고 page-level parsing 을 지원**한다고 명시한다.

여기서 나오는 실무 해석은 아래다.

- `PaddleOCR-VL-1.5`는 "이 instruction 에 해당하는 클릭 타깃이 어디인가"를 정하는 primary planner 로 쓰기보다,
- **텍스트 근거와 구조적 읽기 결과를 공급하는 sidecar/primary extractor**로 쓰는 편이 맞다.

이 해석은 공식 문서의 태스크 범위와 모델 카드 설명을 근거로 한 추론이다.

## 4. UI-Venus로 가능한 것

공식 GitHub/모델 카드 기준 `UI-Venus`는 **screenshot only** 기반 UI agent다.

공식 기준 강점:

- screenshot 만으로 GUI element grounding 가능
- screenshot 만으로 mobile / desktop / web navigation 가능
- `ScreenSpot-Pro`, `ScreenSpot-v2`, `OSWorld-G`, `AndroidWorld` 등 GUI benchmark 에서 강한 성능
- `visual-only reasoning`을 강조

특히 `UI-Venus-1.5-8B` 모델 카드는 `2026-02-09` 공개된 1.5 technical report 기준으로, grounding 과 navigation 을 모두 포함하는 unified GUI agent 로 설명한다.

실무적으로 번역하면 아래와 같다.

1. 화면 전체를 보고 "이 instruction 에 맞는 UI 요소가 무엇인가"를 고르는 능력이 강하다.
2. text 만이 아니라 icon, panel, 탭, 버튼, 메뉴 같은 **UI 의미 단위**를 해석하는 능력이 강하다.
3. accessibility tree 없이도 screenshot 만으로 작업하는 구조라서, 현재 `poc/work2` 같은 screenshot-centric 흐름과 잘 맞는다.

## 5. UI-Venus의 한계

공식 자료는 UI-Venus를 **grounding / navigation 모델**로 소개하지, dense OCR parser 로 소개하지는 않는다.

따라서 아래 해석이 타당하다.

- 작은 숫자, 긴 코드, dense parameter table, multiline text block, 보고서형 panel 의 **정확한 문자 추출**은 `PaddleOCR-VL-1.5` 쪽이 더 신뢰할 만하다.
- `UI-Venus`는 "무엇이 중요한가 / 어디를 봐야 하는가 / 어떤 control 이 instruction 에 맞는가"를 정하는 데 더 적합하다.

또 하나 중요한 점은 공식 UI-Venus repo 가 grounding benchmark 와 navigation benchmark 를 강하게 내세운다는 것이다. 이건 장점이면서도 한계다.

- 장점: UI 의미 해석은 강하다.
- 한계: **문자 자체를 authority 로 삼는 extraction engine**으로 쓰기에는 설계 초점이 다르다.

즉, `UI-Venus` 단독으로도 어느 정도 text-heavy 화면을 읽을 수는 있겠지만, **정확한 text extraction source-of-truth**로 두는 것은 비효율적일 가능성이 높다.

이 부분 역시 공식 benchmark 범위와 모델 설명을 바탕으로 한 추론이다.

## 6. 각 모델의 역할 분리

가장 실용적인 역할 분리는 아래다.

| 역할 | 더 적합한 모델 | 이유 |
|------|----------------|------|
| 정확한 text 읽기 | `PaddleOCR-VL-1.5` | OCR / spotting / document parsing 특화 |
| 좌표 있는 text anchor 생성 | `PaddleOCR-VL-1.5` 또는 일반 PaddleOCR | text polygon / bbox / spotting 가능 |
| 표/차트/수식/체크박스 같은 구조 읽기 | `PaddleOCR-VL-1.5` | 공식 태스크 범위에 직접 포함 |
| instruction 기반 UI 타깃 선택 | `UI-Venus` | grounding / navigation 특화 |
| icon / button / tab / menu 의미 해석 | `UI-Venus` | screenshot-only GUI reasoning 특화 |
| ambiguous crop 재해석 | `UI-Venus` + OCR 재호출 | 의미 해석과 텍스트 근거를 함께 재확인 가능 |

## 7. 최적 파이프라인: extraction-heavy 기준

사용자 질문이 "이미지에서 정보를 잘 추출하고 싶다" 쪽에 가깝다면, 추천 기본형은 아래다.

### 7.1 추천 기본형

1. **입력 triage**
   이미지가 `clean screenshot`인지, `camera/photo/remote-capture`인지 먼저 나눈다.
2. **필요 시 PaddleOCR 전처리**
   `orientation`, `unwarping`, `textline orientation`은 왜곡 이미지에만 켠다.
3. **OCR first pass**
   - 빠른 좌표+텍스트가 필요하면 일반 PaddleOCR
   - 현재 proxy 체계만 쓸 경우 `PaddleOCR-VL-1.5`
4. **OCR 결과 정규화**
   raw text dump 대신 `text`, `bbox/polygon`, `confidence`, `reading_order`, `block_type` 중심 JSON으로 압축한다.
5. **UI-Venus semantic pass**
   전체 스크린샷과 함께 "무슨 정보를 뽑아야 하는지"를 주고, OCR anchor 중 어떤 것이 relevant 한지 고르게 한다.
6. **zoom-in / crop escalation**
   low-confidence 영역이나 dense panel 은 crop 해서 `PaddleOCR-VL-1.5`와 `UI-Venus`를 다시 돌린다.
7. **merge**
   exact string 은 OCR 결과를 우선하고, field 선택/의미 연결은 UI-Venus를 우선한다.
8. **verify**
   서로 충돌하면 unresolved 로 남기거나 second-pass 를 수행한다.

### 7.2 왜 이 순서가 좋은가

- text extraction 의 실패는 보통 "못 읽음" 문제이고, 이건 OCR 쪽이 더 강하다.
- UI extraction 의 실패는 보통 "읽긴 했지만 어떤 값이 중요한지 모름" 문제이고, 이건 UI-Venus 쪽이 더 강하다.
- 따라서 extraction-heavy 업무는 **OCR 로 증거를 만들고 UI-Venus 가 의미를 정리하는 흐름**이 가장 안정적이다.

## 8. 최적 파이프라인: grounding-heavy 기준

반대로 목표가 "버튼/탭/입력칸을 찾고 click target 을 정한다"면 순서를 뒤집는 편이 낫다.

### 8.1 추천 기본형

1. `UI-Venus` full-screen pass
2. low-confidence 또는 text-dependent case 만 OCR 호출
3. label 근처 crop 생성
4. OCR text 와 UI-Venus grounding 결과를 합쳐 좌표 확정
5. action 이후 verify screenshot 재확인

### 8.2 언제 이 모드가 맞는가

- "Save 버튼 눌러라"
- "Recipe name 입력칸 찾기"
- "Main tab 중 View/List 를 선택"
- "작은 text label 옆의 input box 찾기"

이 경우는 extraction 보다 **grounding**이 핵심이므로 `UI-Venus`가 선행하는 게 맞다.

## 9. `OCR:`만 쓸지 `Spotting:`도 쓸지

현재 `poc/work2/prompts/ocr_assist.py`는 `PaddleOCR-VL-1.5`에 대해 `OCR:`만 보내도록 되어 있다.

이 기본값은 나쁘지 않지만, 항상 최선은 아니다.

권장 분기:

- 전체 텍스트를 읽고 싶다: `OCR:`
- 텍스트와 위치를 같이 잡고 싶다: `Spotting:`
- 표 형태 결과가 중요하다: `Table Recognition:`
- plot/chart 내 값을 읽고 싶다: `Chart Recognition:`

즉, 현재 구조를 유지하더라도 **task keyword 를 상황별로 바꿀 수 있게 만드는 것**이 파이프라인 품질에 직접 도움이 된다.

## 10. 전처리를 언제 켜야 하는가

PaddleOCR 공식 문서 기준 `Document Image Preprocessing Pipeline`은 아래 2개를 제공한다.

- `orientation classification`
- `geometric distortion correction (unwarping)`

또 일반 OCR 쪽에는 `text line orientation classification`도 있다.

실무 규칙은 아래 정도가 적당하다.

- native screenshot: 전처리 기본 `off`
- 모니터 사진 / 휴대폰 촬영 / 왜곡된 문서: 전처리 기본 `on`
- 세로/거꾸로 캡처가 자주 섞임: orientation `on`
- remote-desktop 사진처럼 휘거나 찍힘: unwarping `on`

즉, 전처리는 "항상 켜는 옵션"이 아니라 **입력 품질에 따라 조건부로 켜는 단계**로 보는 편이 맞다.

## 11. 실제 운영 추천안

### 추천안 A: 현재 저장소를 크게 안 흔드는 방향

1. 기본 GUI 분석: `UI-Venus`
2. 텍스트 보강: `PaddleOCR-VL-1.5`
3. extraction task 에서는 OCR 결과를 먼저 구조화
4. UI-Venus 에는 raw OCR full dump 대신 compact hint 만 전달
5. low-confidence 영역만 crop 재호출

이 방식은 현재 `poc/work2/flask_vlm.py` 구조와 가장 잘 맞는다.

### 추천안 B: extraction 품질을 더 올리는 방향

1. 일반 PaddleOCR fast pass 추가
2. `PaddleOCR-VL-1.5`는 dense/complex/ambiguous 영역에만 escalation
3. `UI-Venus`는 최종 semantic resolver 로 사용

이 방식은 latency 와 precision 을 같이 잡기 좋다. 다만 현재 repo 에는 일반 PaddleOCR local pipeline 이 아직 없다.

## 12. 하지 말아야 할 것

1. `PaddleOCR-VL-1.5`를 click grounding primary 로 강제 사용하지 않는다.
2. `UI-Venus`를 exact text authority 로 강제 사용하지 않는다.
3. 전체 OCR dump 를 그대로 user prompt 에 길게 붙이지 않는다.
4. full-screen 단일 pass 만 믿지 말고, **crop/zoom 재시도**를 기본 전략으로 둔다.
5. clean screenshot 에도 무조건 doc preprocessing 을 다 켜지 않는다.

## 13. 저장소 기준 최종 권고

현재 저장소 맥락에서 가장 현실적인 최종 권고는 아래다.

- **정보 추출이 목적이면** `PaddleOCR-VL-1.5 -> UI-Venus`
- **조작 대상 찾기가 목적이면** `UI-Venus -> PaddleOCR-VL-1.5`
- `PaddleOCR-VL-1.5`는 text/structure evidence engine
- `UI-Venus`는 UI meaning / grounding engine
- 둘 사이 연결 포맷은 raw plain text 가 아니라 `compact OCR JSON hints`
- small text / crowded panel / low-confidence 는 crop-based escalation

즉, "둘 중 하나를 메인으로 고르고 다른 하나를 덤으로 붙인다"보다, **task 에 따라 선후를 바꾸는 이중 파이프라인**이 가장 좋다.

## 14. 바로 적용 가능한 작업 항목

1. `poc/work2/prompts/ocr_assist.py`에 `OCR:` 고정 대신 task keyword 분기 지점 추가
2. OCR 결과를 `text`, `coords`, `score`, `block_type` 형태로 정규화하는 helper 추가
3. `UI-Venus` prompt 에 OCR hint 를 compact JSON 으로 넣는 포맷 고정
4. low-confidence crop retry 규칙 추가
5. screenshot / photo 입력 구분에 따라 전처리 on/off 분기 추가

## 참고 source

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
