# `deploy_vlms` 배치 모델 역할/강점/한계/파이프라인 조사 메모 (2026-03-17)

## 목적

이 문서는 `deploy_vlms/config/models/*.env` 에 등록된 현재 배치 모델 5종을 기준으로, `poc/work2` 파이프라인에서 각 모델을 어디에 쓰는 것이 가장 실용적인지 정리한 운영 메모다.

핵심 질문은 아래 3개다.

1. 각 모델의 강점은 무엇인가?
2. 각 모델의 한계는 무엇인가?
3. `poc/work2` 기준으로 어떻게 조합해야 가장 안정적인 파이프라인이 되는가?

이 문서는 `2026-03-17` 기준으로 작성했으며, **공식 저장소, 공식 모델 카드, 공식 기술 보고서**, 그리고 이 저장소의 배치 설정 파일만 근거로 사용했다. 공식 자료에 명시되지 않은 운영 해석은 문장 안에서 별도로 표시했다.

## 조사 범위

현재 `deploy_vlms` 와 `flask_api/vlm_serve/config.py` 기준 활성 모델은 아래 5개다.

| service slug | served model | port | GPU | 모델 성격 | 1차 권장 역할 |
|--------------|--------------|------|-----|-----------|----------------|
| `ui-venus` | `ui-venus-1.5-8b` | `8001` | `0` | GUI grounding VLM | full-screen 주력 |
| `ui-tars` | `ui-tars-1.5-7b` | `8003` | `0` | GUI agent VLM | full-screen 대안 / 실험축 |
| `mai-ui` | `mai-ui-8b` | `8002` | `1` | GUI grounding + action parser | zoom-in sidecar |
| `paddleocr-vl-1.5` | `paddleocr-vl-1.5` | `8004` | `1` | OCR/document parsing VLM | 기본 OCR sidecar |
| `got-ocr` | `got-ocr-2.0-hf` | `8005` | `1` | OCR specialist | hard OCR fallback |

로컬 배치 해석도 중요하다.

- GPU 0은 `UI-Venus` 와 `UI-TARS` co-location 구조다.
- GPU 1은 `MAI-UI + PaddleOCR-VL + GOT-OCR` co-location 구조다.
- `deploy_vlms/config/common.env` 는 `STRICT_OFFLINE=1` 기준이다.

즉, 현재 구조는 "GUI 주력 1개 + 조건부 sidecar"로 설계하는 편이 맞고, 5개를 매 step 항상 호출하는 구조는 맞지 않는다.

## 한 줄 결론

- **full-screen GUI grounding 기본값**은 `UI-Venus`
- **agent-style 대안 주력**은 `UI-TARS`
- **small crop 재탐색 sidecar**는 `MAI-UI`
- **기본 OCR/구조 추출 sidecar**는 `PaddleOCR-VL-1.5`
- **formatting 민감 / hard crop OCR fallback**은 `GOT-OCR-2.0-hf`

가장 실용적인 운영 원칙은 아래 한 줄로 요약된다.

> full-screen 판단은 GUI 모델이, exact text authority 는 OCR 모델이, small ambiguous crop 재판독은 `MAI-UI` 가 맡는다.

## 모델별 분석

### 1. `UI-Venus-1.5-8B`

### 공식 자료 기준 강점

- screenshot-driven GUI agent 로 소개된다.
- mobile, web, desktop GUI grounding 과 navigation 벤치마크를 전면에 둔다.
- `ScreenSpot-Pro`, `ScreenSpot-v2`, `AndroidWorld`, `OSWorld-G` 등 GUI benchmark 에서 강한 성능을 공개한다.
- 공식 자료가 `visual-only reasoning` 을 전면에 두고 있다.
- 공식 GitHub 는 `+ZoomIn` 결과도 함께 공개하고 있어, crop 기반 retry 패턴과의 궁합이 좋다.

### `poc/work2` 에서 가장 좋은 용도

1. full-screen 1차 state recognition
2. tab, button, panel, icon, menu, input 위치 후보 찾기
3. OCR 결과를 참고 정보로만 받는 semantic resolver
4. click 전 마지막 좌표 후보 결정

### 한계

아래 한계는 공식 benchmark focus 와 공개된 task framing 을 바탕으로 한 운영 해석이다.

- dense text panel, 작은 숫자, 긴 코드, parameter table 처럼 **exact text fidelity** 가 중요한 경우에는 OCR 모델보다 불리하다.
- 작은 타깃이나 촘촘한 toolbar 에서는 full-screen 1회 호출만으로는 흔들릴 수 있다. 공식 자료가 `ZoomIn` 변형을 따로 공개하는 점도 이 해석과 맞는다.
- 최종 문자열 authority 로 두기보다, 무엇이 중요하고 어디를 클릭해야 하는지 고르는 데 두는 편이 더 적합하다.

### 권장 배치 역할

- 기본 `screen_analysis`
- 기본 `main_tabs`
- full-screen primary grounding

### 2. `UI-TARS-1.5-7B`

### 공식 자료 기준 강점

- `UI-TARS-1.5` 는 GUI interaction 용 end-to-end VLM agent 로 소개된다.
- 공식 저장소는 `GROUNDING`, `COMPUTER_USE`, `BROWSER_USE`, `GAME_USE` 같은 task template 를 명확히 제공한다.
- 단일 프레임 grounding 뿐 아니라 multi-step action generation 과 action-space 설계가 비교적 분명하다.
- `2.0` 대비 향상된 성능을 강조하며, browser/computer/game 계열 태스크 전반을 다룬다.

### `poc/work2` 에서 가장 좋은 용도

1. `UI-Venus` 와의 primary head-to-head 비교 대상
2. multi-step action planner 실험
3. "관찰 -> 다음 action 제안" 흐름이 필요한 자동화 agent 실험
4. 모델 출력 형식을 action DSL 로 통일하고 싶을 때의 대안

### 한계

아래 2개는 공식 저장소의 prompt/post-processing 설명과 benchmark framing 을 바탕으로 한 운영 해석이다.

- 공식 저장소가 absolute coordinates 후처리를 별도로 설명할 정도로, 좌표 post-processing 규칙이 중요하다.
- OCR specialist 가 아니므로 dense text authority 로 쓰는 것은 맞지 않다.
- `poc/work2` 처럼 우선 single-shot screenshot grounding 이 핵심인 흐름에서는, 단순 좌표 질의까지 agent-style prompt 로 몰아가면 오히려 출력 일관성이 떨어질 수 있다.
- browser/game/computer-use 범위가 넓은 대신, 산업용 niche Windows UI 에 최적화되었다고 보기는 어렵다. 이 부분은 공식 공개 범위를 바탕으로 한 해석이다.

### 권장 배치 역할

- `UI-Venus` 대안 primary
- action-sequence 실험 모델
- A/B benchmark 축

### 3. `MAI-UI-8B`

### 공식 자료 기준 강점

- `vision-centric action parser` 를 핵심 아이디어로 제시한다.
- `ScreenSpot-Pro`, `ScreenSpot-v2`, `MM-Mind2Web`, `AndroidControl`, `AndroidWorld` 등에서 강한 성능을 강조한다.
- 공식 GitHub 는 `GroundingAgent` 와 `NavigationAgent` 를 분리해 제공한다.
- `AskUser` 와 `MCP` 를 포함한 `Device-Cloud Collaboration` 방향까지 제시한다.
- 공식 GitHub 표에서는 `zoom-in trick` 없이도 강한 grounding 성능을 보여준다.

### `poc/work2` 에서 가장 좋은 용도

1. primary 모델이 좁힌 영역에 대한 zoom-in 재탐색
2. 작은 버튼, 작은 탭, 가까이 붙은 icon cluster 재판독
3. label 옆 input box, dense parameter panel 같은 local grounding
4. full-screen 모델의 low-confidence 결과에 대한 second opinion

### 한계

아래 항목은 공식 자료의 범위와 현재 저장소 운영 제약을 함께 놓고 해석한 것이다.

- 공식 프레이밍은 mobile/web GUI 와 device-cloud collaboration 비중이 크다. 따라서 RCS 같은 산업용 레거시 Windows UI full-screen primary 로 바로 두기에는 일반화 불확실성이 있다.
- `STRICT_OFFLINE=1` 인 현재 배치에서는 `AskUser`, `MCP` 같은 협업/도구 확장 강점을 거의 활용하지 못한다.
- exact OCR authority 모델은 아니다.
- full-screen primary 로 항상 호출하기보다, crop 기반 재판독에 붙일 때 강점이 더 선명하다.

### 권장 배치 역할

- zoom-in sidecar
- ambiguous crop resolver
- small target specialist

### 4. `PaddleOCR-VL-1.5`

### 공식 자료 기준 강점

- document parsing, text spotting, complex element understanding 중심 OCR VLM 이다.
- `109` 개 언어를 지원한다.
- text, table, formula, chart, seal, spotting 을 다룬다.
- irregular-shaped localization, skew, warping, screen photography 같은 어려운 조건에 대한 강건성을 강조한다.
- 공식 사용법이 `OCR:`, `Spotting:`, `Table Recognition:`, `Formula Recognition:`, `Chart Recognition:` 같은 task keyword 중심으로 정리되어 있다.

### `poc/work2` 에서 가장 좋은 용도

1. dense text panel 읽기
2. small text, parameter value, table-like region 추출
3. OCR anchor 생성
4. multilingual / distorted capture 보강

### 한계

- primary click-grounding 모델로 쓰는 것은 맞지 않다.
- exact GUI target selection 보다 extraction engine 으로 두는 편이 맞다.
- chat-completions 형태로 감싸서 써도, 본질적으로는 task keyword 기반 OCR parser 로 접근하는 편이 더 안정적이다.
- 현재 `poc/work2/prompts/ocr_assist.py` 는 `OCR:` 만 고정 사용하고 있어 `Spotting:` 또는 `Table Recognition:` 분기를 아직 활용하지 못한다.

### 권장 배치 역할

- 기본 `ocr_service`
- OCR evidence engine
- structured extraction sidecar

### 5. `GOT-OCR-2.0-hf`

### 공식 자료 기준 강점

- 공식 모델 카드는 `plain texts`, `formatted texts`, `tables`, `charts`, `equations`, `molecular formulas`, `music scores`, `geometric shapes` 같은 다양한 OCR 산출물을 전면에 둔다.
- `General OCR Theory` 라는 이름처럼 다양한 scene OCR 을 광범위하게 포괄하려는 모델이다.
- interactive OCR 을 지원하며 color 와 coordinates 를 통해 특정 영역 OCR 을 수행할 수 있다고 설명한다.
- `dynamic resolution` 과 global/local module 설계를 강조한다.

### `poc/work2` 에서 가장 좋은 용도

1. 작은 crop 안의 exact text fallback
2. formatting 보존이 중요한 OCR
3. 좌표 또는 color 로 특정 region 만 다시 읽는 region OCR
4. `PaddleOCR-VL` 결과가 불충분할 때의 hard OCR backup

현재 저장소의 wrapper 도 이 역할과 잘 맞는다.

- `deploy_vlms/scripts/serve_got_ocr.py` 는 `format_output`, `color`, `box`, `crop_to_patches` 를 직접 노출한다.
- 즉, full-screen GUI VLM 처럼 쓰기보다 "문제 region 을 다시 읽는 OCR endpoint" 로 쓰는 편이 맞다.

### 한계

- 공식 모델 카드도 Hugging Face 구현 경로에서는 plain-text-first 흐름과 후처리 필요성을 함께 보여준다.
- page-level GUI grounding 또는 action planning 용 모델은 아니다.
- full-screen 의미 해석보다는 crop/region OCR 에 더 적합하다.
- 현재 서비스는 Flask proxy 를 통해서도 결국 `/v1/ocr` 형태로 호출해야 하므로, `/v1/chat/completions` 전용인 `poc/work2/vlm_client.py` 로는 바로 재사용되지 않는다.

### 권장 배치 역할

- OCR fallback
- formatting-sensitive OCR
- region-specific OCR tool

## 모델별 역할 요약

| 질문 | 가장 먼저 볼 모델 | 보조 모델 | 비고 |
|------|-------------------|-----------|------|
| "어디를 클릭해야 하나?" | `UI-Venus` | `MAI-UI` | full-screen -> crop retry |
| "다음 action 을 어떤 형식으로 낼까?" | `UI-TARS` | `MAI-UI` | agent-style action DSL 실험에 적합 |
| "이 패널의 정확한 텍스트/숫자가 뭐지?" | `PaddleOCR-VL-1.5` | `GOT-OCR-2.0-hf` | OCR authority 는 OCR 모델이 맡는다 |
| "작은 crop 안에서 label 과 input 이 헷갈린다" | `MAI-UI` | `PaddleOCR-VL-1.5` | local grounding + OCR 조합 |
| "formatting 이 중요한 코드/표/수식이다" | `GOT-OCR-2.0-hf` | `PaddleOCR-VL-1.5` | region OCR 우선 |

## 권장 파이프라인

### 파이프라인 A: 기본 GUI 자동화용

가장 실용적인 기본형은 아래다.

1. full-screen screenshot 을 `UI-Venus` 에 전달해 state, target 후보, confidence 를 얻는다.
2. confidence 가 충분하면 바로 action 하지 말고 target type 과 크기를 먼저 본다.
3. target 이 작거나 주변이 과밀하면 해당 영역 crop 을 `MAI-UI` 로 다시 본다.
4. target 선택에 exact text 가 필요하면 같은 crop 을 `PaddleOCR-VL-1.5` 로 읽는다.
5. `PaddleOCR-VL` 결과가 애매하거나 formatting 보존이 필요하면 `GOT-OCR-2.0-hf` 로 fallback 한다.
6. 좌표는 GUI 모델 출력을 우선하고, exact string 은 OCR 출력을 우선한다.
7. 둘이 충돌하면 unresolved 로 두고 재관찰한다.

### 이 파이프라인이 좋은 이유

- click target 판단은 GUI 모델이 더 잘한다.
- exact string 은 OCR 모델이 더 잘한다.
- small target 은 crop 확대가 필수인 경우가 많다.
- 현재 GPU 배치도 full-screen 주력과 sidecar 를 분리하도록 설계되어 있다.

### 파이프라인 B: 정보 추출 중심

screen analysis 보다 값 추출이 중요할 때는 순서를 바꾸는 편이 낫다.

1. full panel 또는 target panel crop 을 `PaddleOCR-VL-1.5` 로 먼저 읽는다.
2. OCR 결과를 `text`, `bbox` 또는 `polygon`, `score`, `block_type` 형태의 compact JSON 으로 정규화한다.
3. 이 compact OCR hint 를 `UI-Venus` 에 보조 정보로 넣어 semantic role 을 정하게 한다.
4. 작은 숫자, 코드, 포맷 민감 영역만 `GOT-OCR-2.0-hf` 로 재검증한다.

이 순서는 "무엇을 읽었는가"가 먼저이고 "그 값이 어떤 UI 의미를 갖는가"가 나중인 작업에서 더 안정적이다.

### 파이프라인 C: `UI-Venus` vs `UI-TARS` 비교 실험용

둘 다 GUI 주력 후보라면 아래처럼 비교하는 편이 좋다.

1. 동일한 screenshot 세트로 `UI-Venus` full-screen pass 실행
2. 동일한 screenshot 세트로 `UI-TARS` full-screen pass 실행
3. sidecar 규칙은 고정한다.
4. `MAI-UI`, `PaddleOCR-VL`, `GOT-OCR` 는 같은 조건에서만 escalation 한다.
5. hit rate, click drift, retry count, escalation rate 를 비교한다.

이렇게 해야 primary GUI 모델 차이와 sidecar 효과가 섞이지 않는다.

## `poc/work2` 적용 메모

현재 코드 기준 해석은 아래가 가장 자연스럽다.

1. `poc/work2/flask_vlm.py` 의 기본 `screen_analysis_service` 와 `main_tabs_service` 는 계속 `ui-venus` 로 둔다.
2. `ocr_service` 는 계속 `paddleocr-vl-1.5` 로 둔다.
3. `MAI-UI` 는 global default 로 박기보다, `Work2VLMClient(service_slug="mai-ui")` 로 crop case 에만 직접 호출한다.
4. `UI-TARS` 는 `UI-Venus` 대체 후보이므로 A/B 실험 축으로 둔다.
5. `GOT-OCR` 는 chat-completions 공용 경로가 아니라 OCR 전용 helper 로 따로 붙인다.

즉, 지금 구조를 크게 흔들지 않으려면 `purpose-based 기본값 + direct sidecar call` 이 가장 단순하다.

## 하지 말아야 할 것

1. 5개 모델을 매 step 항상 병렬 호출하지 않는다.
2. `PaddleOCR-VL-1.5` 나 `GOT-OCR-2.0-hf` 를 primary click-grounding 모델로 쓰지 않는다.
3. `UI-Venus`, `UI-TARS`, `MAI-UI` 를 exact text authority 로 쓰지 않는다.
4. `MAI-UI` 를 full-screen 기본값으로 고정한 뒤 모든 화면에 태우지 않는다.
5. `UI-TARS` 를 단순 좌표 질의에도 항상 agent-style verbose output 으로만 쓰지 않는다.

## 최종 권고

- 현재 저장소 기준 기본 주력은 `UI-Venus`
- 대안 primary / 비교축은 `UI-TARS`
- zoom-in 전문 sidecar 는 `MAI-UI`
- 기본 OCR sidecar 는 `PaddleOCR-VL-1.5`
- OCR fallback 은 `GOT-OCR-2.0-hf`

가장 실용적인 실제 운영 파이프라인은 아래 한 줄이다.

> `UI-Venus` 로 full-screen grounding -> 필요 시 `MAI-UI` 로 crop 재탐색 -> exact text 가 필요하면 `PaddleOCR-VL-1.5` -> formatting/hard OCR 만 `GOT-OCR-2.0-hf`

`UI-TARS` 는 이 스택을 대체하는 sidecar 가 아니라, **primary GUI 모델 자리를 두고 `UI-Venus` 와 경쟁하는 실험축**으로 두는 편이 가장 깔끔하다.

## Sources

### 로컬 설정

- `deploy_vlms/config/common.env`
- `deploy_vlms/config/models/ui-venus.env`
- `deploy_vlms/config/models/ui-tars.env`
- `deploy_vlms/config/models/mai-ui.env`
- `deploy_vlms/config/models/paddleocr-vl-1.5.env`
- `deploy_vlms/config/models/got-ocr-2.0-hf.env`
- `deploy_vlms/scripts/serve_got_ocr.py`
- `flask_api/vlm_serve/config.py`
- `poc/work2/flask_vlm.py`
- `poc/work2/vlm_client.py`

### 공식 자료

- UI-Venus GitHub: <https://github.com/inclusionAI/UI-Venus>
- UI-Venus model card: <https://huggingface.co/inclusionAI/UI-Venus-1.5-8B>
- UI-Venus technical report: <https://huggingface.co/papers/2602.09082>
- UI-TARS GitHub: <https://github.com/bytedance/UI-TARS>
- UI-TARS model card: <https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B>
- MAI-UI GitHub: <https://github.com/Tongyi-MAI/MAI-UI>
- MAI-UI model card: <https://huggingface.co/Tongyi-MAI/MAI-UI-8B>
- PaddleOCR GitHub: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR OCR pipeline docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>
- PaddleOCR document preprocessing docs: <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/doc_preprocessor.html>
- PaddleOCR-VL-1.5 model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>
- PaddleOCR-VL-1.5 technical report: <https://huggingface.co/papers/2601.21957>
- GOT-OCR-2.0 model card: <https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf>
