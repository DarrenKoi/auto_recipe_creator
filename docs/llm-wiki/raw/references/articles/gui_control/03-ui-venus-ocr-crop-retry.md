# UI-Venus, OCR, 그리고 Crop-Retry

이 문서는 `UI-Venus` 중심 grounding, OCR sidecar, crop-retry 전략을 현재 repo 상태에 맞게 정리합니다.
핵심은 "지금 구현된 것"과 "다음에 붙일 것"을 분리하는 것입니다.

## 1. 현재 mainline 상태

현재 `poc/work2` mainline에서 실제로 구현된 중심 경로는 다음과 같습니다:

- `login_rcs.py`: 로그인 창 read-only 캡처
- `login_benchmark.py`: 동일 이미지에 대한 multi-service 비교
- `ocr_login_check.py`: OCR 성능과 위치 힌트 가능성 분리 검증

즉, 현재 기본 흐름은 "UI-Venus 단독 automation"이 아니라 "여러 GUI 서비스 비교 + OCR sidecar 확인"입니다.

`ui-venus`는 여전히 기본 GUI grounding 서비스이지만, 현재는 benchmark 비교군 중 하나로 다뤄집니다.

## 2. UI-Venus 프롬프팅 규칙

### 2.1 현재 공식 방향

`poc/work2/prompts/prompt_login_rcs_ui_venus.py`를 보면, 현재 공식 방향은 다음과 같습니다:

- 한 번에 하나의 요소를 요청한다
- 출력은 `[x,y]`
- 불가능하면 `[-1,-1]`

이 방식이 UI-Venus 1.5의 공식 grounding 형식과 가장 가깝습니다.

### 2.2 Batch 프롬프트의 위치

같은 파일에는 batch JSON 프롬프트도 남아 있습니다.
다만 이것은 새 mainline 규약이라기보다 benchmark/legacy 호환용으로 보는 편이 맞습니다.

실무 판단:

- 새로운 single-target grounding 실험: 단일 요소 프롬프트 우선
- 기존 login benchmark 계약 유지: batch 프롬프트 허용

### 2.3 좋은 target 서술

좋은 anchor:

- 보이는 label text
- row 관계
- panel 이름
- left/right/above/below 관계
- active/selected/checked 상태

예시:

- `"the editable text field next to the 'User ID' label"`
- `"the 'Log In' button near the bottom of the dialog"`

피할 것:

- 한 프롬프트에 여러 목적 섞기
- OCR과 planning과 grounding을 한 호출에 모두 넣기
- target 설명 없이 포괄적으로 "중요한 UI를 찾아라" 식으로 묻기

## 3. OCR 지원 방식

### 3.1 현재 구현된 OCR 빌더

`poc/work2/prompts/prompt_ocr_assist.py`는 현재 단순하게 `OCR:`만 반환합니다.
즉, mainline prompt helper는 "기본 OCR 호출"만 담당합니다.

### 3.2 OCR 비교 실험은 별도 스크립트에서 한다

`poc/work2/ocr_login_check.py`는 다음을 별도로 비교합니다:

- `PaddleOCR-VL-1.5`의 `OCR:`
- `PaddleOCR-VL-1.5`의 `Spotting:`
- `GOT-OCR` 기본 OCR
- `GOT-OCR` box 지정 OCR

이 구조가 중요한 이유:

- OCR prompt branching을 operational script 안에 과도하게 섞지 않음
- 텍스트 파싱 품질과 위치 힌트 품질을 분리해서 판단할 수 있음
- OCR을 click planner로 과대해석하지 않게 됨

### 3.3 OCR 역할 규칙

현재 권장 역할 분리는 다음과 같습니다:

- exact text recovery: OCR
- label/value 확인: OCR
- clickable surface grounding: GUI 모델
- conflicting evidence resolution: 사람 또는 추가 verification

## 4. Crop Retry를 실행해야 하는 시점

현재 repo에는 unified crop-retry orchestrator가 mainline으로 합쳐져 있지 않습니다.
그래도 다음 중 하나 이상이면 crop-retry를 붙일 가치가 큽니다:

- target이 매우 작다
- 첫 pass가 toolbar 또는 dense grid 주변에 떨어진다
- OCR은 label을 읽었는데 clickable area는 불분명하다
- full-screen grounding 결과가 유사 이웃 여러 개와 충돌한다

권장 구현 규칙:

1. full-screen 후보를 먼저 구한다
2. 후보 주변을 crop한다
3. crop에서 GUI 모델을 다시 호출한다
4. 필요하면 crop에 OCR을 추가한다
5. 원본 좌표계로 remap한다
6. strategy 이름과 evidence source를 JSON에 남긴다

자주 발생하는 실수:

- crop offset remap 누락
- pixel 좌표와 `relative_1000` 좌표 혼용
- crop 크기 기준 변환식을 잘못 적용

## 5. 병합 규칙

### 5.1 Text Button 또는 Tab

- OCR이 target text를 명확히 찾으면 text box를 strong evidence로 사용합니다.
- 최종 click point는 여전히 GUI grounding 결과와 크게 벗어나지 않아야 합니다.

### 5.2 Label 옆 Input Field

- OCR로 row anchor를 찾습니다.
- 최종 field click point는 GUI grounding으로 정합니다.
- field point가 row band와 어긋나면 거부합니다.

### 5.3 OCR 증거가 약한 경우

- GUI grounding 결과를 유지합니다.
- 대신 strategy를 `"gui_only"`처럼 명시해 artifact에 남깁니다.

### 5.4 증거가 충돌하는 경우

- 클릭하지 않습니다.
- unresolved 상태로 남깁니다.
- crop, raw response, overlay를 review용으로 저장합니다.

## 6. 현재 Repo 기준 파일

이 전략과 직접 연결되는 주요 파일:

- `poc/work2/login_rcs.py`
- `poc/work2/login_benchmark.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/flask_vlm.py`
- `poc/work2/prompts/prompt_login_rcs.py`
- `poc/work2/prompts/prompt_login_rcs_ui_venus.py`
- `poc/work2/prompts/prompt_login_rcs_ui_tars.py`
- `poc/work2/prompts/prompt_ocr_assist.py`
- `poc/work2/util/image_utils.py`

참고용 실험 파일:

- `poc/work2/login_rcs_ui_venus.py`
- `poc/work2/login_rcs_ui_venus_rev2.py`
- `poc/work2/login_rcs_ui_tars.py`
- `poc/work2/login_rcs_ui_tars_rev2.py`

위 실험 스크립트는 아이디어 참고에는 유용하지만, 현재 문서 기준 default entrypoint는 아닙니다.

## 7. 권장 구현 순서

1. `login_rcs.py`의 현재 benchmark 계약을 유지한다
2. crop helper와 remap helper를 별도 유틸로 추가한다
3. `mai-ui` 또는 동일 GUI 서비스로 crop 재시도를 붙인다
4. OCR evidence 병합은 `ocr_login_check.py`로 기준을 확인한 뒤 넣는다
5. strategy와 evidence source를 debug JSON에 명시한다
6. action을 붙이더라도 처음에는 read-only verification 단계부터 유지한다

## 8. Debug Artifact 표준

각 target 또는 각 서비스 실행마다 다음을 유지합니다:

- source screenshot JPEG
- 전송 payload WebP
- raw model output
- parsed JSON 또는 error JSON
- 필요 시 crop image
- overlay image
- strategy name과 evidence source를 포함한 최종 decision 기록

이 규칙을 지키면 모델을 바꾸거나 프롬프트를 바꿔도 비교가 가능합니다.
