# UI-Venus, OCR, 그리고 Crop-Retry

이 문서는 밀도 높은 engineering UI를 대상으로 한 `UI-Venus`, OCR sidecar, crop-retry grounding 가이드를 정리한 문서입니다.

## 1. 시작 순서는 작업 유형에 따라 달라진다

### 1.1 Grounding 비중이 큰 작업

예시:

- 로그인 버튼 찾기
- tab 선택하기
- label 옆 input field 클릭하기

권장 순서:

1. `UI-Venus` full-screen pass
2. 필요하면 crop retry
3. 텍스트로 구분이 꼭 필요할 때만 OCR

### 1.2 Extraction 비중이 큰 작업

예시:

- parameter 값 읽기
- table row 추출하기
- 정확한 numeric field 검증하기

권장 순서:

1. 먼저 `PaddleOCR-VL-1.5` 또는 OCR pipeline 사용
2. 텍스트와 좌표를 normalize
3. semantic role 또는 clickable surface 결정에는 `UI-Venus` 사용

## 2. UI-Venus 프롬프팅 규칙

`UI-Venus`는 single-target grounder로 취급합니다.

권장 프롬프트 형태:

- screenshot 하나
- target 하나
- point 하나
- `[-1,-1]` 같은 명시적 refusal 경로

요청을 고정할 때 사용할 anchor:

- 화면에 보이는 label text
- row/column 관계
- panel 또는 dialog 이름
- left/right/above/below 관계
- active, selected, checked 같은 state 표현

더 좋은 예:

- "the editable text field to the right of the visible label 'User ID'"
- "the numeric input field in the 'Exposure' row inside the right parameter panel"

피할 것:

- 하나의 grounding call에 여러 target 넣기
- 점 하나만 필요할 때 긴 JSON schema 요구하기
- planning, OCR, grounding을 하나의 프롬프트에 섞기

## 3. OCR 모드 선택

| Need | Best mode |
| ------ | ----------- |
| 넓은 범위의 텍스트 추출 | `OCR:` |
| 텍스트와 위치가 모두 필요 | `Spotting:` |
| grid/table 구조 | `Table Recognition:` |
| 어려운 crop 재판독 | `GOT-OCR-2.0-hf` |

실무 규칙:

- 최종 click이 텍스트 좌표에 의존하면 `Spotting:` 사용
- 내용만 중요하면 `OCR:` 사용
- report형 화면이나 parameter grid에는 `Table Recognition:` 사용

## 4. Crop Retry를 실행해야 하는 시점

다음 중 하나 이상이 참이면 crop retry를 사용합니다:

- model confidence가 `0.6` 미만
- target의 최소 변 길이가 `40px` 미만
- target area ratio가 `0.003` 미만
- 약 `80px` 안에 유사한 이웃이 `3`개 초과로 존재
- 첫 번째 pass가 밀집된 toolbar 또는 parameter grid에 떨어짐

실무용 crop 정책:

- 첫 번째 예측 point를 중심으로 crop한다
- 짧은 이미지 변 길이의 `20%`~`30%` 정도로 시작한다
- context가 부족하면 더 큰 window로 한두 번 재시도한다
- 실행 전에 crop 좌표를 원본 pixel 좌표로 다시 매핑한다

자주 발생하는 실수:

- remap 시 crop offset을 잊어버리는 경우
- `relative_1000`와 pixel 좌표를 섞는 경우
- 원복 변환에 잘못된 crop width/height를 사용하는 경우

## 5. 병합 규칙

### 5.1 Text Button 또는 Tab

- OCR이 정확한 target text를 강한 box로 찾으면 OCR box 중심을 우선한다
- crop-grounded point가 그 box 주변에서 너무 멀리 벗어나면 병합을 거부한다

### 5.2 Label 옆 Input Field

- OCR로 올바른 row 또는 label anchor를 찾는다
- 최종 clickable field point는 `UI-Venus`로 결정한다
- field point가 anchor row와 맞지 않으면 거부한다

### 5.3 신뢰할 수 있는 OCR Anchor가 없음

- crop-grounded `UI-Venus` point로 fallback한다
- 사용한 전략을 debug JSON에 명시한다

### 5.4 증거가 충돌함

- 클릭하지 않는다
- 결과를 unresolved로 표시한다
- review용 artifact를 저장한다

## 6. 현재 Repo 기준 파일

이 전략과 정렬되어야 하는 주요 파일:

- `poc/work2/login_rcs_ui_venus.py`
- `poc/work2/login_rcs_ui_venus_rev2.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/pipeline_ocr.py`
- `poc/work2/prompts/prompt_login_rcs_ui_venus.py`
- `poc/work2/prompts/prompt_ocr_assist.py`
- `poc/work2/util/image_utils.py`

## 7. 권장 구현 순서

1. full-screen `UI-Venus` pass는 유지한다
2. crop-region helper와 coordinate remap helper를 추가한다
3. `prompt_ocr_assist.py`에 OCR task branching을 추가한다
4. raw OCR dump 대신 compact OCR hint를 저장한다
5. button target과 labeled input용 merge rule을 추가한다
6. 실제 click을 켜기 전에 evidence artifact를 저장한다

## 8. Debug Artifact 표준

각 target마다 다음을 유지합니다:

- source screenshot은 JPEG
- 적용 가능하면 전송 payload는 WebP
- raw model output
- crop image
- full-screen point, crop point, final point를 그린 overlay image
- strategy name을 포함한 최종 decision JSON

추측에 의존하지 않고 grounding 품질을 올리는 가장 빠른 방법입니다.
