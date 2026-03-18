# UI-Venus + OCR 정밀 그라운딩 구현 계획

Date: 2026-03-18

## 목적

이 문서는 [`ui_venus_grounding_and_ocr_for_engineering_ui.md`](./ui_venus_grounding_and_ocr_for_engineering_ui.md) 의 조사 내용을 실제 `poc/work2` 코드에 반영하기 위한 **구현 순서와 파일 단위 작업 계획**을 정리한다.

핵심 목표는 아래 3개다.

1. `UI-Venus` 단일 full-screen grounding 결과를 그대로 쓰지 않고 crop-retry 를 붙인다.
2. `PaddleOCR-VL-1.5` 또는 일반 OCR 좌표를 이용해 label / row / text anchor 를 정밀 보정에 쓴다.
3. 최종 클릭 전후의 evidence 를 JSON/JPEG/WebP 로 남겨 디버깅 가능한 상태를 만든다.

이 계획은 범용 GUI agent 전체를 새로 만드는 계획이 아니다. 우선은 **RCS 로그인 화면**에서 동작하는 좁은 파이프라인을 만든 뒤, 그 결과를 다른 화면으로 확장하는 방식으로 잡는다.

## 현재 코드 기준 출발점

이미 있는 것:

- [`poc/work2/prompts/prompt_login_rcs_ui_venus.py`](../../poc/work2/prompts/prompt_login_rcs_ui_venus.py)
  - `UI-Venus` 공식 single-target prompt 가 이미 있다.
- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py)
  - 단일 target 에 대해 `[x,y]` `relative_1000` 좌표를 읽는 흐름이 이미 있다.
- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py)
  - 현재는 항상 `OCR:` 만 반환한다.
- [`poc/work2/ocr_login_check.py`](../../poc/work2/ocr_login_check.py)
  - `PaddleOCR-VL-1.5` 에 대해 `OCR:` 와 `Spotting:` 를 비교하는 진단 코드가 있다.
- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py)
  - 창 캡처와 WebP 인코딩만 담당한다.
- [`poc/work2/login_benchmark.py`](../../poc/work2/login_benchmark.py)
  - 멀티모델 비교 러너는 있지만, 지금은 crop-retry 와 OCR evidence merge 가 없다.

즉, 필요한 것은 완전한 신규 시스템보다 아래 4개다.

- crop 생성 / 좌표 remap 유틸
- OCR task branching
- UI-Venus + OCR merge contract
- 로그인 화면에서 먼저 쓰는 얇은 orchestration layer

## 목표 상태

로그인 화면 기준 목표 흐름:

1. 로그인 창 캡처
2. `UI-Venus` single-target full-screen grounding
3. target 별로 crop-retry 필요 여부 판단
4. crop 이미지 생성
5. `UI-Venus` crop grounding 재실행
6. crop 에 대해 OCR 실행
   - 기본은 `Spotting:`
   - text readback 용이면 `OCR:`
7. OCR anchor 와 crop grounding 결과를 병합
8. 최종 point + evidence JSON 저장
9. 클릭 또는 입력 후 재캡처로 검증

이 흐름은 처음에는 `userid_input`, `password_input`, `login_button` 정도의 소수 target 에만 적용한다.

## 구현 원칙

- `poc/work2` 안에서만 닫힌 구조로 구현한다.
- `.env` + dataclass config 를 사용하고 `argparse` 는 추가하지 않는다.
- 좌표는 내부적으로 **픽셀 좌표**를 기준으로 저장한다.
- 모델 raw output 은 버리지 않고 별도 artifact 로 남긴다.
- 새 코드가 generic framework 처럼 커지지 않게 login 화면부터 고정 시나리오로 검증한다.

## 1차 구현 범위

이번 라운드에서 바로 구현할 범위:

- `UI-Venus` full-screen -> crop-retry
- crop 기준 OCR `Spotting:` 호출
- OCR anchor 와 `UI-Venus` crop point 병합
- evidence JSON 저장
- login 화면의 일부 target 에 적용

이번 라운드에서 **하지 않을 것**:

- main menu 이후 전체 RCS workflow 일반화
- action planner / state machine 통합
- OCR raw text 전체를 prompt 에 집어넣는 prompt-heavy 방식
- bbox 기반 multi-target universal schema

## 파일별 작업 계획

### A. `poc/work2/util/image_utils.py`

이 파일은 현재 capture + WebP encode 만 있다. 여기 또는 인접 util 파일에 crop / 좌표 변환 함수를 추가한다.

권장 추가 함수:

```python
def crop_image_region(image, region) -> Image.Image
def compute_centered_crop_region(center_x, center_y, image_width, image_height, crop_width, crop_height) -> tuple[int, int, int, int]
def relative_1000_to_pixel(value: float, axis_size: int) -> int
def remap_crop_point_to_original(crop_x: int, crop_y: int, crop_region: tuple[int, int, int, int]) -> tuple[int, int]
```

권장 이유:

- `login_rcs_ui_venus_rev2.py` 의 `_to_pixel()` 과 유사한 로직이 이미 있다.
- crop 좌표 remap 은 여러 스크립트가 재사용할 가능성이 높다.

대안:

- 별도 파일 [`poc/work2/util/crop_utils.py`](../../poc/work2/util) 를 신설할 수도 있다.

실무적으로는 함수 수가 많지 않으므로 1차는 `image_utils.py` 확장이 더 단순하다.

### B. `poc/work2/prompts/prompt_ocr_assist.py`

현재는 무조건 `OCR:` 이다. 여기서 task branching 이 필요하다.

권장 변경:

```python
def build_ocr_assist_prompt(
    width: int,
    height: int,
    context_label: str = "",
    focus_words: Iterable[str] | None = None,
    max_items: int = 12,
    task_name: str = "OCR",
) -> tuple[str, str]:
```

허용 task:

- `OCR`
- `Spotting`
- `Table Recognition`

권장 이유:

- 이미 `ocr_login_check.py` 가 `Spotting:` 를 진단 대상으로 다루고 있다.
- prompt builder 에서 task keyword branching 만 해도 호출부 수정이 단순해진다.

### C. 새 orchestration 레이어

권장 신규 파일:

- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py)

이 파일의 책임:

- `UI-Venus` full-screen 1차 호출
- crop 생성
- crop 재호출
- OCR sidecar 호출
- evidence merge
- 최종 point 선택

권장 dataclass:

```python
@dataclass(frozen=True)
class CropRetryConfig:
    enabled: bool = True
    crop_scales: tuple[float, ...] = (0.20, 0.35)
    ocr_task_name: str = "Spotting"
    use_ocr_refinement: bool = True

@dataclass(frozen=True)
class GroundingEvidence:
    instruction: str
    image_width: int
    image_height: int
    full_point: dict | None
    crop_region: dict | None
    crop_point: dict | None
    ocr_anchor_text: str
    ocr_anchor_box: dict | None
    final_point: dict | None
    final_strategy: str
```

핵심 함수:

```python
def ground_single_target_with_crop_retry(...)
def run_ocr_refinement_on_crop(...)
def merge_ui_venus_and_ocr(...)
```

이 파일은 1차에서는 login 화면 전용으로 좁게 구현해도 된다. 이름만 generic 하게 잡고 내부는 login target 기준 분기여도 괜찮다.

### D. `poc/work2/login_rcs_ui_venus_rev2.py`

이 파일이 1차 통합 포인트다.

현재:

- `TARGET_KEYS = ("userid_input",)`
- full-screen image 에 대해 single-target `UI-Venus` 호출만 수행

권장 변경:

- full-screen result 저장
- crop-retry result 저장
- OCR sidecar result 저장
- merged final point 저장
- overlay 에 full-screen / crop / final point 를 다른 색으로 표시

1차 적용 target:

- `userid_input`
- 그 다음 `password_input`
- 마지막 `login_button`

이 순서가 좋은 이유:

- label-anchor 와 input-field 관계를 검증하기 쉽다.
- 로그인 버튼은 text-bearing target 이라 OCR merge 규칙이 더 단순하다.

### E. `poc/work2/ocr_login_check.py`

이 파일은 구현 대상이라기보다 **검증 도구**로 유지한다.

추가하면 좋은 기능:

- `OCR_TEST_GOT_BOX` 와 같은 방식으로 crop 이미지 자동 입력
- `Spotting:` 응답에서 좌표 힌트가 얼마나 나오는지 summary 강화
- target word 별 anchor hit table 저장

이 스크립트는 main pipeline 에 넣기보다 비교 진단기로 유지하는 편이 낫다.

### F. `poc/work2/login_benchmark.py`

1차에서는 이 파일을 크게 흔들 필요는 없다.

다만 2차에서 아래로 확장 가능하다.

- benchmark result schema 에 `crop_retry_used`, `ocr_refinement_used`, `final_strategy` 추가
- `ui-venus` 전용 benchmark mode 와 `ui-venus+ocr` hybrid mode 비교

즉, 1차 구현은 `login_rcs_ui_venus_rev2.py` 에 넣고, benchmark 확장은 그 다음이다.

## 구현 순서

### Phase 1. 유틸과 prompt 분기

수정 파일:

- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py)
- [`poc/work2/util/__init__.py`](../../poc/work2/util/__init__.py)
- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py)
- [`poc/work2/prompts/__init__.py`](../../poc/work2/prompts/__init__.py)

완료 기준:

- crop region 계산과 remap 이 pure function 으로 분리됨
- `build_ocr_assist_prompt(..., task_name="Spotting")` 가능

### Phase 2. pipeline 모듈 추가

신규 파일:

- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py)

완료 기준:

- login target 1개에 대해 full-screen / crop / OCR / merge JSON 이 모두 나옴
- 최종 point 선택 로직이 함수 단위로 분리됨

### Phase 3. login rev2 스크립트 통합

수정 파일:

- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py)

완료 기준:

- `userid_input` 에 대해 evidence artifact 가 남음
- raw full-screen point 와 final merged point 를 모두 기록함
- overlay 에 final point 가 시각적으로 보임

### Phase 4. target 확대

추가 target:

- `password_input`
- `login_button`

완료 기준:

- input-field type target 과 text-button type target 을 둘 다 다룸

### Phase 5. benchmark / regression 보강

수정 파일:

- [`poc/work2/login_benchmark.py`](../../poc/work2/login_benchmark.py)
- [`poc/work2/ocr_login_check.py`](../../poc/work2/ocr_login_check.py)

완료 기준:

- `ui-venus` 단독 vs `ui-venus+ocr` 비교 수치가 저장됨
- office Windows 에서 어느 target 이 실제로 개선되는지 확인 가능

## OCR merge 규칙

1차에서는 복잡한 score fusion 대신 규칙 기반 merge 로 충분하다.

### Case 1. text-bearing button / tab

- OCR anchor text 가 target text 와 잘 맞으면 OCR box center 우선
- 단, `UI-Venus` crop point 가 OCR box 바깥 멀리 떨어지면 unresolved 로 남김

예:

- `Log In`
- `Cancel`
- `Recipe`

### Case 2. input field next to label

- OCR 은 label row 를 찾는 데 사용
- 최종 클릭은 `UI-Venus` point 사용
- 단, `UI-Venus` point 가 label row 의 수직 범위와 너무 어긋나면 fail

예:

- `User ID`
- `Password`
- `Server`

### Case 3. no OCR anchor

- `UI-Venus` crop point 를 그대로 final 로 사용
- `final_strategy = "ui_venus_crop_only"`

### Case 4. conflicting evidence

- final point 생성하지 않음
- `verification_required = true`
- 디버그 artifact 만 남기고 click 단계는 건너뛴다

이 규칙은 1차 구현에 충분하다. learned fusion 이나 score calibration 은 나중 일이다.

## 환경변수 / 설정 제안

`.env` 또는 코드 기본값으로 아래 정도를 둔다.

```text
WORK2_CROP_RETRY_ENABLED=true
WORK2_CROP_SCALES=0.20,0.35
WORK2_OCR_REFINEMENT_ENABLED=true
WORK2_OCR_REFINEMENT_TASK=Spotting
WORK2_SAVE_CROP_DEBUG=true
```

주의:

- 새 CLI flag 는 만들지 않는다.
- `poc/work2/.env` 나 코드 기본값으로만 제어한다.

## Artifact 계약

debug artifact 는 아래처럼 남기는 것이 좋다.

```text
debug_images/
  <timestamp>_ui-venus_full.jpg
  <timestamp>_ui-venus_input.webp
  <timestamp>_crop_1.jpg
  <timestamp>_crop_1.webp
  <timestamp>_ocr_raw.txt
  <timestamp>_merge_result.json
  <timestamp>_overlay.jpg
```

`merge_result.json` 에는 최소한 아래가 있어야 한다.

- target key
- instruction
- full-screen point
- crop region
- crop point
- OCR anchor text
- OCR anchor box
- final point
- final strategy
- verification required

## 테스트 계획

순수 함수는 macOS 환경에서도 테스트 가능하므로 pytest 를 붙이는 편이 좋다.

권장 신규 테스트:

- [`test/work2/test_prompt_ocr_assist.py`](../../test/work2)
  - `OCR`, `Spotting`, `Table Recognition` keyword branching 확인
- [`test/work2/test_grounding_crop_utils.py`](../../test/work2)
  - crop region 계산
  - `relative_1000 -> pixel` 변환
  - crop 좌표의 원본 remap
- [`test/work2/test_grounding_pipeline_merge.py`](../../test/work2)
  - label-field merge 규칙
  - button OCR merge 규칙
  - conflict 시 unresolved 처리

권장 실행:

```bash
uv run pytest test/work2/test_prompt_ocr_assist.py
uv run pytest test/work2/test_grounding_crop_utils.py
uv run pytest test/work2/test_grounding_pipeline_merge.py
```

Windows 검증은 office 환경에서 아래 순서로 한다.

1. `uv run python poc/work2/open_rcs.py`
2. `uv run python poc/work2/login_rcs_ui_venus_rev2.py`
3. `uv run python poc/work2/ocr_login_check.py`

## 성공 기준

1차 성공 기준:

- `userid_input` 에 대해 full-screen point 보다 crop/final point 가 더 안정적으로 같은 위치로 수렴한다.
- `Spotting:` 이 label anchor 를 일정하게 잡는다.
- evidence JSON 만 봐도 왜 그 point 가 선택됐는지 설명 가능하다.

2차 성공 기준:

- `password_input`, `login_button` 까지 같은 구조로 확장된다.
- login 화면에서 target 별 false click 이 줄어든다.

3차 성공 기준:

- hybrid mode 가 `ui-venus` 단독보다 실제 office Windows 검증에서 더 낫다는 정성/정량 근거가 생긴다.

## 구현 우선순위

가장 먼저 할 일:

1. `prompt_ocr_assist.py` task branching
2. crop util 추가
3. `grounding_pipeline.py` 신설
4. `login_rcs_ui_venus_rev2.py` 에 `userid_input` 만 먼저 연결

그 다음:

5. 테스트 추가
6. `password_input`, `login_button` 확대
7. benchmark 통합

이 순서가 좋은 이유:

- 가장 적은 변경으로 가장 큰 정확도 개선 포인트를 먼저 검증할 수 있다.
- login 화면에서 결과가 안 좋으면 main tab / recipe editor 로 확장할 이유가 없다.

## 권장 다음 작업

이 문서 다음 바로 이어서 코드를 바꾼다면, 가장 좋은 첫 작업 묶음은 아래다.

- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py) 수정
- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py) 수정
- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py) 추가
- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py) 연결
- [`test/work2/test_prompt_ocr_assist.py`](../../test/work2) 추가
- [`test/work2/test_grounding_crop_utils.py`](../../test/work2) 추가

즉, 구현은 login 화면의 `userid_input` 하나부터 시작하는 것이 맞다.
