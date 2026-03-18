# UI-Venus + OCR 정밀 그라운딩 구현 계획

날짜: 2026-03-18

## 목적

이 문서는 [`ui_venus_grounding_and_ocr_for_engineering_ui.md`](./ui_venus_grounding_and_ocr_for_engineering_ui.md)의 조사 내용을 실제 `poc/work2` 코드에 반영하기 위한 **구현 순서와 파일 단위 작업 계획**을 정리한다.

핵심 목표는 아래 3개다.

1. `UI-Venus`의 단일 전체 화면 그라운딩 결과를 그대로 쓰지 않고 크롭 재시도 단계를 붙인다.
2. `PaddleOCR-VL-1.5` 또는 일반 OCR 좌표를 이용해 라벨, 행, 텍스트 앵커를 기준으로 위치를 더 정밀하게 보정한다.
3. 최종 클릭 전후의 근거 자료를 JSON/JPEG/WebP로 남겨 디버깅 가능한 상태를 만든다.

이 계획은 범용 GUI 에이전트 전체를 새로 만드는 계획이 아니다. 우선은 **RCS 로그인 화면**에서 동작하는 좁은 파이프라인을 만든 뒤, 그 결과를 다른 화면으로 확장하는 방식으로 잡는다.

## 현재 코드 기준 출발점

이미 있는 것:

- [`poc/work2/prompts/prompt_login_rcs_ui_venus.py`](../../poc/work2/prompts/prompt_login_rcs_ui_venus.py)
  - `UI-Venus` 공식 단일 타깃 프롬프트가 이미 있다.
- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py)
  - 단일 타깃에 대해 `[x,y]` `relative_1000` 좌표를 읽는 흐름이 이미 있다.
- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py)
  - 현재는 항상 `OCR:`만 반환한다.
- [`poc/work2/ocr_login_check.py`](../../poc/work2/ocr_login_check.py)
  - `PaddleOCR-VL-1.5`에 대해 `OCR:`와 `Spotting:`를 비교하는 진단 코드가 있다.
- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py)
  - 창 캡처와 WebP 인코딩만 담당한다.
- [`poc/work2/login_benchmark.py`](../../poc/work2/login_benchmark.py)
  - 멀티 모델 비교 실행기는 있지만, 지금은 크롭 재시도와 OCR 근거 병합이 없다.

즉, 필요한 것은 완전한 신규 시스템보다 아래 4개다.

- 크롭 생성 및 좌표 역변환 유틸
- OCR 작업 종류 분기
- `UI-Venus` + OCR 병합 계약
- 로그인 화면에서 먼저 쓰는 얇은 오케스트레이션 계층

## 목표 상태

로그인 화면 기준 목표 흐름:

1. 로그인 창 캡처
2. `UI-Venus` 단일 타깃 전체 화면 그라운딩
3. 타깃별로 크롭 재시도 필요 여부 판단
4. 크롭 이미지 생성
5. `UI-Venus`로 크롭 그라운딩 재실행
6. 크롭에 대해 OCR 실행
   - 기본은 `Spotting:`
   - 텍스트 재확인이 목적이면 `OCR:`
7. OCR 앵커와 크롭 그라운딩 결과를 병합
8. 최종 좌표와 근거 JSON 저장
9. 클릭 또는 입력 후 재캡처로 검증

이 흐름은 처음에는 `userid_input`, `password_input`, `login_button` 정도의 소수 타깃에만 적용한다.

## 구현 원칙

- `poc/work2` 안에서만 닫힌 구조로 구현한다.
- `.env` + dataclass 설정을 사용하고 `argparse`는 추가하지 않는다.
- 좌표는 내부적으로 **픽셀 좌표**를 기준으로 저장한다.
- 모델의 원문 출력은 버리지 않고 별도 근거 산출물로 남긴다.
- 새 코드가 범용 프레임워크처럼 커지지 않게 로그인 화면부터 고정 시나리오로 검증한다.

## 1차 구현 범위

이번 라운드에서 바로 구현할 범위:

- `UI-Venus` 전체 화면 -> 크롭 재시도
- 크롭 기준 OCR `Spotting:` 호출
- OCR 앵커와 `UI-Venus` 크롭 좌표 병합
- 근거 JSON 저장
- 로그인 화면의 일부 타깃에 적용

이번 라운드에서 **하지 않을 것**:

- 메인 메뉴 이후 전체 RCS 워크플로우 일반화
- 액션 플래너 / 상태 머신 통합
- OCR 원문 전체를 프롬프트에 집어넣는 프롬프트 중심 방식
- bbox 기반 다중 타깃 범용 스키마

## 파일별 작업 계획

### A. `poc/work2/util/image_utils.py`

이 파일은 현재 캡처와 WebP 인코딩만 있다. 여기 또는 인접 유틸 파일에 크롭 및 좌표 변환 함수를 추가한다.

권장 추가 함수:

```python
def crop_image_region(image, region) -> Image.Image
def compute_centered_crop_region(center_x, center_y, image_width, image_height, crop_width, crop_height) -> tuple[int, int, int, int]
def relative_1000_to_pixel(value: float, axis_size: int) -> int
def remap_crop_point_to_original(crop_x: int, crop_y: int, crop_region: tuple[int, int, int, int]) -> tuple[int, int]
```

권장 이유:

- `login_rcs_ui_venus_rev2.py`의 `_to_pixel()`과 유사한 로직이 이미 있다.
- 크롭 좌표 역변환은 여러 스크립트가 재사용할 가능성이 높다.

대안:

- 별도 파일 [`poc/work2/util/crop_utils.py`](../../poc/work2/util)를 신설할 수도 있다.

실무적으로는 함수 수가 많지 않으므로 1차는 `image_utils.py` 확장이 더 단순하다.

### B. `poc/work2/prompts/prompt_ocr_assist.py`

현재는 무조건 `OCR:`이다. 여기서 작업 종류 분기가 필요하다.

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

허용 작업 종류:

- `OCR`
- `Spotting`
- `Table Recognition`

권장 이유:

- 이미 `ocr_login_check.py`가 `Spotting:`를 진단 대상으로 다루고 있다.
- 프롬프트 빌더에서 작업 종류 분기만 해도 호출부 수정이 단순해진다.

### C. 새 오케스트레이션 계층

권장 신규 파일:

- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py)

이 파일의 책임:

- `UI-Venus` 전체 화면 1차 호출
- 크롭 생성
- 크롭 재호출
- OCR 보조 호출
- 근거 병합
- 최종 좌표 선택

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

이 파일은 1차에서는 로그인 화면 전용으로 좁게 구현해도 된다. 이름만 범용적으로 잡고 내부는 로그인 타깃 기준 분기여도 괜찮다.

### D. `poc/work2/login_rcs_ui_venus_rev2.py`

이 파일이 1차 통합 지점이다.

현재:

- `TARGET_KEYS = ("userid_input",)`
- 전체 화면 이미지에 대해 단일 타깃 `UI-Venus` 호출만 수행

권장 변경:

- 전체 화면 결과 저장
- 크롭 재시도 결과 저장
- OCR 보조 결과 저장
- 병합된 최종 좌표 저장
- 오버레이에 전체 화면 좌표, 크롭 좌표, 최종 좌표를 다른 색으로 표시

1차 적용 타깃:

- `userid_input`
- 그다음 `password_input`
- 마지막 `login_button`

이 순서가 좋은 이유:

- 라벨 앵커와 입력 필드 관계를 검증하기 쉽다.
- 로그인 버튼은 텍스트 기반 타깃이라 OCR 병합 규칙이 더 단순하다.

### E. `poc/work2/ocr_login_check.py`

이 파일은 구현 대상이라기보다 **검증 도구**로 유지한다.

추가하면 좋은 기능:

- `OCR_TEST_GOT_BOX`와 같은 방식으로 크롭 이미지 자동 입력
- `Spotting:` 응답에서 좌표 힌트가 얼마나 나오는지 요약 강화
- 타깃 단어별 앵커 적중표 저장

이 스크립트는 메인 파이프라인에 넣기보다 비교 진단기로 유지하는 편이 낫다.

### F. `poc/work2/login_benchmark.py`

1차에서는 이 파일을 크게 흔들 필요는 없다.

다만 2차에서 아래로 확장 가능하다.

- benchmark 결과 스키마에 `crop_retry_used`, `ocr_refinement_used`, `final_strategy` 추가
- `ui-venus` 전용 benchmark 모드와 `ui-venus+ocr` 하이브리드 모드 비교

즉, 1차 구현은 `login_rcs_ui_venus_rev2.py`에 넣고, benchmark 확장은 그다음이다.

## 구현 순서

### 1단계. 유틸과 프롬프트 분기

수정 파일:

- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py)
- [`poc/work2/util/__init__.py`](../../poc/work2/util/__init__.py)
- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py)
- [`poc/work2/prompts/__init__.py`](../../poc/work2/prompts/__init__.py)

완료 기준:

- 크롭 영역 계산과 좌표 역변환이 순수 함수로 분리됨
- `build_ocr_assist_prompt(..., task_name="Spotting")` 가능

### 2단계. 파이프라인 모듈 추가

신규 파일:

- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py)

완료 기준:

- 로그인 타깃 1개에 대해 전체 화면 / 크롭 / OCR / 병합 JSON이 모두 나옴
- 최종 좌표 선택 로직이 함수 단위로 분리됨

### 3단계. 로그인 `rev2` 스크립트 통합

수정 파일:

- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py)

완료 기준:

- `userid_input`에 대해 근거 산출물이 남음
- 원본 전체 화면 좌표와 병합된 최종 좌표를 모두 기록함
- 오버레이에 최종 좌표가 시각적으로 보임

### 4단계. 타깃 확대

추가 타깃:

- `password_input`
- `login_button`

완료 기준:

- 입력 필드 타입 타깃과 텍스트 버튼 타입 타깃을 둘 다 다룸

### 5단계. 벤치마크 및 회귀 검증 보강

수정 파일:

- [`poc/work2/login_benchmark.py`](../../poc/work2/login_benchmark.py)
- [`poc/work2/ocr_login_check.py`](../../poc/work2/ocr_login_check.py)

완료 기준:

- `ui-venus` 단독 vs `ui-venus+ocr` 비교 수치가 저장됨
- office Windows에서 어느 타깃이 실제로 개선되는지 확인 가능

## OCR 병합 규칙

1차에서는 복잡한 점수 결합 대신 규칙 기반 병합으로 충분하다.

### 경우 1. 텍스트 기반 버튼 / 탭

- OCR 앵커 텍스트가 타깃 텍스트와 잘 맞으면 OCR 박스 중심을 우선
- 단, `UI-Venus` 크롭 좌표가 OCR 박스 바깥으로 멀리 떨어지면 미해결 상태로 남김

예:

- `Log In`
- `Cancel`
- `Recipe`

### 경우 2. 라벨 옆 입력 필드

- OCR은 라벨이 있는 행을 찾는 데 사용
- 최종 클릭은 `UI-Venus` 좌표 사용
- 단, `UI-Venus` 좌표가 라벨 행의 수직 범위와 너무 어긋나면 실패 처리

예:

- `User ID`
- `Password`
- `Server`

### 경우 3. OCR 앵커가 없는 경우

- `UI-Venus` 크롭 좌표를 그대로 최종값으로 사용
- `final_strategy = "ui_venus_crop_only"`

### 경우 4. 근거가 충돌하는 경우

- 최종 좌표를 만들지 않음
- `verification_required = true`
- 디버그 산출물만 남기고 클릭 단계는 건너뜀

이 규칙은 1차 구현에 충분하다. 학습 기반 결합이나 점수 보정은 나중 일이다.

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

- 새 CLI 플래그는 만들지 않는다.
- `poc/work2/.env`나 코드 기본값으로만 제어한다.

## 산출물 계약

디버그 산출물은 아래처럼 남기는 것이 좋다.

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

`merge_result.json`에는 최소한 아래가 있어야 한다.

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

순수 함수는 macOS 환경에서도 테스트 가능하므로 pytest를 붙이는 편이 좋다.

권장 신규 테스트:

- [`test/work2/test_prompt_ocr_assist.py`](../../test/work2)
  - `OCR`, `Spotting`, `Table Recognition` 작업 종류 분기 확인
- [`test/work2/test_grounding_crop_utils.py`](../../test/work2)
  - 크롭 영역 계산
  - `relative_1000 -> pixel` 변환
  - 크롭 좌표의 원본 역변환
- [`test/work2/test_grounding_pipeline_merge.py`](../../test/work2)
  - 라벨-필드 병합 규칙
  - 버튼 OCR 병합 규칙
  - 충돌 시 미해결 처리

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

- `userid_input`에 대해 전체 화면 좌표보다 크롭/최종 좌표가 더 안정적으로 같은 위치로 수렴한다.
- `Spotting:`이 라벨 앵커를 일정하게 잡는다.
- 근거 JSON만 봐도 왜 그 좌표가 선택됐는지 설명 가능하다.

2차 성공 기준:

- `password_input`, `login_button`까지 같은 구조로 확장된다.
- 로그인 화면에서 타깃별 오클릭이 줄어든다.

3차 성공 기준:

- 하이브리드 모드가 `ui-venus` 단독보다 실제 office Windows 검증에서 더 낫다는 정성/정량 근거가 생긴다.

## 구현 우선순위

가장 먼저 할 일:

1. `prompt_ocr_assist.py` 작업 종류 분기
2. 크롭 유틸 추가
3. `grounding_pipeline.py` 신설
4. `login_rcs_ui_venus_rev2.py`에 `userid_input`만 먼저 연결

그다음:

5. 테스트 추가
6. `password_input`, `login_button` 확대
7. benchmark 통합

이 순서가 좋은 이유:

- 가장 적은 변경으로 가장 큰 정확도 개선 지점을 먼저 검증할 수 있다.
- 로그인 화면에서 결과가 안 좋으면 main tab / recipe editor로 확장할 이유가 없다.

## 권장 다음 작업

이 문서 다음 바로 이어서 코드를 바꾼다면, 가장 좋은 첫 작업 묶음은 아래다.

- [`poc/work2/prompts/prompt_ocr_assist.py`](../../poc/work2/prompts/prompt_ocr_assist.py) 수정
- [`poc/work2/util/image_utils.py`](../../poc/work2/util/image_utils.py) 수정
- [`poc/work2/grounding_pipeline.py`](../../poc/work2/grounding_pipeline.py) 추가
- [`poc/work2/login_rcs_ui_venus_rev2.py`](../../poc/work2/login_rcs_ui_venus_rev2.py) 연결
- [`test/work2/test_prompt_ocr_assist.py`](../../test/work2) 추가
- [`test/work2/test_grounding_crop_utils.py`](../../test/work2) 추가

즉, 구현은 로그인 화면의 `userid_input` 하나부터 시작하는 것이 맞다.
