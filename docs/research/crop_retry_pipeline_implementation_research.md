# Crop-Retry 파이프라인 구현 리서치 (2026-03-17)

## 목적

이 문서는 `deploy_vlms_model_roles_and_pipeline_research.md` 에서 제안한 crop-retry 파이프라인을 실제 구현할 때 필요한 3가지 핵심 질문에 대한 구체적인 설계를 정리한다.

1. 타깃이 "작다"는 기준을 어떻게 정의하는가?
2. crop 영역을 어떻게 잘라내고, crop 좌표를 원본 좌표로 어떻게 변환해야 마우스 클릭에 문제가 없는가?
3. VLM 모델 비교 실험 시 동일한 프롬프트를 쓸 수 있는가?

이 문서는 현재 `poc/work2/` 코드 구조를 기준으로 작성했다.

---

## Q1. 타깃이 "작다"의 정의

### 왜 중요한가

`deploy_vlms_model_roles_and_pipeline_research.md` 파이프라인 A 의 핵심 분기는 "target 이 작거나 주변이 과밀하면 crop 을 `MAI-UI` 로 다시 본다" 이다. 이 분기 조건이 명확하지 않으면 파이프라인이 매번 crop 을 태우거나 (느림), 아예 crop 을 안 태우거나 (정확도 하락) 한다.

### 판단 기준: 3가지 조건의 OR

타깃이 "작다" 또는 "crop-retry 가 필요하다" 는 아래 3가지 조건 중 하나라도 해당하면 참으로 한다.

#### 조건 1: 타깃 bbox 면적이 전체 이미지 대비 임계값 미만

```
target_area = bbox_width * bbox_height
image_area  = image_width * image_height
ratio       = target_area / image_area

if ratio < SMALL_TARGET_AREA_RATIO:
    → crop-retry 필요
```

**임계값 제안**: `SMALL_TARGET_AREA_RATIO = 0.003` (0.3%)

근거:
- 1920×1080 해상도에서 0.3% 는 약 6,220 px² → 대략 79×79 px 크기의 정사각형
- RCS 의 작은 버튼, 탭 텍스트, toolbar icon 은 보통 이보다 작다
- 이 값은 `poc/work2/login_rcs.py` 의 `LOGIN_WINDOW_MAX_AREA = 500000` (로그인 대화상자 전체 면적 제한) 과 다른 스케일이다 — bbox 1개 타깃 단위의 면적이다

#### 조건 2: 타깃 bbox 의 최소 변이 절대 픽셀 임계값 미만

```
min_side = min(bbox_width, bbox_height)

if min_side < SMALL_TARGET_MIN_SIDE_PX:
    → crop-retry 필요
```

**임계값 제안**: `SMALL_TARGET_MIN_SIDE_PX = 40` px

근거:
- VLM 입력 해상도는 보통 model-dependent 로 384~768 px 타일 기준이다
- 원본 스크린샷에서 40px 미만인 타깃은 VLM 내부 리사이징 후 몇 px 로 줄어들 수 있다
- 40px 은 일반적인 Windows toolbar icon (16~32px) 과 작은 탭 라벨을 포함한다

#### 조건 3: confidence 가 낮거나 주변에 유사 타깃이 과밀

```
if confidence < CONFIDENCE_THRESHOLD:
    → crop-retry 필요

if neighbor_count_within_radius > DENSE_NEIGHBOR_LIMIT:
    → crop-retry 필요
```

**임계값 제안**:
- `CONFIDENCE_THRESHOLD = 0.6`
- `DENSE_NEIGHBOR_LIMIT = 3` (반경 80px 이내 다른 타깃 bbox 수)

근거:
- 현재 `poc/work2/prompts/screen_analysis.py` 가 이미 `confidence: 0.0-1.0` 을 반환한다 (line 47)
- 0.6 미만이면 VLM 이 "확실하지 않다" 고 스스로 판단한 것이므로 확대 재탐색이 합리적이다
- 과밀 판단은 bbox 중심점 간 유클리드 거리가 80px 이내인 다른 타깃 수로 간단히 셀 수 있다

### bbox 가 없을 때는?

현재 `poc/work2/prompts/login_rcs.py` 는 좌표를 `{x, y}` 점으로 반환한다 (bbox 아닌 point). bbox 가 없으면 조건 1, 2 를 직접 쓸 수 없다.

이 경우 대안은 2가지다:

**대안 A: VLM 에 bbox 도 함께 반환하도록 프롬프트 수정**

```json
{
    "login_button": {
        "x": 500, "y": 800,
        "bbox": {"x1": 420, "y1": 770, "x2": 580, "y2": 830}
    }
}
```

이렇게 하면 조건 1, 2 를 그대로 쓸 수 있다. 단, VLM 출력이 커지고 bbox 정확도가 점 좌표보다 흔들릴 수 있다.

**대안 B: confidence 만으로 분기**

bbox 없이 `{x, y, confidence}` 만 받고, confidence < 0.6 이면 crop-retry 를 태운다.
가장 단순하고 현재 코드 변경이 적다.

**권장**: 초기 구현은 **대안 B** 로 시작하고, crop-retry 가 자주 불필요하게 발동하거나 반대로 누락되면 **대안 A** 로 전환한다.

### 구현 위치 제안

```
poc/work2/
├── crop_retry.py          # NEW: crop-retry 분기 판단 + 좌표 변환
```

`crop_retry.py` 에 `should_crop_retry(target, image_size, confidence)` → `bool` 함수를 두고, 판단 기준을 한 곳에서 관리한다.

---

## Q2. Crop 영역 생성과 좌표 변환

### 전체 흐름

```
[원본 스크린샷]               [crop 영역]                [최종 클릭 좌표]
1920 × 1080 px     →     crop 잘라냄      →     crop 내 좌표를
                          (예: 400×300)           원본 좌표로 변환
                                                    → pynput 클릭
```

핵심은 crop 영역의 offset 을 정확히 기록하고, VLM 이 crop 이미지에 대해 반환한 좌표를 원본 좌표계로 역변환하는 것이다.

### Step 1: crop 영역 결정

primary VLM 이 반환한 타깃 좌표 `(tx, ty)` 를 중심으로 crop 영역을 잡는다.

```python
def compute_crop_region(
    target_x: int,
    target_y: int,
    image_width: int,
    image_height: int,
    crop_size: int = 400,
    padding_ratio: float = 0.5,
) -> tuple[int, int, int, int]:
    """타깃 좌표 중심의 crop 영역 (x1, y1, x2, y2) 을 반환한다.

    crop 은 target 을 정중앙에 두되 이미지 경계를 넘지 않도록 clamp 한다.
    padding_ratio 는 타깃 주변에 얼마나 여유를 두는지 제어한다.
    """
    half = crop_size // 2

    # 일단 타깃 중심으로 잡는다
    x1 = target_x - half
    y1 = target_y - half
    x2 = target_x + half
    y2 = target_y + half

    # 이미지 경계 clamp
    if x1 < 0:
        x2 -= x1    # shift right
        x1 = 0
    if y1 < 0:
        y2 -= y1    # shift down
        y1 = 0
    if x2 > image_width:
        x1 -= (x2 - image_width)    # shift left
        x2 = image_width
    if y2 > image_height:
        y1 -= (y2 - image_height)   # shift up
        y2 = image_height

    # 최종 clamp (이미지가 crop_size 보다 작은 경우)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return x1, y1, x2, y2
```

**`crop_size` 선택 가이드**:

| 원본 해상도 | 권장 crop_size | 비고 |
|------------|---------------|------|
| 1920×1080 | 400~500 px | 타깃 주변 맥락 충분 |
| 2560×1440 | 500~600 px | 4K 스케일 고려 |
| 1280×720 | 300~400 px | 저해상도 |

crop_size 는 너무 크면 crop 의 의미가 없고 (full-screen 과 다름없다), 너무 작으면 VLM 이 주변 맥락을 보지 못해 오인식한다. **원본 해상도의 20~30%** 가 경험적 시작점이다.

### Step 2: 이미지 자르기

```python
from PIL import Image

def crop_image(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    """region = (x1, y1, x2, y2) 로 이미지를 자른다."""
    return image.crop(region)
```

### Step 3: crop 이미지를 VLM 에 전송

crop 이미지를 기존 `Work2VLMClient` 로 전송한다. 이때 **VLM 에는 crop 이미지의 크기를 알려줘야 한다**.

```python
crop_w = x2 - x1   # crop 이미지 너비
crop_h = y2 - y1   # crop 이미지 높이

# 프롬프트에 crop 이미지 크기를 전달
system_msg = (
    f"The screenshot is {crop_w}x{crop_h} pixels. "
    "Coordinate origin (0, 0) is the top-left corner of this cropped image."
)
```

VLM 은 이 crop 이미지 안에서의 좌표를 반환한다. 이 좌표를 `crop_x`, `crop_y` 라 하자.

### Step 4: crop 좌표 → 원본 좌표 변환 (핵심)

```python
def crop_coords_to_original(
    crop_x: int,
    crop_y: int,
    crop_region: tuple[int, int, int, int],
    coord_system: str = "pixel",
    crop_width: int | None = None,
    crop_height: int | None = None,
) -> tuple[int, int]:
    """crop 이미지 내 좌표를 원본 이미지 좌표로 변환한다.

    Args:
        crop_x, crop_y: VLM 이 crop 이미지에서 반환한 좌표
        crop_region: (x1, y1, x2, y2) — 원본 이미지에서의 crop 영역
        coord_system: VLM 이 사용한 좌표계
        crop_width, crop_height: crop 이미지 크기 (relative 좌표 변환에 필요)

    Returns:
        (original_x, original_y): 원본 이미지 좌표
    """
    x1, y1, x2, y2 = crop_region
    cw = crop_width or (x2 - x1)
    ch = crop_height or (y2 - y1)

    # 1) VLM 좌표를 crop 이미지 픽셀 좌표로 변환
    if coord_system == "relative_1000":
        pixel_x = int(round(crop_x / 1000.0 * (cw - 1)))
        pixel_y = int(round(crop_y / 1000.0 * (ch - 1)))
    elif coord_system == "relative_1":
        pixel_x = int(round(crop_x * (cw - 1)))
        pixel_y = int(round(crop_y * (ch - 1)))
    else:  # pixel
        pixel_x = crop_x
        pixel_y = crop_y

    # 2) crop offset 더하기 → 원본 좌표
    original_x = pixel_x + x1
    original_y = pixel_y + y1

    return original_x, original_y
```

### 좌표 변환이 틀리면 어떤 일이 생기는가

이 부분이 GUI 자동화에서 가장 위험하다. 구체적으로:

| 실수 | 증상 |
|------|------|
| crop offset 을 안 더함 | 항상 화면 왼쪽 위를 클릭함 |
| relative_1000 을 pixel 로 착각 | 좌표가 0~1000 범위로 좌상단에 몰림 |
| crop_width 와 실제 crop 이미지 크기 불일치 | 비례 오류로 클릭이 밀림 |
| clamp 로 crop 이 밀렸는데 offset 을 원래 값으로 기록 | 한쪽 방향으로 클릭이 밀림 |

### 안전장치: 역변환 검증

crop 좌표를 원본으로 변환한 뒤, 아래 검증을 반드시 수행한다:

```python
def validate_remapped_coord(
    original_x: int,
    original_y: int,
    crop_region: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> bool:
    """변환된 좌표가 합리적인 범위에 있는지 검증한다."""
    x1, y1, x2, y2 = crop_region

    # 1) 원본 이미지 범위 안에 있는가?
    if not (0 <= original_x < image_width and 0 <= original_y < image_height):
        return False

    # 2) crop 영역 안에 있는가? (crop 안의 타깃이었으므로)
    if not (x1 <= original_x <= x2 and y1 <= original_y <= y2):
        return False

    return True
```

### 전체 흐름 다이어그램

```
원본 스크린샷 (1920×1080)
    │
    ▼
┌───────────────────────────────┐
│  Primary VLM (UI-Venus)       │
│  → target (tx=850, ty=340)    │
│  → confidence=0.45            │
└───────────────────────────────┘
    │
    │  confidence < 0.6 → crop-retry 필요
    ▼
┌───────────────────────────────┐
│  compute_crop_region           │
│  center=(850,340), size=400   │
│  → crop_region=(650,140,1050,540)  │
└───────────────────────────────┘
    │
    │  PIL crop → 400×400 이미지
    ▼
┌───────────────────────────────┐
│  Sidecar VLM (MAI-UI)         │
│  이미지 크기: 400×400         │
│  coord_system: relative_1000  │
│  → crop 내 좌표 (510, 480)    │
└───────────────────────────────┘
    │
    │  crop_coords_to_original
    │  pixel_x = 510/1000 * 399 ≈ 204
    │  pixel_y = 480/1000 * 399 ≈ 192
    │  original_x = 204 + 650 = 854
    │  original_y = 192 + 140 = 332
    ▼
┌───────────────────────────────┐
│  validate_remapped_coord       │
│  (854, 332) in image? ✓       │
│  (854, 332) in crop region? ✓ │
└───────────────────────────────┘
    │
    ▼
  pynput click(854, 332)
```

### `poc/work2` 기존 코드와의 연결

- `poc/work2/util/json_utils.py` 의 `_to_pixel_coordinate()` 는 이미 `relative_1000`, `relative_1`, `pixel`, `percent` 변환을 지원한다. crop 좌표도 이 함수를 재사용하면 된다.
- `poc/work2/util/image_utils.py` 의 `encode_image_webp()` 는 crop 이미지에도 그대로 적용된다. crop 한 PIL Image 를 WebP 로 인코딩해서 VLM 에 보내면 된다.
- `poc/work2/util/debug_image_utils.py` 의 `save_marked_image()` 를 확장해서 crop 영역 사각형과 remapped 좌표를 함께 그리면 디버깅이 쉬워진다.

---

## Q3. VLM 모델 비교 시 동일 프롬프트 사용 가능 여부

### 짧은 답: 가능하지만 주의점이 있다

현재 배치 모델 5개 중 `/v1/chat/completions` 경로를 공유하는 모델은 아래 4개다:

| 모델 | 엔드포인트 | 프롬프트 형식 |
|------|-----------|-------------|
| UI-Venus | `/v1/chat/completions` | system + user (image + text) |
| UI-TARS | `/v1/chat/completions` | system + user (image + text) |
| MAI-UI | `/v1/chat/completions` | system + user (image + text) |
| PaddleOCR-VL-1.5 | `/v1/chat/completions` | task keyword only (`OCR:`) |

`GOT-OCR-2.0-hf` 는 `/v1/ocr` 전용 엔드포인트를 사용하므로 같은 프롬프트로 비교할 수 없다.

### GUI grounding 모델 3개 (UI-Venus, UI-TARS, MAI-UI) 는 동일 프롬프트로 비교 가능하다

이유:

1. **같은 API 형식**: 3개 모두 OpenAI-compatible `/v1/chat/completions` 이다. `poc/work2/vlm_client.py` 의 `Work2VLMClient` 가 `service_slug` 만 바꾸면 동일 요청을 보낸다.

2. **좌표계 통일 가능**: 현재 `poc/work2/prompts/login_rcs.py` 는 `coord_system='relative_1000'` 을 사용한다. 3개 모델 모두 이 좌표계를 이해할 수 있다 (프롬프트에서 명확히 정의하므로).

3. **현재 코드가 이미 이 구조**: `Work2VLMClient(service_slug="ui-venus")` → `Work2VLMClient(service_slug="mai-ui")` 만 바꾸면 같은 prompt 로 같은 이미지를 보낼 수 있다.

### 비교 실험 코드 구조 제안

```python
from poc.work2.vlm_client import Work2VLMClient
from poc.work2.prompts.login_rcs import build_login_rcs_locator_prompt
from poc.work2.util.image_utils import encode_image_webp

# 동일한 이미지와 프롬프트
image_b64, width, height = encode_image_webp(captured_image)
system_msg, user_msg = build_login_rcs_locator_prompt(width, height)

# 모델별 호출 — 프롬프트는 동일
results = {}
for slug in ["ui-venus", "ui-tars", "mai-ui"]:
    client = Work2VLMClient(service_slug=slug)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=system_msg,
        user_text=user_msg,
    )
    results[slug] = response.text
```

### 주의점: 모델마다 다를 수 있는 부분

동일 프롬프트를 쓸 수 있지만, 아래 차이는 알고 있어야 한다.

#### 1. 좌표 출력 형식 차이

같은 `coord_system='relative_1000'` 을 요청해도 모델마다 출력 형식이 다를 수 있다:

```json
// UI-Venus: 보통 깔끔한 JSON
{"login_button": {"x": 500, "y": 800}}

// UI-TARS: agent-style 로 action 문법을 섞을 수 있음
{"action": "click", "target": "login_button", "coords": [500, 800]}

// MAI-UI: 자체 action parser 형식을 섞을 수 있음
{"login_button": {"x": 500, "y": 800, "action": "tap"}}
```

**대응**: `poc/work2/util/json_utils.py` 의 `extract_json()` 과 `parse_coords()` 가 이미 유연한 파싱을 한다. 추가로 모델별 후처리가 필요하면 `parse_coords` 를 확장한다.

#### 2. 좌표 정밀도 차이

모델마다 같은 타깃에 대해 반환하는 좌표가 다를 수 있다. 이것이 비교 실험의 핵심 측정 대상이다:

```
| 타깃          | UI-Venus    | UI-TARS     | MAI-UI      | 실제 정답 |
|---------------|-------------|-------------|-------------|-----------|
| login_button  | (502, 798)  | (495, 810)  | (508, 795)  | (500, 800)|
| server_input  | (680, 210)  | (675, 220)  | (690, 208)  | (685, 215)|
```

**비교 지표 제안**:

- **click drift**: `sqrt((pred_x - gt_x)² + (pred_y - gt_y)²)` — 예측 좌표와 실제 정답 간 유클리드 거리
- **hit rate**: 예측 좌표가 실제 클릭 가능 영역 안에 들어가는 비율
- **retry count**: crop-retry 를 몇 번 탔는지
- **escalation rate**: OCR sidecar 까지 올라간 비율

#### 3. UI-TARS 는 좌표 post-processing 주의

`deploy_vlms_model_roles_and_pipeline_research.md` 에서도 언급했듯이, UI-TARS 공식 저장소는 absolute coordinates 후처리를 별도로 설명한다. 같은 프롬프트를 쓰되, UI-TARS 출력에는 추가 좌표 정규화가 필요할 수 있다.

#### 4. PaddleOCR-VL 은 비교 대상이 아니다

`poc/work2/prompts/ocr_assist.py` 를 보면 PaddleOCR-VL 은 `""`, `"OCR:"` 만 보낸다 (시스템 메시지 없음). GUI grounding 프롬프트를 PaddleOCR-VL 에 보내면 의미 없는 결과가 나온다. OCR 모델은 GUI grounding 비교에서 제외해야 한다.

### 비교 실험 권장 절차

1. **동일 이미지 세트 준비**: 다양한 RCS 화면 (로그인, 메인, 레시피 등) 스크린샷 10~20장
2. **동일 프롬프트 사용**: `build_login_rcs_locator_prompt()` 또는 `build_state_recognition_prompt()` 결과를 3개 모델에 동일하게 전송
3. **ground truth 표기**: 각 이미지에서 실제 클릭해야 할 좌표를 사람이 미리 표기
4. **3개 모델 순차 호출**: 같은 `(image, prompt)` 를 `ui-venus`, `ui-tars`, `mai-ui` 에 각각 전송
5. **결과 수집**: 각 모델의 raw 응답, 파싱된 좌표, confidence 를 JSON 으로 저장
6. **지표 계산**: click drift, hit rate, retry count, escalation rate
7. **sidecar 는 고정**: crop-retry 규칙과 OCR sidecar 는 3개 모델 모두 동일 조건으로 적용

### 프롬프트를 바꿔야 할 유일한 경우

모델의 **공식 문서가 특정 프롬프트 형식을 권장**하는 경우에만 프롬프트를 모델별로 분기한다.

예를 들어 UI-TARS 가 `GROUNDING` task template 을 권장하면:

```python
if slug == "ui-tars":
    system_msg = "GROUNDING: " + system_msg
```

이런 최소 수정만 하고, 나머지는 동일하게 유지한다. 이렇게 해야 "프롬프트 차이" 가 아닌 "모델 능력 차이" 를 비교할 수 있다.

---

## 구현 우선순위 요약

| 순서 | 작업 | 난이도 | 파일 |
|------|------|--------|------|
| 1 | confidence 기반 crop-retry 분기 (대안 B) | 낮음 | `poc/work2/crop_retry.py` (신규) |
| 2 | `compute_crop_region()` + `crop_coords_to_original()` | 중간 | `poc/work2/crop_retry.py` |
| 3 | `validate_remapped_coord()` 안전장치 | 낮음 | `poc/work2/crop_retry.py` |
| 4 | 3-model 비교 실험 스크립트 | 중간 | `poc/work2/compare_vlms.py` (신규) |
| 5 | bbox 기반 면적/최소변 판단 (대안 A 전환) | 중간 | prompt 수정 + `crop_retry.py` 확장 |

---

## Sources

### 로컬 코드

- `poc/work2/vlm_client.py` — `Work2VLMClient`, `ChatImageRequest`
- `poc/work2/util/json_utils.py` — `_to_pixel_coordinate()`, `parse_coords()`
- `poc/work2/util/image_utils.py` — `encode_image_webp()`
- `poc/work2/util/debug_image_utils.py` — `save_marked_image()`
- `poc/work2/prompts/login_rcs.py` — `build_login_rcs_locator_prompt()`, `coord_system='relative_1000'`
- `poc/work2/prompts/screen_analysis.py` — `confidence` 필드 반환
- `poc/work2/prompts/ocr_assist.py` — PaddleOCR-VL task keyword 전용
- `poc/work2/login_rcs.py` — `LOGIN_WINDOW_MAX_AREA` 등 기존 크기 제한
- `deploy_vlms/scripts/serve_got_ocr.py` — GOT-OCR crop_to_patches, box 파라미터

### 관련 리서치

- `docs/research/deploy_vlms_model_roles_and_pipeline_research.md` — 모델 역할/파이프라인 설계
