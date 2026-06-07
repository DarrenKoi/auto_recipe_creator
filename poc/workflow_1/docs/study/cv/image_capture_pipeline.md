# 화면 캡처 → WebP → crop/zoom 파이프라인 (intro)

> 대상: `util/image_utils.py`, `vlm_client.py` (`encode_image_webp`), `debug_artifacts.py`, `__init__.py`
> 상위 개요: `../algorithms/automation_methods_intro.md`

---

## 1. 전체 흐름

```
mss 캡처(PIL Image)  →  [디버그용 JPEG 저장]  →  WebP(q=90) 인코딩  →  VLM API 전송
                     →  [crop → zoom]  (2단계 로케이터용)
```

프로젝트 전역 규약: **디버그 스크린샷은 JPEG 로 로컬에 저장하고, VLM 으로 보낼 때는 WebP(quality=90)로
변환** 합니다. 이렇게 하면 정확도는 유지하면서 payload 를 줄일 수 있습니다.

---

## 2. 캡처 — mss

`capture_window(window)` (in `util/image_utils.py`):
- `mss` 로 창 rect 영역을 빠르게 grab → `BytesIO` → `Image.open()` 으로 PIL Image.
- 캡처는 **물리 픽셀** 기준 (DPI 보정과 함께 봐야 함 → `../algorithms/dpi_coordinate_mapping.md`).

---

## 3. 인코딩 — 왜 WebP 인가

`encode_image_webp(image, quality=90)` (`vlm_client.py` / `util/image_utils.py`):
```python
image = image.convert("RGB")
image.save(buffer, format="WEBP", quality=quality)   # 기본 90
b64 = base64.b64encode(...)
return b64, width, height
```
- WebP 는 같은 화질에서 PNG/JPEG 보다 용량이 작아 **base64 payload 가 가벼워집니다** → 지연·비용 감소.
- VLM 전송 직전에 변환(`analyze_window_target()` 안에서 `encode_image_webp(image)`).
- 메시지 형식: `data:image/webp;base64,...` 를 OpenAI 호환 `image_url` 로 첨부.

---

## 4. crop / zoom — 2단계 로케이터 보조

| 함수 | 역할 |
|---|---|
| `build_relative_crop_box(w,h, l,t,r,b 비율)` | 비율(0~1) → 픽셀 crop box (클램프, 최소 1px span) |
| `crop_image(image, box)` | PIL `image.crop()` |
| `ensure_min_span(start,end,total,min)` | span 이 작으면 대칭으로 키움 (가장자리면 반대로) |
| `point_to_tiny_bbox(point,w,h,radius=10)` | 점 둘레 작은 bbox (오버레이 시각화용) |

List 영역 crop 비율 예(`workflow_select_tool.py`): `LEFT=0.00, TOP=0.10, RIGHT=0.42, BOTTOM=0.98`
— 좌측 패널(MC ID 컬럼)에 집중. (→ `../sem_box/rcs_list_tab_layout.md`)

---

## 5. 디버그 아티팩트 — JPEG/WebP/JSON/오버레이

`debug_artifacts.py`:
- `save_debug_jpeg(image, path)` — JPEG q=95 (사람 검토용).
- `save_debug_webp(image, path, quality=90)` — VLM 입력과 동일 포맷.
- `save_debug_text` / `save_debug_json(ensure_ascii=False, indent=2)`.
- `save_marked_bboxes(image, elements, colors, path)` — bbox 사각형 + 중심 crosshair/원 + 라벨을
  그린 오버레이 (arial 13pt, 없으면 기본 폰트).

저장 위치는 **모델별 폴더** `debug_images/<model-slug>/` 입니다. slug 는 `resolve_debug_model_name()` 가
`VLM_MODEL_NAME` env 나 기본 모델명으로 생성합니다(`_slugify_model_name`). 파일명은 `YYMMDD_HHMMSS_` 접두사
규칙(`debug_image_path()`)을 따라 시간순으로 정렬됩니다.

---

## 6. 디렉터리 상수 (`__init__.py`)

```python
DEBUG_IMAGE_DIR  = .../debug_images       # 모델별 하위폴더
LOG_DIR          = .../logs               # vlm_calls.log / work2.log / 알람 텍스트
RECORDING_DIR    = .../recordings         # CH4 프레임
ALIGN_IMAGES_DIR = .../align_images       # workflow_2 와 공유하는 핸드오프 트리
```

---

## 7. 핵심 상수 한눈에

| 항목 | 값 | 의미 |
|---|---|---|
| VLM 입력 WebP | quality=90 | payload 절감 + 정확도 유지 |
| 디버그 JPEG | quality=95 | 사람 검토용 |
| crop tiny bbox | radius=10 | 점 오버레이 |
| 오버레이 폰트 | arial 13pt | 라벨 |
| debug json | ensure_ascii=False, indent=2 | 한글·가독성 |
