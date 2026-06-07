# 2단계 VLM 로케이터 — coarse → fine → confirm (deep dive)

> 대상: `login_rcs_ui_venus_mai.py`, `ui_venus_mai_locator.py`, `prompts/*`
> 상위 개요: `automation_methods_intro.md` §1

---

## 1. 왜 2단계인가

VLM 에게 **전체 스크린샷에서 픽셀 단위 정밀 좌표** 를 한 번에 달라고 하면 흔들립니다. 화면이 클수록,
요소가 작을수록 더 그렇습니다. 그래서 정밀도를 단계적으로 올립니다.

```
Stage 1 (coarse) : 전체화면에서 "대략 어디"  → bbox (ui-venus)
   ↓ crop + zoom
Stage 2 (fine)   : 확대된 조각에서 "정확히 한 점" → point (mai-ui)
   ↓ confirm (선택)
OCR 확인          : 그 자리 텍스트가 기대 라벨과 맞나
```

모델 분담: **ui-venus = 영역 식별(grounding)**, **mai-ui = 확대 crop 정밀 클릭점**.

---

## 2. Stage 1 — coarse bbox (ui-venus)

`_run_ui_venus_coarse_bbox()` (in `ui_venus_mai_locator.py`).

- 입력: full 스크린샷(WebP b64) + 요소 설명.
- 프롬프트: `build_ui_venus_single_element_bbox_prompt()` (in `prompts/prompt_login_rcs_ui_venus.py`).
- 응답(JSON):
  ```json
  {"coord_system": "relative_1000", "bbox": {"left":0,"top":0,"right":1000,"bottom":1000}}
  ```
- 못 찾으면 refusal: `{"coord_system":"relative_1000","bbox": null}` (또는 공식 grounding 의 `[-1,-1]`).
- 후처리: `normalize_bbox_1000()` → 0–1000 클램프 → `bbox_1000_to_pixels()` 로 픽셀화 → `bbox_center()`.

### "first-letter anchoring" (요소 설명 작성 원칙)
요소를 **화면에 보이는 텍스트 단서로 anchoring** 합니다. 예: "the editable text field next to the **'User ID'**
label". 모델이 라벨 텍스트의 첫 글자를 먼저 찾고, 그것을 기준 삼아 인접한 클릭 대상을 식별하도록
유도하면 더 안정적입니다. (프로젝트 메모리: VLM Prompt-Building Principles)

요소 설명 사전(`UI_VENUS_LOGIN_ELEMENT_DESCRIPTIONS`)에 userid_input/password_input/login_button
등이 정의되어 있습니다.

---

## 3. crop — 대상 둘레를 패딩해서 잘라내기

`_build_crop_box()`. coarse bbox 둘레에 패딩을 줘 잘라냅니다. 패딩은 **요소별로 다릅니다**
(`TargetConfig`):

```python
left_pad_ratio   = 1.25   # bbox 폭 대비 (라벨이 왼쪽에 있어 넓게)
right_pad_ratio  = 0.45
vertical_pad_ratio = 1.6
min_crop_width   = 320
min_crop_height  = 120
```

예: login_button 은 라벨이 안에 있어 `left_pad_ratio=0.6, vertical_pad_ratio=1.0` 처럼 더 타이트.

> **왜 비대칭 패딩인가?** 입력칸은 보통 **라벨이 왼쪽**에 있으므로, 왼쪽을 넓게 잡아야 "User ID" 라벨이
> crop 안에 들어와 fine 단계가 문맥으로 활용할 수 있습니다.

`ensure_min_span()` 으로 최소 320×120 을 보장합니다 (너무 작게 잘려 정밀 단계가 굶지 않도록).

---

## 4. zoom — 확대해서 픽셀을 벌려주기

`_resize_crop_for_mai()`. 잘린 조각을 LANCZOS 로 확대합니다.

```python
TARGET_MIN_RESIZED_WIDTH  = 960
TARGET_MIN_RESIZED_HEIGHT = 320
MAX_RESIZED_WIDTH = 1400
MAX_RESIZED_HEIGHT = 900
MAX_UPSCALE = 4.0
```

scale 은 "최소 960×320 은 되도록" 하되 "최대 4배·1400×900 은 넘지 않도록" 클램프합니다. 작은 버튼도
fine 단계에서 충분한 픽셀을 확보하게 해 주는 단계입니다.

---

## 5. Stage 2 — fine point (mai-ui)

`_run_mai_ui_refinement()`.

- 입력: **확대된 crop 만** (전체화면 아님).
- 프롬프트: `build_mai_ui_zoom_prompt()` (in `prompts/prompt_login_rcs_mai_ui.py`).
- 핵심 규칙(프롬프트에 명시):
  - 좌표는 **이 crop 기준** (전체 스크린샷 아님).
  - 입력칸은 **왼쪽 안쪽 텍스트 시작 영역**, 버튼은 중앙, 콤보는 드롭다운 컨트롤 중앙.
  - 라벨/테두리/그림자/배경 클릭 금지. 라벨은 **문맥으로만** 사용.
- 응답(JSON): `{"coord_system":"relative_1000","userid_input":{"x":500,"y":250}}` 또는 `null`.

---

## 6. 역변환 — fine 좌표를 full image 로 되돌리기

`_map_resized_point_to_full_image()`. 확대 비율과 crop 오프셋을 거꾸로 적용합니다.

```python
crop_x = round(resized_point.x * (crop_w-1) / (resized_w-1))   # 확대 해제
full_x = crop_box["left"] + crop_x                              # crop 오프셋 더하기
# y 동일
```

이 full image 좌표가 이후 `image_point_to_screen()` (DPI 보정) 을 거쳐 실제 클릭 좌표가 됩니다.

---

## 7. confirm — OCR 로 클릭 전 검증 (선택)

`workflow_login.py` 의 클릭 스텝은 옵션으로 OCR 검증을 끼웁니다. 찾은 점 근처의 텍스트를 읽어 기대
라벨과 얼마나 가까운지 확인하고, **확인되지 않으면 클릭하지 않습니다.** (→ `../cv/ocr_spotting_intro.md`)

---

## 8. 좌표계 규약 — 0–1000 정규화

모든 모델은 **0–1000 정규화 좌표** 를 돌려줍니다(`coord_system: "relative_1000"`). 픽셀 변환:

```python
pixel = value / 1000.0 * (axis_size - 1)   # 클램프 [0, axis_size-1]
```

`util/json_utils.py` 의 `normalize_bbox_1000()` / `bbox_1000_to_pixels()` / `parse_coords()` 가
이를 담당합니다. 응답이 markdown fence·trailing comma·유니코드 따옴표로 지저분해도 `extract_json()` 이
3가지 후보(fence 제거 / 첫 balanced object / 원문) × 2 파서(json / ast.literal_eval)로 버텨 냅니다.

---

## 9. 실패 단계 구분 (디버깅용)

`ui_venus_mai_locator.py` 의 exit 코드:

| 코드 | 의미 |
|---|---|
| `success` | 정상 |
| `capture_failed` | 스크린샷 실패 |
| `window_activate_failed` | 창 활성화/foreground 실패 |
| `vlm_no_detection` (`ui_venus`/`mai_ui`) | 해당 단계가 null/무효 반환 |

실패한 단계가 로그에 찍히므로 "coarse 가 못 찾았는지, fine 이 못 찾았는지"를 바로 구분할 수 있습니다.

---

## 10. 핵심 상수 한눈에

| 상수 | 값 | 의미 |
|---|---|---|
| `TARGET_MIN_RESIZED_WIDTH/HEIGHT` | 960 / 320 | 확대 후 최소 크기 |
| `MAX_UPSCALE` | 4.0 | 최대 확대 배율 |
| `min_crop_width/height` | 320 / 120 | crop 최소 크기 |
| `left/right/vertical_pad_ratio` | 1.25 / 0.45 / 1.6 | 기본 패딩 비율 |
| `coord_system` | `relative_1000` | 0–1000 정규화 |
| `temperature` | 0.0 | 결정적 출력 |
| `max_tokens` | 4096 | 응답 상한 |
