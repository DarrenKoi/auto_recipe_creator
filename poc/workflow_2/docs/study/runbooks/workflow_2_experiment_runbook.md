# Workflow 2 — 오프라인 이미지 실험 Runbook

> 범위: `poc/workflow_2/` · 성격: **실행 가능한 실험 순서서(보조 문서)**
> 권위 문서: [`workflow_2_procedure.md`](./workflow_2_procedure.md) (7단계 설계·경계의 단일 권위).
> 본 문서는 그 중 **자산 준비 + Step 1~3 + 클릭 좌표 실험**을, 실장비 없이 캡처 이미지만으로
> 순서대로 돌려보는 runbook 이다. 설계 원칙(특히 VLM/CV 경계)은 권위 문서를 따른다.

---

## 0. 이 실험에서 하려는 것

이미 캡처해 둔 **장비 화면 정적 이미지**만으로 VLM + classical CV 를 조합해서:

1. **SEM Monitor Box 이미지 영역 추출** — 라이브 SEM 영상이 나오는 큰 사각 영역을 식별·crop.
2. **마우스 클릭 위치 추출** — 두 종류:
   - (2A) **일반 RCS UI 클릭** 좌표 — 탭·버튼·입력창 등.
   - (2B) **SEM Monitor recenter 더블클릭** 좌표 — align key 후보 쪽으로 FOV 를 옮기는 점.
3. 추출 결과를 **정적 매칭**으로 검증하고 score 분포를 모아 임계값을 calibration.

### 실행 환경 (중요)
- **Mac (오피스 밖)**: classical CV·합성/정적 매칭·이미지 crop·마킹까지 실행 가능. **VLM 호출 불가**.
- **오피스 (Windows + Flask VLM proxy)**: 위 + VLM probe(단계 1A, 2A 의 VLM 부분) 실행 가능.
- 실시간 장비 조작(actuation)은 본 runbook **범위 밖**. 좌표는 계산·마킹까지만, 실제 클릭 금지.

### 코드 위치 주의
성숙한 코드는 `poc/workflow_2/` 에 있고 내부적으로 `poc.workflow_1` 을 import 한다
(`vlm_sem_monitor_box.py` 등). CLAUDE.md 가 설명하는 `poc/work2/` 와는 별개 트리다.
아래 경로는 모두 **실재하는 `poc/workflow_2/`** 기준이다.

---

## 단계 0 — 자산 준비 & 검증

캡처 이미지를 표준 레이아웃에 배치한다:

```
align_images/<eqp_id>/<class_name>/<recipe_name>/
├─ align_img_from_rcp/    IMAP0001.*(OM)   IMAP0002.*(SEM)   # 레시피 등록 align key
├─ align_img_from_msr/    S*/E*                              # 측정 궤적 (E 접두 = fail step)
└─ captured_img_from_rcs/ <tag>_rcs.jpg                      # fail 시점 RCS 전체 화면 캡처
```

- stem 규약: `IMAP0001`=OM, `IMAP0002`=SEM (`poc/workflow_2/__init__.py` 의 `RCP_OM_STEM`/`RCP_SEM_STEM`).
- `current_sem` = `align_img_from_msr` 의 최신 `E*`(fail step).
- 경로 해석은 `align_fail_assets.py` 의 `resolve_assets_auto()` 가 단일 창구 (최신 폴더 자동 선택,
  또는 `ALIGN_EQP_ID` / `ALIGN_CLASS_NAME` / `ALIGN_RECIPE_NAME` 환경변수 override).
  → `recipe_om`(IMAP0001) / `recipe_sem`(IMAP0002) / `current_sem` 으로 노출.

| 파일 | 종류 | 역할 |
|---|---|---|
| `poc/workflow_2/align_fail_assets.py` | ✅ 기존 (import 헬퍼, 단독 실행 X) | `resolve_assets_auto()` 로 자산 경로 해석·로딩 |
| `poc/workflow_2/__init__.py` | ✅ 기존 | 레이아웃 상수(`FROM_RCP_DIRNAME` 등) 정의 |

**놓치지 말 것**
- OM/SEM 파일명 구분 규칙(토큰 vs 순번)은 **오피스 미확정 게이트**. 자산을 받으면 실제 규칙부터 확인.
- `current_sem` 은 fail 시점 파일이 장비에 저장되지 않을 수 있음 → 라이브 캡처가 원본
  (`adr/0001-current-sem-is-live-captured-not-downloaded.md` 참조).

---

## 단계 1 — SEM Monitor Box 이미지 추출 (VLM + CV)

### 1A. VLM probe — VLM 이 Box 를 그릴 수 있는가
- VLM 에게 **오버레이 컨트롤 패널(‘Optics’, ‘Function’, ‘AMP’ …)에 가려진 영역까지 포함한
  라이브 SEM 영상 전체 사각형**을 하나의 bbox 로 반환하게 한다. 오버레이만 잡거나 가려지지
  않은 일부만 잡으면 오답이다.
- 박스 상단의 모드 라벨(예: ‘OM’, ‘Optics’)이 1차 앵커.

### 1B. CV landmark crop — 정밀 영역 추출
- `sem_panel_locator.py`: model 별 landmark crop 을 `cv2.matchTemplate(TM_CCOEFF_NORMED)` 로
  찾아 panel ROI 도출. 신뢰 floor `LANDMARK_CONF_MIN = 0.70`.
- 자산 필요: `poc/workflow_2/templates/sem_panel_landmarks/<model_id>/{landmark.jpg, meta.json}`.
  `meta.json` = `{"panel_offset": [dx, dy, w, h], "nm_per_pixel": <float|null>}`.

### 1C. 조합
- VLM bbox = **coarse 힌트/검증**, CV landmark = **정밀 crop**.
- crop 은 `util/image_utils.crop_image` + `build_relative_crop_box` 로 잘라 단계 3 입력으로 저장.

| 파일 | 종류 | 명령 / 사용 |
|---|---|---|
| `vlm_sem_monitor_box.py` | ✅ 기존 (runnable) | `uv run python poc/workflow_2/vlm_sem_monitor_box.py` (오피스) |
| `vlm_sem_monitor_box_realtime.py` | ✅ 기존 (runnable) | 실시간 변형 probe |
| `sem_panel_locator.py` | ✅ 기존 (import 헬퍼) | `locate_panel()` 호출, landmark 자산 필요 |
| `util/image_utils.py` | ✅ 기존 | `crop_image`, `build_relative_crop_box` |

**놓치지 말 것**
- 좌표계 `relative_1000` → `json_utils.bbox_1000_to_pixels` 로 픽셀 변환. prompt 에 width/height 전달.
- **tool model 마다 패널 레이아웃이 다름** → landmark 템플릿을 model 별로 준비.
- crop 결과를 단계 3 매칭의 입력으로 남길 것.

### 1D. landmark 란? & CV 정밀 추출 검증 절차

> **landmark = 화면에서 변하지 않는 시각적 기준점(앵커).** 진짜 찾고 싶은 Live SEM
> 박스는 영상 내용이 매 프레임 바뀌므로 그 자체를 기준으로는 찾을 수 없다. 그래서 박스 **바깥**의
> 안 변하는 UI 조각(패널 타이틀바·코너 아이콘·고정 라벨)을 landmark 로 삼아, "landmark 를
> 찾고 → 거기서 고정 거리(`panel_offset`)만큼 떨어진 곳이 SEM 박스" 로 역산한다.
> 비유: *"빨간 우체통(landmark)을 찾아라. 거기서 오른쪽 3m·아래 1m 가 우리집 문(SEM 박스)이다."*

좋은 landmark 조건: ① 고대비·distinctive(matchTemplate peak 또렷) ② **Live SEM 박스 바깥**
(내부는 매 프레임 변해서 매칭 불가) ③ 모든 프레임에 항상 보임 ④ model 별로 따로 준비.

검증은 **2단(코드 → 가설)** 으로 분리한다:

- **Mac (코드 검증, 실데이터 0)** — `test_sem_panel_locator.py` (합성 self-test, 기대 5/5):
  멀티-landmark argmax / `panel_offset` 산술 / frame 경계 clamp / confidence floor(없으면 None)
  의 4개 분기가 의도대로 도는지만 못 박아 둔다. 합성 프레임은 **SEM 박스 내부를 매 케이스 다른
  random 텍스처로 채우고 landmark 는 박스 바깥에 두어서**, "라이브 영상은 변해도 바깥
  landmark 는 안정적"이라는 운영 전제를 모사한다. ⚠️ PASS 해도 *가설* 증명은 아니다.
- **오피스 (가설 검증, 실데이터)** — 실제로 Live SEM 이 landmark 에서 고정 offset 인지:
  1. `capture_window_frames_tool.py` 로 Tool 창 프레임 캡처.
  2. **landmark 제작(수작업 1회)**: 한 프레임에서 안 변하는 UI 조각을 오려
     `templates/sem_panel_landmarks/<model_id>/landmark.jpg`, 그 조각→SEM 박스 거리를 재서
     `meta.json` 의 `panel_offset:[dx,dy,w,h]` 기입.
  3. `test_match_on_captured_frames.py` 실행(내부에서 `locate_panel` 호출).
  4. **확인 2가지**: overlay 의 ROI 박스가 모든 프레임에서 Live SEM 에 정확히 얹히는가 /
     `confidence` 가 일관되게 ≥ `LANDMARK_CONF_MIN`(0.70)인가. 경계값이면 더 distinctive 한
     landmark 로 교체한다. 창 리사이즈/DPI 변화에 약하면 ORB 기반 검출로의 폴백을 검토한다.

> 스크린샷은 **오피스 머신(Claude Code 직접 실행) 안에서만** 쓰이므로 건물 밖으로 나가지 않는다 —
> 보안 반출 금지와 충돌하지 않는다.

| 파일 | 종류 | 명령 / 사용 |
|---|---|---|
| `test_sem_panel_locator.py` | 🆕 신규 (runnable, Mac) | `uv run python poc/workflow_2/test_sem_panel_locator.py` (기대 5/5) |

---

## 단계 2 — 마우스 클릭 위치 추출 (VLM + CV)

> **VLM/CV 경계 원칙**: 최종 좌표는 **CV 가 결정**. VLM 은 영역 식별·feasibility·coarse 방향만.

### 2A. 일반 RCS UI 클릭 좌표
- 기존 locator 프롬프트 패턴(`prompts/prompt_login_rcs.py`, `prompts/prompt_rcs_main_tabs.py`)
  으로 VLM 이 `coord_system="relative_1000"` 좌표 반환.
- `json_utils.parse_coords(data, keys, img_w, img_h)` 로 픽셀 변환.
- CV 보정: 버튼/아이콘 template match 또는 edge snap 으로 클릭점 정밀화.
- `debug_image_utils.save_marked_image` 로 십자선·라벨 마킹 저장.
- **신규 실험 스크립트(작성 권장)** `poc/workflow_2/exp_click_locator.py`:
  정적 RCS 캡처 1장에 대해 VLM 좌표 + CV 보정 + 마킹 저장. 인자 없이 `uv run`
  (CLAUDE.md 규약: argparse 금지, 설정은 상수/env). *코드는 별도 승인 후 작성 — 현재는 명세만.*

### 2B. SEM Monitor recenter 더블클릭 좌표
- **CV 가 좌표 결정**: `align_key_matcher.compute_align_key_score(...)` 로 SEM Monitor crop 내
  best candidate 위치 산출 → 그 점이 더블클릭(=recenter, 클릭점이 FOV 중심) 목표.
- VLM 은 `vlm_align_key_box.py` 의 coarse `roi_hint` 로만 보조(범위 좁히기).
- ROI 내부 좌표 → screen 좌표 환산은 `live_align_search.py` 의 `move_to_point` 로직 /
  `util/window_utils.image_point_to_screen` 참고 (실제 actuation 은 범위 밖, 계산·마킹까지만).

| 파일 | 종류 | 명령 / 사용 |
|---|---|---|
| `prompts/prompt_login_rcs.py`, `prompts/prompt_rcs_main_tabs.py` | ✅ 기존 | UI locator 프롬프트 빌더 |
| `util/json_utils.py` | ✅ 기존 | `parse_coords`, `bbox_1000_to_pixels` |
| `util/debug_image_utils.py` | ✅ 기존 | `save_marked_image`, `save_marked_bboxes` |
| `exp_click_locator.py` | 🆕 신규(2A) | `uv run python poc/workflow_2/exp_click_locator.py` |
| `align_key_matcher.py` | ✅ 기존 | `compute_align_key_score` (2B 좌표 결정) |
| `match_recipe_key_on_crop.py` | ✅ 기존 (runnable) | crop 위 recipe key 매칭 |
| `vlm_align_key_box.py` | ✅ 기존 (runnable) | `roi_hint` 보조 |
| `live_align_search.py` | ✅ 기존 (runnable) | `move_to_point` 환산 로직 참고 |

**놓치지 말 것**
- 더블클릭 = recenter, wheel = FOV 중심 discrete 배율 (물리 규약 고정).
- actuation(`util/mouse_utils.py`)은 SAFE_MODE dry-run 이 기본이다. 본 실험에서 실제 클릭은 금지.
- VLM confidence 를 calibrated score 처럼 쓰지 말 것 (정량 score 는 CV).

---

## 단계 3 — 정적 매칭 검증 & calibration (CV)

- `compare_align_images.py` (Step 3): 등록 SEM(template) ↔ 현재 SEM(crop) 구조 비교 →
  score/decision/overlay + 한 줄 verdict(`match` / `adjust` / `low`). 자산 없으면 합성 self-test.
- `match_recipe_key_on_crop.py`: 단계 1 crop 위에서 recipe key 매칭.
- `test_match_on_captured_frames.py`: 실 캡처 프레임으로 score 분포 산출 →
  `STRUCTURE_POLICY` 가중치·임계값·`MIN_CONFIRM_SCALE`·`candidate_score` calibration (현재 cold-start).
- `test_align_key_match.py`: 합성 smoke test (10/10) 회귀 확인.

| 파일 | 종류 | 명령 |
|---|---|---|
| `compare_align_images.py` | ✅ 기존 | `uv run python poc/workflow_2/compare_align_images.py` |
| `match_recipe_key_on_crop.py` | ✅ 기존 | `uv run python poc/workflow_2/match_recipe_key_on_crop.py` |
| `test_match_on_captured_frames.py` | ✅ 기존 | `uv run python poc/workflow_2/test_match_on_captured_frames.py` |
| `test_align_key_match.py` | ✅ 기존 | `uv run python poc/workflow_2/test_align_key_match.py` |

**놓치지 말 것**
- 픽셀 동일성이 아니라 **edge 구조(Chamfer 위주)** 로 매칭한다 — align fail 자체가 live key 가 등록과
  "다르게" 보여서 발생하기 때문이다.
- 저배율 miniature 는 변별력 낮음 → broad 는 후보 제안, 확정은 zoom-in + scale~1.0 + ORB.
- `STRUCTURE_POLICY` 임계값은 cold-start → 실데이터로 반드시 calibration.

---

## 단계 4 — (옵션) 커서 / 클릭 이벤트 사후 분류

- `filter_frames_by_change.py`: 정적 프레임 제거 → `change_events.json` (✅ 완료).
- `vlm_cursor_click_filter.py`: 커서 위치 + 변화 영역으로 클릭 이벤트 분류.

| 파일 | 종류 | 명령 |
|---|---|---|
| `filter_frames_by_change.py` | ✅ 기존 | `uv run python poc/workflow_2/filter_frames_by_change.py` |
| `vlm_cursor_click_filter.py` | ✅ 기존 | `uv run python poc/workflow_2/vlm_cursor_click_filter.py` (오피스) |

---

## 단계별 python 파일 요약

| 단계 | 핵심 파일 (기존✅ / 신규🆕) | 산출물 |
|---|---|---|
| 0 자산 준비 | `align_fail_assets.py`✅, `__init__.py`✅ | 표준 레이아웃 자산, 경로 해석 |
| 1 SEM Box 추출 | `vlm_sem_monitor_box.py`✅, `sem_panel_locator.py`✅, `test_sem_panel_locator.py`🆕, `util/image_utils.py`✅ | Box bbox 마킹 이미지 + crop, locator self-test 5/5 |
| 2A UI 클릭 | `prompts/*`✅, `json_utils.py`✅, `debug_image_utils.py`✅, `exp_click_locator.py`🆕 | 클릭 좌표 마킹 이미지 |
| 2B recenter 클릭 | `align_key_matcher.py`✅, `match_recipe_key_on_crop.py`✅, `vlm_align_key_box.py`✅ | 더블클릭 목표점 + overlay |
| 3 정적 매칭 | `compare_align_images.py`✅, `test_match_on_captured_frames.py`✅, `test_align_key_match.py`✅ | score/decision/overlay, 분포 |
| 4 이벤트 분류(옵션) | `filter_frames_by_change.py`✅, `vlm_cursor_click_filter.py`✅ | `change_events.json`, 분류 결과 |

---

## 전역 — 놓치지 말아야 할 포인트

- **좌표계**: VLM 출력은 `relative_1000` 가정. prompt 에 width/height 전달,
  `json_utils.parse_coords` / `bbox_1000_to_pixels` 로 픽셀 변환.
- **VLM/CV 경계**: 좌표·정량 score 는 CV, 영역 식별·애매한 상황 설명·feasibility 는 VLM.
  VLM 단독 좌표 결정 금지, 낮은 CV score 를 VLM 답변만으로 override 금지.
- **SEM 패널 model 의존**: 패널 레이아웃이 model 마다 다르므로 landmark 템플릿을 model 별 준비.
- **코드 컨벤션 (CLAUDE.md)**: 한국어 docstring, `print` 기반 `[INFO]/[ERROR]/[WARNING]` 로깅,
  **argparse/CLI 인자 금지**(설정은 상수·env), `__future__` import 금지,
  디버그 이미지는 **JPEG 저장 / VLM 전송은 WebP(q90)**, SAFE_MODE 기본 `true`.
- **산출물 경로**: 디버그 이미지 `debug_images/<model-slug>/`, 녹화·필터 결과 `recordings/`.
- **오프라인 한계**: Mac 은 CV·합성/정적만, VLM 은 오피스 전용. Mac 검증은 production accuracy 보장 X.

---

## 검증 (Verification)

```bash
# 단계 1A — VLM 이 SEM Monitor Box 를 그릴 수 있는지 (오피스: Flask VLM 필요)
uv run python poc/workflow_2/vlm_sem_monitor_box.py

# 단계 1B/1D — CV landmark locator 합성 self-test (Mac, 실데이터 불필요, 기대 5/5)
uv run python poc/workflow_2/test_sem_panel_locator.py

# 단계 3 — 등록 SEM ↔ 현재 SEM 정적 비교 (자산 없으면 합성 self-test)
uv run python poc/workflow_2/compare_align_images.py

# 단계 2B / 3 — crop 위 recipe key 매칭
uv run python poc/workflow_2/match_recipe_key_on_crop.py

# 매칭 엔진 합성 smoke test (10/10 통과 유지 회귀)
uv run python poc/workflow_2/test_align_key_match.py

# 단계 4(옵션) — 정적 프레임 제거
uv run python poc/workflow_2/filter_frames_by_change.py
```

> `align_fail_assets.py` 와 `sem_panel_locator.py` 는 단독 실행 진입점이 없는 **import 헬퍼**다.
> 자산 해석은 `resolve_assets_auto()`, 패널 추출은 `locate_panel()` 을 다른 스크립트에서 호출해 검증한다.

**기대 산출물**: SEM Box 마킹 bbox 이미지, panel crop, 클릭 좌표 마킹 이미지,
매칭 score/decision/overlay, score 분포 표.
