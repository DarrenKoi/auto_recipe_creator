# Workflow 2 — Align Key 탐색 절차 및 구현 가이드

> 최종 갱신: 2026-05-25 · 범위: `poc/workflow_2/`
> 본 문서는 기존 `sem_monitor_control_implementation_map.html` 을 대체·통합한 **단일 권위 문서**다.
> (매니저 보고용 슬라이드는 `generate_status_report.py` → `workflow_2_status_report.{html,pptx}` 로 별도 유지.
> Chamfer/ORB/Hamming/filter 알고리즘 해설 HTML 들은 참고용 교육 자료로 유지.)

---

## 1. 개요와 경계

`poc/workflow_1/` 은 **RCS 로그인 → Tool 선택 → Align Fail 알람 감지 → Tool 화면 진입**까지 책임진다.
`poc/workflow_2/` 는 그 다음, **Align Fail 이 난 Tool 의 SEM Monitor 를 조작해 레시피에 등록된
align key 와 같은 위치를 찾아내는 흐름**을 담는다.

핵심 설계 원칙(2026-05-25 grill 결과 확정):

- **좌표는 CV 가 결정한다.** VLM 은 영역 식별·애매한 상황 설명·feasibility 평가에만 쓰고,
  최종 align key 좌표를 단독으로 결정하지 않는다.
- **Align fail 은 대개 live key 가 등록 이미지와 "다르게" 보이기 때문에 발생**한다. 따라서 픽셀
  동일성이 아니라 **edge 구조(Chamfer 위주)** 로 매칭하고, hard match 를 강요하지 않으며,
  최종 책임 판정은 **best candidate 를 엔지니어에게 넘기는 것**으로 둔다.

**보정 흐름의 핵심(2026-05-27 사용자 확정):** Align Fail 이 나면 SEM Monitor 는 paused live
화면에 **crosshair(가로/세로 십자선)** 를 그려 *현재(=잘못된)* align 위치를 표시하고 멈춘다.
엔지니어가 등록 이미지에 **그려 둔 박스의 중심(보통 정중앙)이 target point** 이고, 박스 *안의
모양*이 live 에도 보여야 한다. 보정 = crosshair 를 recipe-matched 점으로 옮기고(더블클릭=recenter)
**OK 버튼**을 눌러 진행. **align key 는 대개 잘못된 crosshair 근처에 이미 보이므로, 즉시
reposition+OK 하는 것이 PRIMARY 경로**이고, pan/zoom two-phase 탐색(Step 5~8)은 *아무것도 안
보일 때만* 도는 **FALLBACK** 이다. 둘을 가르는 단일 기준이 `key_visibility_gate` 다.

---

## 2. 트리거와 데이터 흐름

```
[workflow_1] ALID=9006(Align Fail) 감지
      │   align_fail_alarm.py 가 알림 + (RECIPE_ID 있으면) 장비 접속 후
      │   RCS 화면 캡처(rcs_screenshot.py) → captured_img_from_rcs 적재
      ▼
align_images/<eqp_id>/<class>/<recipe>/   (오피스 MES 생성 + 우리 캡처)
      ├─ align_img_from_rcp/  IMAP0001.*(OM)  IMAP0002.*(SEM)   (등록 align key)
      ├─ align_img_from_msr/  S*/E*                              (측정 궤적, E=fail)
      └─ captured_img_from_rcs/  <tag>_rcs.jpg                   (fail 시점 RCS 캡처)
      ▼
[workflow_2]
  Step 1·2 : VLM probe       (정적 파일, throwaway 평가)
  Step 3   : CV 정적 비교      (정적 파일)
  Step 4   : PRIMARY 보정      (paused 프레임 매칭 → reposition + OK)
      │  key 가 보이면(가시성 게이트) 여기서 끝.
      ▼  안 보이면(low) ↓
  Step 5~8 : FALLBACK live two-phase 탐색 (실시간 SEM Monitor pan/zoom)
```

> 경로 해석은 `poc/workflow_2/align_fail_assets.py` 의 `resolve_assets_auto()` 가 단일
> 창구로 담당한다(최신 폴더 자동 선택, 또는 `ALIGN_EQP_ID`/`ALIGN_CLASS_NAME`/`ALIGN_RECIPE_NAME`
> 환경변수로 override). 루트는 `poc/workflow_1/__init__.py` 의 `ALIGN_IMAGES_DIR`.
> **`recipe_om`(IMAP0001) / `recipe_sem`(IMAP0002) / `current_sem`(from_msr 최신 E*)** 로 노출된다.

---

## 3. 단계별 절차 · 파일 매핑 · 구현 상태

| 단계 | 내용 | 담당 파일 | 상태 |
|---|---|---|---|
| **Step 1** | 레시피 등록 이미지(OM·SEM)에서 VLM 이 align key 박스를 그릴 수 있는지 평가 | `vlm_align_key_box.py` | 🟡 코드 완성 / VLM 호출은 오피스 전용 |
| **Step 2** | 현재(실패) SEM 이미지에서 VLM 이 align key 박스를 그릴 수 있는지 평가 | `vlm_align_key_box.py` (동일 스크립트) | 🟡 코드 완성 / VLM 호출은 오피스 전용 |
| **Step 3** | 등록 SEM ↔ 현재 SEM 을 classical CV 로 정적 비교(구조 위주) | `compare_align_images.py`, `align_key_matcher.py` | ✅ Mac 검증 완료 |
| **Step 4 (PRIMARY)** | paused 프레임을 near-native scale 로 매칭 → 가시성 게이트 → best_xy 를 더블클릭 recenter → VLM 으로 OK 버튼 찾아 screen 클릭 | `align_fail_correct.py`, `vlm_ok_button_box.py` | 🟡 로직 검증(Mac dry-run) / 실장비·VLM 오피스 전용 |
| **Step 5 (FALLBACK)** | (게이트 low 일 때만) live SEM Monitor 를 더블클릭(recenter)으로 이동하며 탐색 | `live_align_search.py` | ✅ 로직 검증(mock) / 🔴 실장비 adapter 미구현 |
| **Step 6 (FALLBACK)** | 매 프레임 실시간 스코어링, match 또는 budget 까지 반복 | `live_align_search.py` + `align_key_matcher.py` | ✅ 로직 검증(mock) |
| **Step 7 (FALLBACK)** | 저배율 zoom-out → miniature 후보 탐색 → recenter → zoom-in 확정(two-phase) | `live_align_search.py` | ✅ 로직 검증(mock) |
| **Step 8 (FALLBACK)** | pan 10회 budget, 초과 시 best candidate 보고/escalation | `live_align_search.py` | ✅ 로직 검증(mock) |

범례: ✅ 구현·검증 완료 · 🟡 코드 완성(오피스에서만 실행) · 🔴 미구현 · 🟠 부분 구현

---

## 4. 파일별 구현 가이드

### 4.1 이미 구현·검증된 것

- **`align_key_matcher.py`** — Chamfer + ORB/RANSAC 매칭 엔진(좌표 결정 주체).
  - `MatchPolicy` / `DEFAULT_POLICY`(기존 §7.3 값, smoke test 호환) / `STRUCTURE_POLICY`(0.8 chamfer·0.2 orb, 낮은 임계값 — drift 견고).
  - `compute_align_key_score(..., scales=, policy=)` — scale band·정책을 호출부에서 주입.
  - `BROAD_SCALES`(0.15~0.5) — 저배율 miniature 탐색용. `DEFAULT_SCALES` 는 불변(기존 negative 케이스 보장).
  - 합성 smoke test `test_align_key_match.py` **10/10 통과 유지**.
- **`align_fail_assets.py`** — `align_images/<eqp>/<class>/<recipe>/` 에서 `recipe_om`(IMAP0001)/`recipe_sem`(IMAP0002)/`current_sem`(from_msr 최신 E*) 해석·로딩 공용 헬퍼(`resolve_assets_auto`).
- **`compare_align_images.py` (Step 3)** — 등록 SEM 을 template, 현재 SEM 을 search 대상으로 비교.
  - 두 crop 이 비슷한 크기여도 매칭되도록 frame 에 replicate 여백(pad)을 둘러 template 이 항상 들어가게 함.
  - 결과: score/decision + overlay + 한 줄 verdict(`match`/`adjust`/`low`). 합성 self-test 로 파이프라인 검증됨.
- **`live_align_search.py` (Step 4~7)** — two-phase 탐색 오케스트레이션.
  - 물리 규약: **더블클릭=클릭점 recenter, wheel=FOV중심 discrete 배율, mode(OM/SEM) 별 template routing.**
  - Phase A(broad): zoom-out + `WIDE_SCALES` 로 후보 탐색, 없으면 사각 spiral pan(10회 budget).
  - Phase B(confirm): 후보를 recenter → 단계별 zoom-in → scale~1.0 + ORB 일치 시 확정 match.
  - **terminal match 가드**: `best_scale ≥ MIN_CONFIRM_SCALE(0.6) AND orb_inlier_ratio > 0`
    → tiny-scale chamfer 단독 과신으로 인한 거짓 종료 방지.
  - 실장비 연결은 `SEMMonitorController` Protocol(아래)로만 분리. Mac 은 배율 시뮬 mock 으로 검증됨.
- **`align_fail_correct.py` (Step 4 PRIMARY)** — paused 화면 즉시 보정 오케스트레이션.
  - `key_visibility_gate(result)` — primary vs fallback 의 단일 분기. paused 프레임은 *레시피 등록
    배율*에서 멈춘 것이므로 `PAUSED_SCALES`(=near-native `DEFAULT_SCALES`)로 매칭한다. broad
    miniature band 를 쓰면 tiny-scale chamfer 과신으로 featureless 프레임도 거짓 가시가 된다.
    게이트는 `best_scale>=MIN_CONFIRM_SCALE` 를 요구하고, 약한 `adjust` 는 `orb>0`(feature 보강)일
    때만 가시로 인정해 배경 거짓양성을 차단(강한 `match` 는 edge 구조만으로 인정). cold-start.
  - 게이트 True → `clamp_to_fov(best_xy)` → `move_to_point`(더블클릭 recenter) → `capture_screen()`
    에서 OK 버튼 찾아 `click_screen`. False → `live_align_search` 위임.
  - `build_templates_from_assets(assets, crop_to_box=False)` — `recipe_om`/`recipe_sem` → `{OM,SEM}`.
    `extract_annotation_box()` 로 엔지니어 박스 내부만 crop 하는 옵션은 **미보정, 기본 off**(align key
    자체가 box-in-box 일 수 있어 주석 사각형과 혼동 가능 → 오피스 실파일로 calibration 후 결정).
  - `dry_run=True`(기본): best_xy·OK 좌표를 계산·로그·overlay 만 하고 actuation 안 함(§5 Phase 3).
    Mac self-test `test_align_fail_correct.py` 4/4 통과(gate / primary / fallback 위임 / OK 매핑).
- **`vlm_ok_button_box.py`** — paused 다이얼로그의 OK(확인) 버튼을 VLM 으로 찾아 *screen* 좌표 반환.
  ROI crop 이 아니라 전체 화면 프레임을 받으며(`capture_screen()`), Cancel/닫기와 구분. 좌표 결정이
  아니라 UI 버튼 *영역* 식별이므로 doc §8 의 CV/VLM 경계를 지킨다.
  - **좌표공간 분리(중요)**: `move_to_point`(reposition)은 SEM ROI 내부 FOV-local 픽셀,
    `click_screen`(OK)은 화면 절대 픽셀. 둘은 서로 다른 좌표계라 변환을 공유하지 않는다.

### 4.2 부분 구현(재사용 가능, 자산/튜닝 필요)

- **`sem_panel_locator.py`** — Tool 창 전체에서 SEM Monitor ROI 를 landmark 매칭으로 추출.
  → `templates/sem_panel_landmarks/<model>/{landmark.jpg, meta.json}` 자산 추가 필요.
- **`vlm_sem_monitor_box.py`** — SEM Monitor Box 영역을 VLM 으로 식별(좌표 결정 아님).
- **`vlm_cursor_click_filter.py`** — 커서 위치 + 변화 영역으로 click event 분류.
- **`filter_frames_by_change.py`** — 정적 프레임 제거(✅ 완료).
- **`search_align_key.py`** — 초기 spiral search shell(주입점 구조). live loop 의 spiral stepper(`_square_spiral_step`)를 `live_align_search.py` 가 재사용.

### 4.3 미구현 — 다음에 만들 것 (순서대로)

> 아래 순서는 의존성과 위험도를 고려한 권장 순서다.

1. **(오피스, 최우선) workflow_1 캡처 핸들러**
   - 파일: `poc/workflow_1/rcs_screenshot.py` (align fail 시 `align_fail_alarm.py` 가 연계 호출; 단독 실행도 가능)
   - 할 일: ALID=9006 + RECIPE_ID 시 장비 접속 → RCS 화면 캡처 →
     `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>_rcs.jpg` 로 적재 후 창 닫기.
     레시피 등록 OM/SEM(align_img_from_rcp)·측정 궤적(align_img_from_msr)은 오피스 MES 가 생성.
   - 완료 기준: Step 1~3 스크립트가 자산을 그대로 읽어 동작. **창 닫기 전략 확정**.

2. **(오피스, 최우선) 실장비 `SEMMonitorController` 구현**
   - 파일: 신규 `poc/workflow_2/rcs_sem_controller.py` (가칭)
   - 할 일: `live_align_search.SEMMonitorController` Protocol 의 **6개 메서드** 구현
     - `capture()` — RCS 창에서 SEM Monitor ROI 만 잘라 grayscale numpy 반환(`sem_panel_locator` 활용).
     - `move_to_point(fov_x, fov_y)` — ROI 내부(FOV-local) 좌표를 screen 좌표로 환산 후 더블클릭(recenter).
     - `zoom(direction)` — wheel up/down 한 단계(FOV 중심 기준 discrete 배율).
     - `read_mode()` — monitor mode label(OM/SEM) 읽기(`vlm_sem_monitor_box` 결과 또는 OCR).
     - `capture_screen()` — **(Step 4 PRIMARY 필수)** SEM ROI 가 아니라 전체 화면(또는 RCS 창)을
       반환. OK 같은 dialog 버튼이 ROI 밖에 있어 `vlm_ok_button_box` 가 이 프레임에서 OK 를 찾는다.
     - `click_screen(screen_x, screen_y)` — **(Step 4 PRIMARY 필수)** screen 절대 좌표 single click.
       `move_to_point`(FOV-local 더블클릭)과 좌표공간이 다르다. OK 버튼 클릭에 쓴다.
   - 완료 기준: Phase 1~4(아래 §5) dry-run 통과. **`capture_screen`/`click_screen` 누락 시
     `correct_align_fail` 이 reposition 후 OK 단계에서 AttributeError → 정렬 미완료.**

3. **(최우선) Safety gate**
   - 파일: `live_align_search.py` + `rcs_sem_controller.py`
   - 할 일: `SAFE_MODE=true` 기본값, ROI 밖 클릭 차단, 1회 이동량·zoom 횟수 상한, emergency stop.
   - 현재 `LiveSearchConfig` 에 budget/zoom 상한은 있으나 **actuation 단계의 안전장치는 미구현**.

4. **(높음) 실데이터 calibration**
   - 파일: `test_match_on_captured_frames.py` 확장 + 신규 calibration 스크립트
   - 할 일: positive/negative/hard-negative 실 SEM 샘플로 score 분포 산출,
     `STRUCTURE_POLICY` 의 가중치·임계값·`MIN_CONFIRM_SCALE`·`candidate_score` 보정(현재 cold-start).

5. **(중간) Engineer escalation 산출물**
   - 파일: `live_align_search.py` 의 `notify_fn` 구현(신규 헬퍼)
   - 할 일: 최근 N회 score, 레시피 template, best SEM frame, edge overlay 를 3-pane 이미지로 저장·전송.

6. **(probe 결과에 따라 결정) VLM 보조 broad spotting**
   - 파일: `live_align_search.py` (옵션 플래그, 기본 off) + `vlm_align_key_box.py` 출력 연계
   - 할 일: Step 1·2 probe 가 "VLM 이 miniature key 를 식별 가능"으로 나오면,
     broad 단계에서 VLM 이 coarse box(=`roi_hint`)를 제안하고 CV 가 그 안에서 최종 좌표를 결정.
     **doc 의 좌표 결정 경계는 유지**(VLM 은 영역만, 좌표는 CV).

---

## 5. 실장비 통합 단계 (Phase)

| Phase | 내용 | 클릭 여부 |
|---|---|---|
| **1** | Dry-run capture loop — RCS 활성화 + SEM Monitor ROI 1~2초 주기 캡처 | ❌ 클릭 없음 |
| **2** | Offline matcher calibration — 실 template·live FOV 로 score 분포 확인 | ❌ |
| **3** | Move adapter dry-run — 계산된 더블클릭 좌표·zoom 을 로그/overlay 로만 검증 | ❌ |
| **4** | Bounded live action — 이동량·zoom 횟수·budget 제한 하에 실제 동작 | ✅ 제한적 |

---

## 6. 알려진 한계 (중요)

- **저배율 miniature 단계의 CV 변별력은 낮다.** template 을 크게 축소(~0.15~0.3)하면 edge 픽셀이
  적어 feature 없는 배경에서도 chamfer 가 높게 나올 수 있고, ORB 도 그 스케일에선 무력하다.
  → broad 단계는 *후보 제안* 수준이며, 진짜 판정은 confirm 단계(zoom-in 후 scale~1.0 + ORB)로 미룬다.
  → "miniature 가 정말 align key 인가"를 저배율에서 직접 판단하는 일은 CV 만으로 신뢰도가 낮으며,
  **이것이 Step 1·2 의 VLM probe 로 평가하려는 바로 그 능력**이다(§4.3-6 참조).
- `STRUCTURE_POLICY` 의 임계값은 모두 **cold-start** 값이다. §4.3-4 의 실데이터 calibration 필요.
- Mac 검증은 합성 데이터·배율 mock 한정이며 production accuracy 를 보장하지 않는다.

---

## 7. 실행 방법

```bash
# Step 4 PRIMARY — paused 보정 데모(Mac: 합성 프레임 + mock, dry-run)
uv run python poc/workflow_2/align_fail_correct.py
uv run python poc/workflow_2/test_align_fail_correct.py   # gate/primary/fallback/OK 매핑 self-test (4/4)

# Step 3 — 정적 비교(자산 없으면 합성 self-test)
uv run python poc/workflow_2/compare_align_images.py

# Step 1·2 — VLM probe (오피스: Flask VLM 필요)
uv run python poc/workflow_2/vlm_align_key_box.py

# Step 4~7 — live 탐색 로직 데모(Mac: 가상 wafer + 배율 mock)
uv run python poc/workflow_2/live_align_search.py

# 매칭 엔진 합성 smoke test (10/10)
uv run python poc/workflow_2/test_align_key_match.py
```

---

## 8. VLM 사용 경계 (재확인)

| 써도 되는 영역 | 피해야 할 영역 |
|---|---|
| SEM Monitor Box 식별 | 최종 align key 좌표 단독 결정 |
| OK dialog 버튼 위치 식별(screen 좌표) | crosshair reposition 좌표(=best_xy) 결정 |
| feature 없는 FOV 인지 설명 | VLM confidence 를 calibrated score 처럼 사용 |
| adjust 구간 coarse 방향 힌트 / broad roi_hint(평가 후) | OpenCV 낮은 score 를 VLM 답변만으로 override |
| engineer review 요약 생성 | 반복 가능성이 필요한 stage 이동 판정 |

> 운영 원칙: **정량 score 는 OpenCV 가, 화면 이해·애매한 상황 설명은 VLM 이.**
