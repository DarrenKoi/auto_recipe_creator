# Search-around 도달 범위 진단 — fallback spiral 이 왜 "근처만 건드리는가"

> **2026-08-28 저녁 갱신** — 본문의 "실제 pan 4회 / 예산 5" 는 진단 당시 값이다. 같은 날
> 사용자 결정으로 `pan_budget` 5→10, `low_streak_limit` 11(예산+1, `cycle.py` 도 함께 주입)로
> 바뀌어 **10회를 다 돌고 exhausted → cube 알림**이 나간다. 단, 반경 문제(step ≤ 0.38 FOV,
> zoom-out 부재)는 그대로라 10회로도 ~1 FOV 를 못 벗어난다. 다음 단계는 **(c) 절대 배율
> zoom-out 을 설계해 오피스에서 엔지니어와 실장비 테스트** 하는 것이다(사용자 결정).
> 배율/FOV 상수는 `hitachi_mag_fov_pixel_260828.md` 참조.


작성일: 2026-08-28 · 작성자: research note (Claude) · 상태: 소스 추적 보고서, 코드 미수정

> **질문**: align fail 보정 루프에서 착지 후 FOV 안에 align key 가 안 보이면 fallback
> "search around" 가 넓게 훑지 않고 착지점 근처만 살짝 움직인다. 소스에서 이유를 찾고,
> Hitachi CD-SEM 배율 물리로 지금 spiral 이 실제로 몇 µm 를 덮는지, 진짜 search-around 는
> 무엇이 필요한지 수치로 답하라.
>
> 근거는 소스 코드가 권위다. 배율 상수는 `skewnono_v3_nuxt/docs/datatables/hitachi/mag_pixel.txt`
> 하나뿐이며, 모든 µm 수치는 그 상수(135,000 µm, OFFICE-VERIFY)에 종속된다.

---

## 0. TL;DR

1. **step 이 픽셀이다.** spiral 한 칸은 `pan_step_px = 220` FOV 픽셀 (`live_search.py:133`) 이고
   `clamp_to_fov(margin 0.12)` 로 한 번 더 깎여 실효 step = `min(220, 0.38·fw)` px — 어떤 배율에서도
   **FOV 의 40 % 를 넘지 못한다.** 더블클릭 recenter 가 원리상 FOV 절반이 상한이라 이보다 클 수도 없다.
2. **4번 움직이고 끝난다.** 운영 기본 `pan_budget=5` (`config.py:328`) 인데 `low_streak_limit=5`
   (`live_search.py:130`) 가 먼저 걸려 아무것도 안 보이는 경우 **pan 4회 후 `escalated`** 로 종료한다
   (`live_search.py:277-283`). 4칸 spiral 의 최대 도달은 √2·step ≈ 0.54 FOV, 훑은 면적은 ~1.8×1.8 FOV.
3. **zoom-out 이 사실상 없다.** Phase A 의 "broad" 는 wheel 3 notch (`initial_zoom_out_steps=3`,
   `zoom_scroll_dy=1`) 인데, 같은 저장소의 zoom ladder 는 "1-2 notch 로는 배율이 거의 무의미하게
   바뀐다" 며 rung 당 5 notch 를 쓴다 (`config.py:299-300`). wheel↔배율 비는 미캘리브레이션
   (`controller.py:25, 205-206`) 이고 어떤 tool 은 wheel 이 배율을 안 바꾼다 (`config.py:303-309`).
4. **µm 로 환산하면**: SEM 30K (FOV 4.5 µm) 에서 최대 도달 ≈ **2.4 µm**, 100K 에서 ≈ **0.7 µm**.
   OM 104 에 SEM 공식을 그대로 적용하면 ≈ 700 µm 이지만 `mag_pixel.txt` 가 "OM Field_Size 는 값이
   없고 SEM 상수로 채우면 안 된다" 고 못박아 OM 수치는 근거 없음(OFFICE-VERIFY).
5. **빠진 것**: (a) 배율→FOV 를 아는 단위계(controller 에 `nm_per_pixel=None`, `controller.py:305`),
   (b) 실제로 배율을 바꾸는 zoom-out(PM 드롭다운은 check-only 전용, 보정 컨트롤러엔 미배선),
   (c) 1 FOV 를 넘는 step 을 만드는 recenter chain, (d) 출발점 기억/복귀. 전형적 align 오차 반경은
   소스 어디에도 없다(OFFICE-VERIFY) — 이것이 없으면 "얼마나 넓게" 를 정할 수 없다.

---

## 1. 물리 — 배율, FOV, 더블클릭 한 번의 이동 상한

### 1.1 공식 (단일 출처)

`mag_pixel.txt` §2:

```
FOV_µm       = 135,000 / Mag
PixelSize_nm = FOV_µm × 1000 / N_pixels
```

- 135,000 의 단위는 µm (2026-07-25 정정, §1). 기준 화면 ≈ 13 cm SEM 모니터 규격.
- office 근거는 cond.txt 표본 **1건** (30000× ↔ 4.499 µm, 곱 134,970). 상수가 정확히 135,000 인지는
  **OFFICE-VERIFY** (§1 "기준 화면").
- **픽셀 수는 FOV 를 바꾸지 않는다** (§1 첫 항목). 512 든 1024 든 같은 배율이면 같은 영역. 따라서
  아래 "pan 거리" 계산에 픽셀 수는 안 들어가고 *FOV 의 몇 분율* 만 들어간다.
- **OM 은 이 공식이 적용된다는 근거가 없다.** §4: "OM 은 key 만 있고 값이 없습니다 — SEM 상수로
  계산해 채우면 안 됩니다". 이 문서의 OM 행은 *공식을 기계적으로 넣으면 이렇게 된다* 는 참고값이며
  물리적으로 맞다는 뜻이 아니다.

### 1.2 대표 배율별 FOV 와 pan 상한

| Mag | FOV (µm) | nm/px @512 | 더블클릭 1회 이론 상한 (½ FOV) | 실효 상한 (0.38 FOV, margin 0.12) |
|---|---|---|---|---|
| OM 104 *(근거 없음)* | 1,298 | 2,535 | 649 | 493 |
| OM 210 *(근거 없음)* | 643 | 1,256 | 321 | 244 |
| SEM 5K | 27.0 | 52.7 | 13.5 | 10.3 |
| SEM 10K | 13.5 | 26.4 | 6.75 | 5.13 |
| SEM 20K | 6.75 | 13.2 | 3.38 | 2.57 |
| SEM 30K (cond.txt 표본) | 4.50 | 8.79 | 2.25 | 1.71 |
| SEM 50K | 2.70 | 5.27 | 1.35 | 1.03 |
| SEM 100K | 1.35 | 2.64 | 0.675 | 0.513 |
| SEM 200K | 0.675 | 1.32 | 0.338 | 0.257 |

- "이론 상한 ½ FOV" 는 `live_search.py:13` 의 물리 규약("한 번에 최대 ~FOV 절반 pan") 이자
  recenter-on-point 의 기하(가장자리를 클릭하면 그 점이 중심으로 = 절반 이동).
- "실효 상한 0.38 FOV" 는 `clamp_to_fov` (`live_search.py:179-191`) 가 `click_margin_ratio=0.12`
  (`live_search.py:137`) 로 클릭점을 `[0.12·fw, 0.88·fw]` 에 가두기 때문. 중심에서 최대 0.38·fw.
- 배율 계열 표: CG 23단 / GT 28단 (`mag_pixel.txt` §3). 5K→8K→10K→20K→50K 처럼 **이산** 이라
  "FOV 를 정확히 R 로" 는 불가능하고 가장 가까운 단을 고른다.

---

## 2. 현재 구현 추적

### 2.1 fallback 진입 조건 (primary → fallback)

`correction.py:correct_align_fail`:

1. paused 프레임 1장 캡처 → `compute_align_key_score_ensemble(scales=PAUSED_SCALES=DEFAULT_SCALES
   (0.7~1.4), policy=STRUCTURE_POLICY)` (`correction.py:257-259`, `engine.py:47`).
2. `key_visibility_gate` (`correction.py:160-200`):
   `best_scale < MIN_CONFIRM_SCALE(0.6)` → fallback; `decision=="match"` 또는
   `("adjust" and distinctive)` 가 아니면 → fallback (`:187-193`).
3. `fallback_search_enabled=False` 면 actuation 0회로 `escalated_key_not_visible` (`:297-320`,
   kill switch `ALIGN_FAIL_FALLBACK_SEARCH`, 기본 **on** `config.py:325, :460`).
4. on 이면 `live_align_search(controller, templates, config=fallback_config, ...)` 위임
   (`:322-331`). `dry_run` 인자는 **live_align_search 에 전달되지 않는다** — 실제 움직임 게이트는
   컨트롤러의 `action_enabled = settings.action_enabled and not settings.correction_dry_run`
   (`cycle.py:850, :869-878`) 하나다. production 진입점은 `_apply_live_mode_defaults` 가
   `SAFE_MODE=0` + `ALIGN_FAIL_CORRECTION_DRY_RUN=0` 을 못박으므로 (`align_fail_monitor.py:730-731`)
   **실운전에서 fallback 은 실제로 stage 를 움직인다.** 2026-08-24 오피스에서 실제로 그랬다
   (commit `ffd2d50`, `70acf9b`).

운영 주입값 (`cycle.py:987-1004`): `CorrectionConfig(fallback_search_enabled=settings.…)` +
`LiveSearchConfig(pan_budget=settings.search_pan_budget)` — **pan_budget 만 주입**, 나머지
(`pan_step_px`, `initial_zoom_out_steps`, `low_streak_limit`, `max_zoom_in_steps`) 는 라이브러리
기본값 고정. `workflow_3_config.example.py` 에는 fallback/pan 관련 상수가 없다(grep 0건) — 셸 env
`ALIGN_FAIL_FALLBACK_SEARCH` / `ALIGN_FAIL_SEARCH_PAN_BUDGET` 만 있다 (`config.py:460-461`).

### 2.2 Phase A 시작 — "zoom-out" 의 실체

```
for _ in range(max(0, config.initial_zoom_out_steps)):   # 3
    controller.zoom(-1)                                   # live_search.py:221-222
```

- `RCSSEMMonitor.zoom(direction)` = panel 중심에서 `scroll_at_screen(dy = direction * zoom_scroll_dy)`
  (`controller.py:202-215`), `zoom_scroll_dy=1` (`config.py:335`, env `ALIGN_SEM_ZOOM_SCROLL_DY`
  `:465`). 즉 **wheel 총 3 notch**.
- 미캘리브레이션 명시: "wheel 1단계 ↔ 배율 비율 (zoom_scroll_dy)" (`controller.py:25`),
  "TODO(캘리브레이션): wheel 1단계당 배율 변화율을 오피스에서 측정" (`:205-206`).
- 같은 저장소의 check-only zoom ladder 는 실측 후 **rung 당 5 notch** 로 올렸다: "1-2 notch 로는
  배율이 거의 무의미하게 바뀌어 … 5 로 상향" (`config.py:299-300`). 그 기준이면 3 notch 는
  1 rung 도 안 된다.
- 어떤 tool 은 wheel 이 배율을 **아예** 안 바꾸고 RCS 가 live box 안 wheel 을 recenter 로 오해한다
  (`config.py:303-309`, `zoom_method="auto"` 가 PM 드롭다운으로 폴백하는 이유). 그 tool 은 fab-out
  됐다고 적혀 있지만, `RCSSEMMonitor.zoom()` 에는 PM 드롭다운 경로가 없다 — `_run_pm_dropdown_arms`
  는 check-only 사이클 (`cycle.py:1690, :2095`) 에만 배선돼 있다.
- 결론: **Phase A 가 몇 배율로 내려가는지 코드로는 알 수 없고, 실측 근거로는 "거의 안 내려간다"
  쪽이다.** Mac mock 은 notch 1개 = zoom factor 한 단(4.0→3.0→2.0→1.4→1.0, `live_search.py:502`) 이라
  데모에서는 3 notch 가 3× 광각이 되지만, 실장비는 그 모델을 따를 근거가 없다.

### 2.3 spiral step 과 도달 거리

`square_spiral_step(idx, step)` (`search_pattern.py:4-21`): leg 길이 1,1,2,2,3,3,… 인 정사각 spiral,
반환은 **직전 위치 대비 delta** (`step` 단위).

`_do_pan` (`live_search.py:324-339`):

```
dx, dy = square_spiral_step(state.spiral_idx, config.pan_step_px)      # 220 px
target = clamp_to_fov(fw//2 + dx, fh//2 + dy, fw, fh, 0.12)
controller.move_to_point(target)                                       # 더블클릭 recenter
```

- step 단위 = **현재 프레임의 픽셀**. 프레임 = `capture()` 가 잘라낸 live SEM box ROI
  (`controller.py:122-131`, `panel_roi` = VLM 이 검출한 box 의 화면 픽셀 크기 `:296-305`).
  fw 는 RCS 창 크기·DPI 에 따르며 소스에 실측값이 없다(OFFICE-VERIFY). 테스트/데모는 320·768 px.
- 실효 step: `220 ≤ 0.38·fw ⇔ fw ≥ 579`. fw 가 그보다 작으면 clamp 가 잘라 **0.38·fw**.
  - fw=512 → 194 px = 0.38 FOV
  - fw=800 → 220 px = 0.275 FOV
- **µm 환산은 픽셀 수와 무관**: pan_µm = (step_px / fw) × FOV_µm. 배율을 올리면 FOV 가 줄어 같은
  픽셀 step 이 그만큼 짧아진다 — step 이 프레임 분율이라 **배율에 반비례해 수축**한다.

**pan 횟수 — budget 5 보다 escalation 이 먼저**. broad 분기 (`live_search.py:275-284`):

```
if state.pan_count >= config.pan_budget: return exhausted      # 5
state.low_streak += 1
if state.low_streak >= config.low_streak_limit: return escalated   # 5
_do_pan(...)
```

아무것도 안 보이면 iter 0..3 에서 pan 4회, iter 4 에서 `low_streak=5` → **`escalated`, pan_count=4**.
`low_streak` 은 후보(`score ≥ candidate_score=0.40`, `engine.py:172`) 가 나와 confirm 으로 갈 때만
0 으로 리셋된다 (`:272-274`). 즉 budget 5 는 후보를 쫓다 되돌아오는 경우에만 소진된다.

**4 pan 의 위치 (step 단위, 원점=착지점)**: (+1,0) → (+1,−1) → (0,−1) → (−1,−1).
최대 반경 √2 step, 훑은 FOV 중심의 외접 사각형 2×1 step. 방문 FOV 들의 합집합 ≈
(1 + 2·0.38) FOV ≈ **1.76 FOV 한 변** (fw=512 기준) — 겹침이 크고 3사분면 하나는 안 간다.

| Mag | FOV µm | 실효 step (0.38 FOV) | 4-pan 최대 도달 (√2 step) | 훑은 영역 한 변 (~1.76 FOV) |
|---|---|---|---|---|
| OM 104 *(근거 없음)* | 1,298 | 493 | 697 | 2,285 |
| SEM 10K | 13.5 | 5.13 | 7.3 | 23.8 |
| SEM 30K | 4.50 | 1.71 | 2.42 | 7.9 |
| SEM 50K | 2.70 | 1.03 | 1.45 | 4.8 |
| SEM 100K | 1.35 | 0.51 | 0.73 | 2.4 |

budget 10 이 온전히 pan 에 쓰이는 가상 경우(후보 없이는 불가능) 에도 10번째 위치는 (2,0) 으로
최대 반경 √5 step ≈ 0.85 FOV — 30K 에서 3.8 µm. **어떤 설정에서도 1 FOV 를 못 벗어난다.**

### 2.4 왜 "갔다가 돌아오는" 것처럼 보이는가

- candidate 임계 0.40 은 chamfer 비중 0.8 의 STRUCTURE_POLICY (`engine.py:168-173`) 인데 모듈
  docstring 이 "축소 template 의 chamfer 는 featureless 배경에서도 높게 나온다" 고 경고한다
  (`live_search.py:24-36`). 거짓 후보 → `move_to_point(후보)` + `zoom(+1)` 반복 (`:286-293`,
  최대 4회) → 놓치면 `zoom(-1)`×zoom_in_count 로 복귀 후 `_do_pan` (`:295-303`). spiral 의 다음
  delta 는 대개 방금 온 방향의 반대다 — 2026-08-24 오피스 관찰 "align key 로 잘 찾아갔다가 갑자기
  반대 방향으로 간다" 의 코드 경로 (commit `ffd2d50` 본문).
- confirm 종료 조건은 `decision=="match" and best_scale>=0.6 and orb_inlier_ratio>0`
  (`:251-255`). live search 는 non-ensemble `compute_align_key_score` 를 쓰므로 ORB 가 계산된다
  (`engine.py:1012-1018`) — 이 가드 자체는 살아 있다(ensemble 경로의 orb=0 고정과 다름).

### 2.5 탐색 중심과 복귀

- 중심 = fallback 진입 시점의 stage 위치, 즉 장비가 스스로 찾아간 fail 위치(=recipe 등록 좌표로
  간 결과). recipe 좌표계에서 "기대 위치" 를 따로 계산하는 코드는 없다 — `SEMMonitorController`
  Protocol 은 `capture/move_to_point/zoom/read_mode/capture_screen/click_screen` 뿐이고
  (`live_search.py:78-111`) stage 절대 좌표·`nm_per_pixel` 이 없다 (`controller.py:305 nm_per_pixel=None`).
- 종료 후 복귀 없음: `exhausted/escalated/best_candidate` 모두 `_finish_with_best` 로 즉시 반환
  (`:353-360`); 실 컨트롤러는 자기 위치를 모르므로 원점으로 돌아갈 수단 자체가 없다. best 후보의
  `fov_xy` 는 "그때 프레임의 픽셀" 이라 stage 가 움직인 뒤엔 의미가 없다.

---

## 3. 진단 — sweep 이 국소에 머무는 이유 (열거)

| # | 원인 | 근거 | 효과 |
|---|---|---|---|
| 1 | step 이 **프레임 픽셀**(220) + margin clamp | `live_search.py:133, :137, :335-337` | step ≤ 0.38 FOV, 배율 오를수록 µm 로 수축 |
| 2 | 더블클릭 recenter 는 원리상 **½ FOV** 가 상한이고 step 1회 = 클릭 1회 | `live_search.py:13`, `controller.py:177-188` | 1 FOV 이상 step 은 이 primitive 로 불가 |
| 3 | `low_streak_limit=5` 가 `pan_budget=5` 보다 먼저 | `live_search.py:130, :277-283` | 무후보 시 **pan 4회** 로 종료 |
| 4 | 4-pan spiral 은 ring 1 의 절반 | `search_pattern.py:8-21` | 최대 √2 step ≈ 0.54 FOV, 한 사분면 미방문 |
| 5 | zoom-out = wheel 3 notch, 미캘리브레이션 | `live_search.py:128, :221`, `controller.py:25, :205`, `config.py:299-300` | 배율이 유의미하게 안 내려감 → "broad" 가 아님 |
| 6 | wheel 이 배율을 안 바꾸는 tool 대비 PM 드롭다운이 보정 컨트롤러에 없음 | `config.py:303-309`, `cycle.py:1690, :2095` (check-only 전용) | 그 tool 에선 zoom(-1) 이 no-op 또는 recenter 오작동 |
| 7 | 배율→FOV 단위계 부재 (`nm_per_pixel=None`, PM 판독은 mode 힌트로만 씀) | `controller.py:305, :220-235` | "몇 µm 훑었다" 를 코드가 알 수 없음 |
| 8 | 거짓 후보(chamfer 0.40) → confirm 왕복이 pan 예산·시간을 먹음 | `live_search.py:24-36, :266-274, :286-303` | 왕복 후 spiral 재개 = 반대 방향 |
| 9 | 원점 기억/복귀 없음 | `live_search.py:353-360` | 끝나면 stage 가 spiral 끝점에 방치 |
| 10 | 운영 주입은 `pan_budget` 하나 | `cycle.py:1004` | step/zoom/streak 은 코드 수정 없인 못 바꿈 |

**fallback 자체는 기본 on** 이고 production 에서 실클릭이다(§2.1). "안 도는" 것이 아니라
"돌아도 1 FOV 안" 이다. `key_visibility_gate` 는 `best_scale<0.6` 또는 non-match 면 항상 fallback 을
고르므로 Phase A 진입 자체는 막히지 않는다.

---

## 4. 진짜 search-around 에 필요한 것 — 최소 변경 순

전제: **전형적 align 오차 반경 R 이 소스 어디에도 없다(OFFICE-VERIFY).** E 이미지·cond.txt
crosshair 이력에서 뽑을 수 있을 것이나 이 문서 범위 밖. 아래 도달값은 30K/100K SEM 기준.

| 순위 | 옵션 | 무엇이 바뀌나 | 도달 (30K, FOV 4.5 µm) | 도달 (100K, FOV 1.35 µm) | 제약 |
|---|---|---|---|---|---|
| (d) | **recenter chain**: spiral 1 step = 더블클릭 N회 연쇄 (N=3 → 1.14 FOV) | `_do_pan` 만 | step 5.1 µm; ring-1(8 cell) 후 반경 ≈ 7.2 µm | step 1.5 µm; ≈ 2.2 µm | 클릭 3배, settle 0.5 s×3 (`config.py:334`); low_streak 를 pan 단위로 재정의 필요 |
| (a) | **step 을 stage 단위(1 FOV)로** 정의 = (d) + "1 FOV = fw px" 규약 | `pan_step_px` → 분율 1.0 | 동일 | 동일 | 배율 무관하게 FOV 분율이라 µm 는 여전히 배율에 반비례 — 절대 µm 를 원하면 (b)/(c) 필요 |
| (c) | **PM 드롭다운으로 절대 배율 선택** 후 탐색 (`choose_step_targets` 재사용, `pm_dropdown.py:247-269`) | `RCSSEMMonitor.zoom` 에 pm_dropdown 경로 배선 | 30K→10K: 1 FOV = 13.5 µm, 1 pan(0.38) = 5.1 µm; →5K: 27 µm / 10.3 µm | 100K→50K: 2.7 / 1.0 µm; →20K: 6.75 / 2.6 µm | check-only 에서 오피스 검증된 코드; 드리프트 없음. template scale 하한 참고 |
| (b) | **먼저 R 을 덮는 배율로 zoom-out** 해 한 프레임에서 매칭 | (c) + R | Mag ≤ 135,000/(2R). 예: R=10 µm → ≤6.75K → CG 단 5K (FOV 27 µm) | R=2 µm → ≤33.75K → 30K | **R 미상(OFFICE-VERIFY)**. template 은 `WIDE_SCALES` 최소 0.15 (`live_search.py:66`, `engine.py:67`) → 배율 축소 ≤ 6.7×(30K→≥4.5K, 100K→≥15K). presence/confirm 은 `best_scale≥0.6` 요구라 찾은 뒤 다시 zoom-in 필수(Phase B 가 그 역할, 단 wheel 4 notch 상한 `live_search.py:129`) |

보조 관찰:

- (b) 의 scale 하한은 modality 에 따라 실질이 다르다. 메모리(`project_om_sem_mag_key_fills_frame`)
  대로 SEM key 가 프레임 80-100 % 면 0.15× 에서도 ~77 px(512 기준) 로 남지만, OM key 가 10-20 % 면
  0.15× 에서 8-15 px 로 사실상 매칭 불가 — OM 은 zoom-out 이 아니라 pan 으로 풀어야 한다.
- 어느 옵션이든 **원점 복귀** 가 있어야 한다: 못 찾으면 착지점으로 돌아와 엔지니어에게 넘기는 것이
  "spiral 끝에 방치" 보다 낫다. (d) 는 delta 누적만으로 역경로가 가능하고, (c) 는 baseline 배율
  복귀가 zoom ladder 에 이미 있다 (`cycle.py:1953-2000`).
- 시간 예산: 2026-08-24 결정은 "10회면 엔지니어가 개입 못 한다" 였다 (`live_search.py:123-126`).
  **2026-08-28 사용자 결정으로 이 제약은 버린다** — 긴급 해제 래치(`ffd2d50`)가 개입 경로를
  대신하므로 클릭 횟수는 더 이상 탐색 반경의 제약이 아니다. 따라서 (d) recenter chain 이 다시
  선택지에 들어오고, 모달리티별로 갈린다: **SEM = (c) 절대 배율 zoom-out**(scale 을 알게 되어
  1-scale 매칭), **OM = (d) pan**(key 가 프레임 10-20 % 라 0.15× 에서 매칭 불가). kill switch 와
  `pan_budget` 값 자체는 그대로다.

---

## 5. 출처

- `/Users/daeyoung/Codes/skewnono_v3_nuxt/docs/datatables/hitachi/mag_pixel.txt` — §1 기준 화면
  135,000 µm(OFFICE-VERIFY, 표본 1건), §2 공식/표, §3 CG/GT 배율 단, §4 OM Field_Size 값 없음.
- `poc/workflow_3/align/live_search.py` — `:13` 물리 규약, `:66` WIDE_SCALES, `:70` MIN_CONFIRM_SCALE,
  `:123-137` LiveSearchConfig, `:179-191` clamp_to_fov, `:221-222` 초기 zoom-out, `:251-255` 종료 가드,
  `:265-303` broad/confirm 분기, `:324-339` `_do_pan`, `:353-360` 종료, `:502` mock zoom_factors.
- `poc/workflow_3/align/search_pattern.py:4-21` — square spiral.
- `poc/workflow_3/align/correction.py` — `:47-56` import, `:60-65` PAUSED_SCALES, `:76-98`
  CorrectionConfig, `:160-200` key_visibility_gate, `:297-347` kill switch / fallback 위임,
  `:514-566` `correct_align_fail_auto`.
- `poc/workflow_3/sem_monitor/controller.py` — `:1-33` 좌표 계약·미캘리브레이션 목록, `:122-131`
  capture ROI, `:177-188` move_to_point(더블클릭), `:202-215` zoom(wheel), `:220-235` read_mode,
  `:296-305` panel_roi/nm_per_pixel=None.
- `poc/workflow_3/sem_monitor/pm_dropdown.py:247-269` — `choose_step_targets`.
- `poc/workflow_3/sem_monitor/sem_box_detect.py:138-162` — `parse_pm_magnification`.
- `poc/workflow_3/config.py` — `:283-310` zoom ladder(특히 `:299-300` notch 실측, `:303-309` wheel
  무효 tool), `:320-328` fallback_search_enabled/search_pan_budget, `:334-335` settle/zoom_scroll_dy,
  `:447-465` env 배선.
- `poc/workflow_3/monitor/cycle.py` — `:850` action_enabled 게이트, `:869-878` build_rcs_sem_monitor,
  `:927-1004` `_exec_run_correction`(`:1004` pan_budget 만 주입), `:1690` / `:1748-1760` zoom ladder
  는 check-only, `:2095-2135` PM 드롭다운 arms, `:2340-2345` verdict 별 out/in 순서.
- `poc/workflow_3/monitor/align_fail_monitor.py:710-733` — 실운전 기본값(SAFE_MODE=0, DRY_RUN=0).
- `poc/workflow_3/align/matching/engine.py` — `:47` DEFAULT_SCALES, `:67` BROAD_SCALES, `:168-173`
  STRUCTURE_POLICY, `:1012-1018` non-ensemble ORB.
- `poc/workflow_3/workflow_3_config.example.py` — fallback/pan 상수 없음(grep 0건).
- git: `ffd2d50` (2026-08-24, kill switch + 오피스 관찰 "반대 방향"), `70acf9b` (2026-08-24,
  pan_budget 10→5 + `fallback_config` 미전달 배선 결함 수정).
- `.claude/skills/workflow3-env-flags/SKILL.md:26` — zoom ladder / PM 드롭다운 운영 레퍼런스.
- `poc/workflow_3/align/test_fallback_kill_switch.py:1-18` — 08-24 배경 서술.

## 5. R = 30 µm 시험값으로 계산 (2026-08-28 사용자 설정, 실측 아님)

탐색 박스 = 2R = 60 µm. 한 프레임으로 덮으려면 Mag ≤ 135,000/60 = 2,250 → CG 단 **2K**
(FOV 67.5 µm). 그러나 30K 등록 SEM key(≈4 µm)는 2K 에서 ≈30 px, scale 0.135 로 매칭 하한
0.15 아래다 → **(c) 단독으로는 못 덮는다. (c) 로 하한까지 내리고 (d) 로 격자 pan** 이 필요하다.

| 등록 | 내릴 수 있는 최저 단 | FOV | key px | 60 µm 격자 | 1 FOV step 당 recenter |
|---|---|---|---|---|---|
| 30K | 5K (하한 4.5K) | 27 µm | 77 | 3×3 = 9 FOV | 2회 |
| 30K | 10K | 13.5 µm | 154 | 5×5 = 25 FOV | 2회 |
| 100K | 20K (하한 15K) | 6.75 µm | 92 | 9×9 = 81 FOV | 2회 |

- 30K 등록이면 **5K + 3×3 격자**가 현실적이다: 착지 셀 제외 8 셀 × 2 recenter = 16 클릭 +
  후보마다 Phase B. `pan_budget` 은 pan 단위라 1 FOV step 을 1 pan 으로 세면 10 으로 부족하지
  않지만(8 step), 지금 `_do_pan` 은 0.38 FOV 한 번이라 **step 정의를 바꿔야** 한다.
- **align key 는 보통 50K 이하로 등록·운영된다(사용자 확인 2026-08-28) — 100K 케이스는 없다.**
  그래서 위 100K 행은 참고용이고, 설계 상한은 50K 다.
- 50K 등록 (key ≈ 2.4 µm): scale 하한 0.15 → Mag ≥ 7.5K → **8K** (FOV 16.9 µm, key 74 px)
  → 60 µm 격자 **4×4 = 16 FOV**. 5K 까지 내리면 3×3 이지만 key 가 46 px / scale 0.10 으로
  하한 아래 — **매칭 하한을 0.15→0.10 으로 내릴 수 있으면 ≤50K 전부 5K + 3×3 로 통일된다.**
  그 검증(46 px key 에서 구조 매칭이 사는가)이 첫 실장비 테스트의 세 번째 산출물이다.
- 내릴 배율은 recipe 마다 cond.txt `Magnification` 에서 계산한다: 등록 배율 × 0.15 이상인
  가장 낮은 CG 단 (30K→5K, 50K→8K).
- R 자체가 시험값이므로 첫 실장비 테스트의 산출물 하나는 "실제 R 이 얼마였나" 다.

