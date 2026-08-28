# Search-around 재설계 — 절대 배율 zoom-out + FOV 격자 sweep (설계)

- 날짜: 2026-08-28
- 대상:
  - `poc/workflow_3/align/live_search.py` (Phase A 교체: grid sweep + collect-then-chase)
  - `poc/workflow_3/align/search_pattern.py` (step 단위를 FOV 비율로)
  - `poc/workflow_3/sem_monitor/controller.py` (`set_magnification` / `nm_per_pixel` / odometry)
  - `poc/workflow_3/sem_monitor/pm_dropdown.py` (재사용, 변경 없음 예정)
  - `poc/workflow_3/align/matching/engine.py` (base scale 정규화 진입점)
  - `poc/workflow_3/monitor/cycle.py` (등록 배율·SEM box rect·PM 옵션을 fallback 에 주입)
  - `poc/workflow_3/config.py` (`ALIGN_FAIL_SEARCH_*` 신규 필드)
- 상태: **구현 완료 2026-08-28**(Mac 테스트 31 통과), 오피스 실장비 검증 대기. 구현 중 확정된 편차:
  - confirm 게이트는 `decision=="match" and scale>=0.6` — legacy 의 `orb>0` 는 쓰지 않는다
    (합성 box 도 실제 SEM junction key 도 ORB 특징점이 빈약해 진짜 match 가 orb=0 으로 거부됐다).
    `distinctive` 는 engine 규약(chamfer-top 의 유일성, hard gate 금지)대로 **soft advisory** 로
    `meta.confirm_distinctive`/`confirm_second_ratio` 에 기록만 한다(code-review 반영).
  - 배율 판독 실패는 세 곳에서 갈린다: zoom-out 뒤 → `degraded(mag_unreadable)`, 옵션 0개 →
    `degraded(no_mag_options)`(selector 가 PM 버튼 재클릭으로 드롭다운 닫기 시도), 추격 중 confirm
    배율 → 추격 중단 + 복귀(`reason=mag_unreadable_confirm`). 복귀 실패는 `meta.restore_failed`.
  - PM 드롭다운 클릭도 보정 이중 게이트(`SAFE_MODE=0 and CORRECTION_DRY_RUN=0`)를 지키고, abort 래치가
    삼킨 클릭(`click_at_screen`→False)은 열린/선택된 것으로 간주하지 않는다.
  - 셀 step 은 x=fw, y=fh(프레임이 정사각이 아니면 세로 FOV 가 짧다); n 은 짧은 변 FOV 로 계산.
  - 추격은 `decision!="low"` 인 상위 `max_chase`(3)개만 — 전 셀 추격은 배율 왕복 9회가 됐다.
  - `MagnificationControl.options` 는 lazy(`options_fn`): 실장비에서 옵션 읽기 = 드롭다운 열기라 선택
    직전에만 열고 그 열린 목록에서 바로 누른다.
  - settle 은 controller(`move_to_point`)와 selector(`zoom_probe_settle_sec`)가 이미 하므로 grid 는 0.
  - PM 드롭다운 actuator 를 `cycle.py:_PMDropdownSelector` 로 추출(본문 verbatim 이동) — **zoom ladder 오피스 재검증 필요**.
- 근거: `docs/study/search_around_fov_reach_260828.md`(진단), `docs/study/hitachi_mag_fov_pixel_260828.md`(물리),
  `docs/opencode/2026-08-28-search-around-zoomout-grid-debate.md`(검토 기록)

## 변경 이유

현재 fallback 은 착지점을 못 벗어난다. step 이 프레임 픽셀(220px, 0.38 FOV clamp)이고 더블클릭
recenter 는 half-FOV 상한이며 zoom-out 은 배율을 거의 안 바꾸는 휠 3 notch 라, `pan_budget` 을
10 으로 올려도 ~1 FOV(30K 에서 4.5 µm) 안이다. 사용자 시험값 R=30 µm(박스 60 µm)를 덮으려면
**FOV 를 키우고, step 을 FOV 단위로 재정의**해야 한다.

## 확인된 도메인 사실

- `FOV_µm = 135,000 / Mag` (상수 OFFICE-VERIFY, 표본 1건 0.02% 차). 더블클릭 recenter 1회 = 최대
  half-FOV, 0.12 margin clamp 를 지키면 **0.38 FOV/click → 1 FOV = 3 click**.
- align key 는 **≤50K 로 등록·운영**(사용자 2026-08-28). 100K 케이스 없음.
- SEM key 는 등록 배율에서 프레임 80-100 %, OM key 는 10-20 % → **OM 은 zoom-out 불가, pan 만**.
- PM 드롭다운 경로는 check-only 사이클에서 오피스 검증됨(`pm_dropdown.read_dropdown_options` /
  `nearest_option` / `choose_step_targets`, `cycle.py` zoom ladder). **옵션은 런타임에 읽는다** —
  등록 배율(예: 30K)이 드롭다운에 없을 수 있다. PM box OCR 판독(`_capture_rung`)이 배율 피드백.
- 매칭 scale 은 **template-px → frame-px** 비율이다(`engine._resize_template`). 프레임은 SEM box
  화면 crop 이라 폭 `fw` 는 512 가 아니다(런타임 값, `detect_sem_box` rect).
- 클릭 횟수는 제약 아님(사용자 2026-08-28, hotkey 래치가 개입 경로). **시간은 제약**.
- `cv2.phaseCorrelate` 는 `align/consensus_cv.py:59` 에 이미 쓰인다.

## 설계

### 0. 단위계 — base scale 정규화 (모든 것의 전제)

fallback 진입 시 한 번:

    base   = fw / template_w          # 등록 배율에서 key 가 프레임을 채운다는 사실에서
    tpl_n  = resize(template, base)   # 이후 모든 scale 은 순수 배율비 new_mag/reg_mag
    nm_px  = 135_000_000 / (reg_mag × fw)   # controller.nm_per_pixel (현재 None)

`reg_mag` 는 cond.txt `Magnification`(str→int). 없으면 fallback 은 **기존 경로(휠+spiral)로
degrade** 하고 `search_mode="legacy_no_reg_mag"` 를 outcome 에 남긴다.

### 1. zoom-out 단 선택 — 최소 key 픽셀 기준

    key_px(m) = fw × (m / reg_mag)          # 정규화 후 key ≈ 프레임 폭
    target    = min{ m ∈ dropdown_options : key_px(m) ≥ MIN_KEY_PX }

`MIN_KEY_PX` 기본 **60**(env `ALIGN_FAIL_SEARCH_MIN_KEY_PX`, 오피스 실측으로 확정할 상수 —
"가장 작은 key 픽셀에서 구조 매칭이 사는가" 가 첫 실장비 테스트 산출물 ③). 고정 scale 0.15 는
쓰지 않는다(fw=320 이면 5K key 가 48px 로 무너진다 — 검토 반론 2). 선택은 `nearest_option` 이
아니라 **조건을 만족하는 가장 낮은 옵션**이며, 드롭다운 선택 후 PM OCR 로 실제 배율을 읽어
`cur_mag` 로 삼는다(명령값이 아니라 판독값).

### 2. 격자 sweep — collect-then-chase

    fov_um   = 135_000 / cur_mag
    n        = ceil(2R / fov_um)  (홀수로 올림, 착지 셀 중심)      # R: ALIGN_FAIL_SEARCH_RADIUS_UM=30
    cells    = square_spiral 순서로 n×n − 1 개
    step     = 1 FOV = 3 × recenter(0.38 FOV)                    # search_pattern 이 FOV 비율로 냄

셀마다: settle → capture → **3채널 ensemble + rerank**(경량 score 아님 — 프레임이 ≤ 24 장이라
비용 감당 가능, SEM 병목은 rank-1 precision) → `(cell, score, xy)` 기록. **sweep 중에는 후보를
추격하지 않는다**(검토 반론 3: 오탐 하나가 격자 bookkeeping 을 깨뜨린다). sweep 이 끝나면
`candidate_score` 이상인 셀을 점수순으로 추격한다.

예산: `pan_budget`(10) 은 **step(=셀) 단위**로 센다. 4×4 는 15 셀이라 초과 → 격자 크기가
예산을 넘으면 시작 전에 경고하고 예산 안의 spiral 순서대로만 돈다(바깥 링 일부 생략).
`low_streak_limit` 은 sweep 에는 적용하지 않는다(sweep 은 "안 보임" 이 정상 상태).

### 3. odometry — 측정된 이동량 누적 + 게이트

click 마다: settle(`settle_sec`) → capture → `cv2.phaseCorrelate(prev, cur)` → `measured_px`.

    if |measured − commanded| > ODOM_TOL × FOV:   # 기본 0.15
        use commanded; drift_flags += 1           # 주기 구조에서 한 주기 어긋난 값이 높은 cc 로 나온다
    path.append(delta_used)

원점 복귀·후보 셀 복귀는 `path` 의 누적으로 한다. `drift_flags` 와 (measured, commanded) 쌍은
debug json 에 남긴다 — 실장비 테스트 산출물 ④ "recenter 가 명령대로 움직이는가".
**settle 없이 찍은 프레임과 상관하면 0 이 나온다**(모니터는 스캔 속도로 갱신) — 기존
`settle_sec` 을 그대로 지킨다.

### 4. 후보 추격 (Phase B 수정)

best 셀로 `path` 역행/전진 → 후보 xy 로 recenter → **드롭다운으로 `reg_mag` 에 가장 가까운
옵션** 선택 → PM OCR 판독 → confirm 매칭. confirm scale 은 `cur_mag / reg_mag`(정규화 후 순수
배율비)이고 `MIN_CONFIRM_SCALE 0.6` 은 그 값에 적용한다(20K/30K = 0.67 통과; 정규화 없이는
fw 에 따라 0.6 아래로 떨어져 "찾고도 exhausted" 가 된다 — 검토 반론 1 잔여). 실패하면 다음
후보; 모두 실패하면 §5.

### 5. 종료

- 성공: 기존 `LiveSearchOutcome` 규약(status `match`, best xy) → 상위가 reposition/OK 로 이어감.
- 실패: `path` 역행으로 **원점 복귀** → `reg_mag` 최근접 옵션 복귀 → `exhausted` → 기존 배선으로
  cube 알림(status ≠ corrected). 복귀 실패는 `restore_failed=True` 로 outcome 에 남긴다.
- abort 래치는 click 마다 확인(기존 계약 유지).

### 6. 모달리티

`read_mode()=="OM"` 이면 §1 을 건너뛰고 현재 배율에서 §2-3 만(격자 pan, step = 1 FOV). OM 은
FOV 를 모르므로(Field_Size 공란) n 은 계산하지 못한다 → `pan_budget` 만큼 spiral.

## 계약

1. **배율을 바꾸는 유일한 경로는 드롭다운 + OCR 판독**이다. 휠은 이 설계에서 쓰지 않는다.
   판독 실패 시 배율을 모르는 상태로 격자를 계산하지 않는다 → legacy 경로로 degrade.
2. **sweep 중 stage 는 격자 순서로만 움직인다.** 후보 추격은 sweep 종료 후.
3. **모든 scale/거리는 정규화된 배율비·FOV 비율**이다. 프레임 픽셀 상수(220px)는 사라진다.
4. **측정값은 게이트를 지나야 누적된다.** 게이트 실패는 명령값 폴백 + 기록이지 중단이 아니다.
5. **어떤 경로로 끝나도 원점·배율 복귀를 시도한다**(teardown 과 같은 `finally` 성격).

## 오피스 실장비 테스트 — 첫 실행의 산출물

| # | 질문 | 어디서 읽나 |
|---|---|---|
| ① | 실제 R 은 얼마인가 (30 µm 는 시험값) | 성공 시 best 셀 좌표 × FOV |
| ② | 드롭다운 옵션에 등록 배율이 있는가 / 5K·8K 프레임에서 SEM key 가 잡히는가 | `read_dropdown_options` 로그, 셀별 score |
| ③ | 매칭이 사는 최소 key 픽셀 (→ `MIN_KEY_PX`) | 5K·8K·10K 셀 점수 비교 |
| ④ | recenter 가 명령대로 움직이는가 (측정/명령 비율, drift_flags) | odometry debug json |
| ⑤ | 드롭다운 배율 변경 부작용(초점/밝기/settle 시간) | 셀 프레임 육안 + settle 실측 |
| ⑥ | 사이클 시간 (3×3 ≈ 1 분 / 4×4 ≈ 2 분 추정) | manifest |

`SAFE_MODE=1` 리허설은 드롭다운·클릭이 모두 막혀 의미가 없다 — 이 설계는 실클릭 테스트만 검증한다.

## 롤백

- `ALIGN_FAIL_SEARCH_MODE=legacy` → 기존 휠+spiral 경로 그대로(코드 삭제 없음).
- `ALIGN_FAIL_FALLBACK_SEARCH=0` → 탐색 자체 off(기존 kill switch).

## 테스트 (Mac, 실장비 없이)

- 단위계: fw/template_w/reg_mag 조합에서 `key_px`, `n`, step 클릭 수(3) 산출.
- 단 선택: 옵션 목록 + fw 로 "MIN_KEY_PX 를 만족하는 가장 낮은 단" (fw=320 이면 5K→8K 로 밀림).
- sweep 순서: mock controller 로 n×n−1 셀 방문, sweep 중 추격 0회, 예산 초과 시 바깥 링 생략.
- odometry 게이트: 측정이 한도를 넘으면 명령값 사용 + flag; 원점 복귀가 path 누적과 일치.
- confirm scale: 정규화 후 20K/30K 가 0.6 을 통과, 정규화 없이는 실패하는 회귀.
- 종료: 실패 경로에서 원점·배율 복귀 호출 보장, `exhausted` status.
- degrade: reg_mag 없음 / OCR 실패 / OM → legacy 또는 pan-only.
