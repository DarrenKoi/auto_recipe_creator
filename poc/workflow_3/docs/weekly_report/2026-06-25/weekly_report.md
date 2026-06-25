# Weekly Report — 재등록 리포트(Phase 1·2) · 측정 실패 abort 잡(workflow_3e) · template-bank matcher 벤치

**기간:** 2026.06.19 ~ 06.25
**대상:** ① `poc/workflow_2/golden_reregister_report_cond.py` (align-key 재등록 우선순위 리포트) · ② `poc/workflow_3e/` (신규 확장 — 측정 연속 실패 자동 abort) · ③ `poc/workflow_2` template-bank matcher 벤치
**커밋:** 약 40개, 전량 `main` 직접 반영
**작업 방식:** Mac blind 작성 → 오피스 `git pull` 실행 → digest/경고 라인 relay 반복 (fab 데이터 Mac 반입 불가)

> 용어가 낯설면 같은 폴더의 **[concepts_explained.md](concepts_explained.md)** — template bank / heatmap / RRF / baseline / digest / in_topk vs rank-1 / kill-test 설명서 — 를 먼저 읽으면 좋다.

---

## 요약

이번 주는 **"align-key가 변별력이 약하다"는 하나의 근본 병목**을 세 갈래로 공략했고, 그 과정에서 **"매처를 더 정교하게 만드는 길은 막혀 있고, 진짜 레버는 align-key 재등록"**이라는 결론에 도달했다. 별개로 **두 번째 MES 알람 클래스(측정 연속 실패) 자동 abort** 잡을 프로덕션 루프에 추가했다.

1. **재등록 리포트 Phase 1 (완료)** — success 프레임만으로 변별력 약한 align-key를 latent-risk로 랭킹 + 대체 박스 제안.
2. **재등록 리포트 Phase 2 (실험 종료/코드 보존)** — 같은 free-search를 fail(E) 프레임에 돌려 S→E 점수 collapse로 승급 시도. **오피스 실측에서 신호 신뢰 불가로 판명 → 종료.**
3. **측정 실패 abort 잡 (코드 shipped/오피스 게이트)** — `workflow_3` core 무수정, `workflow_3e` 확장 패키지로 구현. 한 프로세스/한 루프 직렬 처리. notify-only 기본값 + 이중 게이트.
4. **template-bank matcher 벤치 (결론: 기각 — ADR 0006)** — heatmap soft-voting + RRF 2-arm 구현 + 오피스 평가. `in_topk`는 이겼으나 **rank-1 ≈ 0.5(반반) → 출하 불가**. SEM은 어떤 멤버 융합으로도 못 푸는 ranking 문제로 확정.

> 핵심 교훈 (한 줄): **재등록 리포트가 "약하다"고 flag하는 recipe는 애초에 success에서도 약한 key** → "S에서 높다가 E에서 무너지는" collapse 신호 자체가 성립 안 함(Phase 2 종료). 그리고 매처를 어떻게 융합해도(heatmap/RRF/기타) **rank-1이 0.5에 막힘** → 매처가 아니라 **align-key 재등록이 유일한 레버**임을 두 갈래에서 동시에 확인.

---

## 1. 재등록 리포트 Phase 1 — S-only 스크리닝 · **완료**

success(S) 프레임만으로 align-key 변별력을 평가해 재등록 우선순위를 랭킹하는 드라이버를 완성했다.

- **3단 evidence tier + risk score + 랭킹** (`71b9fd0`, `d038fec`, `dd1e57d`) — recipe별 증거를 집계해 latent-risk 순으로 정렬.
- **C1 스크리닝 리포트 드라이버** (`84cb662`) — STRONG = GT-absent + frac floor 기준, S-only 정직한 생존 편향 스크리닝.
- **C2 대체 박스 제안 + 오버레이** (`24f86f8`, `e9b51b6`, `ca81da4`) — 더 변별력 있는 영역을 박스로 제안하고 마킹 이미지로 시각화.

### box-fidelity 버그 추적 (06-23 집중 디버그)

box-suggestion 경로에서 fidelity가 **전 recipe에서 0**으로 떨어져 제안이 0건이던 문제를 끝까지 추적해 해결.

- **근본 원인**: 매칭 엔진은 후보 `xy`를 **patch 중심**으로 반환하고 `align_offset`을 적용하지 않는다. off-center 박스(특히 OM unique-area)는 후보가 crosshair 근처에 절대 안 떨어져 fidelity가 전부 0 (`f7501fb` — `_box_offset_xy` 신설, `expected = gt_xy + offset*scale`).
- **단계적 자가-진단**: all-zero 경고를 `exc/empty/offtarget`로 분해 + call-site 태그 + top1 후보 덤프 → 오피스 한 줄 relay만으로 원인 좁힘 (`47cda7e`, `6aa23af`, `ca31403`).
- **tight scale band + tolerance widen**: scale 0.60이 작은 박스를 주기 SEM distractor에 매칭시키던 것을 `(0.85,1.0,1.15)` band로 차단 (`463426b`); GT tolerance `0.20 → 0.30`으로 참 localization(0.20~0.24)을 1~6px 차로 놓치던 것 복구, distractor(≥0.42)는 계속 기각 (`b7cb567`).
- **결과**: `w_sugg 0 → 1` (양 modality). box-fidelity 경로 정상화.

> 신규 env 노브(`REREGISTER_MAX_RECIPES` / `REREGISTER_FIDELITY_SCALES` / `REREGISTER_GT_TOL_NORM`) 전부 `golden_eval_config` 브리지.

---

## 2. 재등록 리포트 Phase 2 — E-frame confirmation · **실험 종료 / 코드 보존**

Phase 1의 latent-risk를 fail(E) 프레임 증거로 **confirmed**로 승급하려는 시도. SDD로 6태스크 전량 구현 + 태스크별 리뷰 + 최종 whole-branch opus 리뷰 완수 (`af18d7b`..`61f5222`, 48 테스트 통과).

**설계 핵심**: `TIER_WEIGHT`에 `E_CONFIRMED` 최상위 추가 + `_e_confirm(s_rep, e_rep)` 규칙 — high-S 전제(`s_rep >= S_FLOOR`) 후 `(s_rep - e_rep >= COLLAPSE_MARGIN) or (e_rep <= E_FLOOR)`. E 프레임 free-search proposer는 bit-parity로 구현. 부수 인프라(`039303b`): 전용 `EFRAME_ROOT` 분리 + 시작 시 dataset-health 사전점검(12 신규 테스트).

### 왜 멈췄나 — 오피스 실측 2회로 신호 신뢰 불가 확정

**실행 #1 (기본 임계) — confirmed 0:**
```
dataset health: 117 recipes | confirm-capable 28 | E-bearing 28 | incomplete 89
[DIGEST] om[strong 42, confirmed 0] | sem[strong 84, confirmed 0]
STRONG 샘플: 0.206->-(n_e=0)  0.308->-(n_e=0)  0.244->0.187(n_e=1)
```
실제 점수대는 **~0.2-0.3**인데 기본 임계는 ~0.5-0.6 가정 → `_e_confirm` 첫 줄에서 100% 탈락. 게다가 117 중 **28만 E-bearing**, 그 28도 per-recipe E 장수가 얇음(n_e=1).

**실행 #2 (임계 낮춰 smoke test) — 5/6이 false positive:** SEM 3건 전부 점수대가 ~0.15-0.25라 S·E 둘 다 floor 밑 → '낮은 점수'를 'collapse'로 오판.

### 결론 (사용자 합의)

**근본 원인 = rcp align-key가 충분히 distinctive하지 않다.** s_rep가 *success* 프레임에서도 낮음(0.15-0.31). key가 S에서도 약하니 score-collapse는 **무너질 높이가 없는 구조적 모순.** → **Phase 2는 이 데이터에서 신뢰할 신호가 아니므로 실험 종료.** 코드/테스트(60 tests)·`EFRAME_ROOT`·dataset-health는 보존.

---

## 3. 측정 실패 abort 잡 — `workflow_3e` · **코드 shipped / 오피스 게이트**

효율적 장비 운영을 위해 **두 번째 MES 알람 클래스 — "측정 연속 실패"를 자동 abort**하는 잡을 추가했다. align fail(ALID=9006)만 처리하던 루프가, 이제 정렬은 성공했지만 측정 포인트가 임계치(예: 100점 중 ~20점 연속 실패)를 넘겨 계속 실패하는 run을 **감지 → 접속 → 증거 캡처 → Stop/Abort 클릭**으로 중단해 wafer 1장을 통째로 태우는 손실을 막는다.

### 구조를 결정한 제약

시스템은 **하나의** RCS 클라이언트를 **하나의** OS 커서로 구동한다. 따라서 **actuation은 직렬이어야 하고**(하드웨어 사실), **detection은 싸고 이미 통합돼 있다**(같은 MES 조회, 다른 ALID). 사용자 확인: abort는 **큐잉 가능**(연속 실패는 천천히 쌓임, 분 단위 지연 허용 → 선점 불필요), **MES가 임계 알람을 발화**(감지기는 stateless, align edge-trigger의 거울상).

### 결정 — 한 프로세스, 한 루프, 직렬 공유 GUI 위의 두 잡 타입

```
monitor_loop()                              # 단일 while 루프
├─ alarms = source.poll()                   # tick 당 MES 1회 조회
├─ filter_align_fail(alarms)   → align rows → process_fail_rows  (기존, blocking)
└─ measurement_fail_rows(..)   → abort rows → process_abort_rows (신규, blocking)
```

별도 프로세스는 기각(어차피 같은 커서 경합 → 락 필요 + 공유 dedup 상태 손실). "abort는 큐잉 가능"이므로 기존 **blocking 사이클 모델 자체가 직렬화**다. 신규 잡은 align 사이클의 RCS-readiness·connect·창대기·점유팝업 처리·teardown을 **그대로 재사용** — 진짜 새 자동화 표면은 **UI 컨트롤 1개**(Stop/Abort 버튼 + 확인 다이얼로그)뿐.

### 왜 별도 `workflow_3e` 패키지인가

새 알람 잡마다 `config.py` 플래그와 `cycle.py` 분기가 쌓이는 증식을 격리. **`workflow_3e → workflow_3` 단방향 import**(역방향 금지), `workflow_3`는 **편집 0건**. 통합 슈퍼바이저 `workflow_3e/monitor.py`가 새 진입점.

| 모듈 | 역할 |
| --- | --- |
| `monitor.py` | 통합 슈퍼바이저: MES 1회 polling → align/abort 분배, 단일 프로세스 직렬 |
| `detector.py` | `filter_measurement_fail` — raw 알람 위 stateless 필터(streak은 MES 소유) |
| `meas_alarm_source.py` | 측정 실패 rows 공급: 전용 office provider 우선, 없으면 ALID 필터 폴백 |
| `dispatch.py` | `process_abort_rows`(edge-trigger + 점유 cooldown) + manifest 기록 |
| `abort_cycle.py` | `run_abort_cycle` — connect/캡처/teardown 재사용, `_exec_abort_measurement`만 신규 |
| `abort_button.py` | Stop/Abort 버튼 + 확인 다이얼로그 VLM locator(VLM은 영역만 식별) |
| `notify.py` | `notify_abort_outcome` — cube rich 알림 |
| `config.py` | `Workflow3eSettings` + `MEAS_FAIL_*` 필드 |

### 안전 — notify-only 로 출하

프로덕션 측정 run 자동 abort는 **파괴적·외부 영향**이라 CV 보정과 동일한 **이중 게이트**(`SAFE_MODE=0` **그리고** `MEAS_FAIL_ABORT_DRY_RUN=0`)로 보호하고 **기본값은 notify-only**: 감지 → 접속 → 캡처 → Abort 버튼 locate → **엔지니어 cube 알림**, 검증 전까지 **클릭 안 함**.

| Env (`MEAS_FAIL_*`) | 기본값 | 의미 |
| --- | --- | --- |
| `MEAS_FAIL_ABORT_ENABLED` | `1` | abort 잡 마스터 토글(감지 + 알림) |
| `MEAS_FAIL_ALID` | `""` | 임계 알람 ALID — **오피스 확인 필요**; 비면 검출 안 함 |
| `MEAS_FAIL_ABORT_DRY_RUN` | `1` | actuation 게이트. 실제 클릭은 `SAFE_MODE=0` **그리고** 이 값 `=0`일 때만 |
| `MEAS_FAIL_ABORT_BUTTON_SERVICE` | `ui-venus` | Stop/Abort locator VLM route_slug |

### 검증 / 오피스 활성화 게이트

- **6개 테스트 파일 전량 통과** (detector 4/4, config 4/4, dispatch 3/3, abort_button 3/3, abort_cycle 2/2, meas_alarm_source 4/4). dry-run 경로(검출→접속→캡처→`[DRY-RUN]` 좌표 로깅→manifest→notify, **클릭 없음**) 확인.
- **남은 오피스 입력**: ① `office_meas_many_fails.py` 구현(연속 실패 임계 카운팅 = MES 소유) ② `MEAS_FAIL_ALID` 확정 ③ 실장비 Abort 버튼 calibrate(dry-run 캡처 검증) ④ 무장(`SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0`).

---

## 4. template-bank matcher 벤치 — `workflow_2` · **결론: 기각 (ADR 0006)**

직전에 지목한 "매처 개선"을 7-task TDD로 구현하고 오피스 골든셋으로 평가해 **결론을 냈다.** (bench 전용, `workflow_3` 무수정 bit-parity fork.) 용어 상세는 [concepts_explained.md](concepts_explained.md) §3~§9.

### 동기와 가설

rcp align-key가 충분히 distinctive하지 않아 프로덕션 매처가 **success 프레임에서도** 약하게(best-candidate ~0.2–0.3) localize한다. 지배적 실패는 **key 영역 *내부*의 주기적 distractor**.

- **H1(주력)**: N개 S-crop을 **개별(sharp)** 유지하고 dense match response를 **heatmap soft-voting(SUM)**으로 융합하면, median-consensus(blur로 뭉갬)보다 잘 localize한다. RRF는 "discrete 융합이 추가 가치를 주나?"의 **extra arm**.
- **H0(반대가설, kill-test로 먼저 배제)**: 주기 distractor가 S들 사이 일관되면 모든 멤버가 **같은 wrong lattice**를 지명 → 합의가 distractor를 *강화* → bank가 median보다 *나쁠* 수 있다.

### 평가 결과 — 3단 판정

1. **kill-test 통과** — near_periodic이 om 0.014/sem 0.052로 낮음. distractor 강화(H0)는 **일어나지 않음**. ✅
2. **in_topk는 consensus를 이김** — 정답을 후보 안엔 잘 넣는다. ✅
3. **그러나 rank-1 ≈ 0.5** (OM/SEM 둘 다) — **1등으로 꼽는 건 동전 던지기.** ❌ RRF는 OM에서 0.902지만 이미 redundant(consensus와 동급).

### 결론

실전은 후보 8개가 아니라 **1등 좌표 하나를 클릭**하므로 **rank-1이 실제 출하 성능**(`in_topk`는 천장일 뿐). rank-1 0.5는 **출하 불가**. 게다가 heatmap·RRF·기타 3가지 융합이 **모두 같은 벽**에 막힘 → **SEM은 어떤 멤버 융합(member-fusion)으로도 풀 수 없는 ranking/distinctiveness 문제**로 확정. 매처-융합은 소진(exhausted)됐고, 레버는 **upstream 재등록(re-registration)**.

> 코드는 `TBANK_HEATMAP=0` kill switch 뒤에 보존(16/16 테스트 통과), **`workflow_3` 포팅 안 함.** **벤치 A/B는 앞으로 `in_topk`가 아니라 rank-1으로 비교**(가장 비싸게 배운 규율).

---

## 진행 현황 한눈에

| 항목 | 상태 |
| --- | --- |
| 재등록 Phase 1 — S-only 스크리닝 + box-fidelity 버그 | ✅ 완료 (`w_sugg 0→1`) |
| 재등록 Phase 2 — E-frame confirmation 구현 | ✅ 코드 shipped (60 tests) |
| 재등록 Phase 2 — 오피스 실측 검증 | ⛔ 종료 (신호 신뢰 불가) |
| 측정 실패 abort — `workflow_3e` 구현 + 이중 게이트 | ✅ 코드 shipped (6 test 파일) |
| 측정 실패 abort — 오피스 detection 입력 + 무장 | ⏳ 오피스 게이트 |
| template-bank matcher — 벤치 2-arm + kill-test | ✅ 구현·평가 완료 |
| template-bank matcher — 출하 판정 | ⛔ 기각 (rank-1 0.5, ADR 0006) |
| template-bank matcher → `workflow_3` 포팅 | ⛔ 안 함 |

---

## 다음 단계 / 미해결

- **재등록(re-registration)이 유일한 레버** — Phase 1 신호로 변별력 약한 align-key를 더 distinctive한 영역으로 재등록(Phase 3 worklist spec 별도 진행).
- **abort 잡 오피스 활성화**: `office_meas_many_fails.py` + `MEAS_FAIL_ALID` 확정 + 실장비 버튼 calibrate → dry-run 검증 → 무장.
- **규율 정착**: 벤치 A/B는 `in_topk`(천장)가 아니라 **rank-1**(출하 성능)으로 비교.
- **미규명(이월)**: rcp **이미지** 약함 vs **matcher** 약함 분리 — template-bank가 "매처 융합으로는 안 풀린다"를 보였으므로 무게추는 **이미지(재등록) 쪽**으로.
