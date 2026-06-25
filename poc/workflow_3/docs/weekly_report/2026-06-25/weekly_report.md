# Weekly Report — 측정 실패 abort 잡(workflow_3e) + template-bank matcher 벤치

**기간:** 2026.06.25 (직전 리포트 06.19~06.24 이후)
**대상:** ① `poc/workflow_3e/` (신규 확장 패키지 — 측정 연속 실패 시 자동 abort) · ② `poc/workflow_2` template-bank matcher 벤치
**커밋:** 약 15개, 전량 `main` 직접 반영
**작업 방식:** Mac blind 작성 → 오피스 `git pull` 실행 → digest/경고 라인 relay 반복 (fab 데이터 Mac 반입 불가)

---

## 요약

효율적 장비 운영(effective tool management)을 위해 **두 번째 MES 알람 클래스 — "측정 연속 실패"를 자동 abort** 하는 잡을 프로덕션 루프에 추가했다. align fail(ALID=9006)만 처리하던 루프가, 이제 정렬은 성공했지만 측정 포인트가 임계치(예: 100점 레시피 중 ~20점 연속 실패)를 넘겨 계속 실패하는 run 을 **감지 → 접속 → 증거 캡처 → Stop/Abort 클릭**으로 중단해 wafer 1장을 통째로 태우는 손실을 막는다.

1. **측정 실패 abort 잡 (착수·코드 shipped)** — `workflow_3` core 를 **무수정**으로 둔 채 `workflow_3e` 확장 패키지로 구현. 단일 RCS 커서를 공유하므로 **한 프로세스/한 루프에서 직렬** 처리(별도 lock 불필요). abort 클릭은 CV 보정과 동일한 **이중 게이트 + notify-only 기본값**으로 보호. 6개 테스트 파일 전량 통과.
2. **template-bank matcher 벤치 (구현 완료·오피스 평가 대기)** — 직전 리포트의 "다음 단계"였던 매처 개선 실험을 7-task TDD 계획대로 구현. **heatmap soft-voting(primary) + RRF(extra) 2-arm** + kill-test 리포트. 아직 오피스 골든셋 평가 미실행(office-gated).

> 핵심 설계 결정: **"abort 는 GUI 행동이고 큐잉 가능하다"** — 두 잡이 같은 커서를 동시에 만질 수 없다는 하드웨어 사실에서 출발해, 별도 프로세스/스레드/락 대신 기존 blocking 사이클을 재사용해 직렬화를 공짜로 얻었다. **연속 실패 카운팅은 MES 가 소유**(임계 알람 발화)하므로 감지기는 align edge-trigger 와 똑같이 stateless.

---

## 1. 측정 실패 abort 잡 — `workflow_3e` · **착수 / 코드 shipped**

### 배경 — 왜 필요한가

프로덕션 루프는 지금까지 **align fail(ALID=9006)** 한 종류만 처리했다. 두 번째 실패 클래스가 범위에 들어왔다: 레시피가 *측정 중*(정렬은 성공)인데 포인트가 계속 **측정 실패**하면(정렬 drift / 잘못된 레시피 / wafer 편차 등) 신뢰도 낮은 데이터가 쌓인다. fab 은 실패가 임계치를 넘으면 **run 을 abort** 해 wafer 전체를 낭비하지 않기를 원한다.

### 구조를 결정한 제약

시스템은 **하나의** RCS 클라이언트를 **하나의** OS 커서(pywinauto + pynput)로 구동한다. align 사이클은 connect → correct → engineer-watch → close 동안 커서를 물리적으로 독점한다. 따라서:

- **Actuation 은 직렬이어야 한다.** 두 잡이 동시에 GUI 를 만질 수 없다(코드 구조가 아니라 하드웨어 사실).
- **Detection 은 싸고 이미 통합돼 있다.** align 트리거는 MES 한 번 조회 → ALID 필터. 측정 실패 트리거는 *같은 조회, 다른 ALID*.

사용자 확인 사항:
- **Abort 는 RCS GUI 행동**(Stop/Abort 컨트롤 locate + 클릭 + 확인 다이얼로그) — align 사이클과 커서를 두고 경합한다.
- **Abort 는 큐잉 가능** — 진행 중 align 사이클이 끝날 때까지 기다려도 무방(연속 실패는 천천히 쌓이는 조건, 분 단위 지연 허용). 선점(preemption) 불필요.
- **MES 가 임계 알람을 발화** — 시스템이 측정 결과를 직접 누적하지 않는다. 감지기는 **stateless**, align edge-trigger 의 정확한 거울상.

### 결정 — 한 프로세스, 한 루프, 직렬 공유 GUI 위의 두 잡 타입

별도 OS 프로세스는 기각: 둘 다 같은 커서를 두고 싸워(어차피 프로세스 간 락 필요) 공유 dedup 상태(`active_tools`)도 잃는다. "abort 는 큐잉 가능"이므로 기존 **blocking 사이클 모델 자체가 직렬화**다 — 스레드·락·잡 큐 전부 불필요.

```
monitor_loop()                              # 단일 while 루프
├─ alarms = source.poll()                   # tick 당 MES 1회 조회
├─ filter_align_fail(alarms)   → align rows → process_fail_rows  (기존, blocking)
└─ measurement_fail_rows(..)   → abort rows → process_abort_rows (신규, blocking)
```

먼저 검출된 잡이 먼저 행동하고 다른 잡은 현재 사이클 꼬리 또는 다음 tick 을 기다린다. 신규 잡은 align 사이클의 RCS-readiness·connect·창대기·점유(select)팝업 처리·teardown 을 **그대로 재사용**한다. 진짜 새로운 자동화 표면은 **UI 컨트롤 1개**(Stop/Abort 버튼 + 확인 다이얼로그)뿐.

### 왜 `workflow_3` 가 아니라 별도 `workflow_3e` 패키지인가

새 알람 잡을 추가할 때마다 `workflow_3/config.py` 에 `ALIGN_FAIL_*` 류 플래그가 쌓이고 `cycle.py`/`align_fail_monitor.py` 에 분기가 늘어난다. 그 증식을 격리하려고 **확장 패키지**로 분리했다.

- **`workflow_3e` → `workflow_3` 단방향 import**(확장이 core 를 import, 역방향 금지 — legacy 패키지와 같은 의존 규칙). `workflow_3` 는 **편집 0건**.
- **통합 슈퍼바이저** `workflow_3e/monitor.py` 가 새 진입점. MES 를 한 번 polling 하고 align rows 는 `workflow_3` 의 `process_fail_rows`, abort rows 는 `workflow_3e` 의 `process_abort_rows` 로 분배.
- abort 사이클은 `workflow_3` step executor(`_exec_ensure_rcs_ready`/`_exec_connect_tool`/`_exec_wait_tool_window`/`_exec_capture_screen`)·`WorkflowRunner`·`CycleResult`·teardown 을 **재사용** — `_exec_abort_measurement` + 버튼 locator 만 신규.
- `align_fail_monitor.py` 는 align-only 로 standalone 동작 유지(하위 호환).

### 패키지 구조

| 모듈 | 역할 |
| --- | --- |
| `monitor.py` | 통합 슈퍼바이저: MES 1회 polling → align/abort 분배, 단일 프로세스 직렬 |
| `detector.py` | `filter_measurement_fail(rows, alid)` — raw 알람 DataFrame 위 stateless 필터(streak 은 MES 소유) |
| `meas_alarm_source.py` | 측정 실패 rows 공급: 전용 office provider 우선, 없으면 ALID 필터 폴백 |
| `dispatch.py` | `process_abort_rows`(edge-trigger + 점유 cooldown) + `append_abort_manifest`(`measurement_abort_cycles.csv`) |
| `abort_cycle.py` | `run_abort_cycle` — connect/창대기/캡처/teardown 재사용, `_exec_abort_measurement` 만 신규 |
| `abort_button.py` | Stop/Abort 버튼 + 확인 다이얼로그 VLM locator(`align/ok_button.py` 와 동형; VLM 은 영역만 식별) |
| `notify.py` | `notify_abort_outcome` — cube rich 알림(`workflow_3` office 어댑터 경유) |
| `config.py` | `Workflow3eSettings(Workflow3Settings)` + `load_workflow3e_settings()` — `MEAS_FAIL_*` 필드 추가 |

### 안전 — notify-only 로 출하

프로덕션 측정 run 을 자동 abort 하는 것은 **파괴적·외부 영향**이라 커서를 미는 것과는 차원이 다르다. CV 보정과 동일한 **이중 게이트**(`SAFE_MODE=0` **그리고** `MEAS_FAIL_ABORT_DRY_RUN=0`)로 보호하고 **기본값은 notify-only**다: 알람 감지 → 접속 → 증거 캡처 → Abort 버튼 locate → **엔지니어에게 cube 알림**, 단 검증 전까지 **클릭 안 함**. dry-run 이 VLM locate 를 포함한 전체 경로를 그대로 태우고 최종 클릭만 게이트 — 보정을 검증했던 방식 그대로.

| Env (`MEAS_FAIL_*`) | 기본값 | 의미 |
| --- | --- | --- |
| `MEAS_FAIL_ABORT_ENABLED` | `1` | abort 잡 마스터 토글(감지 + 알림) |
| `MEAS_FAIL_ALID` | `""` | 임계 알람 ALID — **오피스 확인 필요**; 비면 검출 안 함 |
| `MEAS_FAIL_ABORT_DRY_RUN` | `1` | actuation 게이트. 실제 클릭은 `SAFE_MODE=0` **그리고** 이 값 `=0` 일 때만 |
| `MEAS_FAIL_ABORT_BUTTON_SERVICE` | `ui-venus` | Stop/Abort locator VLM route_slug |

### 검증

- **6개 테스트 파일 전량 통과** — `test_detector`(4/4), `test_config`(4/4: 기본 dry-run, SAFE_MODE 강제 dry-run, 무장 조건, env override), `test_dispatch`(3/3: edge-trigger/dedup + manifest 기록), `test_abort_button`(3/3: 미검출 None, 확인 다이얼로그 스키마 공유), `test_abort_cycle`(2/2: RCS 부재 안전 종료, step 형태), `test_meas_alarm_source`(4/4: provider 우선·예외 폴백·템플릿 스키마).
- **dry-run 경로**: 검출 → 접속 → 증거 캡처 → `[DRY-RUN]` 버튼 좌표 로깅 → abort manifest 1줄 → cube-notify, **클릭 없음** 확인.

### 오피스 활성화 게이트

action 측은 완성, **detection 측 입력 1가지**만 오피스에서 필요:

1. **`filter_measurement_fail` / `get_measurement_fail_alarms` 제공** — `temp_office_meas_many_fails.py` → `office_meas_many_fails.py`(오피스 gitignore) 복사 후 함수 1개 구현(연속 N회 임계 넘긴 tool 의 표준 알람 rows 반환). 미제공 시 잡은 경고와 함께 self-disable(align 경로 무영향).
2. **`MEAS_FAIL_ALID` 확정** (provider 없이 ALID 필터로 갈 경우).
3. **Abort 버튼 calibrate** — 실장비에서 locator 가 맞는 버튼을 잡는지 dry-run 으로 캡처 검증.
4. **무장** — dry-run 검증 후에만 별도 명시 단계로 `SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0`.

---

## 2. template-bank matcher 벤치 — `workflow_2` · **구현 완료 / 오피스 평가 대기**

직전 리포트가 매처 개선을 "다음 단계"로 지목했고, 그 실험을 7-task TDD 계획대로 구현했다(bench 전용, `workflow_3` 무수정 bit-parity fork — `ensemble_lab.py` 패턴).

### 동기

rcp align-key 가 충분히 distinctive 하지 않아 프로덕션 매처가 **success 프레임에서도** 약하게(best-candidate proposer ~0.2–0.3) localize 한다. SEM proposer-recall(`gt_in_topk`) ~68% 가 알려진 병목이고, 지배적 실패는 **key 영역 *내부*의 주기적 distractor**(ROI 좁히기가 구조적으로 무용).

### 핵심 가설 — 그리고 명시적 반대가설

- **H1**: N 개의 S-crop 을 **개별(sharp)** 로 유지하고 **dense match response 를 멤버 간 합의로 융합**하면, median-consensus(distinctive 구조를 blur 로 뭉갬)보다 align point 를 더 잘 localize 한다.
- **primary fusion = soft-voting heatmap(dense), not RRF(discrete)**: 프로덕션이 이미 discrete-candidate → RRF → rerank 인데 ~0.2–0.3/68% 에 갇혀 있다. RRF-bank 는 같은 융합 family 라 천장을 물려받을 공산. heatmap 은 멤버별 top-K 에 못 든 약한 응답도 **합산**해 *아무 멤버도 랭크하지 않은* 진짜 점을 끌어올려 `gt_not_in_topk` 를 정면으로 친다(자유 파라미터도 적어 lift 조작 여지 ↓). RRF-bank 는 "discrete 융합이 dense 누적 대비 보태는 게 있나?"의 **extra arm** 으로 보존.
- **H0(반대가설, Codex 리뷰 — 먼저 배제할 것)**: 주기적 distractor 가 S 프레임 간 *일관*되면 모든 sharp 멤버가 **같은 wrong lattice 점**을 지명해 합의가 distractor 를 *강화* → bank 가 median 보다 *나쁠* 수 있다. H1 은 데이터의 성질이지 보장이 아니다 — 실험의 첫 임무는 H1 vs H0 를 싸게 판정하는 것.

### 구현 (커밋)

- **config 노브**(`9426972`) — heatmap/rrf arm + 파라미터.
- **`bank_build`**(`fe77228`) — N 개 S-crop 을 *개별* `AlignKeyTemplate` 로 빌드(median 합치기 없음), consensus 와 동일 `min_s` 게이트 + coregister 전처리.
- **`bank_match_heatmap`**(`b0a8f0c`, primary) — soft-voting dense 누적.
- **`bank_match_rrf`**(`2379945`, extra) — one-vote 공간 RRF + max-member NCC.
- **kill-test 리포트**(`51b93d6`, `8bb1672`) — lattice-period 추정 + GT-bucket `classify_winner`(near_periodic 등)로 H0(distractor 강화)를 **명시적으로 측정**, consensus 평가 드라이버에 Phase 로 통합(office-gated).
- **bench eval helpers**(`315a315`) — bootstrap CI, bucket aggregate, bank digest.
- **parity·perf 정리**(`26e9e7e`, `b7bc6d4`, `73bb04e`, `ac69876`) — bank LOO `min_s` A/B parity(`<2` 게이트), None-safe digest, baseline intersection(phantom-zero 제거), bank 1회 빌드 후 LOO 슬라이스(이중 coregister 제거), frame_dt arm 간 공유.

### 상태

bench 코드는 완성. **오피스 골든셋 평가 미실행** — heatmap-primary 가 rcp-only/median-consensus baseline 대비 `gt_in_topk`·rank-1 을 (OM/SEM 층화로) 끌어올리는지, H0 가 발화하지 않는지는 오피스 실행 digest 로 확정해야 한다. `workflow_3/align` 포팅은 **양성 + 귀속 결과가 나온 뒤** 별도 spec 으로 게이트.

---

## 진행 현황 한눈에

| 항목 | 상태 |
| --- | --- |
| 측정 실패 abort — `workflow_3e` 패키지 구현 | ✅ 코드 shipped (6 test 파일 통과) |
| 측정 실패 abort — 이중 게이트 + notify-only 기본 | ✅ 적용 |
| 측정 실패 abort — dry-run 경로 검증(클릭 없음) | ✅ 확인 |
| 측정 실패 abort — 오피스 detection 입력(provider/ALID) | ⏳ 오피스 게이트 |
| 측정 실패 abort — 실장비 버튼 calibrate + 무장 | ⏳ dry-run 검증 후 |
| template-bank matcher — 벤치 2-arm + kill-test 구현 | ✅ 완료 |
| template-bank matcher — 오피스 골든셋 평가 | 🔬 평가 대기(office-gated) |
| template-bank matcher → `workflow_3` 포팅 | ⛔ 양성 결과 게이트 |

---

## 다음 단계 / 미해결

- **abort 잡 오피스 활성화**: `office_meas_many_fails.py` 구현(연속 실패 임계 카운팅 = MES 소유) + `MEAS_FAIL_ALID` 확정 + 실장비 Abort 버튼/확인 다이얼로그 calibrate → dry-run 검증 → 무장.
- **template-bank 평가 실행**: 오피스 골든셋에서 heatmap vs RRF vs baseline digest 확보, H1/H0 판정. 양성이면 포팅 spec 착수.
- **미규명(이월)**: rcp **이미지** 약함 vs **matcher** 약함 분리 — template-bank 결과가 이 귀속을 일부 밝힐 것.
