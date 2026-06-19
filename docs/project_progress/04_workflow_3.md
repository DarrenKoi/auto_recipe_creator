# 04. workflow_3 — 실시간 Align Fail 모니터링 루프 (Production, 현재 주력)

> 목적: workflow_1(RCS GUI 자동화) + workflow_2(CV align-key 보정)의 production 경로를 하나의
> end-to-end 실시간 루프로 통합한다. **현재 주력 패키지.**

근거: `poc/workflow_3/README.md`, `poc/workflow_3/docs/`.

## 1. 루프 개요

```
알람 감지(ALID=9006) → RCS 장비 접속 → CV align fail 보정
→ 실패 시 cube rich notification → 상시 screenshot 녹화(엔지니어 수동 조작 포함)
→ tool 닫기 → 다음 장비 대기
```

특징:
- popup 직후, `run_alarm_cycle`과 **겹쳐** daemon thread로 consensus gather가 실행 — 해당 recipe의
  최근 성공 S 이미지를 `align_consensus_cache/`에 stage(보정용 consensus 재료 확보).
- office 모듈 부재 시 자동 비활성 → **기존 동작·루프 응답성 불변**(회귀 위험 0).

## 2. 4-Layer 모듈 아키텍처 (DAG)

의존 방향: `monitor → {rcs, align, sem_monitor, runner, vlm, util}`. workflow_3는 workflow_1/2를
import하지 않는다(legacy가 wf3를 import하는 방향만 허용).

```
Layer 4  monitor/        루프 본체 (오케스트레이터)
Layer 3  align/ · rcs/ · sem_monitor/ · recording_filter/   (capabilities)
Layer 2  vlm/ · runner/  (services)
Layer 1  util/           (leaf)
```

| 서브패키지 | 내용 |
|-----------|------|
| `monitor/` | 알람 폴링(`align_fail_monitor.py` 진입점), 알람별 사이클(`cycle.py`), 상시 녹화(`recording.py`), 알림(`notify.py`), 엔지니어-done 감지, consensus gather glue, 알람 소스, office adapter 로딩 |
| `rcs/` | RCS GUI 자동화 — 실행/로그인/tool 선택·종료/캡처 |
| `align/` | Align fail 보정 도메인 — 자산 해석, 보정 orchestration(`correction.py`), 라이브 탐색, consensus, cond/crop helper |
| `align/matching/` | align-key matcher 엔진 + ensemble proposer (좌표 authority) |
| `align/diagnostics/` | 오피스/개발 검증용 probe, feasibility mark, crop 비교 |
| `sem_monitor/` | SEM Monitor panel 위치 검출 + 실장비 controller adapter |
| `vlm/` | VLM 클라이언트/서비스 레지스트리/프롬프트 |
| `runner/` | WorkflowRunner — step/precondition/journal |
| `util/` | env/image/json/time + 선택적 mouse(pynput)/window(pywinauto) |

## 3. 핵심 능력

### (1) Per-alarm cycle + 보장 teardown (`cycle.py`)
- 알람당 step 시퀀스: RCS 준비 → 팝업 닫기 → tool 접속 → 창 대기 → 녹화 시작 → SEM panel ROI →
  CV 보정.
- cleanup(녹화 중지·tool 닫기·팝업 backstop)은 step이 아니라 `try/finally`로 **반드시 실행**.

### (2) Consensus 라우팅 보정
- stage된 S로 **consensus template(최근 S median)** 을 빌드해 등록 rcp 대신 라우팅
  (`align/consensus_resolve.resolve_templates`, modality별 consensus-or-rcp).
- modality당 S가 `ALIGN_FAIL_CONSENSUS_MIN_S`(기본 4) 미만/blur 미통과/캐시 부재/예외면 그 modality는
  rcp로 폴백(**회귀 위험 0**). cache cold면 1회 bounded sync 후 진행.
- `ALIGN_FAIL_CONSENSUS=0`이면 순수 rcp(기존 동작) — 롤아웃 킬스위치.

### (3) 상시 녹화 (`recording.py`)
- 변화 감지 적응 캡처: 변화 있으면 ~0.3s 간격, 없으면 5s heartbeat. delta>15 다운샘플 픽셀 수로 판정.
- RCS 원격 화면이라 **장비측 커서·엔지니어 수동 조작까지 프레임에 보존** → 후속 분석 데이터.

### (4) Engineer-done 감지 (`engineer_done_align_adjustment.py`)
- Recipe Monitor 측정 카운터(N/M)를 hybrid(VLM grounding + CV gate + OCR)로 읽어, 엔지니어가
  측정을 시작하면(분자 N>5 연속 2회) watch를 조기 종료하고 tool을 자동으로 닫는다.

### (5) Feasibility 판정 & 재등록 플래깅 (`diagnostics/feasibility_check.py`)
- 보정 가능/불가/모호(`possible`/`not_visible`/`ambiguous`)를 판정. 모호(2nd/best ratio>τ)면
  "이 recipe의 align key를 더 distinctive한 영역으로 재등록 권고"를 audit log에 남김.

### (6) Zoom ladder / PM dropdown (check-only)
- feasibility가 모호/미발견일 때 live SEM box의 배율을 mouse wheel 또는 **PM 버튼 드롭다운**으로
  바꿔가며 각 배율에서 재매칭(어느 배율에서 key가 보이는지 탐색). 일부 장비는 wheel이 배율을 안 바꿔
  PM 드롭다운이 기본.

### (7) Check-only 변형 (`align_fail_monitor_only_check.py`)
- 접속 → 1프레임 캡처 → feasibility 판정 → 닫기만 수행(실보정·녹화·watch 없음). 진단·캘리브레이션용.

## 4. 안전 장치 (Safe-mode gating)

| env | 기본 | 의미 |
|-----|------|------|
| `SAFE_MODE` | 0 | 1이면 모든 마우스/키보드 차단(전역 dry-run) |
| `ALIGN_FAIL_CORRECTION` | 1 | CV 보정 단계 수행 여부 |
| `ALIGN_FAIL_CORRECTION_DRY_RUN` | 1 | 보정 move/click 차단. 실클릭은 `SAFE_MODE=0` **그리고** 이 값=0일 때만 |
| `ALIGN_FAIL_CONSENSUS` | 1 | consensus 라우팅 마스터 토글(킬스위치) |
| `ALIGN_FAIL_ENGINEER_DONE_DETECT` | 0 | 측정-시작 감지(캘리브레이션 후 1) |

→ 실보정은 **두 단계 게이트**(`SAFE_MODE=0` + `DRY_RUN=0`)를 모두 통과해야만 작동.

## 5. 산출물 경로

- 알람 로그: `logs/align_fail_alarms.txt`
- 사이클 manifest: `logs/align_fail_cycles.csv` (알람 1건 = 1줄)
- step journal: `logs/workflow_runs/<run_id>_align_fail_cycle_<eqp>/`
- 녹화: `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/`
- consensus 캐시: `align_consensus_cache/<eqp>/<class>/<recipe>/events/<event_id>/`

## 6. 현재 상태

- ✅ 코드 완료: 루프 골격, primary 보정(reposition+OK), fallback live search, consensus 라우팅(코드),
  box-crop cond-aware 템플릿, 재등록 플래깅, check-only 진단.
- 🟡 활성화 대기: `office_success_downloader`(S 이미지 공급) 구현, 오피스 캘리브레이션(zoom/click 좌표,
  engineer-done), pilot 실보정.

상세 현황·로드맵은 [05_status_roadmap.md](05_status_roadmap.md).
