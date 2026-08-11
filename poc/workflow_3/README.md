# workflow_3 — 실시간 Align Fail 모니터링 시스템

workflow_1(RCS GUI 자동화) + workflow_2(CV align-key 보정)의 production 경로를
전면 이전해 하나의 end-to-end 루프로 통합한 패키지.

```
알람 감지(ALID=9006) → RCS 장비 접속 → CV align fail 보정
→ 실패 시 cube rich notification → 상시 screenshot 녹화(엔지니어 수동 조작 포함)
→ tool 닫기 → 다음 장비 대기
```

popup 직후, `run_alarm_cycle` 과 **겹쳐** daemon thread 로 consensus gather 가 실행된다:
해당 recipe 의 최근 성공 S 이미지+cond 를 `align_consensus_cache/` 에 stage 해
consensus 빌드 재료를 확보한다(`ALIGN_FAIL_GATHER_SUCCESS` 게이트 — 기본 on; gather 는 TTL
내면 재fetch skip).
office 모듈(`office_success_downloader.py`)이 없으면 자동 비활성(루프 응답성·기존 동작 불변).

보정 단계(`run_correction`)는 stage 된 S 이미지로 **consensus template(최근 S median)** 을
빌드해 등록 rcp align key 대신 라우팅 template 으로 쓴다(`align/consensus_resolve.resolve_templates`,
modality별 consensus-or-rcp). modality당 S 가 `ALIGN_FAIL_CONSENSUS_MIN_S`(기본 4) 미만이거나
blur 가드 미통과/캐시 부재/예외면 그 modality 는 기존 rcp 로 폴백(회귀 위험 0). cache cold(최초
fail)면 1회 bounded sync(`wait_for_gather`, `ALIGN_FAIL_CONSENSUS_SYNC_TIMEOUT`) 후 진행.
`ALIGN_FAIL_CONSENSUS=0` 으로 끄면 순수 rcp(기존 동작). office downloader 가 캐시를 채워야
실제 활성 — 없으면 자동으로 rcp.
검증: bench cond A/B in_topk 0.434→0.876, rank1 0.318→0.764(min_s=3 기준; prod 기본 min_s=4 는
의도적 정책). 알고리즘은 workflow_2 bench bit-parity 포팅.

## 패키지 구조

| 서브패키지 | 내용 |
|---|---|
| `monitor/` | 루프 본체 — 알람 폴링(`align_fail_monitor.py` 진입점), 알람별 사이클(`cycle.py`), 상시 녹화(`recording.py`, 변화 감지 적응 캡처 — RCS 원격 화면이라 장비측 커서가 프레임에 찍힘), 알림(`notify.py`), 알람 소스(`alarm_source.py`), office adapter 로딩(`integration_loader.py`) |
| `rcs/` | RCS GUI 자동화 — 실행/로그인/tool 선택·종료/캡처 |
| `align/` | Align fail 보정 도메인 — 자산 해석(`assets.py`), 보정 orchestration(`correction.py`), 라이브 탐색(`live_search.py`), consensus gather, cond/crop helper |
| `align/matching/` | Align-key matcher 엔진과 ensemble proposer |
| `align/diagnostics/` | 오피스/개발 검증용 probe, feasibility mark, crop/캡처 비교 스크립트 |
| `sem_monitor/` | SEM Monitor panel 위치 검출과 실장비 controller adapter |
| `vlm/` | VLM 클라이언트/서비스 레지스트리/프롬프트 |
| `runner/` | WorkflowRunner — step/precondition/journal (`logs/workflow_runs/`) |
| `util/` | env/image/json/time + 선택적 mouse(pynput)/window(pywinauto) |

의존 방향: `monitor → {rcs, align, sem_monitor, runner, vlm, util}`. workflow_3 는
poc.workflow_1/2 를 import 하지 않는다(legacy 가 wf3 를 import 하는 방향만 허용).

## 실행

```bash
uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

개발 PC dry-run (office 모듈 없이):

```bash
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay \
ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

replay CSV 컬럼: `EQP_ID,ALID,ALARM_NAME,UTC9,RECIPE_ID,OPERATION_DESC,LOT_TYPE_CD`
(UTC9 는 로드 시 현재 시각으로 재기록되어 윈도우 필터를 통과한다).

캘리브레이션 (office Windows, 측정 중 tool 대상):

```bash
uv run python poc/workflow_3/monitor/engineer_done_align_adjustment.py
```

측정 중 tool 로 done-감지 체인(grounding/CV/OCR) 즉시 검증. `ALIGN_DONE_CALIB_EQP_ID` 로 대상 tool 지정 (기본: 열려 있는 아무 Remote Monitoring 창). debug crop 은 run 별로 `debug_images/engineer_done_calib/<tool>_<yymmdd_HHMMSS>/` 에 보존된다 (운영 cycle 은 `debug_images/engineer_done/<eqp>_<tag>/`). 참고: 캘리브레이션 tick 간격은 `poll_sec` 그대로지만, 운영 watch 에서는 내부 2s sleep 이 더해져 실제 감지 주기가 `poll_sec + OCR 지연 + 2s` 정도로 약간 길다.

기본값(2026-08-11~): done 판정은 **두 조건을 모두** 요구한다 — `ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA=6`(watch 시작 이후 측정 카운터 분자가 최소 6 회 늘어야 함) **그리고** `ALIGN_FAIL_ENGINEER_DONE_OK_STREAK=6`(Assist Window 가 현재 연속 정상(검정) 측정 6 회를 보여야 함). 재정렬 직후 카운터가 잠깐 보였다 사라지는 false-start 나, 카운터만 오르고 Assist 는 아직 안 좋은 상태로 너무 일찍 닫히지 않게 두 임계를 함께 높였다. **`ALIGN_FAIL_ENGINEER_WATCH_SEC=300`**(5분) 은 done 미감지 시의 backstop cap 이다 — 측정이 시작되면 그 전에 조기 종료한다.

- 캘리브레이션 의도는 **threshold 가 아니라 grounding/CV/OCR 체인 검증**이다. 분자가 6 까지 오르고 Assist 연속 정상이 6 을 채우길 기다리면 길어질 수 있으니, 빠른 검증에는 `ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA=2 ALIGN_FAIL_ENGINEER_DONE_OK_STREAK=2 ALIGN_DONE_CALIB_SEC=180` 처럼 임시로 낮춰 done=True 까지 확인한 뒤, 운영 `.env` 에는 기본 6/6 을 쓴다.
- 체인이 정상(ROI 로그 → tick 마다 `changed=True` + `n` 증가)인데 시간 내 done 만 안 나면 임계/시간 문제이지 grounding 실패가 아니다.

### 엔지니어 수동 조작 녹화 (알람 불필요)

엔지니어와 약속한 뒤, 이미 열려 있는 Remote Monitoring 창을 그 자리에서 녹화한다.
접속(tool 더블클릭)은 하지 않는다.

```bash
uv run python poc/workflow_3/monitor/manual_record.py
```

- 모니터링 창이 여러 개면 목록을 출력하고 종료한다. `MANUAL_RECORD_EQP_ID` 로 지정한다.
- 기본 상한은 **600초(10분)**. 프레임 4000장 / 2000MB 는 백스톱이라 정상이면 걸리지 않는다.
- Ctrl+C 로 종료. 창을 닫아도 자동 종료된다.
- 저장 경로: `align_images/<EQP>/_manual/<tag>/recording/`
- 분석은 별도 실행: `RECORDING_FILTER_INPUT_DIR=<경로> uv run python poc/workflow_3/recording_filter/filter_recording.py`
- 프레임 JPEG 품질은 기본 85 다(`MANUAL_RECORD_JPEG_QUALITY`). 알람 사이클 녹화는 종전대로 95 를 쓴다.

| env | 기본값 | 역할 |
|-----|--------|------|
| `MANUAL_RECORD_MAX_SEC` | 600 | 시간 상한(0=무제한) |
| `MANUAL_RECORD_MAX_FRAMES` | 4000 | 프레임 백스톱 |
| `MANUAL_RECORD_MAX_DISK_MB` | 2000 | 디스크 백스톱 |
| `MANUAL_RECORD_POLL_SEC` | 0.2 | 샘플링 요청 간격 |
| `MANUAL_RECORD_EQP_ID` | (빈값) | 창이 여럿일 때 지정 |
| `MANUAL_RECORD_META` | 1 | 사이드카 메타 기록 |

#### 첫 오피스 분석 실행 권장 설정 (VLM 호출 상한)

수동 세션 분석은 Stage 2a(커서 VLM)의 호출 예산이 기본 무제한(`RECORDING_FILTER_MAX_VLM_CALLS=0`)
이다. 엔지니어 커서가 라이브 SEM 박스 위에 머무는 동안은 프레임마다 승격되므로, 10분
세션이면 프록시 호출이 500~2000 회 / 30~120분까지 늘어날 수 있다. 기본값은 그대로 둔다
(줄이면 기존 알람 사이클 분석이 조용히 잘린다). 대신 **첫 실행만** 상한을 걸고 감을 잡는다:

```bash
RECORDING_FILTER_INPUT_DIR=<녹화 경로> RECORDING_FILTER_MAX_VLM_CALLS=300 \
  uv run python poc/workflow_3/recording_filter/filter_recording.py
```

상한에 걸려 잘린 분량은 숨기지 않는다 - `summary.json` 의 `truncated` / `skipped_due_to_cap`
에 그대로 남고, 실제 호출량은 `vlm_calls_stage1_5_region_map` / `vlm_calls_stage2a_cursor` /
`vlm_calls_stage2c_label_estimate` / `vlm_calls_total_estimate` 로 스테이지별로 확인한다.
게이트/가림이 이벤트를 전부 걷어내면 상태가 `all_events_discarded` 로 끝나며(exit 1),
`ambient_events_dropped` / `occluded_events_excluded` 로 원인을 가른다.

## 주요 env (기존 이름 유지)

| env | 기본 | 의미 |
|---|---|---|
| `SAFE_MODE` | 0 | 1 이면 모든 마우스/키보드 차단 (전역 dry-run) |
| `ALIGN_FAIL_CORRECTION` | 1 | CV 보정 단계 수행 여부 |
| `ALIGN_FAIL_CORRECTION_DRY_RUN` | 1 | 보정의 move/click 차단. 실 클릭은 SAFE_MODE=0 **그리고** 이 값=0 일 때만 |
| `ALIGN_FAIL_RCS_RECOVERY` | 0 | RCS 메인 창 부재 시 재실행+재로그인 복구 |
| `ALIGN_FAIL_BLOCK_INPUT` | 0 | 자동 GUI 구간 동안 사용자 물리 마우스/키보드 차단(Win32 BlockInput). 사용자가 다른 앱을 쓰면 foreground lock 으로 RCS 가 안 떠서 방해되는 문제 대응. SAFE_MODE=0 일 때만 적용, engineer watch 구간은 제외, Ctrl+Alt+Del 로 항상 해제. 합성 클릭(자동화)은 차단 중에도 통과 |
| `ALIGN_FAIL_POLL_SEC` / `ALIGN_FAIL_WINDOW_SEC` | 10 / 60 | 폴링 주기 / 감지 look-back |
| `ALIGN_FAIL_RECORDING_POLL_SEC` | 0.3 | 녹화 샘플링 간격 (변화 감지용 빠른 폴링) |
| `ALIGN_FAIL_RECORDING_HEARTBEAT_SEC` | 5.0 | 변화 없어도 이 간격마다 1장 저장 |
| `ALIGN_FAIL_RECORDING_CHANGE_MIN_PX` | 4 | 변화 판정: delta>15 인 다운샘플 픽셀 최소 개수 (커서 이동도 감지) |
| `ALIGN_FAIL_RECORDING_MAX_SEC` | 900 | 녹화 상한 |
| `ALIGN_FAIL_ENGINEER_WATCH_SEC` | 300 | 미보정 시 엔지니어 조작 녹화 대기 상한(5분) |
| `ALIGN_FAIL_ENGINEER_DONE_DETECT` | 0 | 측정 카운터 delta + Assist Window 연속 정상(streak) 두 조건을 모두 충족하면 engineer watch 조기 종료 + cycle teardown 의 tool 창 자동 닫기. 캘리브레이션(`monitor/engineer_done_align_adjustment.py` 단독 실행, 측정 중 tool 대상) 검증 후 `1`. |
| `ALIGN_FAIL_ENGINEER_DONE_POLL_SEC` | 8.0 | watch 안 감지기 호출 간격 |
| `ALIGN_FAIL_ENGINEER_DONE_MIN_DELTA` | 6 | done 조건 1/2 - watch 시작 이후 측정 카운터 분자가 이만큼 늘어야 함 |
| `ALIGN_FAIL_ENGINEER_DONE_OK_STREAK` | 6 | done 조건 2/2 - Assist Window 연속 정상(검정) 측정이 이만큼 있어야 함 (둘 다 충족 시 watch 종료+tool 닫기) |
| `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE` | `ui-venus` | 분자 위치 grounding 서비스 (route_slug, 모델명 아님) |
| `ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE` | `paddleocr-vl-1.5` | 분자 OCR 서비스 |
| `ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX` | 4 | CV gate 변화 픽셀 임계(다운샘플) - 감지가 둔하면 낮추고 과민하면 올린다 |
| `ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS` | 3 | 변화 후 OCR 연속 미검출 N회 시 ROI 재grounding(패널 드래그 대비) |
| `ALIGN_FAIL_ENGINEER_DONE_REGROUND_SEC` | 30.0 | grounding 거부 후 재시도 간격. 재정렬 중에는 카운터(N/M)가 빈칸이라 거부가 정상 - 측정 시작 시 재시도가 성공한다 |
| `ALIGN_DONE_CALIB_EQP_ID` | (빈값) | 캘리브레이션 대상 tool (빈값=열려있는 아무 Remote Monitoring 창) |
| `ALIGN_DONE_CALIB_SEC` | 120 | 캘리브레이션 최대 실행 시간 |
| `ALIGN_FAIL_ALARM_SOURCE` | office | `office` \| `replay` |
| `ALIGN_SEM_MODE_OVERRIDE` | (없음) | read_mode v0 강제값 (`OM`/`SEM`) |
| `ALIGN_FAIL_GATHER_RCP_MSR` | 1 | rcp/msr 입력 이미지 office 다운로드 활성(사이클 직전 동기). downloader 부재 시 자동 skip(MES 직접 적재 전제) |
| `ALIGN_FAIL_GATHER_SUCCESS` | 1 | consensus gather 활성(최근 S 이미지 stage) — 0 으로 끄면 gather 전체 skip |
| `ALIGN_FAIL_GATHER_MAX_EVENTS` | 8 | 한 알람당 stage 할 최근 성공 event 수 (이미지 수 아님; OM/SEM split 후도 min_s 확보) |
| `ALIGN_FAIL_CONSENSUS` | 1 | consensus 라우팅 마스터 토글 — 0 이면 보정이 순수 rcp(기존 동작). 롤아웃 킬스위치 |
| `ALIGN_FAIL_CONSENSUS_MIN_S` | 4 | modality당 consensus 빌드·신뢰 최소 S 장수(floor 3). 미만이면 그 modality rcp 폴백 |
| `ALIGN_FAIL_CONSENSUS_SYNC_TIMEOUT` | 8.0 | cache cold 시 1회 bounded gather 대기(초). 초과 시 rcp, 백그라운드가 다음 데움 |
| `ALIGN_FAIL_CONSENSUS_REFRESH_TTL` | 21600 | gather 재fetch TTL(초, 6h). 이내면 FTP skip(캐시 재사용) |
| `ALIGN_FAIL_FEASIBILITY_MARK` | 1 | 점검 모니터(`align_fail_monitor_only_check`) 전용. 캡처 후 rcp 엔진으로 보정 가능/불가/모호를 판정해 캡처 옆 `<tag>_rcs_marked.jpg` + `<tag>_feasibility.json` 생성, consensus cache S event 수도 표기. production 보정 루프엔 영향 없음 |
| `ALIGN_CONSENSUS_CACHE_DIR` | `poc/workflow_3/align_consensus_cache` | staged S 이미지 캐시 루트 override |
| `WORKFLOW3_FILE_LOG_DETAIL` | 0 | 1 이면 `logs/*.log` 에 info 이벤트/VLM 성공 호출까지 기록. 기본은 warning/error 만 파일 기록 |
| `WORKFLOW3_LOG_LEVEL` | INFO | 파일 로거 레벨. 구버전 `WORK2_LOG_LEVEL` 도 fallback 으로 읽음 |

## 산출물 경로

- 파일 로그: 기본은 warning/error 만 기록 (`WORKFLOW3_FILE_LOG_DETAIL=1` 일 때 상세 info 기록)
- 알람 로그: `poc/workflow_3/logs/align_fail_alarms.txt`
- 사이클 manifest: `poc/workflow_3/logs/align_fail_cycles.csv` (알람 1건 = 1줄: run_status/failed_step/outcome/녹화 경로)
- step journal: `poc/workflow_3/logs/workflow_runs/<run_id>_align_fail_cycle_<eqp>/`
- 녹화 프레임: `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<utc9_tag>/recording/`
  (RECIPE_ID 없으면 `align_images/<eqp>/_unregistered/<tag>/recording/`)
- `ALIGN_IMAGES_DIR` 물리 경로는 **현재 `poc/workflow_1/align_images`** (오피스 MES 도구가
  직접 타겟). → `poc/workflow_3/align_images` 로 이전 예정: 아래 **align_images 루트 이전
  체크리스트** 참조. (옮기는 동안은 env `ALIGN_IMAGES_DIR` 한 줄로 검증/전환.)
- consensus gather 캐시: `align_consensus_cache/<eqp>/<class>/<recipe>/events/<event_id>/`
  — fail 알람 시 최근 성공 S 이미지+cond 를 stage(consensus 빌드 재료). replace-if-non-empty:
  새 이미지 ≥1 장이면 기존 set 을 원자적(.events_new/.events_old)으로 교체(읽는 중에도 events/
  유효), 0장/실패면 기존 보존. TTL(`ALIGN_FAIL_CONSENSUS_REFRESH_TTL`) 이내면 재fetch skip.

## align_images 루트 이전 체크리스트 (workflow_1 → workflow_3)

production(workflow_3)이 legacy 폴더에 쓰지 않게, MES 산출물 루트를
`poc/workflow_1/align_images` → `poc/workflow_3/align_images` 로 이전한다. recipe 별 입력
(from_rcp/from_msr + cond.txt)과 우리 산출물(captured_img_from_rcs/녹화)이 한 트리에 모인다.
cond.txt 는 localization·consensus eval 에서 white box/crosshair 제거용으로 읽기만 한다
(workflow_3 는 cond.txt 를 쓰지 않음 — 장비 다운로더가 이미지 옆에 함께 받아온다).

> **순서 중요** — MES 출력 경로를 먼저 안 옮기고 코드 default 만 바꾸면, align 은 새(빈)
> 트리를 읽고 MES 는 옛 트리에 계속 써서 매칭이 통째로 빈다.

1. **gitignore 가드 (이전 전 필수, 완료)** — `poc/workflow_3/align_images/` 가 .gitignore 에
   있는지 확인. fab 이미지 + cond.txt(`.txt` 는 전역 `*.jpg` 무시에 안 걸림)가 절대 커밋되지 않게.
2. **office MES 출력 경로 재설정** — from_rcp/from_msr **및 cond.txt 다운로더**가
   `poc\workflow_3\align_images\<eqp>\<class>\<recipe>\...` 에 쓰도록 office 도구 설정 변경.
   (office_success_downloader 는 `ALIGN_IMAGES_DIR` 로 읽으므로 env/default 만 맞으면 자동 추종.)
3. **기존 데이터 이동** — 숨김 사이드카(`.<image>/cond.txt`)까지 통째로:
   ```
   robocopy poc\workflow_1\align_images poc\workflow_3\align_images /E /MOVE
   ```
   (`/E` 중첩·숨김 폴더 포함, `/MOVE` 이동 후 원본 삭제. `move` 는 중첩 숨김폴더를 빠뜨릴 수 있어 robocopy 권장.)
4. **env 로 먼저 검증** (default 안 건드림):
   ```
   set ALIGN_IMAGES_DIR=<repo>\poc\workflow_3\align_images
   uv run python -c "from poc.workflow_3 import ALIGN_IMAGES_DIR as d; print(d, 'exists=', d.exists()); print(list(d.glob('*/*/*'))[:3])"
   ```
   새 트리에서 eqp/class/recipe 가 보이는지 확인.
5. **녹화/캡처 경로 확인** — `ALIGN_IMAGES_DIR` 유지한 채 SAFE_MODE=1 dry-run 알람 1회 →
   `captured_img_from_rcs/<tag>/recording/` 이 새 루트 아래 생기는지.
6. **default 상수 교체** — 검증 끝나면 `poc/workflow_3/__init__.py` 의 `ALIGN_IMAGES_DIR`
   default 를 `WORKFLOW_3_DIR.parent / "workflow_1" / "align_images"` →
   `WORKFLOW_3_DIR / "align_images"` 로. 이후 env 불필요. 본 README 산출물 경로 +
   CLAUDE.md 의 "물리 경로 workflow_1 그대로" 문구도 함께 갱신.
7. **legacy 정리** — `poc\workflow_1\align_images` 잔여 빈 디렉터리 제거.

## 오피스 PC 이전 체크리스트 (단계별 활성화)

1. **office_* 복사** — git pull 후 정위치(`poc/workflow_3/monitor/`)로 둔다:
   ```
   copy poc\workflow_1\office_align_fail_alarm.py poc\workflow_3\monitor\
   copy poc\workflow_1\office_rich_notify.py      poc\workflow_3\monitor\
   ```
   (workflow_3 는 정위치에서만 로드한다 — legacy 위치 fallback 은 제거됨. 정위치에
   없으면 해당 integration 은 비활성으로 떨어지고 경고만 남긴다.)

   **`office_success_downloader.py`** — `poc/workflow_3/monitor/` 에 신규 작성(사용자 담당,
   gitignore). `make_success_downloader()` 팩토리를 노출해 `SuccessDownloader` Protocol 을
   구현한다. 스켈레톤은 plan 문서
   (`poc/workflow_2/docs/superpowers/plans/2026-06-10-consensus-gather-in-loop.md` Task 6) 참조.
   검증: 알람 1회 후 콘솔에 `consensus gather: ... reason=ok` 로그 + `align_consensus_cache/`
   하위 events/ 디렉터리 생성 확인.
   권장: `office_rich_notify.send_cube_align_fail_info` 에 optional `summary: str = ""`
   파라미터를 추가하면 cube 메시지에 보정 결과 요약(status/best_xy/녹화 경로)이 실린다.

   **`office_rcp_msr_downloader.py`** (선택) — MES 가 align_img_from_rcp/msr 를
   `ALIGN_IMAGES_DIR` 트리에 **직접 적재하지 못하는** 환경에서만 작성한다(사용자 담당,
   gitignore). `make_rcp_msr_downloader()` 팩토리를 노출해 `RcpMsrDownloader` Protocol
   (`monitor/rcp_msr_gather.py`)을 구현한다: `download_rcp_msr(eqp_id, recipe_id, *, dest_dir)`
   가 `dest_dir`(=`ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe>`) 아래에 office MES 와 동일한
   레이아웃(align_img_from_rcp/IMAP0001·IMAP0002, align_img_from_msr/S*·E*)으로 쓰고 이미지
   수를 반환. 알람마다 **사이클 직전에 동기** 호출되어 feasibility/보정이 빈 트리를 읽는
   레이스를 막는다. 모듈/팩토리 부재 시 자동 skip(MES 직접 적재 전제). `ALIGN_FAIL_GATHER_RCP_MSR=0`
   으로 비활성. 주의: 이 다운로드를 `office_rich_notify`(cube 알림) 안에 넣지 말 것 — 점검
   모니터는 알림을 호출하지 않으므로 다운로드가 누락된다.
2. **import sweep** — `uv run python -c "import poc.workflow_3.monitor.align_fail_monitor"`
   (경고 없이 로드되는지).
3. **SAFE_MODE=1 실알람 dry-run** — 클릭 0회, journal/알림/manifest 만 확인.
4. **record-only 패리티** — `ALIGN_FAIL_CORRECTION=0` + SAFE_MODE=0:
   기존 align_fail_alarm_record 동작(접속→캡처→닫기) + 상시 녹화 재현 확인.
5. **보정 dry-run** — `ALIGN_FAIL_CORRECTION=1` (DRY_RUN 은 기본 1 유지):
   CV 가 좌표 계산·overlay 저장·cube 알림까지, 클릭은 dry-run 로그만.
6. **캘리브레이션** — tool model 별로:
   - SEM panel landmark crop + meta.json → `poc/workflow_3/templates/sem_panel_landmarks/<model_id>/`
   - 유휴 장비에서 더블클릭 recenter 이동량, wheel 1단계↔배율(`ALIGN_SEM_ZOOM_SCROLL_DY`) 측정
   - read_mode 실제 판독(모드 라벨 OCR/픽셀 휴리스틱) 구현
7. **pilot actuation** — 단일 장비에서 `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.

## Legacy

- `poc/workflow_1/` — CCTV/DVR 경로와 초기 실험 스크립트만 잔류 (동결).
- `poc/workflow_2/` — 평가/AB/튜닝 하니스만 잔류. CV 엔진은 `poc.workflow_3.align` 에서 import.
