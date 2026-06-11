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
consensus 빌드 재료를 확보한다(`ALIGN_FAIL_GATHER_SUCCESS` 게이트 — 기본 on).
office 모듈(`office_success_downloader.py`)이 없으면 자동 비활성(루프 응답성·기존 동작 불변).

## 패키지 구조

| 서브패키지 | 내용 |
|---|---|
| `monitor/` | 루프 본체 — 알람 폴링(`align_fail_monitor.py` 진입점), 알람별 사이클(`cycle.py`), 상시 녹화(`recording.py`, 변화 감지 적응 캡처 — RCS 원격 화면이라 장비측 커서가 프레임에 찍힘), 알림(`notify.py`), 실장비 adapter(`sem_controller.py`), 알람 소스(`alarm_source.py`) |
| `rcs/` | RCS GUI 자동화 — 실행/로그인/tool 선택·종료/캡처 |
| `vision/` | CV 엔진 — 매칭(ensemble)/자산 해석/보정(`align_fail_correct`)/라이브 탐색 |
| `vlm/` | VLM 클라이언트/서비스 레지스트리/프롬프트 |
| `runner/` | WorkflowRunner — step/precondition/journal (`logs/workflow_runs/`) |
| `util/` | env/image/json/time + 선택적 mouse(pynput)/window(pywinauto) |

의존 방향: `monitor → {rcs, vision, runner, vlm, util}`. workflow_3 는
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

## 주요 env (기존 이름 유지)

| env | 기본 | 의미 |
|---|---|---|
| `SAFE_MODE` | 0 | 1 이면 모든 마우스/키보드 차단 (전역 dry-run) |
| `ALIGN_FAIL_CORRECTION` | 1 | CV 보정 단계 수행 여부 |
| `ALIGN_FAIL_CORRECTION_DRY_RUN` | 1 | 보정의 move/click 차단. 실 클릭은 SAFE_MODE=0 **그리고** 이 값=0 일 때만 |
| `ALIGN_FAIL_RCS_RECOVERY` | 0 | RCS 메인 창 부재 시 재실행+재로그인 복구 |
| `ALIGN_FAIL_POLL_SEC` / `ALIGN_FAIL_WINDOW_SEC` | 10 / 60 | 폴링 주기 / 감지 look-back |
| `ALIGN_FAIL_RECORDING_POLL_SEC` | 0.3 | 녹화 샘플링 간격 (변화 감지용 빠른 폴링) |
| `ALIGN_FAIL_RECORDING_HEARTBEAT_SEC` | 5.0 | 변화 없어도 이 간격마다 1장 저장 |
| `ALIGN_FAIL_RECORDING_CHANGE_MIN_PX` | 4 | 변화 판정: delta>15 인 다운샘플 픽셀 최소 개수 (커서 이동도 감지) |
| `ALIGN_FAIL_RECORDING_MAX_SEC` | 900 | 녹화 상한 |
| `ALIGN_FAIL_ENGINEER_WATCH_SEC` | 300 | 미보정 시 엔지니어 조작 녹화 대기 상한(5분) |
| `ALIGN_FAIL_ENGINEER_DONE_DETECT` | 0 | 측정-시작(Recipe Monitor 분자) 감지로 engineer watch 조기 종료 + cycle teardown 의 tool 창 자동 닫기. 캘리브레이션(`monitor/engineer_done_align_adjustment.py` 단독 실행, 측정 중 tool 대상) 검증 후 `1`. |
| `ALIGN_FAIL_ENGINEER_DONE_POLL_SEC` | 8.0 | watch 안 감지기 호출 간격 |
| `ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT` | 6 | done(=watch 종료+tool 닫기) 최소 분자값, N>5 연속 2회 확인 |
| `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE` | `ui-venus` | 분자 위치 grounding 서비스 (route_slug, 모델명 아님) |
| `ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE` | `paddleocr-vl-1.5` | 분자 OCR 서비스 |
| `ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX` | 4 | CV gate 변화 픽셀 임계(다운샘플) - 감지가 둔하면 낮추고 과민하면 올린다 |
| `ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS` | 3 | 변화 후 OCR 연속 미검출 N회 시 ROI 재grounding(패널 드래그 대비) |
| `ALIGN_FAIL_ENGINEER_DONE_REGROUND_SEC` | 30.0 | grounding 거부 후 재시도 간격. 재정렬 중에는 카운터(N/M)가 빈칸이라 거부가 정상 - 측정 시작 시 재시도가 성공한다 |
| `ALIGN_DONE_CALIB_EQP_ID` | (빈값) | 캘리브레이션 대상 tool (빈값=열려있는 아무 Remote Monitoring 창) |
| `ALIGN_DONE_CALIB_SEC` | 120 | 캘리브레이션 최대 실행 시간 |
| `ALIGN_FAIL_ALARM_SOURCE` | office | `office` \| `replay` |
| `ALIGN_SEM_MODE_OVERRIDE` | (없음) | read_mode v0 강제값 (`OM`/`SEM`) |
| `ALIGN_FAIL_GATHER_SUCCESS` | 1 | consensus gather 활성(최근 S 이미지 stage) — 0 으로 끄면 gather 전체 skip |
| `ALIGN_FAIL_GATHER_MAX_EVENTS` | 5 | 한 알람당 stage 할 최근 성공 event 수 (이미지 수 아님) |
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
  새 event ≥1 건이면 기존 set 교체, 0건/실패면 기존 보존.

## align_images 루트 이전 체크리스트 (workflow_1 → workflow_3)

production(workflow_3)이 legacy 폴더에 쓰지 않게, MES 산출물 루트를
`poc/workflow_1/align_images` → `poc/workflow_3/align_images` 로 이전한다. recipe 별 입력
(from_rcp/from_msr + cond.txt)과 우리 산출물(captured_img_from_rcs/녹화)이 한 트리에 모인다.
cond.txt 는 localization·consensus eval 에서 white box/crosshair 제거용으로 읽기만 한다
(workflow_3 는 cond.txt 를 쓰지 않음 — 장비 다운로더가 이미지 옆에 함께 받아온다).

> **순서 중요** — MES 출력 경로를 먼저 안 옮기고 코드 default 만 바꾸면, vision 은 새(빈)
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

1. **office_* 복사** — git pull 후:
   ```
   copy poc\workflow_1\office_align_fail_alarm.py poc\workflow_3\monitor\
   copy poc\workflow_1\office_rich_notify.py      poc\workflow_3\monitor\
   ```
   (원본은 legacy 스크립트용으로 유지. 복사 전에도 legacy 위치 fallback 으로 동작하지만 경고가 뜬다.)

   **`office_success_downloader.py`** — `poc/workflow_3/monitor/` 에 신규 작성(사용자 담당,
   gitignore). `make_success_downloader()` 팩토리를 노출해 `SuccessDownloader` Protocol 을
   구현한다. 스켈레톤은 plan 문서
   (`poc/workflow_2/docs/superpowers/plans/2026-06-10-consensus-gather-in-loop.md` Task 6) 참조.
   검증: 알람 1회 후 콘솔에 `consensus gather: ... reason=ok` 로그 + `align_consensus_cache/`
   하위 events/ 디렉터리 생성 확인.
   권장: `office_rich_notify.send_cube_align_fail_info` 에 optional `summary: str = ""`
   파라미터를 추가하면 cube 메시지에 보정 결과 요약(status/best_xy/녹화 경로)이 실린다.
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
- `poc/workflow_2/` — 평가/AB/튜닝 하니스만 잔류. CV 엔진은 `poc.workflow_3.vision` 에서 import.
