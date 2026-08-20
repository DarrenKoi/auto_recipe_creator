# 사이클 1건 = 폴더 1개 (per-take run directory, 설계)

- 날짜: 2026-08-20
- 대상:
  - `poc/workflow_3/run_dirs.py` (신규 - 유일한 경로 구성점)
  - `poc/workflow_3/monitor/cycle.py`
  - `poc/workflow_3/monitor/align_fail_monitor.py`, `align_fail_monitor_only_check.py`
  - `poc/workflow_3/runner/workflow_runner.py`
  - `poc/workflow_3/logger.py`
  - `poc/workflow_3/rcs/workflow_select_tool.py`, `rcs/view_list_tab_rcs.py`
  - `poc/workflow_3/recording_filter/filter_recording.py`
  - `poc/workflow_3/monitor/make_demo_video.py`
  - `poc/workflow_3/workflow_extract/extract_workflow.py`
  - `poc/workflow_3e/dispatch.py`, `poc/workflow_3e/abort_cycle.py`
  - `poc/workflow_3/config.py`, 저장소 `.gitignore`
- 상태: 설계 확정(구현 전). oc-discuss(glm-5.3, 2R) 검토 반영 - `docs/opencode/2026-08-20-per-take-run-directory-debate.md`
- 선행 설계: `2026-08-18-occupied-share-request-recording-design.md`,
  `2026-08-10-manual-recording-session-design.md`

## 변경 이유

알람 1건("한 테이크")이 남기는 증거가 **네 개 루트에 흩어져 있고, 이름 규칙이 세 가지**다.
사이클이 실패했을 때 엔지니어는 폴더를 옮겨 다니며 같은 사건의 조각을 손으로 맞춰야 한다.

| 산출물 | 현재 위치 | 키 |
|---|---|---|
| 본 녹화 (+prelude) | `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/` | recipe 경로 |
| 첫 화면 캡처 (check-only) | `align_images/.../captured_img_from_rcs/<tag>/<tag>_rcs.jpg` | recipe 경로 |
| 보정 산출물 | `debug_images/align_fail_cycle/<tag>/` | tag |
| 공유/접근 요청 crop | `debug_images/{share_request,access_request}/<tag>/` | tag |
| 점유자 crop | `debug_images/row_occupant/<model-slug>/` | **모델 slug** |
| 로그인/tool 선택 crop | `debug_images/<model-slug>/<ts>_*.jpg` | **모델 slug + 자체 ts** |
| engineer_done crop | `debug_images/engineer_done/<eqp>_<tag>/` | eqp + tag |
| step 저널 | `logs/workflow_runs/<자체ts>_align_fail_cycle_<eqp>/` | **자체 타임스탬프** |
| 콘솔 | `logs/work2.log` (모든 사이클이 뒤섞임) | 없음 |

## 확인된 사실 (2026-08-20 코드 실측)

- `runner/workflow_runner.py:169-170` 이 **자기 `run_id` 를 새로 찍어** step 저널 폴더를 만든다.
  사이클 tag 와 어긋나므로 **이름으로 join 이 불가능**하다.
- `align_images/` 는 오피스 MES 와의 입력 계약 루트인데 루프가 같은 트리에 녹화/캡처를 쓴다.
  README.md:342 기준 오피스 `ALIGN_IMAGES_DIR` 은 아직 `poc/workflow_1/align_images` 를
  가리킬 수 있다. 즉 **우리 증거물이 env 한 줄로 통째로 옮겨 다니는 남의 루트에** 쌓인다.
- `debug_images/` 하위는 사이클 tag 기반과 모델 slug 기반 두 키를 섞어 쓴다. 후자는
  사이클 경계를 표현하지 못한다.
- 녹화 소비자 3곳이 `align_images` 레이아웃에 묶여 있다: `filter_recording.py:63-68`(고정 glob
  3종), `make_demo_video.find_recording_dirs`(`rglob`, 루트만 바꾸면 됨),
  `workflow_extract/extract_workflow.py:220`(경로 모양으로 EQP/`_manual` 판별).
- `log_work2_event` 호출부는 **42곳이고 `tag=` 를 넘기는 곳은 0곳**이다
  (`cycle.py` 9, `align/correction.py` 8, `rcs/*` 12 …). 시그니처가 `**fields` 라 필드 추가
  자체는 자유롭다(`logger.py:137`).

### tag 는 테이크 단위가 아니다 (설계 전제의 오류, oc-discuss 지적)

`align_fail_monitor.py:445` 는 `tag=_alarm_time_to_tag(info["utc9"])` 를 넘긴다. tag 는
캡처 시각이 아니라 **알람 이벤트의 UTC9** 에서 나온다 (`:130-141`, 의도가 docstring 에 명시).
실패 사이클은 `_defer_retry` 로 cooldown 에 들어가고, 알람이 해제되지 않았으므로 만료 후
**같은 MES row 를 다시 읽는다 → 같은 UTC9 → 같은 tag**. 즉 tag 는 알람 1건당 유일하지,
테이크 1건당 유일하지 않다.

프레임 파일명은 `<tag>_rcs_<seq:04d>_<elapsed_ms:08d>ms.jpg` (`recording.py:230`) 이고 seq 는
세션마다 0에서 다시 시작하며 elapsed 는 세션 시작 기준이다. 따라서 재시도의 피해는
덮어쓰기보다 **조용한 뒤섞임**이 지배적이다:

- 하드 덮어쓰기: `0000_00000000ms` 프레임, 그리고 `summary.json` / `step_*.json` (고정 이름).
- 뒤섞임: 나머지 프레임이 한 폴더에 섞여 **하나의 연속 타임라인처럼 보인다**. 이것이
  `make_demo_video`(elapsed_ms 로 시간축 복원)와 `recording_filter` Stage 1(인접 프레임
  변화 검출)을 조용히 오염시킨다 - 두 세션 경계를 프레임 변화로 읽는다.

**이 결함은 오늘 이미 존재한다** (`captured_img_from_rcs/<tag>/recording/`). 본 설계가
만드는 것이 아니라 물려받는 것이지만, "폴더 하나 = 테이크 하나" 를 표방하는 이상 여기서
고쳐야 한다. `workflow_3e/dispatch.py:103` 도 같은 `_alarm_time_to_tag` 를 쓰므로 같은 결함을
공유한다.

## 문제의 핵심

1. **join key 부재** - 같은 사건의 폴더 이름이 서로 다르다 (tag / 자체ts / model-slug).
2. **테이크 식별자 부재** - tag 는 알람 키이지 테이크 키가 아니다.
3. **소유권 혼재** - 입력 계약(`align_images`)과 파생 증거(녹화)가 같은 트리에 산다.
4. **집계 축과 이벤트 축의 혼동** - `debug_images/<도구>/<tag>` 는 "도구별로 모아 보기"에
   최적화된 배치다. 그런데 실제 조회 질문은 언제나 "**방금 그 알람**에서 무슨 일이 있었나" 다.

## 설계

### 새 루트: `runs/` - 폴더 하나가 곧 한 테이크

```
poc/workflow_3/runs/                          # env ALIGN_FAIL_RUNS_DIR 로 override
├─ LATEST.txt                                 # 가장 최근 '시작된' 테이크의 절대경로 한 줄
└─ 260820/                                    # 날짜 버킷 (보존 정책 + 디렉터리 엔트리 수 관리)
   └─ 260820_113045_MCD916_CLASS-ARDL/        # <tag>_<eqp>_<recipe-slug>[__a2]
      ├─ summary.json                         # 제일 먼저 여는 파일
      ├─ events.log                           # 이 테이크의 work2 이벤트만 추출
      ├─ 00_connect/                          # RCS 준비/로그인/List 탭/tool row crop + OCR
      ├─ 01_gates/                            # row_occupant, share_request, access_request
      ├─ 02_recording/                        # 본 녹화 (+ prelude/ 하위)
      ├─ 03_correction/                       # _marked.jpg, _feasibility.json, consensus crop
      ├─ 04_engineer/                         # engineer_done 카운터 crop
      └─ steps/step_*.json                    # runner 저널
```

숫자 접두어는 사이클의 **시간 순서**다. 폴더를 위에서 아래로 읽으면 그것이 사건의 경과다.
`workflow_3e` 의 abort 테이크는 같은 루트에 살되 슬롯이 다르다: `A0_abort/`.

### 이름 규칙 (세 가지가 모두 필요하다)

- **recipe slug** - `recipe_id` 는 `"<class>/<recipe>"` 라 경로 구분자를 품는다
  (`rcp_msr_gather.py:163` 주석이 3단 중첩을 명시). 그대로 f-string 에 넣으면 디렉터리가
  중첩된다. `/` → `-` 치환 후 `manual_record.sanitize_eqp_for_path` 와 같은 규약으로 정제한다.
- **미등록** - `recipe_id` 가 비면 `_unregistered` 를 쓴다(현행 폴백과 같은 이름).
- **테이크 suffix** - 이벤트 폴더가 이미 있고 그 안에 `summary.json` 이 있으면
  `__a2`, `__a3` … 를 붙인다. **파일시스템에서 유도하며 모니터 상태를 쓰지 않는다** -
  cooldown 재시도, 프로세스 재시작, 두 진입점(align/check-only), `workflow_3e` 어디서 와도
  같은 규칙이 성립해야 하기 때문이다. `view_only_attempts` 는 eqp 키에 알람 해제 시 리셋되므로
  테이크 카운터로 쓸 수 없다.

### 단일 경로 구성점

`poc/workflow_3/run_dirs.py` 하나만 경로를 만든다. consensus 캐시에서 이미 채택한
`consensus_gather._events_dir_for` 규약과 같은 이유 - 경로 구성이 흩어지면 누군가 축을
다시 끼워 넣는다. **suffix 로직도 여기에 산다**; `workflow_3e` 가 자기 이름 규칙을 따로 두면
충돌이 그대로 재현된다.

```python
SLOT_CONNECT   = "00_connect"      # 오탈자로 03_corrections/ 가 생기지 않도록 상수 고정
SLOT_GATES     = "01_gates"
SLOT_RECORDING = "02_recording"
SLOT_CORRECTION= "03_correction"
SLOT_ENGINEER  = "04_engineer"
SLOT_STEPS     = "steps"
SLOT_ABORT     = "A0_abort"

def new_event_dir(tag, eqp_id, recipe_id="", *, job="align") -> Path   # 테이크 시작 시 1회
def event_subdir(event_dir: Path, slot: str) -> Path
def write_summary(event_dir: Path, payload: dict) -> None
def update_latest(event_dir: Path) -> None
def preflight(runs_root: Path) -> None                                 # 실패 시 예외
```

### `summary.json` - 이벤트 폴더의 색인

`CycleResult` 를 직렬화하되 경로는 **이벤트 폴더 기준 상대경로**로 적는다(절대경로만 적으면
폴더를 Mac 으로 복사했을 때 전부 깨진다). `attempt` 필드로 몇 번째 테이크인지 남긴다.

```json
{
  "tag": "260820_113045", "attempt": 2, "job": "align",
  "eqp_id": "MCD916", "recipe_id": "CLASS/ARDL",
  "alid": "9006", "utc9": "...", "alarm_name": "...",
  "run_status": "failed", "failed_step": "locate_sem_panel",
  "failure_class": "panel_not_found",
  "outcome_status": "", "key_decision": "", "best_xy": "",
  "occupancy": "free", "frame_count": 1832, "prelude_frame_count": 0,
  "artifacts": {"recording": "02_recording", "correction": "03_correction",
                "steps": "steps", "events": "events.log"},
  "notes": ["..."]
}
```

### `events.log` - 콘솔 tee 를 쓰지 않는 이유

프로세스 stdout 을 사이클마다 갈아 끼우는 방식은 **채택하지 않는다**. 모니터는 녹화 스레드,
engineer-done watcher, 비차단 `gather_success_async` 가 함께 도는 장수 프로세스라 tee 를
중간에 바꾸면 스레드 간 print 가 엉뚱한 사이클 로그로 들어가고, teardown 구간이 유실된다.
콘솔이 cp949 인 점도 tee writer 의 인코딩을 새 실패 지점으로 만든다.

대신 이미 인코딩이 해결된 `log_work2_event` 감사 로그에서 이 테이크의 이벤트만 뽑아
`events.log` 로 쓴다. 뽑는 기준은 시간창 추정이 아니라 **명시적 `tag` 필드**다.

호출부가 42곳이고 지금 `tag=` 를 넘기는 곳이 0곳이므로, 42곳을 손으로 고치는 대신
`logger.py` 에 **현재 테이크 바인딩**(모듈 전역 + `set_current_take()/clear_current_take()`)을
두고 `log_work2_event` 가 필드가 없을 때 채워 넣는다. 편집 지점이 2곳(`logger.py`, `cycle.py`)이
된다. 정직하게 적어 둘 한계: 이는 **모니터가 알람을 순차 처리한다**는 사실에 기댄다
(`workflow_3e/dispatch.py` 의 통합 supervisor 도 한 번에 한 잡만 돌린다 - 단일 RCS 커서를
직렬화하려는 원래 이유와 같다). 사이클 밖에서 도는 `gather_success_async` 의 늦은 로그는
이전 테이크로 잘못 귀속될 수 있다. 그 오귀속은 `tag` 필드 불일치로 **보이기라도 한다** -
tee 방식의 유실은 보이지 않는다.

리터럴 콘솔 전사가 필요한 촬영/시연은 종전대로 오피스에서 프로세스 레벨 tee 를 건다
(PowerShell `Tee-Object` → `DEMO_VIDEO_CONSOLE_LOG`, `make_demo_video.py:61-75`).

### 계약 (테스트로 고정할 것)

1. **이벤트 폴더는 테이크 시작 시 만들어지고, 어떤 실패 경로로도 `summary.json` 이 남는다.**
   기록 위치는 기존 teardown 과 같은 `try/finally` 다.
2. **증거물은 절대 `align_images/` 에 쓰지 않는다.** 반대로 `align_images/` 가 읽기 전용인
   것은 **아니다** - `rcp_msr_gather.py:164` 가 `ALIGN_IMAGES_DIR/<eqp>/<recipe_id>` 로 MES
   자료를 내려받고 `office_success_downloader` 도 거기에 쓴다. 정확한 계약은
   **"MES 입력은 쓰되 증거물은 없다(MES-input-writable, evidence-free)"** 이다.
3. **`tag` + `attempt` 가 테이크를 유일하게 식별하며, 폴더 이름이 곧 join key 다.**
   `WorkflowRunner` 는 자기 타임스탬프를 찍지 않고 호출자가 준 `run_dir` 을 쓴다.
   인자를 안 주면 종전 경로로 폴백한다(다른 호출자 무영향).
4. **`runs/` 접근 실패는 기동 시에 터진다.** 루프 시작 전에 `preflight()` 가 루트를 만들고
   probe 파일을 쓰고 지운다. 실패하면 배너를 찍고 **루프를 시작하지 않는다**.
   사이클 중간 폴백은 두지 않는다 - 기동 시점에는 사람이 있고, 녹화 중에는 없다.

### 바뀌지 않는 것

- `logs/work2.log`, `logs/vlm_calls.log`, `logs/align_fail_cycles.csv` 는 **그대로 유지**한다.
  이벤트 폴더는 "한 건 깊게", 이들은 "여러 건 넓게" 보는 축이다. `align_fail_cycles.csv` 에
  `event_dir` 컬럼을 하나 추가해 두 축을 잇는다.
- `RecordingSession` 의 동작(예산/heartbeat/manifest/파일명). 출력 경로 주입만 바뀐다.
- `align_img_from_rcp` / `cond.txt` / consensus 캐시 경로.
- `debug_images/` 는 남되 **사이클이 없는 도구 전용**이 된다: `bench_*`,
  `align/diagnostics/*`, `demonstration_rcs_control`, `engineer_done_calib`.

## 깨지는 것과 대응

| 깨지는 곳 | 왜 | 대응 |
|---|---|---|
| `filter_recording.py:63-68` | `align_images` 고정 glob 3종 | `runs/*/*/02_recording` 패턴 추가. 기존 3종은 **남긴다**(과거 녹화가 그 자리에 있다) |
| `make_demo_video.find_recording_dirs` | 루트로 `ALIGN_IMAGES_DIR` 전달 | 두 루트를 훑고 mtime 순 병합 |
| `workflow_extract/extract_workflow.py:220` | 경로 모양으로 EQP/`_manual` 판별 | `summary.json` 이 있으면 먼저 읽고, 없으면 종전 경로 파싱 |
| `workflow_3e/dispatch.py:103`, `abort_cycle.py` | 같은 tag 규칙 | `run_dirs` 를 공유하고 `A0_abort` 슬롯 사용 |
| `align_fail_monitor_only_check.py` | 같은 `_alarm_time_to_tag` | 같은 suffix 로직 사용(`job="check"`) |
| `README.md` 111/322/325/340-341, `docs/runbooks/demo_video_shot_list.md` 271-273 | 문서화된 경로 | 문서 갱신 |
| `.gitignore` | `runs/` 미등재 | **1단계 이전에 등재**(아래) |
| 오피스에 이미 쌓인 녹화 | 위치가 다름 | **이동하지 않는다.** 소급 마이그레이션 없음 |

## 마이그레이션 순서 (각 단계 단독 배포 가능)

0. **`.gitignore` 가드 (첫 실행 이전 필수)** - `poc/workflow_3/runs/` 등재. 20fps × 10분이면
   테이크당 1GB 대다. `align_images` 이전 때와 같은 순서다(README "align_images 루트 이전
   체크리스트" 1번과 동일한 이유).
1. `run_dirs.py` + 단위 테스트만 추가. 아무도 호출하지 않는다. suffix/슬러그/`_unregistered`
   /preflight 를 여기서 전부 덮는다. (무위험)
2. `WorkflowRunner` 에 `run_dir` 주입 파라미터 + 폴백. 사이클이 `runs/<...>/steps` 를 넘긴다.
3. `debug_images` 계열 7개 호출부(`cycle.py:387,546,960,1084,1338` + `rcs/` 2개 모듈 상수)를
   `event_subdir` 로 전환. **녹화는 아직 옮기지 않는다.**
4. `logger.py` 테이크 바인딩 + `summary.json` + `LATEST.txt` + `events.log`,
   CSV 에 `event_dir` 컬럼 추가, 기동 `preflight()`.
5. 녹화/캡처(`_recording_dir_for`, `_capture_dir_for`)를 옮기고 **같은 커밋에서** 소비자
   3곳을 갱신한다.
6. `workflow_3e` 를 같은 규약으로 옮긴다.

## 위험과 롤백

- `ALIGN_FAIL_RUN_LAYOUT=legacy` 가 5단계만 되돌린다(녹화가 `align_images` 로 복귀).
  이 스위치는 **기동 배너에 찍고 CSV manifest 컬럼에 남긴다** - 오피스 피드백이 붙여넣은
  콘솔 텍스트뿐이라, 어느 레이아웃으로 돌았는지가 그 텍스트에 보여야 한다.
- 3단계까지는 debug 산출물 위치 변경뿐이라 스위치를 두지 않는다. 스위치가 늘면 두 레이아웃을
  영구히 유지해야 하고 소비자 glob 이 영원히 이중이 된다.
- 가장 큰 위험은 **5단계에서 프레임을 잃는 것**이다. 완화는 사이클 폴백이 아니라 기동
  preflight(계약 4)다.
- 경로 길이: 새 경로가 종전보다 짧아 Windows `MAX_PATH` 여유는 늘어난다
  (`runs/<yymmdd>/<tag>_<eqp>_<slug>/02_recording/<파일>` vs
  `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/<파일>`).

## 보존 정책

`runs/` 는 무한히 자란다. 날짜 버킷(`runs/<yymmdd>/`)을 전제로:

- **프레임만 회수한다.** `02_recording/` 와 `prelude/` 만 용량 상한으로 오래된 것부터 지운다.
- **`summary.json` / `steps/` / `events.log` / crop 은 남긴다.** 킬로바이트 단위이고,
  "그때 무슨 일이 있었나" 를 답하는 것은 이쪽이다.
- 오피스에는 스케줄러를 걸 사람이 없으므로 **인자 없는 스크립트** 로 만든다:
  `uv run python poc/workflow_3/monitor/prune_runs.py` (`RUNS_RETENTION_*` env 로 상한 조정).

## 남은 열린 질문

- 수동 녹화(`manual_record.py`, `_manual/<tag>`)와 시연(`demonstration_rcs_control`)도
  `runs/` 로 통일할 것인가. 개념상 같은 "테이크"지만 이번 범위 밖으로 둔다.
- `LATEST.txt` 는 job 구분 없이 "가장 최근 시작된 테이크" 하나만 가리킨다. align 과 abort 를
  따로 보고 싶다는 요구가 생기면 `LATEST_<job>.txt` 로 늘린다.
