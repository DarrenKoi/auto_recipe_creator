# Align Fail Recovery Playbook - 증분 1·2 구현 프롬프트 (티켓 10–17)

이 리포는 `/Users/daeyoung/Codes/auto_recipe_creator` (오피스에서는 pull 받은 같은 리포) 다. 아래를 **순서대로** 읽고 시작한다.

1. `CLAUDE.md` (코드 규약·패키지 지도·테스트 명령), `CONTEXT.md` (도메인 용어집 - Episode/Guard/Action/Verification/Outcome 은 이 정의대로만 쓴다)
2. `docs/issues/align-fail-recovery-playbook/spec.md` - **정본**. 티켓과 다르면 spec 이 이긴다. spec 은 수정하지 않는다.
3. `docs/issues/align-fail-recovery-playbook/issues/10-*.md` ~ `17-*.md` - 구현 티켓. `01`~`09` 는 이미 닫힌 결정 티켓이라 배경으로만 본다.
4. `docs/agents/issue-tracker.md` - 티켓 파일 규약(`Status:` 줄, `## Comments`).

## 범위

티켓 **10 → 11 → 14 → 12 → 13 → 15 → 16 → 17** 을 이 순서로 구현한다(각 티켓의 `Blocked by:` 가 이 순서를 정한다; 11 뒤에 14 와 12 는 어느 쪽이 먼저여도 된다). **18 은 사람이 오피스에서 하는 gate 이고 19–27 은 18 뒤다 - 손대지 않는다.** 17 이 끝나면 멈추고 보고한다.

## 작업 방식

- 티켓 하나씩. 한 티켓 안에서는 TDD - 실패하는 테스트 하나 → 통과하는 최소 구현 → 반복. 테스트는 spec 의 "Testing Decisions" 가 정한 seam 에서만 쓴다: 티켓 10/14 = `process_fail_rows` (fake alarm row + 주입된 fake `run_alarm_cycle`), 11/12 = `RecordingSession`/cycle 이 쓰는 폴더와 manifest, 13/15/16 = attempt 폴더에 남는 record 파일, 17 = 순수 함수 + Episode 파일의 final Outcome/digest. private helper 호출 순서를 단정하는 테스트는 쓰지 않는다.
- 새 테스트는 `poc/workflow_3/monitor/test_recovery_episode.py` (10/11/14), 나머지는 해당 모듈 옆 `test_*.py`. pytest 스타일, Mac 에서 Windows 의존성 없이 돌아야 한다(`tmp_path` 로 `ALIGN_IMAGES_DIR` 대체).
- 기존 테스트 패턴을 재사용한다: `poc/workflow_3/monitor/test_failure_cooldown.py` 의 `_stub_deps`(process_fail_rows 의 부수효과 함수를 no-op 로 교체) 와 `_cycle_returning`(fake `run_alarm_cycle`).
- 티켓마다 끝나면: 그 티켓이 적은 spec 테스트 번호가 덮였는지 체크박스를 채우고, 티켓 파일의 `Status:` 를 `resolved` 로 바꾸고 `## Comments` 에 (무엇을 어디에 넣었는지 / 테스트 수 / spec 과 달리 한 점) 을 적는다.
- 티켓마다 한 번 **pathspec 커밋** (`git add <내가 만든/고친 파일들>` - `git add -A` 금지, 다른 세션이 같은 리포를 동시에 편집한다), 메시지 예: `feat(episode): ticket 10 - Episode identity through alarm-row processing`. 커밋 전 `git show --stat` 로 범위를 확인한다. main 에 직접 커밋한다(브랜치 만들지 않음).

## 코드 규약 (CLAUDE.md 요약 - 위반 시 리뷰에서 반려)

- Python 3.10+, `uv run`. **새 의존성 금지** (json/uuid/hashlib/dataclasses/os.replace 만).
- 한국어 docstring. `from __future__ import ...` 금지. 로깅은 `print("[INFO] ...")`/`[WARNING]`/`[ERROR]` 이고 `logging` 모듈 금지. **print 문자열 안에 em-dash(U+2014) 금지**(오피스 콘솔 cp949).
- **argparse/CLI 인자 금지.** 설정은 `poc/workflow_3/config.py` 의 `Workflow3Settings` 필드 + `load_workflow3_settings()` 의 `env_flag/env_int/env_float` 한 줄, 그리고 `poc/workflow_3/workflow_3_config.example.py` 에 같은 이름의 상수 한 줄(주석 포함).
- 절대 import `from poc.workflow_3.xxx import ...`. `poc/workflow_4/playbook/` 는 **workflow_3 를 import 하지 않는다**(plain dict/dataclass 만 받는다).
- 새 플래그는 **기본 off**. off 면 기존 동작·기존 테스트가 byte 단위로 그대로다.
- `WorkflowRunner`, `poc/workflow_4/framework/engine.py`, teardown/notify/cooldown/abort 계약은 건드리지 않는다.
- 파일 쓰기는 temp + `os.replace` 원자적 (예: `poc/workflow_4/framework/graph_view.py:38-43`).

## 이미 확인된 seam 과 설계 결정 (다시 조사하지 말고 이대로)

**티켓 10**
- 진입점: `poc/workflow_3/monitor/align_fail_monitor.py` `process_fail_rows(fails, active_tools, settings, occupied_cooldown=None, view_only_attempts=None)`. `_collapse_rows_by_tool` 이 만드는 `info` dict 키: `eqp_id, alarm_time, utc9, alarm_name, alid, recipe_id, operation_desc, lot_type_cd`. 알람 fingerprint = `eqp_id + alid + recipe_id + utc9`.
- **주의:** 메인 루프는 알람이 전부 사라진 poll 에서 `process_fail_rows` 를 부르지 않고 직접 `active_tools.clear()` 한다(같은 파일 `while True:` 안의 `_alarm_rows_empty` 분기). Episode clearance 는 두 곳 모두에서 닫아야 한다.
- 새 모듈 `poc/workflow_3/monitor/recovery_episode.py` 에 `EpisodeTracker` (메모리 eqp→열린 Episode 맵): `begin_attempt(info, settings) -> handle(episode_id, attempt_seq, tag, root)`, `finish_attempt(handle, cycle: CycleResult)`, `fail_attempt(handle, reason)`, `close_cleared(current_eqp_ids)`. `process_fail_rows` 에 `episodes=None` kwarg 로 주입(None = 수집 off). 메인 루프는 `settings.episode_collect_enabled` 일 때만 tracker 를 만든다.
- Episode root = cycle 의 `_capture_dir_for` 와 **같은 경로**: `ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>` (recipe_id 는 `class/recipe` 문자열, 슬래시로 단계 분리) 또는 recipe 없으면 `ALIGN_IMAGES_DIR/<eqp>/_unregistered/<tag>`. 기존 `rcs/rcs_screenshot.captured_dir_for` 는 Windows 전용 모듈을 import 하므로 새 모듈에 **순수 경로 함수**를 두고 `cycle._capture_dir_for`/`_recording_dir_for` 가 그것을 쓰게 바꾼다(티켓 11 의 prefactor 시작). tracker 는 `images_root` 인자를 받아 테스트에서 `tmp_path` 를 넣는다.
- tag: `_alarm_time_to_tag(info["utc9"]) or make_timestamp_tag()` (`poc/workflow_3/util/time_utils.py`) 를 `process_fail_rows` 에서 한 번 계산해 `run_alarm_cycle(tag=)` 로 넘긴다. 재시도는 Episode 가 기억한 tag 를 다시 쓴다. tag 는 위치이지 identity 가 아니다 - identity 는 `uuid4`.
- `recovery_episode.json` v1 필드: `schema_version="recovery_episode.v1"`, `observation_contract="align_fail_observation.v1"`, `bindings_version=null`(티켓 26 전까지), `episode_id`, `alarm`(info 전부), `fingerprint`, `execution_mode="live"`, `tag`, `state` open|closed, `opened_at`/`closed_at`, `next_event_seq`, `attempts[]`, `events[]`, `outcome="unknown"`(티켓 17 이 채움), `recovery_actors=[]`, `complete`, `incomplete_reasons[]`. attempt 항목: `attempt_seq`(1부터), `started_at/finished_at`, `execution_mode`, `settings={safe_mode, correction_dry_run, action_enabled}`, `run_id`(= `CycleResult.run_dir` 의 basename, 경로 아님), `run_status`, `failed_step`, `failure_class`, `outcome_status`, `artifacts={...Episode-relative 경로만}`, `complete`, `incomplete_reason`.
- `event_seq` 는 Episode 전체 단조 카운터 하나(`next_event_seq`), 이벤트는 `attempt_seq`(episode-level 이면 null) 를 함께 가진다. 이벤트 kind: `attempt_started`, `attempt_finished`, `alarm_cleared`, `attempt_error`.
- attempt 는 `run_status ∈ {error, rcs_unavailable, cycle_disabled}` 또는 예외면 `complete=false` + reason. Episode `complete` = 모든 attempt complete; `incomplete_reasons` 는 `attempt_<n>:<reason>` 로 모은다. **파일은 절대 지우지 않는다.**
- 로드 시 artifacts 의 모든 경로 문자열에 대해 절대 경로와 `..` 를 거부한다.
- Config: `episode_collect_enabled: bool = False`, env `ALIGN_FAIL_EPISODE_COLLECT`, example 파일 상수 `EPISODE_COLLECT = None`.
- Mac 확인 명령(문서에 적는다): `ALIGN_FAIL_EPISODE_COLLECT=1 SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> uv run python poc/workflow_3/monitor/align_fail_monitor.py` - RCS 모듈이 없어 attempt 는 `rcs_unavailable` 로 incomplete 가 되지만 Episode 생성 → 빈 poll clearance → closed 가 파일로 보인다.

**티켓 11**
- cycle 의 `_recording_dir_for`(cycle.py 약 745행)와 `_capture_dir_for`(약 1506행)를 하나의 attempt-dir resolver 로 합친다. Episode 수집 on 이면 `<root>/attempt_<n>/` 아래(`recording/`, `recording/prelude/`, 캡처 jpg, matcher/feasibility 산출물), off 면 종전 `<tag>/` 그대로. attempt_seq 는 `run_alarm_cycle` 에 kwarg 로 넘긴다.
- `poc/workflow_3/recording_filter/filter_recording.py` 64–66행의 glob 3개에 `*/attempt_*/recording` 깊이를 **추가**(기존 것 유지). `make_demo_video` 는 `rglob("recording")` 이라 무변경 - 그 사실을 고정하는 테스트만 추가.

**티켓 12**
- 사이드카 래퍼는 `poc/workflow_3/monitor/manual_record.py` `_make_capture_fn` + `poc/workflow_3/monitor/frame_meta.py` `FrameMetaWriter`. 이것을 공용 함수로 올려(예: `frame_meta.py` 안) cycle 의 본 녹화 `RecordingSession(... capture_fn=capture_window)` (cycle.py 약 818행) 가 같은 래퍼를 쓰게 한다. manifest 는 `poc/workflow_3/monitor/recording.py` 가 쓴다 - `episode_id/attempt_seq/capture_completeness` 를 additive 로.

**티켓 13**
- 점유 3상태: `poc/workflow_3/rcs/row_occupant.py` (`FREE/OCCUPIED_BY_OTHER/UNKNOWN`). key 가시성/유일성: `poc/workflow_3/align/diagnostics/feasibility_check.py` `mark_align_feasibility -> FeasibilityResult`(verdict, `second_ratio`, `reregister_recommended`). mode: `sem_box_detect` 의 `pm_mode` (cycle 이 `mode_hint` 로 주입). Guard record 는 attempt 폴더에 JSON(`guards.json` 또는 JSONL) 로.

**티켓 15**
- Assist 패널: `poc/workflow_3/sem_monitor/assist_score.py` `locate_assist_panel(...)`, `read_assist_state(panel_image)` - 행 band 만 센다. 열 분리 reader 를 **만들지 않는다**. stub 은 crop 만 저장하고 `unknown(reason="reader_not_calibrated")`.

**티켓 16**
- `poc/workflow_3/monitor/engineer_done_align_adjustment.py` `EngineerDoneDetector(..., debug_dir=None)`, `__call__ -> bool`. per-read 판독을 attempt 폴더에 JSONL 로 추가 기록(반환 계약 무변경).

**티켓 17**
- 새 패키지 `poc/workflow_4/playbook/` (`__init__.py` + `outcome.py` 등). 순수 함수 `derive_outcome(records) -> "recovered"|"escalated"|"aborted"|"unknown"`. cycle/monitor 가 Episode close 시 호출해 `outcome` 을 쓰고 `print("[DIGEST] episode ...")` 한 줄. 테스트는 `uv run pytest poc/workflow_4/`.

## 회귀 테스트 (티켓마다 마지막에 전부)

```bash
uv run pytest poc/workflow_3/monitor -q
uv run pytest poc/workflow_3/recording_filter poc/workflow_3/workflow_extract -q
uv run pytest poc/workflow_4 -q
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_manual_record.py
uv run pytest poc/workflow_3/monitor/test_prelude_recording.py poc/workflow_3/monitor/test_make_demo_video.py -q
```

## 하지 말 것

spec.md 수정 / 티켓 18 이후 착수 / 새 의존성 / 열 분리 Measurement reader 작성 / tag 충돌 결함을 attempt 폴더와 별도로 고치기 / `WorkflowRunner` 나 workflow_4 엔진에 필드 추가 / 손으로 적은 "실제 오피스 산출물 모양" fixture(합성 입력은 되지만 `recovery_episode.json` 의 진짜 견본은 티켓 18 이 가져온다) / 실제 마우스·키보드 코드 경로 변경.

## 최종 보고 형식

티켓마다 한 줄(커밋 해시, 추가/변경 파일, 테스트 수), 전체 회귀 결과, spec 과 달리 결정한 점, 열린 질문. 이미지 없이 텍스트만.
