# 10 — Episode identity through alarm-row processing

Type: task
Status: resolved
Blocked by: None — can start immediately
Spec: [spec.md](../spec.md) (Recovery Episode lifecycle and persistence)

## What to build

알람 모니터의 fail-row 처리가 ALID=9006 active interval 하나에 Recovery Episode 하나를 만든다.
Episode 정본 `recovery_episode.json` 은 기존 per-alarm capture 폴더 `captured_img_from_rcs/<tag>/`
(recipe 없는 알람은 `_unregistered/<tag>/`) 루트에 **GUI 작업이 시작되기 전에** 원자적으로 써진다.
cooldown 재시도는 같은 Episode 를 재개하며 `attempt_seq` 를 올리고, 알람이 poll 에서 사라지면
clearance 이벤트를 남기고 Episode 를 닫는다. 이후 같은 장비·레시피의 재발은 새 Episode 다.

Episode 수집은 기본 off 인 플래그 뒤에 있고, off 면 현재 모니터 동작과 기존 테스트가 그대로다.
Mac 에서는 replay CSV 소스(첫 poll 에 알람, 다음 poll 은 빈 목록)로 생성 → 재시도 → clearance 경로를
실장비 없이 돌려 볼 수 있어야 한다.

## Acceptance criteria

- [x] 같은 알람의 cooldown 재시도는 같은 `episode_id` 를 재사용하고 `attempt_seq` 가 1, 2, 3… 으로 증가해 파일의 ordered attempts 에 남는다.
- [x] 알람이 poll 에서 사라지면 clearance 이벤트가 기록되고 Episode 가 닫힌다; 같은 EQP/recipe 의 후속 알람은 다른 `episode_id` 를 받는다.
- [x] `episode_id` 는 UUID 이고 어떤 경로·타임스탬프에서도 재구성되지 않는다. 알람 fingerprint = 장비 + alarm code + recipe + 원 UTC9 는 별도 필드다.
- [x] 초기 Episode 파일은 첫 GUI step 전에 존재한다. cycle 이 예외로 끝나도 파일은 남고 `incomplete` + reason 이 기록된다(삭제 없음).
- [x] 저장되는 artifact 경로는 전부 Episode-relative 다. 절대 경로와 `..` 탈출은 로드 시 거부된다.
- [x] 쓰기는 temp + atomic replace 다. `_unregistered/<tag>/` 도 같은 규약이다.
- [x] 파일에 schema 버전, observation-contract 버전, `bindings_version`, `execution_mode="live"` + settings snapshot 참조(safe_mode/dry-run 이 provenance 로 보임)가 stamp 된다.
- [x] 수집 플래그 off 에서 기존 monitor/cycle 테스트가 무변경으로 통과한다.
- [x] replay CSV 로 Mac 에서 생성 → clearance 경로가 돈다(테스트 또는 문서화된 실행 예).
- [x] spec 테스트 1(identity 부분), 2, 15/34(경로), 16(incomplete 보존)을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- 새 모듈 `poc/workflow_3/monitor/recovery_episode.py` - `EpisodeTracker`
  (`begin_attempt`/`finish_attempt`/`fail_attempt`/`close_cleared`), 순수 경로 함수
  `episode_root_for`, `alarm_fingerprint`, `load_episode`(경로 규약 검증),
  `attempt_dirname`. 쓰기는 temp + `os.replace`.
- `monitor/align_fail_monitor.py` - `process_fail_rows(..., episodes=None)` kwarg.
  Episode 는 알람 row 처리 **맨 앞**(gather/사이클 전)에서 열리고, 사이클 결과가
  `finish_attempt`, 예외가 `fail_attempt` 로 간다. clearance 는 두 곳에서 닫는다:
  `process_fail_rows` 의 `close_cleared(current_tools)` 와 메인 루프의
  `_alarm_rows_empty` 분기(`close_cleared(())`). 후자를 빠뜨리면 알람이 전부 사라진
  poll 에서 Episode 가 열린 채 남아 다음 알람이 잘못 재개된다.
- `config.py` - `episode_collect_enabled: bool = False` + `ALIGN_FAIL_EPISODE_COLLECT`.
  `workflow_3_config.example.py` 에 `EPISODE_COLLECT = None`, loader 매핑 1줄.

**테스트** `poc/workflow_3/monitor/test_recovery_episode.py` 6개 (전부 Mac, RCS 없이).
spec 테스트 1(identity), 2, 15/34(경로), 16(incomplete 보존) 커버.
회귀: monitor 510 / recording_filter+workflow_extract 231 / workflow_4 36 통과.

**Mac replay 실행 확인** — 아래를 8~20초 돌려 파일로 생성 -> attempt -> clearance ->
`state=closed` 를 확인했다(UTC9 는 `ALIGN_FAIL_WINDOW_SEC` 안이어야 방출된다):

```
ALIGN_FAIL_EPISODE_COLLECT=1 SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay \
  ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

**spec 과 다르게 한 점 / 판단**
- `CAPTURED_RCS_DIRNAME` 을 `rcs/rcs_screenshot.py` 에서 import 하지 않고 새 모듈에
  같은 값으로 다시 뒀다. 그 모듈이 pywinauto 를 최상단에서 끌어와 Mac 에서 import
  자체가 실패하는데, Episode 경로 계산은 실장비 없이 성립해야 한다. 반대 방향
  (`rcs` -> `monitor`) import 는 계층 규약 위반이라 택하지 않았다.
- 수집 off 일 때 tag 계산 경로를 종전 그대로 뒀다(`_alarm_time_to_tag` 결과를 그대로
  `run_alarm_cycle` 에 넘기고 None 폴백은 사이클이 한다). 수집 on 일 때만 모니터가
  tag 를 확정해 Episode 에 기억시킨다 - off 에서 byte 단위 동일을 지키기 위함이다.
- `run_status="aborted"`(runner step 실패)는 `complete=true` 다. 티켓이 지정한
  incomplete 집합은 `{error, rcs_unavailable, cycle_disabled}` + 예외뿐이며,
  step 실패는 '수집이 깨진 것' 이 아니라 관측된 결과이기 때문이다.
