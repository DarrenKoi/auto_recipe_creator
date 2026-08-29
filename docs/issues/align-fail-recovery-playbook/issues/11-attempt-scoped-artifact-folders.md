# 11 — Attempt-scoped artifact folders

Type: task
Status: resolved
Blocked by: 10
Spec: [spec.md](../spec.md) (Recovery Episode lifecycle and persistence; Operational entrypoints and compatibility)

## What to build

per-alarm cycle 이 attempt 마다 `<episode_root>/attempt_<attempt_seq>/` 를 만들고 그 attempt 의
recording(prelude 하위 폴더 포함), capture 프레임, matcher/feasibility 산출물을 전부 그 아래에 쓴다.
지금은 cooldown 재시도가 같은 `<tag>/recording/` 에 두 테이크를 뒤섞는데(알려진 tag 충돌 결함),
이 폴더 구조가 그 결함을 닫는다 - 별도 수정을 만들지 않는다.

**Prefactor 먼저:** cycle 에는 tag 로 recording 폴더를 정하는 resolver 와 capture 폴더를 정하는
resolver 가 따로 있다. 둘을 하나의 attempt-dir resolver 로 합친 뒤 attempt 깊이를 넣는다.

runner journal 은 Episode root 밖 제자리에 남고 run id 로만 참조된다. 오프라인 소비자 중
`recording_filter` 의 고정 깊이 발견 glob 만 `attempt_*` 깊이를 얻고, `make_demo_video` 는 재귀 발견이라
무변경, manual 녹화 `_manual/<tag>/recording/` 도 무변경이다.

## Acceptance criteria

- [x] cooldown 재시도 2회가 `attempt_1/recording/` 과 `attempt_2/recording/` 으로 갈리고 프레임·manifest 가 섞이지 않는다.
- [x] prelude 녹화는 `attempt_<n>/recording/prelude/` 에 간다.
- [x] Episode 파일의 각 attempt 항목이 자기 폴더와 산출물을 Episode-relative 로 가리킨다.
- [x] `recording_filter` 발견이 `<tag>/recording/` 과 `<tag>/attempt_<n>/recording/` 을 둘 다 찾는다(구 녹화 호환); manual 경로는 무변경.
- [x] `make_demo_video` 가 코드 변경 없이 attempt 하위 녹화를 찾는다(재귀 발견을 고정하는 테스트).
- [x] runner journal 위치는 그대로이고 Episode 파일은 run id 로 참조한다.
- [x] 수집 플래그 off 면 종전 `<tag>/recording/` 그대로다.
- [x] spec 테스트 1(폴더 부분), 35 를 덮는다.

## Comments

구현 완료 (2026-08-30).

**Prefactor** - `cycle.py` 의 `_recording_dir_for` 와 `_capture_dir_for` 를 하나의
`_attempt_dir_for(eqp_id, recipe_id, tag, attempt_seq=None)` 로 합쳤다. 경로 계산 자체는
`recovery_episode.episode_root_for`(순수 함수) 가 소유하므로 Episode 정본과 산출물이
같은 자리를 가리키는 것이 구조적으로 보장된다. `_recording_dir_for` 는 그 위의 얇은
`/recording` 래퍼로 남겼다 - `test_prelude_recording.py` 가 이 이름을 monkeypatch 하므로
이름을 없애면 그 테스트가 깨지는데, 폴더 결정은 이미 한 곳(`_attempt_dir_for`)이라
prefactor 의 목적은 달성된다.

**무엇을 어디에**
- `cycle.py` - `_attempt_dir_for` 신설, `_capture_dir_for` 삭제(호출부 1곳 교체),
  `run_alarm_cycle(..., attempt_seq=None)` -> `context["attempt_seq"]`. 본 녹화와
  prelude 가 그 값을 읽어 `attempt_<n>/recording[/prelude]` 로 간다. 쓰이지 않게 된
  `captured_dir_for` import 도 걷어냈다(같은 try 블록의 `login_rcs_common` 이 이미
  RCS 가용성을 대표하므로 `RCS_MODULES_AVAILABLE` 판정은 그대로다).
- `align_fail_monitor.py` - 수집 on 일 때만 `attempt_seq=` 를 넘긴다(off 면 kwarg 자체를
  안 넘겨 사이클 호출 규약이 종전과 동일).
- `recovery_episode.py` - `relative_ref()` 로 `CycleResult.recording_dir`/`prelude_dir` 을
  Episode-relative 로 접어 `artifacts` 에 적는다. root 밖이면 **참조하지 않는다**(빈
  문자열) - 밖을 가리키는 절대 경로를 적느니 비우는 편이 규약에 맞다.
- `recording_filter/filter_recording.py` - `_discover_recording_dirs()` 로 분리하고
  `attempt_*` 깊이 2개를 **추가**했다(기존 3개 유지). 교체가 아니라 추가여야 구 녹화가
  계속 보인다.

**테스트** (+7)
- `test_recovery_episode.py` +4: attempt 폴더 분리(프레임/manifest 안 섞임), prelude 가
  같은 attempt 아래, 수집 off 면 종전 `<tag>/recording`, Episode 파일의 artifacts 가
  Episode-relative.
- `recording_filter/test_filter_recording.py` +2: 구/신 경로 동시 발견, prelude 제외.
- `monitor/test_make_demo_video.py` +1: `rglob` 재귀 발견이라 무변경이라는 사실 고정.
- `test_failure_cooldown.py` 의 `_cycle_returning` fake 가 `**_kwargs` 를 받도록 한 줄
  수정(사이클 인자를 검사하지 않는 fake 라 계약 변화 아님).

회귀: monitor 515 / recording_filter+workflow_extract 233 / workflow_4 36,
failure_cooldown·manual_record(47)·engineer_done(53) 통과.

**spec 과 다르게 한 점 / 판단**
- matcher/feasibility 디버그 산출물은 attempt 폴더로 옮기지 **않았다**. 그것들은
  `DEBUG_IMAGE_DIR/align_fail_cycle/<tag>` 에 쌓이고 `cycle_images.gather_and_report` 가
  테이크 단위로 모으는 별도 파이프라인이라, 옮기면 그 수집이 통째로 빈다. spec 은
  "Episode 는 기존 산출물을 **가리킨다**" 와 "저장 경로는 Episode-relative 뿐" 을 둘 다
  요구하므로, 해법은 옮기는 것이 아니라 **Episode 파일에 그 경로를 적지 않는 것**이다.
  Guard 3 의 evidence 는 티켓 13 이 attempt 폴더 안에 따로 남긴다.
