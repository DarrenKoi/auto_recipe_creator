# 13 — Durable Guard readings

Type: task
Status: resolved
Blocked by: 11, 12
Spec: [spec.md](../spec.md) (Capture and durable Recovery Guards)

## What to build

attempt 마다 Episode-level Recovery Guard 정확히 세 종류를 읽어 record 로 저장한다.

1. 화면 관측 가능성/가림 - 사이드카 occlusion + 창 rect 존재.
2. 점유/control - tool row 점유 3상태(`occupied_by_other`→`false`, `free`→`true`, `unknown`→`unknown`)
   와 화면 공유 요청 결과.
3. SEM mode + align key 가시성/유일성 - mode 가 읽혔고, 그 mode 의 템플릿이 매칭됐고, key 가 유일할 때만
   `true`. feasibility 의 `ambiguous`/`candidate` 는 `true` 가 아니다. 읽은 OM/SEM 값은 detail/provenance
   에만 남고 Guard 값이 아니다(v1 signature 밖).

기존 matcher/feasibility/점유 코드는 어댑트만 하고 동작을 바꾸지 않는다. OK-control availability 는
Guard 가 아니라 `confirm_align` 의 precondition 기록이다.

## Acceptance criteria

- [x] 각 reading = `true`/`false`/`unknown`, reason, observation time, Episode-relative evidence ref. Guard kind 는 셋뿐이고 추가 경로가 없다.
- [x] 읽기/파싱/stale 사이드카/asset 없음/mode 미판독/capture 실패/matcher 예외는 전부 `unknown` 이다 - `false` 나 `true` 로 새지 않는다.
- [x] feasibility `candidate`/`ambiguous` 에서 Guard 3 은 `true` 가 아니다; 읽은 mode 는 detail 에만 있다.
- [x] OK-control availability 는 precondition 기록으로 저장되고 Guard 목록에 없다.
- [x] 기존 matcher/occupancy/share 테스트가 무변경으로 통과한다.
- [x] spec 테스트 5, 6 을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- 새 모듈 `monitor/guard_readings.py` - 순수 분류 함수 3개(`screen_observability_guard`,
  `occupancy_guard`, `align_key_guard`) + `ok_control_precondition` + `write_guard_records`.
  `GUARD_KINDS` 는 닫힌 3-튜플이고 다른 kind 를 만드는 경로가 없다. 새 관측을 하지 않고
  호출부가 이미 얻은 값을 **분류만** 하므로 matcher/occupancy/share 동작은 무변경이다.
- `monitor/cycle.py` - `collect_attempt_guards()` 가 사이클 문맥에서 값을 모으고
  `write_attempt_guards()` 가 attempt 폴더에 `guards.json` 을 쓴다. 호출은
  `run_alarm_cycle` 의 **finally, teardown 앞**이다: 사이클이 깨진 attempt 도 기록이
  있어야 하고(그때 값은 전부 unknown 이며 그것이 정직한 관측이다), teardown 뒤로 밀면
  창이 닫혀 사이드카 마지막 레코드가 stale 로 샌다.
- `monitor/frame_meta.py` - `FrameMetaRecorder` 가 `last_record`/`last_at` 를 노출한다
  (Guard 가 파일을 다시 파싱하지 않고 "그때 창이 보였는가" 를 묻고 stale 도 잰다).
- `monitor/cycle.py` - 화면 공유 요청 status 를 `context["share_status"]` 로 남긴다
  (provenance 전용, 분기 없음).
- `monitor/recovery_episode.py` - `_ATTEMPT_RECORD_FILES` 에 있는 record 파일 중
  **실제로 존재하는 것만** attempt artifacts 에 Episode-relative 로 건다. 티켓 15/16 은
  여기 한 줄씩 추가하면 된다.

**값 표현** - `True`/`False`/`None`(=unknown). 문자열 `"false"` 를 쓰면 파이썬에서 truthy 라
`if value:` 한 줄로 unknown/false 가 활성화 쪽으로 새는데, `None` 은 그 실수까지 안전한 쪽으로
떨어진다. JSON 에서는 `null` 이라 unknown 이 명시적으로 실린다.

**판정 규약**
- Guard 1: `none`->true, `partial`/`full`->false(부분 가림도 근거로 못 쓴다), 사이드카
  부재/stale/rect 없음/`unknown`->unknown.
- Guard 2: `free`->true, `occupied_by_other`->false, `unknown`->unknown. 화면 공유 승낙은
  **관전**이지 제어가 아니므로 값을 올리지 않고 detail 에만 남는다.
- Guard 3: mode 판독 + 템플릿 매칭 + 유일성이 **모두** 성립할 때만 true. ambiguous /
  구조 유일성 없는 candidate / 키 부재는 false(관측된 부정). mode 미판독 / no_assets /
  matcher 예외 / matcher 미실행 / **유일성 미판독(`second_ratio is None`)** 은 unknown -
  matcher 의 `distinctive` 는 데이터 결손 시 false-flag 를 피하려고 True 를 기본값으로
  가지므로, 그 True 를 '유일함이 확인됐다' 로 읽으면 안 된다.
- 읽은 OM/SEM 은 detail 에만 있다. OK 컨트롤 가용성은 `preconditions` 목록에 따로 있다.

**테스트** `monitor/test_guard_readings.py` 12개. spec 테스트 5, 6 커버.
회귀: monitor 536 / recording_filter+workflow_extract+workflow_4 269 /
row_occupant+share(85) / align correction(16) 통과.

**spec 과 다르게 한 점 / 판단**
- Guard 2 의 `evidence` 는 **빈 문자열**이다. 점유 판독의 근거 crop 은
  `debug_images/row_occupant/` 에 살고 `read_occupancy` 는 그 경로를 반환하지 않는다.
  선택지는 (a) 밖을 가리키는 절대 경로를 적는다 (b) 그 crop 을 attempt 로 복사한다
  (c) 비운다 였다. (a)는 "Episode-relative 만" 규약을 깨고 폴더를 옮기면 끊긴다. (b)는
  `read_text_near_point`/`read_occupancy` 반환 계약을 바꿔야 해서 "어댑트만 한다"를
  어긴다. 그래서 (c)를 골랐고 점유 상태·공유 status 는 detail 에 남는다. 티켓 21
  (annotation)이 실제 근거를 사람 손으로 붙일 수 있는 자리다.
- Guard 기록 자체도 `episode_collect_enabled` 게이트 뒤에 둔다(티켓 10~12 과 같은 규약).
