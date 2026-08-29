# 12 — Frame metadata for automatic recordings

Type: task
Status: resolved
Blocked by: 11
Spec: [spec.md](../spec.md) (Capture and durable Recovery Guards)

## What to build

manual recording 런처에만 붙어 있는 프레임 사이드카(창 rect, 전면 창 제목, 가림 여부, 로컬 커서)
capture 래퍼를 공용 모듈로 올려, 알람 사이클의 `RecordingSession` 도 같은 래퍼를 `capture_fn`
주입점으로 쓴다. 녹화기 자체는 포크하거나 바꾸지 않는다.

capture manifest 는 `episode_id`, `attempt_seq`, capture completeness 를 additive 로 얻는다. 새 필드를
모르는 기존 소비자(`make_demo_video`, `recording_filter`, demo 로그 패널)는 그대로 동작해야 한다.

## Acceptance criteria

- [x] 알람 사이클 녹화가 manual 녹화와 같은 스키마의 `frame_meta.jsonl` 을 attempt 의 recording 폴더에 쓴다; manual 경로 출력 규약은 무변경.
- [x] manifest 에 `episode_id` / `attempt_seq` / completeness 가 추가되고 기존 manifest 소비자 테스트가 무변경으로 통과한다.
- [x] 사이드카 실패(1회 경고 후 영구 비활성)에도 녹화는 계속되고 Episode 는 보존되며 completeness 에 사유가 남는다.
- [x] 커서 좌표는 기록만 한다 - Action 이나 의도 claim 을 만들지 않는다.
- [x] spec 테스트 4, 5(사이드카 실패 부분)를 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- `monitor/frame_meta.py` - `FrameMetaRecorder` 신설(공용 래퍼). `wrap(capture_fn)` 이
  캡처 함수를 감싸 프레임마다 사이드카 1줄을 남기고, `completeness(frames)` 가 manifest
  용 요약을 낸다. 수집 실패는 **1회 경고 후 영구 비활성** + 사유 보존이다(기존
  `manual_record._make_capture_fn` 은 프레임마다 경고를 찍어 20fps 에서 콘솔을 도배했다).
  가림 판정 핸들은 창에서 직접 뽑고(`_handles_of`), 못 뽑으면 빈 집합 = `unknown` 판정이다.
- `manual_record.py` - `_make_capture_fn` 삭제, 공용 래퍼로 교체. 출력 규약(`frame_meta.jsonl`
  경로/스키마/`seq_%04d`/`t_sec`)은 무변경.
- `monitor/recording.py` - `manifest_extra_fn` 옵션 하나만 추가. 녹화기가 Episode 개념을
  알 필요는 없으므로 dict 를 만드는 쪽은 호출부이고, 종료 시점에 불러 최종값을 반영한다.
  실패해도 기본 manifest 는 쓰인다.
- `monitor/cycle.py` - 본 녹화가 같은 래퍼를 `capture_fn` 으로 받고, manifest 에
  `episode_id`/`attempt_seq`/`capture_completeness` 를 additive 로 싣는다. teardown 의
  녹화 중지 직후 사이드카 핸들을 닫는다. `run_alarm_cycle(..., episode_id=)` 추가.

**테스트** `monitor/test_frame_meta_recorder.py` 5개 (신규). spec 테스트 4, 5(사이드카
실패) 커버. 회귀: monitor 524 / recording_filter+workflow_extract+workflow_4 269 /
manual_record 47 통과.

**spec 과 다르게 한 점 / 판단**
- 사이드카와 manifest 확장을 `episode_collect_enabled` **게이트 뒤**에 뒀다. 티켓 본문은
  게이트를 명시하지 않지만, 프레임마다 Win32 조회(WindowFromPoint x5 + GetCursorPos +
  GetForegroundWindow)가 붙는 상시 비용이고 구현 프롬프트의 규약이 "새 플래그는 기본 off,
  off 면 기존 동작이 byte 단위로 그대로" 이기 때문이다. off 동작을 고정하는 테스트를 함께 뒀다.
- `capture_completeness.meta_records` 는 **샘플 수**(= manifest 의 `sampled_count`)이고
  `frames` 는 **저장 프레임 수**다. 변화 없는 샘플은 저장되지 않으므로 둘은 정상적으로
  어긋나며, 그래서 분석 단계가 seq 가 아니라 `t_sec` 으로 조인한다. 테스트가 이 관계를 고정한다.
- cycle 이 캡처 람다를 주입하게 되면서 녹화 캡처가 `cycle.capture_window` 를 거친다
  (수집 on 일 때). 같은 util 함수라 동작은 같지만, 테스트는 실제 경로를 patch 하도록 맞췄다.
