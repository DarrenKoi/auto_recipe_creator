# 12 — Frame metadata for automatic recordings

Type: task
Status: ready-for-agent
Blocked by: 11
Spec: [spec.md](../spec.md) (Capture and durable Recovery Guards)

## What to build

manual recording 런처에만 붙어 있는 프레임 사이드카(창 rect, 전면 창 제목, 가림 여부, 로컬 커서)
capture 래퍼를 공용 모듈로 올려, 알람 사이클의 `RecordingSession` 도 같은 래퍼를 `capture_fn`
주입점으로 쓴다. 녹화기 자체는 포크하거나 바꾸지 않는다.

capture manifest 는 `episode_id`, `attempt_seq`, capture completeness 를 additive 로 얻는다. 새 필드를
모르는 기존 소비자(`make_demo_video`, `recording_filter`, demo 로그 패널)는 그대로 동작해야 한다.

## Acceptance criteria

- [ ] 알람 사이클 녹화가 manual 녹화와 같은 스키마의 `frame_meta.jsonl` 을 attempt 의 recording 폴더에 쓴다; manual 경로 출력 규약은 무변경.
- [ ] manifest 에 `episode_id` / `attempt_seq` / completeness 가 추가되고 기존 manifest 소비자 테스트가 무변경으로 통과한다.
- [ ] 사이드카 실패(1회 경고 후 영구 비활성)에도 녹화는 계속되고 Episode 는 보존되며 completeness 에 사유가 남는다.
- [ ] 커서 좌표는 기록만 한다 - Action 이나 의도 claim 을 만들지 않는다.
- [ ] spec 테스트 4, 5(사이드카 실패 부분)를 덮는다.
