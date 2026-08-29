# 11 — Attempt-scoped artifact folders

Type: task
Status: ready-for-agent
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

- [ ] cooldown 재시도 2회가 `attempt_1/recording/` 과 `attempt_2/recording/` 으로 갈리고 프레임·manifest 가 섞이지 않는다.
- [ ] prelude 녹화는 `attempt_<n>/recording/prelude/` 에 간다.
- [ ] Episode 파일의 각 attempt 항목이 자기 폴더와 산출물을 Episode-relative 로 가리킨다.
- [ ] `recording_filter` 발견이 `<tag>/recording/` 과 `<tag>/attempt_<n>/recording/` 을 둘 다 찾는다(구 녹화 호환); manual 경로는 무변경.
- [ ] `make_demo_video` 가 코드 변경 없이 attempt 하위 녹화를 찾는다(재귀 발견을 고정하는 테스트).
- [ ] runner journal 위치는 그대로이고 Episode 파일은 run id 로 참조한다.
- [ ] 수집 플래그 off 면 종전 `<tag>/recording/` 그대로다.
- [ ] spec 테스트 1(폴더 부분), 35 를 덮는다.
