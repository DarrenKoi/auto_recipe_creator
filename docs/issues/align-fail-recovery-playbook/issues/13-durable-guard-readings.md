# 13 — Durable Guard readings

Type: task
Status: ready-for-agent
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

- [ ] 각 reading = `true`/`false`/`unknown`, reason, observation time, Episode-relative evidence ref. Guard kind 는 셋뿐이고 추가 경로가 없다.
- [ ] 읽기/파싱/stale 사이드카/asset 없음/mode 미판독/capture 실패/matcher 예외는 전부 `unknown` 이다 - `false` 나 `true` 로 새지 않는다.
- [ ] feasibility `candidate`/`ambiguous` 에서 Guard 3 은 `true` 가 아니다; 읽은 mode 는 detail 에만 있다.
- [ ] OK-control availability 는 precondition 기록으로 저장되고 Guard 목록에 없다.
- [ ] 기존 matcher/occupancy/share 테스트가 무변경으로 통과한다.
- [ ] spec 테스트 5, 6 을 덮는다.
