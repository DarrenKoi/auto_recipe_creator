# 18 — Office gate: first real Episode

Type: task
Status: ready-for-human
Blocked by: 10, 11, 12, 13, 14, 15, 16, 17
Spec: [spec.md](../spec.md) (Delivery boundary and rollout order; Testing Decisions - office acceptance)

## What to build

코드가 아니라 오피스 실행이다. 10–17 을 push 하고 오피스에서 pull 한 뒤 production 알람 모니터를
Episode 수집 on 으로 띄운다. 실제 ALID=9006 알람 1건이 지나가면 `captured_img_from_rcs/<tag>/` 에
`recovery_episode.json` 과 `attempt_<n>/` 폴더가 생겼는지 확인하고 **텍스트만** 집으로 가져온다. 그
파일들이 19–27 의 producer fixture 가 된다 - 손으로 적은 consumer-shape fixture 는 이후 받지 않는다.

Outcome 은 무관하다. `incomplete` 여도 gate 는 통과한다. 파일이 아예 생기지 않으면 10 번으로 돌아간다.

## Acceptance criteria

- [ ] `captured_img_from_rcs/<tag>/recovery_episode.json` 과 `attempt_<n>/` 가 존재하고 `[DIGEST] episode` 한 줄이 복사됐다.
- [ ] 집으로 가져온 것: digest, `recovery_episode.json`, capture manifest, `frame_meta.jsonl` 앞부분, Guard/Measurement/numerator record. 이미지·원본 프레임은 없다.
- [ ] 가져온 파일이 리포의 fixture 위치에 들어갔다(장비/레시피 식별자 마스킹 여부는 사용자가 정한다).
- [ ] 오피스에서 본 것 중 스펙과 어긋난 점(경로, 필드, 예외)이 이 티켓의 Comments 에 텍스트로 남았다.
