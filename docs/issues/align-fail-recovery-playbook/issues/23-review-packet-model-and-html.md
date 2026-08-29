# 23 — Review Packet model and HTML

Type: task
Status: ready-for-agent
Blocked by: 21, 22
Spec: [spec.md](../spec.md) (Review Packet, annotations, and approval records; Further Notes)

## What to build

Episode 하나 × candidate rule 하나 × effective annotation cut 에서 결정론적으로 파생되는 Review Packet
모델: 대표 before/after frame 참조, Guard reading, 정규화 step, Verification, Outcome, branch reason, scope,
열린 질문(21 번의 파생 그대로). 정본 파일이 아니라 view 다.

이 모델을 기존 오프라인 렌더링 패턴으로 self-contained 로컬 HTML 로 낸다. 읽기 전용이고 Episode-relative
evidence 를 링크/embed 하며 어떤 정본도 쓰지 않는다. 오피스 승인 전에는 소비자가 없으므로 증분 4 안에서
마지막에 짓고 첫 승인이 보일 때까지 밀려도 된다 - 21 번의 reviewer 와 `annotations.jsonl` 이 먼저다.

## Acceptance criteria

- [ ] 같은 (Episode, rule, cut) 에 같은 모델이 나온다.
- [ ] HTML 은 외부 자원 없이 열리고 정본 파일을 만들거나 바꾸지 않는다.
- [ ] 열린 질문이 21 번 파생과 일치하고, 해소된 질문은 packet 에서 사라진다.
- [ ] spec 테스트 13(packet 부분)을 덮는다.
