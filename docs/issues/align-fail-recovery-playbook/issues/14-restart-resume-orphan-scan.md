# 14 — Restart resume and orphan scan

Type: task
Status: ready-for-agent
Blocked by: 10
Spec: [spec.md](../spec.md) (Recovery Episode lifecycle and persistence)

## What to build

모니터 프로세스가 재시작돼도 Episode 식별이 깨지지 않는다. 재시작 후 같은 알람이 오면 fingerprint
가 **완전히** 일치할 때만 열린 Episode 를 재개한다. 일부라도 다르면 이전 Episode 를 reason 과 함께
`incomplete` 로 닫고 새 Episode 를 만든다. 첫 poll 에서 capture tree 를 한 번 스캔해, 열려 있지만
알람 목록에 없는 Episode 를 `incomplete(alarm_gone_during_restart)` 로 닫는다. 장비→Episode 맵은
메모리에만 있고 디스크에서 재구성하는 경로는 이 스캔뿐이다.

## Acceptance criteria

- [ ] 재시작 후 같은 fingerprint 알람은 열린 Episode 를 재개하고 `attempt_seq` 를 이어 간다.
- [ ] fingerprint 가 하나라도 다르면 이전 Episode 는 `incomplete` + reason, 새 Episode 가 생성된다.
- [ ] 첫 poll 스캔이 알람 없는 open Episode 를 `incomplete(alarm_gone_during_restart)` 로 닫는다.
- [ ] 스캔은 깨진 Episode 파일이나 큰 tree 에서 모니터를 죽이지 않는다(경고 후 건너뜀).
- [ ] spec 테스트 3 을 덮는다.
