# 14 — Restart resume and orphan scan

Type: task
Status: resolved
Blocked by: 10
Spec: [spec.md](../spec.md) (Recovery Episode lifecycle and persistence)

## What to build

모니터 프로세스가 재시작돼도 Episode 식별이 깨지지 않는다. 재시작 후 같은 알람이 오면 fingerprint
가 **완전히** 일치할 때만 열린 Episode 를 재개한다. 일부라도 다르면 이전 Episode 를 reason 과 함께
`incomplete` 로 닫고 새 Episode 를 만든다. 첫 poll 에서 capture tree 를 한 번 스캔해, 열려 있지만
알람 목록에 없는 Episode 를 `incomplete(alarm_gone_during_restart)` 로 닫는다. 장비→Episode 맵은
메모리에만 있고 디스크에서 재구성하는 경로는 이 스캔뿐이다.

## Acceptance criteria

- [x] 재시작 후 같은 fingerprint 알람은 열린 Episode 를 재개하고 `attempt_seq` 를 이어 간다.
- [x] fingerprint 가 하나라도 다르면 이전 Episode 는 `incomplete` + reason, 새 Episode 가 생성된다.
- [x] 첫 poll 스캔이 알람 없는 open Episode 를 `incomplete(alarm_gone_during_restart)` 로 닫는다.
- [x] 스캔은 깨진 Episode 파일이나 큰 tree 에서 모니터를 죽이지 않는다(경고 후 건너뜀).
- [x] spec 테스트 3 을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- `recovery_episode.EpisodeTracker.resume_from_disk(current_fingerprints)` - 프로세스당
  1회(`_scanned` 플래그)만 capture tree 를 `rglob(recovery_episode.json)` 로 훑는다.
  `state="open"` 인 것 중 fingerprint 가 이번 poll 의 알람과 **완전히** 일치하면 메모리
  맵에 다시 올려 재개하고, 아니면 `incomplete(alarm_gone_during_restart)` 로 닫는다.
  스캔 전체와 파일 하나가 각각 try/except 라 깨진 JSON 은 경고 후 건너뛴다(삭제 없음).
- `begin_attempt` - 같은 장비에 열린 Episode 가 있어도 fingerprint 가 다르면
  `incomplete(fingerprint_changed)` 로 닫고 새 Episode 를 연다. 재시작이 아니라 같은
  프로세스 안에서 알람이 갈리는 경우(cooldown 재시도 사이에 알람 시각이 바뀜)를 덮는다.
- `_mark_episode_incomplete()` - attempt 사유(`attempt_<n>:<reason>`)와 Episode 수준
  사유를 구분해 적는다.
- `align_fail_monitor` - `process_fail_rows` 는 `by_tool` 을 만든 직후
  `resume_from_disk(fingerprints)` 를, 메인 루프의 빈-poll 분기는 `resume_from_disk(())`
  를 부른다. **빈 분기에도 넣어야** 재시작 직후 첫 poll 이 비어 있을 때 고아 Episode 가
  영영 열린 채 남지 않는다(그 분기는 process_fail_rows 를 거치지 않는다).

**테스트** `test_recovery_episode.py` +4 (총 14): 재시작 재개(attempt 1 -> 2),
fingerprint 변경 시 이전 Episode 닫힘 + 새 Episode, 알람 없는 open Episode 의
`alarm_gone_during_restart`, 깨진 파일에도 스캔이 계속됨. spec 테스트 3 커버.
회귀: monitor 519 / recording_filter+workflow_extract 233 / workflow_4 36 통과.

**판단**
- 스캔에 개수 상한은 두지 않았다. align_images 트리는 (eqp, class, recipe, tag) 조합이라
  현실적 규모에서 rglob 1회가 문제되지 않고, 상한을 두면 "일부만 닫혔다" 는 더 나쁜
  상태(어느 것이 남았는지 모름)가 된다. 요구는 '죽지 않는다' 이고 그건 예외 처리로 만족한다.
