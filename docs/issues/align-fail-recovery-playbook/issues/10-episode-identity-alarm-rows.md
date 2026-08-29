# 10 — Episode identity through alarm-row processing

Type: task
Status: ready-for-agent
Blocked by: None — can start immediately
Spec: [spec.md](../spec.md) (Recovery Episode lifecycle and persistence)

## What to build

알람 모니터의 fail-row 처리가 ALID=9006 active interval 하나에 Recovery Episode 하나를 만든다.
Episode 정본 `recovery_episode.json` 은 기존 per-alarm capture 폴더 `captured_img_from_rcs/<tag>/`
(recipe 없는 알람은 `_unregistered/<tag>/`) 루트에 **GUI 작업이 시작되기 전에** 원자적으로 써진다.
cooldown 재시도는 같은 Episode 를 재개하며 `attempt_seq` 를 올리고, 알람이 poll 에서 사라지면
clearance 이벤트를 남기고 Episode 를 닫는다. 이후 같은 장비·레시피의 재발은 새 Episode 다.

Episode 수집은 기본 off 인 플래그 뒤에 있고, off 면 현재 모니터 동작과 기존 테스트가 그대로다.
Mac 에서는 replay CSV 소스(첫 poll 에 알람, 다음 poll 은 빈 목록)로 생성 → 재시도 → clearance 경로를
실장비 없이 돌려 볼 수 있어야 한다.

## Acceptance criteria

- [ ] 같은 알람의 cooldown 재시도는 같은 `episode_id` 를 재사용하고 `attempt_seq` 가 1, 2, 3… 으로 증가해 파일의 ordered attempts 에 남는다.
- [ ] 알람이 poll 에서 사라지면 clearance 이벤트가 기록되고 Episode 가 닫힌다; 같은 EQP/recipe 의 후속 알람은 다른 `episode_id` 를 받는다.
- [ ] `episode_id` 는 UUID 이고 어떤 경로·타임스탬프에서도 재구성되지 않는다. 알람 fingerprint = 장비 + alarm code + recipe + 원 UTC9 는 별도 필드다.
- [ ] 초기 Episode 파일은 첫 GUI step 전에 존재한다. cycle 이 예외로 끝나도 파일은 남고 `incomplete` + reason 이 기록된다(삭제 없음).
- [ ] 저장되는 artifact 경로는 전부 Episode-relative 다. 절대 경로와 `..` 탈출은 로드 시 거부된다.
- [ ] 쓰기는 temp + atomic replace 다. `_unregistered/<tag>/` 도 같은 규약이다.
- [ ] 파일에 schema 버전, observation-contract 버전, `bindings_version`, `execution_mode="live"` + settings snapshot 참조(safe_mode/dry-run 이 provenance 로 보임)가 stamp 된다.
- [ ] 수집 플래그 off 에서 기존 monitor/cycle 테스트가 무변경으로 통과한다.
- [ ] replay CSV 로 Mac 에서 생성 → clearance 경로가 돈다(테스트 또는 문서화된 실행 예).
- [ ] spec 테스트 1(identity 부분), 2, 15/34(경로), 16(incomplete 보존)을 덮는다.
