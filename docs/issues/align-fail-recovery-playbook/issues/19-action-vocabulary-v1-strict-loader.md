# 19 — Action vocabulary v1 JSON and strict loader

Type: task
Status: ready-for-agent
Blocked by: 18
Spec: [spec.md](../spec.md) (Recovery Action vocabulary and classification)

## What to build

Recovery Action 닫힌 어휘 v1 을 버전 붙은 JSON 데이터 `recovery_actions.v1.json` 으로 두고, workflow_4
Playbook 패키지의 strict 로더로 읽는다. 다섯 kind 의 parameter 스키마, precondition, 필요 binding
capability 를 담는다: `reposition_to_align_key`(기호 target `matched_recipe_box_center`), `pan_fov`
(`direction` 8방위 + `magnitude` small/medium/large, bin 임계값도 이 파일에), `set_magnification`
(양수 `mag`), `confirm_align`(precondition `ok_control_available`), `handoff_to_engineer`. 연속값은 어떤
parameter 계약에도 들어가지 않는다.

## Acceptance criteria

- [ ] 파일에 5 kind + parameter 스키마 + precondition + capability + pan bin 임계값이 있고 로더가 이를 검증한다.
- [ ] 파일 없음/깨짐/kind 중복/미지원 schema 버전은 예외로 hard fail 한다 - 조용히 전부 `unclassified` 가 되는 경로가 없다.
- [ ] 버전 N 로더는 N 밖의 kind 를 `unclassified` 로 내고, v2 파일을 옆에 두면 두 버전이 함께 읽힌다.
- [ ] 로더는 workflow_3 를 import 하지 않고, workflow_3 는 경로로 로드한다.
- [ ] spec 테스트 12 를 덮는다.
