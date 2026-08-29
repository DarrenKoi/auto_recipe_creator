# 20 — Pure Recovery Action classifier

Type: task
Status: ready-for-agent
Blocked by: 19
Spec: [spec.md](../spec.md) (Recovery Action vocabulary and classification)

## What to build

workflow extraction 의 grouping 뒤에 순수 분류기 하나를 붙인다. 입력은 grouped step, 그 step 의 대표
전·후 frame 기계 판독(matcher 점수, PM 배율 OCR, OK 라벨 OCR), 시퀀스 위치. 출력은 vocabulary 버전,
kind 또는 `unclassified`, 정규화 parameter, source ∈ {rule, cv, ocr}, evidence ref, hint, `inferred`.
Outcome/Verification 은 시그니처에 없고 좌표 필드도 없다.

두 pass 로 결정론적으로 돈다: 먼저 이웃에 의존하지 않는 control click 과 PM diff, 그 다음 live-image
double-click 을 다음 semantic step 으로 분류한다(다음이 confirm 이면 reposition, 아니면 pan). after-frame
matcher 가 calibrated ensemble 임계를 넘을 때만 cv corroboration 이고 before frame 은 쓰지 않는다.
`pan_fov` 는 bin 된 direction/magnitude 만 내고 원 `(dx, dy)` 는 provenance 에 남긴다.

산출물은 기존 `workflow.json`/`workflow.md` 옆의 additive 분류 파일이고, 콘솔 digest 줄
`<seq> <kind>(<params>) src=<rule|cv|ocr> inferred=<0|1> t=<sec>` 을 낸다.

## Acceptance criteria

- [ ] 규칙 5개(reposition/pan/set_magnification/confirm/unclassified) 각각 positive 1 + 인접 negative 1 테스트, next-step confirm 문맥과 PM same/diff/unknown 포함.
- [ ] 두 pass 순서가 고정이고 같은 입력에 같은 출력이다.
- [ ] 같은 bin 에 떨어지는 두 offset 은 동일 parameter 이고, 원 offset 은 provenance 에만 있다.
- [ ] 함수 시그니처에 Outcome/Verification 이 없고 출력에 screen 좌표가 없다.
- [ ] 기존 `workflow.json`/`.md` 와 flat step 은 무변경이다.
- [ ] 18 번의 실제 Episode fixture 의 step 으로 smoke 가 돈다.
- [ ] spec 테스트 10, 11 을 덮는다.
