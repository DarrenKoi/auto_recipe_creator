# 대표 Recovery Trace corpus 를 선택한다

Type: task
Mode: HITL
Status: resolved
Blocked by: none

## Question

정규화와 분기 설계를 실제 근거에 대조할 수 있도록, 사용 가능한 Recovery Trace 중 가장 작은
대표 corpus 를 선택한다.

- 관측 가능한 성공 근거가 있는 Episode 하나 이상
- 알려진 엔지니어 행동 경로가 다르다면 각 경로를 보여주는 Episode
- 있다면 `escalated` / `aborted` / `unknown` Episode
- 가능한 경우 recording manifest, `frame_meta.jsonl`, `interaction_timeline.json`, `workflow.json`,
  대표 전·후 프레임

임의의 개수를 채우지 말고, 현재 알려진 경로와 충돌을 드러낼 최소 세트만 선택한다. 원본을
저장소에 복사할 수 없으면 분석 가능한 경로와 접근 제약만 기록한다.

## Comments

### 2026-08-29 — 집 환경 접근 제약

- 사용자는 현재 집에서 작업 중이며 오피스의 Recovery Trace 저장소에 접근할 수 없다.
- 현재 checkout과 로컬 `Codes`/`Documents`, Git history에는 `recording_manifest.json`,
  `frame_meta.jsonl`, `interaction_timeline.json`, `workflow.json`을 갖춘 실제 Trace가 없다.
- 로컬 `align_fail_cycles.csv`에는 Recovery Action 전에 끝난 `rcs_unavailable` 및
  `rcs_recovery_launch_error` abort만 있다. 성공 Trace 대체물로 쓰지 않는다.
- synthetic matcher/debug 산출물이나 `workflow_4` demo run도 실제 Recovery Episode로
  승격하지 않는다.
- 이 확인 시점에는 성공 사례 유무가 확정되지 않아 claimed 상태를 유지했다. 이후 사용자
  확인으로 아래 Answer에서 해결했다.

오피스 재개 체크리스트:

1. 실제 `ALIGN_IMAGES_DIR`을 확인한다.
2. `captured_img_from_rcs/<tag>/recording/` 및 `_manual/<tag>/recording/` 후보를 나열한다.
3. 각 후보에서 pre-action Align Fail, Recovery Action provenance, post-action Measurement
   정상화 또는 측정 재개 근거가 같은 Episode에 연결되는지 확인한다.
4. 성공 Episode 하나를 먼저 선택하고, 행동 경로나 Outcome이 실제로 다를 때만 후보를
   추가한다. 원본은 복사하지 않고 절대 경로와 접근 제약만 기록해도 된다.

## Answer

사용자 확인 결과, 현재까지 성공한 Recovery Episode 자체가 없다. 집에서 오피스 저장소에
접근할 수 없는 일시적 문제가 아니라 아직 선택할 성공 corpus가 존재하지 않는 상태다.

따라서 현재 대표 corpus는 다음처럼 확정한다.

- `recovered`: 0건
- 서로 다른 성공 행동 경로: 0건
- 실제 Recovery Action을 포함한 `escalated` / `unknown`: 0건
- 로컬 pre-action abort: `rcs_unavailable`, `rcs_recovery_launch_error`가 있으나 Recovery
  Playbook 근거 corpus에서는 제외한다.

synthetic matcher 자료, `workflow_4` demo, `corrected`/`completed` 상태도 성공 Trace로
대체하지 않는다. 실제 성공 근거가 생기기 전에는 Playbook candidate의 offline replay나
승인을 주장할 수 없다.

첫 corpus는 앞으로 생기는 **첫 번째 검증 가능한 성공 Episode 한 건**으로 시작한다. 이후
Trace는 행동 경로, 관측 Guard, Verification 또는 Outcome이 실제로 다를 때만 추가한다.
첫 성공 Episode에 필요한 관측·식별자·산출물 계약은
[`부족한 Recovery 관측 신호의 최소 보강 경계를 결정한다`](08-minimal-observation-gap-contract.md)에서
확정한다.
