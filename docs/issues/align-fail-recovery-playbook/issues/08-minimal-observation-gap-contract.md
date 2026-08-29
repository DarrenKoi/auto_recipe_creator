# 부족한 Recovery 관측 신호의 최소 보강 경계를 결정한다

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 01, 02

## Question

[`현재 관측 신호의 Recovery 의미를 확정한다`](01-current-observation-semantics.md)와
[`대표 Recovery Trace corpus 를 선택한다`](02-representative-trace-corpus.md)를 근거로,
`recovery_playbook.json` 이 `recovered`를 주장하고 3상태 Recovery Guard를 평가하는 데 필요한
최소 관측 계약은 무엇인가? 현재 성공 corpus가 0건이므로, 앞으로 생길 첫 성공 Episode를
검증 가능한 Trace로 남기는 수집 계약도 함께 확정한다.

- Recipe Monitor 분자 증가, Assist quality, alarm 해제 중 어떤 조합을 `recovered`의 필수
  Verification으로 삼을지
- matcher/점유/화면 사용 가능성 중 어떤 pre-action 상태를 durable Recovery Guard로 남길지
- detector 실패와 결측을 어떤 구조로 `unknown` 및 escalation 근거에 연결할지
- retry attempt, action provenance, Verification, Episode Outcome을 어떤 식별자로 연결할지
- 첫 성공 Episode에서 반드시 남길 원본·구조화 산출물과 수집 실패 시 판정 경계
- 현재 신호로 충분한 부분과 추가 관측이 필요한 부분의 경계를 확정한다.

새 detector 알고리즘이나 production 클릭 구현은 설계하지 않는다.

## Answer

### `recovered` Verification

`recovered`는 개별 Action이나 attempt가 아니라 Recovery Episode의 Outcome이다. 다음 우선순위를
적용한다.

1. Primary는 Action 후 Assist `Measurement`의 정상화 또는 측정 재개다. 판독 가능한
   post-baseline 변화와 정상 row, red 없음이 근거여야 한다.
2. `Measurement`가 읽을 수 없을 때만 Recipe Monitor 분자의 엄격 증가를 fallback으로 쓴다.
3. Alarm 해제는 보조 증거일 뿐 단독으로 `recovered`를 만들지 않는다.
4. cursor idle, Remote Monitoring 창 종료, `corrected`, `completed`, OK click은 성공
   Verification이 아니다.

첫 Trace에서는 자동 판독이 부족하면 엔지니어가 근거 frame을 가리켜 `Measurement`를 annotation할
수 있다. annotation은 관측 판정의 출처이지 독립적인 성공 신호가 아니다. 현재 코드의 Assist 기본
off/분자 primary 동작은 이 정본 계약과 다르며, 첫 qualified Trace를 수집하기 전에 맞춰야 할
구현 gap이다.

### Durable Recovery Guard

Action 선택을 바꾸는 다음 세 종류만 저장한다.

1. 화면 관측 가능성 및 occlusion
2. 점유 및 제어 가능성
3. SEM mode와 align key 가시성·유일성

각 Guard는 `true` / `false` / `unknown`, 판정 이유, 관측 시각, evidence reference를 가진다.
OK control 존재는 Episode Guard가 아니라 해당 Action의 실행 전제조건이다. 실제 Trace가 다른
사전 상태의 필요성을 입증하기 전에는 Guard를 추가하지 않는다.

### Episode와 event 연결

- 하나의 ALID=9006 활성 구간에 하나의 opaque `episode_id`를 부여한다.
- 같은 알람을 처리하는 retry는 `attempt_seq`, attempt 안의 관측·Action·Verification은 단조 증가
  `event_seq`로 정렬한다.
- 파일 경로나 timestamp는 identity가 아니다. EQP, recipe, alarm 시각은 Episode 속성이다.
- Alarm 해제 후 다시 발생하면 같은 EQP/recipe라도 새 Episode다.
- Alarm이 해제돼도 승인된 Verification이 없으면 Outcome은 `unknown`이다.

### `unknown`과 수집 실패

- 필수 Guard가 `unknown`이면 자동 Action을 선택하지 않고 escalation한다.
- Action 후 Verification이 `unknown`이면 evidence를 보존하고 Episode Outcome도 `unknown`으로
  둔다.
- `unknown`을 `false`나 다른 Action branch로 바꾸지 않는다.
- 필수 원본이 빠진 Trace는 삭제하지 않고 `incomplete`로 보존하지만 승인 corpus에는 넣지 않는다.
  재생성 가능한 파생 파일이 없다는 이유만으로 원본 Trace를 폐기하지 않는다.

### 첫 qualified Trace의 필수 evidence bundle

1. `episode_id`와 `attempt_seq`가 포함된 capture manifest
2. Recovery Action 전부터 Verification 후까지의 원본 JPEG
3. `frame_meta.jsonl`의 시간축, window geometry, occlusion, cursor 관측
4. 위 세 Recovery Guard 판정과 근거 frame
5. Recovery Action 및 Measurement/분자 Verification 판정과 근거 frame

`change_events.json`, `region_map.json`, `interaction_timeline.json`, `workflow.json`은 위 원본에서
만드는 파생물이므로 capture 시 필수는 아니다.

Episode 정본은 작은 `recovery_episode.json` 하나다. `episode_id`, EQP, recipe, alarm 시각,
attempt 목록, Guard/Action/Verification evidence 경로, 최종 Outcome과 `complete`/`incomplete`
상태만 저장한다. 기존 `recording_manifest.json`과 `run_state.json`은 각각 capture/attempt
세부를 계속 소유하며, Episode 정본은 내용을 복제하지 않고 경로로 연결한다.

현재 alarm-cycle 녹화의 `frame_meta.jsonl` 부재, Episode/attempt/event 식별자 부재, Assist
판정의 structured decision 부재, alarm-clear의 비영속성은 첫 qualified Trace 수집 전 보강해야 할
경계다. 이 티켓은 관측·수집 계약만 확정하며 detector 알고리즘과 production GUI Action은
설계하지 않는다.
