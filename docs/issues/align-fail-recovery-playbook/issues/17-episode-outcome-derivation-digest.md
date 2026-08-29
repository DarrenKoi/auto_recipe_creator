# 17 — Episode Outcome derivation and the `[DIGEST] episode` line

Type: task
Status: resolved
Blocked by: 13, 15, 16
Spec: [spec.md](../spec.md) (Structured Verification and Outcome; Evaluator and compiler)

## What to build

workflow_4 의 Playbook 도메인 패키지에 **첫 순수 evaluator 조각** 을 만든다: Guard reading, Measurement
record, numerator 기록, clearance, abort, handoff 기록을 plain data 로 받아 Outcome ∈ {`recovered`,
`escalated`, `aborted`, `unknown`} 을 낸다. workflow_3 를 import 하지 않는다. 24 번이 이 evaluator 를
Guard 평가/rule 선택으로 확장하므로, Verification 우선순위와 Outcome 파생은 여기가 유일한 소유자다.

per-alarm cycle 은 Episode 를 닫을 때 이 함수로 final Outcome 을 `recovery_episode.json` 에 쓰고, 오피스에서
집으로 복사할 한 줄 `[DIGEST] episode …` 를 찍는다. 이 digest 가 18 번 오피스 gate 의 산출물이다.

## Acceptance criteria

- [x] primary Measurement `success` → recovered; `failure` → recovered 아님; `unknown` 일 때만 numerator fallback 을 보고, fallback 은 strictly increasing 연속만 success 다.
- [x] clearance, OK click, `corrected`, runner 완료, cursor idle, 창 닫힘, probable-close 는 단독으로 recovered 를 만들지 못한다.
- [x] 실패/unknown attempt 뒤 qualified recovered attempt 가 오면 Episode Outcome 은 recovered 이고 attempt 이력은 보존된다; 명시 handoff 기록 → escalated; abort latch → aborted; 그 외 unknown.
- [x] `handoff` 라는 노드/상태 이름만으로는 escalated 가 되지 않는다.
- [x] digest 는 한 줄이며 최소 episode id(축약), EQP, recipe, attempts 수, outcome, Guard 3값, verification 경로(primary/fallback/unknown), complete|incomplete(reason) 를 담는다.
- [x] spec 테스트 8(의미 부분), 9, 21 을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- 새 패키지 `poc/workflow_4/playbook/` (`__init__.py`, `outcome.py`). `derive_outcome()` /
  `evaluate_attempt()` / `format_episode_digest()` 는 전부 순수 함수이고 **workflow_3 를
  import 하지 않는다**(AST 로 검사하는 테스트로 고정 - 문서에서 언급하는 것은 허용,
  의존만 금지).
- `monitor/recovery_episode.py`
  - `build_episode_evidence()` 가 attempt 폴더의 record 3종
    (`measurement_verification.json` / `numerator_reads.jsonl` / `guards.json`)을 읽어
    plain data 로 만든다. **파일을 파싱하는 일은 생산자(workflow_3) 쪽**이 한다.
  - `_close()` 가 Episode 를 닫을 때 `derive_outcome` 으로 `outcome` +
    `outcome_detail`(verification_path/reason/deciding_attempt)을 쓰고
    `[DIGEST] episode ...` 한 줄을 찍는다. workflow_4 import 는 지연·보호라 그 계층이
    없어도 알람 처리가 깨지지 않는다(그때 outcome 은 `unknown` 으로 남는다).
  - `finish_attempt` 가 **명시** abort/handoff 기록을 남긴다.

**판정 규약**
- primary = Measurement. `success` -> recovered. `failure` -> recovered 아니며
  **fallback 을 열지도 않는다**(관측된 실패를 카운터 증가로 뒤집으면 화면이 깨진 채로
  성공이 된다). `unknown` 일 때만 fallback.
- fallback = 분자 기록의 **엄격 증가 연속**. `ocr_miss`/`equal_or_decrease`/
  `reground_reset` 이 연속을 끊고, `not_sampled` 는 세지도 끊지도 않는다. 증가가 한 번도
  없는 같은 값 반복은 연속이 아니다. 요구 표본 수는
  `engineer_done_numerator_increase_reads` 를 따른다(설정과 판정이 갈리면 안 된다).
- Episode = 자격 있는 recovered attempt > 명시 abort > 명시 handoff > unknown.
  실패/unknown attempt 뒤의 recovered 가 이기고 attempt 이력은 보존된다.
  abort 가 handoff 보다 앞인 이유는 긴급 해제가 더 종결적인 사실이기 때문이다.
- 알람 해제 / OK 클릭 / `corrected` / runner 완료 / 커서 정지 / 창 닫힘 / 닫기 정황은
  단독으로 recovered 를 만들지 못한다(테스트로 고정).

**테스트** `poc/workflow_4/playbook/test_outcome.py` 13개(`uv run pytest poc/workflow_4`
= 49) + `monitor/test_recovery_episode.py` +7(총 21). spec 테스트 8(의미), 9, 21 커버.

**Mac 실행 확인** - replay 한 바퀴에서 실제로 나온 digest:

```
[DIGEST] episode id=1485c7c0 eqp=MCD916 recipe=RJ1BXXX/Z_RJ1B_CBLHM2_FULL attempts=1 \
  outcome=unknown guards=screen:unknown,occupancy:unknown,align:unknown verify=unknown complete=yes
```

**spec 과 다르게 한 점 / 판단**
- "명시 handoff" 의 생산자를 정해야 했다. 노드/상태 **이름**은 근거가 아니라는 규약을
  지키되 Outcome 이 영원히 `unknown` 이면 오피스 gate 가 의미를 잃으므로, 자동 보정이
  **의도적으로 엔지니어에게 넘긴** outcome status 넷
  (`awaiting_engineer_ok`, `escalated_ambiguous_key`, `escalated_key_not_visible`,
  `escalated_no_ok`)만 명시 handoff 기록으로 승격했다. 사이클 실패나 관전
  (`view_only_observation`)은 승격되지 않는다(테스트로 고정). evaluator 자신은 이 목록을
  모르며 `{"explicit": true}` 기록만 본다 - 목록은 생산자 쪽 판단이다.
- abort 기록은 긴급 해제 래치(`util.abort_switch`)에서 읽는다. 래치는 전역이고 한 번
  걸리면 모니터 루프가 종료되므로 실제로는 그 attempt 하나만 표시된다.
