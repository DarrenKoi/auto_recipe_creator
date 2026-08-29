# Recovery Action 어휘와 step 분류 계약을 결정한다

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 04, 05, 07

## Question

Recovery Playbook 이 참조할 수 있는 Recovery Action 의 닫힌 어휘는 무엇이고, `workflow_extract`
가 만든 평평한 step(`double_click` / `select_from_dropdown` / `type_text` / `click_repeat` /
`click`)과 전·후 frame 의 기계 판독(matcher, PM 배율 OCR, OK 버튼 locator)을 그 어휘로 바꾸는
분류 계약은 어떤 모양이어야 하는가?

[`workflow_4` compiler 경계](07-workflow4-compiler-boundary.md)가 요구한 것이다: shadow digest 의
"엔지니어가 다르게 했다" 는 boolean 으로는 집에서 [병합](04-trace-merge-and-branch-rules.md)을
할 수 없으므로, 엔지니어의 관측된 행동이 **정규화된 도메인 Action 열**(`unclassified` 를 1급
값으로)로 텍스트에 실려야 한다. 어떤 분류가 기계적으로 가능하고 어떤 것이
[Recovery Annotation](05-engineer-review-prototype.md) 으로 가야 하는지, parameter 정규화, 어휘
확장 경로, digest 한 줄 형식, 분류기가 사는 위치를 확정한다.

## Answer

사용자 위임(2026-08-29)에 따라 추천안을 채택하되
[opencode 토론](../../../opencode/2026-08-29-recovery-action-vocabulary-debate.md)으로 압박했다.
두 라운드에서 분류 규칙의 입력이 두 번 정정됐다.

### Recovery Action 닫힌 어휘 v1

| kind | parameters | 바인딩 (workflow_3) | Trace 에서의 관측 신호 |
|---|---|---|---|
| `reposition_to_align_key` | `target=matched_recipe_box_center` (기호값, 좌표 아님) | `move_to_point(best_xy)` - best_xy 는 실행 시 matcher 가 계산 | live box `double_click` 의 **다음 step 이 `confirm_align`** + after frame 의 matcher corroboration |
| `pan_fov` | `offset=(dx, dy)` live-box 비율 | `move_to_point` (비-key 점) | live box `double_click` 뒤에 OK 없이 항법이 이어짐 |
| `set_magnification` | `mag` 절대값 | `MagnificationControl` (PM 드롭다운; wheel 단계는 바인딩의 선택) | PM 배율 OCR before/after diff 또는 PM 컨트롤의 R2 |
| `confirm_align` | 없음; precondition `ok_control_available` | `click_screen(ok_xy)` (VLM locator) | `ui_control` click 의 OCR 라벨 OK/확인 |
| `handoff_to_engineer` | 없음 | terminal `handoff` ([compiler 경계](07-workflow4-compiler-boundary.md)) | Trace 에서 오지 않는 시스템 전용 action |

어휘가 아닌 것: RCS 수준 동작(tool 열기/닫기, 화면 공유 요청)은 `run_correction` 을 감싸는 cycle 의
고정 frame 이다. 비-PM 컨트롤의 `type_text`/`select_from_dropdown`, 미지 컨트롤 `click` 은
`unclassified` 다. mode(OM/SEM)는 Guard 이지 parameter 가 아니다.

### 분류 계약

분류기는 `workflow_extract` grouping 뒤의 **순수 함수**
`f(step, 전·후 frame 의 기계 판독, 시퀀스 위치) -> {action_kind ∈ 어휘 ∪ {unclassified}, parameters,
classification_source ∈ {rule, cv, ocr}, evidence_ref, hint}` 다.

- **Outcome 과 Verification 은 입력이 아니다.** kind 가 결과를 부호화하면 [병합 규칙](04-trace-merge-and-branch-rules.md)이
  `unresolved` 로 보내야 할 것을 분류기가 인과 분기로 날조하고, 늦게 오는 Verification 이 이미
  확인된 분류를 뒤집어 append-only 를 깬다. "그 reposition 이 성공한 것이었는가" 는 Verification 의 일이다.
- 규칙 표(위에서 아래로 첫 일치):
  1. `live_image` `double_click` 이고 다음 step 이 `confirm_align` -> `reposition_to_align_key`
     (src=rule). after frame 에서 matcher 가 ensemble 의 calibrated threshold 를 넘으면 src=cv 로
     corroborated. before frame 의 matcher 는 쓰지 않는다 - 회복 Trace 는 key 가 안 잡혀서 생긴다.
  2. `live_image` `double_click` 뒤에 OK 없이 항법(다른 `double_click`, 배율 변경)이 이어짐 ->
     `pan_fov(offset)`.
  3. PM 배율 diff 는 3값이다. `same` -> 동작 없음; `diff` + 값 판독 -> `set_magnification(mag)`;
     `diff` 인데 값 없음 또는 `unknown` -> `unclassified(hint=magnification_change)`.
  4. `ui_control` click 의 OCR 라벨이 OK/확인 -> `confirm_align`.
  5. 그 외 -> `unclassified`. target 라벨, `target_kind`, `value` 를 그대로 실어 엔지니어가 무엇을
     분류할지 보게 한다.
- 분류기는 **좌표를 방출하지 않는다.** 오분류가 조작된 점을 바인딩할 수 없다.
- 기계 분류는 제안이다. [검토 묶음](05-engineer-review-prototype.md)대로 inferred 는 확인 질문,
  `unclassified` 는 분류 질문(닫힌 어휘 또는 "기타")을 받는다. "기타" 는 action 이 되지 않고
  `unresolved` 어휘 제안으로 남는다. 병합 입력이 되는 kind 는 행동 수행자가 확인한 것뿐이다.
- 캘리브레이션은 집/오피스 루프로 한다: 오피스 digest 가 step 마다 (제안 kind, after-frame matcher
  점수, 확인 kind) 를 내고, 집에서는 그 불일치 표로 threshold 를 조정한다. 초기값은 ensemble 의
  Youden threshold 다.

### 어휘의 위치와 버전

어휘 표(kind, parameter 스키마, precondition)는 Python import 가 아니라 **버전 붙은 JSON 데이터**
`poc/workflow_4/playbook/recovery_actions.v1.json` 이며, evaluator 와 workflow_3 분류기가 경로로
로드한다. 표가 없거나 못 읽으면 분류는 **hard fail** 이다(조용히 전부 `unclassified` 가 되는 digest 는
없다). playbook 과 분류 출력 모두 `action_vocabulary_version` 을 stamp 한다. bump 는 새 파일을 옆에
추가하고 구버전은 계속 읽히며, 버전 N 을 로드한 분류기는 N 밖의 kind 를 `unclassified` 로 낸다.
미러의 opt-in soft import 와는 무관하다.

### Digest 한 줄

```text
<seq> <kind>(<params>) src=<rule|cv|ocr|annotation> inferred=<0|1> t=<sec>
```

extraction 시점에 확정되며 Verification 을 기다리지 않는다. [승격 gate](06-offline-replay-approval-gate.md)
shadow digest 의 (c) 항목이 같은 형식이다.

### 검증

Mac: 분류기는 순수라 합성 step/판독으로 단위 테스트한다. 오피스: 첫 Episode 에서 digest 와 불일치
표가 텍스트로 돌아온다. 현재 Episode 0건.

