# 현재 관측 신호의 Recovery 의미를 확정한다

Type: task
Mode: AFK
Status: resolved
Blocked by: none

## Question

현재 `recording/`, `recording_filter`, `workflow_extract`, engineer-done detector, align correction 경로가
실제로 남기는 상태·행동·결과 근거를 목록화해 다음을 구분한다.

- Recovery Guard 로 평가할 수 있는 관측
- Recovery Verification / Outcome 근거로 쓸 수 있는 관측
- provenance 로만 보존해야 하는 추론 이벤트
- 관측 실패 시 `unknown` 을 내야 하는 경계

각 항목에 생산 코드와 산출물 필드를 근거로 연결한다. 새 detector 설계는 하지 않는다.

## Answer

### 판정 원칙

- Recovery Guard 는 Recovery Action **전**에 평가하는 관측 상태다. Action 기록이나 엔지니어
  의도는 Guard 가 아니다.
- Recovery Verification 은 Action **후** 기대 상태 변화가 일어났다는 근거다. 명령을 보냈거나
  GUI 분기를 완주했다는 사실만으로는 Verification 이 아니다.
- Recovery Outcome 은 개별 시도 상태가 아니라 Recovery Episode 의 종료 결과다. 여러 retry 를
  잇지 못하는 현재 attempt 산출물은 그 자체로 Episode Outcome 이 아니다.
- 생산자가 안전을 위해 `candidate` 로 통과시킨 값은 `true` 가 아니다. 읽기 실패, 결측, 오래된
  sidecar, 파싱 실패는 Playbook 에서 `unknown` 이다.

### Recovery Guard 로 평가할 수 있는 현재 관측

| 관측 | 현재 생산 코드와 필드 | Guard 의미 | `unknown` 경계 |
|---|---|---|---|
| Align Fail trigger | `monitor/align_fail_monitor.py`의 alarm row `ALID`, `EQP_ID`, `RECIPE_ID`; `align_fail_cycles.csv`의 `alid`, `eqp_id`, `recipe_id` | `ALID=9006` 발생 여부와 Episode scope 를 정할 수 있다. | alarm 조회/파싱 실패 또는 Episode 와 row 를 잇지 못하면 `unknown`. |
| 관측 화면 사용 가능성 | 수동 `recording/frame_meta.jsonl`의 `t_sec`, `window_rect`, `foreground_title`, `occlusion`, `cursor_screen_xy`, `cursor_in_window` (`monitor/frame_meta.py:156-175`) | 창 기하와 가림 여부를 action 전제조건으로 평가할 수 있다. 로컬 cursor 위치는 엔지니어 의도나 클릭 버튼 상태가 아니다. | `occlusion="unknown"`, cursor/rect 결측, 10초를 넘긴 timestamp join 은 `unknown`. 자동 alarm-cycle 녹화는 현재 sidecar 를 쓰지 않으므로 이 관측이 없다 (`recording_filter/click_detect.py:115-123`). |
| live SEM 영역/레이아웃 | `recording_filter/region_map.json`의 `region_maps[].generation/live_box`; `change_events.json`의 `verdict`, `region`, `occlusion`, `generation` (`recording_filter/filter_recording.py:154-184`) | `live_box` 존재, UI/live-image 영역, 관측 가림 여부를 평가할 수 있다. 이는 Recovery 상태보다 **관측 가능성 Guard**다. | live-box 검출 실패, sidecar 미조인, cursor/프레임 좌표 변환 실패 시 pipeline 은 `candidate` 로 통과시킨다 (`region_gate.py:341-417`). Playbook 값은 `true` 가 아니라 `unknown`. |
| RCS 점유/제어 가능성 | cycle context의 `occupancy`; `view_only_observation`, `corrected_unverified` 변환 (`monitor/cycle.py:907-925`) | 다른 엔지니어 점유 중인지와 action 이 장비에 반영될 수 있는지를 Guard 로 쓸 수 있다. | 점유 `unknown` 은 action 반영 여부 `unknown`; 코드도 `corrected`를 `corrected_unverified`로 강등한다. |
| SEM mode와 key 가시성/유일성 | `CorrectionOutcome.key_decision`, `second_ratio`, `score_gap`, `distinctive`; `history[].stage="paused_match"`의 `mode`, `decision`, `score`, `chamfer`, `best_scale`, `best_xy`, `scale_pinned` (`align/correction.py:102-127`, `:277-301`) | 검증된 frame/template/mode에서 `key_visibility_gate`가 `act`, `fallback_search`, `engineer_review`를 결정한다. | 자산 없음, mode 미확정, capture/matcher 오류는 `unknown`. `engineer_review`는 key 존재와 자동 선택 가능성을 분리해야 한다. |
| OK control 존재 | `CorrectionOutcome.ok_screen_xy`, `error`; status `escalated_no_ok`/`ok_detect_error` (`align/correction.py:395-498`) | reposition 뒤 OK를 진행할 수 있는지의 action Guard 로 쓸 수 있다. | locator 없음, not-found, detector 예외를 서로 구분하며, 어느 경우도 recovery 실패나 성공을 뜻하지 않는다. |

원본 JPEG 자체는 위 Guard detector 들의 provenance 다. 현재 `recording/` 또는
`interaction_timeline.json` 안에는 임의의 화면 상태를 이름 붙은 3상태 Guard 로 저장하는 범용
필드가 없다.

### Recovery Action provenance 로만 보존할 현재 근거

| 산출물 | 실제 필드/규칙 | 확정 의미와 제한 |
|---|---|---|
| `recording/` JPEG + `recording_manifest.json` | 파일명 elapsed-ms; manifest `tag`, `started_epoch`, `capture_source`, `frame_count`, `sampled_count`, sampling/budget 설정, `stop_reason` (`monitor/recording.py:259-283`) | 시간순 화면 provenance 와 capture 완전성 근거다. 프레임 저장은 변화/heartbeat 기반이라 모든 입력 순간을 기록하지 않는다. |
| `change_events.json` | `frame_path`, `prev_frame_path`, `timestamp_sec`, `frame_index`, `change_bbox`, `largest_blob_area_px`, `changed_pixels`와 선택적 gate 필드 | 화면 변화 관측이다. 사람의 action 이나 UI 상태 의미는 아니다. |
| `interaction_timeline.json` click/type | `action`, `coords`, `element`, `element_source`, `target_kind`, `region`, `generation`, `occlusion`, `cursor_source`, `confidence`, `source_frames` (`recording_filter/timeline.py:32-89`) | cursor 위치 + ROI pixel 변화 + OCR/VLM으로 만든 action 후보다. sidecar도 mouse-button hook이 아니므로 `click`은 여전히 추론이다. source/confidence/frame을 함께 보존해야 한다. |
| `probable_close_click` | `confidence=0.35`, `evidence="window_gone + top_right_change + cursor_vlm_missing"`, `replayable=false`, `candidate_box` (`recording_filter/close_click_evidence.py:107-171`) | 닫기 버튼 click의 저신뢰 사후 정황이다. `ClickEvent`나 실행 가능한 Recovery Action으로 승격하지 않는다. |
| `workflow.json` | step `action`, `target`, `target_kind`, `value`, `value_source`, `coords_in_live_box`, `t_sec`, `generation`, `grouping_rule`, `inferred`, `intent`, `count`, `raw_events`, `frame` (`workflow_extract/steps.py:8-42`) | 단일 Trace의 평평한 action 후보 목록이다. R1 `double_click`과 R2 `select_from_dropdown`은 결과/기하 추론이고 `inferred=true`; R3/R4/R5도 Guard/Verification을 만들지 않는다. `raw_events`로 원본 연결을 유지한다. |
| align correction 결과 | `CorrectionOutcome.status/path/key_decision/best_xy/ok_screen_xy/fallback/error/history`; `LiveSearchOutcome.status/final_decision/best/pan_count/history/meta` | matcher 판단과 **명령 경로** provenance 다. `dry_run=true`여도 correction 분기는 결과 status를 만들 수 있고, live action도 post-action readback이 없다. `corrected`, `fallback_match`, `run_status=completed`를 `recovered`로 읽지 않는다. |
| runner/cycle journal | `run_state.json`, `step_<id>.json`, `align_fail_cycles.csv`; `run_status`, `failure_class`, `outcome_status`, `outcome_path`, `key_decision`, `best_xy`, `run_dir` | attempt 실행/오류/인계 근거다. cycle `StepResult.verification_result`는 현재 항상 `null` (`monitor/cycle.py:229-256`). settings snapshot을 함께 봐야 `safe_mode`/dry-run/action gate를 구분할 수 있다. |

`workflow_extract`는 `probable_close_click`과 `replayable=false`를 grouping 입력에서 제거하고
(`workflow_extract/extract_workflow.py:247-260`), `region_map.json`/`change_events.json`이 없으면 R1/R2를
평범한 click으로 degrade 한다. 이 동작은 action 날조를 줄이지만, 결측을 Recovery Guard의
`false`로 바꾸지는 않는다.

### Recovery Verification / Outcome 으로 쓸 수 있는 현재 관측

| 신호 | 현재 구조화 근거 | 판정 |
|---|---|---|
| Recipe Monitor 분자 N의 엄격 증가 | `engineer_done/numerator_decision_NNN.json`: `poll`, `reading`, `sampled`, `value`, `sequence`, `reset_reason`, `assist_unusable_streak`, `fallback_open`, `done` (`monitor/engineer_done_align_adjustment.py:463-490`) | 현재 유일하게 **지속되는 구조화된 성공 Verification 후보**다. 기본 설정은 Assist off, 3회의 엄격 증가를 요구한다. `done=true`는 측정 재개 근거가 될 수 있다. OCR miss/equal/decrease/reground은 성공 근거가 아니며, 읽기 실패는 `unknown`. |
| Assist panel row/red | live `AssistObservation.status/ok_row_count/has_red/reason`; 변경 시 `assist_panel_*_rowsN_redM.jpg` (`engineer_done_align_adjustment.py:587-628`) | usable + 최소 row + red 없음은 live Verification 후보지만 기본 off다. 현재 CV는 세 열을 구분하지 않고 검정 band를 세므로 (`sem_monitor/assist_score.py:87-114`) offline artifact만으로 `recovered`를 확정하지 않는다. structured `done` record도 없다. |
| 로컬 cursor idle | `last_debug.cursor_idle_sec`와 console; 기본 120초 (`engineer_done_align_adjustment.py:263-306`) | 엔지니어가 손을 뗐다는 약한 종료 휴리스틱뿐이다. 측정/품질/align 성공 Verification이 아니며 `recovered` 근거로 쓰지 않는다. 읽기 실패는 `unknown`. |
| Remote Monitoring 창 종료 | `recording_manifest.stop_reason="window_gone"` | 명시적 수동 종료 신호지만 recovery 성공을 뜻하지 않는다. `probable_close_click`을 보조할 provenance이며 성공 근거가 없으면 Outcome은 `unknown`. |
| 자동 correction/cycle status | `corrected`, `awaiting_engineer_ok`, `escalated_*`, `corrected_unverified`, runner `completed/aborted` | `escalated_*`/handoff와 오류는 attempt의 escalation/abort 근거가 될 수 있다. `corrected`와 `completed`는 open-loop 실행 상태라 `recovered`가 아니다. retry들을 Episode로 stitch하지 못하므로 최종 Outcome은 별도 종료 근거가 필요하다. |

### 반드시 `unknown` 으로 남길 경계

1. 프레임/capture/JSON을 읽지 못했거나, sidecar join이 없거나 오래됐거나, VLM/OCR/CV가
   usable 결과를 내지 못한 경우.
2. `recording_filter`가 안전 폴백으로 `candidate`를 준 경우. 이는 “사람 action일 수 있으니
   버리지 않음”이지 Guard `true`가 아니다.
3. `workflow_extract`가 optional 입력 결측으로 R1/R2를 비활성화한 경우. 남은 R5 click은
   모르는 의미를 채운 결과가 아니다.
4. engineer-done detector가 capture/ground/OCR 실패로 `False`를 반환한 경우. 현재 bool API가
   `false`와 `unknown`을 합치므로, Playbook 변환 시 실패 사유를 보지 않고 `false`로 옮기면 안 된다.
5. `stop_reason`이 `max_sec`, budget, teardown, interrupt/error인 경우와 cursor idle만 확인된 경우.
6. `corrected`/`fallback_match`/`run_status=completed`만 있고 post-action measurement, quality 또는
   alarm-clear 근거가 없는 경우.

### 현재 산출물의 빈칸

- alarm-cycle `RecordingSession`에는 `frame_meta.jsonl`이 없어 자동 Trace의 cursor/occlusion/window
  기하가 구조화되지 않는다.
- `_engineer_watch`는 numerator/Assist/cursor 중 **무엇으로** 종료했는지를 CycleResult나 recording
  manifest에 연결하지 않는다. numerator만 별도 debug JSON의 `done`으로 사후 복원 가능하다.
- alarm 해제는 monitor가 console에 관측하지만 (`Align Fail 해제`) Episode/tag와 연결된 durable
  Verification으로 저장하지 않는다.
- correction status는 post-action readback이 없고, current cycle journal의 `verification_result`도
  비어 있다.
- 현재 checkout의 `align_fail_cycles.csv`는 production 성공 Trace 없이 pre-correction abort만
  포함한다. 실제 성공/충돌 판단은 [대표 Recovery Trace corpus 를 선택한다](02-representative-trace-corpus.md)의
  실증 범위다.

따라서 현 상태에서 Playbook 후보가 안전하게 자동 채울 수 있는 것은 matcher/점유/관측 가능성
Guard, action provenance, 그리고 `numerator_decision.done=true`의 측정 재개 Verification까지다.
그 밖의 Recovery Outcome은 성공 근거가 연결되지 않으면 `unknown` 또는 명시적 인계가 확인된
`escalated`로 남긴다.

검증: 관련 offline 계약 테스트
`uv run pytest poc/workflow_3/recording_filter poc/workflow_3/workflow_extract poc/workflow_3/monitor/test_manual_record.py poc/workflow_3/monitor/test_engineer_done_align_adjustment.py poc/workflow_3/align/test_correction.py poc/workflow_3/monitor/test_cycle_report.py poc/workflow_3/monitor/test_failure_cooldown.py -q`
→ `374 passed, 57 warnings` (기존 bool-return pytest warning).
