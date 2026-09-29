# CEO 시연 영상 2편 설계 - Codex 협의용 brief (2026-09-29)

## 배경 (이미 main 에 있음)

- `poc/workflow_3/monitor/screen_video.py` `ScreenVideoRecorder`: 주 모니터 30fps mp4 + events.json
  (monitor rect, 프레임별 커서, 클릭/휠 시각). 시간축 = perf_counter 경과초.
- `poc/workflow_3/monitor/demo_record_rcs.py`: `demonstration_rcs_control.main(settings, stage_fn=...)`
  을 녹화로 감싸 stages.json(단계 start/end) + subtitles.json 을 남긴다.
- `poc/workflow_3/monitor/polish_demo_video.py`: SEQUENCE(card/clip) 조립. 커서/클릭 원/샷 단위 확대/
  idle cut/자막/카드/letterbox. 출력 1920x1080 H.264 mp4 (PPT 삽입용).

## 사용자 요구 (이번)

1. RCS 시연을 "매크로가 아니라 AI 가 읽고 판단한다" 로 바꾼다. 채택된 장면:
   - (1) List 탭에서 장비별 Connection User 판독 -> 비었으면 접속, 사용 중이면 건너뜀
   - (2) tool monitor 화면 판독: live SEM box 검출 + PM 배율 -> OM/SEM 모드 (클릭 없음)
   - (4) 판독 결과를 MemoPrint 에 **영어로** 기록 (한글 입력 불가, Shift 기호 불가 - 원격이 수정자를 못 건넘)
   - (5) 창 닫고 RCS 로 돌아와 다음 장비
   - 이미지 모드 전환(OM->SEM)은 가동 중 장비라 금지.
2. 이어 붙일 in-tool 편은 **Align Fail 알람 발생부터** 시작해야 한다. 감지 popup(Windows MessageBox)도 보이면 좋다.
3. 두 편 모두 polish_demo_video.py 로 다듬을 수 있어야 한다.

## 설계 A - 순찰 라운드 (구현 중)

- `demonstration_rcs_control`: 상단 상수 `INSPECT = True` (env `DEMO_RCS_INSPECT`).
  - 접속: `connect_to_tool(..., require_occupancy_check=INSPECT)` (production 점유 게이트 그대로).
    `rcs_occupied`/`rcs_occupancy_unknown` 이면 visit status `occupied`/`occupancy_unknown`, 더블클릭이 없었으므로
    창 대기/닫기 생략.
  - `ToolSelectionResult.occupancy_boxes` (신규, 표시 전용): 점유 판독이 읽은 MC ID / Connection User 셀의
    화면 rect. `occupancy_screen_boxes(report, to_screen)` 가 `check_tool_occupancy` report 의 layout 으로 만든다.
  - 창 안: `detect_sem_box(capture)` -> 관찰 {live, mode, pm} -> `inspection_memo()` 영어 4줄
    (`MCD019 AUTO CHECK 2026-09-29 1432` / `CONNECTION USER - NONE` / `LIVE IMAGE - FOUND, MODE OM, PM 210` /
    `CHECKED BY AI AGENT`) -> 기존 memo_print 흐름에 그 문구로.
  - `main(settings, stage_fn=None, note_fn=None)`: 판독마다 `note_fn(title, lines, boxes)` (한글, 영상용).
    사용자 이름은 note 에 절대 싣지 않는다("다른 사용자 접속 중").
- `demo_record_rcs`: note 를 `{t, title, lines, boxes}` 로 notes.json 에.
- `polish_demo_video`: notes.json -> (a) boxes 를 원본 좌표에서 강조 사각형(fade) (b) 우상단 "AI 판독" 패널
  (c) note 표시 중 idle cut 금지 (d) boxes 합집합이 확대 화면에 들어가면 카메라 초점 후보(샷 계획에 합류).

## 설계 B - 알람부터 in-tool 녹화 (제안, 미구현)

- 새 진입점 `demo_record_alarm.py`: `align_fail_monitor.monitor_loop(settings)` 을 **그대로** 돈다(포크 금지).
  그 모듈의 전역 `notify_align_fail_popup` / `run_alarm_cycle` 를 녹화 래퍼로 바꿔 끼운다
  (`process_fail_rows` 가 호출 시점에 모듈 전역을 조회하므로 production 코드 무수정).
  - popup 래퍼: 녹화 시작(PRE_ROLL 1s) -> stage `alarm` start -> 원래 popup.
    popup off 면 cycle 래퍼가 녹화를 시작.
  - cycle 래퍼: stage `cycle` -> 원래 run_alarm_cycle -> 녹화 정지 -> `_demo/alarm_<eqp>_<tag>/`.
    알람마다 clip 하나, Ctrl+C 까지 계속 (또는 첫 1건 후 종료 옵션).
  - 사이클 내부 단계(connect/capture/correction/ok/close)는 runner journal
    (`<take>/runs/*/step_*.json`: timestamp(초 해상도, 종료 시각) + elapsed_ms) 을 사후에 읽어
    stages.json 으로 변환. recorder 가 `wall_t0`(time.time) 을 events.json 에 남겨 경과초로 환산.
  - 단계별 자막 표 `STEP_SUBTITLES = {step_id: 한글}`.
- polish: clip 이름 `alarm_` 으로 같은 SEQUENCE 에 이어 붙임.

## 묻고 싶은 것

1. 설계 A 의 note 채널(note_fn -> notes.json -> polish) 이 적절한가? 더 단순하거나 더 견고한 경로?
2. `ToolSelectionResult` 에 표시 전용 필드를 더하는 것 vs 다른 방법 (production rcs 모듈 오염 우려).
3. 설계 B 의 모듈 전역 교체(monkeypatch) 방식이 production 무수정이라는 이점 대비 위험? 대안(명시 hook 인자)?
4. journal 초 해상도(최대 1s 오차)로 단계 자르기가 충분한가, StepResult 에 epoch 필드를 더해야 하나?
5. 알람 in-tool 편에서 "AI 판독" 패널에 무엇을 보여야 하나 (align key 매칭 위치/점수, reposition, OK 버튼)?
   그 데이터를 cycle 산출물(debug_images, correction outcome, step json)에서 사후에 얻을 수 있나?
6. 놓친 위험 (녹화 부하와 상시 RecordingSession 동시 실행, popup 이 MB_SYSTEMMODAL 이라 캡처/커서에 주는 영향 등).
