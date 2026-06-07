# Workflow 1 — RCS 자동화 절차 및 구현 가이드

> 최종 갱신: 2026-06-07 · 범위: `poc/workflow_1/`
> 본 문서는 workflow_1 의 **단일 권위 문서(single source of truth)** 다. 알고리즘 해설은
> `../algorithms/`, `../cv/`, 설계 결정은 `../adr/` 참조.

---

## 1. 개요와 경계

`poc/workflow_1/` 은 **RCS 로그인 → Tool 선택 → Align Fail 알람 감지(ALID=9006) → Tool 화면 캡처**
까지 책임진다. 그 다음 단계, 즉 멈춘 Tool 의 SEM Monitor 를 조작해 align key 위치를 찾는 일은
`poc/workflow_2/` 의 몫이다.

핵심 설계 원칙:

- **좌표는 VLM(coarse→fine)이 제안하고, OCR 은 확인만 한다.** 미확인 시 클릭하지 않는다.
  (workflow_2 의 "좌표는 CV 가 결정, VLM 은 식별만" 과 짝을 이루는 workflow_1 판 안전 규칙.)
- **RCS 는 legacy GUI** 라 내부 컨트롤이 pywinauto UIA/win32 에 잡히지 않는다 → 스크린샷→VLM→pynput
  경로로 조작한다. pywinauto 는 창 띄우기·제목 탐색·foreground 에만 쓴다.
- **DPI 배율(125/150%)을 항상 보정** 한다. import 시 DPI awareness 선언 + 클릭 직전 rect/screenshot
  비율 보정.

---

## 2. 트리거와 데이터 흐름

```
[workflow_1] ALID=9006(Align Fail) 감지   ← align_fail_alarm.py 폴링 루프
      │   알림(팝업/로그) + (RECIPE_ID 있으면) Tool 접속
      │   → RCS 화면 캡처(rcs_screenshot.py) → captured_img_from_rcs 적재
      ▼
align_images/<eqp_id>/<class>/<recipe>/
      ├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   (등록 align key, MES)
      ├─ align_img_from_msr/   S*/E*                              (측정 궤적, E=fail, MES)
      └─ captured_img_from_rcs/ <tag>_rcs.jpg                     (fail 시점 RCS 캡처, workflow_1)
      ▼
[workflow_2] 가 이 트리를 읽어 align key 탐색 시작
```

루트 상수: `ALIGN_IMAGES_DIR` (in `__init__.py`). workflow_1 이 쓰는 폴더는
`captured_img_from_rcs/`. (자세한 layout 은 프로젝트 메모리 project_align_images_layout.)

---

## 3. 단계별 절차

### Step 0 — RCS 실행
- `open_rcs.py` → `RcsMainHD.exe` 기동. pywinauto 는 exe 실행 + 창 제목 탐지에만.

### Step 1 — 로그인 (2단계 VLM)
- `workflow_login.py` / `login_rcs_ui_venus_mai.py`.
- 요소(Server/UserID/Password/Login)마다: **ui-venus coarse bbox → crop+zoom → mai-ui fine point**
  → (선택) OCR 확인 → DPI 보정 → pynput 클릭/입력.
- 상세: `../algorithms/two_stage_vlm_locator.md`.

### Step 2 — List 탭 진입
- `view_list_tab_rcs.py` — List 탭을 찾아 클릭(2단계 VLM).

### Step 3 — Tool 선택·더블클릭
- `workflow_select_tool.py` / `connect_tool.py`.
- 먼저 VLM 으로 영역을 제안(coarse→fine, 최대 2회)하고, 실패하면 OCR fallback + **Tool ID 정규화
  매칭** 으로 넘어간다. 그래도 못 찾으면 List 영역을 스크롤(변화 감지, 최대 8회)하고, 찾으면 더블클릭한다.
- List 레이아웃: 최좌측 MC ID 컬럼, 좌측 신호등(녹=On/검=Off). → `../sem_box/rcs_list_tab_layout.md`.
- 상세 매칭: `../algorithms/tool_name_canonicalization.md`.

### Step 4 — Tool 화면 캡처
- `rcs_screenshot.py` — Remote Monitoring 창 탐색 → `settle_sec=2.0` 대기 → 1프레임 캡처
  (정적 모니터라 1장이면 충분) → `captured_img_from_rcs/<tag>/<tag>_rcs.jpg`.

### Step 5 — Tool 창 닫기
- `workflow_close_tool.py` — 제목에 tool_id 가 포함된 "Remote Monitoring System - ..." 창을 찾아
  **제목을 검증한 뒤** 닫는다(close()→WM_CLOSE→SC_CLOSE→빨간 X 사다리).

### (상시) 알람 폴링
- `align_fail_alarm.py` / `align_fail_alarm_record.py` 가 위 Step 1~5 를 알람 트리거로 한데 묶는다.
- 상세: `../algorithms/alarm_polling_loop.md`.

### (옵션) DVR/CCTV 분석
- `monitor_align_fail.py` → CH4 프레임 캡처(`capture_window_frames_ch4.py`) → 커서 검출
  (`locate_cursor_in_captured_frames.py`). → `../cv/cursor_detection.md`.

---

## 4. 오케스트레이션 — WorkflowRunner

`workflow_runner.py` 는 entry point 가 아니라 **라이브러리**다.

- `WorkflowRunner.run(steps, context, executor)` — `list[WorkflowStep]` 을 순차 실행.
  - 각 스텝은 `depends_on` → `skip_if` → `preconditions` → 실행 → `success_criteria` 순으로 검사한다.
  - 실패하면 abort(중단)한다. 결과는 `logs/workflow_runs/<run_id>_<name>/` 에 저널링한다
    (`run_state.json`, `step_<id>.json`).
- `ConditionChecker` — 사전/사후 조건 평가. 조건 타입: `WINDOW_VISIBLE`, `WINDOW_FOUND`,
  `WINDOW_APPEARED`, `DIALOG_DISAPPEARED`, `TEXT_APPEARED`, `FIELD_READY_FOR_INPUT` 등. 그룹은
  `ALL`(AND) / `ANY`(OR).

설정은 `WorkflowSettings`(`workflow_config.py`), `load_workflow_settings()` 로 로드(env override):

| 설정 | 기본 | 의미 |
|---|---|---|
| `total_retry_budget` | 10 | 전체 재시도 예산 |
| `service_fallback_order` | (`ui-venus`,`mai-ui`) | VLM 우선순위 |
| `pre_click_settle_sec` | 0.2 | 클릭 전 대기 |
| `post_double_click_settle_sec` | 0.5 | 더블클릭 후 대기 |
| `char_type_delay_sec` | 0.03 | 타이핑 간 지연 |
| `login_verify_timeout_sec` | 15.0 | 로그인 검증 타임아웃 |

---

## 5. 안전장치 모음

- **safe mode**: `SAFE_MODE` 는 실제 마우스/키보드 출력을 차단한다(dry-run). `action_enabled`/
  `typing_enabled` 는 기본적으로 `SAFE_MODE` 의 반대 값을 따른다.
- **미확인 시 클릭 금지**: OCR 확인에 실패하면 클릭하지 않는다.
- **Tool ID 모호 시 자동선택 포기**: 여러 행이 매칭되면 VLM 에 위임한다.
- **점유 대비 창 탐색 상한**: `RCS_WINDOW_MAX_TRIALS=10` (select 팝업 무한 폴링 방지).
- **RECIPE_ID 있을 때만 자동 접속**: 없으면 엔지니어가 직접 처리한다.

---

## 6. 실행법

각 모듈 실행법은 `../runbooks/module_run_guide.md` 를 참조한다. 모든 스크립트는 CLI 인자 없이
`uv run python <script>.py` 로 실행하며, 설정은 env 와 WorkflowSettings 로 한다.
