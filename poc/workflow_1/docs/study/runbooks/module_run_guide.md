# Workflow 1 모듈 실행 가이드

> 범위: `poc/workflow_1/` · 규칙: **CLI 인자 없음.** 설정은 env / `WorkflowSettings` 로만.
> 환경: 실제 실행은 **Windows**(오피스)에서 한다. Mac 에서는 safe mode dry-run 만 가능하고 창이 없다는 한계가 있다.

---

## 1. 환경 준비

```bash
uv sync --extra dev               # 코어 + dev 도구
# 또는
uv pip install -r requirements.txt
```

- Windows 전용 패키지(pywinauto, pynput)는 절대 제거하지 않는다.
- VLM 은 Flask proxy 를 경유하므로 PC별 `.env` 가 필요 없다(기본 엔드포인트가 하드코딩되어 있다). → `../adr/0002-flask-proxy-vs-direct-vlm.md`

---

## 2. 단계별 모듈 (workflow_1 procedure 순서)

```bash
# Step 0 — RCS 실행
uv run python poc/workflow_1/open_rcs.py

# Step 1 — 로그인
uv run python poc/workflow_1/workflow_login.py
uv run python poc/workflow_1/login_rcs_ui_venus_mai.py   # 2단계 로케이터 단독

# Step 2 — List 탭
uv run python poc/workflow_1/view_list_tab_rcs.py

# Step 3 — Tool 선택
uv run python poc/workflow_1/workflow_select_tool.py
uv run python poc/workflow_1/connect_tool.py             # 임의 Tool 수동 접속

# Step 4 — Tool 화면 캡처
uv run python poc/workflow_1/rcs_screenshot.py

# Step 5 — Tool 창 닫기
uv run python poc/workflow_1/workflow_close_tool.py
```

---

## 3. 알람 감지 (상시 트리거)

```bash
uv run python poc/workflow_1/align_fail_alarm.py          # 폴링 + 알림 + 캡처
uv run python poc/workflow_1/align_fail_alarm_record.py   # 풀 사이클: 접속→기록→Tool 닫기→팝업 닫기
uv run python poc/workflow_1/monitor_align_fail.py        # 알람 + DVR(CCTV) 열기 + CH4 캡처
```

주요 env 토글:

| env | 기본 | 의미 |
|---|---|---|
| `ALIGN_FAIL_POLL_SEC` | 10 | 폴링 주기(초) |
| `ALIGN_FAIL_WINDOW_SEC` | 60 | 탐지 윈도우(초) |
| `ALIGN_FAIL_POPUP_TIMEOUT_SEC` | 60 | 팝업 자동 닫힘 |
| `ALIGN_FAIL_CONNECT_TOOL` | on | RECIPE_ID 있을 때 자동 접속 |
| `ALIGN_FAIL_CONNECT_ACTION` | — | 자동 접속 dry-run 토글 |
| `SAFE_MODE` | — | 실제 마우스/키보드 차단(dry-run) |

---

## 4. DVR/CCTV 프레임 분석 (옵션)

```bash
uv run python poc/workflow_1/capture_window_frames_ch4.py        # CH4 프레임 캡처
uv run python poc/workflow_1/extract_recorded_ch4_frames.py      # 녹화에서 프레임 추출
uv run python poc/workflow_1/locate_cursor_in_captured_frames.py # 프레임에서 커서 끝점 검출
```

---

## 5. 테스트

```bash
uv run python poc/workflow_1/test_tool_name_match.py             # Tool ID 정규화 매칭
uv run python poc/workflow_1/test_vlm_popup_and_cursor_on_frames.py
```

---

## 6. 산출물 위치

| 폴더 | 내용 |
|---|---|
| `debug_images/<model-slug>/` | VLM 입력/오버레이 디버그 이미지 |
| `logs/vlm_calls.log` | VLM 호출 감사 로그(회전) |
| `logs/work2.log` | 일반 이벤트 로그(회전) |
| `logs/align_fail_alarms.txt` | 알람 텍스트 기록 |
| `logs/align_fail_records.csv` | 캡처 매니페스트 |
| `logs/workflow_runs/<run>/` | WorkflowRunner 저널 |
| `recordings/` | CH4 프레임 |
| `align_images/.../captured_img_from_rcs/` | **workflow_2 핸드오프 지점** |

---

## 7. Mac vs Windows

- **Mac**: RCS 가 없으므로 GUI 자동화는 dry-run 으로만 돌고 창이 뜨지 않는다. 코드 편집과 push 가 주 작업이다.
- **Windows**: 실제 실행과 디버깅을 한다. 콘솔 출력과 `debug_images/` 로 결과를 확인한다.
- 오피스 데이터는 Mac 으로 반입할 수 없다 → blind 로 작성한 뒤 오피스에서 실행하고, 텍스트 digest 로 피드백을 받는다.
