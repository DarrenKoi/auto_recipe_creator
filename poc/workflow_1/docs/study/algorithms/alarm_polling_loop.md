# 알람 폴링 루프 — ALID=9006 edge-triggered 감지 (deep dive)

> 대상: `align_fail_alarm.py`, `align_fail_alarm_record.py`, `office_align_fail_alarm.py`, `monitor_align_fail.py`
> 상위 개요: `automation_methods_intro.md` §5

---

## 1. ALID=9006 이 무엇인가

**ALID = Alarm ID.** `9006` 은 **Align Fail**(정렬 실패) 알람 코드입니다.

```python
# office_align_fail_alarm.py
ALIGN_FAIL_ALID = "9006"
```

`filter_align_fail(alarms)` 는 숫자 ALID 필드와 텍스트 알람명에 "align fail"(대소문자 무시)이
들어 있는지를 함께 보고 거릅니다. 둘 중 하나만 와도 잡히도록 견고하게 만들었습니다.

> 의미: 이 알람이 뜨면 장비가 레시피에 등록된 align key 위치를 못 찾아 멈춘 상태 → workflow_2 가
> 이어받아 위치를 찾아야 하는 **핸드오프 트리거**.

---

## 2. 루프 구조 — 폴링 주기 ≠ 탐지 윈도우

`align_fail_alarm.py` 의 `monitor_loop()`:

```python
POLL_INTERVAL_SEC    = env_int("ALIGN_FAIL_POLL_SEC", 10)    # 10초마다 조회
DETECTION_WINDOW_SEC = env_int("ALIGN_FAIL_WINDOW_SEC", 60)  # 최근 60초 알람만 본다
```

**왜 둘을 분리하나?** 알람 보고는 지연될 수 있어서, "지금 막 뜬 것"만 보면 놓치기 쉽습니다.
10초마다 조회하되 **최근 60초 윈도우** 안의 알람을 보면, 보고가 몇 초 늦어도 놓치지 않습니다.
`filter_rows_within_window()` 가 UTC9 시각 컬럼을 `pd.to_datetime()` 으로 파싱해 `datetime.now()`
기준으로 필터합니다.

---

## 3. edge-triggered dedup — 처음 뜬 순간만 처리

알람은 **해제될 때까지 목록에 계속 남습니다.** 폴링할 때마다 알림을 쏘면 같은 팝업이 반복됩니다.
그래서 집합(set) 차집합으로 "새로 뜬 것"만 골라 처리합니다.

```python
active_tools: set[str] = set()          # 현재 알람 떠 있는 EQP_ID 들

# process_fail_rows() 안:
new_tools     = current_tools - active_tools   # 이번에 새로 등장 → 처리
cleared_tools = active_tools - current_tools   # 사라짐 → active 에서 제거(재발 시 다시 처리 가능)
```

- EQP_ID 가 `active_tools` 에 들어가면, **해제되어 빠지기 전까지** 다시 알림이 안 옵니다.
- 해제는 로그로 남깁니다: `[INFO] Align Fail 해제: EQP_ID=...`.

비유하자면, 자동차 경고등은 켜진 동안 계속 켜져 있지만 알림음은 "켜지는 순간"에만 한 번 울리는 것과 같습니다.

---

## 4. 탐지 시 동작

순서대로:

1. **텍스트 로그** — `append_alarm_record()` → `logs/align_fail_alarms.txt` 에 한 줄:
   ```
   {detected_at} | EQP_ID=.. | ALID=9006 | ALARM_NAME=.. | UTC9=.. | RECIPE_ID=.. | OPERATION_DESC=.. | LOT_TYPE_CD=..
   ```
2. **Windows 팝업** — `_show_popup_windows()` (데몬 스레드). `MessageBoxTimeoutW`(비공식 API) 있으면
   `POPUP_TIMEOUT_SEC=60` 후 자동 닫힘, 없으면 `MessageBoxW`(수동). 플래그
   `MB_ICONWARNING | MB_SYSTEMMODAL | MB_SETFOREGROUND`.
3. **Tool 자동 접속 — RECIPE_ID 있을 때만** — `_connect_to_tool_sync()` →
   `workflow_select_tool.connect_to_tool()`. **느슨하게 1회**, 짧은 타임아웃
   (`CONNECT_TOOL_WINDOW_TIMEOUT_SEC=3s`)으로 루프를 막지 않음. env: `ALIGN_FAIL_CONNECT_TOOL`(기본 on),
   `ALIGN_FAIL_CONNECT_ACTION`(dry-run 토글).
   > **왜 RECIPE_ID 조건인가?** RECIPE_ID 가 없으면 어떤 레시피를 보정할지 알 수 없으므로, 자동
   > 접속하지 않고 엔지니어가 직접 처리합니다. (프로젝트 메모리: feedback_no_connect_when_recipe_id)

---

## 5. record 변형 — 접속→캡처→닫기 4단계

`align_fail_alarm_record.py` 는 탐지 후 전체 사이클을 한 바퀴 돕니다:

```
① connect_to_tool()        — List 에서 Tool 더블클릭
② _close_alert_window()    — SYSTEMMODAL 팝업을 캡처 전에 치움
③ record_rcs_window()      — 단일/다중 프레임 캡처
④ close_tool()             — 제목의 tool_id 매칭으로 Tool 창 닫기
```

- 창 탐색은 `RCS_WINDOW_MAX_TRIALS=10` 회로 제한 — RCS 가 타인 점유 중(select 팝업)일 때 무한 폴링
  방지. (프로젝트 메모리: project_rcs_occupied_select_popup)
- 매니페스트 `logs/align_fail_records.csv` 에 EQP_ID/RECIPE_ID 별 캡처 프레임을 기록.

캡처 산출물 경로(`rcs_screenshot.record_rcs_window`):
```
align_images/<eqp_id>/<class>/<recipe>/captured_img_from_rcs/<tag>/<tag>_rcs.jpg
```
`tag` 는 이벤트 UTC9(있으면) 또는 벽시계 타임스탬프. 이 폴더가 **workflow_2 가 읽는 핸드오프 지점**.

---

## 6. 정적 모니터에 맞춘 단순 캡처

Align fail 시 장비가 멈추면서 **SEM Monitor 화면이 고정**됩니다. 그래서 캡처는 보통 **1프레임이면
충분**합니다(`duration_sec<=0` → 1장). 전환 감지나 중복 제거 같은 로직도 필요 없습니다.
(프로젝트 메모리: project_sem_monitor_static_at_align_fail) `settle_sec=2.0` 으로 SEM 렌더만 기다립니다.

---

## 7. 상사 예상 질문

**Q. 1분마다 도는데 알람을 놓치진 않나?**
A. 조회는 10초마다, 보는 범위는 최근 60초입니다. 보고 지연이 있어도 윈도우 안에 들어오면 잡힙니다.

**Q. 같은 장비가 계속 알람이면 매번 접속하나?**
A. 아니요. edge-triggered 라 **처음 뜬 순간만** 1회 처리합니다. 해제 후 재발하면 다시 1회.

**Q. RCS 가 남이 쓰고 있으면?**
A. 창 탐색을 10회로 제한하고 다음 알람 대기로 넘어갑니다. select(공유/종료) 팝업은 무시합니다.

---

## 8. 핵심 상수 한눈에

| 상수 | 값 | 의미 |
|---|---|---|
| `ALIGN_FAIL_ALID` | "9006" | Align Fail 알람 코드 |
| `POLL_INTERVAL_SEC` | 10s | 조회 주기 (`ALIGN_FAIL_POLL_SEC`) |
| `DETECTION_WINDOW_SEC` | 60s | 탐지 look-back 윈도우 |
| `POPUP_TIMEOUT_SEC` | 60s | 팝업 자동 닫힘 |
| `CONNECT_TOOL_WINDOW_TIMEOUT_SEC` | 3s | 자동 접속 창 탐색 타임아웃 |
| `RCS_WINDOW_MAX_TRIALS` | 10 | Tool 창 탐색 시도 상한(점유 대비) |
| capture `settle_sec` | 2.0s | SEM 렌더 대기 |
