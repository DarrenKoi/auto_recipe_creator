# 다중 tool align fail — robust serial 큐 + 측정-시작 기반 녹화 조기종료 (spec)

날짜: 2026-06-11
대상: 여러 tool 에서 align fail 이 동시/근접 발생할 때, **하나도 흘리지 않으면서
한 번에 한 대만 GUI 자동화**하는 robust serial 큐. 더불어 미보정 후 엔지니어 수동
조작 녹화를 고정 timeout 이 아니라 **측정 시작(=align 완료) 감지**로 조기 종료해
큐 대기시간을 줄인다.
대상 파일: `poc/workflow_3/monitor/align_fail_monitor.py`,
`poc/workflow_3/monitor/cycle.py`, 신규 `poc/workflow_3/monitor/engineer_done.py`,
신규 `poc/workflow_3/vlm/prompts/prompt_recipe_monitor_counter.py`,
`poc/workflow_3/config.py`.

---

## 1. 배경·목표

현재 루프(`align_fail_monitor.monitor_loop` → `process_fail_rows` → `cycle.run_alarm_cycle`)는
한 poll 안의 여러 EQP_ID 를 `_collapse_rows_by_tool` 로 합쳐 **edge-trigger** 로 신규만
처리한다. 즉 다중 tool 을 *감지*는 한다. 그러나:

- **처리는 완전 blocking·직렬**: 한 tool 의 사이클(접속→녹화→보정→미보정 시
  `_engineer_watch` 대기, 분 단위)이 끝나야 다음 tool 로 넘어간다. 그 사이 메인 루프는
  poll 도 못 한다.
- **알림 지연**: tool B 의 감지 popup·cube 알림·gather 가 B 차례(=A 사이클 종료 후)에야
  발동한다. 동시에 fail 한 B 를 엔지니어가 **A 처리 내내 모른다** → robust 큐 위반.
- **`_engineer_watch` 가 고정 timeout**(`engineer_watch_sec`)으로 루프를 막아, 큐의 두 번째
  알람이 그만큼 늦어진다.

**사용자 결정(brainstorming):**
- **동시성 모델 = robust serial.** 한 번에 한 tool 만 능동 제어. (병렬 제어/2nd-monitor
  창 이동/배경 녹화는 범위 아님.)
- **A 를 완전히 끝내고 B.** 단, `_engineer_watch` 를 **측정 시작 감지**로 조기 종료해
  "완전히 끝내기"를 빠르게 만든다.
- **감지 방식 = hybrid.** Recipe Monitor 의 측정 카운터 **분자(numerator)** 가 align 완료
  후 증가(1/350→2/350→3/350...)하는 것을 신호로 쓴다. tool 마다 창 위치가 다르고
  드래그 가능하므로 **VLM 으로 위치를 사이클당 1회 grounding → 캐시**, 이후는 **CV
  변화감지(gate) → 변화 시에만 OCR 확인(N≥2)**.

**목표:** (a) 동시 다발 fail 을 즉시 전부 acknowledge(log+popup+cube+gather)하고, 제어는
FIFO 직렬로 한 대씩. (b) 미보정 watch 를 측정-시작 감지로 조기 종료(고정 timeout 은 cap).

**핵심 단순화:** serial + 사이클 `finally` 의 `close_tool(A)` 가 B 사이클 시작 전에 A 창을
닫으므로 **Remote Monitoring 창은 항상 1개만 열림** → 같은 좌표 겹침 없음 →
`capture_window`(mss 화면영역 grab)의 occlusion 문제·창 이동 primitive **불필요**.

---

## 2. 범위·불변 원칙

**포함**
- `process_fail_rows` two-pass 재구성(acknowledge-all → control-serial, FIFO).
- `cycle._engineer_watch` 를 done-detector 조기종료로 업그레이드(+cap backstop).
- 신규 `EngineerDoneDetector`(VLM 1회 grounding + CV gate + OCR confirm).
- 신규 grounding prompt builder.
- `Workflow3Settings` 필드 + env 배선.

**제외(비범위)**
- 병렬 제어, 핸드오프 배경 녹화, 2nd-monitor 창 이동/move-window primitive.
- durable cross-poll pending 큐(§7 가정이 깨질 때만 재검토).
- Recipe Monitor ROI 의 **오피스 캘리브레이션**(grounding 프롬프트 문구 미세조정,
  numerator 패딩 실측값) — 기본값 제공 + 오피스에서 검증/조정.
- OCR/grounding 서비스 자체 구현(기존 `paddleocr-vl-1.5`/`ui-venus` 재사용).

**불변 원칙**
- **경계 유지:** monitor = 폴링/큐/배선, vision/vlm = 인식. 의존 방향 `monitor → vlm`
  준수(`engineer_done` 은 monitor 안, vlm client/prompt 만 호출). workflow_1/2 import 0.
- **graceful degrade:** grounding 거부(`[-1,-1]`)·OCR 실패·detector 예외 어느 것도 루프/
  watch 를 죽이지 않는다 → 항상 `engineer_watch_sec` cap 으로 안전 복귀.
- **기본 비활성:** `engineer_done_detect_enabled` 기본 False(오피스 캘리브레이션·검증
  전까지 기존 고정 timeout 동작 유지 → 회귀 위험 0).
- **CV 가 전이를 판정하는 규칙 존중:** VLM 은 위치 grounding·OCR 보조만, 측정-시작
  판정의 정량 근거는 CV 변화감지 + OCR 숫자(`CLAUDE.md` vision 규칙).

---

## 3. 아키텍처

의존 방향: `align_fail_monitor`(monitor) → `cycle`(monitor) → `engineer_done`(monitor) →
`vlm.vlm_client` + `vlm.prompts.prompt_recipe_monitor_counter` + `util.capture_window`.
순환 없음.

| 파일 | 상태 | 책임 |
|---|---|---|
| `monitor/align_fail_monitor.py` | 수정 | `process_fail_rows` two-pass(acknowledge-all → control-serial, FIFO by UTC9). `active_tools` 편입 시점을 pass 1 로. |
| `monitor/cycle.py` | 수정 | `_engineer_watch` 를 done-detector 조기종료로. watch 진입 시 detector 빌드(enabled 시), 기존 `vlm_client` 재사용. |
| `monitor/engineer_done.py` | 신규 | `EngineerDoneDetector`(callable) + `build_engineer_done_detector()`. VLM 1회 grounding → ROI 캐시 → CV gate → OCR confirm(N≥2). |
| `vlm/prompts/prompt_recipe_monitor_counter.py` | 신규 | 단일요소 grounding `(system,user)`; ui-venus `[x,y]` 0–1000, 거부 `[-1,-1]`. |
| `config.py` | 수정 | `Workflow3Settings` 측정-감지 필드 + `load_workflow3_settings` env override. |

---

## 4. 상세 설계

### 4-A. two-pass `process_fail_rows` (`align_fail_monitor.py`)

현재: `new_tools = current_tools - active_tools` 를 `sorted()`(EQP_ID 알파벳) 순회하며
tool 별로 record→popup→gather→**cycle(blocking)**→`active_tools.add` 를 한 번에 한다.

신규(시그니처·반환 불변, 내부만 2-pass):

```
by_tool = _collapse_rows_by_tool(fails)
current_tools = set(by_tool)
cleared_tools = active_tools - current_tools           # 기존 해제 처리 동일
active_tools.difference_update(cleared_tools)
new_tools = current_tools - active_tools

# FIFO 정렬: 알람 시각(UTC9) 오름차순, 동률/파싱불가는 EQP_ID 보조키
ordered = _order_new_tools_fifo(new_tools, by_tool)

# Pass 1 — acknowledge ALL (빠르고 비차단)
for eqp_id in ordered:
    info = by_tool[eqp_id]
    append_alarm_record(...)                            # 빠른 파일 append
    if settings.popup_enabled: notify_align_fail_popup(...)   # daemon thread (비차단)
    gather_success_async(eqp_id, info["recipe_id"], settings) # daemon thread (비차단)
    active_tools.add(eqp_id)                            # ownership 을 ack 시점에 확보

# Pass 2 — control SERIAL (blocking, FIFO)
for eqp_id in ordered:
    info = by_tool[eqp_id]
    cycle = run_alarm_cycle(...) if settings.cycle_enabled else CycleResult(...)
    append_cycle_manifest(info, cycle)

return len(ordered)
```

근거:
- **Pass 1 은 전부 비차단**(popup=`show_popup_windows` daemon thread, gather=daemon
  thread, record=짧은 파일 append). 따라서 동시 fail 전부가 *즉시* 가시화된다.
- **`active_tools` 편입을 pass 1 로** 이동: ack 시점에 ownership 확보 → pass 2 가 중간에
  끊겨도(KeyboardInterrupt 등) 다음 poll 에서 같은 tool 재-ack/재-cycle 안 함. (프로세스
  재시작 시 `active_tools` 가 in-memory 라 자연 재처리 — 기존과 동일.)
- **FIFO(UTC9)**: 큐 의미상 먼저 fail 한 tool 먼저 제어. 기존 EQP_ID 알파벳 정렬은 우연.
- `_order_new_tools_fifo`: `pd.to_datetime(UTC9, errors="coerce")` 로 정렬, NaT 는 뒤로
  보내고 EQP_ID 보조정렬(결정적). UTC9 컬럼은 이미 `_collapse_rows_by_tool` 가 보존.

부수효과 메모: 같은 제목("CD-SEM Align Fail 감지") popup 이 여럿 떠도
`close_alert_window` 는 제목으로 전부 닫으므로(루프) A 사이클의 `close_alert_popup`
step 이 B popup 도 함께 닫는다. B 는 이미 log+cube 로 통지됐고 popup 은 backstop
timeout 이 있어 허용. (필요 시 추후 per-tool 제목 분기 — 비범위.)

### 4-B. 완료-게이트 `_engineer_watch` (`cycle.py`)

현재:
```
def _engineer_watch(recording, watch_sec):
    deadline = time.time() + watch_sec
    while time.time() < deadline and recording.is_alive():
        time.sleep(2.0)
```

신규(첫 조건 충족 시 종료):
```
def _engineer_watch(recording, watch_sec, *, done_detector=None, poll_sec=8.0):
    deadline = time.time() + watch_sec
    next_check = 0.0
    while time.time() < deadline and recording.is_alive():
        if done_detector is not None and time.time() >= next_check:
            try:
                if done_detector():                    # True = 측정 시작 확인
                    print("[INFO] 측정 시작 감지(align 완료 추정) - 녹화 조기 종료")
                    log_work2_event(component=LOG_COMPONENT, message="engineer_done_detected")
                    break
            except Exception as exc:
                print(f"[WARNING] done detector 예외(무시, cap 으로 진행): {exc}")
            next_check = time.time() + poll_sec
        time.sleep(2.0)
```

종료 조건(OR): ① 녹화 스레드 사망(창 닫힘/max_sec, 기존) ② `done_detector()` True
③ `watch_sec` cap(이제 주 bound 가 아니라 backstop). detector 예외는 삼켜 ②만 무력화,
①③ 은 유지.

`run_alarm_cycle` watch 호출부:
```
if recording is not None and (outcome is None or outcome.status != "corrected"):
    done_detector = None
    if settings.engineer_done_detect_enabled and context.get("tool_window") is not None:
        try:
            done_detector = build_engineer_done_detector(
                context["tool_window"], settings, vlm_client=context.get("vlm_client"))
        except Exception as exc:
            print(f"[WARNING] done detector 생성 실패(고정 timeout 로 진행): {exc}")
    _engineer_watch(recording, settings.engineer_watch_sec,
                    done_detector=done_detector, poll_sec=settings.engineer_done_poll_sec)
```

메모: `_exec_run_correction` 가 만든 `vlm_client` 를 `context["vlm_client"]` 로 보관해
detector 가 재사용(없으면 detector 가 자체 생성 시도). 보정 성공(corrected) 경로는 watch
자체를 안 타므로 grounding VLM 호출 0(불필요 비용 없음).

### 4-C. `EngineerDoneDetector` (`monitor/engineer_done.py`, 신규)

Recipe Monitor 측정 카운터 분자로 "측정 시작=align 완료" 판정. callable 1개가 watch
iteration 마다 호출되며 사이클당 상태(ROI 캐시, 직전 crop, 직전 N)를 보유.

```
class EngineerDoneDetector:
    def __init__(self, tool_window, settings, *, vlm_client=None, ocr_client=None):
        self.tool_window = tool_window
        self.s = settings
        self._vlm = vlm_client                 # grounding(ui-venus) — lazy 생성 허용
        self._ocr = ocr_client                 # OCR(paddleocr-vl) — lazy 생성 허용
        self._roi_ratios = None                # (l,t,r,b) tool-window 상대비율 — 캐시
        self._localize_tried = False
        self._prev_gray = None
        self._last_n = None
        self._ocr_miss_streak = 0

    def __call__(self) -> bool:
        if self._roi_ratios is None and not self._localize_tried:
            self._localize_tried = True
            self._roi_ratios = self._localize_numerator()   # 1회 grounding
        if self._roi_ratios is None:
            return False                                     # grounding 실패 → cap 에 위임
        crop = self._crop_numerator()                        # tool-window 캡처 → 상대ROI crop
        if crop is None:
            return False
        gray = _to_diff_gray(crop)
        changed = _frame_changed(self._prev_gray, gray, self.s.engineer_done_change_min_px)
        self._prev_gray = gray
        if not changed:
            return False                                     # 정적 → OCR 안 함(싸게)
        n = self._ocr_numerator(crop)                        # 변화 시에만 OCR
        if n is None:
            self._ocr_miss_streak += 1
            if self._ocr_miss_streak >= self.s.engineer_done_relocalize_after_miss:
                self._roi_ratios = None; self._localize_tried = False  # 드래그 대비 1회 재grounding
            return False
        self._ocr_miss_streak = 0
        is_done = n >= self.s.engineer_done_min_count and (self._last_n is None or n >= self._last_n)
        self._last_n = n
        return is_done
```

세부:
- `_localize_numerator()`: `capture_window(tool_window)` → `prompt_recipe_monitor_counter`
  로 grounding → 단일 점 `[x,y]`(0–1000). 거부 `[-1,-1]`/파싱불가 → None. 점을
  numerator cell 로 확장(`engineer_done_roi_pad_*` 비율 패딩) → tool-window **상대비율**로
  저장(전체 창 이동/재캡처에 견고).
- `_crop_numerator()`: 매 호출 tool-window 재캡처 후 상대비율→픽셀 crop. 캡처 실패는
  None(그 회차 skip).
- `_to_diff_gray`/`_frame_changed`: `recording.py` 의 다운샘플+delta 임계 로직을 재사용
  (import 또는 동등 helper). gate 가 정적 구간(align-fix 중)에 OCR 호출 0 을 보장.
- `_ocr_numerator(crop)`: tight crop 만 OCR(전체 스크린샷 환각 회피 — memory
  `paddleocr_vl_screenshot_hallucination`). 결과 텍스트에서 정수 추출(첫 연속 숫자열).
  "N/M" 이 잡혀도 분자만 취함. 숫자 없으면 None.
- 판정: `N ≥ engineer_done_min_count(기본 2)` **且** 비감소. 최초 충족 즉시 done.
- 드래그 견고성: OCR 가 변화 후에도 `relocalize_after_miss` 회 연속 숫자 미검출이면 ROI
  1회 재grounding(엔지니어가 패널을 옮긴 경우). 과도 호출 방지를 위해 streak 기반.

`build_engineer_done_detector(tool_window, settings, *, vlm_client=None)`:
config 게이트 확인 후 detector 생성(필요 시 OCR client lazy 생성). RCS/VLM 모듈 부재
환경에선 None 반환(개발 PC) → watch 는 고정 timeout.

### 4-D. grounding prompt (`vlm/prompts/prompt_recipe_monitor_counter.py`, 신규)

기존 prompt builder 규약: `(system_message, user_message)` 반환, `width`/`height` +
타깃 파라미터. ui-venus 공식 grounding 형식(memory `ui_venus_official_grounding`):
단일 요소, `[x,y]` 0–1000, 미발견 시 `[-1,-1]`. 첫-글자 anchoring 원칙 적용.

대상 문구(초안, 오피스 조정): "Recipe Monitor 영역에서 Port/Slot/Recipe 행의 현재
측정 점 카운터(예: `2/350`)의 **분자 숫자**(슬래시 앞 정수)의 중심점." 보이지 않으면
`[-1,-1]`.

### 4-E. `Workflow3Settings` (`config.py`)

| 필드 | 기본 | env | 의미 |
|---|---|---|---|
| `engineer_done_detect_enabled` | `False` | `ALIGN_FAIL_ENGINEER_DONE_DETECT` | 측정-시작 조기종료 on/off. 캘리브레이션 전 False. |
| `engineer_done_poll_sec` | `8.0` | `ALIGN_FAIL_ENGINEER_DONE_POLL_SEC` | detector 호출 간격(watch 안). |
| `engineer_done_min_count` | `2` | `ALIGN_FAIL_ENGINEER_DONE_MIN_COUNT` | done 으로 보는 최소 분자값. |
| `engineer_done_change_min_px` | `4` | `ALIGN_FAIL_ENGINEER_DONE_CHANGE_MIN_PX` | CV gate 변화 픽셀 임계(다운샘플). |
| `engineer_done_relocalize_after_miss` | `3` | `ALIGN_FAIL_ENGINEER_DONE_RELOCALIZE_MISS` | 변화 후 OCR 연속 미검출 N회 시 ROI 재grounding. |
| `engineer_done_roi_pad_x` / `_y` | `0.03` / `0.02` | (env 생략 가능) | grounding 점→numerator crop 확장 비율(상대). |
| `engineer_done_vlm_service` | `ui-venus` 계열 기본 | `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE` | grounding 서비스 slug. |
| `engineer_done_ocr_service` | `paddleocr-vl-1.5` | `ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE` | numerator OCR 서비스 slug. |

`engineer_watch_sec`(기존)은 cap 으로 의미 재정의(주석만 갱신).

---

## 5. 데이터 흐름

```
poll → filter_align_fail → window 필터
 → process_fail_rows:
     [Pass 1] ack 전부: log + popup(비차단) + cube? + gather(비차단) + active_tools.add
     [Pass 2] FIFO 직렬:
        run_alarm_cycle(A): RCS→connect→record→SEM panel→correction
           ├ corrected      → notify 생략 → finally(record stop→close_tool(A))
           └ not corrected  → cube notify → _engineer_watch(A):
                 매 poll_sec: EngineerDoneDetector()
                    grounding(1회 캐시) → CV gate(정적=skip) → 변화 시 OCR → N≥2 → break
                 (또는 창 닫힘 / watch_sec cap)
              → finally(record stop → close_tool(A))     # A 창 닫힘 → 겹침 없음
        run_alarm_cycle(B): (A 창 닫힌 뒤 시작)
```

---

## 6. 에러 처리·degrade

- grounding 거부/실패 → ROI None → detector 항상 False → `watch_sec` cap 종료(기존 동작).
- OCR 실패 → 그 회차 미판정, miss streak 누적 시 1회 재grounding, 끝내 안 되면 cap.
- detector 예외 → `_engineer_watch` 가 삼킴(②만 무력화, ①③ 유지).
- detector 생성 실패 → 고정 timeout watch.
- Pass 1 best-effort(popup/gather) 실패는 기존대로 삼킴. 사이클 `try/finally` teardown 불변.

---

## 7. 열린 가정 (오피스 확인 필요)

MES 알람 피드가 **진행 중** align fail 을 poll 마다 **최신 UTC9** 로 재보고한다고 가정
(현 edge-trigger·window 필터 설계 전제). 만약 **onset 시각**으로만 보고하면, 긴 사이클
뒤에 큐잉된 tool 이 `detection_window_sec` 밖으로 aging 되어 누락될 수 있다 → 그 경우
durable cross-poll pending 큐 변형(비범위)로 재검토. 본 spec 은 piece-of-poll 동시 fail
(같은 poll 에 함께 등장) 의 robust 직렬화를 보장하며, 이 가정에 의존하지 않는다(두 tool 이
같은 poll 에 보이면 둘 다 같은 window 안). 가정은 *사이클 중 새로 발생한* tool 에만 영향.

---

## 8. 테스트 (전부 Mac/dev, RCS·VLM 불요)

신규 `poc/workflow_3/monitor/test_multi_tool_queue.py`:
- **two-pass 순서**: fake cycle fn 으로 [B(이른 UTC9), A(늦은)] 입력 → ack 가 cycle 보다
  먼저 전부 호출됨(호출 로그 순서 검증), cycle 은 FIFO(B→A).
- **active_tools 편입**: pass 1 후 전 tool 이 active. 다음 poll 동일 입력 → 신규 0.
- **해제**: current 에서 빠진 tool 이 active 에서 제거.
- **gather/popup 호출 수**: 신규 tool 수만큼 1회씩(monkeypatch).

신규 `poc/workflow_3/monitor/test_engineer_done.py`:
- fake `capture`(시퀀스 이미지) + fake grounding + fake OCR 주입.
- 정적 구간 → CV gate False → OCR 미호출(call count 0).
- 카운터 변화 + OCR "2" → True(첫 충족). "1" → False. 비감소 위반(3→2) → False.
- grounding 거부([-1,-1]) → 항상 False(call 후 ROI None).
- OCR miss streak ≥ 임계 → 재grounding 1회 트리거.
- detector 예외 → `_engineer_watch` 가 cap 으로 정상 종료(별도 watch 단위 테스트).

`_engineer_watch` 단위(fake recording.is_alive + fake detector):
- detector True → cap 전 조기 break. recording 사망 → break. detector 예외 → cap 까지.

---

## 9. 미해결·추후

- Recipe Monitor grounding 문구·numerator 패딩·`min_count`·`poll_sec` **오피스 실측 튜닝**
  후 `engineer_done_detect_enabled=True` 승격.
- durable cross-poll 큐(§7 가정 깨질 때).
- per-tool popup 제목 분기(B popup 이 A 사이클에 닫히는 것 보존하고 싶을 때).
- 측정-시작 외 추가 done 신호(알람 해제 cross-check 등) — 현재 불필요.
