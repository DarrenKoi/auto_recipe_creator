# 실시간 루프 실패경로 하드닝 (Loop Failure-Path Hardening)

- 작성일: 2026-08-06
- 대상: `poc/workflow_3/monitor/` (+ `poc/workflow_3e/abort_cycle.py`)
- 상태: 설계 승인됨 (구현 계획 대기)

## 1. 배경과 동기

CV 매칭 정확도 축(verifier)은 2026-07-21 registration A/B 로 사실상 소진됐다.
2026-08-06 전체 표본(67/67 recipe, 334점) 재측정에서도 같은 결론이 재현됐다 —
`route_sw` r1=0.826, oracle=0.865, `allmiss[pm=39/re=6]`. 즉 어떤 rerank 로도 못
고치는 몫이 87%(pm=39)이고 verifier 여지는 6점뿐이다.

따라서 무게중심을 **실시간 루프가 오피스에서 안 죽고 도는가**로 옮긴다. 이 문서는
그중 **area 1 — 실패/예외 경로**만 다룬다 (area 2 rerank 실구동 확인, area 3 office
어댑터/경로, area 4 장시간 무인 구동은 별도 spec).

**동기는 예방적이다.** 아래 F1~F5 는 오피스에서 관측된 장애가 아니라 코드 감사로
찾은 결함이다. 실전 테스트(엔지니어 합동 평가) 전에 닫는 것이 목적이다.

## 2. 감사 결과 (F1~F5)

| ID | 위치 | 내용 | 심각도 |
|----|------|------|--------|
| F1 | `monitor/cycle.py:1641-1652` | check-only teardown 이 **엔지니어 물리 입력을 잠근 채로 끝날 수 있음** | 최고 |
| F2 | `monitor/align_fail_monitor.py:370-378` | 실패 tool 을 매 poll 재시도 → 단일 RCS 커서 독점(starvation) | 높음 |
| F3 | `monitor/rcp_msr_gather.py:101` | 동기 office 다운로드에 timeout 없음 → 무한 정지 가능 | 높음 |
| F4 | 세 사이클의 `finally` 전반 | "teardown 은 항상 돈다"가 **강제되지 않는 관행** | 중간(잠재) |
| F5 | `align_fail_monitor.py:321-378` | tool 1대 예외가 같은 poll 의 나머지 tool 을 건너뜀 | 낮음 |

### F1 — check-only teardown 의 입력 잠금

`run_check_only_cycle` 의 `finally` 는 `close_tool`(보호됨) → `close_alert_window`
(**미보호**) → `block_input(False)` 순이다. `close_alert_window` 가 던지면 해제가
실행되지 않아 엔지니어의 마우스·키보드가 잠긴 채 남는다. 탈출은 Ctrl+Alt+Del 뿐이며,
엔지니어가 그 사실을 알아야만 한다.

`run_alarm_cycle`(`cycle.py:632`)과 `workflow_3e/abort_cycle.py:218`은 해제를 **먼저**
하므로 이 결함이 없다. 즉 **순서 규약이 주석에만 있어서 한 곳에서 뒤집힌 사고**다.

### F2 — 실패 cooldown 부재

`_OCCUPIED_FAILURE_CLASSES` 만 cooldown 을 받는다. 그 외 사유로 반복 실패하는 tool 은
매 poll 재시도되어, 직렬화된 단일 RCS 커서를 독점하고 다른 알람을 굶긴다.

### F3 — 무한 블로킹 office 호출

`gather_rcp_msr` 은 예외는 삼키지만 **timeout 이 없다**. `download_rcp_msr` 이 걸리면
모니터 전체가 무한 정지한다. 동기 설계 자체는 **옳고 문서화돼 있다** — feasibility/보정이
assets 를 읽기 전에 디스크에 있어야 하므로 async 로 바꾸면 "보정 불가" 오판이 난다.
따라서 고칠 것은 동기성이 아니라 **경계(bound)** 다.

### F4 — 강제되지 않는 teardown 불변식

세 `finally` 는 `recording.stop()`·`close_alert_window()` 가 내부적으로 방어적이라는
사실에 의존한다. 2026-08-06 확인 결과 **오늘은 실제로 방어적이다**(`_write_manifest` 가
자체 예외를 삼키고, `join` 은 bounded). 그러나 이를 강제하는 장치가 없어, `notify.py`
나 `recording.py` 한 줄 수정이 teardown 을 조용히 무력화할 수 있다. 같은 형태가
`workflow_3e/abort_cycle.py` 에도 복제돼 있어 두 곳이 동시에 깨진다.

### F5 — poll 배치 중단

`run_alarm_cycle` 은 내부에서 예외를 잡지만, `append_cycle_manifest`·`gather_*`·팝업은
tool 별 보호 밖에 있다. 하나가 던지면 그 poll 의 남은 tool 이 전부 건너뛰어진다
(다음 poll 에 재시도되므로 유실이 아니라 지연).

## 3. 채택 방안

**방안 B — guarded-teardown 헬퍼 + 지점 수정.**

검토한 대안:

- **A. 지점 수정만** — 최소 diff. 그러나 F4 가 남아 "teardown 은 항상 돈다"가 계속
  읽어서 확인해야 하는 성질로 남고, 다음 수정이 조용히 되깨뜨릴 수 있다.
- **C. watchdog/supervisor** — 사이클 전체를 워커 스레드로 시간 제한. **기각.** Python
  에서 GUI 자동화 중인 스레드를 안전하게 죽일 수 없고, 단일 RCS 커서를 중심으로
  **의도적으로 직렬**인 루프에 동시성 모델을 도입한다.

B 를 택한 이유: 모든 수정이 이 코드베이스에 **이미 있는 패턴**을 따른다 — 보호된
teardown 은 기존 `close_tool` 가드와 같고, cooldown 은 `occupied_cooldown` 을 재사용하며,
bound 는 `success_gather.wait_for_gather()` 의 관용구를 재사용한다. 새로 발명할 개념이
없고, F4 를 경계심이 아니라 구조로 닫는다.

## 4. 설계

### 4.1 `monitor/teardown.py` (신규)

```python
def run_teardown(steps, *, label=""):
    """teardown 단계를 각각 독립 보호하며 순서대로 실행한다.

    한 단계가 던져도 다음 단계는 반드시 실행된다 - teardown 은 "가능한 만큼 정리"가
    계약이고, 중간 실패가 뒤(특히 입력 차단 해제/tool 닫기)를 막으면 안 된다.
    반환: 실패한 (이름, 오류) 목록 - 호출부가 result.notes 에 남길 수 있다.
    """
```

- 각 step 은 `(name, callable)` 쌍.
- 헬퍼 자체는 **절대 던지지 않는다**.
- `cycle.py`(1663줄) 가 아니라 별도 모듈에 두는 이유: (a) `cycle.py` 는 더 커지면
  안 되는 파일이고, (b) `workflow_3e` 가 import 해야 하는데 함수 하나 때문에
  1663줄 오케스트레이터를 끌어오게 할 수 없다.

**순서 규약을 명시적으로 만든다.** 오늘 "입력 해제를 먼저" 는 세 사이클 중 둘이
우연히 지키는 관행이다. 헬퍼 docstring 이 이를 규약으로 적고, 테스트가 검증한다:
**모든 teardown 목록의 첫 단계는 `block_input(False)` 다.** F1 을 실제로 막는 것은
try/except 가 아니라 이 순서를 검사 가능한 성질로 만드는 것이다.

### 4.2 `_teardown_steps(...)` 추출

각 사이클은 teardown 목록을 이름 있는 함수로 만든다:

```python
def _teardown_steps(...) -> list[tuple[str, Callable]]:
```

`finally` 는 `run_teardown(_teardown_steps(...))` 만 호출한다. 이유는 두 가지다 —
`finally` 블록이 짧아지고(1663줄 파일에서 중요), **RCS 스택 없이 순서 불변식을
테스트할 수 있다**(§6 참조).

호출부 형태(`run_alarm_cycle`, 예시):

```python
finally:
    failures = run_teardown(_teardown_steps(...), label=f"align_fail_cycle {eqp_id}")
    result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)
```

`recording_stop` 단계는 `stop()` 과 `result.recording_dir`/`frame_count` 갱신을 **함께**
감싼다 — 여기서 실패하면 두 필드가 조용히 비는 대신 note 가 남고 나머지 teardown 은
계속된다.

**적용 대상 3곳**: `run_alarm_cycle`, `run_check_only_cycle`(F1 해소),
`workflow_3e/abort_cycle.run_abort_cycle`(F4 해소).

**단계 목록은 사이클마다 다르다** — `recording_stop` 은 `run_alarm_cycle` 에만 있다
(check-only 와 abort 는 녹화하지 않는다). 세 사이클에 공통으로 강제되는 규약은 하나뿐:
**첫 단계는 항상 입력 해제**. 위 예시는 `run_alarm_cycle` 의 목록이다.

### 4.3 F2 — 실패 cooldown (occupied 기구 재사용)

기존 `occupied_cooldown` dict 를 "이 tool 은 T 까지 쉰다" 범용 레지스트리로 확장한다.

**값 타입은 `float` 만료시각 그대로 둔다**(`(expiry, reason)` 튜플로 바꾸지 않는다).
`test_occupied_popup.py:173` 이 이 dict 를 위치인자로 넘기고, reason 은 로그 한 줄에만
쓰이므로 등록 시점에 출력하면 된다. 개념은 하나로 합치되 타입 변경은 없다.

**트리거**: 사이클이 정상 완료되지 않은 경우 — `run_status == "error"` 또는 runner 가
`failed_step` 으로 중단. **트리거 아님**: 정상 수행 후 correction fallback 으로 끝난
경우(엔지니어 인계라는 정상 경로이며, 이미 알람 해제까지 `active_tools` 에 머문다).

이 구분은 코드로 확인됨(2026-08-06): `_exec_run_correction`(`cycle.py:468-475`)은
`outcome.status` 와 무관하게 `"success"` 를 반환하고 **예외일 때만** `failed_step`
(`correction_error`)을 세운다. 따라서 fallback 계열 outcome 은 cooldown 을 유발하지
않는다 — 구현 시 이 성질에 의존해도 된다.

신규 env `ALIGN_FAIL_FAILURE_COOLDOWN_SEC`, 기본 300(=`ALIGN_FAIL_OCCUPIED_COOLDOWN_SEC`
와 동일). `Workflow3Settings` 필드로 추가(No-CLI-args 규약).

### 4.4 F3 — office 다운로드 경계

`gather_rcp_msr` 의 **동기 계약은 유지**한다(사이클이 읽기 전 디스크 적재 보장).
바뀌는 것은 대기가 bounded 라는 점이며, `success_gather.wait_for_gather()` 가 이미
쓰는 관용구를 재사용한다 — daemon thread + `join(timeout)` 후 결과와 무관하게 진행.

신규 env `ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC`, 기본 60.

**트레이드오프(명시)**: timeout 시 받은 만큼으로 진행하므로 assets 가 없거나 부분일 수
있고, feasibility 가 "보정 불가" 오판을 낼 수 있다 — 동기 설계가 막으려던 바로 그
실패 모드다. 알람 1건의 bounded 오답이 전체 루프의 무한 정지보다 낫다고 판단하되,
이 교환은 숨기지 않고 기록한다.

**in-flight 가드는 필수 동반**: timeout 된 스레드가 계속 쓰는 동안 사이클이 읽으므로,
`success_gather` 와 같이 동일 recipe 의 gather 가 진행 중이면 새 gather 를 건너뛴다.
이 가드 없이 timeout 만 넣으면 "보이는 정지"가 "조용한 부분읽기 경쟁"으로 바뀌어
원래 결함보다 나빠진다.

### 4.5 F5 — tool 별 가드

`process_fail_rows` 루프의 tool 처리 본문을 try/except 로 감싼다. 예외를 낸 tool 은
§4.3 의 cooldown 도 함께 받는다 — 그러지 않으면 다음 poll 에 같은 예외를 반복해
아무것도 고쳐지지 않는다.

### 4.6 적용 범위 — 두 모니터 모두

`align_fail_monitor_only_check.py:114` 는 **자체 `process_fail_rows` 사본**을 갖고 있고
cooldown 인자조차 없다. F2/F5 를 두 모니터에 동일하게 적용한다(코드가 평행해 delta 는
작고, check-only 도 오피스에서 무인 실행된다).

## 5. 데이터 흐름 / 오류 처리 요약

```
poll → filter → process_fail_rows
                  └─ per-tool try (F5)
                       ├─ gather_success_async        (기존 비차단)
                       ├─ gather_rcp_msr [bounded]    (F3: daemon+join(timeout)+in-flight)
                       ├─ run_alarm_cycle
                       │    └─ finally: run_teardown([input_unblock, recording_stop,
                       │                              close_tool, close_alert])   (F1/F4)
                       └─ 실패면 cooldown 등록, active 미등록                      (F2)
```

오류 처리 계약:
- teardown 단계 실패 = 치명적이지 않음. 로그 + `result.notes` 에 기록하고 계속.
- tool 처리 실패 = 그 tool 만 cooldown, 나머지 tool 계속.
- gather timeout = 경고 후 진행(assets 부분 가능성 명시).
- 루프는 어떤 경우에도 다음 poll 로 진행한다.

## 6. 테스트 (fault-injection 단위 테스트)

**제약**: Mac 에서 `RCS_MODULES_AVAILABLE` 이 False 라 `run_check_only_cycle` 은
`cycle.py:1588` 에서 조기 반환한다 → 사이클 전체 fault-injection 은 불가능하다.
§4.2 의 `_teardown_steps` 추출이 이 제약의 해법이다(RCS 스택 없이 목록을 만들어 검사).

| 파일 | 검증 |
|------|------|
| `monitor/test_teardown.py` (신규) | 던지는 단계가 뒤를 막지 않음 / 실패가 `(name, error)` 로 순서대로 반환 / 헬퍼는 절대 안 던짐 / 빈 목록 no-op |
| 순서 테스트 (동 파일) | 세 사이클 모두 `_teardown_steps(...)[0]` 이 입력 해제 단계 — **F1 의 회귀 테스트** |
| `monitor/test_failure_cooldown.py` (신규) | `run_status="error"` → cooldown 등록 + `active_tools` 미등록 / 만료 후 재시도 / **정상 fallback 은 cooldown 없음**(과잉 트리거 방지) / tool 1대 예외 시 나머지 처리됨 + 예외 tool cooldown |
| `monitor/test_rcp_gather_timeout.py` (신규) | timeout 초과 downloader 에서 bound 내 반환 / 동일 recipe 진행 중이면 새 gather skip |

기존 `test_occupied_popup.py` 는 수정 없이 통과해야 한다(§4.3 의 타입 유지 근거).

## 7. 설정 표면

| env | 기본 | 용도 |
|-----|------|------|
| `ALIGN_FAIL_FAILURE_COOLDOWN_SEC` | 300 | F2 실패 cooldown |
| `ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC` | 60 | F3 동기 다운로드 경계 |

둘 다 `Workflow3Settings` 필드 + `workflow_3_config.example.py` 항목으로 추가한다.

## 8. Non-goals

- watchdog/supervisor 도입(방안 C 기각).
- correction·CV·rerank 경로 변경(area 2).
- office 이음새 하드닝 — `ALIGN_IMAGES_DIR` 불일치, 어댑터 부재, RCS 창 탐색 실패(area 3).
- 장시간 무인 구동 검증 — keep-awake, foreground 탈취(area 4).

## 9. 검증의 한계 (명시)

이 spec 의 테스트는 **주입된 예외에 대해 루프의 제어 흐름이 살아남는지**를 증명한다.
`close_tool` 이 실제 RCS 창을 닫는지, `block_input(False)` 가 실제로 물리 입력을
푸는지는 증명하지 못한다 — 그것은 오피스 실행 확인 사항이다. 테스트 통과를 실장비
동작 보증으로 읽지 말 것.
