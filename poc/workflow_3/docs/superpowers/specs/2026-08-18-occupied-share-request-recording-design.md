# 점유 tool 화면 공유 요청과 엔지니어 작업 녹화 (설계)

- 날짜: 2026-08-18
- 대상:
  - `poc/workflow_3/monitor/share_request.py` (신규)
  - `poc/workflow_3/monitor/occupied_popup.py`
  - `poc/workflow_3/monitor/cycle.py`
  - `poc/workflow_3/monitor/notify.py`
  - `poc/workflow_3/monitor/align_fail_monitor.py`
  - `poc/workflow_3/rcs/tool_row_verify.py`
  - `poc/workflow_3/vlm/prompts/prompt_share_options.py` (신규)
  - `poc/workflow_3/config.py`
- 상태: 설계 확정, 구현 전
- 선행 설계: `2026-08-06-loop-failure-path-hardening-design.md`

## 변경 이유

align fail 알람으로 tool 에 접속하려 할 때 다른 엔지니어가 이미 그 tool 을 점유하고 있으면
RCS 가 `Select` 팝업을 띄운다. 현재 구현(`occupied_popup.detect_select_popup`)은 이 팝업을
검출만 하고 세 옵션 중 무엇도 클릭하지 않은 채 접속을 포기한다. 그 결과 사이클은
`failure_class="rcs_occupied_select"` 로 실패하고 상위 루프가 cooldown 에 등록한다.

이 경로는 안전하지만, 프로젝트가 가장 필요로 하는 데이터를 버린다. 점유 중이라는 것은
**엔지니어가 지금 그 장비에서 align 을 수동으로 잡고 있다**는 뜻이고, 그 조작 과정이야말로
모방 학습과 절차 분석의 원천 데이터다. 지금은 그 순간마다 우리가 자리를 뜬다.

RCS 의 `Select` 팝업은 화면 공유 요청 옵션을 제공한다. 이 요청을 보내고 상대가 승낙하면
같은 제목의 Remote Monitoring 창이 열리므로, 기존 창 탐색과 `RecordingSession` 을 그대로
재사용해 그 엔지니어의 작업을 녹화할 수 있다.

## 확인된 도메인 사실

사용자 확인(2026-08-18):

- `Select` 팝업 옵션은 제어 공유 요청 / 화면 공유 요청 / 기존 사용자 강제 종료 요청이며,
  `Request` 와 `Cancel` 버튼이 있다.
- 화면 공유를 요청하면 **상대 엔지니어의 RCS 에 승인 팝업이 뜨고, 상대가 수락해야** 우리 쪽에
  화면이 열린다. 우리는 상대 화면에서 벌어지는 일을 제어하지 않으며 승낙/거절만 판별하면 된다.
- 거절되면 **그 엔지니어가 점유하는 동안 접근이 불가**하다.
- 수락되면 열리는 창은 **같은 `Remote Monitoring System - <EQP>` 창**이다. 제어 공유인지
  화면 공유인지에 따라 조작 가능 여부가 갈리며, 화면 공유는 관전만 가능하다.
- 한 번 승낙된 뒤에는 팝업 없이 **바로 접근**되는 경우가 있다.
- RCS List 탭에는 **점유자 ID/이름 컬럼**이 있어 누가 쓰고 있는지 읽을 수 있다.

## 문제의 핵심

점유 상태는 두 갈래이며, 두 번째가 기존 코드에 조용한 오보를 만든다.

| 경로 | 화면 | 현재 코드의 반응 |
|---|---|---|
| (a) 점유 + 미승인 | `Select` 팝업이 뜬다 | 검출 → 접속 포기 |
| (b) 점유 + 이미 승낙됨 | 팝업 없이 창이 열린다 | **접속 성공으로 판단** |

(b)에서 열린 창이 화면 공유(view-only)면 우리 클릭은 장비에 먹지 않는다. 그런데
`correct_align_fail_auto` 는 클릭 결과를 화면으로 되읽지 않는 open-loop actuator 라, 클릭이
무시된 사실을 모른 채 `corrected` 를 반환할 수 있다. `notify_correction_outcome` 은
`corrected` 면 cube 알림을 생략하므로, **보정되지 않았는데 아무도 통보받지 않는** 상태가 된다.

이는 새 기능이 만드는 결함이 아니라 이미 잠복해 있던 구멍이다. 다만 지금은 우리가 공유 요청을
보내지 않아 (b)에 도달할 일이 드물었을 뿐이고, 요청을 보내기 시작하면 (b)가 상시 경로가 된다.
따라서 이 설계는 공유 요청 기능과 view-only 판별을 **함께** 넣어야 한다.

## 목표

1. `Select` 팝업이 뜨면 "화면 공유 요청" 을 선택하고 `Request` 를 눌러 요청을 보낸다.
2. 잘못된 옵션(특히 강제 종료)을 누르지 않도록 클릭 전에 라벨을 확인한다.
3. 승낙되면 그 세션을 녹화한다. 거절/무응답이면 기존 cooldown 경로로 돌아간다.
4. view-only 세션에서는 보정을 수행하지 않으며, 수행한 것처럼 보고하지도 않는다.

## 비목표

- 제어 공유(`share control`) 요청. 화면 공유만 다룬다.
- 기존 사용자 강제 종료. 어떤 경로로도 클릭하지 않으며, 오히려 명시적 거부 대상이다.
- 상대 엔지니어 쪽 승인 UI 에 대한 어떤 조작이나 추론.
- 공유 세션에서의 자동 보정. 관전과 녹화만 한다.

## 선택한 접근

### A. detector 와 actuator 를 다른 모듈로 분리 (채택)

`occupied_popup.py` 는 지금 그대로 **검출 전용**으로 둔다. 이 모듈은 모든 예외를 `False` 로
흡수하는 fail-open 정책이며, 그것이 옳다 — 검출 실패가 접속을 막으면 안 된다.

클릭하는 쪽은 `share_request.py` 로 새로 만든다. actuator 는 fail-**closed** 여야 한다 —
확신이 없으면 누르지 않는다. 두 정책을 한 파일에서 유지하면 어느 쪽 규칙이 적용되는 코드인지
읽는 사람이 매번 판별해야 한다.

### B. 한 모듈에 검출과 클릭을 함께 (기각)

호출부가 하나 줄어든다는 이점이 있으나, 위의 상반된 오류 정책이 한 파일에 섞인다. 특히
"예외를 삼켜 False" 라는 기존 관용구가 actuator 경로로 새어 들어가면 클릭 실패가 조용히
성공으로 보고될 수 있다.

## 설계

### ① `monitor/share_request.py` — 팝업 actuator

```text
request_screen_share(vlm_client, settings) -> ShareRequestResult
  status ∈ {requested, confirm_failed, not_found, blocked_safe_mode, error}
```

절차:

1. `Select` 창을 찾아 캡처한다.
2. 2단계 로케이터(`mai-ui` coarse → `mai-ui` fine)로 두 좌표를 얻는다.
   - "Request to share the screen" 라디오
   - "Request" 버튼
3. **확인 게이트** — 각 좌표 주위를 좁게 crop 해 `vlm/label_verify.py` 로 OCR 한다.
4. 두 확인이 모두 통과할 때만 라디오 → `Request` 순으로 클릭한다.

로케이터가 주는 점은 **이미지 픽셀 좌표**이므로, 클릭 직전 반드시 `image_point_to_screen`
으로 창 rect / 이미지 크기 배율 보정을 거친다. 이 변환을 빠뜨리면 확인 게이트가
**무의미해진다** - 점 A 의 라벨을 읽고 점 B 를 누르게 되어, 오피스 125/150% 배율에서
어긋난 클릭이 하필 강제 종료 라디오에 떨어질 수 있다. 변환 실패는 클릭하지 않는다.
(2026-08-18 code-review 에서 실제로 이 구멍이 잡혔다 - 주입식 `click_fn` 덕에 단위
테스트는 전부 통과하고 있었다. 회귀: `test_share_click_converts_image_point_to_screen`.)

확인 게이트 판정 규칙:

| 대상 | 통과 조건 | 즉시 중단 조건 |
|---|---|---|
| 라디오 | `share` 와 `screen` 토큰이 **둘 다** 읽힘 | `control` 또는 `terminat` 토큰이 읽힘 |
| 버튼 | `request` 토큰이 읽힘 | `cancel` 토큰이 읽힘 |

정책 `ALIGN_FAIL_SHARE_CONFIRM` 은 세 값을 받는다.

| 정책 | 판정 |
|---|---|
| `strict` (기본) | 통과 조건을 만족해야만 클릭. 못 읽었으면 클릭하지 않는다 |
| `lenient` | 즉시 중단 토큰이 읽히지 않았으면 클릭. 못 읽은 것은 통과로 본다 |
| `off` | 확인하지 않고 클릭 (진단용) |

`tool_row_verify` 의 기본값이 `lenient` 인 것과 달리 여기는 `strict` 를 기본으로 한다 —
남의 세션에 영향을 주는 클릭이므로 "읽지 못했으면 누르지 않는다" 가 옳은 기본값이다.
`off` 는 오피스에서 좌표 자체를 진단할 때만 쓰며 운영 기본값이 아니다.

`terminat` 을 명시적 거부 사유로 두는 이유는 라디오 3개가 세로로 인접해 fine 단계가 한 칸
어긋나는 것이 가장 현실적인 실패 형태이고, 그 어긋남의 최악이 강제 종료이기 때문이다.

확인 실패 시 팝업에 아무 조작도 하지 않고 `confirm_failed` 를 반환한다. 호출부는 이를 기존
포기 경로와 같게 처리한다.

### ② 승낙/거절 판별

`Request` 클릭 후 `share_wait_sec`(기본 45초) 동안 폴링한다.

- 제목에 `eqp_id` 를 가진 Remote Monitoring 창 등장 → `accepted`
- 시간 초과 → `denied_or_timeout`

거절과 무응답을 하나의 상태로 합친다. 거절 시 RCS 가 무엇을 보여주는지 확정되지 않았고,
어느 쪽이든 결론은 "그 엔지니어가 점유하는 동안 접근 불가" 로 동일해 동작이 갈리지 않기
때문이다. 두 경우의 구분은 manifest 에 남겨 오피스 확인 후 필요하면 분리한다.

`accepted` 가 아니면 팝업을 닫고 기존 `rcs_occupied_select` cooldown 으로 간다.

**팝업은 `Cancel` 버튼 좌표를 찍어 클릭하지 않고 `close_window()` 로 닫는다.** 좌표 클릭은
로케이션이 방금 실패했을 수도 있는 시점에 같은 팝업을 다시 겨냥하는 것이라, 확인 게이트를
통과하지 않은 클릭을 fail-closed 원칙을 어기고 최악의 순간에 내보내게 된다. 창 핸들은 이미
가지고 있으므로 VLM 도 좌표도 필요 없다.

### ③ 점유 3-상태 판별과 view-only

점유는 참/거짓이 아니라 **3-상태**다. "모른다" 를 "비어 있다" 로 접으면 조용한 오보가 된다.

| 상태 | 판별 | 보정 | outcome |
|---|---|---|---|
| `occupied_by_other` | 점유자 컬럼에서 ID 를 읽음, 또는 이번 사이클에서 우리가 공유 요청을 보냄 | 건너뜀 | `view_only_observation` |
| `free` | 점유자 컬럼 읽기 성공 + 점유자 토큰 없음 | 수행 | 기존 그대로 |
| `unknown` | 점유자 컬럼 읽기 실패 | 수행 | `corrected_unverified` |

`unknown` 에서 보정을 막지 않는 이유는, 먹지 않는 클릭 자체는 무해하고 진짜 피해는
**"보정했다" 고 보고하며 알림을 생략하는 것**이기 때문이다. 그래서 클릭을 막는 대신
불확실성을 outcome 에 실어 **알림이 반드시 나가게** 한다. 오피스 OCR 캘리브레이션이
끝나기 전에도 조용한 성공이 불가능해진다.

이 판별에는 **cross-cycle 기억을 두지 않는다.** 승낙 이력을 TTL 캐시로 들고 있으면, 동료가
작업을 끝내고 tool 을 놓아준 뒤에도 그 EQP 가 view-only 로 굳어 보정·알림·재시도가 한꺼번에
죽는다(2026-08-18 oc-discuss 에서 발견). 판별은 매 사이클 새로 한다.

#### 점유자 컬럼은 반드시 별도 crop 으로 읽는다

`tool_row_verify` 가 이미 하는 행 strip OCR 을 넓혀 한 번에 읽고 싶은 유혹이 있으나,
**해서는 안 된다.** `classify_tokens` 는 목표 ID 를 못 읽은 상태에서 ID 모양 토큰
(`_looks_like_tool_id`: 영숫자 + 글자·숫자 혼재 + 길이 범위)을 만나면 `mismatch` 를 낸다.
점유자 ID(`KIM0234` 등)가 정확히 그 모양이다. 게다가 `accepts()` 는 `lenient` 에서도
`mismatch` 를 거부한다. 따라서 crop 을 넓히면 지금은 무해하게 통과하던 `unreadable` 이
`mismatch` 로 **승격**되어 정상 행의 클릭이 거부되고, 그것도 점유자 컬럼이 채워진 행에서만
— 이 기능이 겨냥한 바로 그 케이스에서 — 발생한다. `PointTextRead.tokens` 는 좌표를 버리므로
사후에 컬럼별로 나눌 수도 없다.

따라서 `read_row_occupant()` 는 **자기 crop 과 자기 OCR 호출**을 가진다. `tool_row_verify`
는 손대지 않는다. 비용은 접속당 OCR 1회 추가이며, 사이클이 이미 수 분 단위(녹화 300초)인
것에 비하면 무시할 수 있다.

### ④ `cycle.py` 흐름

`_exec_wait_tool_window` 의 점유 감지 분기를 확장한다.

```text
abort_check 가 Select 팝업 감지
  share_request_enabled ?
    request_screen_share()
      requested        -> wait_share_response()
                            accepted          -> occupancy=occupied_by_other
                                                 창 대기 계속 -> success
                            denied_or_timeout -> close_window(popup)
                                                 -> failed(rcs_occupied_select)
      confirm_failed   -> 팝업 미조작 + 진단 산출물 저장
                          -> failed(rcs_share_confirm_failed)
      not_found/error  -> failed(rcs_occupied_select)
  아니면
    기존 failed(rcs_occupied_select)
```

팝업이 뜨지 않은 정상 경로에서는 `select_tool` 단계의 `read_row_occupant()` 결과가
`context["occupancy"]` 를 채운다. 즉 (a)는 위 분기가, (b)는 행 읽기가 채우며, 둘 다
실패하면 `unknown` 이다.

`run_correction` 단계는 `context["occupancy"]` 에 따라 갈린다. `occupied_by_other` 면 보정을
건너뛰고 `CorrectionOutcome(status="view_only_observation")` 을, `unknown` 이면 보정을
수행하되 결과 status 를 `corrected_unverified` 로 바꿔 넣는다. 이것이 이 설계에서 가장
중요한 부분이다 — 없으면 위의 "문제의 핵심" 에서 서술한 조용한 오보가 그대로 발생한다.

녹화와 engineer watch 는 수정하지 않는다. 두 새 status 모두 `corrected` 가 아니므로 기존
조건(`cycle.py:740`, `outcome.status != "corrected"`)이 그대로 watch 를 태워
`engineer_watch_sec`(300초) cap 녹화가 돈다. 코드 확인 결과 `corrected` 비교는
`notify.py:286` 과 `cycle.py:740` 두 곳뿐이고 **모두 정확 비교(`==`/`!=`)** 이므로 새 status
가 접두사로 스며들 여지는 없다.

### ⑤ 알림

**cube 알림을 생략하지 않는다.** 초안은 "그 엔지니어가 이미 붙어 있으니 알림은 방해" 라고
보았으나 두 전제가 틀렸다. cube 는 **이 알람의 담당 엔지니어**에게 가는데 tool 앞의 사람은
다른 사람일 수 있고, 점유자가 알람을 남긴 채 자리를 뜨면 아무도 통보받지 못한다. 알림 채널을
"지금 누가 서 있는가" 라는 추측으로 바꾸는 셈이다.

대신 문구를 구분한다.

| status | cube |
|---|---|
| `corrected` | 발송 안 함 (기존) |
| `view_only_observation` | "다른 엔지니어 점유 중 - 관전·녹화만 수행" |
| `corrected_unverified` | "점유 여부 확인 불가 - 보정을 시도했으나 반영 여부 미확인" |

### ⑥ 상위 루프

`align_fail_monitor.py`:

- `rcs_share_confirm_failed` 를 `_RETRY_LATER_FAILURE_CLASSES` 에 추가한다. 확인 실패는
  장비 탓이 아니라 우리 인식 실패이므로 `wrong_tool_opened` 과 같은 성격이다.
- **outcome 기반 재시도 집합을 새로 둔다.**

  ```python
  _RETRY_LATER_OUTCOME_STATUSES = {"view_only_observation", "corrected_unverified"}
  ```

  `_cycle_failed()` 는 `run_status`/`failed_step` 만 보므로 완주한 사이클은 실패가 아니고,
  그대로 두면 `active_tools` 에 등록되어 **알람이 해제될 때까지 영영 재시도되지 않는다.**
  점유자가 tool 을 놓아준 뒤에도 우리는 돌아가지 않는다. 두 status 는 어디에서도 성공으로
  등록하지 않고 cooldown 경로로 보낸다 — 그래야 tool 이 풀렸을 때 실제 보정이 돌아간다.

- **재시도 상한을 둔다.** cooldown 300초 + 사이클당 녹화 300초이므로, 알람이 오래 유지되면
  시간당 약 6회 재시도와 그만큼의 cube 가 나가고 단일 RCS 커서를 계속 점유한다.
  EQP 별 연속 `view_only`/`unverified` 횟수를 세어 `share_max_attempts`(기본 2)를 넘으면
  `active_tools` 로 넘겨 멈춘다. 카운터는 알람 해제 시 `active_tools`/`occupied_cooldown` 과
  같은 시점에 정리한다.

### ⑦ 환경 변수

기본 ON 으로 배포한다. 안전은 env 게이트가 아니라 확인 게이트가 담당한다.

| 변수 | 기본 | 의미 |
|---|---|---|
| `ALIGN_FAIL_SHARE_REQUEST` | `1` | 공유 요청 발송 활성화 |
| `ALIGN_FAIL_SHARE_CONFIRM` | `strict` | 확인 게이트 정책 |
| `ALIGN_FAIL_SHARE_WAIT_SEC` | `45` | 승낙 대기 상한(초) |
| `ALIGN_FAIL_SHARE_MAX_ATTEMPTS` | `2` | EQP 별 연속 view-only 재시도 상한 |

승낙 대기를 90초에서 45초로 낮춘 이유는 이 대기가 `_exec_wait_tool_window` 안에서
**블로킹**이고, 단일 RCS 커서를 모든 tool 의 알람이 직렬로 공유하기 때문이다. 점유 tool
하나의 대기가 다른 모든 장비의 알람 처리 지연으로 그대로 전가된다.

`SAFE_MODE=0` 은 실클릭 조건으로 계속 필요하다. `align_fail_monitor` 가 이미 그 기본값으로
뜨므로 운영자가 추가로 할 일은 없다.

## 오류 처리

| 상황 | 동작 |
|---|---|
| 확인 게이트 미통과 | 클릭 없음 + **진단 산출물 저장**, `rcs_share_confirm_failed`, cooldown |
| 라디오 클릭 후 `Request` 클릭 실패(예외) | `close_window()` 로 팝업 닫아 원상 복구, cooldown |
| 승낙 대기 timeout | `close_window()` 로 닫고 `rcs_occupied_select`, cooldown |
| 점유자 컬럼 OCR 실패 | `unknown` → 보정은 수행, outcome `corrected_unverified`, cube 발송 |
| `share_request` 예외 | 삼키지 않고 `error` 로 반환, 사이클은 기존 포기 경로 |

`strict` 를 기본값으로 두는 대가로 **확인 실패는 반드시 자기 진단이 되어야 한다.** 게이트가
막을 때마다 팝업 crop, OCR 원문 응답, 로케이터가 찍은 좌표를 `debug_images/` 에 저장한다.
Mac 에서는 팝업을 볼 수 없으므로, 첫 오피스 실행이 실제 문구를 알려주는 유일한 경로다. 토큰
매칭은 대소문자 무시 부분 일치로 하고 국문 표기도 함께 받는다.

## 테스트

VLM 과 실장비 없이 Mac 에서 도는 순수 로직으로 작성한다 (`monitor/test_share_request.py`).

- 확인 게이트: `share`+`screen` 통과, `control` 거부, `terminat` 거부, `cancel` 거부
- 정책 3종(`strict` / `lenient` / `off`) 별 판정
- 좌표 미검출, OCR 빈 응답, OCR 예외
- 승낙 판별: 창 등장 → `accepted`, timeout → `denied_or_timeout`
- 점유 3-상태: 점유자 토큰 있음/없음/읽기 실패 → `occupied_by_other`/`free`/`unknown`
- `occupied_by_other` → `run_correction` skip 회귀 (`cycle` 쪽)
- `unknown` → 보정 수행하되 status 가 `corrected_unverified` 로 치환되는 회귀
- 두 새 status 모두 cube 가 **발송되는** 회귀 (`notify` 쪽 — 생략되면 실패)
- 두 새 status 모두 `active_tools` 가 아니라 cooldown 으로 가는 회귀 (`align_fail_monitor`)
- 재시도 상한: 연속 `share_max_attempts` 회 후 `active_tools` 로 전환, 알람 해제 시 리셋

`tdd` 스킬로 진행한다 (전역 지침의 "테스트 우선 개발" 항목).

## 오피스 확인이 필요한 항목

설계는 아래를 보수적 기본값으로 잡아두었으므로 틀려도 위험하지 않으나, 첫 오피스 실행에서
확인하고 조정해야 한다.

1. **거절 시 RCS 화면** — 지금은 무응답과 합쳐 timeout 으로 처리한다. 거절이 별도 팝업이나
   메시지로 나타난다면 대기를 일찍 끊을 수 있다.
2. **List 점유자 컬럼의 위치와 폭** — `read_row_occupant()` 의 crop 기하. 여기가 안 맞으면
   대부분의 사이클이 `unknown` → `corrected_unverified` 로 떨어진다(안전하지만 시끄럽다).
3. **확인 게이트의 실제 OCR 토큰** — 팝업 옵션 문구가 영문인지 국문인지, 줄바꿈으로 잘리는지.
   확인 실패 시 저장되는 진단 산출물이 이 답을 준다.

## 설계 변경 이력

- 2026-08-18 초안 작성.
- 2026-08-18 `oc-discuss`(glm-5.3, 3라운드) 후 7개 항목 수정. 상세는
  `docs/opencode/2026-08-18-occupied-share-request-debate.md`.
  요지: view-only 판별의 fail-open 근거가 틀렸고(재시작 후 (b) 경로), grant 캐시는 해제된
  tool 을 view-only 로 굳혔으며, cube 생략은 점유자와 알람 담당자를 동일시했고, 좌표로 Cancel
  을 누르는 것은 fail-closed 원칙 위반이었고, 행 strip crop 확장은 기존 게이트의
  `unreadable` 을 `mismatch` 로 승격시켜 정상 클릭을 거부했을 것이다. 추가로 새 outcome
  status 의 재시도 생애주기가 정의되지 않아 완주 사이클이 `active_tools` 로 굳는 구멍과,
  점유 오판독 시의 cube spam 상한이 빠져 있었다.
