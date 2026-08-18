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

`Request` 클릭 후 `share_wait_sec`(기본 90초) 동안 폴링한다.

- 제목에 `eqp_id` 를 가진 Remote Monitoring 창 등장 → `accepted`
- 시간 초과 → `denied_or_timeout`

거절과 무응답을 하나의 상태로 합친다. 거절 시 RCS 가 무엇을 보여주는지 확정되지 않았고,
어느 쪽이든 결론은 "그 엔지니어가 점유하는 동안 접근 불가" 로 동일해 동작이 갈리지 않기
때문이다. 두 경우의 구분은 manifest 에 남겨 오피스 확인 후 필요하면 분리한다.

`accepted` 가 아니면 팝업을 `Cancel` 로 닫고 기존 `rcs_occupied_select` cooldown 으로 간다.

### ③ view-only 판별 — 이중 안전장치

| 경로 | 판별 방법 | 신뢰도 |
|---|---|---|
| (a) 우리가 요청 → 승낙 | 우리가 보냈으므로 확실히 안다 | 확정 |
| (b) 팝업 없이 진입 | List 행의 점유자 ID 컬럼 OCR | 캘리브레이션 필요 |

(b)를 위해 `select_tool` 이 이미 확정한 행 좌표를 재사용해 strip 을 조금 넓게 잘라 점유자
컬럼을 읽는다. 새 좌표를 만들지 않으므로 "좌표는 VLM 이 정하고 OCR 은 확인만 한다" 규칙을
지킨다. 컬럼 위치와 폭은 오피스 캘리브레이션 항목이므로 `read_row_occupant()` 한 지점에
격리한다.

읽기 실패 시에는 경고와 manifest 기록 후 기존 동작(보정 수행)을 유지한다. fail-closed 로
잡으면 OCR 이 흔들릴 때 보정이 영영 돌지 않게 되는데, (b)는 우리가 요청을 보낸 이력이
있어야 도달 가능한 상태라 (a)의 확정 판별이 대부분을 덮는다.

추가로 `share_grant_cache: {eqp_id: 만료 epoch}` 를 두어, 우리가 승낙받은 EQP 는 TTL 동안
(b)로 진입해도 view-only 로 기억한다. 프로세스 재시작 시 캐시는 사라지며 그때는 List 읽기가
유일한 신호다.

### ④ `cycle.py` 흐름

`_exec_wait_tool_window` 의 점유 감지 분기를 확장한다.

```text
abort_check 가 Select 팝업 감지
  share_request_enabled ?
    request_screen_share()
      requested        -> wait_share_response()
                            accepted          -> view_only=True, 창 대기 계속 -> success
                            denied_or_timeout -> Cancel 로 닫기 -> failed(rcs_occupied_select)
      confirm_failed   -> 팝업 미조작 -> failed(rcs_share_confirm_failed)
      not_found/error  -> failed(rcs_occupied_select)
  아니면
    기존 failed(rcs_occupied_select)
```

`run_correction` 단계는 `context["view_only"]` 가 참이면 보정을 건너뛰고
`CorrectionOutcome(status="view_only_observation")` 을 넣는다. 이것이 이 설계에서 가장
중요한 한 줄이다 — 없으면 위의 "문제의 핵심" 에서 서술한 조용한 오보가 그대로 발생한다.

녹화와 engineer watch 는 수정하지 않는다. `view_only_observation` 은 `corrected` 가 아니므로
기존 조건(`cycle.py`, outcome != corrected)이 그대로 watch 를 태워
`engineer_watch_sec`(300초) cap 녹화가 돈다.

### ⑤ 알림

`notify.py` 의 cube 생략 대상에 `view_only_observation` 을 추가한다. 그 엔지니어가 이미
장비에 붙어 작업 중인 상황에서 "OK 를 눌러달라" 는 알림은 도움이 아니라 방해다. manifest 와
audit log 에는 그대로 남긴다.

### ⑥ 상위 루프

`align_fail_monitor.py`:

- `rcs_share_confirm_failed` 를 `_RETRY_LATER_FAILURE_CLASSES` 에 추가한다. 확인 실패는
  장비 탓이 아니라 우리 인식 실패이므로 `wrong_tool_opened` 과 같은 성격이다.
- 공유 세션 녹화가 정상 완료된 사이클은 실패가 아니므로 `active_tools` 에 등록된다. 알람이
  해제될 때까지 같은 EQP 를 다시 붙잡지 않는다 — 한 번 녹화했으면 충분하다.

### ⑦ 환경 변수

기본 ON 으로 배포한다. 안전은 env 게이트가 아니라 확인 게이트가 담당한다.

| 변수 | 기본 | 의미 |
|---|---|---|
| `ALIGN_FAIL_SHARE_REQUEST` | `1` | 공유 요청 발송 활성화 |
| `ALIGN_FAIL_SHARE_CONFIRM` | `strict` | 확인 게이트 정책 |
| `ALIGN_FAIL_SHARE_WAIT_SEC` | `90` | 승낙 대기 상한(초) |
| `ALIGN_FAIL_SHARE_GRANT_TTL_SEC` | `7200` | 승낙 기억 TTL(초) |

`SAFE_MODE=0` 은 실클릭 조건으로 계속 필요하다. `align_fail_monitor` 가 이미 그 기본값으로
뜨므로 운영자가 추가로 할 일은 없다.

## 오류 처리

| 상황 | 동작 |
|---|---|
| 확인 게이트 미통과 | 클릭 없음, `rcs_share_confirm_failed`, cooldown 후 재시도 |
| 라디오 클릭 후 `Request` 클릭 실패(예외) | `Cancel` 로 팝업 닫아 원상 복구, cooldown |
| 승낙 대기 timeout | `Cancel` 로 닫고 `rcs_occupied_select`, cooldown |
| 점유자 컬럼 OCR 실패 | 경고 + manifest 기록, 기존 동작 유지 |
| `share_request` 예외 | 삼키지 않고 `error` 로 반환, 사이클은 기존 포기 경로 |

## 테스트

VLM 과 실장비 없이 Mac 에서 도는 순수 로직으로 작성한다 (`monitor/test_share_request.py`).

- 확인 게이트: `share`+`screen` 통과, `control` 거부, `terminat` 거부, `cancel` 거부
- 정책 3종(`strict` / `lenient` / `off`) 별 판정
- 좌표 미검출, OCR 빈 응답, OCR 예외
- 승낙 판별: 창 등장 → `accepted`, timeout → `denied_or_timeout`
- `view_only` → `run_correction` skip 회귀 (`cycle` 쪽)
- `view_only_observation` → cube 생략 회귀 (`notify` 쪽)

`tdd` 스킬로 진행한다 (전역 지침의 "테스트 우선 개발" 항목).

## 오피스 확인이 필요한 항목

설계는 아래 둘을 보수적 기본값으로 잡아두었으므로 틀려도 위험하지 않으나, 첫 오피스 실행에서
확인하고 조정해야 한다.

1. **거절 시 RCS 화면** — 지금은 무응답과 합쳐 timeout 으로 처리한다. 거절이 별도 팝업이나
   메시지로 나타난다면 대기를 일찍 끊을 수 있다.
2. **List 점유자 컬럼의 위치와 폭** — `read_row_occupant()` 의 crop 기하.
3. **확인 게이트의 실제 OCR 토큰** — 팝업 옵션 문구가 영문인지 국문인지, 줄바꿈으로 잘리는지.
   토큰 매칭 규칙은 이에 맞춰 조정한다.
