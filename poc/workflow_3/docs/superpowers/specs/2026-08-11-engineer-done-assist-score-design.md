# engineer-done 판정을 Assist Window score 색상으로 (설계)

- 날짜: 2026-08-11
- 대상: `poc/workflow_3/sem_monitor/assist_score.py` (신규), `poc/workflow_3/monitor/engineer_done_align_adjustment.py` (개조)
- 상태: 설계 확정, 구현 전

## 배경

engineer-done 감지기는 align fail 후 엔지니어가 수동 정렬을 끝냈는지를 판단해, watch 를
조기 종료하고 tool 창을 닫는다. 오탐의 대가는 **엔지니어가 작업 중인데 창이 닫히고 녹화가
끊기는 것**이라 비대칭적으로 나쁘다.

현재 구현은 Recipe Monitor 의 측정 카운터 `N/M` 만 본다:

```
n >= min_count(6)  and  _last_n is not None  and  n >= _last_n
```

이 판정의 오탐 경로를 세어보니 7가지가 나왔고, 그중 셋은 구조적이었다.

| # | 경우 | 통과하는 이유 |
|---|---|---|
| 1 | 분모를 읽음 | `_INT_RE = r"\d+"` 가 첫 정수만 봄. `350` 은 크고 상수라 두 조건 동시 충족 |
| 2 | 이전 런의 잔존 카운터 | N 이 낮게 시작해야 한다는 요구가 없음. 시작 시 `350/350` 이면 즉시 충족 |
| 3 | 정의 간극 | 감지기가 답하는 건 "측정이 **시작**됐다". 필요한 건 "엔지니어가 **끝냈다**" |
| 4 | ROI 오grounding | grounding 1회 + 확인 게이트 없음. 인접 숫자 필드를 잡으면 통과 가능 |
| 5 | 등호가 정지를 진행으로 인정 | `n >= _last_n`. 변화 게이트는 숫자가 아니라 crop 픽셀이라 커서·리페인트로도 발화 |
| 6 | `_last_n` 이 재grounding 을 건너 생존 | 리셋 목록에서 이 필드만 빠져 옛 ROI 값과 새 ROI 값을 비교 |
| 7 | OCR 숫자 병합/환각 | 작은 crop 이 완화책이나 0은 아님 |

1은 실제로는 확률이 낮다. crop 이 N 과 M 을 함께 담으면 변화 게이트는 N 이 바뀔 때 발화하고,
그 순간 OCR 이 하필 M 을 첫 정수로 돌려줘야 하는 동시 우연이 필요하다. 단독 상수 필드만
잡혔다면 변화 자체가 없어 OCR 까지 가지도 않는다.

3이 근본이다. "측정 시작"과 "엔지니어 완료"는 다른 사건이다. 엔지니어는 정렬을 시험하려
측정을 돌려보고, 실패를 보고, 다시 조정할 수 있다.

## 도메인 사실 (오피스 확인, 2026-08-11)

- Recipe Monitor **Assist Window** 는 **tool 창 내부 패널**이다. 별도 최상위 창이 아니므로
  기존 `capture_window(tool_window)` 캡처에 그대로 들어온다.
- 레이아웃: **3열 × 7행**. 열 = Addressing1 / Addressing2 / Measurement. 각 칸에 측정
  이미지 썸네일과 score 숫자가 실시간으로 찍힌다. 7행 = 최신 7회 측정.
- **Addressing2 는 대개 비어 있다** (거의 없음). Addressing1 과 Measurement 가 중요하다.
- **score 숫자 색이 판정 신호다**: 검정 = 정상 측정, 빨강 = 측정 실패.
  **Addressing1 또는 Measurement 중 하나라도 빨강이면 그 측정은 실패.**
- 색 상태는 검정/빨강 **둘뿐**이다. 측정이 진행 중인 행은 **빈칸**이다.
- 측정 1회는 레시피에 따라 **2~5초**.
- 최신 행이 **맨 아래**에 쌓인다.
- 측정점이 6개 미만인 레시피는 **사실상 없다** (있어도 0.01% 미만).

## 판정 정의

> 문제 없이 측정이 계속 진행되면 engineer-done 으로 인정한다.
> "문제 없이" = Assist Window 의 score 가 연속 6회 정상(검정).
> 단, watch 시작 이후 실제로 새로 진행된 측정이어야 한다.

빨강이 하나 나오면 연속 streak 은 0부터 다시 센다.

## 접근 선택

두 축을 각각 골랐다.

**축 1 — "새 측정 6회"를 무엇으로 세는가: 카운터 + Assist 하이브리드 (채택)**

- 대안 B(Assist 단독, 카운터 폐기)는 ROI 가 하나로 줄고 카운터 관련 오탐이 전부 사라지지만,
  "6회"가 창 크기에 묶여 사실상 7회가 되고 score 값 지문 충돌이라는 새 실패 모드가 생긴다.
- 대안 C(행 추가 이벤트 증분 추적)는 6회를 정확히 세지만 스크롤로 밀리는 행의 시퀀스 정합이
  필요해 셋 중 가장 복잡하다.
- 하이브리드를 고른 이유: 카운터는 이미 있고, 절대 진행량을 주는 유일한 신호다. 그리고
  아래 오탐 표대로 카운터의 기존 오탐이 차분 판정에서 대부분 자동 소멸한다.

**축 2 — 폴링마다 무엇을 실행하는가: CV-per-poll (채택)**

- 영역 로케이트만 VLM 1회(`analyze_window_target`), 이후 폴링은 순수 CV.
- 대안(OCR-per-poll)은 8초마다 VLM 호출이 들어가고 PaddleOCR 환각 리스크를 떠안는다.
  우리가 필요한 건 값이 아니라 **색**이므로 값을 읽지 않으면 그 리스크가 설계 단계에서 없어진다.
- 색 판정이 순수 CV 라는 점은 프로젝트 설계 규칙("VLM 은 영역만 식별, 정량 판단은 CV")과 맞다.

## 구조

`sem_monitor/` 계층에 판독 능력을 두고 `monitor/` 가 소비한다(4-layer DAG 유지:
util → {vlm, runner} → {align, rcs, sem_monitor, ...} → monitor).

```
sem_monitor/assist_score.py                                     [신규]
  locate_assist_layout(window, title, backend, image) -> AssistLayout | None
      VLM 1회. 패널 박스 + 열 박스(Addressing1/Addressing2/Measurement). 캐시된다.
  read_row_states(image, layout) -> list[RowState]
      순수 CV. 행 분할(수평 투영) -> 열별 잉크 색 분류.
  classify_ink(crop) -> "black" | "red" | "blank"
      chroma 기반. sem_box_detect 의 grey/chroma 판정과 같은 계열.
  ok_streak(rows) -> int
      최신 행부터 연속 ok 개수. 순수 함수.

monitor/engineer_done_align_adjustment.py                       [개조]
  ROI grounding + 카운터 OCR 경로는 유지. 판정식만 교체.
```

`RowState` = `{addr1, addr2, meas: "black"|"red"|"blank"}` + 파생 `verdict`:

- `fail` — addr1 또는 meas 가 red
- `pending` — 판정에 필요한 칸이 blank (측정 진행 중)
- `ok` — 그 외 (addr2 는 비어 있어도 무방)
- `unknown` — 색 분류가 흑/적 경계에 걸림

## 알고리즘

폴링마다 캡처 1회로 카운터와 Assist 를 함께 읽는다.

```python
n = read_counter(image)                 # 기존 경로
rows = read_row_states(image, layout)   # 순수 CV
if n is None or not rows:
    return False

if baseline_n is None:                  # watch 시작 기준점
    baseline_n = n
    return False

delta = n - baseline_n                  # watch 시작 이후 새 측정 횟수
streak = ok_streak(rows)                # 화면상 최신 연속 정상 개수

done = (delta >= min_delta) and (streak >= ok_streak_required)
```

두 조건은 각각 다른 일을 한다. `delta` 는 잔존 카운터를 걷어내고, `streak` 은 품질을 본다.
하나만으로는 둘 다 막지 못한다.

### 임계값(6) < 창 크기(7) 이 주는 성질

화면은 **항상 판정에 필요한 증거를 전부 담고 있다.** 빨강이 화면 밖으로 밀려났다는 것은 그
뒤로 이미 7회 이상 정상이 지났다는 뜻이고, 그건 어차피 done 조건을 만족한다. 따라서:

- 폴링 사이에 행을 놓칠 걱정이 없다.
- 스크롤 부기(어느 행이 새 행인지 추적)가 필요 없다.
- 폴링 간격 튜닝이 판정 정확도에 영향을 주지 않는다.

`streak` 은 매 폴링 화면에서 새로 계산되므로 폴링 간 누적 상태가 없다. 유지되는 상태는
`baseline_n` 하나뿐이다.

### 행 방향

`ASSIST_NEWEST_ROW_AT = "bottom"` (모듈 상수). 오피스 확인 완료 — 최신 행이 맨 아래다.
상수로 남겨두는 이유는 tool 버전에 따라 달라질 경우 한 줄로 대응하기 위해서다.

## 오탐 처리 대응표

| # | 경우 | 새 설계에서의 처리 |
|---|---|---|
| 1 | 분모 오독 | ΔN=0 이라 영영 done 이 아님 (무해) |
| 2 | 잔존 카운터 | `delta >= 6` 이 차단 |
| 3 | 정의 간극 | `streak >= 6` 이 "문제 없이 진행 중"을 직접 측정 |
| 4 | ROI 오grounding | 엉뚱한 숫자가 6 이상 증가 **그리고** Assist 6행 검정을 동시 요구 |
| 5 | 등호가 정지를 통과 | `>= 6` 차분이 대체 |
| 6 | `_last_n` 생존 | 폴링 간 상태가 `baseline_n` 하나뿐이고, 재grounding 시 무효화한다 |
| 7 | OCR 환각 | 색만 보므로 판정 경로에서 제거 |

## 에러 처리

실패는 전부 `False` 를 돌려주고 watch cap 이 안전망 역할을 한다.

| 상황 | 처리 |
|---|---|
| `locate_assist_layout` 실패 | 경고 1회 + 이번 watch 동안 감지 비활성 -> cap 대기 |
| 행 분할이 0행 또는 8행 이상 | layout 무효화 후 1회 재로케이트, 그 폴링은 `False` |
| 카운터 읽기 실패 | `False`. 3연속 실패 시 재grounding (기존 동작 유지) |
| 재grounding 발생 | `baseline_n = None` 로 무효화 -> 새 ROI 에서 baseline 재설정 |
| 색 분류 애매 | 그 행은 `unknown` -> streak 을 끊는다 (ok 로 세지 않음) |
| 그 외 예외 | 삼키고 `False` |

판정의 대가가 비대칭이라 모든 애매함은 "아직 아님" 쪽으로 넘어간다. 늦게 감지하면 cap 까지
몇 분 더 기다릴 뿐이지만, 잘못 감지하면 엔지니어 작업 중에 창이 닫힌다.

## 설정

`Workflow3Settings` (`ALIGN_FAIL_*` 네임스페이스):

| 필드 | 기본 | 비고 |
|---|---|---|
| `engineer_done_ok_streak` | 6 | 신규. 연속 정상 요구 횟수 |
| `engineer_done_min_delta` | 6 | 신규. watch 시작 이후 최소 새 측정 횟수 |
| `engineer_done_min_count` | — | **제거**. 절대값 기준이 사라짐 |
| `engineer_done_detect_enabled` | False | 유지. 오피스 검증 전까지 off |

모듈 상수: `ASSIST_NEWEST_ROW_AT` ("bottom").

## 디버그 산출물

폴링마다가 아니라 **판정이 바뀔 때만** 오버레이 1장을 저장한다: 패널 박스 + 열 박스 +
행별 verdict. 오피스가 행 방향·열 매핑·색 임계를 한 장으로 검증할 수 있다.
`debug_artifacts` 를 쓰므로 콘솔 스팸은 없다.

## 테스트

전부 합성 이미지 기반이라 Mac 에서 돈다.

1. `classify_ink` — 검정 글자 / 빨강 글자 / 빈칸 / 회색 경계값
2. `read_row_states` — 합성 7행 x 3열 패널 -> RowState 목록 (Addressing2 없는 케이스 포함)
3. `ok_streak` — 경계: 정확히 6, 5, 최신행 fail, 중간 fail, pending·unknown 혼재
4. 판정 통합 — 잔존 카운터(시작 시 7행 검정 + delta=0 -> False), delta 6 + streak 5 -> False,
   둘 다 충족 -> True
5. 재grounding 시 baseline 무효화
6. 기존 `test_engineer_done_align_adjustment.py` 18/18 유지

기존 `capture_fn`/`ground_fn`/`ocr_fn` 주입 패턴을 그대로 따라 `rows_fn` 주입점을 추가하면
통합 판정 테스트도 실이미지 없이 돌아간다.

## 범위 밖

- 확인 게이트(오탐 #4의 잔여분): grounding 점이 정말 카운터인지 OCR 로 확인하는 장치는 넣지
  않는다. 이중 신호 요구로 실질적으로 닫히고, 넣으면 폴링마다 OCR 이 다시 들어온다.
- Assist Window 가 없는 tool: 감지 비활성 + cap 대기로 충분하다. 별도 폴백을 만들지 않는다.
- 측정점이 6개 미만인 레시피: streak 6 에 도달하지 못해 영영 감지되지 않는다. 오피스 확인
  결과 그런 레시피는 사실상 없다(0.01% 미만). 설령 걸려도 실패 방향이 안전하다(cap 까지
  대기 = 기존 동작). 데이터가 없어 미룬 것이 아니라, 데이터를 보고 다루지 않기로 한 것이다.
