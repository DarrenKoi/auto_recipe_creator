# engineer-done 우선순위 신호와 Assist 판독 보강 (설계)

- 날짜: 2026-08-12
- 대상:
  - `poc/workflow_3/monitor/engineer_done_align_adjustment.py`
  - `poc/workflow_3/monitor/cycle.py`
  - `poc/workflow_3/monitor/recording.py`
  - `poc/workflow_3/sem_monitor/assist_score.py`
  - `poc/workflow_3/recording_filter/`
- 상태: 설계 확정, 구현 전
- 선행 설계: `2026-08-11-engineer-done-assist-score-design.md`

## 변경 이유

기존 구현은 engineer-done 을 다음 두 조건의 AND 로 판정한다.

```text
watch 시작 후 numerator delta >= 6
AND
Assist 최신 정상 streak >= 6
```

오피스 확인 결과 numerator 는 측정 실패 중에도 계속 증가한다. 따라서 numerator 는 측정이
문제없이 진행된다는 품질 신호가 아니다. Assist 를 읽을 수 있을 때 numerator 를 동급 조건으로
강제하면 정상 완료를 늦추고, Assist 를 읽을 수 없을 때 numerator 를 단독 품질 신호로 취급하면
실패 측정을 정상으로 오인할 수 있다.

또한 엔지니어가 작업 완료를 확신하면 Remote Monitoring 창 우상단 X 를 직접 눌러 창을
나간다. 이 행동은 모델 추정보다 강한 명시적 완료 신호다. 단, 원격 쪽 마우스 움직임과 엔지니어
쪽 움직임이 분리되어 있고 화면 가장자리에서는 X 모양 커서의 아래쪽 절반만 보일 수 있어,
마지막 프레임만 보고 커서를 확정하는 것은 어렵다.

## 확인된 도메인 사실

- Recipe Monitor Assist Window 에서 필요한 열은 `Measurement` 와 선택적인
  `Addressing1` 이다.
- `Measurement` 는 항상 존재하며 최종 측정 품질을 나타낸다.
- `Addressing1` 은 레시피에 따라 없을 수 있다. 존재하고 빨강일 때만 실패 veto 로 쓴다.
- `Addressing2` 는 engineer-done 판정에 필요하지 않다.
- `Measurement` 는 창 오른쪽 벽에 거의 붙어 있어 헤더 전체가 보이지 않거나 OCR 결과가
  `Measuremen`, `Measu` 같은 접두 조각일 수 있다.
- 현재 디버그 이미지의 `assist_panel_crop_region` 은 오피스에서 올바른 영역을 담는다.
  crop 기하는 바꾸지 않는다.
- 현재 2단계 로케이터의 기본 서비스는 실제로 `mai-ui > mai-ui` 다. 디버그 이름의
  `ui_venus` 는 옛 단계명일 뿐, 현재 ui-venus 서비스를 호출한다는 뜻이 아니다.
- Remote Monitoring 창을 닫으면 녹화 캡처가 실패하고, 현재 `RecordingSession` 은 연속
  실패 약 5초 후 `stop_reason="window_gone"` 으로 끝난다.

## 목표

engineer-done 은 다음 우선순위로 판정한다.

1. 엔지니어가 Remote Monitoring 창을 닫음: 즉시 완료.
2. Assist 를 신뢰성 있게 읽음: Assist 품질 판정으로 완료 또는 계속 대기.
3. Assist 를 반복해서 읽을 수 없음: numerator 가 세 번 연속 엄격히 증가할 때 자동 완료.
4. 어느 신호도 확정하지 못함: 기존 `engineer_watch_sec` cap 까지 대기.

## 선택한 접근

### A. 우선순위 상태기계 (채택)

Assist 와 numerator 를 동시에 OR 로 묶지 않고, Assist 가 사용 가능하면 항상 Assist 를
우선한다. numerator 는 Assist 판독이 연속 실패했을 때만 활성화한다.

장점:

- 화면에 빨간 실패가 보이는데 numerator 증가가 이를 덮는 경로가 없다.
- `Addressing2` 또는 잘린 전체 헤더 때문에 Assist 전체를 포기하지 않는다.
- 현재 detector 의 안전 cap 과 graceful failure 방식을 유지한다.

### B. Assist OR numerator 병렬 판정 (기각)

구현은 단순하지만 측정 실패 중에도 numerator 가 증가하므로, 읽을 수 있는 빨간 Assist 를
numerator 가 덮고 창을 닫을 수 있다.

### C. numerator 우선 + Assist veto (기각)

Assist 가 잠깐 unreadable 이거나 stale 하면 veto 가 사라진다. 품질 신호가 보조 역할로
밀려 도메인 우선순위와 반대다.

## 완료 판정 상태기계

### 1. 명시적 창 닫기

`RecordingSession.stop_reason == "window_gone"` 은 엔지니어가 Remote Monitoring 을
직접 닫은 것으로 취급한다. `_engineer_watch` 는 녹화 스레드 종료를 보고 즉시 빠져나온다.
이 경로는 이미 존재하므로 의미가 드러나는 로그와 테스트만 보강한다.

`max_sec`, 예외, 외부 `stop()` 같은 다른 녹화 종료 사유를 엔지니어 완료라고 부르지 않는다.
watch 루프 종료 자체는 기존대로 수행하되, 결과/로그에는 실제 `stop_reason` 을 남긴다.

### 2. Assist primary

Assist layout 은 다음 두 열만 필수로 만든다.

- `Measurement`: 필수
- `Addressing1`: 선택

`Addressing2` 는 OCR 에 잡혀도 디버그 정보로만 남기며 grid/완료 판정의 필수조건에서 뺀다.
`Measurement` 헤더는 영숫자 정규화 후 최소 5자 접두 일치를 허용한다. 예를 들어
`Measurement`, `Measurement:`, `Measuremen`, `Measu` 를 같은 열로 본다. 5자 미만 조각은
오탐 가능성이 커서 인정하지 않는다.

행 판정:

```text
Measurement red                         -> fail
Addressing1 red (열/값이 존재할 때)      -> fail
Measurement blank                       -> pending
Measurement unknown 또는 Addressing1 unknown -> unknown
그 외                                   -> ok
```

완료는 최신부터 연속 `ok` 6행을 요구한다. watch 시작 전부터 남아 있던 검정 7행으로 즉시
완료하지 않도록, 첫 유효 Assist 관측을 baseline 으로 저장하고 그 뒤 Measurement 영역의
픽셀 지문이 최소 한 번 바뀐 뒤에만 streak 완료를 허용한다. verdict 목록이 모두 `ok` 여도
점수/썸네일 픽셀이 바뀌면 새 측정 활동으로 인정한다.

Assist 에서 `fail` 을 한 번이라도 실제로 관측하면 `assist_failure_seen=True` 를 watch 동안
유지한다. 이 상태에서는 numerator fallback 을 사용하지 않는다. 이후 Assist 가 다시 읽혀
최신 정상 6행을 직접 확인한 경우에만 primary 경로로 완료할 수 있다.

### 3. numerator fallback

Assist 가 다음 이유로 **3회 연속 unusable** 일 때만 fallback 을 연다.

- 패널 grounding 실패
- `Measurement` 헤더/열 식별 실패
- grid 생성 실패
- Measurement 셀들이 전부 `blank`/`unknown` 이라 품질을 판정할 수 없음

Assist 가 한 번이라도 정상적으로 읽히면 unusable streak 은 0으로 리셋한다. 빨강을 읽은
경우에도 Assist 는 usable 이므로 fallback 으로 내려가지 않는다.

fallback 완료 조건은 세 번의 연속 OCR reading 이 엄격한 증가열을 만드는 것이다.

```text
10 -> 11 -> 12    완료
10 -> 10 -> 11    미완료, 10에서 새 sequence 시작
10 -> 12 -> None  미완료, sequence 리셋
10 -> 9  -> 10    미완료, 9에서 새 sequence 시작
```

즉 세 개의 성공 reading `n1 < n2 < n3` 를 요구한다. 증가 폭은 묻지 않는다. 동일값,
감소, OCR 미검출, ROI 재grounding 은 연속 증가 sequence 를 리셋한다.

`assist_failure_seen=True` 면 numerator 가 계속 증가해도 fallback 완료를 내지 않는다.

## Assist crop 과 오른쪽 경계

오피스에서 `assist_panel_crop_region` 이 잘 맞는 것이 확인되었으므로 기존 point 중심 crop
비율과 좌표를 유지한다. 오른쪽 벽에 가까운 `Measurement` 대응은 crop 확대/이동이 아니라:

1. `Addressing2` 필수조건 제거
2. `Measurement` 접두 헤더 허용
3. 숫자 bbox 가 보이면 그 x 범위를 우선 사용
4. 끝까지 Measurement 를 식별하지 못하면 Assist unusable 로 처리 후 bounded fallback

으로 해결한다. 검증된 crop 을 움직여 다른 패널을 섞는 새 위험을 만들지 않는다.

## 우상단 X / 커서의 last-resort 확인

이 로직은 **완료를 결정하지 않는다**. 창 닫힘은 `window_gone` 이 이미 결정하며, 커서 추정은
녹화 후 행동 복원에서 "엔지니어가 X 를 눌렀다"는 설명을 보강하는 증거일 뿐이다.

마지막 성공 캡처에서 VLM 이 커서를 찾지 못했을 때만 다음 조건을 모두 확인한다.

1. 바로 뒤 녹화 종료 사유가 `window_gone` 이다.
2. 마지막 인접 프레임의 변화 bbox 가 창 우상단 close-button 영역과 겹친다.
3. 마지막 프레임 우상단에 가장자리에서 잘린 X/커서 후보가 있다.

세 조건을 만족하면 `probable_close_click` 로 기록하고 근거를
`window_gone + top_right_change + cursor_vlm_missing` 로 남긴다. 좌표는 우상단 후보 영역으로
표시하되 confidence 는 낮게 둔다.

고정 close-button X 는 항상 존재하므로, 정적 한 프레임이나 "VLM 이 못 찾았다" 하나만으로
이 fallback 을 발화하지 않는다. 이 추정 결과로 클릭을 재생하거나 창을 닫지도 않는다.

## detector 인터페이스

현재 `rows_fn() -> list[RowState]` 만으로는 "Assist 없음"과 "7행 pending"을 구분할 수 없다.
판독 결과를 명시적인 observation 으로 바꾼다.

```python
AssistObservation(
    status="usable" | "unusable",
    rows=list[RowState],
    panel_fingerprint=str | None,
    reason=str,
)
```

`EngineerDoneDetector` 는 다음 watch-local 상태만 가진다.

```text
assist_unusable_streak
assist_baseline_fingerprint
assist_changed_since_start
assist_failure_seen
numerator_increase_sequence
```

Assist 판독과 numerator OCR 의 localization/throttle 은 서로 독립적이어야 한다. numerator
grounding 실패가 Assist primary 판정을 막지 않고, Assist 실패가 numerator 샘플 수집 자체를
막지 않게 한다.

## 디버그 산출물

한 calibration/alarm run 의 모든 Assist 파일을 동일한 run-specific `debug_dir` 에 저장한다.
현재처럼 locator 파일은 전역 `debug_images/assist_score`, OCR/grid 파일은
`engineer_done_calib/...` 로 갈라지지 않게 한다.

실제 서비스와 단계를 이름에 드러낸다.

```text
assist_panel_capture.jpg
assist_panel_coarse_mai_ui_response.txt
assist_panel_refine_mai_ui_response.txt
assist_panel_locator_overlay.jpg
assist_panel_crop_region.jpg
assist_ocr_read_<reason>.jpg
assist_ocr_read_<reason>.json
assist_grid_<seq>.jpg
```

locator overlay 에는 다음을 서로 다른 색과 label 로 한 장에 그린다.

- `coarse_mai_ui_bbox`
- `assist_panel_crop_region`
- `refine_mai_ui_point`

coarse bbox 와 refine point 가 겹치는 것은 정상이다. overlay/JSON 에 실제
`coarse_service="mai-ui"`, `refine_service="mai-ui"` 를 함께 남겨 옛 `ui_venus` 단계명과
실제 서비스 호출을 혼동하지 않게 한다.

numerator fallback 은 reading, 현재 sequence, reset 이유를 저장한다. top-right cursor
fallback 은 원본 마지막 프레임, change bbox, 후보 ROI, `window_gone` 근거를 같은 run 에
남긴다.

## 에러 처리

| 상황 | 처리 |
|---|---|
| Remote Monitoring `window_gone` | 명시적 완료로 engineer watch 종료 |
| Assist usable + red | 미완료, numerator fallback 금지 |
| Assist usable + 6 ok + fresh change | 완료 |
| Assist unusable 1~2회 | 미완료, 재시도 |
| Assist unusable 3회 이상 | numerator fallback 평가 가능 |
| numerator 3연속 증가 | Assist failure 미관측 때만 완료 |
| 모든 신호 불확정 | watch cap 까지 대기 |
| cursor VLM 미검출 | 완료 판정에는 영향 없음; window_gone 동반 시 last-resort 증거만 시도 |

## 테스트

### `sem_monitor/test_assist_score.py`

1. `Measurement` + `Addressing1` 만으로 grid 생성
2. `Measurement` 단독으로 grid 생성
3. `Addressing2` 누락이 실패 원인이 아님
4. `Measu` / `Measuremen` 오른쪽 잘림 접두 인식
5. `Meas` 같은 5자 미만 조각 거부
6. Addressing1 blank + Measurement black -> ok
7. Addressing1 red 또는 Measurement red -> fail

### `monitor/test_engineer_done_align_adjustment.py`

1. Assist 6 ok 이지만 baseline 이후 변화 없음 -> 미완료
2. Assist 6 ok + baseline 이후 fingerprint 변화 -> 완료
3. Assist red 관측 후 numerator 3연속 증가 -> 미완료
4. Assist unusable 2회 + numerator 증가 -> 미완료
5. Assist unusable 3회 + reading `10, 11, 12` -> 완료
6. `10, 10, 11`, `10, 12, None`, `10, 9, 10` -> 미완료
7. Assist 가 다시 usable 이 되면 unusable streak 리셋
8. numerator ROI 재grounding 시 증가 sequence 리셋

### `monitor` / `recording_filter`

1. `window_gone` 으로 watch 가 조기 종료되고 명시적 완료 로그를 남김
2. `max_sec` 종료를 엔지니어의 X 클릭으로 기록하지 않음
3. cursor VLM 미검출 + top-right change + `window_gone` -> `probable_close_click`
4. 고정 X 만 있고 변화 없음 -> fallback 없음
5. top-right change 가 있어도 `window_gone` 이 아니면 fallback 없음

## 오피스 검증

1. Remote Monitoring 에서 측정 중인 tool 로 calibration 실행
2. `assist_panel_crop_region` 이 기존처럼 올바른지 확인
3. `Measurement` 전체 또는 접두가 인식되고 grid 가 오른쪽 score 에 놓이는지 확인
4. 검정 6행과 새 panel 변화 후 Assist primary 로 완료되는지 확인
5. Assist locator 를 의도적으로 실패시켜 numerator `n1 < n2 < n3` fallback 확인
6. 빨간 Measurement 가 보이는 동안 numerator 가 증가해도 완료되지 않는지 확인
7. 엔지니어가 우상단 X 를 눌렀을 때 약 5초 안에 `window_gone` 으로 종료되는지 확인
8. 마지막 커서가 안 잡힌 경우에만 `probable_close_click` 증거가 생성되는지 확인

## 범위 밖

- 우상단 X 를 자동 클릭하는 동작
- 추정한 `probable_close_click` 을 GUI 재생에 사용하는 동작
- 검증된 `assist_panel_crop_region` 기하 변경
- `Addressing2` 판정 복원
- ui-venus 서비스 재도입

