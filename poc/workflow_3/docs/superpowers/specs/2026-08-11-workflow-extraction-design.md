# 수동 녹화 → workflow 추출 (Workflow Extraction)

- 작성일: 2026-08-11
- 대상: `poc/workflow_3/recording_filter/` (Stage 2a 확장, Stage 2b 신규), `poc/workflow_3/workflow_extract/` (신규)
- 상태: 설계 승인 대기
- 선행: `2026-08-10-manual-recording-session-design.md` (녹화 + 사이드카 + Stage 1.5/2c)

## 1. 배경과 동기

`monitor/manual_record.py` 로 엔지니어 수동 조작을 녹화하는 경로는 완성됐고, 실제
세션 프레임도 확보했다. `recording_filter` 는 그 프레임을
`interaction_timeline.json` 까지 뽑는다 — **"언제 어디를 클릭했나"의 평평한 목록**이다.

없는 것은 그 다음이다. 타임라인은 이벤트를 하나씩 독립적으로 판정할 뿐, 이벤트 7번과
8번이 "PM 드롭다운을 열고 210 을 골랐다"는 하나의 조작이라는 것을 모른다. 절차로
읽으려면 **순서 의미(sequence semantics)** — 묶기, 순서, 의도 이름 — 가 필요하다.
이것은 지금까지의 프레임 단위 CV/VLM 작업과 다른 종류의 문제라서, Stage 2c 에 덧붙이는
것은 잘못된 이음매다.

선행 설계(`2026-08-10`, §6)는 workflow 추출과 재생기를 명시적으로 범위 밖에 뒀다.
이 문서가 그 중 **추출** 부분을 채운다. 재생기는 여전히 범위 밖이다.

## 2. 결정 사항 (브레인스토밍 합의, 2026-08-11)

| 항목 | 결정 | 근거 |
|------|------|------|
| 산출물 | `workflow.json` 이 진실, `workflow.md` 는 렌더 | 기계 소비(재생기)와 사람 판단(엔지니어 확인) 용도가 둘 다 필요 |
| step 단위 | **의미 단위 묶음** (평평한 목록, phase 계층 없음) | 10분 세션이 15~30 step. 클릭 1:1 은 60~150 step 이라 절차로 안 읽히고, 2계층 phase 는 국면 이름이 VLM 추측이라 오차원이 하나 더 는다 |
| 원본 보존 | 모든 step 이 `raw_events` 로 원본 이벤트 인덱스를 든다 | 그룹핑 규칙이 틀려도 되돌릴 수 있어야 한다 (§5 불변식) |
| 커서 출처 | 사이드카 `cursor_xy` 우선, 없으면 기존 VLM 경로 | 수동 세션의 로컬 커서는 엔지니어의 커서 그 자체 — 추론이 아니라 실측 |
| 타이핑 | **포커스 연결 OCR** (Stage 2b 신규) | 화면에 글자가 렌더되므로 OCR 로 값 복원 가능. 직전 클릭이 필드를 지정한다 |
| 더블클릭 | 기본 `click`. 라이브 박스 recenter 시그니처일 때만 `double_click(inferred)` | Linux 툴이라 더블클릭이 드물고, **SEM Monitor FOV 이동만** 더블클릭 (사용자 확인) |
| 입력 후킹 | **여전히 없음.** 마우스 버튼 상태도 폴링하지 않는다 | 선행 설계의 동의 약속을 그대로 유지. 더블클릭이 한 경우뿐이라 버튼 폴링의 값어치가 없다 |
| VLM 사용 | `workflow_extract` 는 **VLM 콜 0회** | 그룹핑 규칙은 튜닝 회차가 가장 많은 단계다. 재실행이 공짜여야 한다 |

## 3. 구조

```
recording/  (frames + frame_meta.jsonl + manifest)
   │
   ├── recording_filter/            기존 패키지, 2곳 추가
   │      Stage 1    frame_reduce        (무변경)
   │      Stage 1.5  region_gate         (무변경)
   │      Stage 2a   click_detect        ← 사이드카 커서 소스 추가
   │      Stage 2b   type_detect         ← 신규: 타이핑 구간 + 포커스 연결 OCR
   │      Stage 2c   element_label       (무변경)
   │      └─▶ interaction_timeline.json     (click + type_text 이벤트)
   │
   └── workflow_extract/            신규 패키지, 순수 소비자
          settings.py      WorkflowExtractSettings (WORKFLOW_EXTRACT_* env)
          steps.py         WorkflowStep dataclass + 스키마
          grouping.py      timeline events → steps  (순수 함수, I/O 없음, VLM 없음)
          render.py        steps → 한국어 markdown + step 별 표시 프레임
          extract_workflow.py   엔트리포인트
          └─▶ workflow.json + workflow.md + steps/*.jpg
```

**Stage 2b 가 `recording_filter` 에 있는 이유:** 프레임 단위 탐지이기 때문이다.
`timeline.py:32` 의 `build_timeline(click_events, typing_events=None, ...)` 과 이벤트
스키마의 `text` 필드는 선행 설계가 바로 이 용도로 미리 뚫어둔 이음매다. 새로 만들 것이
아니라 예약된 자리를 채운다.

**`workflow_extract` 가 VLM 을 쓰지 않는 이유:** 그룹핑은 타임라인이 이미 들고 있는
라벨만 읽고, 렌더링은 템플릿이다. 규칙 임계값을 만지며 수십 번 재실행할 단계이므로
네트워크 의존이 있으면 튜닝 자체가 비싸진다.

## 4. Stage 2a — 사이드카 커서 소스 (기존 모듈 확장)

현재 `click_detect._locate_cursor` 는 생존 프레임마다 VLM 을 1회 불러 커서 bbox 를
얻는다. 알람 녹화에는 사이드카가 없으니 그 길밖에 없지만, **수동 세션은
`frame_meta.jsonl` 에 `cursor_screen_xy` 가 이미 있다.**

변경은 순수 가산(additive)이다:

```
프레임의 meta 에서 커서를 해석할 수 있는가?
  ├ 예  → region_gate.screen_point_to_frame(cursor_xy, rect, frame_wh) 로 프레임 좌표 산출
  │        (VLM 콜 없음. cursor_visible=True, cursor_source="sidecar")
  └ 아니오 → 기존 VLM 경로 그대로 (cursor_source="vlm")
```

- 좌표 변환은 **이미 있는 함수**를 쓴다 (`region_gate.py:163`). DPI 배율 보정이 그
  안에 있고, 단순 뺄셈으로 좌표계를 섞는 버그(2026-08-10 FINDING 2)를 이미 고쳐뒀다.
- 클릭 판정(ROI 변화 픽셀 세기)은 **바뀌지 않는다**. 사이드카는 "어디"만 대신 답하고
  "눌렀는가"는 기존 cv2 로직이 그대로 판정한다.
- `cursor_source` 필드를 `ClickEvent` 와 타임라인에 남긴다 — 나중에 정확도를 따질 때
  실측 좌표와 VLM 추정 좌표를 섞어 보면 안 되기 때문이다.
- 알람 녹화(사이드카 없음)는 분기에 들어오지 않으므로 **동작이 오늘과 동일**하다.

부수 효과로 수동 세션의 Stage 2a VLM 콜이 사실상 0이 된다. 이것이 §5 그룹핑 규칙을
싸게 튜닝할 수 있게 하는 실질적 전제다.

## 5. Stage 2b — 타이핑 구간 탐지 (신규 모듈)

### 5.1 신호

타이핑은 **마우스가 멈춘 채 픽셀이 계속 바뀌는** 유일한 조작이다 — 클릭(커서 이동 후
1회 국소 변화)의 정확한 반대다.

```
구간 시작 조건:  cursor_xy 가 고정(이동 ≤ cursor_still_px)
                + 연속 change event 가 같은 작은 영역에 반복(≥ min_burst_events)
구간 종료 조건:  커서 이동, 또는 change 없음이 burst_idle_sec 이상 지속
```

### 5.2 커서 깜빡임(caret blink) 배제

텍스트 캐럿도 "커서 이동 없는 국소 반복 변화"라 같은 신호를 낸다. 판별은 OCR 두 번:

```
구간 시작 프레임의 필드 ROI → OCR → before
구간 종료 프레임의 필드 ROI → OCR → after

before, after 둘 다 비어 있음 → 캐럿 아님. **OCR 이 아무것도 못 읽은 것**이다.
                                 구간은 발행하고 값만 비운다(value_source="none")
before == after (둘 다 비지 않음) → 타이핑 아님(캐럿 깜빡임). 구간 폐기
before != after                   → 진짜 타이핑. value = after
```

구간당 OCR 2콜. 캐럿 판별과 값 복원을 같은 두 콜로 동시에 해결한다.

**빈 문자열 두 개를 "변화 없음"으로 묶지 않는 이유** (2026-08-11 Task 4 리뷰): ROI 가
어긋나거나 OCR 이 오독하면 양끝이 모두 `""` 로 나온다. 이것을 캐럿 깜빡임과 같이
취급하면 **실제 타이핑이 조용히 사라진다** — OCR 이 타이핑 값을 얻는 유일한 경로이므로
이 모듈에서 가장 비싼 실패 형태다. §8 의 "OCR 실패도 구간은 발행" 규칙과 같은 취지다.

또한 타이핑 이벤트의 `target_kind` 는 하드코딩하지 않고
`timeline.derive_target_kind(region, element_source)` 로 파생한다. 라벨 출처가 없으면
`unknown` 이어야 한다 - 그러지 않으면 OCR 이 실패한 타이핑이 "다른 장비로 이식 가능한
UI 컨트롤"이라고 잘못 주장한다.

### 5.3 필드 지정 (포커스 연결)

구간 직전 `focus_max_sec`(기본 2.0) 안에 클릭이 있으면 그 클릭이 필드를 지정한 것으로
보고, 그 클릭의 `element` 를 `target`, 클릭 좌표 주변을 필드 ROI 로 삼는다.

Tab/단축키로 포커스를 옮긴 경우 직전 클릭이 없다. 이때는 `target=null` 로 두되 **값은
그대로 복원한다** — 필드 이름을 추측해 채워 넣지 않는다(추측 라벨은 새 오차원이다).

## 6. workflow_extract — step 스키마

```json
{ "seq": 3,
  "action": "type_text",
  "target": "Recipe Name",
  "target_kind": "ui_control",
  "value": "MCD916_ALIGN_02",
  "value_source": "ocr",
  "coords_in_live_box": null,
  "t_sec": [52.1, 59.6],
  "generation": 0,
  "grouping_rule": "R3",
  "inferred": false,
  "raw_events": [23, 24, 25, 26],
  "frame": "seq_0284.jpg" }
```

| 필드 | 의미 |
|------|------|
| `action` | `click` \| `double_click` \| `select_from_dropdown` \| `type_text` \| `click_repeat` |
| `target` / `target_kind` | 라벨과 이식성 종류. `target_kind` 는 타임라인에서 그대로 승계 |
| `value` / `value_source` | 드롭다운 선택값 또는 타이핑 값. `value_source` ∈ `ocr` \| `vlm` \| `none` |
| `coords_in_live_box` | `live_image` step 만. **라이브 박스 내부 정규화 좌표** (0~1) |
| `t_sec` | `[시작, 끝]`. 단일 이벤트면 두 값이 같다 |
| `grouping_rule` | 이 step 을 만든 규칙 ID. 오작동 규칙을 특정하기 위함 |
| `inferred` | 관측이 아니라 결과로부터의 추론일 때 `true` (현재는 `double_click` 만) |
| `intent` | 규칙이 의도를 특정할 수 있을 때만. 현재는 R1 의 `fov_move` 하나 |
| `count` | `click_repeat` 만. 반복 횟수 |
| `raw_events` | 원본 타임라인 이벤트의 `seq` 목록 |

### 6.1 입력 파일

`workflow_extract` 는 `recording_filter` 산출 폴더에서 **세 파일**을 읽는다.

| 파일 | 쓰임 | 없을 때 |
|------|------|---------|
| `interaction_timeline.json` | step 의 원천 이벤트 | 에러 종료 (§8) |
| `region_map.json` | 세대별 `live_box` — R1 의 면적 비율, `coords_in_live_box` 정규화 | R1 degrade + 정규화 좌표 생략 (§8) |
| `change_events.json` | R1 이 클릭 직후의 변화 면적을 재는 데 쓴다 (Stage 1 통계) | R1 degrade (§8) |

**`coords_in_live_box` 를 창 픽셀 대신 정규화 좌표로 두는 이유:** 창 위치·크기에
독립적이고, 동시에 "이건 좌표가 아니라 영상 내용에 의존한다 → 재생하려면 CV 재해석이
필요하다"는 것을 스키마 자체로 드러낸다. 선행 설계 §6 의 이식성 구분과 같은 취지다.

## 7. 그룹핑 규칙

타임라인을 `t_sec` 순으로 좌→우 1회 순회하며, 각 위치에서 아래 우선순위대로 규칙을
시도하는 **greedy 단일 패스**다. 먼저 맞는 규칙이 이벤트를 가져간다.

| # | 규칙 | 트리거 | 산출 |
|---|------|--------|------|
| R1 | `double_click` (FOV 이동) | `region == "live_image"` 클릭 + 직후 `recenter_window_sec`(1.5s) 안의 change event 중 하나라도 **`change_bbox ∩ live_box` 면적이 `live_box` 면적의 `recenter_min_ratio`(0.40) 이상** | `action=double_click, intent=fov_move, inferred=true` |
| R2 | `select_from_dropdown` | ui_control **A** 클릭 → `dropdown_max_sec`(5.0) 안에 `dropdown_region_below(A)` 내부를 클릭한 **B** | `target=A.element, value=B.element` |
| R3 | `type_text` | Stage 2b 구간. 직전 `focus_max_sec`(2.0) 안의 같은 필드 클릭을 **포커스로 흡수** | `target, value, value_source` |
| R4 | `click_repeat` | **동일 대상**을 `repeat_window_sec`(6.0) 안에 `repeat_min_count`(3)회 이상 클릭 | `count=N` |
| R5 | `click` | 위에 안 걸린 나머지 | 1:1 |

**R4 의 "동일 대상" 정의:** `element` 라벨이 있으면 라벨이 같을 것. 라벨이 `null` 이면
클릭 좌표가 `same_target_px`(24px) 안에 있을 것. 라벨과 좌표를 섞어 비교하지 않는다 —
같은 버튼을 두 번 눌렀는데 한 번만 OCR 이 성공한 경우를 억지로 묶으면, 묶임 여부가
OCR 운에 좌우되어 재현되지 않는다.

### 7.1 R1 — 더블클릭을 결과로 판별하는 이유

프레임 주기가 실측 ~3-4fps(`poll_sec=0.2` + 처리시간)인데 사람의 더블클릭은 두 번의
누름이 150~250ms 간격이다. **두 누름이 한 프레임 간격 안에 들어가므로 시간 분리로는
구분할 수 없다** — 더 좋은 알고리즘의 문제가 아니라 그 정보가 애초에 샘플되지 않았다.

대신 결과가 남는다. FOV recenter 는 라이브 박스 전체를 다시 그리고, 라이브 박스 안의
단발 클릭(마커/선택)은 국소 변화만 남긴다. 사용자 확인에 따르면 이 툴에서 더블클릭은
**SEM Monitor FOV 이동 한 가지뿐**이므로 규칙 하나가 전체를 덮는다.

`inferred=true` 를 남기는 이유: 이것은 여전히 결과로부터의 추론이다. "버튼이 두 번
내려가는 것을 봤다"와 "그림이 움직였으니 아마 더블클릭"은 재생기가 구분할 수 있어야 한다.

### 7.2 R2 — 관측자와 실행기가 같은 기하를 쓴다

`sem_monitor/pm_dropdown.dropdown_region_below(button_xy, frame_wh)` 를 그대로 쓴다.
PM 드롭다운 실행기가 이미 쓰는 함수다. 관측이 인식할 수 있는 드롭다운 선택은 곧
실행기 기하가 수행할 수 있는 것이 되어, 관측 가능 범위와 재생 가능 범위가 어긋날 수
없다.

**단, 그 비율 상수는 PM 전용으로 보정돼 있다** (`PM_DD_LEFT_RATIO=0.04`,
`RIGHT=0.12`, `DOWN=0.45`). 더 넓은 다른 드롭다운은 놓칠 수 있다. 첫 실측 후 일반
비율셋이 필요한지 판단한다 — 지금 일반화하지 않는 이유는 실제로 어떤 드롭다운이
쓰이는지 데이터가 없기 때문이다.

### 7.3 불변식 (그룹핑 왜곡에 대한 안전장치)

> **모든 타임라인 이벤트는 정확히 하나의 step 의 `raw_events` 에 나타난다.
> 누락도 중복도 없다.**

코드에서 assert 하고 테스트로 덮는다. 이 불변식 덕에 잘못된 그룹핑은 **항상 되돌릴 수
있다** — `workflow.json` 만으로 기계적 클릭 단위 뷰를 재구성할 수 있고,
`grouping_rule` 이 있으므로 오작동한 규칙을 지목할 수 있다(의심에 그치지 않는다).

## 8. 오류 처리

`run_filter` 의 기존 원칙(조용한 성공을 보고하지 않는다)을 따른다.

| 상황 | 동작 |
|------|------|
| `interaction_timeline.json` 없음 | 에러 종료. `filter_recording.py` 를 먼저 돌리라고 안내 |
| 타임라인은 있으나 이벤트 0건 | `no_events`, **비정상 종료 코드** (`all_events_discarded` 와 같은 규율) |
| `region_map.json` 또는 `change_events.json` 없음 | R1 이 평범한 `click` 으로 degrade(라이브 박스나 변화 통계가 없으면 비율을 못 잰다). `coords_in_live_box` 도 생략. 1회 경고 후 계속 |
| Stage 2b OCR 실패 | `value=null, value_source="none"`. step 은 그대로 발행 |
| 그룹핑 불가 이벤트 | R5 로 떨어진다. **버리지 않는다** (§7.3 불변식) |
| 라벨 없는 클릭 (`element=null`) | `target=null` 로 발행. 라벨을 추측해 채우지 않는다 |

## 9. 테스트

전부 오프라인, Mac 에서 실행 가능, VLM 불필요. `recording_filter` 와 같은 pytest 스타일.

- `grouping.py` 는 dict 를 받아 dict 를 내는 순수 함수 → 합성 타임라인 픽스처로 덮는다.
  **R1~R5 각각 발화 테스트 + 비발화 테스트**(경계에서 잘못 발화하지 않는지).
- **불변식 테스트**: 입력 이벤트가 정확히 한 번씩 `raw_events` 에 나타난다.
- **캐럿 깜빡임 배제**: before/after OCR 텍스트가 같은 구간은 `type_text` 가 되면 안 된다.
- **사이드카 없는 회귀**: 사이드카 없는 입력(알람 녹화)에서 Stage 2a 동작이 오늘과
  동일해야 한다. 기존 71개 테스트가 지키는 행동을 보호하는 장치다.
- **R1 degrade**: `region_map.json` 이 없을 때 `double_click` 이 안 나오고 `click` 이 나온다.
- `render.py` 는 step 목록 → markdown 문자열이므로 스냅샷 비교로 덮는다.

## 10. 오피스 실행 순서

각 단계는 앞 단계가 정상일 때만 의미가 있다.

1. `RECORDING_FILTER_MAX_VLM_CALLS=300 uv run python poc/workflow_3/recording_filter/filter_recording.py`
2. **`region_map_gen0.jpg` 의 시안 박스가 실제 라이브 SEM 영역과 맞는지 확인.**
   틀리면 여기서 멈춘다 — R1 과 게이트 판정 전부가 무효다.
3. `summary.json` 의 `gate_passed / total_change_events` 확인 (90%+ 제거가 정상,
   0% 면 사이드카 조인 의심). `cursor_source` 분포도 본다 — 수동 세션인데 `vlm` 이
   대부분이면 사이드카 조인이 깨진 것이다.
4. `uv run python poc/workflow_3/workflow_extract/extract_workflow.py`
   (VLM 콜 0회 — 임계값을 만지며 몇 번이든 재실행)
5. **`workflow.md` 를 엔지니어와 함께 읽고 실제 질문을 던진다: "이게 당신이 한
   절차가 맞습니까?"** 이번 작업의 산출물은 자동화가 아니라 이 질문에 대한 답이다.

## 11. 한계 (`workflow.md` 푸터에 명시한다)

과신을 막기 위해 산출 문서 자체에 적는다.

| 항목 | 상태 |
|------|------|
| 키 입력 | **기록하지 않는다.** 화면에 렌더되는 값만 OCR 로 복원 |
| Enter/Tab/단축키 | 관측 불가 (화면 변화의 결과로만 간접 추정 가능, 하지 않는다) |
| 드래그 | 관측 불가 (버튼 눌림 상태를 폴링하지 않으므로 누름·이동·뗌을 구분 못 한다) |
| 스크롤/휠 | 관측 불가 (커서 위치에 픽셀 시그니처가 없다) |
| `double_click` | **관측이 아니라 추론** (`inferred=true`) |
| 라이브 영상 위 조작 | 좌표가 아니라 내용 의존 → 재생하려면 CV 재해석 필요 |

## 12. 범위 밖

- **재생기**: 이번에도 범위 밖이다. 선행 설계 §6 의 판단(만들 때 새로 만들 것이 거의
  없다 — `analyze_window_target` + row confirm gate 패턴)은 그대로 유효하다.
- **2계층 phase 구조**: 국면 이름이 VLM 추측이라 오차원이 하나 더 는다. 평평한 의미
  단위 step 이 먼저 실측으로 검증된 뒤에 다시 검토한다.
- **VLM 응답 캐시 + circuit breaker**: 2026-08-11 대화에서 설계했으나
  `deploy_vlms` 를 mai-ui/paddleocr 2종으로 줄여 OOM 압력이 낮아졌으므로 후속으로
  미룬다. 다만 `click_detect` 의 per-event `except Exception` 이 죽은 의존성까지
  삼켜 **조용한 성공**을 만드는 문제는 별개로 남아 있다(서버가 중간에 죽어도 exit 0 +
  그럴듯한 부분 타임라인). 별도 이슈로 추적한다.
- **Stage 2b 를 넘어선 값 복원**: 화면에 안 보이는 입력은 복원하지 않는다.
