# RCS 비디오-투-액션 구현 가이드 (2026-03-13)

## 문서 목적

이 문서는 아래 목표를 실제 구현 관점으로 풀어 쓴 가이드다.

- `AVI` 녹화 영상에서 엔지니어의 수동 조작 구간을 추출한다.
- 수동 조작 구간을 `episode -> step -> target -> effect -> objective` 형태의 trajectory 로 바꾼다.
- trajectory 를 검색 가능한 메모리로 저장한다.
- 현재 `RCS` 화면에 대해 과거 trajectory 를 검색해 VLM에게 다음 마우스/키보드 액션을 결정하게 한다.
- 액션 후에는 step verification 으로 성공/실패를 판정한다.

이 문서는 코드를 작성하지 않고도 팀이 구현 범위, 모듈 분할, 입출력 스키마, 검증 기준을 합의할 수 있도록 만드는 것을 목표로 한다.

## 구현 범위

이번 구현의 1차 범위는 아래로 제한한다.

1. `video-only` 환경을 전제로 한다.
2. `AVI` 안에서 **흰색 마우스 커서**가 보인다고 가정한다.
3. 1차 action 범위는 `click`, `double_click`, `drag`, `scroll`, `type_candidate`, `wait`, `verify`, `ask_human` 으로 제한한다.
4. 실제 키보드 원시 입력은 복원하지 않는다.
5. first-class target 은 `recipe setup` 과 직접 관련된 `input`, `button`, `tab`, `dropdown`, `table row`, `list row`, `dialog action`, `parameter cell` 이다.

## 구현하지 않을 것

아래는 1차 구현 범위에서 제외한다.

- 실제 `keydown` / `keyup` 로그 수준의 키 이벤트 복원
- 강화학습 또는 대규모 fine-tuning
- 일반 Windows 전체 자동화 프레임워크와의 깊은 통합
- RCS 외 일반 desktop agent 범용화
- 모든 tool UI 변형에 대한 완전한 rule set

## 핵심 산출물

구현이 끝나면 최소 아래 산출물이 있어야 한다.

1. `manual-control episode` 목록
2. episode 별 `step trajectory`
3. step 별 `pre_frame`, `post_frame`, `cursor`, `action_type`, `target`, `effect`
4. retrieval 가능한 trajectory 저장소
5. `next_action` JSON planner prompt/spec
6. action 후 verification 규칙 세트
7. 평가 리포트

## 전체 시스템 그림

```text
AVI videos
-> low-rate scan
-> manual-control episode mining
-> high-rate episode decoding
-> cursor tracking
-> event extraction
-> step segmentation
-> target grounding + OCR diff
-> objective summarization
-> trajectory storage
-> retrieval
-> next-action planning
-> action execution
-> verification
```

## 기존 저장소와의 연결

현재 저장소에서 직접 연결되는 지점은 아래와 같다.

- [`/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/extractor.py)
  - 영상 열기, 프레임 추출
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py)
  - action sequence 추출 자리
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py)
  - `ActionSequence`, `FrameData`, `RAGContext` 등 데이터 모델
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_context_manager.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_context_manager.py)
  - 현재 화면과 과거 trajectory 연결
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_prompt_builder.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_prompt_builder.py)
  - trajectory 를 VLM 프롬프트로 주입
- [`/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py)
  - 현재 screenshot 기반 state / UI 분석
- [`/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/pipeline_ocr.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/pipeline_ocr.py)
  - OCR sidecar
- [`/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/flask_vlm.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/flask_vlm.py)
  - `UI-Venus`, `PaddleOCR-VL` 등 팀 기본 route 설정

즉, 구현 방향은 기존 코드를 폐기하는 것이 아니라 아래처럼 확장하는 것이다.

- `test/video_frame_parser/`는 오프라인 데이터 추출 계층
- `test/vlm_input_control/`는 retrieval / prompt 계층
- `poc/work2/`는 online execution / verification 계층

## 단계별 구현 가이드

## 1. 데이터 인벤토리 정리

첫 단계는 모델보다 데이터 정리다.

### 목적

- 2주 분량의 AVI가 실제로 어떤 해상도, FPS, duration, codec 분포를 가지는지 파악한다.
- tool 별, UI 종류별, shift 별, 엔지니어 별 편차를 파악한다.

### 해야 할 일

1. 모든 AVI에 대해 메타데이터를 수집한다.
2. 아래 필드를 테이블로 만든다.

| 필드 | 설명 |
|------|------|
| `video_id` | 내부 고유 ID |
| `file_path` | 파일 위치 |
| `tool_id` | 어떤 장비인지 |
| `rcs_session_id` | 세션 식별자 |
| `engineer_id` | 엔지니어 식별자 |
| `start_time` | 시작 시간 |
| `duration_sec` | 재생 시간 |
| `fps` | 프레임 레이트 |
| `width` | 너비 |
| `height` | 높이 |
| `codec` | 코덱 |
| `notes` | 비고 |

3. tool 유형별로 대표 샘플 영상을 5~10개 고른다.
4. 대표 샘플에 대해 사람이 눈으로 보며 아래를 확인한다.

- 커서 형태가 일정한가
- RCS 오버레이가 있는가
- UI zoom / scaling 이 일정한가
- 글자가 OCR 가능한 수준인가
- engineer 개입이 자주 일어나는 화면 종류가 무엇인가

### 완료 기준

- 최소 1개의 메타데이터 CSV 또는 테이블이 있다.
- 대표 샘플 영상 세트가 정리되어 있다.
- tool / UI 별 변형 포인트가 문서화되어 있다.

## 2. Manual-Control Episode Mining

이 단계의 목표는 전체 영상 중 수동 조작이 있었던 구간만 남기는 것이다.

### 왜 필요한가

전체 영상의 90%가 자동 운전이면, 전체를 같은 비용으로 처리하는 것은 낭비다.
먼저 수동 조작 후보 episode 를 잘 찾는 것이 전체 파이프라인 효율을 좌우한다.

### 입력

- 전체 AVI 영상

### 출력

- `manual_control_episode` 목록

### 권장 방식

#### 2.1 저속 1차 스캔

- 전체 영상을 `1~2 fps`로 훑는다.
- 각 프레임에 대해 아래 feature 를 계산한다.

| feature | 설명 |
|---------|------|
| `global_change_score` | 전체 프레임 변화량 |
| `local_change_score` | 상위 N개 패치 변화량 |
| `cursor_presence_score` | 커서 후보 존재 여부 |
| `cursor_motion_score` | 직전 프레임 대비 커서 이동량 |
| `text_change_score` | OCR 텍스트 변화량 |
| `focus_change_score` | input focus, highlight, selection 변화량 |
| `dialog_jump_score` | dialog open/close 같은 구조 변화량 |

#### 2.2 후보 burst 찾기

아래 패턴을 수동 조작 후보로 본다.

- 장시간 안정 상태 후 커서가 갑자기 이동
- 작은 영역에만 집중 변화 발생
- focus/caret 변화 후 텍스트 diff 발생
- 목록/테이블/패널이 부분 스크롤
- dialog / dropdown / tab 전환

#### 2.3 window 확장

후보 burst 를 찾으면 `-10초` 에서 `+20초` 정도로 확장한다.
이 확장 구간이 episode 후보가 된다.

#### 2.4 episode 레벨 분류

episode 후보를 아래 3개로 분류한다.

- `manual_control`
- `auto_running`
- `uncertain`

`uncertain` 은 나중에 human review 또는 teacher model 로 재분류한다.

### 구현시 주의점

- 전체 화면 변화가 크다고 수동 조작인 것은 아니다.
- 반대로 수동 조작은 종종 아주 작은 영역 변화만 만든다.
- cursor 가 계속 보이는지와 cursor 가 실제로 이동하는지는 구분해야 한다.

### 저장 스키마 예시

```json
{
  "episode_id": "ep_0001",
  "video_id": "vid_001",
  "start_time": 1052.0,
  "end_time": 1074.5,
  "label": "manual_control",
  "confidence": 0.88,
  "signals": {
    "cursor_motion_score": 0.91,
    "text_change_score": 0.63,
    "dialog_jump_score": 0.52
  }
}
```

## 3. White Cursor Detection and Tracking

이 단계는 전체 파이프라인의 핵심이다.

### 목표

- 각 프레임에서 커서 위치를 추정한다.
- 프레임 간 연속성을 유지한다.
- 커서 모양이 바뀌어도 최대한 안정적으로 추적한다.

### 권장 탐지 계층

#### 3.1 template bank

아래 형태를 별도 템플릿 세트로 둔다.

- white arrow
- white hand
- I-beam
- crosshair
- wait cursor 계열이 있으면 추가

템플릿은 여러 크기와 약간의 회전을 포함하는 편이 좋다.

#### 3.2 후보 생성

각 프레임에서 아래 방식으로 커서 후보를 만든다.

- 밝기 threshold
- 가장자리/형태 특징
- template similarity
- 이전 프레임 위치 주변 search window

#### 3.3 temporal linking

후보가 여러 개면 아래 우선순위로 고른다.

1. 이전 프레임 cursor 위치와 가장 가까운 후보
2. template score 가 높은 후보
3. local background 대비 충분히 distinct 한 후보
4. 일정 시간 연속 추적이 되는 후보

#### 3.4 추적 실패 회복

추적이 끊기면 아래 규칙으로 복구한다.

- full-frame search fallback
- 최근 10프레임 중 가장 안정적이었던 위치 재시도
- cursor shape 를 바꿔 재탐색
- recover 못하면 `cursor_missing`

### 저장 스키마 예시

```json
{
  "frame_id": "vid_001_f00012345",
  "cursor": {
    "x": 812,
    "y": 436,
    "bbox": [804, 428, 820, 444],
    "cursor_type": "white_arrow",
    "score": 0.93,
    "tracking_state": "tracked"
  }
}
```

### 품질 체크 포인트

- cursor recall
- cursor position error
- track fragmentation count
- shape-switch recovery rate

## 4. Event Extraction

cursor trajectory 를 action event 로 바꾸는 단계다.

### 공통 원칙

event 는 “이 프레임에서 무엇을 했는가”보다 “어떤 변화가 일어났는가”를 같이 저장해야 한다.
즉 아래 3개를 항상 묶어야 한다.

- cursor behavior
- local UI change
- pre/post stable frame

### 4.1 click

click 후보는 아래 조합으로 찾는다.

- cursor 가 짧게 정지한다
- 정지 직후 target 근방의 local pixel 변화가 발생한다
- 버튼 highlight, focus, selection 같은 local response 가 생긴다

#### 저장 필드

- `action_type=click`
- `cursor_x`, `cursor_y`
- `target_window_bbox`
- `pre_frame_id`
- `post_frame_id`
- `event_confidence`

### 4.2 double click

double click 후보는 아래 패턴을 본다.

- 같은 위치 부근에서 짧은 간격의 click 2회
- 두 번째 click 뒤 open/expand/select-all 같은 더 큰 반응이 생김

### 4.3 drag

drag 후보는 아래 패턴을 본다.

- cursor 가 누른 상태처럼 연속 이동
- selection box, splitter, scroll thumb, slider 가 함께 이동
- 시작점과 종료점이 명확함

drag 는 아래 타입으로 더 세분화하는 것이 좋다.

- `drag_select`
- `drag_scrollbar`
- `drag_splitter`
- `drag_slider`
- `drag_unknown`

### 4.4 scroll

scroll 후보는 아래 패턴을 본다.

- cursor 위치는 거의 고정
- list/table/panel 내용만 수직 또는 수평 이동
- scroll bar thumb 변화가 동반되면 confidence 상승

### 4.5 type_candidate

video-only 환경에서는 `type` 이 아니라 `type_candidate` 로 저장하는 것이 더 안전하다.

#### 판단 기준

- input field 가 focus 상태가 됨
- caret 또는 selected text 상태가 보임
- 직후 OCR text diff 발생

#### 저장 방식

- 실제 key stroke 를 저장하지 않는다
- `before_text`, `after_text`, `inserted_text_candidate`, `confidence` 를 저장한다

### 4.6 hotkey_candidate

마우스 이동 없이 화면 전환이 일어났을 때 hotkey 를 의심할 수 있다.
하지만 menu action 과 구분이 어려우므로 보수적으로 저장한다.

권장 타입:

- `hotkey_candidate`
- `system_event_candidate`
- `hotkey_or_system_event`

### 4.7 wait

툴 자동 동작을 기다리는 구간은 action sequence 상에서 중요하다.

`wait` 는 아래 상황에 넣는다.

- 직전 action 후 결과가 나타날 때까지 시간이 필요함
- 사용자가 추가 입력 없이 화면 변화를 관찰함

이 이벤트는 later planning 에서 “즉시 추가 클릭하지 말라”는 신호가 된다.

## 5. Step Segmentation

event 를 나열하는 것만으로는 trajectory 가 되지 않는다.
원자적 step 으로 쪼개야 한다.

### step 정의

1개 step 은 아래를 포함한다.

- stable `pre_frame`
- action moment
- stable `post_frame`
- target summary
- effect summary

### step boundary 규칙

- 새로운 cursor motion burst 시작
- action 후 화면이 안정화되는 시점
- dialog open/close
- list scroll 종료
- input 편집 완료 후 caret 안정화

### 권장 구조

```json
{
  "step_id": "step_001",
  "episode_id": "ep_0001",
  "pre_frame_id": "f100",
  "post_frame_id": "f124",
  "action": {
    "type": "click",
    "cursor": [812, 436]
  },
  "effect": {
    "type": "dialog_open",
    "summary": "Recipe parameter dialog opened"
  }
}
```

### 구현시 주의점

- 하나의 user intention 이 여러 low-level event 로 나뉠 수 있다.
- 반대로 하나의 long drag 를 여러 step 으로 쪼개면 안 된다.
- step granularity 는 `planner` 가 그대로 참고할 수 있을 정도로 작아야 한다.

## 6. Target Grounding

action 을 재사용하려면 좌표만이 아니라 “무엇을 눌렀는가”가 필요하다.

### 역할 분담

- `UI-Venus` 또는 `UI-TARS`
  - 현재/과거 screenshot 에서 candidate UI element grounding
- `MAI-UI`
  - 작은 crop 재탐색
- `PaddleOCR-VL`
  - dense text / list / table / panel 텍스트
- `GOT-OCR`
  - 작은 숫자/코드 fallback

### target grounding 절차

1. `pre_frame` 에서 primary VLM 으로 UI candidate 목록 추출
2. cursor 종점과의 거리 기준으로 target 후보 정렬
3. 후보가 작거나 crowded 하면 crop 생성
4. crop 을 `MAI-UI`로 다시 확인
5. label/number 가 중요하면 OCR sidecar 호출
6. 최종 target 을 `bbox + label + element_type + confidence` 형태로 저장

### 저장 스키마 예시

```json
{
  "target": {
    "element_type": "button",
    "label": "Apply",
    "bbox": [780, 420, 860, 452],
    "grounded_by": "ui-venus+mai-ui",
    "confidence": 0.87
  }
}
```

### 구현시 주의점

- cursor 위치만으로 target 을 확정하지 않는다.
- OCR 텍스트만으로도 target 을 확정하지 않는다.
- `cursor proximity`, `GUI grounding`, `OCR evidence` 를 합쳐야 한다.

## 7. Parameter Value Reconstruction

recipe automation 에서 가장 중요한 것은 최종 parameter value 다.

### 목표

- 엔지니어가 어떤 값을 입력/수정했는지 추정한다.

### 절차

1. `type_candidate` step 에서 focused input bbox 를 찾는다.
2. `pre_frame` / `post_frame` 의 해당 영역을 crop 한다.
3. OCR로 전후 문자열을 읽는다.
4. diff 를 계산해 삽입/삭제 후보를 만든다.
5. 숫자, 단위, code, recipe name 은 별도 태그를 붙인다.

### 저장 스키마 예시

```json
{
  "value_change": {
    "field_label": "Dose",
    "before_text": "12.0",
    "after_text": "12.5",
    "delta_type": "replace",
    "inserted_text_candidate": "12.5",
    "confidence": 0.81
  }
}
```

### 실패 케이스

- caret 만 보이고 값 변화가 없을 수 있음
- 숫자 `1`, `7`, `.` 같은 문자는 OCR 오류가 잦음
- selection overwrite 와 append 를 구분하기 어려울 수 있음

이 경우 low confidence 로 남기고 later human review 대상에 넣는다.

## 8. Local Objective Generation

trajectory 재사용성을 높이려면 좌표 나열이 아니라 “무슨 작업을 위한 step 인가”를 붙여야 한다.

### 목표 필드

- `global_task`
- `local_objective`
- `subgoal_completion_signal`

### 생성 절차

1. episode 내 step 전후 화면, target, OCR text 를 모은다.
2. step cluster 를 의미 단위로 묶는다.
3. teacher model 또는 템플릿 규칙으로 목적을 생성한다.

예시:

- global task: `Create new recipe`
- local objective: `Open focus parameter panel`
- subgoal completion signal: `Parameter panel title visible`

### 구현시 주의점

- objective 는 너무 길면 retrieval 품질이 나빠진다.
- objective 는 screen-specific 한 문장보다 task-specific 한 문장이 좋다.

## 9. Trajectory Storage Design

trajectory 는 단순 JSON dump 가 아니라 retrieval 대상 데이터여야 한다.

### 저장 단위

- `video`
- `episode`
- `step`
- `target`
- `value_change`
- `verification`

### 필수 인덱스

- `video_id`
- `episode_id`
- `step_id`
- `tool_id`
- `task_type`
- `local_objective`
- `target_label`
- `action_type`
- `success`

### 검색용 표현

각 step 또는 episode 에 대해 아래 표현을 별도로 저장하는 것이 좋다.

- visual embedding
- OCR text summary
- local objective text
- action summary text

즉 retrieval 은 한 종류가 아니라 아래 조합이 된다.

- visual similarity
- OCR text similarity
- objective similarity
- target label similarity

## 10. Retrieval Layer

실행 시점에서 현재 화면에 맞는 과거 예시를 찾아야 한다.

### 검색 목표

- 현재 화면과 유사한 step/episode
- 현재 subgoal 과 유사한 trajectory
- 비슷한 target label/element 를 다룬 사례

### 검색 전략

1. visual top-k 검색
2. OCR text top-k 검색
3. objective text top-k 검색
4. action/target metadata re-rank

### retrieval 결과 예시

```json
{
  "query": {
    "tool_id": "sem_tool_a",
    "current_goal": "Change focus parameter"
  },
  "results": [
    {
      "step_id": "step_1842",
      "score": 0.89,
      "local_objective": "Open focus parameter panel",
      "action_type": "click",
      "target_label": "Focus"
    }
  ]
}
```

### 구현시 주의점

- visual similarity 만 믿으면 다른 tool 의 비슷한 패널이 섞일 수 있다.
- OCR text 만 믿으면 레이아웃이 다른 화면이 섞일 수 있다.
- hybrid retrieval 이 기본이어야 한다.

## 11. Next-Action Planning

retrieval 결과를 바탕으로 VLM에게 다음 step 만 결정하게 하는 계층이다.

### planner 입력

- 현재 screenshot
- 현재 task goal
- 최근 1~3 step history
- OCR summary
- retrieved trajectory examples
- 허용 action schema

### planner 출력

반드시 단일 step JSON 만 내게 한다.

```json
{
  "next_action": {
    "type": "click",
    "target": {
      "description": "Focus parameter tab",
      "bbox": [720, 155, 820, 186],
      "confidence": 0.84
    },
    "args": {},
    "expected_effect": "Focus parameter panel becomes visible",
    "reason": "Current panel matches prior recipe-edit trajectory"
  }
}
```

### planner 규칙

- 한번에 한 step 만 내게 한다.
- 불확실하면 `ask_human` 을 내게 한다.
- `expected_effect` 를 반드시 쓰게 한다.
- target bbox 없이 click 을 내지 못하게 한다.

### 역할 분담

- `UI-Venus` / `UI-TARS`
  - planner + grounding
- `MAI-UI`
  - target이 작거나 crowded 할 때만 escalation
- `Kimi-K2.5`
  - ambiguous case 또는 teacher planning 비교용

## 12. Action Execution Layer

planner 결과를 실제 RCS mouse/keyboard action 으로 변환하는 계층이다.

### 입력

- `next_action` JSON

### 출력

- 실제 mouse/keyboard operation
- `pre_action_screenshot`
- `post_action_screenshot`
- execution log

### 실행 규칙

- 좌표는 현재 screenshot 기준이어야 한다.
- scaling / monitor offset / RCS viewport offset 을 별도 보정한다.
- drag 는 시작점과 끝점을 둘 다 검증한다.
- type action 은 실제 실행 전 target input 을 다시 한 번 확인한다.

### safety 규칙

- low confidence action 은 실행하지 않는다.
- destructive action 은 `ask_human` 으로 전환한다.
- 연속 실패 시 중단한다.

## 13. Step Verification

action 후 실제로 의도한 효과가 발생했는지 확인한다.

### verification 입력

- pre-action screenshot
- post-action screenshot
- action JSON
- expected_effect

### verification 방법

#### 13.1 rule-based

- dialog title visible
- tab selected state change
- input focus acquired
- value changed
- row selected

#### 13.2 OCR-based

- 기대한 문자열이 생겼는가
- 이전 값이 새 값으로 바뀌었는가

#### 13.3 VLM-based

- post-frame 이 expected_effect 와 일치하는가
- target element 주변 상태가 맞는가

### verification 출력

```json
{
  "verification": {
    "success": true,
    "confidence": 0.9,
    "method": "rule+ocr",
    "evidence": [
      "Focus tab highlighted",
      "Panel title changed to Focus"
    ]
  }
}
```

### 구현시 주의점

- verification 은 planner 와 분리하는 편이 좋다.
- action success 여부를 planner 가 자기 자신에게 판정하게 하면 오류가 누적된다.

## 14. Human-in-the-Loop

초기에는 semi-automatic 모드가 필요하다.

### human approval 이 필요한 상황

- target confidence 가 낮음
- OCR 값이 애매함
- destructive action 가능성 있음
- recovery path 가 여러 개임
- 반복 실패가 발생함

### human correction 을 저장해야 하는 이유

human correction 은 가장 가치 있는 학습 데이터다.
특히 아래 필드를 남기는 것이 중요하다.

- planner 가 제안한 action
- 실제 human 이 고친 action
- correction 이유
- correction 후 결과

## 15. 평가 체계

구현 후 반드시 수치화해야 할 항목은 아래다.

### episode mining

- precision
- recall
- uncertain rate

### cursor tracking

- cursor recall
- mean position error
- track fragmentation

### event extraction

- click precision / recall
- scroll precision / recall
- drag precision / recall
- type_candidate accuracy

### target grounding

- target hit rate
- bbox IoU
- label match accuracy

### planning

- step success rate
- retry count
- ask_human rate
- average steps to completion

### end-to-end

- task completion rate
- average completion time
- human takeover rate

## 16. 권장 구현 순서

아래 순서가 가장 안전하다.

1. representative AVI set 정리
2. manual-control episode mining
3. white cursor tracking
4. click / scroll / drag extraction
5. type_candidate + OCR diff
6. target grounding
7. step verification
8. retrieval memory
9. next-action planning
10. semi-automatic deployment

## 17. 팀 합의가 필요한 의사결정

구현 전에 아래 항목은 팀에서 합의해야 한다.

1. 어떤 tool UI를 1차 대상에 넣을 것인가
2. 어느 수준의 오탐/누락을 허용할 것인가
3. `type_candidate` low confidence 를 자동 실행에서 허용할 것인가
4. human approval threshold 를 어떻게 둘 것인가
5. trajectory 저장소를 어디에 둘 것인가
6. evaluation gold set 을 누가 만들 것인가

## 18. 문서 사용 방법

이 문서는 아래 순서로 쓰는 것이 좋다.

1. 본 문서로 구현 범위를 합의한다.
2. `task breakdown` 문서로 실제 작업을 나눈다.
3. 이후 각 단계별 상세 설계 문서를 별도 추가한다.

## 관련 문서

- [RCS 비전-투-액션 연구 메모](/Users/daeyoung/Codes/auto_recipe_creator/docs/research/rcs_video_to_action_research.md)
