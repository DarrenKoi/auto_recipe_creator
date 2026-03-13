# RCS 비전-투-액션 연구 메모 (2026-03-13)

## 목적

이 문서는 아래 질문에 답하기 위한 운영 연구 메모다.

1. `RCS(Remote Control System)` 화면만 보고 VLM이 다음 마우스/키보드 액션을 결정하게 하려면 어떤 구조가 맞는가?
2. 2주 분량의 `AVI` 녹화 영상에서 엔지니어의 수동 조작 구간(전체의 약 10%)을 어떻게 추출해 학습/가이드 데이터로 바꿀 것인가?
3. 현재 저장소에서 이미 쓰고 있는 `UI-Venus`, `UI-TARS`, `MAI-UI`, `PaddleOCR-VL`, `GOT-OCR`, `Kimi-K2.5`를 각각 어떤 역할로 쓰는 것이 맞는가?

핵심 결론부터 말하면, **처음부터 “비디오로 모델을 바로 재학습”하는 접근보다, 영상에서 `상태 -> 액션 -> 결과` 형태의 전문가 궤적을 먼저 추출하고, 이를 RAG/예시 기반으로 다음 액션 결정에 주입하는 방식이 현실적**이다.

특히 현재 조건에서는 다음 3단 구조가 맞다.

1. `AVI -> manual-control episode -> step trajectory` 로 오프라인 지식화
2. `현재 화면 + 목표 + 과거 전문가 step 예시` 로 다음 액션 결정
3. 액션 실행 후 `pre/post screenshot` 기반 step verification 으로 성공/실패 판정

## 왜 이 문제가 어려운가

RCS 기반 원격 화면은 일반 Windows UI 자동화와 다르다.

- 실제 툴 UI가 RCS 스트림 안에 렌더링되므로 `pywinauto` 같은 표준 접근성 API가 잘 안 먹을 가능성이 크다.
- 즉, 이 프로젝트의 핵심은 selector 기반 자동화가 아니라 **screenshot-first GUI grounding** 이다.
- 또 전체 가동 시간의 90%는 툴이 자동 운전이므로, 녹화 영상 대부분은 “아무 것도 가르쳐주지 않는 구간”일 가능성이 높다.

따라서 전체 비디오를 그대로 모델에 넣는 것은 비효율적이다. 먼저 **수동 조작이 실제로 일어난 10%만 찾아내고**, 그 안에서도 **행동 단위(step)** 로 잘게 쪼개야 한다.

## 외부 연구 기준에서 본 시사점

최근 GUI agent 연구는 지금 상황과 매우 비슷한 결론을 보여준다.

- `ScreenSpot-Pro`는 전문 소프트웨어 고해상도 화면에서 GUI grounding 이 여전히 어렵다는 점을 보여준다. 반도체 장비 UI도 이 범주에 가깝다.
- `GUI Narrator / Act2Cap`은 GUI 영상에서 primitive action 을 이해하려면 **cursor, keyframe, temporal boundary** 가 중요하다고 본다.
- `VideoAgentTrek`은 **라벨 없는 screen recording** 에서도 `Video2Action` 형태의 inverse dynamics pipeline 으로 step 데이터를 자동 채굴할 수 있음을 보였다.
- `Learning from Online Videos at Inference Time for Computer-Use Agents`는 비디오 전체를 통째로 넣지 않고, **짧은 trajectory 로 구조화한 뒤 현재 subgoal 에 맞는 것만 동적으로 골라 넣는 방식**이 효과적이라고 본다.
- `STEVE`는 행동 전/후 화면을 비교해 각 step 의 정오를 라벨링하는 것이 agent 학습과 품질 관리에 매우 중요하다고 본다.
- `PC-Agent`는 raw interaction log 를 “cognitive trajectory” 로 바꾸는 후처리 계층이 중요하다고 본다.

이 흐름을 RCS에 그대로 적용하면,

- **비디오를 바로 teacher 로 쓰지 말고**
- **비디오에서 step trajectory 를 뽑아낸 다음**
- **실행 시점에는 필요한 trajectory 만 골라 현재 화면에 붙여주는 구조**

가 가장 현실적이다.

## 현재 모델 스택에서의 역할 분담

현재 저장소 기준 권장 역할은 아래와 같다.

| 역할 | 모델 | 권장 용도 |
|------|------|-----------|
| Primary GUI grounding | `UI-Venus-1.5-8B` 또는 `UI-TARS-1.5-7B` | 현재 화면에서 클릭/입력 대상 후보 찾기, step-level action planning |
| Zoom-in re-grounding | `MAI-UI-8B` | 작은 버튼, crowded toolbar, 애매한 crop 재탐색 |
| Dense text reading | `PaddleOCR-VL-1.5` | parameter panel, list/table, 작은 텍스트 판독 |
| Hard OCR fallback | `GOT-OCR-2.0-hf` | format 민감 숫자, 코드, 작은 crop fallback |
| Teacher / verifier / hard reasoning | `Kimi-K2.5` | trajectory 설명 생성, ambiguous step 판정, pseudo-label 보강 |

현재 단계에서 `MAI-UI`를 primary executor 로 쓰기보다 **sidecar** 로 두는 것이 맞다. 저장소에서도 이미 `UI-Venus + PaddleOCR-VL` 구성이 기본이며, `MAI-UI`는 조건부 확대 판독 역할로 두는 편이 자연스럽다.

## “모델에게 다음에 무엇을 할지” 어떻게 알려줄 것인가

가장 중요한 점은 **모델에게 자유 서술형으로 “다음 뭐 할까?”를 묻지 않는 것**이다.

다음 액션 결정을 위해 모델 입력은 아래 6가지로 구조화하는 편이 좋다.

1. 현재 screenshot
2. 현재 목표(task goal)
3. 직전 1~3 step action history
4. OCR sidecar 가 읽은 핵심 텍스트
5. 검색된 전문가 trajectory 예시 1~3개
6. 허용 action schema

권장 action schema 예시는 아래와 같다.

```json
{
  "next_action": {
    "type": "click | double_click | right_click | drag | scroll | hotkey | type | wait | verify | ask_human | stop",
    "target": {
      "description": "예: Recipe Name input",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0
    },
    "args": {
      "text": "",
      "keys": [],
      "scroll_delta": 0,
      "drag_to": null
    },
    "expected_effect": "예: 입력창에 커서가 들어가고 기존 값이 선택됨",
    "reason": "한 줄 설명"
  }
}
```

즉, 모델은 “정답 문장”이 아니라 **실행 가능한 step JSON** 을 내야 한다.

또 `expected_effect` 필드를 강제하는 것이 중요하다. 이 값이 있어야 액션 후 검증을 할 수 있다.

## RCS용 전체 아키텍처 권장안

### 1. 오프라인: AVI에서 전문가 궤적 추출

```text
AVI -> 프레임 추출 -> manual-control episode mining -> step segmentation
    -> action parameterization -> target grounding -> effect labeling
    -> trajectory store (검색 가능)
```

### 2. 온라인: 현재 화면에서 다음 step 결정

```text
현재 screenshot -> state summary + OCR hint
    -> similar trajectory retrieval
    -> primary VLM action planning
    -> conditional MAI-UI / OCR escalation
    -> mouse/keyboard 실행
    -> step verification
```

### 3. 안전 제어

- `SAFE_MODE=true` 기본 유지
- 위험 액션은 승인 필요
- 불확실성 높으면 `ask_human`
- 실패 2회 이상 시 자동 중단

## AVI 처리 파이프라인: 무엇을 추출해야 하는가

영상에서 필요한 것은 “비디오” 자체가 아니라 아래 구조다.

```text
observation_t
action_t
observation_t+1
effect_t
```

좀 더 구체적으로는 아래 필드가 필요하다.

| 필드 | 설명 |
|------|------|
| `pre_frame` | 액션 직전 화면 |
| `post_frame` | 액션 직후 화면 |
| `action_type` | click, drag, type, scroll, hotkey 등 |
| `action_args` | 좌표, 텍스트, 키 조합, scroll delta |
| `target_bbox` | 클릭/입력 대상 위치 |
| `target_text` | OCR 또는 VLM이 읽은 타깃 라벨 |
| `effect_summary` | 액션 후 화면 변화 요약 |
| `task_objective` | 이 step 이 속한 로컬 목표 |
| `success` | 성공 여부 |
| `confidence` | 자동 추출 신뢰도 |

## 현재 확보 조건이 주는 의미

현재 추가로 확인된 조건은 아래 2가지다.

- `AVI` 안에 **흰색 마우스 커서가 보인다**
- **입력 로그 없이 비디오만 있다**

이 조건은 extraction 전략에 직접 영향을 준다.

### 1. 흰색 커서가 보인다는 점은 큰 장점이다

이 경우 아래 조합이 현실적으로 가능하다.

1. white cursor template matching
2. 밝기/색상 threshold 기반 후보 검출
3. optical flow 기반 이동 연속성 추적
4. 화면 모서리/고정 UI와 구분하는 negative mask

즉 click/drag/scroll 복원의 출발점을 “화면 전체 변화량”이 아니라 **cursor trajectory** 에 둘 수 있다.

### 2. 비디오만 있다는 점은 keyboard 복원 한계를 만든다

반대로 아래 항목은 video-only 조건에서 정확 복원이 어렵다.

- 실제 key sequence
- modifier 조합(`Ctrl`, `Alt`, `Shift`)의 정확한 시점
- backspace/delete/edit 과정
- 단축키와 menu click 의 완전 구분

따라서 `type`/`hotkey` 는 deterministic truth 로 저장하지 말고 아래처럼 저장하는 편이 맞다.

```json
{
  "type": "type",
  "text_candidate": "recipe_01",
  "recovery_method": "ocr_diff",
  "confidence": 0.71
}
```

즉 video-only 조건에서는 **click / drag / scroll 은 비교적 강하게**, **type / hotkey 는 후보 + confidence 로 약하게** 다루는 것이 맞다.

## 1단계: 90% 자동 운전 구간 제거

지금 가장 큰 효율 포인트는 **manual-control 구간만 남기는 것**이다.

권장 순서는 아래와 같다.

1. `1~2 fps` 저속 샘플링으로 전체 영상을 훑는다.
2. 아래 신호를 사용해 수동 조작 가능성이 높은 구간을 찾는다.

- 커서가 장시간 정지 상태에서 갑자기 이동함
- 클릭 직후 특정 영역의 픽셀 변화가 발생함
- 텍스트 입력 직후 focused region 의 OCR 결과가 바뀜
- 팝업/드롭다운/탭 전환 같은 UI state jump 가 발생함
- 화면 변화량은 작지만 특정 region 만 자주 변함

3. 이런 burst 를 중심으로 `±10~20초` window 를 잡아 episode 후보로 만든다.
4. 후보 episode 를 `manual_control`, `auto_running`, `uncertain` 으로 1차 분류한다.

여기서 중요한 점은 **프레임 변화량이 큰 구간만 고르면 안 된다**는 것이다. 수동 조작은 오히려 “작은 영역의 의미 있는 변화”일 수 있다.

## 2단계: episode 내부를 step으로 쪼개기

한 episode 안에서 다시 atomic step 경계를 잡아야 한다.

권장 step boundary 신호:

- 커서 정지 -> 이동 시작
- 클릭 시점 추정
- 입력 focus 획득
- 드롭다운/다이얼로그 open/close
- 스크롤 후 UI 목록 정지
- 액션 후 0.3~1.0초 내 stable frame 도달

실무적으로는 아래처럼 잡는 편이 좋다.

1. `cursor movement start`
2. `action moment`
3. `screen settles`

즉 step 은 대략 아래처럼 저장한다.

```text
pre-frame (stable)
-> action frame(s)
-> post-frame (stable)
```

## 3단계: action type 추론

비디오만 있을 때 action type 은 아래 방식으로 추론한다.

### click / double click / right click

- 커서가 특정 위치에 정지
- 직후 작은 local visual change 발생
- 짧은 시간 내 같은 위치 반복이면 double click 후보
- context menu 출현 시 right click 후보
- 흰색 커서 template 의 peak 안정도가 높으면 click 후보 신뢰도를 올린다

### drag

- 커서가 버튼 press 상태처럼 보이는 동안 연속 이동
- 객체/selection box/slider/thumb 이 같이 이동

### scroll

- 커서는 거의 고정인데 list/table/panel 내용이 수직 이동
- 흰색 커서 위치는 유지되지만 panel content 만 moving pattern 을 보이면 scroll 후보로 본다

### type

- focus ring / caret / selected input field 감지
- 직후 OCR text diff 로 삽입 문자열 추정
- video-only 환경에서는 실제 key stroke 가 아니라 text delta 를 action 결과로 저장한다

### hotkey

- 마우스 이동 없이 dialog open/close, tab change, selection action 발생
- 단축키 후보는 후처리에서 유추
- confidence 가 낮으면 `hotkey_or_system_event` 로 보수적으로 저장한다

비디오만으로 모든 키 입력을 100% 복원하기는 어렵다. 따라서 `type` 과 `hotkey` 는 **후보 문자열/후보 키 조합 + confidence** 로 저장하는 편이 안전하다.

## 4단계: target grounding

행동 종류만 알면 부족하다. **어디를 눌렀는지**를 알아야 한다.

여기서 현재 모델 스택을 다음 순서로 쓰는 것이 좋다.

1. `UI-Venus` 또는 `UI-TARS`로 pre-frame 에서 후보 UI element grounding
2. 커서 종점과 가장 가까운 interactable element 후보 추출
3. 작은 요소면 해당 crop 을 `MAI-UI`로 재해석
4. 텍스트가 중요하면 같은 crop 에 `PaddleOCR-VL`
5. 작은 숫자/코드류는 `GOT-OCR` fallback

즉 영상에서 추정한 커서 위치를 **VLM grounding 의 약한 supervision** 으로 쓰는 구조다.

이 방식은 `GUI Narrator`가 cursor 를 중요한 visual prompt 로 다루는 방향과도 맞다.

## 5단계: typed text / parameter 값 복원

레시피 생성에서 중요한 것은 클릭보다도 실제 parameter 값이다.

권장 방식:

1. type action 전후 `input bbox` 를 잘라낸다.
2. `PaddleOCR-VL`로 pre/post 텍스트를 읽는다.
3. diff 를 계산해 삽입/삭제 문자열을 추정한다.
4. confidence 가 낮으면 `GOT-OCR` fallback
5. 값이 숫자/코드/recipe parameter 면 별도 high-priority tag 부여

이렇게 해야 “엔지니어가 어떤 값을 넣었는가”가 남는다. 단순 click trajectory 만 저장하면 recipe automation 에 필요한 핵심 정보가 빠진다.

## 6단계: 로컬 목표(task objective) 생성

step 을 모아도 그냥 좌표 나열이면 나중에 재사용하기 어렵다.

각 step 묶음마다 아래 둘이 필요하다.

1. `global task`: 예: "새 recipe 생성"
2. `local objective`: 예: "Exposure 조건 입력창에 기준 값 입력"

이 objective 는 아래 조합으로 생성하는 것이 좋다.

- step 전/후 screenshot
- action list
- OCR text
- 필요 시 `Kimi-K2.5` teacher summarization

이렇게 생성된 objective 가 있어야 나중에 retrieval 할 때 “현재 subgoal 과 비슷한 예시”를 찾을 수 있다.

## 7단계: step verification

다음 액션을 잘 내는 것만큼 중요한 것이 **잘못 눌렀을 때 알아차리는 것**이다.

권장 검증 구조:

1. action 전 screenshot 저장
2. action 실행
3. action 후 screenshot 저장
4. 아래 3종 검증 수행

- rule-based verification
  - 팝업 열림/닫힘
  - input 값 변화
  - target 영역 highlight/focus 여부
- OCR verification
  - 기대 텍스트가 나타났는가
- VLM verification
  - `expected_effect` 와 post-frame 이 일치하는가

`STEVE` 식 사고를 적용하면, step 의 성공/실패 binary label 을 축적할 수 있고 나중에 SFT/DPO/KTO 계열 데이터로도 쓸 수 있다.

## 영상으로 “가르치는” 가장 현실적인 방법

### 1단계: 즉시 가능한 방법

**fine-tuning 없이 RAG memory 로 쓰기**

- 전문가 영상에서 추출한 trajectory 를 저장
- 현재 화면이 들어오면 유사한 trajectory 검색
- 그 예시를 붙여 primary VLM 에게 다음 액션만 결정하게 함

이 방식의 장점:

- 라벨 품질이 완벽하지 않아도 쓸 수 있다
- 현재 저장소의 `RAGContextManager`, `RAGPromptBuilder` 방향과 맞다
- 모델 재학습 없이 바로 실험 가능하다

### 2단계: 데이터가 쌓인 뒤

**step-level supervised fine-tuning**

권장 시작 기준:

- 검증된 step 수가 최소 `5k~20k`
- 실패/복구 케이스 포함
- click 뿐 아니라 `type`, `scroll`, `drag`, `verify`, `ask_human` 이 모두 포함

이때도 처음부터 foundation model 전체를 크게 바꾸기보다,

- action formatting 안정화
- domain vocabulary 적응
- local workflow pattern 학습

용도의 경량 tuning 이 맞다.

### 3단계: 그 다음

**step verification 기반 preference / reward 학습**

- 좋은 step / 나쁜 step binary label
- recovery trajectory
- ambiguous case 에서의 human correction

이 데이터가 쌓이면 policy 품질이 많이 올라간다.

## 저장소에 바로 연결하는 방법

현재 저장소에는 이미 필요한 뼈대가 일부 있다.

- [`/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py)
  - 자동 action 추출은 아직 기본 stub 수준
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py)
  - `ActionSequence`, `RAGContext` 데이터 모델 존재
- [`/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_context_manager.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_context_manager.py)
  - 현재 화면과 과거 프레임/작업 시퀀스를 연결하는 RAG 방향 존재
- [`/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py)
  - 현재 screenshot 기반 분석 루프 존재

즉, 새 시스템을 처음부터 만들 필요는 없고 아래 순서가 맞다.

1. `test/video_frame_parser/` 에 episode mining + cursor/event extraction 추가
2. `ActionSequence.actions` 포맷을 step JSON schema 로 정규화
3. trajectory/objective 저장
4. `test/vlm_input_control/` 또는 `poc/work2/` 에 retrieval-augmented next-action planner 추가
5. `work2` executor 에 step verification 추가

## 가장 먼저 구현할 최소 기능(MVP)

MVP는 아래 정도면 충분하다.

1. `AVI`에서 `manual-control episode` 후보 추출
2. 각 episode 에서 `white cursor tracking` 기반 `click / scroll / drag` 를 우선 복원
3. `type` 은 `OCR diff` 기반 후보 텍스트만 우선 복원
3. 각 step 에 대해
   - pre-frame
   - post-frame
   - action type
   - cursor location
   - target bbox
   - OCR diff
   - local objective
   저장
4. keyboard-like event 는 `candidate + confidence` 로 저장
5. 현재 화면과 유사한 과거 step 을 검색해 prompt 에 붙이기
6. model 이 JSON으로 `next_action` 을 내고, 실행 후 verify 하기

이 정도면 “녹화 영상으로부터 전문가 가이드를 끌어와 다음 행동을 결정하는 구조”를 이미 검증할 수 있다.

## 추천 프롬프트 구조

실행 프롬프트는 아래처럼 짧고 제한적으로 가져가는 편이 좋다.

```text
당신은 RCS를 통해 장비 UI를 조작하는 GUI agent다.

목표:
- {task_goal}

현재 화면 관찰:
- OCR 핵심 텍스트: {ocr_summary}
- 최근 실행 step: {recent_steps}

참고 전문가 예시:
- 예시 1: {trajectory_1}
- 예시 2: {trajectory_2}

규칙:
- 한 번에 한 step만 제안한다.
- 허용 action 은 click, double_click, drag, scroll, type, hotkey, wait, verify, ask_human, stop 뿐이다.
- 좌표는 반드시 현재 screenshot 기준이다.
- target 이 애매하면 ask_human 을 반환한다.
- expected_effect 를 반드시 적는다.

JSON으로만 답하라.
```

이 구조면 모델이 과도한 자유도를 갖지 않고, retrieval 예시도 바로 활용할 수 있다.

## 중요한 운영 원칙

### 1. “전체 영상”이 아니라 “짧은 검증된 step”이 자산이다

비디오를 많이 모으는 것보다, **검증된 step trajectory** 를 모으는 것이 훨씬 중요하다.

### 2. RCS 특성상 selector보다 screenshot memory가 더 중요하다

원격 툴 화면은 OS accessibility 가 약하므로, 이 프로젝트의 핵심 자산은 `DOM` 이 아니라 `screenshot + action + effect` 다.

### 3. OCR은 보조이고 grounding은 GUI 모델이 담당한다

텍스트 판독은 OCR이 잘하지만, “무엇을 눌러야 하는가”는 GUI grounding 모델이 더 중요하다.

### 4. action decision 과 action verification 을 분리해야 한다

결정 모델과 검증 모델을 분리해야 실수에서 회복할 수 있다.

### 5. 가능하면 앞으로는 입력 로그도 같이 수집하는 편이 훨씬 좋다

현재는 `AVI`만 있다고 가정했지만, 앞으로는 가능하면 RCS client 측에서 아래를 같이 기록하는 것이 좋다.

- mouse move/down/up
- keyboard keydown/keyup
- active window title
- timestamp

비디오만으로 복원하는 것보다 정확도와 개발 속도가 훨씬 좋아진다.

## 최종 권고

- 지금 단계에서 가장 현실적인 방향은 **비디오 기반 trajectory mining + retrieval-augmented action planning** 이다.
- 처음부터 VLM을 재학습하려 하지 말고, **AVI에서 manual-control step을 구조화해 전문가 메모리로 만드는 것**이 먼저다.
- 실행 시점에는 `UI-Venus` 또는 `UI-TARS`를 primary planner/grounder 로 쓰고, `MAI-UI`는 zoom-in sidecar, `PaddleOCR-VL/GOT-OCR`는 text sidecar, `Kimi-K2.5`는 teacher/verifier 로 두는 구성이 가장 실용적이다.
- 영상 처리 파이프라인의 본질은 `AVI -> episode -> step -> target -> effect -> objective` 변환이다.
- 이 구조가 잡히면, 그 다음부터는 semi-automatic 모드에서 human correction 을 계속 수집해 policy를 더 강하게 만들 수 있다.

## 다음 구현 우선순위

1. `manual-control episode mining`
2. `cursor / click / type / scroll` step extraction
3. `target grounding + OCR diff`
4. `trajectory retrieval`
5. `next-action JSON planner`
6. `step verification`

## 참고 자료

- UI-Venus Technical Report / GitHub: <https://github.com/inclusionAI/UI-Venus>
- UI-Venus paper: <https://arxiv.org/abs/2508.10833>
- UI-TARS / GitHub: <https://github.com/bytedance/UI-TARS>
- UI-TARS paper: <https://arxiv.org/abs/2501.12326>
- UI-TARS Desktop: <https://github.com/bytedance/UI-TARS-desktop>
- MAI-UI project: <https://tongyi-mai.github.io/MAI-UI/>
- MAI-UI paper: <https://arxiv.org/abs/2512.22047>
- PaddleOCR / PaddleOCR-VL: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR-VL paper: <https://arxiv.org/abs/2510.14528>
- GOT-OCR2.0 / GitHub: <https://github.com/Ucas-HaoranWei/GOT-OCR2.0>
- GOT-OCR2.0 paper: <https://arxiv.org/abs/2409.01704>
- Kimi K2.5 tech blog: <https://www.kimi.com/blog/kimi-k2-5.html>
- Kimi K2.5 paper: <https://arxiv.org/abs/2602.02276>
- ScreenSpot-Pro benchmark: <https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding>
- GUI Narrator / Act2Cap: <https://showlab.github.io/GUI-Narrator/>
- GUI Narrator paper: <https://arxiv.org/abs/2406.13719>
- VideoAgentTrek paper: <https://arxiv.org/abs/2510.19488>
- Learning from Online Videos at Inference Time for Computer-Use Agents: <https://arxiv.org/abs/2511.04137>
- STEVE paper: <https://arxiv.org/abs/2503.12532>
- PC-Agent / GitHub: <https://github.com/GAIR-NLP/PC-Agent>
- PC-Agent paper: <https://arxiv.org/abs/2412.17589>
