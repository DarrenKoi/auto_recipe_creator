# Video Frame Trajectory Extraction

이 문서는 `poc/workflow_1/capture_window_frames_ch4.py`가 저장하는 DVR player 프레임 묶음을
레시피 자동화에 재사용 가능한 `trajectory memory`와 `workflow step`으로 바꾸는 방법을 정리한다.
목표는 "영상에서 뭔가 일어났다" 수준이 아니라, 온라인 GUI 제어가 다시 사용할 수 있는
구조화된 action / schema / verification / fallback 계약을 만드는 것이다.

중요한 기준은 다음과 같다.

- raw video를 곧바로 action policy의 입력으로 쓰지 않는다.
- 먼저 `frame -> state change -> action candidate -> reviewed workflow step`으로 압축한다.
- 온라인 실행에서는 과거 step을 그대로 재생하지 않고, 현재 화면을 다시 관찰하고 검증한다.
- `observe -> decide -> act -> verify` 루프를 깨지 않는 형태로 offline 자산을 만든다.

## 1. `capture_window_frames_ch4.py`가 이미 제공하는 것

[`poc/workflow_1/capture_window_frames_ch4.py`](../../poc/workflow_1/capture_window_frames_ch4.py)는
다음 산출물을 만든다.

- `frames/frame_XXXX_YYYYYYYYms.jpg`
- `summary.json`
- `timeline.txt`

이 스크립트의 장점:

- 실제 Windows DVR player 창만 잘라서 일정 간격으로 저장한다.
- 프레임 번호와 경과 시간이 파일명, `summary.json`, `timeline.txt`에 모두 남는다.
- `workflow_select_ch4_cctv.py`와 같은 동일한 player-window 탐색 로직을 재사용하므로
  "어떤 창을 기록했는가"가 명확하다.

이 스크립트의 한계:

- cursor 위치 정보가 없다.
- OCR 결과가 없다.
- before/after state 요약이 없다.
- 어떤 프레임이 manual action 구간이고 어떤 프레임이 단순 playback 인지 분리하지 않는다.
- offline extraction 결과가 온라인 workflow schema와 직접 연결되지 않는다.

즉 현재 캡처 스크립트는 "좋은 raw evidence producer"이고,
여기에 후속 extraction 계층을 붙여야 한다.

## 2. 목표 산출물

video frame extraction의 최종 산출물은 아래 4종으로 나누는 것이 가장 실용적이다.

### 2.1 Frame Observation

각 프레임의 관찰 결과를 저장한다.

용도:

- OCR 텍스트 인덱싱
- UI state 비교
- event 후보 탐지
- retrieval memory 검색

권장 파일:

- `frame_observations.jsonl`

권장 스키마:

```json
{
  "frame_id": "250416_101530_player_f000123",
  "video_id": "250416_101530_player",
  "frame_index": 123,
  "timestamp_sec": 12.3,
  "frame_path": "frames/frame_0123_00012300ms.jpg",
  "width": 1280,
  "height": 720,
  "change_score": 0.084,
  "is_stable": false,
  "cursor": {
    "visible": true,
    "x": 941,
    "y": 612,
    "confidence": 0.82,
    "method": "template+motion"
  },
  "ocr_lines": [
    {
      "text": "Image",
      "bbox": [801, 84, 864, 109],
      "confidence": 0.98
    }
  ],
  "ui_hints": [
    {
      "type": "tab",
      "label": "Image",
      "bbox": [790, 74, 875, 116],
      "confidence": 0.74,
      "source": "ocr+vlm"
    }
  ]
}
```

### 2.2 Event Candidate

프레임 간 변화에서 "action이 있었을 가능성"이 높은 지점을 저장한다.

용도:

- review 비용 절감
- keyframe 선택
- action inference 입력

권장 파일:

- `event_candidates.json`

권장 스키마:

```json
{
  "event_id": "ev_0008",
  "start_frame_index": 118,
  "peak_frame_index": 123,
  "end_frame_index": 129,
  "start_time_sec": 11.8,
  "end_time_sec": 12.9,
  "change_score_peak": 0.121,
  "cursor_stop_point": [941, 612],
  "candidate_action_types": ["double_click", "click"],
  "evidence_sources": ["frame_diff", "cursor_track", "ocr_diff"],
  "confidence": 0.76
}
```

### 2.3 Step Candidate

실제로 자동화 가능한 step 후보를 저장한다.
이 레벨부터는 `workflow_1` 또는 `work2`의 step schema와 맞닿아야 한다.

권장 파일:

- `step_candidates.json`

권장 스키마:

```json
{
  "step_id": "step_0004",
  "action_type": "double_click",
  "target_description": "DVR player bottom-right Channel 4 viewport",
  "target_key": "ch4_cctv",
  "timestamp_sec": 12.3,
  "before_frame_index": 118,
  "after_frame_index": 129,
  "coordinates": [941, 612],
  "target_bbox": [640, 360, 1279, 719],
  "input_text": null,
  "preconditions": ["player_window_visible"],
  "success_criteria": ["single_channel_view_appeared"],
  "verification_plan": {
    "primary": "vlm_before_after",
    "secondary": "window_layout_check"
  },
  "confidence": 0.88,
  "confidence_tier": "high",
  "review_status": "pending",
  "evidence": {
    "before_frame_path": "frames/frame_0118_00011800ms.jpg",
    "after_frame_path": "frames/frame_0129_00012900ms.jpg",
    "ocr_diff_texts": ["Image", "Function", "Queue"]
  }
}
```

### 2.4 Reviewed Workflow Annotation

자동 추출만으로는 recipe automation에 바로 쓰기 어렵다.
최종적으로는 사람이 검토하거나 규칙 기반 auto-accept를 거친
`workflow annotation`으로 승격해야 한다.

이 단계에서는 기존
[`test/workflow_extractor/models.py`](../../test/workflow_extractor/models.py)의
`WorkflowStep`, `WorkflowAnnotation`과 호환되도록 맞추는 것이 좋다.

## 3. Recipe Automation 관점의 Action Taxonomy

레시피 자동화에서 필요한 action 분류는 넓게 잡을 필요가 없다.
처음부터 아래처럼 GUI replay 친화적인 소수 타입으로 고정하는 편이 좋다.

권장 분류:

- `click`
- `double_click`
- `right_click`
- `drag`
- `scroll`
- `type`
- `hotkey`
- `select`
- `wait`
- `verify`

추가 메타데이터:

- `target_kind`: `button`, `tab`, `tree_item`, `grid_cell`, `dialog`, `input`, `canvas`
- `target_locator`: `text`, `bbox`, `ocr_anchor`, `visual_description`, `window_region`
- `state_effect`: `dialog_opened`, `dialog_closed`, `tab_changed`, `field_focused`, `value_changed`

이 분류가 중요한 이유:

- CCTV/화면 녹화에서 keyboard 이벤트를 완전히 복원하기 어렵다.
- 그래서 모든 action을 low-level event replay로 만들면 정확도가 떨어진다.
- 반대로 high-level GUI step으로 저장하면 온라인 시점에 다시 탐지하고 검증할 수 있다.

예를 들어 영상에서 `user id 입력`은 실제 keypress를 모두 복원하기보다,
다음 형태로 저장하는 것이 낫다.

```json
{
  "action_type": "type",
  "target_kind": "input",
  "target_locator": {
    "locator_type": "ocr_anchor",
    "label": "User Name"
  },
  "input_text": "<REDACTED_OR_TEMPLATE>",
  "state_effect": "value_changed"
}
```

## 4. 권장 Extraction Workflow

`capture_window_frames_ch4.py` 출력에서 step을 만들 때는 아래 순서를 권장한다.

### 4.1 Stage A: Capture Manifest 정규화

입력:

- `summary.json`
- `timeline.txt`
- `frames/*.jpg`

처리:

- `video_id` 생성
- 각 프레임에 `frame_id` 부여
- 상대 경로를 기준 경로와 함께 정규화
- 프레임 누락 여부 확인

출력:

- `capture_manifest.json`

이 단계는 꼭 필요하다.
현재 `summary.json`은 충분히 유용하지만, downstream 파이프라인이 요구하는
`video_id`, `frame_id`, 상대 경로 기준을 별도 manifest로 명시하는 편이 안전하다.

### 4.2 Stage B: Frame Observation 생성

각 프레임에 대해 다음을 계산한다.

- grayscale diff 기반 `change_score`
- `is_stable`
- OCR 결과
- cursor 후보
- coarse UI hint

구현 매핑:

- frame iteration / 메타데이터 틀: [`test/video_frame_parser/extractor.py`](../../test/video_frame_parser/extractor.py)
- 저장 모델 참조: [`test/video_frame_parser/models.py`](../../test/video_frame_parser/models.py)

실무 팁:

- OCR은 전체 프레임에 무조건 돌리지 말고 `change_score`가 낮은 안정 프레임 위주로 우선 적용한다.
- 동일한 정적 화면 구간에서는 OCR 결과를 재사용한다.
- JPEG 저장은 유지하되 VLM/OCR 전송용 payload는 WebP 변환을 재사용한다.

### 4.3 Stage C: Stable Segment / Event Boundary 추출

단순 frame diff threshold 하나로는 부족하다.
다음 3개 신호를 같이 본다.

- `frame_diff_score`
- `cursor_motion_score`
- `ocr_diff_score`

경계 판단 규칙:

- 변화량이 커지고 cursor가 특정 위치에 멈춘 뒤 화면 변화가 생기면 `click` 후보
- 변화량이 2회 짧게 반복되면 `double_click` 후보
- cursor 이동 경로가 길고 대상 영역이 함께 움직이면 `drag` 후보
- 텍스트 블록이 연속적으로 바뀌면 `type` 후보
- 화면 변화 없이 시간만 지나면 `wait` 후보

출력:

- `event_candidates.json`

기존 `test/workflow_extractor/auto_extractor.py`는
"프레임 추출 -> 변화 감지 -> keyframe -> VLM 추론"의 큰 흐름은 맞지만,
recipe automation 수준에서는 cursor / OCR / verification plan을 추가해야 한다.

### 4.4 Stage D: Action Inference

이 단계에서는 event 후보마다 `before`, `peak`, `after` 프레임을 묶어
action을 추론한다.

권장 입력:

- `before_frame`
- `peak_frame`
- `after_frame`
- 해당 구간 OCR diff
- cursor 마지막 안정 위치
- 필요 시 crop 이미지

권장 추론 순서:

1. 규칙 기반 1차 분류
2. OCR anchor 보강
3. VLM으로 target 설명과 verification plan 생성

VLM에 바로 "무슨 액션이었나?"만 묻지 말고 아래처럼 나눠 묻는 편이 낫다.

- 질문 1: 화면 상태 변화가 무엇인가
- 질문 2: cursor 종착점이 어떤 control과 가장 관련 있는가
- 질문 3: 이 변화가 click / double_click / type / drag 중 무엇에 가장 가까운가
- 질문 4: 실행 후 무엇을 검증해야 같은 step으로 간주할 수 있는가

이렇게 분해하면 action label과 verification criteria를 한 번에 같이 얻을 수 있다.

### 4.5 Stage E: Step Assembly

action candidate를 곧바로 최종 step으로 확정하지 않고,
아래 필드를 채운 `step_candidate`로 조립한다.

- `step_id`
- `action_type`
- `target_description`
- `target_key`
- `coordinates`
- `target_bbox`
- `before_frame_index`
- `after_frame_index`
- `preconditions`
- `success_criteria`
- `verification_plan`
- `confidence`
- `review_status`

핵심은 action 자체보다 `success_criteria`를 같이 묶는 것이다.
온라인 자동화는 결국 "무엇을 눌렀는가"보다 "그 결과 어떤 상태가 나와야 하는가"가 더 중요하다.

### 4.6 Stage F: Review / Acceptance

다음 기준으로 자동 승인 범위를 좁게 둔다.

auto-accept 가능:

- `double_click` on `ch4_cctv`처럼 window region 규칙이 강한 step
- OCR anchor와 cursor 종착점이 일치하는 text-labeled step
- before/after 변화가 명확한 dialog open/close

반드시 human review:

- password 또는 민감 입력
- drag path가 긴 step
- 복수 control 후보가 겹치는 crowded UI
- error recovery step

출력:

- `workflow_annotation.json`
- 필요 시 `review_queue.json`

## 5. 온라인 자동화와 연결되는 Schema

offline extraction 결과는 온라인 실행기로 바로 넘길 수 있어야 한다.
이때 기준이 되는 것은 현재 `workflow_1`의 step contract다.

참조 파일:

- [`poc/workflow_1/workflow_types.py`](../../poc/workflow_1/workflow_types.py)
- [`poc/workflow_1/workflow_runner.py`](../../poc/workflow_1/workflow_runner.py)

따라서 step export 시 아래 필드는 유지하는 편이 좋다.

```json
{
  "step_id": "open_recipe_tab",
  "step_type": "double_click",
  "target_description": "Recipe tab in main SEM control screen",
  "target_key": "recipe_tab",
  "preconditions": {
    "group_type": "all",
    "conditions": [
      {"condition_type": "window_visible"}
    ]
  },
  "success_criteria": {
    "group_type": "all",
    "conditions": [
      {
        "condition_type": "text_appeared",
        "target_key": "recipe_tab",
        "expected_text": "Recipe"
      }
    ]
  },
  "safety_tier": 2,
  "max_retries": 2,
  "retry_profile": "detect_then_verify",
  "idempotent": true,
  "verify_timeout_sec": 10.0
}
```

여기서 핵심은 video extraction이 low-level event log가 아니라
`WorkflowStep` 후보를 만드는 작업이어야 한다는 점이다.

## 6. Failure Class와 Fallback Strategy

video extraction은 detection 실패를 피할 수 없다.
중요한 것은 실패를 동일하게 다루지 않는 것이다.

권장 `failure_class`:

- `frame_missing`
- `frame_corrupted`
- `cursor_not_visible`
- `ocr_unreliable`
- `state_change_ambiguous`
- `action_type_ambiguous`
- `target_not_grounded`
- `verify_plan_missing`
- `non_idempotent_unresolved`

권장 대응:

| failure_class | 1차 대응 | 2차 대응 | 3차 대응 |
|---|---|---|---|
| `frame_missing` | 누락 프레임 스킵 후 재세그먼트 | 구간 폐기 | human review |
| `frame_corrupted` | 원본 재디코드 | 인접 프레임 보간 | human review |
| `cursor_not_visible` | OCR/state diff 중심 추론 | target region prior 사용 | human review |
| `ocr_unreliable` | 안정 프레임 재-OCR | crop OCR | VLM semantic text 보조 |
| `state_change_ambiguous` | 더 긴 before/after window 사용 | peak frame 교체 | human review |
| `action_type_ambiguous` | 규칙 기반 재분류 | VLM 재질문 | human review |
| `target_not_grounded` | OCR anchor 탐색 | crop-grounding | 다른 VLM fallback |
| `verify_plan_missing` | state_effect 템플릿 매핑 | VLM으로 verify plan 재생성 | human review |
| `non_idempotent_unresolved` | 자동 승인 금지 | redacted template 처리 | human review |

오프라인 추출에서도 `retry routing`이 필요하다.
이는 온라인 실행에서 사용하는 재시도 전략과 같은 철학을 따라가야 한다.

## 7. `capture_window_frames_ch4.py` 기준 구현 순서

현재 repo 기준으로는 아래 순서가 가장 현실적이다.

### Phase 1: Capture Output 읽기

새 모듈 예시:

- `poc/workflow_1/video_trajectory_extractor.py`

필수 함수:

- `load_capture_summary(summary_path) -> CaptureSession`
- `iter_capture_frames(session) -> Iterator[FrameRecord]`
- `build_capture_manifest(session) -> dict`

이 단계에서 `summary.json`과 실제 `frames/` 내용을 대조해
누락 프레임, 해상도 불일치, 시간 역전 여부를 먼저 검사한다.

### Phase 2: Observation 생성

필수 함수:

- `compute_change_scores(frames) -> list[FrameObservation]`
- `run_ocr_on_stable_frames(frames) -> list[FrameObservation]`
- `track_cursor(frames) -> list[FrameObservation]`

권장 구현:

- OCR은 `work2`에서 이미 쓰는 sidecar 또는 기존 OCR 실험 자산을 재사용
- cursor 추적은 처음부터 완전 자동을 기대하지 말고 `visible/unknown` 2단계로 시작

### Phase 3: Event Candidate 생성

필수 함수:

- `segment_event_windows(observations) -> list[EventCandidate]`
- `classify_event_candidates(events, observations) -> list[EventCandidate]`

여기서는 `Channel 4 확대`, `View tab 전환`, `tool selection` 같은
반복 패턴부터 우선 지원하는 것이 좋다.

### Phase 4: Step Candidate 생성

필수 함수:

- `infer_step_candidates(events, observations) -> list[StepCandidate]`
- `attach_verification_plan(step_candidates) -> list[StepCandidate]`
- `export_workflow_annotation(step_candidates) -> dict`

이 단계에서 기존
[`test/workflow_extractor/models.py`](../../test/workflow_extractor/models.py)와
[`poc/workflow_1/workflow_types.py`](../../poc/workflow_1/workflow_types.py)
사이를 연결하는 translator를 두는 것이 좋다.

### Phase 5: Review와 Ingest

필수 함수:

- `build_review_queue(step_candidates) -> dict`
- `ingest_reviewed_annotation(annotation_path)`

DB/RAG ingestion이 필요하면 기존
[`test/workflow_extractor/workflow_ingester.py`](../../test/workflow_extractor/workflow_ingester.py)를
재사용할 수 있다.

## 8. 구현 시 주의점

### 8.1 Keyboard 입력은 "복원"보다 "의도"를 저장한다

CCTV나 screen recording에서 password, shortcut, free-text 입력은
완전 복원이 어려운 경우가 많다.

따라서 저장 기준은 다음이 낫다.

- 실제 문자열이 확인되면 `input_text`
- 확인되지 않으면 `input_template` 또는 `input_kind`
- 민감 정보면 redacted 처리

예시:

```json
{
  "action_type": "type",
  "target_description": "Recipe name field",
  "input_text": null,
  "input_template": "{recipe_name}",
  "redacted": false
}
```

### 8.2 좌표는 절대값만 저장하지 않는다

절대 좌표만 저장하면 해상도, 창 위치, 스케일 차이에서 깨진다.
항상 함께 저장할 것:

- frame 기준 좌표
- bbox 기준 상대 위치
- 창 영역 정보
- target description

즉 offline step은 좌표 replay가 아니라 target grounding 힌트여야 한다.

### 8.3 Verification 계획이 없는 step은 반쪽짜리다

모든 step candidate는 최소한 아래 둘 중 하나가 있어야 한다.

- 상태 기반 success criterion
- 화면 기반 before/after verification plan

없으면 온라인 실행기로 넘어가면 안 된다.

## 9. Repo 기준 권장 책임 분리

문서 기준 권장 분리는 다음과 같다.

- `poc/workflow_1/capture_window_frames_ch4.py`
  창 프레임 캡처만 담당
- `poc/workflow_1/video_trajectory_extractor.py`
  capture 산출물을 observation / event / step으로 변환
- `test/workflow_extractor/`
  annotation review / VLM-assisted extraction 실험
- `test/video_frame_parser/`
  범용 프레임 / OCR / embedding / DB 모델 재사용
- `poc/work2/`
  온라인 실제 automation entrypoint

즉, offline 추출 자산은 `workflow_1` 또는 `test/`에서 만들고,
실제 운영 execution은 `work2` 또는 `workflow_1 runner`가 담당하는 구조가 맞다.

## 10. 현실적인 첫 버전 범위

첫 버전은 아래 범위로 제한하는 편이 좋다.

- 입력 소스는 `capture_window_frames_ch4.py` 결과만 사용
- action type은 `click`, `double_click`, `wait`, `type`만 우선 지원
- OCR anchor가 있는 control만 우선 자동 추출
- `Channel 4 확대`, `탭 전환`, `리스트 선택` 같은 반복 패턴 우선
- 민감 입력과 drag는 human review 필수

이렇게 시작해야 하는 이유:

- 현재 repo는 이미 online GUI grounding과 verification 쪽 자산이 있다.
- 가장 부족한 것은 "frame capture 이후 어떤 schema로 축약할 것인가"이다.
- 따라서 v1은 coverage보다 contract 품질이 더 중요하다.

## 11. 추천 구현 체크리스트

1. `capture_window_frames_ch4.py` 결과를 읽는 manifest loader를 만든다.
2. 프레임별 `change_score`, `ocr_lines`, `cursor`를 계산해 `frame_observations.jsonl`을 만든다.
3. 연속 변화 구간을 `event_candidates.json`으로 압축한다.
4. 각 event에 대해 `before/peak/after` 기반 `step_candidates.json`을 만든다.
5. 각 step candidate에 `success_criteria`와 `verification_plan`을 붙인다.
6. 사람이 검토하거나 auto-accept 규칙을 통과한 step만 `workflow_annotation.json`으로 승격한다.
7. 온라인 실행기에서는 annotation을 그대로 replay하지 말고 현재 화면 기준으로 재탐지 후 실행한다.

이 문서의 핵심은 하나다.
`capture_window_frames_ch4.py`는 이미 충분히 좋은 raw source이고,
실제로 필요한 것은 그 위에 "관찰", "행동", "검증"을 분리한 extraction contract를 쌓는 일이다.
