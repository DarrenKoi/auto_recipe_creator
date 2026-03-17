# RCS 비디오-투-액션 개요 (2026-03-17)

## 목적

이 문서는 `RCS` 녹화 영상을 어떻게 자동화 자산으로 바꿀지 큰 그림만 빠르게 합의하기 위한 canonical 개요 문서다.

핵심 질문은 아래 3개다.

1. 비디오를 바로 학습 데이터로 넣는 것이 맞는가
2. `video-only` 조건에서 어떤 산출물을 먼저 만들어야 하는가
3. 현재 저장소에서는 어떤 계층으로 연결하는 것이 맞는가

세부 구현은 [`rcs_video_action_implementation_guide.md`](./rcs_video_action_implementation_guide.md), 실제 작업 패키지는 [`rcs_video_action_task_breakdown.md`](./rcs_video_action_task_breakdown.md) 에 둔다.

## 핵심 결론

가장 현실적인 방향은 **비디오를 바로 재학습하지 않고, 영상에서 전문가 trajectory를 먼저 구조화하는 것**이다.

즉, 아래 3단 구조가 기준이다.

1. `AVI -> manual-control episode -> step trajectory` 로 오프라인 자산화
2. `현재 화면 + 현재 목표 + 과거 trajectory 예시` 로 다음 액션 결정
3. 액션 후 `pre/post screenshot` 기반 verification 으로 성공 여부 판정

## 왜 이 문제가 어려운가

`RCS` 화면은 일반 Windows UI 자동화와 다르다.

- 실제 툴 UI가 원격 화면 스트림 안에 렌더링된다.
- accessibility tree 나 selector 기반 자동화가 약할 가능성이 높다.
- 전체 가동 시간의 대부분은 자동 운전이라, 비디오 대부분이 학습 가치가 낮다.

따라서 전체 영상을 통째로 다루기보다, 먼저 **사람이 실제로 개입한 짧은 구간**을 찾고, 그 구간을 **step 단위**로 쪼개는 편이 맞다.

## 권장 산출물

영상에서 최종적으로 필요한 것은 "비디오 파일"이 아니라 아래 구조다.

```text
observation_t
action_t
observation_t+1
effect_t
```

최소 필드는 아래 정도가 필요하다.

- `pre_frame`
- `post_frame`
- `action_type`
- `action_args`
- `target_bbox`
- `target_text`
- `effect_summary`
- `task_objective`
- `success`
- `confidence`

## video-only 조건에서의 해석

현재 전제는 아래 두 가지다.

- `AVI` 안에 흰색 마우스 커서가 보인다.
- 입력 로그 없이 비디오만 있다.

이 조건에서 얻는 장단점은 분명하다.

### 장점

- cursor tracking을 복원의 출발점으로 삼을 수 있다.
- click, double click, drag, scroll 후보를 화면 전체 변화량보다 안정적으로 찾을 수 있다.

### 한계

- 실제 key sequence 복원은 어렵다.
- typed text는 OCR diff 기반 `type_candidate` 수준으로만 다루는 것이 현실적이다.
- 그래서 1차 범위는 raw keyboard replay가 아니라 step memory 구축이어야 한다.

## 현재 저장소와의 연결

역할 분리는 아래처럼 두는 편이 자연스럽다.

- `test/video_frame_parser/`: 오프라인 episode / step 추출
- `test/vlm_input_control/`: retrieval / prompt 구성
- `poc/work2/`: online observation / planning / verification

즉, 기존 코드를 버리는 방향이 아니라, 오프라인 memory 계층을 추가해 현재 `poc/work2` 실행 흐름을 보강하는 방향이다.

## 모델 역할 분리

현재 스택은 아래처럼 두는 편이 맞다.

| 역할 | 권장 모델 |
|------|-----------|
| primary GUI grounding | `UI-Venus` 또는 `UI-TARS` |
| zoom-in sidecar | `MAI-UI` |
| dense text reading | `PaddleOCR-VL-1.5` |
| hard OCR fallback | `GOT-OCR-2.0-hf` |
| teacher / verifier | `Kimi-K2.5` |

핵심은 **primary GUI 모델이 다음 행동 후보를 고르고, OCR 계열이 exact text를 보강하며, verification을 별도 단계로 분리하는 것**이다.

## 구현 우선순위

큰 순서는 아래면 충분하다.

1. representative AVI set 정리
2. manual-control episode mining
3. cursor tracking
4. event extraction
5. target grounding + OCR diff
6. retrieval memory
7. next-action planning
8. step verification

## 이 문서 다음에 볼 것

- 구현 상세와 스키마: [`rcs_video_action_implementation_guide.md`](./rcs_video_action_implementation_guide.md)
- 작업 패키지와 완료 기준: [`rcs_video_action_task_breakdown.md`](./rcs_video_action_task_breakdown.md)
