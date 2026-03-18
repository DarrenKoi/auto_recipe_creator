# RCS Video-To-Action

이 문서는 RCS video overview, 구현 가이드, task breakdown을 하나의 roadmap으로 합친 문서입니다.

## 1. 핵심 결정

raw AVI 파일을 곧바로 training input으로 취급하지 않습니다.

처음으로 유용한 자산은 다음입니다:

`video -> manual-control episode -> step trajectory -> retrieval memory`

이 방식은 직접 end-to-end 재학습하는 것보다 더 현실적이며, 현재 저장소 구조에도 더 잘 맞습니다.

## 2. 목표 출력

각 유의미한 step은 최종적으로 최소한 다음 정보를 가져야 합니다:

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

목표는 raw replay만이 아니라 검색 가능한 trajectory memory입니다.

## 3. 권장 Pipeline

1. 모든 AVI asset을 inventory한다
2. 대표성 있는 gold set을 고른다
3. manual-control episode를 추출한다
4. 흰색 cursor를 detect하고 track한다
5. click, double-click, drag, scroll, `type_candidate`를 추출한다
6. VLM과 OCR로 target을 grounding한다
7. 로컬 objective와 effect를 요약한다
8. trajectory record를 저장한다
9. online 실행 중 유사 record를 retrieval한다
10. 새로운 screenshot으로 post-action 성공 여부를 verify한다

## 4. Video-Only가 어려운 이유

알려진 제약:

- 녹화된 시간 대부분이 자동 실행 구간일 수 있어 유용하지 않을 수 있음
- keyboard event는 직접 관찰되지 않음
- text entry는 종종 OCR diff로 재구성해야 함
- cursor shape 변화가 단순 tracking을 깨뜨릴 수 있음

그래서 cursor와 국소 UI 변화 신호가 핵심 feature가 됩니다.

## 5. 작업 패키지 압축

기존 task list는 실무적으로 다음 다섯 단계로 읽을 수 있습니다:

### Phase A: Data And Sampling

- inventory table
- representative gold set

### Phase B: Episode And Cursor Extraction

- manual-control episode mining
- white cursor detection
- cursor tracking 및 recovery

### Phase C: Event Building

- click / double-click / drag / scroll 추출
- OCR-diff 기반 `type_candidate`
- step segmentation

### Phase D: Grounding And Memory

- VLM/OCR 기반 target grounding
- local objective 생성
- trajectory 저장 및 retrieval schema

### Phase E: Online Planner And Verification

- next-action planning
- post-action verification
- human-in-the-loop 규칙
- evaluation protocol

## 6. Repo 매핑

- `test/video_frame_parser/`: offline frame 및 episode 추출
- `test/vlm_input_control/`: retrieval context 및 이전 prompt-building 로직
- `poc/work2/`: online state recognition, action planning, verification

이 구조는 `poc/work2`를 대체하는 것이 아니라 확장하는 구조입니다.

## 7. 평가 우선순위

다음을 측정합니다:

- cursor recall 및 위치 오차
- click/double-click 추출 품질
- 입력 값에 대한 OCR-diff 품질
- target grounding 정확도
- planner 성공률
- end-to-end step verification 정확도

## 8. 실용적인 범위 규칙

첫 버전의 범위는 좁게 유지합니다:

- video-only 가정
- white cursor 가정
- recipe 관련 GUI action만 대상
- 전체 keyboard replay 재구성은 제외

이 범위는 충분히 유용하면서도 실제 구현 가능한 크기입니다.
