# RCS Video-To-Action

이 문서는 RCS video-to-action 작업의 장기 roadmap을 현재 repo 구조에 맞춰 다시 정리한 것입니다.
중요한 점은, 이것이 현재 `poc/work2` mainline entrypoint를 대체하는 문서가 아니라는 점입니다.

## 1. 현재 위치

현재 mainline은 다음에 가깝습니다:

- live screenshot 기반 분석
- 로그인 창 grounding benchmark
- OCR sidecar 검증

video-to-action은 이 흐름을 대체하는 것이 아니라, offline memory와 retrieval 기반 planning을 보강하는 확장 경로입니다.

## 2. 핵심 결정

raw AVI를 곧바로 training input 또는 action 정책의 단일 source로 보지 않습니다.

처음으로 유용한 자산은 다음입니다:

`video -> episode -> step trajectory -> retrieval memory`

이 방식이 더 현실적인 이유:

- 현재 repo에 이미 frame parsing 실험이 존재함
- online planner와 offline memory를 느슨하게 결합할 수 있음
- end-to-end 재학습 없이도 바로 실험 가능함

## 3. 목표 출력

각 유의미한 step은 최소한 다음 정보를 가져야 합니다:

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

추가로 현재 repo와 맞추려면 다음도 있으면 좋습니다:

- `state_id`
- `service_hint`
- `evidence_sources`
- `verification_status`

목표는 raw replay가 아니라 검색 가능한 trajectory memory입니다.

## 4. 권장 Pipeline

1. AVI asset inventory를 만든다
2. representative gold set을 고른다
3. manual-control episode를 추출한다
4. cursor를 detect하고 track한다
5. click, double-click, drag, scroll, `type_candidate`를 추출한다
6. frame 전후 차이와 OCR로 text evidence를 만든다
7. VLM으로 target grounding과 objective를 요약한다
8. trajectory record를 저장한다
9. online 실행 시 유사 record를 retrieval한다
10. 새로운 screenshot으로 post-action 성공 여부를 verify한다

## 5. Video-Only가 어려운 이유

알려진 제약:

- 녹화 중 상당 구간이 자동 실행일 수 있음
- keyboard event는 직접 보이지 않음
- text entry는 OCR diff나 context reconstruction이 필요함
- cursor shape, remote session artifacts, compression noise가 tracking을 깨뜨릴 수 있음

따라서 핵심 feature는 다음입니다:

- cursor movement
- local UI change
- OCR diff
- 전후 frame의 semantic state 변화

## 6. 작업 패키지

### Phase A: Data And Sampling

- asset inventory
- representative gold set
- labeling 기준 정리

### Phase B: Episode And Cursor Extraction

- manual-control episode mining
- cursor detection
- cursor tracking recovery

### Phase C: Event Building

- click / double-click / drag / scroll 추출
- OCR diff 기반 `type_candidate`
- step segmentation

### Phase D: Grounding And Memory

- VLM/OCR 기반 target grounding
- local objective 생성
- trajectory schema 설계
- retrieval memory 저장

### Phase E: Online Planner And Verification

- next-action planning
- post-action verification
- human-in-the-loop 규칙
- evaluation protocol

## 7. Repo 매핑

- `test/video_frame_parser/`: offline frame parsing 및 관련 테스트
- `test/video_frame_parser/tests/`: parser unit test
- `test/vlm_input_control/`: retrieval context 및 이전 input-control 실험
- `test/workflow_extractor/`: workflow extraction 실험
- `poc/work2/`: online screen analysis, benchmark, verification entrypoint

즉:

- offline memory 쪽은 `test/` 아래 실험 자산
- online operational entrypoint는 여전히 `poc/work2/`

## 8. 평가 우선순위

다음을 측정합니다:

- cursor recall 및 위치 오차
- click/double-click 추출 품질
- OCR diff 기반 text recovery 품질
- target grounding 정확도
- retrieval hit usefulness
- planner 성공률
- post-action verification 정확도

## 9. 실용적인 범위 규칙

첫 버전의 범위는 좁게 유지합니다:

- video-only 가정
- recipe 관련 GUI action 우선
- white cursor 또는 명확한 cursor 전제
- 전체 keyboard replay 복원은 제외
- offline memory 생성과 online retrieval 연결까지만 우선

이 범위가 현실적인 이유는, 현재 repo mainline도 아직 perception/verification 안정화 단계이기 때문입니다.
