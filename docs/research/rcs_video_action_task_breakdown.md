# RCS 비디오-투-액션 작업 분해 문서 (2026-03-13)

## 목적

이 문서는 `RCS` 비디오-투-액션 시스템을 실제 작업 단위로 쪼개기 위한 문서다.
각 작업 패키지마다 아래를 정의한다.

- 목적
- 입력 / 출력
- 담당 모듈
- 선행 조건
- 완료 기준
- 리스크

## 전체 작업 패키지 목록

| ID | 작업 패키지 | 우선순위 | 상태 |
|----|-------------|----------|------|
| `WP-01` | 데이터 인벤토리 정리 | P0 | pending |
| `WP-02` | representative AVI gold set 선정 | P0 | pending |
| `WP-03` | manual-control episode miner 설계 | P0 | pending |
| `WP-04` | white cursor detection 설계 | P0 | pending |
| `WP-05` | cursor tracking / smoothing 설계 | P0 | pending |
| `WP-06` | click / double click 추출 설계 | P0 | pending |
| `WP-07` | drag / scroll 추출 설계 | P0 | pending |
| `WP-08` | type_candidate / OCR diff 설계 | P0 | pending |
| `WP-09` | step segmentation / schema 정리 | P0 | pending |
| `WP-10` | target grounding with VLM/OCR 설계 | P0 | pending |
| `WP-11` | local objective 생성 방식 정리 | P1 | pending |
| `WP-12` | trajectory storage / retrieval schema 정리 | P1 | pending |
| `WP-13` | next-action planner prompt/spec 정리 | P1 | pending |
| `WP-14` | step verification 규칙 설계 | P1 | pending |
| `WP-15` | human-in-the-loop 운영 규칙 정리 | P1 | pending |
| `WP-16` | evaluation protocol / gold label 설계 | P1 | pending |

## WP-01 데이터 인벤토리 정리

### 목적

전체 AVI 자산의 메타데이터를 정리해 이후 작업의 입력 기반을 만든다.

### 입력

- 원본 AVI 파일 전체

### 출력

- video inventory table

### 관련 모듈

- [`extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/extractor.py)
- [`models.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py)

### 선행 조건

- 파일 접근 가능
- 파일명 규칙 또는 세션 정보 일부 확보

### 완료 기준

- 모든 비디오에 `video_id`, duration, fps, resolution, codec 가 정리되어 있다.
- tool / session / engineer 매핑 가능 여부가 표시되어 있다.

### 리스크

- 파일명만으로 세션 정보를 알 수 없을 수 있음
- 메타데이터만으로 tool 종류를 모를 수 있음

## WP-02 representative AVI gold set 선정

### 목적

정확도 평가와 알고리즘 설계를 위한 대표 샘플 세트를 만든다.

### 입력

- inventory table

### 출력

- gold set video list
- tool 별 대표 샘플 목록

### 선행 조건

- `WP-01` 완료

### 완료 기준

- 최소 10~20개 대표 영상이 선정되어 있다.
- easy / medium / hard 난이도 구분이 있다.

### 리스크

- 너무 쉬운 샘플만 고르면 실제 성능을 과대평가할 수 있음

## WP-03 manual-control episode miner 설계

### 목적

자동 운전 90%를 제거하고 manual episode 후보만 남기는 규칙을 정리한다.

### 입력

- representative AVI

### 출력

- episode mining signal list
- episode JSON schema

### 관련 모듈

- [`extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/extractor.py)
- [`action_extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py)

### 완료 기준

- candidate signal 정의가 문서화되어 있다.
- episode 분류 라벨 `manual_control`, `auto_running`, `uncertain` 이 합의되었다.

### 리스크

- manual과 auto의 경계가 흐린 구간이 많을 수 있음

## WP-04 white cursor detection 설계

### 목적

흰색 커서를 frame-level로 검출하는 전략을 정리한다.

### 입력

- representative frame samples

### 출력

- cursor template bank spec
- detection rule spec

### 관련 모듈

- [`action_extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py)

### 완료 기준

- 커서 shape 목록이 정리되어 있다.
- 탐지 score 정의가 있다.
- 추적 실패 시 fallback 규칙이 정리되어 있다.

### 리스크

- tool UI의 흰색 아이콘과 커서를 헷갈릴 수 있음
- RCS 압축 노이즈로 template score 가 흔들릴 수 있음

## WP-05 cursor tracking / smoothing 설계

### 목적

검출된 cursor 후보를 시간축으로 연결해 안정적 trajectory 를 만든다.

### 입력

- frame-level cursor candidates

### 출력

- cursor track schema
- smoothing / relink 규칙

### 관련 모듈

- [`action_extractor.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/action_extractor.py)

### 완료 기준

- `tracked`, `recovered`, `missing` 상태 정의가 있다.
- fragmentation 처리 규칙이 있다.

### 리스크

- cursor shape 전환 시 track 이 끊길 수 있음

## WP-06 click / double click 추출 설계

### 목적

cursor 정지와 local UI response 를 결합해 click 이벤트를 추출한다.

### 입력

- cursor trajectory
- local change maps

### 출력

- click event schema
- click confidence rule

### 완료 기준

- click / double_click 구분 기준이 있다.
- context menu 기반 right_click 후보 처리 규칙이 있다.

### 리스크

- hover highlight 를 click 으로 오탐할 수 있음

## WP-07 drag / scroll 추출 설계

### 목적

drag 와 scroll 을 분리해 복원한다.

### 입력

- cursor trajectory
- panel motion evidence

### 출력

- `drag_*` subtype spec
- `scroll` event spec

### 완료 기준

- drag 와 scroll 의 분리 규칙이 있다.
- scroll bar thumb evidence 활용 여부가 정리되어 있다.

### 리스크

- panel redraw 를 scroll 로 잘못 볼 수 있음

## WP-08 type_candidate / OCR diff 설계

### 목적

video-only 환경에서 keyboard input 을 text delta 수준으로 복원한다.

### 입력

- focused input region
- pre/post OCR text

### 출력

- `type_candidate` schema
- OCR diff confidence rule

### 관련 모듈

- [`pipeline_ocr.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/pipeline_ocr.py)

### 완료 기준

- before/after text 저장 형식이 정해져 있다.
- low-confidence 처리 방식이 정리되어 있다.

### 리스크

- 숫자와 기호 OCR 오차가 큼
- overwrite / append 구분 실패 가능

## WP-09 step segmentation / schema 정리

### 목적

event 를 planner 에 쓸 수 있는 step 단위로 변환한다.

### 입력

- event list

### 출력

- step schema
- pre/post stable frame 정의

### 관련 모듈

- [`models.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py)

### 완료 기준

- step JSON field 가 확정되어 있다.
- granularity 원칙이 문서화되어 있다.

### 리스크

- step 이 너무 크거나 너무 잘게 쪼개질 수 있음

## WP-10 target grounding with VLM/OCR 설계

### 목적

좌표 중심 action 을 의미 있는 target element 로 변환한다.

### 입력

- pre_frame
- cursor endpoint
- candidate UI crops

### 출력

- target grounding rule
- model routing policy

### 관련 모듈

- [`vlm_screen_analysis.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py)
- [`flask_vlm.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/flask_vlm.py)

### 완료 기준

- `UI-Venus / UI-TARS / MAI-UI / PaddleOCR / GOT-OCR` 역할 분담이 확정되어 있다.
- target schema 가 정리되어 있다.

### 리스크

- small widget grounding 실패
- OCR과 VLM 판단이 충돌

## WP-11 local objective 생성 방식 정리

### 목적

trajectory retrieval 품질을 높이기 위해 local objective 를 생성하는 규칙을 정리한다.

### 입력

- step cluster
- OCR text
- target summary

### 출력

- objective naming convention
- global/local task taxonomy

### 완료 기준

- objective 길이와 형식이 합의되었다.

### 리스크

- objective naming 이 제각각이면 검색 품질이 나빠짐

## WP-12 trajectory storage / retrieval schema 정리

### 목적

offline 에서 만든 trajectory 를 online retrieval 에 바로 쓸 수 있게 저장 구조를 정의한다.

### 입력

- step / episode / objective 데이터

### 출력

- storage schema
- retrieval index spec

### 관련 모듈

- [`rag_context_manager.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_context_manager.py)
- [`models.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/models.py)

### 완료 기준

- visual / text / metadata 검색 조합이 정의되어 있다.
- 저장 단위와 인덱스가 정리되어 있다.

### 리스크

- visual similarity 와 task similarity 가 서로 다른 결과를 낼 수 있음

## WP-13 next-action planner prompt/spec 정리

### 목적

VLM이 자유 서술이 아니라 single-step JSON 을 반환하게 하는 prompt / output schema 를 확정한다.

### 입력

- 현재 screenshot
- current goal
- retrieved examples

### 출력

- planner prompt spec
- next_action JSON schema

### 관련 모듈

- [`rag_prompt_builder.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/vlm_input_control/rag_prompt_builder.py)
- [`vlm_screen_analysis.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py)

### 완료 기준

- 허용 action type 이 확정되어 있다.
- `expected_effect` 필드가 포함되어 있다.

### 리스크

- 모델이 설명 문장을 길게 쓰고 JSON 형식을 깨뜨릴 수 있음

## WP-14 step verification 규칙 설계

### 목적

실행한 action 의 성공/실패를 후속 판단 가능한 형태로 남긴다.

### 입력

- pre/post screenshot
- next_action
- expected_effect

### 출력

- verification schema
- rule / OCR / VLM verification policy

### 관련 모듈

- [`vlm_screen_analysis.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/vlm_screen_analysis.py)
- [`pipeline_ocr.py`](/Users/daeyoung/Codes/auto_recipe_creator/poc/work2/pipeline_ocr.py)

### 완료 기준

- success criteria 가 action type 별로 정리되어 있다.
- verification evidence 형식이 있다.

### 리스크

- planner 와 verifier 가 같은 bias 를 공유할 수 있음

## WP-15 human-in-the-loop 운영 규칙 정리

### 목적

초기 semi-automatic 운영에서 언제 human approval 을 받을지 명확히 한다.

### 입력

- planner output
- verification result

### 출력

- approval threshold
- escalation policy
- human correction logging spec

### 완료 기준

- `ask_human` 전환 조건이 합의되었다.
- human correction 저장 필드가 정의되었다.

### 리스크

- 기준이 느슨하면 위험 action 이 실행될 수 있음
- 기준이 너무 엄격하면 자동화 가치가 줄어듦

## WP-16 evaluation protocol / gold label 설계

### 목적

각 단계별 품질을 측정하기 위한 기준 데이터와 지표를 정한다.

### 입력

- representative AVI set

### 출력

- gold label protocol
- metric definition

### 완료 기준

- episode / cursor / click / target / end-to-end 단계별 metric 이 있다.
- 최소한의 annotated gold set 이 정의되어 있다.

### 리스크

- gold label 비용이 큼
- annotator 간 기준 차이 발생 가능

## 권장 실행 순서

아래 순서로 진행하는 것이 좋다.

1. `WP-01`
2. `WP-02`
3. `WP-03`
4. `WP-04`
5. `WP-05`
6. `WP-06`
7. `WP-07`
8. `WP-08`
9. `WP-09`
10. `WP-10`
11. `WP-14`
12. `WP-12`
13. `WP-13`
14. `WP-15`
15. `WP-16`
16. `WP-11`

## 각 단계에서 문서로 반드시 남길 것

각 작업 패키지는 끝날 때 아래를 별도 메모로 남겨야 한다.

1. 최종 입력 / 출력 스키마
2. 실패 케이스 예시
3. 사람이 봤을 때 직관적으로 이해되는 before/after 사례
4. threshold 와 그 이유
5. 다음 단계에 넘길 제약 조건

## 관련 문서

- [RCS 비디오-투-액션 개요](/Users/daeyoung/Codes/auto_recipe_creator/docs/research/rcs_video_to_action_overview.md)
- [RCS 비디오-투-액션 구현 가이드](/Users/daeyoung/Codes/auto_recipe_creator/docs/research/rcs_video_action_implementation_guide.md)
