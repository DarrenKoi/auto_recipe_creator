# 연구 문서 인덱스

`docs/research/` 안의 문서는 성격이 다른 메모가 섞여 있었고, 일부는 같은 내용을 요약본과 상세본으로 반복하고 있었다. `2026-03-17` 기준으로 아래 원칙으로 다시 정리했다.

- 토픽마다 먼저 읽어야 할 **canonical entry 문서**를 1개 둔다.
- 세부 근거, 모델별 deep dive, 구현 상세는 별도 문서로 둔다.
- 얇은 중복 요약본은 삭제하고, 필요한 내용만 새 개요 문서로 합쳤다.

## 빠른 시작

| 토픽 | 먼저 읽을 문서 | 이 문서를 읽는 이유 |
|------|----------------|---------------------|
| GUI 모델 선택 / 벤치마크 | [`gui_model_selection_and_benchmark_plan.md`](./gui_model_selection_and_benchmark_plan.md) | 어떤 모델부터 비교하고, 어떤 순서로 sidecar를 붙일지 빠르게 결정하기 좋다. |
| RCS 비디오-투-액션 | [`rcs_video_to_action_overview.md`](./rcs_video_to_action_overview.md) | 비디오를 어떻게 trajectory 자산으로 바꿀지 큰 그림을 먼저 잡기 좋다. |
| GUI 자동화 아키텍처 | [`vlm_gui_automation_for_engineering.md`](./vlm_gui_automation_for_engineering.md) | 전체 GUI agent 설계 방향과 연구 흐름을 잡기 좋다. |

## GUI 자동화 / 아키텍처

| 문서 | 역할 |
|------|------|
| [`vlm_gui_automation_for_engineering.md`](./vlm_gui_automation_for_engineering.md) | 엔지니어링 GUI 자동화의 전체 연구 배경, 상태 머신, 메모리 기반 실행 전략 |
| [`dynamic_screen_automation_phase3.md`](./dynamic_screen_automation_phase3.md) | 정적 화면이 아니라 실시간 SEM 이미지가 움직이는 동적 화면 대응 설계 |

## 모델 선택 / OCR / 런타임

| 문서 | 역할 |
|------|------|
| [`gui_model_selection_and_benchmark_plan.md`](./gui_model_selection_and_benchmark_plan.md) | 현재 저장소 기준 비교 순서, sidecar escalation 규칙, 측정 항목 |
| [`deploy_vlms_model_roles_and_pipeline_research.md`](./deploy_vlms_model_roles_and_pipeline_research.md) | 배치된 5개 모델의 역할, 강점, 한계, `poc/work2` 파이프라인 적용 해석 |
| [`paddleocr_vl_ui_venus_pipeline_research.md`](./paddleocr_vl_ui_venus_pipeline_research.md) | `PaddleOCR-VL-1.5`와 `UI-Venus` 역할 분리와 조합 방식 |
| [`ui_venus_grounding_and_ocr_for_engineering_ui.md`](./ui_venus_grounding_and_ocr_for_engineering_ui.md) | `UI-Venus` grounding 프롬프트, crop retry, OCR 좌표 보정까지 포함한 엔지니어링 UI 정밀 그라운딩 가이드 |
| [`omniparser_v2_integration_research.md`](./omniparser_v2_integration_research.md) | OCR sidecar를 OmniParser 계열로 바꿀 때의 기대 효과와 리스크 |
| [`vllm_runtime_and_unsloth_finetuning.md`](./vllm_runtime_and_unsloth_finetuning.md) | vLLM 서빙 구조, tokenizer/processor가 왜 필요한지, 작은 GPU 환경의 LoRA/QLoRA 학습 전략 |
| [`encode_decode_and_model_architecture.md`](./encode_decode_and_model_architecture.md) | tokenizer의 encode/decode와 모델 architecture의 encoder/decoder 차이, 왜 어떤 모델은 한쪽만 가지는지 정리 |
| [`pagedattention_and_prefix_caching.md`](./pagedattention_and_prefix_caching.md) | `PagedAttention`과 `prefix caching`이 각각 무엇을 최적화하는지 설명 |

## RCS 비디오-투-액션

| 문서 | 역할 |
|------|------|
| [`rcs_video_to_action_overview.md`](./rcs_video_to_action_overview.md) | 왜 비디오를 바로 재학습하지 않고 trajectory memory로 바꿔야 하는지 정리한 개요 |
| [`rcs_video_action_implementation_guide.md`](./rcs_video_action_implementation_guide.md) | `episode -> step -> target -> verification` 구현 상세 가이드 |
| [`rcs_video_action_task_breakdown.md`](./rcs_video_action_task_breakdown.md) | 실제 작업 패키지 분해, 선행 조건, 완료 기준 |

## 이번 정리에서 합쳐진 문서

| 기존 문서 | 현재 기준 |
|----------|-----------|
| `engineering_screen_vlm_recommendations.md` | `gui_model_selection_and_benchmark_plan.md` + `deploy_vlms_model_roles_and_pipeline_research.md` 로 통합 |
| `gui_vlm_benchmark_report.md` | `gui_model_selection_and_benchmark_plan.md` 로 통합 |
| `rcs_video_to_action_research.md` | `rcs_video_to_action_overview.md` + `rcs_video_action_implementation_guide.md` 로 통합 |

## LLM 실행 기초

LLM에서 tokenizer는 **문자열을 모델이 처리할 수 있는 토큰 ID 시퀀스로 바꾸는 변환기**다. 모델은 글자나 단어를 직접 읽는 것이 아니라, tokenizer가 만든 토큰 ID를 embedding으로 바꾼 뒤 다음 토큰 확률을 계산한다.

- 입력 시에는 `text -> tokens` 변환이 필요하다. prompt를 숫자 ID 배열로 바꿔야 GPU 위 모델이 prefill/decode를 수행할 수 있다.
- 출력 시에는 `tokens -> text` 변환이 필요하다. 모델이 생성한 다음 토큰 ID들을 다시 사람이 읽을 수 있는 문자열로 복원해야 한다.
- 같은 base model이라도 **반드시 그 모델에 맞는 tokenizer**를 써야 한다. vocabulary, special token, chat template이 어긋나면 출력 품질이 깨지거나 프롬프트 형식이 틀어진다.
- VLM에서는 보통 텍스트는 tokenizer가, 이미지/비디오는 processor가 맡는다. 둘 다 결국 모델이 처리 가능한 token/embedding 형태로 바꾸는 전처리 단계다.

즉, tokenizer는 선택 부품이 아니라 **모델 실행 경로의 일부**다. tokenizer 없이 LLM은 자연어 입력을 숫자 시퀀스로 바꾸지 못하고, 생성 결과도 다시 텍스트로 해석할 수 없다. 관련 상세 메모는 [`vllm_runtime_and_unsloth_finetuning.md`](./vllm_runtime_and_unsloth_finetuning.md)를 보면 된다.
