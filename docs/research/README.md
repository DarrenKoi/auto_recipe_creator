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
| [`omniparser_v2_integration_research.md`](./omniparser_v2_integration_research.md) | OCR sidecar를 OmniParser 계열로 바꿀 때의 기대 효과와 리스크 |
| [`vllm_runtime_and_unsloth_finetuning.md`](./vllm_runtime_and_unsloth_finetuning.md) | vLLM 서빙 구조와 작은 GPU 환경의 LoRA/QLoRA 학습 전략 |

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
