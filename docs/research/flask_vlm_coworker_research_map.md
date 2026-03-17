# Flask VLM Coworker Research Map

이 문서는 동료 소개용 발표를 만들 때 `docs/research/` 안의 문서를 어떤 순서로 읽으면 되는지 빠르게 정리한 안내서다.
목표는 "모델 비교 기준 -> 모델 역할 분리 -> OCR/파이프라인 해석" 순서로 읽게 만드는 것이다.

## 1. 먼저 읽을 canonical research 문서

1. [`gui_model_selection_and_benchmark_plan.md`](./gui_model_selection_and_benchmark_plan.md)
   GUI primary 비교 순서, sidecar escalation 규칙, 측정 항목을 가장 빠르게 이해할 수 있다.
2. [`deploy_vlms_model_roles_and_pipeline_research.md`](./deploy_vlms_model_roles_and_pipeline_research.md)
   현재 배치된 5개 모델의 역할, 강점, 한계, 권장 파이프라인을 한 번에 정리한다.
3. [`paddleocr_vl_ui_venus_pipeline_research.md`](./paddleocr_vl_ui_venus_pipeline_research.md)
   `UI-Venus`와 `PaddleOCR-VL-1.5`를 어떤 순서로 조합해야 안정적인지 설명한다.

## 2. 발표 중 필요할 때만 펼쳐볼 deep dive

- [`omniparser_v2_integration_research.md`](./omniparser_v2_integration_research.md)
  OCR sidecar를 OmniParser 계열로 바꿀 때의 기대 효과와 리스크를 설명한다.
- [`vllm_runtime_and_unsloth_finetuning.md`](./vllm_runtime_and_unsloth_finetuning.md)
  vLLM 런타임 구조, tokenizer/processor 역할, 작은 GPU 환경의 학습 전략을 정리한다.
- [`encode_decode_and_model_architecture.md`](./encode_decode_and_model_architecture.md)
  tokenizer encode/decode와 model encoder/decoder 개념을 발표 보충용으로 설명할 때 쓴다.
- [`pagedattention_and_prefix_caching.md`](./pagedattention_and_prefix_caching.md)
  서빙 성능 설명이 필요할 때만 본다.

## 3. 발표 범위에서 후순위인 문서

- `rcs_video_*`
- `dynamic_screen_automation_phase3.md`
- `vlm_gui_automation_for_engineering.md`

이 문서들은 장기 GUI agent 설계나 비디오-투-액션 연구에는 중요하지만,
이번 coworker introduction deck의 1차 핵심 범위는 아니다.

## 4. 발표용 핵심 메시지

- 비교는 `UI-Venus` vs `UI-TARS` 같은 primary head-to-head부터 시작한다.
- `MAI-UI`는 primary 경쟁자라기보다 zoom-in sidecar에 가깝다.
- exact text authority 는 OCR 모델이 맡는다.
- 5개 모델을 항상 동시에 부르는 구조는 피한다.
- `observe -> decide -> act -> verify` 흐름으로 pipeline을 설명한다.
