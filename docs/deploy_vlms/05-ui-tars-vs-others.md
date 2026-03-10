# UI-TARS 와 다른 모델의 차이

이 문서는 현재 `docs/deploy_vlms`에서 같이 다루는 3개 모델:

- `UI-TARS-1.5-7B`
- `UI-Venus-1.5-8B`
- `MAI-UI-8B`

사이의 차이를, 모델 카드와 파일 구성 기준으로 빠르게 정리한 메모다.

## 한 줄 요약

- `UI-TARS-1.5-7B`는 `Qwen2.5-VL` 기반이고, 범용 computer-use / browser / virtual-world 성향이 가장 강하다.
- `UI-Venus-1.5-8B`는 `Qwen3-VL` 기반이고, grounding + mobile + web을 하나로 합친 unified GUI agent 쪽에 더 가깝다.
- `MAI-UI-8B`도 `Qwen3-VL` 기반이지만, mobile/real-world deployment, agent-user interaction, MCP tool call, device-cloud collaboration을 더 강하게 전면에 둔다.

## 비교표

| 항목 | UI-TARS-1.5-7B | UI-Venus-1.5-8B | MAI-UI-8B |
|------|----------------|-----------------|-----------|
| 공개 모델 | `ByteDance-Seed/UI-TARS-1.5-7B` | `inclusionAI/UI-Venus-1.5-8B` | `Tongyi-MAI/MAI-UI-8B` |
| 기반 아키텍처 | `qwen2_5_vl` | `Qwen3-VL` 계열 | `Qwen3-VL` 계열 |
| 공식 설명의 핵심 | reasoning-heavy multimodal agent, computer use, browser use, phone use, game/virtual world | unified, end-to-end GUI agent, grounding + mobile + web 통합 | real-world centric GUI agent, mobile 중심, user interaction + MCP + device-cloud collaboration |
| 강하게 강조하는 능력 | thought-before-action, inference-time scaling, 범용 GUI/게임/가상환경 | grounding 성능 + cross-platform navigation + unified training/merge | 실제 배포성, 동적 환경 대응, cloud-device routing |
| 배포 관점 차이 | `chat_template.json`, `preprocessor_config.json`, 다수 safetensor shard, `qwen2_5_vl` runtime 필요 | vLLM quick start가 명시적이고 GUI 전용 학습 파이프라인 설명이 상세함 | vLLM quick start가 명시적이고 `vllm>=0.11.0`, `transformers>=4.57.0` 요구사항이 모델 카드에 직접 나옴 |

## UI-TARS 가 특히 다른 점

### 1. 베이스 모델 계열이 다르다

`UI-TARS-1.5-7B`는 Hugging Face 메타데이터상 `qwen2_5_vl` 계열이다. 반면 `UI-Venus-1.5-8B`와 `MAI-UI-8B`는 둘 다 최신 `Qwen3-VL` 계열을 기반으로 설명된다.

운영 관점에서는 이것이 가장 중요하다. `UI-TARS`는 서버 런타임이 `Qwen2.5-VL`을 제대로 읽을 수 있어야 하고, `UI-Venus`/`MAI-UI`는 `Qwen3-VL` 지원 런타임과 더 잘 맞는다.

### 2. 목표가 더 범용 computer-use 쪽이다

`UI-TARS-1.5` 모델 카드는 단순한 mobile GUI 모델이라기보다:

- computer use
- browser use
- phone use
- game / virtual world

를 같이 강조한다. 특히 "reason through its thoughts before taking action"와 inference-time scaling을 전면에 둔다.

즉, `UI-TARS`는 "스크린샷을 보고 바로 조작"만이 아니라, 더 일반적인 에이전트 reasoning 스타일을 강하게 밀고 있는 모델이다.

### 3. 파일 구성부터 runtime 힌트가 다르다

`UI-TARS-1.5-7B` 파일 목록에는 아래가 같이 보인다.

- `chat_template.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- 여러 개의 `model-0000x-of-00007.safetensors`

이 점은 배포에서 중요하다. 단순히 `MODEL_ID` 경로만 있다고 끝나는 것이 아니라, 멀티모달 processor / tokenizer / chat template / sharded weight가 모두 맞아야 한다.

여기서의 배포 해석은 모델 파일 구성에서 도출한 추론이다:

- `UI-TARS` 실패 원인이 `trust_remote_code` 하나가 아니라
- `Qwen2.5-VL` 지원 런타임 부족
- 누락된 `chat_template.json` 또는 `preprocessor_config.json`
- shard/weight 불완전 복사

같은 쪽일 가능성이 더 높다.

### 4. 용량 체감도 다를 수 있다

`UI-TARS-1.5-7B` 페이지는 7B급이지만 파일 트리 기준 약 `33.2 GB`이고, 모델 페이지 메타데이터에 tensor type이 `F32`로 보인다. 그래서 "7B인데 왜 이렇게 크지?"라는 체감이 생기기 쉽다.

즉, 실무에서 `UI-TARS`는 parameter 수보다 실제 weight 배포 크기와 shard 수 때문에 더 까다롭게 느껴질 수 있다.

## UI-Venus 와 비교하면

- `UI-Venus-1.5-8B`는 공식 설명 자체가 "unified, end-to-end GUI Agent"다.
- training pipeline도 `Mid-Training -> Offline-RL -> Online-RL -> Model Merge`로 명시되어 있다.
- browser, mobile, grounding을 한 family 안에서 묶는 방향이 강하다.

정리하면:

- `UI-TARS`는 범용 computer-use / reasoning 에이전트 느낌이 더 강하고
- `UI-Venus`는 GUI grounding + navigation 통합 완성도 쪽이 더 강하다.

## MAI-UI 와 비교하면

- `MAI-UI-8B`는 "real-world centric foundation GUI agents"를 전면에 둔다.
- 모델 카드에서 agent-user interaction, MCP tool calls, device-cloud collaboration을 명확히 말한다.
- mobile navigation benchmark와 실제 배포 architecture를 더 운영적인 언어로 설명한다.

정리하면:

- `UI-TARS`는 범용 GUI agent / computer-use reasoning 쪽
- `MAI-UI`는 실제 모바일/실서비스 배치와 협업 구조를 더 강조하는 쪽

으로 읽는 것이 맞다.

## 이 저장소에서의 실무 해석

이 문단은 위 모델 카드/파일 목록과 현재 저장소 구현을 합쳐서 정리한 추론이다.

- `UI-TARS`는 다른 두 모델보다 runtime 민감도가 높게 느껴질 수 있다.
- 그래서 `docs/deploy_vlms/scripts/serve_vlm.py`에서는 `UI-TARS` 계열일 때 `Qwen2.5-VL` runtime 가용성, `preprocessor_config.json`, `tokenizer_config.json`, `chat_template.json`, sharded weight 존재 여부를 먼저 확인하도록 바꿨다.
- `CHAT_TEMPLATE`를 비워 두면 모델 디렉터리 안의 `chat_template.json`을 자동 사용하도록 한 이유도 이 차이 때문이다.

## 출처

- UI-TARS-1.5-7B model card: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B
- UI-TARS-1.5-7B files: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/tree/main
- UI-Venus-1.5-8B model card: https://huggingface.co/inclusionAI/UI-Venus-1.5-8B
- MAI-UI-8B model card: https://huggingface.co/Tongyi-MAI/MAI-UI-8B
- MAI-UI-8B files: https://huggingface.co/Tongyi-MAI/MAI-UI-8B/tree/main
