# vLLM 런타임 구조와 Unsloth 기반 파인튜닝 메모 (2026-03-13)

## 목적

이 문서는 아래 질문에 답하기 위한 연구 메모다.

1. `vLLM`은 LLM/VLM을 어떤 메커니즘으로 실행하는가?
2. 작은 GPU 환경에서 모델을 어떻게 파인튜닝하는 것이 현실적인가?
3. `Unsloth`는 좋은 선택인가?
4. 학습 데이터는 어떻게 모으고, 어떻게 학습하고, 학습 결과를 어떻게 실제 서빙 모델에 적용하는가?

핵심 전제는 하나다.

- `vLLM`은 기본적으로 **추론/서빙 런타임**이다.
- 실제 **SFT/LoRA/QLoRA 파인튜닝**은 보통 `Transformers + TRL + PEFT` 또는 그 위를 가속하는 `Unsloth`에서 수행한다.

이 구분을 먼저 명확히 해야 전체 스택이 헷갈리지 않는다.

## 결론 요약

- `vLLM`의 본질은 **고처리량 inference engine + OpenAI 호환 서버**다.
- LLM은 `vllm.LLM` 또는 `vllm serve`로 실행하고, 내부적으로는 **`PagedAttention`, `continuous batching`, `prefix caching`, `chunked prefill`** 같은 최적화로 GPU 메모리와 요청 스케줄링을 효율화한다.
- VLM도 같은 런타임 위에서 돌지만, 텍스트 프롬프트 외에 `multi_modal_data`로 이미지/비디오/오디오를 함께 넘기고, 모델별 Hugging Face 포맷에 맞는 프롬프트를 써야 한다.
- 파인튜닝 프레임워크의 기본 축은 `Transformers + Datasets + PEFT + TRL`이다.
- `Unsloth`는 이 Hugging Face 축을 유지하면서 **저 VRAM, 단일/소수 GPU, 빠른 LoRA/QLoRA 반복**에 특히 강한 선택지다.
- 자원 관점에서 보면 text LLM `3B/7B/8B`는 `QLoRA`로 단일 consumer GPU에서도 시작 가능하지만, 긴 context, 멀티모달 vision layer 튜닝, `DPO/RLHF` 단계는 요구 자원이 빠르게 커진다.
- 작은 GPU에서 처음부터 full fine-tuning으로 가는 것보다 **LoRA/QLoRA SFT -> 평가 -> 필요 시 DPO/RLHF** 순서가 현실적이다.
- 학습 데이터는 공개 데이터셋만으로 끝나는 경우가 드물고, 실제 운영 태스크에서는 **자체 로그/스크린샷/정답 JSON/사람 교정 데이터**가 성능 차이를 만든다.
- 학습 결과 적용은 보통 2가지다: base model + LoRA adapter를 따로 둔 채 `vLLM`에서 LoRA 서빙하거나, adapter를 16-bit로 merge해서 일반 모델처럼 `vLLM serve` 한다.

## 1. `vLLM`은 무엇인가

vLLM 공식 문서는 이 프로젝트를 "**LLM inference and serving**" 라이브러리로 설명한다. 즉, 기본 역할은 모델 학습이 아니라 모델 실행이다.

실무적으로 보면 `vLLM`은 아래 3개 층으로 이해하면 가장 쉽다.

1. **모델 로딩 층**
- Hugging Face 모델/토크나이저/프로세서를 읽어온다.
- 언어 모델뿐 아니라 이미지/비디오/오디오를 받는 멀티모달 모델도 지원한다.

2. **엔진 층**
- 요청을 큐에 넣고 batching, KV-cache 관리, 토큰 생성 스케줄링을 처리한다.
- 처리량과 메모리 효율을 위해 PagedAttention, continuous batching, prefix caching 같은 최적화를 쓴다.

3. **서빙 층**
- Python 코드 안에서 `LLM(...)`로 직접 호출하거나
- `vllm serve ...`로 OpenAI 호환 HTTP 서버를 띄울 수 있다.

즉, `vLLM`은 "모델 파일을 GPU에 올린 뒤 빠르게 여러 요청을 처리하는 런타임"이라고 보면 된다.

## 2. LLM을 `vLLM`으로 실행하는 메커니즘

### 2.1 기본 실행 흐름

LLM 쪽 흐름은 대략 아래와 같다.

1. Hugging Face에서 base model과 tokenizer를 읽는다.
2. 요청 프롬프트를 tokenizer/chat template 기준으로 토큰화한다.
3. prefill 단계에서 입력 prefix에 대한 KV-cache를 만든다.
4. decode 단계에서 다음 토큰들을 반복 생성한다.
5. 여러 요청을 continuous batching으로 섞어 GPU를 최대한 바쁘게 유지한다.
6. prefix가 겹치면 prefix caching으로 기존 KV-cache를 재사용한다.
7. 결과를 텍스트 스트림 또는 완료 응답으로 돌려준다.

vLLM Quickstart 문서도 `llm.generate(...)`가 입력 프롬프트를 엔진의 waiting queue에 넣고 high-throughput으로 실행한다고 설명한다.

### 2.2 왜 빠른가

공식 문서와 논문 기준 핵심은 아래 4개다.

#### a. PagedAttention

vLLM 논문은 KV-cache가 크고 동적으로 늘었다 줄었다 해서 fragmentation과 중복 메모리 낭비가 심해진다고 설명한다. 이를 해결하려고 **OS의 virtual memory/paging과 비슷한 방식**으로 KV-cache를 block/page 단위로 관리하는 PagedAttention을 제안했다.

실무 해석은 이렇다.

- 요청마다 KV-cache를 커다란 연속 메모리 덩어리로 잡지 않는다.
- block 단위로 끊어서 필요할 때 할당한다.
- 그래서 batch를 더 크게 잡을 수 있고 throughput이 올라간다.

### b. Continuous batching

공식 문서는 incoming requests를 continuous batching한다고 설명한다. 전통적인 "batch 하나 끝나고 다음 batch" 방식보다, 들어오는 요청들을 decode/prefill 시점에 계속 섞어 넣어 GPU idle을 줄이는 방식이다.

### c. Prefix caching

vLLM의 APC 문서는 동일 prefix를 가진 요청이 있으면 기존 KV-cache를 재사용해서 shared prefix 계산을 건너뛸 수 있다고 설명한다. 긴 system prompt, 같은 문서에 대한 반복 질의, 같은 UI instruction prefix 같은 패턴에서 효율이 크다.

### d. Chunked prefill / quantization / CUDA graph

공식 문서 기준 `chunked prefill`, 여러 quantization 방식, CUDA/HIP graph, FlashAttention/FlashInfer 연계도 주요 최적화다. 긴 컨텍스트나 많은 동시 요청에서 유리하다.

## 3. VLM을 `vLLM`으로 실행하는 메커니즘

### 3.1 LLM과 동일한 점

VLM도 기본 구조는 같다.

- base model을 로딩한다.
- 요청을 엔진 큐에 넣는다.
- batching, KV-cache, sampling은 동일한 런타임이 처리한다.

즉, VLM이라고 해서 별도 엔진이 있는 것이 아니라, 같은 vLLM 엔진 위에서 **멀티모달 입력 전처리**가 추가된다고 보는 편이 맞다.

### 3.2 다른 점: `multi_modal_data`

vLLM 멀티모달 입력 문서 기준 offline inference에서는 입력이 아래 구조를 따른다.

- `prompt`: Hugging Face 모델 문서가 요구하는 형식의 텍스트 프롬프트
- `multi_modal_data`: 이미지/비디오/오디오 등의 실제 입력 payload

예를 들어 이미지 VLM이면 `multi_modal_data={"image": image}` 같은 구조가 된다. 여러 이미지를 한 번에 넣으려면 `limit_mm_per_prompt`를 설정하고 이미지 리스트를 넘긴다.

즉, VLM 실행 메커니즘은 아래처럼 이해하면 된다.

1. 텍스트 prompt를 모델 포맷에 맞춰 구성한다.
2. 이미지/비디오/오디오를 `multi_modal_data`에 넣는다.
3. 모델의 processor가 이를 embedding/token 형태로 변환한다.
4. 이후는 일반 LLM decode와 같은 런타임 경로로 들어간다.

### 3.3 멀티모달에서 주의할 점

- 프롬프트 포맷은 모델별로 다르다. `Qwen-VL`, `LLaVA`, `Phi-vision`은 placeholder 형식이 다를 수 있다.
- 여러 이미지 입력은 지원 모델과 설정이 맞아야 한다.
- 일부 멀티모달 LoRA는 language backbone만 바로 붙일 수 있고, vision tower/connector까지 건드린 경우는 merge가 더 안전하다.

## 4. `vLLM`은 학습 프레임워크인가?

엄밀히 말하면 **주력 역할은 아니다**.

`vLLM` 문서에는 RLHF/TRL 연계 문서도 있지만, 공식 홈페이지와 주요 가이드는 여전히 inference/serving을 중심으로 구성되어 있다. 따라서 SFT/LoRA/QLoRA 관점에서는 아래처럼 보는 편이 실무적으로 맞다.

- `vLLM`: 추론, 배포, throughput 최적화
- `Transformers/TRL/PEFT`: 학습
- `Unsloth`: 위 학습 스택을 저 VRAM 환경에 맞게 가속/단순화한 프레임워크

위 구분은 공식 문서 구조를 바탕으로 한 실무적 해석이다.

## 5. 파인튜닝 프레임워크는 무엇을 쓰는가

### 5.1 기본 축: Hugging Face 스택

가장 표준적인 조합은 아래다.

- `transformers`: base model/processor/tokenizer 로딩
- `datasets`: 학습 데이터 로딩/전처리
- `peft`: LoRA/adapter 계열
- `trl`: `SFTTrainer`, DPO/GRPO 등 trainer 계열
- `bitsandbytes`: 4-bit/8-bit quantization, QLoRA
- `accelerate` 또는 `deepspeed/fsdp`: 분산 학습 확장

이 조합의 장점은 가장 범용적이고, 모델 지원과 생태계 문서가 풍부하다는 점이다.

### 5.2 `PEFT`와 `LoRA`

Hugging Face PEFT 문서는 LoRA를 "원래 weight는 freeze하고, 작은 저랭크 update matrix만 학습하는 방식"으로 설명한다.

장점은 명확하다.

- trainable parameter 수가 줄어든다.
- VRAM 부담이 작다.
- base model 하나에 task별 adapter를 여러 개 둘 수 있다.
- 필요하면 merge해서 standalone 모델처럼 쓸 수 있다.

즉, 작은 GPU 환경이면 full fine-tuning보다 LoRA가 보통 첫 선택이다.

### 5.3 `QLoRA`

Transformers/bitsandbytes 문서는 QLoRA를 "4-bit quantization + trainable LoRA weights" 방식으로 설명한다.

실무 의미는 아래다.

- base model은 4-bit로 줄여 메모리를 아낀다.
- 학습 가능한 건 LoRA adapter만 유지한다.
- 그래서 작은 GPU에서도 7B급, 경우에 따라 더 큰 모델까지 만질 수 있다.

작은 GPU에서 첫 실험은 보통 `LoRA`보다 `QLoRA`가 더 현실적이다.

### 5.4 `TRL`

TRL의 `SFTTrainer`는 텍스트 LLM뿐 아니라 VLM도 지원한다. 공식 문서 기준 VLM 학습은 `image` 또는 `images` 컬럼이 포함된 데이터셋을 넣으면 된다.

즉, supervised fine-tuning 기준 기본 뼈대는 이미 `TRL` 쪽에 있다.

## 6. `Unsloth`는 좋은가

### 짧은 대답

작은 GPU에서 open model을 빠르게 파인튜닝하고 싶다면 **좋은 선택**이다. 다만 "만능 기본값"으로 보기보다, **Hugging Face 호환 가속 레이어**로 이해하는 것이 가장 정확하다.

### 좋은 이유

공식 문서와 Hugging Face TRL 연동 문서 기준 `Unsloth`는 아래 강점이 있다.

- Hugging Face 호환 워크플로를 유지한다.
- `SFTTrainer`와 호환된다.
- LoRA, QLoRA, full fine-tuning, VLM, RLHF까지 폭넓게 다룬다.
- 문서가 `vLLM`, `llama.cpp`, `Ollama` 배포까지 바로 연결한다.
- 공식 설명상 학습 속도와 VRAM 효율을 크게 끌어올리는 방향으로 설계되어 있다.

### 언제 특히 잘 맞는가

- single GPU 또는 소수 GPU
- 3B/7B/8B급 open model 실험
- LoRA/QLoRA 위주 실험
- 빠른 시행착오가 중요한 초기 연구 단계
- 학습 후 곧바로 `vLLM`에 붙여 서빙하고 싶은 경우

### 한계와 주의점

이 부분은 공식 문서 기반 정보와 실무 판단을 합친 내용이다.

- 최신 커스텀 커널과 모델별 지원 상태를 확인해야 한다.
- 아주 보수적인 재현성/장기 유지보수 관점에서는 여전히 순정 `Transformers + TRL + PEFT` baseline을 같이 유지하는 편이 안전하다.
- 대규모 multi-node 학습이나 아주 세밀한 분산 최적화가 중심이라면, 처음부터 `Accelerate/DeepSpeed/FSDP` 중심 파이프라인을 잡는 편이 더 단순할 수 있다.
- 멀티모달 모델에서 vision tower/connector까지 적극적으로 튜닝했다면 export 방식과 `vLLM` 적용 경로를 사전에 확인해야 한다.

내 판단은 아래와 같다.

- **작은 GPU에서 첫 번째 현실적 선택지로는 좋다.**
- 다만 production 학습 파이프라인의 최종 표준을 곧바로 이것 하나로 고정하기보다, **baseline은 Hugging Face 스택으로 재현 가능하게 유지**하는 편이 좋다.

## 7. 학습 데이터는 어떻게 구하는가

### 7.1 데이터 원천

실무에서는 보통 4종류를 섞는다.

1. 공개 데이터셋
- Hugging Face Hub에서 instruction/chat/VLM 데이터셋을 가져온다.
- 초기 warm-up이나 포맷 검증에 좋다.

2. 사내/자체 로그
- 실제 사용자 질문, 실제 화면 캡처, 실제 실패 케이스가 가장 중요하다.
- 공개 데이터셋보다 도메인 적합성이 높다.

3. 전문가 작성 데이터
- 정답 품질은 높지만 비용이 많이 든다.
- eval set과 hard case set에 특히 유용하다.

4. teacher model 기반 synthetic data
- 부족한 case를 빠르게 메울 수 있다.
- 단, teacher 편향과 환각을 사람이 샘플 검사해야 한다.

### 7.2 데이터 포맷

Hugging Face Datasets 문서 기준 텍스트는 `csv/jsonl/parquet`, 이미지는 `imagefolder`와 `metadata.csv/jsonl/parquet`로 만들 수 있다.

예를 들어 VLM 데이터는 아래 2가지 방식이 실무적이다.

#### 방식 A: `imagefolder + metadata`

```text
dataset/
  train/
    0001.jpg
    0002.jpg
    metadata.csv
```

```csv
file_name,instruction,answer
0001.jpg,Login button을 찾아라,"{""x"": 812, ""y"": 516, ""label"": ""login_button""}"
0002.jpg,현재 화면 상태를 요약하라,"{""state"": ""rcs_main"", ""active_tab"": ""view""}"
```

#### 방식 B: conversation JSONL

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "이 화면에서 Login 버튼 위치를 JSON으로 반환해라."},
        {"type": "image", "image": "images/0001.jpg"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{\"target\":\"login_button\",\"x\":812,\"y\":516}"}
      ]
    }
  ]
}
```

둘 다 가능하지만, GUI/VLM 파이프라인에서는 **conversation format + strict JSON answer**가 나중에 trainer/평가/배포와 잘 맞는다.

### 7.3 이 저장소 도메인에 맞는 데이터 추천

이 저장소가 다루는 것은 CD-SEM/VeritySEM recipe setup과 GUI automation이므로, 공개 general VLM 데이터보다 아래 컬럼이 있는 자체 데이터가 훨씬 중요하다.

- `image_path`
- `screen_id`
- `workflow_step`
- `instruction`
- `ocr_text`
- `expected_json`
- `bbox` 또는 click point
- `human_verified`
- `failure_type`

특히 아래 식의 정답 포맷이 중요하다.

- 화면 요약 JSON
- target element bbox/click point JSON
- action sequence JSON
- reason/error class JSON

### 7.4 데이터 수집 원칙

- 운영 화면과 최대한 같은 캡처 해상도를 유지한다.
- 비슷한 화면만 과도하게 모으지 않는다.
- 개인정보/사내 민감정보는 먼저 마스킹한다.
- train/valid/test는 화면 패밀리나 workflow 기준으로 분리한다.
- 같은 스크린샷의 augmentation 변형본이 test로 새지 않게 한다.

## 8. 어떻게 학습할 것인가

### 8.1 가장 현실적인 시작점

작은 GPU 환경이면 아래 순서가 좋다.

1. base instruct model 선택
2. LoRA 또는 QLoRA SFT
3. holdout eval
4. 실패 케이스 보강
5. 필요하면 preference/RL 단계 추가

처음부터 full fine-tuning이나 RLHF로 가면 비용 대비 신호가 약한 경우가 많다.

### 8.2 모델 선택

실무적으로는 아래 원칙이 안전하다.

- text-only task면 text instruct model
- image grounding/screen reasoning이 중요하면 VLM
- 작은 GPU면 너무 큰 base보다 작은 instruct checkpoint로 먼저 성공 경로를 확인

GUI 자동화라면 대개 아래 2개 축을 따로 보는 편이 좋다.

- text LLM: planning, JSON repair, OCR post-processing
- VLM: screen understanding, element localization, state detection

### 8.3 `Unsloth`로 LLM SFT 하는 기본 흐름

아래는 개념 예시다.

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=2048,
    dtype="auto",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

dataset = load_dataset("json", data_files="train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="outputs/llm_lora",
        max_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
    ),
    train_dataset=dataset,
)
trainer.train()
```

핵심은 아래다.

- 작은 GPU면 `load_in_4bit=True`로 시작
- 처음엔 `r=16` 또는 `r=32` LoRA부터
- epoch를 과하게 늘리지 말고 eval로 조절
- 출력 포맷이 중요한 task면 system/instruction을 엄격하게 고정

### 8.4 `Unsloth`로 VLM SFT 하는 기본 흐름

VLM은 `TRL` 문서 기준 `image` 또는 `images` 컬럼을 가진 데이터셋을 넣을 수 있고, `Unsloth`는 vision data collator를 제공한다.

실무 절차는 아래와 같다.

1. 이미지 + instruction + answer를 conversation 형식으로 준비
2. `FastVisionModel` 계열로 모델 로딩
3. 필요한 layer에 LoRA 적용
4. vision data collator로 SFT
5. evaluation set에서 grounding/state accuracy 확인

중요한 점은 VLM에서 sequence truncation이 image token을 잘라버릴 수 있으므로, `TRL` 문서가 권고하듯 길이 설정을 조심해야 한다는 점이다.

### 8.5 어느 정도 데이터가 필요하나

정답은 task와 품질에 따라 달라진다. 다만 경험적으로는 아래 원칙이 더 중요하다.

- 100만 건 generic 데이터보다 5천 건의 high-quality domain data가 더 낫다.
- GUI task는 "어려운 실패 케이스"가 일반 케이스보다 훨씬 중요하다.
- 데이터 양보다 정답 consistency가 중요하다.

따라서 처음에는 아래를 목표로 잡는 것이 현실적이다.

- text SFT: 수천 ~ 수만 건의 고품질 instruction-response
- GUI VLM: 수천 건 수준이라도 hard case와 strict answer schema를 확보

정확한 규모는 모델 크기와 목표 성능에 따라 달라진다. 이 부분은 일반 원칙 기반의 실무 추정이다.

### 8.6 파인튜닝에 얼마나 많은 자원이 요구되나

이 질문의 핵심은 "모델 크기" 하나만이 아니라 아래 5개가 같이 결정한다는 점이다.

- 모델 파라미터 수 (`3B`, `7B`, `14B`, `32B` 등)
- 학습 방식 (`QLoRA`, `LoRA`, full fine-tuning, `DPO`, `RLHF/GRPO`)
- context length
- batch size / gradient accumulation
- 텍스트 전용인지, VLM인지

#### 8.6.1 가장 중요한 자원은 GPU VRAM

Unsloth 요구사항 문서는 LLM 파인튜닝 VRAM의 **absolute minimum**을 아래처럼 제시한다.

| Model | QLoRA (4-bit) 최소 VRAM | LoRA (16-bit) 최소 VRAM |
| --- | ---: | ---: |
| `3B` | `3.5 GB` | `8 GB` |
| `7B` | `5 GB` | `19 GB` |
| `8B` | `6 GB` | `22 GB` |
| `14B` | `8.5 GB` | `33 GB` |
| `32B` | `26 GB` | `76 GB` |
| `70B` | `41 GB` | `164 GB` |

이 표는 분명한 방향을 준다.

- 작은 GPU면 거의 항상 `QLoRA`가 출발점이다.
- `7B/8B`를 `LoRA 16-bit`로 바로 가려면 consumer GPU 한 장으로는 빡빡해질 수 있다.
- `14B+`부터는 `QLoRA`와 `LoRA`의 진입 장벽 차이가 매우 커진다.

다만 Unsloth 문서도 명시하듯 이 수치는 **절대 최소값**이다. 실제로는 아래 이유로 더 여유가 필요하다.

- batch size를 조금만 높여도 OOM이 날 수 있다.
- evaluation, checkpoint 저장, dataloader worker, 긴 context가 추가된다.
- 모델 구현에 따라 layer별 메모리 패턴이 다르다.

따라서 실무적으로는 **표의 최소치 + 20~50% 여유**를 두고 잡는 편이 안전하다.

#### 8.6.2 왜 모델 크기보다 더 많은 메모리가 드나

Transformers의 model memory anatomy 문서는 학습 메모리가 단순히 "모델 weight 크기"로 끝나지 않는다고 설명한다. GPU에는 아래가 함께 올라간다.

- model weights
- optimizer states
- gradients
- forward activations
- temporary buffers

같은 문서는 mixed precision training에서 전형적으로 **parameter당 18 bytes + activation memory**가 들 수 있다고 설명한다. 즉, "7B 모델이니까 7B weight 크기만큼만 있으면 된다"라는 식으로 계산하면 거의 항상 과소추정이 된다.

그리고 activation memory는 특히 아래 변수에 민감하다.

- sequence length
- hidden size
- batch size

실무적으로 OOM을 줄이는 첫 번째 레버는 보통 아래 순서다.

1. `per_device_train_batch_size`를 `1`로 낮춘다.
2. `gradient_accumulation_steps`로 유효 batch를 맞춘다.
3. context length를 줄인다.
4. `QLoRA` 또는 8-bit optimizer를 쓴다.

#### 8.6.3 context length가 길어지면 요구 자원이 급격히 커진다

Unsloth 문서는 batch size가 높을수록 OOM이 쉬워진다고 명시하고, 별도 benchmark 문서에서는 같은 `Llama 3.1 8B` `QLoRA`에서도 context 길이에 따라 가능한 설정이 크게 달라진다고 보여준다.

예를 들어 Unsloth benchmark 기준, `Llama 3.1 8B` `QLoRA`에서 batch size `1`일 때 대략 아래 수준의 장기 context가 가능했다.

- `8 GB`: 약 `2,972` tokens
- `12 GB`: 약 `21,848` tokens
- `16 GB`: 약 `40,724` tokens
- `24 GB`: 약 `78,475` tokens

이 수치는 특정 benchmark 조건에서의 결과지만, 방향성은 명확하다.

- 같은 모델이라도 context를 2배, 4배로 늘리면 필요 VRAM이 급격히 증가한다.
- "짧은 instruction tuning은 되는데 긴 document/UI history tuning은 안 된다" 같은 일이 흔하다.

즉, `8B QLoRA가 6GB 최소`라는 표만 보고 계획을 세우면 부족하다. **실제 context 목표를 같이 봐야 한다.**

#### 8.6.4 VLM은 text LLM보다 보수적으로 잡아야 한다

TRL 문서는 VLM 학습에서 image token truncation을 피하려면 `max_length=None`을 권고한다. 이는 VLM이 텍스트 외에 image token도 sequence budget을 먹는다는 뜻이다.

추가로 Unsloth의 Gemma 3n 문서는 기본 Colab 노트북이 text layer 미세조정에 맞춰져 있고, vision/audio layer까지 건드리면 **15GB 무료 Colab/Kaggle 범위를 넘는 훨씬 큰 VRAM**이 필요할 수 있다고 설명한다.

실무 해석은 아래가 맞다.

- 같은 `3B/4B/7B` 계열이라도 VLM은 text LLM보다 더 보수적으로 VRAM을 잡아야 한다.
- language backbone만 LoRA 하는 경우와 vision tower/connector까지 튜닝하는 경우의 요구 자원이 크게 다를 수 있다.
- screen/UI 태스크라면 처음에는 `text-only 보조 모델` 또는 `language layer 위주 LoRA`부터 시작하는 것이 안전하다.

#### 8.6.5 `DPO`와 `RLHF/GRPO`는 `SFT`보다 더 비싸다

`SFT`는 보통 가장 시작하기 쉬운 학습 단계다.

- `DPO`는 `chosen/rejected` preference pair를 다뤄야 해서, 같은 모델과 context 기준으로 보면 대체로 `SFT` 이상 자원이 든다.
- `RLHF/GRPO`는 생성(inference)과 학습을 함께 굴리는 성격이 강해서 훨씬 비싸다.

Unsloth RL 가이드는 `GRPO`의 안전한 경험칙으로, `QLoRA 4-bit`에서는 **모델 parameter 수와 비슷한 수준의 GB VRAM**을 잡고, `LoRA 16-bit`는 **최소 4배 이상** VRAM이 필요할 수 있다고 설명한다. 또한 decent result를 위해 **12시간 이상** 학습이 걸릴 수 있다고 안내한다.

즉, 작은 GPU 환경에서는 보통 아래 순서가 현실적이다.

1. `SFT`
2. hard case 보강
3. 필요하면 `DPO`
4. 정말 alignment/reasoning 개선이 필요할 때만 `RLHF/GRPO`

#### 8.6.6 실무적으로 어느 정도 장비를 상정하면 되나

아래는 공식 문서 수치와 일반적인 single-GPU 실험 경험을 섞은 **실무 추정**이다.

- `8~12 GB` GPU: `3B` text `QLoRA`, 짧은 context의 `7B` text `QLoRA` 입문
- `16 GB` GPU: `7B/8B` text `QLoRA`의 가장 현실적인 출발점, 짧은 context의 작은 VLM 실험
- `24 GB` GPU: `8B/14B` text `QLoRA`를 조금 더 여유 있게, eval/longer context도 상대적으로 수월
- `48 GB+` GPU: `27B/32B` `QLoRA`, 더 공격적인 VLM, RL 계열 실험 검토 가능

CPU RAM과 디스크도 무시하면 안 된다. 아래는 역시 실무 추정이다.

- system RAM: single GPU라도 `32 GB`는 있으면 편하고, `14B+` 또는 VLM이면 `64 GB+`가 안전하다.
- SSD free space: base model, dataset cache, checkpoint, merged export까지 감안해 **최소 수십 GB**, 여러 실험을 병행하면 `100 GB+`를 보는 편이 안전하다.

이 문서 목적에 맞춘 보수적 추천은 아래다.

- 첫 실험: `7B/8B` text model + `QLoRA` + `16~24 GB` GPU
- GUI/VLM 첫 실험: 작은 VLM + language layer 위주 LoRA + `24 GB` 전후에서 시작
- `DPO/RLHF`는 `SFT` baseline이 이미 충분히 돌아간 뒤에만 검토

## 9. 학습 후 모델을 어떻게 적용하는가

### 9.1 방법 A: LoRA adapter를 분리해서 적용

장점:

- 파일이 작다.
- task별 adapter를 여러 개 바꿔 끼우기 쉽다.
- 실험 속도가 빠르다.

vLLM 공식 문서는 `--enable-lora`와 `--lora-modules`로 LoRA adapter를 함께 서빙할 수 있다고 설명한다. runtime에서 adapter를 동적으로 load/unload하는 API도 있다.

예:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --enable-lora \
  --lora-modules my-task=/models/my-task-lora
```

이 방식은 여러 task adapter를 운용할 때 특히 좋다.

### 9.2 방법 B: base + adapter를 merge해서 적용

PEFT 문서는 LoRA를 merge하면 별도 adapter loading에서 오는 latency를 줄일 수 있다고 설명한다. Unsloth 문서도 `vLLM`용 저장은 `merged_16bit`를 기본 경로로 안내한다.

예:

```python
model.save_pretrained_merged(
    "exports/my_model_for_vllm",
    tokenizer,
    save_method="merged_16bit",
)
```

그리고 나서 일반 모델처럼:

```bash
vllm serve exports/my_model_for_vllm
```

### 9.3 어느 쪽을 선택할까

- adapter를 자주 바꾸면: **분리 LoRA**
- 단순하고 안정적인 배포가 우선이면: **merged model**
- 멀티모달 tower/connector까지 바꿨으면: **merge 우선 고려**

## 10. `Unsloth -> vLLM` 추천 워크플로

가장 실용적인 end-to-end 경로는 아래다.

1. base model 선정
- text task면 small instruct LLM
- screen task면 supported VLM

2. 데이터 수집
- 실사용 화면/로그/정답 JSON 확보

3. `Unsloth`로 LoRA/QLoRA SFT
- 작은 GPU면 먼저 QLoRA

4. holdout 평가
- exact JSON accuracy
- bbox/click-point error
- workflow completion rate

5. export
- 실험 단계: LoRA adapter만 저장
- 배포 단계: `merged_16bit` 저장

6. `vLLM` 서빙
- text model: 일반 `vllm serve`
- VLM: 모델별 프롬프트 포맷 + `multi_modal_data`

7. 실제 automation 파이프라인 연결
- strict JSON validation
- low-confidence fallback
- OCR sidecar와 조합

## 11. 이 저장소 기준 추천안

현재 저장소 성격을 고려하면 아래 순서가 합리적이다.

### 11.0 현재 인프라 가정

현재 전제는 아래와 같다.

- 집: GPU 없음
- 일반 사무실 환경: GPU 없음
- office server/cloud: 최대 `H200` `4`장까지 할당 가능

이 조건이면 워크플로는 사실상 **remote-first**로 잡는 편이 맞다.

- 로컬 환경에서는 데이터 정리, 프롬프트 설계, 정답 라벨링, 결과 검수, 경량 API 호출 테스트를 한다.
- 실제 SFT/DPO/VLM 파인튜닝과 대규모 eval은 office server/cloud에서 수행한다.
- 로컬에서 돌아가는 파이프라인을 전제로 설계하기보다, `vLLM` 또는 학습 job이 떠 있는 원격 서버를 기준으로 자동화 흐름을 붙이는 편이 낫다.

### 1단계

`poc/work2`에서 이미 있는 화면 캡처, OCR 보강, strict JSON 응답 흐름을 유지하면서 **학습 데이터 수집 파이프라인**을 먼저 만든다.

추천 저장 컬럼:

- 스크린샷 경로
- 프롬프트
- OCR 텍스트
- 모델 응답
- 사람 교정 정답
- 클릭 좌표 또는 bbox
- 성공/실패 라벨

### 2단계

text-only 보조 모델부터 미세조정한다.

- OCR 결과 정리
- malformed JSON repair
- step description -> strict action JSON 변환

이 단계는 데이터가 만들기 쉽고, 현재 환경에서는 **office server 1 GPU부터** 빠르게 성과를 보기 좋다. 처음부터 `H200 x4` 전체를 쓰기보다 아래처럼 가는 편이 단순하다.

- 첫 baseline: `7B/8B` text model + `QLoRA` + single-GPU
- context, batch, eval 규모가 커질 때만 자원을 늘린다.

### 3단계

그 다음 VLM을 미세조정한다.

- login button 찾기
- View/List tab 찾기
- tool screen state 분류
- panel 요약

이 단계부터는 멀티모달 token과 이미지 처리 때문에 자원이 더 커지므로, 필요하면 `2~4 GPU`까지 확장한다. 특히 아래 조건이면 multi-GPU를 검토할 만하다.

- 더 큰 VLM backbone을 쓰고 싶을 때
- vision tower/connector까지 적극적으로 튜닝할 때
- 긴 context 또는 다중 이미지 입력을 유지하고 싶을 때
- DPO/RL 계열 후속 학습까지 붙일 때

### 4단계

서빙은 `vLLM`로 일원화한다.

- base + LoRA 실험 단계
- 안정화되면 merged 16-bit 모델 배포

현재 환경에서는 서빙도 로컬이 아니라 office server/cloud 쪽에 두는 편이 자연스럽다.

- 로컬/사무실 PC는 `vLLM` 서버의 client 역할만 수행
- 학습 중에는 가능하면 서빙 GPU와 학습 GPU를 분리
- 자원이 넉넉하면 1장은 서빙/eval, 나머지는 학습에 쓰는 방식이 운영상 편하다

## 12. 최종 판단

- `vLLM`은 모델을 **학습시키는 프레임워크**라기보다 **빠르게 돌리는 프레임워크**다.
- 파인튜닝의 기본 축은 `Transformers + TRL + PEFT`이고, 작은 GPU 환경에서는 `Unsloth`가 매우 실용적이다.
- `Unsloth`는 "좋다"라고 말할 수 있지만, 더 정확히는 **저 VRAM LoRA/QLoRA 실험을 빠르게 반복하게 해주는 Hugging Face 호환 가속 프레임워크**라고 보는 것이 맞다.
- 가장 현실적인 경로는 `Unsloth`로 SFT/LoRA를 만들고, 결과를 `vLLM`에 adapter 또는 merged model로 올리는 방식이다.
- GUI/VLM 프로젝트에서 성능을 결정하는 것은 프레임워크 이름보다 **도메인 데이터 품질, strict output schema, hard case 수집**이다.

## 참고 링크

- vLLM 메인 문서: https://docs.vllm.ai/en/stable/
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vLLM offline inference: https://docs.vllm.ai/en/latest/serving/offline_inference/
- vLLM multimodal inputs: https://docs.vllm.ai/en/v0.15.0/features/multimodal_inputs/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
- vLLM LoRA adapters: https://docs.vllm.ai/en/stable/features/lora/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html
- vLLM PagedAttention 논문: https://arxiv.org/abs/2309.06180
- Hugging Face PEFT LoRA guide: https://huggingface.co/docs/peft/main/conceptual_guides/lora
- Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization
- Hugging Face Transformers bitsandbytes/QLoRA: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Hugging Face Transformers model memory anatomy: https://huggingface.co/docs/transformers/model_memory_anatomy
- Hugging Face TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face TRL Unsloth integration: https://huggingface.co/docs/trl/en/unsloth_integration
- Hugging Face Datasets create dataset: https://huggingface.co/docs/datasets/create_dataset
- Hugging Face image dataset guide: https://huggingface.co/docs/datasets/en/image_dataset
- Unsloth docs: https://docs.unsloth.ai/
- Unsloth requirements: https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements
- Unsloth fine-tuning guide: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
- Unsloth benchmarks context lengths and memory usage: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl
- Unsloth vision fine-tuning: https://docs.unsloth.ai/basics/vision-fine-tuning
- Unsloth Gemma 3n fine-tuning: https://docs.unsloth.ai/basics/gemma-3n-fine-tuning
- Unsloth RL guide: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
- Unsloth saving to vLLM: https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-vllm
- Unsloth vLLM deployment guide: https://docs.unsloth.ai/basics/inference-and-deployment/vllm-guide
