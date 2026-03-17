# OmniParser v2 설치 메모

`OmniParser v2`는 현재 `deploy_vlms`의 기존 `vLLM OpenAI server` 흐름과는 분리해서 보는 편이 맞다. 이 모델은 `served-model-name + port` 중심의 `vLLM` 배포 대상이라기보다, `별도 Python runtime + 로컬 weights + OCR + caption loader`를 가진 `screen parser` 쪽에 가깝다.

이 문서는 `사내 private cloud 에서 outbound network 가 막힌 Linux 환경`을 기준으로 다시 정리한다. 핵심은 단순하다.

- private cloud 안에서는 Hugging Face나 GitHub로 나가지 않는다고 가정한다.
- 외부에서 `GitHub code repo`와 `Hugging Face model/processor/custom-code`를 모두 미리 내려받아 반입한다.
- `OmniParser-v2.0`만 받으면 끝이 아니다. 기본 V2 weight 외에 `Florence-2-base` 관련 processor/tokenizer 파일이 추가로 필요하다.
- 현재 upstream loader와 `OmniParser-v2.0/icon_caption/config.json`을 기준으로 보면, `Florence-2-base-ft`의 custom code 파일도 오프라인에서 같이 준비하는 편이 안전하다.
- 반대로 `Florence-2-large`, `Florence-2-large-ft`는 기본 `OmniParser v2` 경로에는 필요 없다.
- 이 저장소 기준 권장 설치 방식은 `같은 서버 안의 별도 uv venv`다. 기존 `vLLM` runtime과 섞어 올리기보다 분리하는 편이 운영 리스크가 낮다.

## 1. OmniParser v2가 실제로 하는 일

`OmniParser`는 스크린샷을 바로 agent가 쓰기 쉬운 `구조화된 UI 요소 목록`으로 바꾸는 parser다. 논문과 model card 기준 핵심 구성은 아래 3개다.

- `icon_detect`: 클릭 가능한 영역과 상호작용 후보를 찾는 detection model
- `OCR`: 화면에 이미 있는 text를 읽는 경로
- `icon_caption`: 탐지된 아이콘이나 UI 요소의 기능을 짧은 텍스트로 설명하는 caption model

즉, `Florence`는 여기서 `전체 화면을 다시 이해하는 범용 VLM` 역할이 아니라, `detected UI element의 기능 설명`을 붙이는 caption backbone 역할이다.

실무적으로는 아래처럼 이해하면 된다.

- `YOLO 계열 detection`이 box를 만든다.
- `OCR`이 text를 읽는다.
- `Florence 기반 caption model`이 `search`, `settings`, `trash`, `attachment`, `submit` 같은 아이콘 기능 설명을 만든다.
- 이 정보를 합쳐 downstream agent가 `어디를 눌러야 하는지`를 더 안정적으로 판단한다.

## 2. Florence는 무엇을 내려받아야 하나

질문에 대한 짧은 답부터 적으면 아래와 같다.

| 항목 | 기본 OmniParser v2에 필요한가 | 이유 |
|------|-------------------------------|------|
| `microsoft/OmniParser-v2.0` | 예 | 실제 V2 icon detect / icon caption weight |
| `microsoft/Florence-2-base` | 예 | upstream code가 processor를 이 repo id에서 직접 로드 |
| `microsoft/Florence-2-base-ft` | 부분적으로 예 | caption model config의 `auto_map`이 이 repo의 custom code를 참조 |
| `microsoft/Florence-2-large` | 아니오 | upstream 기본 경로에서 쓰지 않음 |
| `microsoft/Florence-2-large-ft` | 아니오 | upstream 기본 경로에서 쓰지 않음 |

정리:

- `Florence-2-base`는 받아 두는 편이 맞다.
- 다만 `Florence-2-base`의 `모델 weight 전체`가 꼭 필요한 것은 아니다.
- 기본적으로 필요한 것은 `processor/tokenizer/custom code`다.
- `Florence-2-base-ft`도 `전체 weight`까지는 기본 경로에서 불필요하지만, `configuration_florence2.py`, `modeling_florence2.py` 같은 custom code 파일은 오프라인에서 필요해질 수 있다.

위 판단은 `OmniParser` upstream loader가 `AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)`를 호출하는 점과, `OmniParser-v2.0/icon_caption/config.json`의 `auto_map`가 `microsoft/Florence-2-base-ft`를 가리키는 점을 근거로 한 해석이다.

## 3. 왜 Hugging Face 준비가 `OmniParser-v2.0` 하나로 끝나지 않나

공식 README는 아래 정도만 적고 있다.

- GitHub repo clone
- `python==3.12`
- `requirements.txt` 설치
- `OmniParser-v2.0` weight 다운로드
- `icon_caption -> icon_caption_florence` rename

하지만 private cloud 오프라인 관점에서는 이 설명만으로는 부족하다.

부족한 이유:

- upstream code는 caption processor를 `microsoft/Florence-2-base`에서 직접 로드한다.
- `trust_remote_code=True`를 사용하므로, processor/model custom code 파일도 Hub에서 찾는다.
- `icon_caption_florence/config.json`은 모델 custom code를 `microsoft/Florence-2-base-ft` 쪽에 매핑한다.

즉, 인터넷이 되는 환경에서 한 번 실행하면 자동 cache로 넘어갈 수도 있지만, `private cloud 안에서는 첫 실행 자동 다운로드`에 기대면 안 된다.

## 4. 현재 환경 기준 권장 설치 전략

이 저장소의 다른 문서와 현재 클라우드 기준을 반영하면, 권장 경로는 아래다.

- runtime은 `vLLM`과 분리한다.
- cloud 안에서는 `Python 3.11` 기준 별도 `uv venv`를 먼저 시도한다.
- upstream README는 `Python 3.12`를 적고 있으므로, `3.11`에서 dependency/runtime 문제가 나면 그때만 `3.12` venv로 올린다.
- 모델/processor/custom code는 외부 연결이 되는 staging machine에서 모두 준비한 뒤 반입한다.
- private cloud에서는 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 등을 켠다.

`Python 3.11` 우선 권장은 `이 저장소의 현재 클라우드 문서 기준`에 맞춘 운영 판단이다. upstream이 `3.12`를 명시하므로, strict하게 upstream 재현이 더 중요하면 처음부터 `3.12` venv를 잡아도 된다.

## 5. 권장 디렉터리 예시

예시는 아래처럼 두는 편이 관리가 쉽다.

```text
/project/day/workSpace/itc-1stop-solution/
├── OmniParser/
│   ├── gradio_demo.py
│   ├── requirements.txt
│   ├── util/
│   └── weights/
│       ├── icon_detect/
│       └── icon_caption_florence/
└── itc-1stop-solution-gpu-image/
    └── data/
        └── models/
            └── omniparser-v2/
                ├── hf_stage/
                │   ├── OmniParser-v2.0/
                │   ├── Florence-2-base/
                │   └── Florence-2-base-ft/
                └── hf-cache/
```

핵심은 아래 2개다.

- 실행 code는 `OmniParser/`
- Hugging Face artifact와 cache는 `/project/.../data/models/omniparser-v2/` 아래

## 6. staging machine에서 미리 준비할 것

### 6.1 GitHub code repo

`OmniParser` local 실행에 필요한 `gradio_demo.py`, `util/`, `omnitool/` 기준 코드는 여전히 GitHub upstream에 있다. 즉, `Hugging Face repo`만으로는 stock local runtime을 대체하지 못한다.

따라서 인터넷이 되는 외부 환경에서 먼저 아래를 준비한다.

```bash
cd /tmp
git clone https://github.com/microsoft/OmniParser.git
cd /tmp/OmniParser
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

위 `venv`는 `HF cache prewarm` 용 임시 runtime으로만 써도 충분하다. 필요하면 이 디렉터리를 tarball로 묶어서 private cloud로 반입하면 된다.

### 6.2 V2 weight

공식 README가 안내하는 기본 weight는 아래다.

```bash
mkdir -p /tmp/omniparser-stage/hf/OmniParser-v2.0

for f in \
  icon_detect/train_args.yaml \
  icon_detect/model.pt \
  icon_detect/model.yaml \
  icon_detect/LICENSE \
  icon_caption/config.json \
  icon_caption/generation_config.json \
  icon_caption/model.safetensors \
  icon_caption/LICENSE
do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" \
    --local-dir /tmp/omniparser-stage/hf/OmniParser-v2.0
done
```

이후 private cloud나 staging area에서 최종 폴더명을 아래처럼 맞춘다.

```text
weights/
  icon_detect/
  icon_caption_florence/
```

즉, `icon_caption` 폴더는 반드시 `icon_caption_florence`로 rename 해야 한다.

### 6.3 Florence-2-base processor/tokenizer/custom code

`Florence-2-base`에서는 아래 파일을 미리 받아 두는 편이 안전하다.

```bash
mkdir -p /tmp/omniparser-stage/hf/Florence-2-base

for f in \
  config.json \
  configuration_florence2.py \
  preprocessor_config.json \
  processing_florence2.py \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  LICENSE
do
  huggingface-cli download microsoft/Florence-2-base "$f" \
    --local-dir /tmp/omniparser-stage/hf/Florence-2-base
done
```

주의:

- `Florence-2-base` 전체 snapshot은 약 `930 MB` 수준이다.
- 하지만 OmniParser v2 기본 경로에서는 `Florence-2-base`의 full model weight가 직접 필요하지는 않다.
- 오프라인 목적이면 위 processor/tokenizer/custom code 위주로 먼저 준비하는 편이 더 경제적이다.

### 6.4 Florence-2-base-ft custom code

`OmniParser-v2.0/icon_caption/config.json`은 모델 custom code를 `microsoft/Florence-2-base-ft` 쪽에 매핑한다. 따라서 최소한 아래 파일은 같이 받아 두는 편이 안전하다.

```bash
mkdir -p /tmp/omniparser-stage/hf/Florence-2-base-ft

for f in \
  configuration_florence2.py \
  modeling_florence2.py \
  LICENSE
do
  huggingface-cli download microsoft/Florence-2-base-ft "$f" \
    --local-dir /tmp/omniparser-stage/hf/Florence-2-base-ft
done
```

중요:

- `Florence-2-base-ft`의 `full model weight`까지 받을 필요는 기본 경로에서는 없다.
- 필요한 것은 custom code 파일 쪽이다.
- 즉, `Florence-2-large*`는 물론이고 `Florence-2-base-ft` full snapshot도 기본 smoke test 목적에는 과한 편이다.

## 7. 가장 안전한 오프라인 반입 방식: HF cache까지 같이 준비

private cloud에서 code patch 없이 upstream를 그대로 돌릴 생각이면, `파일만 로컬 폴더에 두는 것`보다 `Hugging Face cache까지 미리 채워서 반입하는 방식`이 가장 덜 헷갈린다.

이유:

- upstream code가 repo id 문자열을 직접 사용한다.
- `trust_remote_code=True`라서 dynamic module cache도 같이 필요하다.
- cache가 비어 있으면 offline mode에서도 시작 단계에서 막힐 수 있다.

인터넷 되는 staging machine에서 아래처럼 한 번 cache를 만들어 두는 방식이 가장 단순하다.

```bash
cd /tmp/OmniParser
mkdir -p weights
cp -R /tmp/omniparser-stage/hf/OmniParser-v2.0/icon_detect weights/
cp -R /tmp/omniparser-stage/hf/OmniParser-v2.0/icon_caption weights/icon_caption_florence

HF_HOME=/tmp/omniparser-stage/hf-cache \
uv run --python .venv/bin/python - <<'PY'
from transformers import AutoProcessor, AutoModelForCausalLM

AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(
    "weights/icon_caption_florence",
    trust_remote_code=True,
)
print("HF cache prepared")
PY
```

위 스크립트가 성공하면 아래 성격의 hidden dependency가 cache에 같이 채워진다.

- `Florence-2-base` processor/tokenizer/custom code
- `Florence-2-base-ft` custom code

그 다음 아래 둘을 같이 반입하면 된다.

- `OmniParser/` source tree
- `/tmp/omniparser-stage/hf-cache/`

## 8. private cloud에서 설치

### 8.1 소스와 staged artifact 배치

예:

```bash
cd /project/day/workSpace/itc-1stop-solution
tar xf OmniParser.tar

mkdir -p /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/omniparser-v2
tar xf omniparser-hf-cache.tar -C /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/omniparser-v2
```

weight를 수동으로 옮길 때는 최종 구조를 아래처럼 맞춘다.

```text
/project/day/workSpace/itc-1stop-solution/OmniParser/weights/
  icon_detect/
    model.pt
    model.yaml
    train_args.yaml
  icon_caption_florence/
    config.json
    generation_config.json
    model.safetensors
```

### 8.2 dedicated uv venv

private cloud에서 기존 `vLLM` runtime과 분리하려면 아래처럼 두는 편이 가장 단순하다.

```bash
cd /project/day/workSpace/itc-1stop-solution/OmniParser
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

만약 `3.11`에서 dependency/runtime 문제가 나면, 같은 방식으로 `3.12` venv를 다시 만드는 편이 낫다.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

### 8.3 offline env

실행 전에 아래 값을 켠다.

```bash
export HF_HOME=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/omniparser-v2/hf-cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN
```

이 문서는 `OmniParser` 전용 문서이므로 `vLLM` 관련 env는 여기서 따로 다루지 않는다.

## 9. 최소 smoke test

가장 단순한 확인은 gradio demo다.

```bash
cd /project/day/workSpace/itc-1stop-solution/OmniParser
uv run --python .venv/bin/python gradio_demo.py
```

확인 포인트:

- startup 중 Hub download 시도가 없어야 한다.
- 첫 로딩은 수십 초 걸릴 수 있다.
- screenshot 1장을 올렸을 때 detection box, OCR text, icon caption이 모두 나와야 한다.

## 10. 자주 걸리는 포인트

### 10.1 `OmniParser-v2.0`만 반입한 경우

가장 흔한 오해다. V2 weight만 있으면 충분해 보이지만, 실제 loader는 `Florence-2-base` processor와 `Florence-2-base-ft` custom code에 간접 의존한다.

### 10.2 `icon_caption_florence` rename 누락

README와 동일하게, 폴더명은 `icon_caption_florence`여야 한다.

### 10.3 full Florence model까지 다 받아야 한다고 생각하는 경우

기본 `OmniParser v2` 목적만 보면 아니다.

- `Florence-2-base`: processor/tokenizer/custom code가 핵심
- `Florence-2-base-ft`: custom code가 핵심
- `Florence-2-large*`: 기본 경로에서는 불필요

즉, `다른 Florence 모델`을 추가로 받을 필요는 없다.

### 10.4 OCR이 GPU를 안 쓰는 것처럼 보이는 경우

현재 upstream code에는 `PaddleOCR(... use_gpu=False)` 초기화가 보인다. 따라서 GPU는 주로 detection/caption 쪽에서 보이고, OCR은 CPU처럼 보일 수 있다.

## 11. 한 줄 결론

`OmniParser v2`를 private cloud에서 안정적으로 올리려면, `microsoft/OmniParser-v2.0`만 받지 말고 `Florence-2-base` processor/tokenizer/custom code와 `Florence-2-base-ft` custom code까지 외부에서 미리 준비해 반입하는 편이 맞다. 기본 경로에서는 `Florence-2-large`나 다른 Florence 변형은 받을 필요가 없다.

## 12. 참고 source

- GitHub README: <https://github.com/microsoft/OmniParser>
- GitHub utils loader: <https://github.com/microsoft/OmniParser/blob/master/util/utils.py>
- Hugging Face model card: <https://huggingface.co/microsoft/OmniParser-v2.0>
- Florence-2-base model card: <https://huggingface.co/microsoft/Florence-2-base>
- Florence-2-base file tree: <https://huggingface.co/microsoft/Florence-2-base/tree/main>
- Florence-2-base-ft file tree: <https://huggingface.co/microsoft/Florence-2-base-ft/tree/main>
- OmniParser technical report: <https://arxiv.org/abs/2408.00203>
