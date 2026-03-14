# OmniParser v2 설치 메모

`OmniParser v2`는 현재 `deploy_vlms`의 기존 `vLLM OpenAI server` 흐름과는 분리해서 보는 편이 맞다. 공식 upstream는 `microsoft/OmniParser` GitHub repo clone + 별도 Python 환경 + 로컬 `weights/` 폴더를 기준으로 설치를 안내한다. 즉, `UI-Venus`, `MAI-UI`, `UI-TARS`, `PaddleOCR-VL`처럼 지금 있는 `serve_vlm.py` 체계에 바로 끼워 넣는 모델은 아니다.

핵심 판단:

- 설치 출발점은 `Hugging Face model repo`가 아니라 `GitHub code repo`다.
- Hugging Face에서 모델 파일을 수동 다운로드하는 것은 가능하지만, `모델만` 받아서는 실행에 필요한 코드가 부족하다.
- 수동 다운로드 시 `weights/icon_caption` 폴더명을 반드시 `weights/icon_caption_florence`로 바꿔야 한다.
- 현재 upstream 코드상 caption processor는 `microsoft/Florence-2-base`에서 따로 로드하므로, 완전 오프라인이면 그 processor cache도 미리 준비해야 한다.
- 이 저장소에는 아직 `poc/work2`용 OmniParser wrapper나 공용 API adapter가 없다. 우선 별도 env에서 smoke test를 끝내고, 그 다음 연동하는 순서가 맞다.

## 1. 왜 기존 vLLM 문서와 분리하는가

공식 설치 경로는 아래 순서다.

1. `microsoft/OmniParser` repo clone
2. 별도 Python env 생성
3. `pip install -r requirements.txt`
4. `weights/` 아래에 V2 checkpoint 배치
5. `python gradio_demo.py`

즉, 현재 `deploy_vlms/scripts/serve_vlm.py`처럼 `MODEL_ID`, `PORT`, `SERVED_MODEL_NAME`만 맞춰 `vLLM` OpenAI-compatible server를 띄우는 구조와 출발점이 다르다.

## 2. 필요한 구성요소

| 구성 | 용도 | 비고 |
|------|------|------|
| `microsoft/OmniParser` GitHub repo | 실행 코드 | `gradio_demo.py`, `util/`, `omnitool/` 포함 |
| `microsoft/OmniParser-v2.0` Hugging Face repo | V2 가중치 | `icon_detect`, `icon_caption` 다운로드 |
| `microsoft/Florence-2-base` processor cache | caption processor 로딩 | 현재 upstream 코드가 repo id로 직접 로드 |

실무적으로는 `GitHub repo clone + Hugging Face weight download` 둘 다 필요하다.

## 3. 권장 설치 절차

### 3.1 별도 clone + env 준비

공식 README 기준 설치 명령은 아래다.

```bash
cd /project/day/workSpace/itc-1stop-solution
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser

conda create -n omni python==3.12
conda activate omni
pip install -r requirements.txt
```

운영 메모:

- 현재 upstream `requirements.txt`에는 `torch`, `torchvision`, `transformers`, `easyocr`, `paddleocr`, `gradio`, `ultralytics==8.3.70` 등이 들어 있다.
- `ultralytics`는 버전을 임의로 올리지 말고 upstream pinned value를 그대로 따르는 편이 안전하다.
- 이 저장소의 기본 `.venv`에 섞지 말고 별도 env로 분리하는 편이 맞다.

### 3.2 Hugging Face weight 다운로드

공식 README의 기본 다운로드 방식은 아래다.

```bash
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} \
         icon_caption/{config.json,generation_config.json,model.safetensors}; do
  huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

mv weights/icon_caption weights/icon_caption_florence
```

수동 다운로드를 할 때도 최종 폴더 구조는 아래처럼 맞춰야 한다.

```text
OmniParser/
  weights/
    icon_detect/
      model.pt
      model.yaml
      train_args.yaml
    icon_caption_florence/
      config.json
      generation_config.json
      model.safetensors
```

다운로드 용량 메모:

- `icon_detect/model.pt`: 약 `40.6 MB`
- `icon_caption/model.safetensors`: 약 `1.08 GB`
- V2 checkpoint 총량: 대략 `1.12 GB`

가능하면 각 하위 폴더의 `LICENSE` 파일도 같이 받아 두는 편이 좋다.

### 3.3 완전 오프라인이면 Florence processor cache도 준비

현재 upstream 코드 `util/utils.py`는 caption model 자체는 로컬 `weights/icon_caption_florence`에서 읽지만, processor는 아래처럼 별도 repo id로 로드한다.

```python
AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
```

따라서 해석은 아래와 같다.

- 인터넷이 열려 있으면 첫 실행에서 processor 관련 파일을 자동 cache할 수 있다.
- 완전 오프라인이면 `microsoft/Florence-2-base` processor/tokenizer cache가 없어서 시작 단계에서 막힐 가능성이 높다.

가장 단순한 준비 방법은 인터넷이 되는 환경에서 한 번 아래를 실행해 cache를 미리 만드는 것이다.

```bash
python - <<'PY'
from transformers import AutoProcessor
AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
print("Florence processor cache ready")
PY
```

위 판단은 upstream 코드 호출 방식을 근거로 한 해석이다.

## 4. 최소 smoke test

가중치와 env 준비가 끝났으면 가장 단순한 검증은 Gradio demo다.

```bash
cd /project/day/workSpace/itc-1stop-solution/OmniParser
python gradio_demo.py
```

기본 실행 확인 포인트:

- 첫 로딩은 모델 초기화 때문에 수십 초 걸릴 수 있다.
- upstream `gradio_demo.py`는 `127.0.0.1:7861`에서 띄우는 구성을 사용한다.
- screenshot 1장을 올려서 bounding box overlay와 parsed text가 모두 나오는지 보면 된다.

## 5. 자주 걸리는 포인트

### 5.1 모델만 받고 code repo를 안 받은 경우

`microsoft/OmniParser-v2.0` Hugging Face repo는 weight 중심 repo다. 현재 기준 top-level에 `handler.py`, `requirements.txt`는 있지만, 일반적인 로컬 실행에 필요한 GitHub 쪽 `gradio_demo.py`, `util/`, `omnitool/` 기준 문서와 동일하지 않다. 따라서 설치 시작점은 GitHub clone이 맞다.

### 5.2 `icon_caption_florence` rename 누락

`gradio_demo.py`는 caption model path를 `weights/icon_caption_florence`로 고정해 사용한다. 폴더 rename을 빼먹으면 바로 파일 경로 에러가 난다.

### 5.3 OCR이 CUDA를 다 쓰지 않는 것처럼 보이는 경우

현재 upstream `util/utils.py`에서는 `PaddleOCR(... use_gpu=False)`로 초기화한다. 즉, GPU는 주로 detection/caption 쪽에 쓰고, PaddleOCR은 충돌 회피를 위해 CPU 경로를 타는 구성이 기본이다.

### 5.4 공용 API endpoint가 바로 필요한 경우

upstream repo에는 `omnitool/omniparserserver/omniparserserver.py`가 있지만, 이 경로는 Gradio demo보다 부가 의존성과 운영 정리가 더 필요하다. 현재 `deploy_vlms` 기준 권장 순서는 아래다.

1. 별도 env에서 `gradio_demo.py` smoke test
2. screenshot parsing 품질 확인
3. 그 다음 `poc/work2`용 thin wrapper 또는 별도 API adapter 설계

## 6. 이 저장소 기준 정리

`OmniParser v2`는 지금 바로 `deploy_vlms/scripts/serve_vlm.py`에 넣어 배포할 대상이라기보다, `별도 clone + 별도 env + 로컬 weights`로 먼저 검증할 모델이다. 현재 repo에는 OmniParser integration 코드가 커밋되어 있지 않으므로, 우선 upstream smoke test를 성공시킨 뒤 `poc/work2`에서 screenshot parser helper로 감싸는 순서가 가장 안전하다.

## 7. 참고 source

- GitHub README: <https://github.com/microsoft/OmniParser>
- GitHub requirements: <https://github.com/microsoft/OmniParser/blob/master/requirements.txt>
- GitHub demo entrypoint: <https://github.com/microsoft/OmniParser/blob/master/gradio_demo.py>
- GitHub utils loader: <https://github.com/microsoft/OmniParser/blob/master/util/utils.py>
- Hugging Face model card: <https://huggingface.co/microsoft/OmniParser-v2.0>
- Hugging Face `icon_detect` tree: <https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_detect>
- Hugging Face `icon_caption` tree: <https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_caption>
