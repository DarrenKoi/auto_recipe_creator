# GOT-OCR-2.0-hf 설치 메모

`stepfun-ai/GOT-OCR-2.0-hf`는 현재 `deploy_vlms`의 기존 `vLLM` 배포 흐름과는 다르게 보는 편이 맞다. 공식 Hugging Face 문서는 `transformers` 기반 추론을 안내하고, 2026-03-11 기준으로 확인한 범위에서는 `docs.vllm.ai`에 이 모델의 공식 배포 가이드나 supported-model entry를 찾지 못했다.

핵심 판단:

- 이 모델은 `새 md 파일`로 분리하는 것이 맞다.
- 현재 확인한 클라우드 런타임 `Python 3.11 + transformers 4.57.6 + torch 2.10.0`이면 별도 대규모 업그레이드 없이 바로 테스트할 수 있다.
- `config/models/got-ocr-2.0-hf.env`에 `GPU_ID`, `DEVICE=cuda`, `TORCH_DTYPE=bfloat16`를 넣고 GPU 실행 경로를 고정해 둘 수 있다.
- 다만 `vLLM` supported models에는 이 모델을 찾지 못했으므로, 현재 기준 배포 방식은 `transformers` 직접 추론이다.
- repo 기본 `.venv`에는 OCR 런타임 의존성을 넣지 않는 쪽이 여전히 낫지만, 클라우드 전용 Python 환경은 이미 준비된 상태로 볼 수 있다.

## 1. 왜 기존 vLLM 문서와 분리하는가

공식 문서 기준 사용 방식은 아래다.

- `AutoProcessor.from_pretrained(...)`
- `AutoModelForImageTextToText.from_pretrained(...)`

즉, 현재 `UI-Venus`, `MAI-UI`, `UI-TARS`처럼 `vLLM OpenAI server`부터 띄우는 문서 구조와 출발점이 다르다.

반대로 `PaddleOCR-VL-1.5`는 공식 문서가 `vLLM` 경로를 함께 안내하므로 기존 체계에 편입하기 쉬웠지만, `GOT-OCR-2.0-hf`는 지금 기준으로는 그렇지 않다.

## 2. 현재 작업 환경에 맞춘 권장 런타임

현재 확인한 클라우드 환경은 아래다.

- Python `3.11`
- `transformers 4.57.6`
- `torch 2.10.0`
- `vLLM 0.17.0`

이 조합에서의 해석은 아래와 같다.

| 항목 | 현재 상태 | 판단 |
|------|-----------|------|
| Python | `3.11` | 적합 |
| `transformers` | `4.57.6` | 적합. `GOT-OCR2` 문서가 이미 `v4.57.0`에 존재 |
| `torch` | `2.10.0` | 적합 |
| `vLLM` | `0.17.0` | 설치는 되어 있지만 공식 supported model로는 확인되지 않음 |

따라서 지금은 `추가 설치보다 smoke test와 wrapper 분리`가 우선이다.

별도 env가 필요한 경우는 아래 둘 중 하나다.

- 클라우드 공용 런타임을 건드리고 싶지 않을 때
- `accelerate`, `pillow`, 후처리 패키지를 모델별로 분리하고 싶을 때

## 3. 현재 클라우드에서 바로 확인하는 절차

클라우드 환경에 이미 `transformers`와 `torch`가 있으므로, 우선 아래처럼 최소 추가 패키지만 확인하면 된다.

```bash
python -m pip install pillow
```

이미 들어 있다면 추가 설치는 생략한다.

그 다음 아래 설정 파일을 먼저 확인한다.

- [got-ocr-2.0-hf.env](./config/models/got-ocr-2.0-hf.env)
- [run_got_ocr.py](./scripts/run_got_ocr.py)

주요 키:

- `MODEL_ID`: 로컬 모델 절대경로
- `IMAGE_PATH`: OCR 대상 이미지 절대경로
- `GPU_ID`: 사용할 GPU 번호
- `DEVICE=cuda`: GPU 실행 강제
- `TORCH_DTYPE=bfloat16`: GPU 기본 dtype

설정 후에는 아래처럼 바로 GPU smoke test를 돌린다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
python scripts/run_got_ocr.py
```

## 4. 별도 env가 필요할 때

공용 클라우드 Python을 건드리지 않으려면 아래처럼 분리한다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
uv venv .venvs/got-ocr-2.0-hf --python python3.11
uv pip install --python .venvs/got-ocr-2.0-hf/bin/python "transformers[torch]" pillow
```

이미 모델을 로컬에 받아 두었다면, 추론 시 Hugging Face repo id 대신 로컬 절대경로를 쓰는 편이 현재 저장소 운영 방식과 맞다.

예:

```text
/ABS/PATH/TO/GOT-OCR-2.0-hf
```

## 5. 최소 smoke test

아래 예시는 별도 env를 만들었을 때도 같은 config 파일을 재사용하는 방식이다.

```bash
uv run --python .venvs/got-ocr-2.0-hf/bin/python - <<'PY'
import runpy
runpy.run_path("scripts/run_got_ocr.py", run_name="__main__")
PY
```

## 6. 팀 공용 endpoint가 필요할 때

이 모델을 팀 공용 API로 노출하고 싶다면, 현재 문서에 있는 `serve_vlm.py`를 바로 재사용하기보다 아래 중 하나를 별도로 잡는 편이 맞다.

- 작은 Flask/FastAPI wrapper 추가
- 별도 추론 worker 프로세스 작성

즉, `GOT-OCR-2.0-hf`는 지금 단계에서는 `로컬 또는 전용 Python runtime`으로 먼저 검증하고, 그 다음에 API 화하는 순서가 적절하다.

## 7. 참고 source

- Hugging Face model card: <https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf>
- Transformers GOT-OCR2 docs: <https://huggingface.co/docs/transformers/v4.57.0/model_doc/got_ocr2>
- Transformers installation: <https://huggingface.co/docs/transformers/main/installation>
- vLLM supported models: <https://docs.vllm.ai/en/latest/models/supported_models.html>
