# PaddleOCR-VL-1.5 배포 메모

`PaddleOCR-VL-1.5`는 현재 `docs/deploy_vlms`에 있는 `vLLM` 운영 체계와 가장 비슷하게 붙일 수 있는 OCR VLM이다. 공식 문서는 `PaddleOCR/PaddlePaddle` 추론 경로와 `vLLM` 경로를 둘 다 안내한다.

핵심 판단:

- 이 저장소 기준 권장 경로는 `기존 GPU 서버 + 기존 vLLM wrapper 재사용`이다.
- 현재 확인한 클라우드 런타임 `Python 3.11 + vLLM 0.17.0 + transformers 4.57.6 + torch 2.10.0`에서는 `vLLM` 경로를 우선 사용하면 된다.
- 반대로 공식 Hugging Face `transformers` 예시는 `transformers>=5.0.0`를 요구하므로, 현재 클라우드의 `transformers 4.57.6`은 이 경로의 권장 기준보다 낮다.
- `paddle`, `paddleocr` 기반 공식 parser 경로는 별도 의존성 설치가 필요하므로, 현재 문서에서는 기존 `vLLM` 재사용을 기본값으로 둔다.

## 1. 공식 설치 방식 요약

공식 문서 기준으로는 아래 2갈래다.

1. `PaddleOCR/PaddlePaddle` 런타임으로 직접 추론
2. `vLLM`으로 OpenAI 호환 서버 구동

`PaddleOCR-VL-1.5` model card도 이 family의 `vLLM` usage guide를 별도 경로로 연결한다. 따라서 `docs/deploy_vlms`와 바로 이어지는 것은 `2번 vLLM 경로`다. 현재 저장소의 [serve_vlm.py](./scripts/serve_vlm.py)는 특정 GUI 모델 전용이 아니라, `MODEL_ID`, `SERVED_MODEL_NAME`, `PORT`, `GPU_ID`를 읽어 일반 `vLLM` 서버를 띄우는 wrapper라서 `PaddleOCR-VL-1.5`에도 그대로 맞는다. 현재 코드에서 별도 특수 처리가 필요한 쪽은 `Qwen2.5-VL` 계열뿐이다.

## 2. 현재 작업 환경에 맞춘 권장 경로

현재 확인한 클라우드 환경:

- OS `Linux`
- Python `3.11`
- `vLLM 0.17.0`
- `transformers 4.57.6`
- `torch 2.10.0`

이 조합에서의 해석은 아래와 같다.

| 경로 | 현재 클라우드 적합성 | 판단 |
|------|----------------------|------|
| `vLLM` 서빙 | 가능 | `vLLM` supported models에 `PaddleOCRVLForConditionalGeneration`가 있다. 현재 `serve_vlm.py` 재사용 가능 |
| `transformers` 직접 추론 | 비권장 | 공식 `PaddleOCR-VL-1.5` card는 `transformers>=5.0.0`를 명시 |
| `paddleocr[doc-parser]` 공식 parser | 별도 준비 필요 | `paddlepaddle>=3.2.1`, `paddleocr[doc-parser]`, special `safetensors`가 추가로 필요 |

따라서 `PaddleOCR-VL-1.5`는 현재 Linux 클라우드에서 아래처럼 가져가는 것이 가장 단순하다.

- Linux GPU 서버: 기존 `vLLM` wrapper로 실제 운영
- 별도 문서 parser가 정말 필요할 때만 `paddleocr` 전용 env 추가

## 3. 이 저장소에서 바로 쓰는 배포 절차

### 3.1 기본 model env 확인

이 문서와 함께 아래 기본 model env와 시작 스크립트를 둔다.

- [paddleocr-vl-1.5.env](./config/models/paddleocr-vl-1.5.env)
- [start_paddleocr_vl.py](./scripts/start_paddleocr_vl.py)

Linux 클라우드 서버에서:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
```

그 다음 `config/models/paddleocr-vl-1.5.env`에서 최소 아래만 확인한다.

- `MODEL_ID`: 이미 내려받아 둔 로컬 절대경로
- `PORT`: 기본 `8004` canary 포트
- `GPU_ID`: 실제 비어 있는 GPU

중요:

- [serve_vlm.py](./scripts/serve_vlm.py)는 `STRICT_OFFLINE=1`일 때 `MODEL_ID`가 `ALLOWED_MODEL_ROOT` 아래에 있어야만 실행된다.
- 즉, 모델이 `/project/.../data/models/` 밖에 있다면 `MODEL_ID`만 바꾸는 것으로 끝나지 않고 `ALLOWED_MODEL_ROOT`도 같이 조정하거나, 모델을 기존 루트 아래로 옮겨야 한다.

### 3.2 서버 기동

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
python scripts/start_paddleocr_vl.py
```

기동 확인:

```bash
curl http://127.0.0.1:8004/v1/models
python scripts/check_vlm.py http://127.0.0.1:8004 paddleocr-vl-1.5
```

`VLM_MODEL_NAME`은 아래처럼 맞추면 된다.

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8004
VLM_MODEL_NAME=paddleocr-vl-1.5
```

## 4. 정리

`PaddleOCR-VL-1.5`는 "기존 문서와 완전히 다른 모델"은 아니고, `기존 vLLM 문서 체계에 편입 가능한 모델`에 가깝다. 현재 Linux 클라우드에서는 `PaddleOCR/PaddlePaddle` 직접 추론보다 `vLLM` 경로를 먼저 쓰는 편이 단순하다.

## 5. 참고 source

- Hugging Face model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-0.9B>
- PaddleOCR repository: <https://github.com/PaddlePaddle/PaddleOCR>
- Transformers PaddleOCR-VL docs: <https://huggingface.co/docs/transformers/v5.1.0/model_doc/paddleocr_vl>
- vLLM PaddleOCR-VL recipe: <https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- vLLM supported models: <https://docs.vllm.ai/en/latest/models/supported_models.html>
