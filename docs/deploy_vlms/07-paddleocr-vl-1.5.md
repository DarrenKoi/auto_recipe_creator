# PaddleOCR-VL-1.5 배포 메모

`PaddleOCR-VL-1.5`는 현재 `docs/deploy_vlms`에 있는 `vLLM` 운영 체계로 바로 가져갈 수 있는 OCR VLM이다. 이 문서는 `PaddleOCR/PaddlePaddle` 직접 추론이 아니라 `vLLM` 배포 기준으로만 정리한다.

핵심 판단:

- 이 저장소 기준 권장 경로는 `기존 GPU 서버 + 기존 vLLM wrapper 재사용`이다.
- 현재 확인한 Linux 클라우드 런타임 `Python 3.11 + vLLM 0.17.0 + transformers 4.57.6 + torch 2.10.0`이면 `vLLM` 경로로 운영하면 된다.
- `vLLM` supported models에 `PaddleOCRVLForConditionalGeneration`가 있으므로, 이 문서에서는 다른 런타임 경로를 따로 확장하지 않는다.
- 즉, `vLLM`으로 쓸 때는 `paddleocr[doc-parser]`를 설치할 필요가 없다.

## 1. 공식 설치 방식 요약

공식 기준으로 중요한 점은 아래 2개다.

1. `vLLM` supported models에 `PaddleOCRVLForConditionalGeneration`가 올라와 있다.
2. `PaddleOCR-VL-1.5` model card도 이 family의 `vLLM` usage guide를 연결한다.

따라서 `docs/deploy_vlms`에서는 `vLLM` 경로만 보면 된다. 현재 저장소의 [serve_vlm.py](./scripts/serve_vlm.py)는 특정 GUI 모델 전용이 아니라, `MODEL_ID`, `SERVED_MODEL_NAME`, `PORT`, `GPU_ID`를 읽어 일반 `vLLM` 서버를 띄우는 wrapper라서 `PaddleOCR-VL-1.5`에도 그대로 맞는다. 현재 코드에서 별도 특수 처리가 필요한 쪽은 `Qwen2.5-VL` 계열뿐이다.

## 2. 현재 작업 환경에 맞춘 권장 경로

현재 확인한 클라우드 환경:

- OS `Linux`
- Python `3.11`
- `vLLM 0.17.0`
- `transformers 4.57.6`
- `torch 2.10.0`

이 조합이면 `PaddleOCR-VL-1.5`를 현재 Linux 클라우드에서 바로 `vLLM`으로 운영할 수 있다. 이 문서 기준으로 추가로 신경 쓸 것은 아래뿐이다.

- Linux GPU 서버에서 기존 `vLLM` wrapper 사용
- `MODEL_ID`, `PORT`, `GPU_ID`, `SERVED_MODEL_NAME`만 맞춤
- `paddleocr[doc-parser]` 같은 별도 PaddleOCR 패키지는 설치하지 않음

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

`PaddleOCR-VL-1.5`는 `기존 vLLM 문서 체계에 그대로 편입 가능한 모델`이다. 현재 Linux 클라우드에서는 `vLLM` 경로만 보면 충분하고, 이 사용 방식에서는 `paddleocr[doc-parser]`를 설치할 필요가 없다.

## 5. 참고 source

- Hugging Face model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5-0.9B>
- PaddleOCR repository: <https://github.com/PaddlePaddle/PaddleOCR>
- vLLM PaddleOCR-VL recipe: <https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- vLLM supported models: <https://docs.vllm.ai/en/latest/models/supported_models.html>
