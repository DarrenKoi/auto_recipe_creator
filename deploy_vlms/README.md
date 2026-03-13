# VLM 배포 가이드

`H200 GPU 2장` 환경에서 GUI 특화 VLM과 일부 OCR VLM을 배포하고, 이 저장소의 테스트 코드와 연결하기 위한 운영 문서 모음이다. 현재 주력 운영 문서는 `vLLM 0.17` 기준이고, OCR 모델은 런타임 성격에 따라 별도 메모를 같이 둔다.

실제 클라우드 기준 주소:

- 클라우드 base URL: `http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/`
- Flask API root: `http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api`
- 클라우드 repo root: `/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image`
- `deploy_vlms` 작업 루트: `/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms`

주의:

- 현재 저장소의 `flask_api` 패키지는 `/api/vlm_serve/<service>/v1/...` 형태의 VLM proxy route를 제공한다.
- 즉, coworkers 용 endpoint는 `http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api/vlm_serve/ui-venus` 같은 형태로 잡으면 된다.

## 권장 시작점

초기 PoC는 아래 2개만 먼저 올리는 구성을 권장한다.

| 순서 | GPU | 포트 | 모델 | served-model-name | 비고 |
|------|-----|------|------|-------------------|------|
| 1 | GPU 0 | `8001` | `UI-Venus-1.5-8B` | `ui-venus-1.5-8b` | 1차 주력 후보 |
| 2 | GPU 1 | `8002` | `MAI-UI-8B` | `mai-ui-8b` | 비교/A-B 후보 |
| 3 | GPU 0 또는 1 | `8003` | `UI-TARS-1.5-7B` | `ui-tars-1.5-7b` | 다음 단계 확장 |

지금은 VRAM을 꽉 채우는 방향보다, `모델 1개 = 포트 1개 = 서비스 1개`로 단순하게 운영하는 편이 PoC 속도와 장애 분석 측면에서 유리하다.

같은 family를 여러 size로 비교해야 하면 `instance=<family>-<size>` 규칙으로 확장하면 된다.

- `ui-venus-2b`
- `ui-venus-7b`
- `ui-venus-30b`
- `mai-ui-2b`
- `ui-tars-30b`

## 2~3개 소형 모델을 한 GPU에 같이 올릴 때

`vLLM`의 `GPU_MEMORY_UTILIZATION`은 고정 상수로 잡기보다, GPU 1장당 공유 예산으로 계산하는 편이 안전하다.

- 기본 식: `u_recommended = ((M_gpu - M_shared) / N_models - M_proc) / M_gpu`
- KV cache 식: `M_kv ~= 2 * L * H_kv * D_head * bytes(dtype) * MAX_MODEL_LEN * MAX_NUM_SEQS / TP`
- 판정 식: `M_weights/TP + M_kv + M_proc <= (M_gpu - M_shared) / N_models`

여기서:

- `M_gpu`: GPU 총 VRAM GiB
- `M_shared`: GPU 전체에서 공통으로 남겨 둘 여유분. 기본 `8 GiB`
- `M_proc`: 프로세스별 CUDA/vLLM/mm encoder 여유분. 기본 `4 GiB`
- `N_models`: 같은 GPU를 공유하는 모델 개수
- `TP`: `TENSOR_PARALLEL_SIZE`

H200 `140GB` 기준 시작값은 대략 아래처럼 보면 된다.

- 8B 2개 공유: `GPU_MEMORY_UTILIZATION ~= 0.44`
- 8B 3개 공유: `GPU_MEMORY_UTILIZATION ~= 0.29`

이 문서의 [serve_vlm.py](./scripts/serve_vlm.py)는 이제 로컬 `config.json`과 weight shard 크기를 읽어서 이 공식을 자동 적용할 수 있다.

```bash
AUTO_TUNE_GPU_MEMORY_UTILIZATION=1
COLOCATED_MODELS_PER_GPU=2
GPU_SHARED_RESERVE_GIB=8
GPU_PROCESS_RESERVE_GIB=4
```

`GPU_MEMORY_UTILIZATION=auto`로 써도 같은 경로를 탄다. `nvidia-smi`를 못 읽는 환경이면 `GPU_TOTAL_MEMORY_GIB=140`을 같이 주면 된다.

`3 x 8B`가 실제로 맞는지는 `num_key_value_heads`와 weight shard 크기에 따라 달라진다. 자동 계산이 실패하면 `MAX_NUM_SEQS`를 먼저 줄이고, 그다음 `MAX_MODEL_LEN`을 내리는 편이 가장 단순하다.

## 포트 정책

- `8000`은 비워 둔다.
- 실제 모델 포트는 `8001`, `8002`, `8003` 순서로 증가시킨다.
- 포트와 모델 alias는 고정한다. 모델 파일 경로만 바꾸더라도 포트와 alias는 가능하면 유지한다.
- 신규 실험 모델은 기존 포트를 덮어쓰지 말고 다음 빈 포트에 먼저 올린다.

권장 포트 예약표:

| 포트 | 용도 |
|------|------|
| `8001` | `UI-Venus-1.5-8B` 운영/기준 |
| `8002` | `MAI-UI-8B` 운영/비교 |
| `8003` | `UI-TARS-1.5-7B` 또는 차기 7B/8B 후보 |
| `8004` | canary, chat-template 실험, revision 검증 |

## 운영 원칙

- `served-model-name`은 Hugging Face repo 이름 대신 짧고 안정적인 alias를 쓴다.
- `MODEL_ID`는 클라우드 서버의 로컬 절대경로를 사용한다. 지금처럼 모델이 이미 `data/models/` 아래에 받아져 있으면 그 경로를 직접 쓰는 편이 가장 안정적이다.
- 공통 설정과 모델별 설정을 분리한다.
- 권한 범위 내에서 수동 실행 + 헬스 체크 기준으로 운영한다.
- 이 저장소에서는 `VLM_API_URL`, `VLM_MODEL_NAME`, `VLM_API_KEY`만 맞추면 바로 붙는다.

## 문서 순서

1. [01-layout-and-settings.md](./01-layout-and-settings.md)
2. [02-serve-ui-venus-and-mai-ui.md](./02-serve-ui-venus-and-mai-ui.md)
3. [03-operations-and-repo-integration.md](./03-operations-and-repo-integration.md)
4. [04-offline-and-network-policy.md](./04-offline-and-network-policy.md)
5. [05-ui-tars-vs-others.md](./05-ui-tars-vs-others.md)
6. [06-multi-size-variants.md](./06-multi-size-variants.md)
7. [07-paddleocr-vl-1.5.md](./07-paddleocr-vl-1.5.md)
8. [08-got-ocr-2.0-hf.md](./08-got-ocr-2.0-hf.md)

## OCR VLM 추가 판단

이번에 확인한 2개 OCR 모델은 같은 폴더 아래에서 관리하되, 배포 방식은 분리하는 편이 맞다.

| 모델 | 공식 런타임 성격 | 이 저장소 권장 경로 |
|------|------------------|---------------------|
| `PaddleOCR-VL-1.5` | `vLLM` | 현재 Linux 클라우드 `Python 3.11 + vLLM 0.17.0` 기준으로는 기존 `deploy_vlms` 체계에 편입. 새 model env만 추가해서 GPU 서버에서 `vLLM`으로 운영 |
| `GOT-OCR-2.0-hf` | `transformers` 중심 | 현재 클라우드 `Python 3.11 + transformers 4.57.6 + torch 2.10.0`에서 직접 추론 가능. 현 문서 기준 `vLLM` 경로와 분리 |

## 실행 스크립트

실행 가능한 예시 스크립트도 같이 추가했다.

- [serve_vlm.py](./scripts/serve_vlm.py)
- [start_model.py](./scripts/start_model.py)
- [check_vlm.py](./scripts/check_vlm.py)
- [prepare_variant_envs.py](./scripts/prepare_variant_envs.py)
- [start_paddleocr_vl.py](./scripts/start_paddleocr_vl.py)
- [run_got_ocr.py](./scripts/run_got_ocr.py)
- [common.env](./config/common.env)
- [ui-venus.env](./config/models/ui-venus.env)
- [mai-ui.env](./config/models/mai-ui.env)
- [ui-tars.env](./config/models/ui-tars.env)
- [paddleocr-vl-1.5.env](./config/models/paddleocr-vl-1.5.env)
- [got-ocr-2.0-hf.env](./config/models/got-ocr-2.0-hf.env)

이제 기본 env 파일도 `config/` 아래에 같이 두므로, 클라우드에서는 필요한 값만 수정한 뒤 바로 실행하면 된다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms

python scripts/start_ui_venus.py
python scripts/start_mai_ui.py
python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
python scripts/check_vlm.py http://127.0.0.1:8002 mai-ui-8b
```

size variant env를 미리 만들고 싶으면 아래처럼 준비하면 된다.

```bash
python scripts/prepare_variant_envs.py ui-venus
python scripts/start_model.py ui-venus 2b
python scripts/check_vlm.py http://127.0.0.1:8102 ui-venus-2b

python scripts/start_model.py ui-venus 30b
python scripts/check_vlm.py http://127.0.0.1:8130 ui-venus-30b
```

이 스크립트들은 다음을 기본 전제로 둔다.

- 모델은 클라우드 서버의 `/project/.../data/models/...` 아래 로컬 절대경로에 있어야 한다.
- 설정 파일은 `deploy_vlms/config/common.env`, `deploy_vlms/config/models/*.env`에 둔다.
- Hugging Face Hub 직접 접근은 금지한다.
- telemetry와 usage stats는 비활성화한다.
- proxy 환경변수는 기본적으로 제거한다.
- 회사 정책상 outbound가 이미 차단되어 있다면, 추가 네트워크 설명은 생략해도 된다.

## 이 저장소와 바로 연결되는 설정 키

`poc/work`는 아래 키를 사용한다.

- `VLM_API_URL`
- `VLM_API_BASE_URL`
- `VLM_API_KEY`
- `VLM_MODEL_NAME`

클라이언트가 `/v1`를 자동 처리하므로 아래 둘 다 사용 가능하다.

- `http://<gpu-server>:8001`
- `http://<gpu-server>:8001/v1`

실제 클라우드 예시:

- direct port: `http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com:8001`
- flask api proxy: `http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api/vlm_serve/ui-venus`

## 빠른 요약

- 먼저 `UI-Venus-1.5-8B -> 8001`, `MAI-UI-8B -> 8002`
- 공통값은 `common.env`, 모델별 값은 `models/<name>.env`
- size variant는 `models/<family>-<size>.env`로 확장하고, port/KV cache 관련 튜닝도 model env에서 개별 관리
- 수동 검증 후 같은 시작 스크립트로 재기동/운영
- PoC 전환은 `poc/work/.env`에서 `VLM_API_URL`과 `VLM_MODEL_NAME`만 바꾸면 된다
