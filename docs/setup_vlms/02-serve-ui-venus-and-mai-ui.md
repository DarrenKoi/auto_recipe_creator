# UI-Venus / MAI-UI 서빙 절차

## 1. 사전 점검

서빙 전에 최소한 아래는 확인한다.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
vllm --help
ss -ltn | grep 800
```

체크 포인트:

- GPU 2장이 모두 보이는지
- `vllm serve` CLI가 실제 서버에서 실행되는지
- `8001`, `8002`가 비어 있는지

## 2. 1차 권장 배치

초기 PoC는 아래처럼 단순하게 시작한다.

| GPU | 포트 | 모델 | 목적 |
|-----|------|------|------|
| `0` | `8001` | `UI-Venus-1.5-8B` | 주력 grounding 후보 |
| `1` | `8002` | `MAI-UI-8B` | 비교/A-B 후보 |

이 구성이 메모리 효율만 보면 매우 여유롭지만, PoC 초반에는 장애 분리와 비교 실험이 훨씬 중요하다.

## 3. 수동 스모크 테스트

직접 긴 `vllm serve ...` 명령을 치는 대신, 아래 예시 스크립트를 복사해서 쓰는 편이 운영상 더 낫다.

- [serve_vlm.py](../../deploy_vlms/scripts/serve_vlm.py)
- [start_model.py](../../deploy_vlms/scripts/start_model.py)
- [start_ui_venus.py](../../deploy_vlms/scripts/start_ui_venus.py)
- [start_mai_ui.py](../../deploy_vlms/scripts/start_mai_ui.py)
- [start_ui_tars.py](../../deploy_vlms/scripts/start_ui_tars.py)
- [prepare_variant_envs.py](../../deploy_vlms/scripts/prepare_variant_envs.py)

스크립트는 오프라인/내부망 전용 환경변수를 먼저 세팅한 뒤 `python -m vllm.entrypoints.openai.api_server`를 실행한다.
`start_model.py`와 `start_*.py`는 기본적으로 nohup 유사 백그라운드 모드로 떠서 터미널을 닫아도 유지된다.
로그는 `deploy_vlms/runtime/logs/<instance>.log`, PID는 `deploy_vlms/runtime/pids/<instance>.pid`에 남는다.
현재 셸에 붙여 디버깅하고 싶으면 [start_model.py](../../deploy_vlms/scripts/start_model.py)의 `RUN_IN_BACKGROUND` 값을 `0`으로 바꾼다.
아래 예시는 클라우드 서버에서 `deploy_vlms`로 이동한 상태를 기준으로 한다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
```

### 3.1 UI-Venus

```bash
python scripts/start_ui_venus.py
```

### 3.2 MAI-UI

다른 셸에서 실행:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms

python scripts/start_mai_ui.py
```

참고:

- 어떤 모델은 별도 template가 필요할 수 있다.
- 그런 경우 `CHAT_TEMPLATE=/absolute/path/to/<name>.jinja`처럼 실제 파일 경로를 넣는다.
- exact flag 이름은 서버의 `vllm 0.17` 빌드 기준으로 `vllm serve --help`에서 최종 확인한다.
- GPU 서버에서는 `deploy_vlms/config` 아래 env 파일만 수정해서 쓰는 방식을 권장한다.

### 3.3 같은 family를 여러 size로 운영할 때

고정 wrapper(`start_ui_venus.py`) 대신 generic wrapper를 쓰는 편이 낫다.

예:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
# 필요하면 config/common.env, config/models/ui-venus.env 값을 먼저 수정한다.
python scripts/prepare_variant_envs.py ui-venus
python scripts/start_model.py ui-venus 2b
```

다른 셸에서:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
python scripts/start_model.py ui-venus 7b
```

`30B`는 기본 variant env가 `GPU_ID=0,1`, `TENSOR_PARALLEL_SIZE=2`로 생성되며, 필요하면 `EXTRA_VLLM_ARGS`에 KV cache 관련 flag를 instance별로 따로 넣을 수 있다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
python scripts/start_model.py ui-venus 30b
```

## 4. 기동 확인

서버에서 아래로 응답을 확인한다.

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models

# 또는
python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
python scripts/check_vlm.py http://127.0.0.1:8002 mai-ui-8b
```

size variant는 `served-model-name` 대신 instance 이름을 넘겨도 된다. `check_vlm.py`가 `config/models/<instance>.env`를 읽어 alias를 자동 해석한다.

```bash
python scripts/check_vlm.py http://127.0.0.1:8102 ui-venus-2b
python scripts/check_vlm.py http://127.0.0.1:8130 ui-venus-30b
```

정상이면 각 포트에서 `data[0].id` 또는 유사 필드에 아래 alias가 보여야 한다.

- `ui-venus-1.5-8b`
- `mai-ui-8b`

## 5. 수동 운영 메모

권한 제약이 있으면 시작 스크립트로 백그라운드 기동 후 포트 헬스 체크와 로그 파일 확인만 하는 방식이 가장 단순하다.

- `UI-Venus`와 `MAI-UI`는 같은 셸에서 순서대로 올려도 된다. 스크립트가 바로 반환된다.
- 중단은 `python scripts/stop_model.py <instance>`로 처리한다.
- 포트 점유 프로세스를 다시 확인할 때는 `ss -ltnp | grep 800` 같은 명령으로 PID를 찾는다.
- 로그 확인은 `tail -f runtime/logs/<instance>.log` 기준으로 본다.

## 6. 다음 단계 확장

`UI-TARS-1.5-7B`를 붙일 때는 기존 포트를 건드리지 말고 `8003`으로 먼저 올린다.

`UI-TARS-1.5-7B`는 Hugging Face 쪽에서 `Qwen2.5-VL` 아키텍처로 배포되어 있고, 모델 파일 목록에 `preprocessor_config.json`, `tokenizer_config.json`, `chat_template.json`, `model.safetensors.index.json`이 같이 들어 있다. 지금 `serve_vlm.py`는 이 파일들을 먼저 확인하고, `CHAT_TEMPLATE`를 비워 두면 모델 디렉터리 안의 `chat_template.json`을 자동으로 사용한다.

예:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
python scripts/start_ui_tars.py
```

단일 GPU로 먼저 확인할 때는 예시 파일 그대로 두고, 다중 GPU throughput 실험이 필요하면 아래처럼 바꾼다.

```bash
GPU_ID=0,1,2,3
DATA_PARALLEL_SIZE=4
```

PoC 단계에서는 `UI-Venus`, `MAI-UI`, `UI-TARS`를 각각 독립 포트로 유지하는 편이 결과 비교와 회귀 추적에 좋다.
