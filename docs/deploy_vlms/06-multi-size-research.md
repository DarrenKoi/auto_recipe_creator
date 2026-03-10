# 같은 family의 multi-size 연구 준비

이 문서는 같은 model family를 `2B`, `7B`, `30B`처럼 여러 size로 비교 실험할 때의 기준을 정리한다.

핵심은 아래 3가지다.

- instance 이름을 `family-size`로 고정한다.
- size별 튜닝값은 `config/models/<instance>.env`에서 override한다.
- 고정 wrapper 대신 generic script를 사용한다.

## 1. instance naming 규칙

권장 규칙:

- `ui-venus-2b`
- `ui-venus-7b`
- `ui-venus-30b`
- `mai-ui-2b`
- `ui-tars-30b`

이 규칙을 쓰면 아래가 한 줄로 정렬된다.

- env 파일명
- start/check 스크립트 인자
- `SERVED_MODEL_NAME`
- 벤치마크 결과 표의 모델 키

## 2. 연구용 env 자동 생성

클라우드 서버에서 아래를 한 번 실행하면 research env 초안을 만든다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
python scripts/prepare_research_envs.py
```

family 하나만 만들고 싶으면:

```bash
python scripts/prepare_research_envs.py ui-venus
```

특정 instance만 만들고 싶으면:

```bash
python scripts/prepare_research_envs.py ui-venus-30b
```

생성 대상:

- `config/models/ui-venus-2b.env`
- `config/models/ui-venus-7b.env`
- `config/models/ui-venus-30b.env`
- `config/models/mai-ui-2b.env`
- `config/models/mai-ui-7b.env`
- `config/models/mai-ui-30b.env`
- `config/models/ui-tars-2b.env`
- `config/models/ui-tars-7b.env`
- `config/models/ui-tars-30b.env`

주의:

- 생성된 `MODEL_ID`는 `SET_ME_...` placeholder를 사용한다.
- 실제 로컬 모델 디렉터리명으로 반드시 바꿔야 한다.
- 기존 파일은 기본적으로 덮어쓰지 않는다. 다시 만들려면 `OVERWRITE_EXISTING=1`을 준다.

## 3. size별 기본값

generator는 아래 운영 가정을 기본값으로 넣는다.

| size | 기본 포트 패턴 | 기본 GPU | 기본 TP | 기본 목적 |
|------|----------------|----------|---------|-----------|
| `2B` | family별 `x102` | `0` | `1` | 빠른 latency/cheap baseline |
| `7B` | family별 `x107` | `1` | `1` | 주력 비교 baseline |
| `30B` | family별 `x130` | `0,1` | `2` | 고품질 비교, 단독 세션 권장 |

family별 포트 예:

| family | `2B` | `7B` | `30B` |
|--------|------|------|--------|
| `ui-venus` | `8102` | `8107` | `8130` |
| `mai-ui` | `8202` | `8207` | `8230` |
| `ui-tars` | `8302` | `8307` | `8330` |

이 값은 시작점일 뿐이다. 실제 메모리/throughput 상태에 따라 아래 값을 env에서 조정하면 된다.

- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `MAX_NUM_SEQS`
- `TENSOR_PARALLEL_SIZE`

KV cache 관련 실험이 더 필요하면 `EXTRA_VLLM_ARGS`를 instance별로 따로 넣는다.

예:

```bash
# EXTRA_VLLM_ARGS=--kv-cache-memory-bytes 40G
EXTRA_VLLM_ARGS=
```

어떤 flag 이름을 쓸지는 GPU 서버의 실제 `vllm --help` 출력 기준으로 맞추는 편이 안전하다.

## 4. 시작과 확인

`2B` 시작:

```bash
python scripts/start_model.py ui-venus 2b
python scripts/check_vlm.py http://127.0.0.1:8102 ui-venus-2b
```

`30B` 시작:

```bash
python scripts/start_model.py ui-venus 30b
python scripts/check_vlm.py http://127.0.0.1:8130 ui-venus-30b
```

`check_vlm.py`는 이제 `served-model-name` 대신 instance 이름을 받아도 된다. `config/models/<instance>.env`에서 alias를 읽어 자동 해석한다.

## 5. 동시에 여러 size를 띄울 때

동시에 띄울 때는 아래처럼 분리해서 보는 편이 낫다.

- `2B`, `7B`: 서로 다른 단일 GPU + 서로 다른 port
- `30B`: 기본적으로 2GPU 전용 세션
- 같은 family라도 port, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `EXTRA_VLLM_ARGS`는 각 env에서 독립 관리

예:

```text
ui-venus-2b  -> port 8102, GPU_ID=0
ui-venus-7b  -> port 8107, GPU_ID=1
ui-venus-30b -> port 8130, GPU_ID=0,1
```

## 6. 이 저장소와 연결

예를 들어 `poc/work`에서 `ui-venus-30b`를 붙일 때는 아래처럼 별도 env를 두면 된다.

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8130
VLM_API_KEY=
VLM_MODEL_NAME=ui-venus-30b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

비교 실험은 `poc/work/.env.ui-venus-2b`, `poc/work/.env.ui-venus-7b`, `poc/work/.env.ui-venus-30b`처럼 분리해 두는 편이 가장 단순하다.

## 7. 운영 해석

- 기존 운영 baseline인 `8001/8002/8003`은 유지한다.
- size variant 연구는 `81xx/82xx/83xx` 대역으로 분리한다.
- `30B`는 기본적으로 2GPU 전용 세션으로 본다.
- 같은 family를 비교할 때는 프롬프트, 이미지, `SAFE_MODE`, post-processing을 고정한다.
- 모델 교체와 프롬프트 수정을 한 번에 하지 않는다.
