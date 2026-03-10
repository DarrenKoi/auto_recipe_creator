# 배포 레이아웃 및 설정 관리

## 1. 전제

이 문서는 아래 환경을 가정한다.

- GPU 서버: `H200 140GB x 2`
- OS: Linux 계열
- 런타임: 별도 Python 환경에 `vLLM 0.17` 설치 완료
- 네트워크: 사내망 또는 내부 VPN 기반
- 모델 파일: 클라우드 서버의 `data/models/` 아래에 Hugging Face weights가 이미 받아져 있는 상태

이 저장소의 `.venv`에는 `vllm`가 들어있지 않을 수 있으므로, 실제 서빙은 GPU 서버의 전용 런타임에서 수행하는 것이 맞다.

## 2. 권장 디렉터리 구조

모델 파일과 운영 설정을 분리하는 것을 권장한다.

```text
DEPLOY_VLMS_ROOT=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms

/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/
├── UI-Venus-1.5-8B/
├── MAI-UI-8B/
└── UI-TARS-1.5-7B/

${DEPLOY_VLMS_ROOT}/
├── scripts/
├── config/
│   ├── common.env
│   └── models/
│       ├── ui-venus.env
│       ├── mai-ui.env
│       └── ui-tars.env
└── templates/
    └── README.md
```

핵심은 아래 3개를 분리하는 것이다.

- `/project/.../data/models/`: 실제 모델 파일
- `${DEPLOY_VLMS_ROOT}/config/common.env`: 공통 옵션
- `${DEPLOY_VLMS_ROOT}/config/models/*.env`: 모델별 포트, GPU, alias

현재 클라우드 기준 모델 경로는 `/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/` 이다. 다른 경로라면 아래 예시의 `MODEL_ID`만 그 경로로 바꾸면 된다. 운영 중에는 가능하면 상대경로보다 절대경로를 쓰는 편이 낫다.

## 3. 공통 설정 파일

`${DEPLOY_VLMS_ROOT}/config/common.env`

```bash
HOST=127.0.0.1
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.80
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=8
TENSOR_PARALLEL_SIZE=1
API_KEY=
```

초기값 의미:

- `HOST=127.0.0.1`: 기본은 서버 로컬 접근만 허용한다. 다른 내부 머신에서 직접 붙어야 하면 내부 IP나 `0.0.0.0`으로 바꾼다.
- `DTYPE=bfloat16`: H200에서 시작값으로 무난하다.
- `GPU_MEMORY_UTILIZATION=0.80`: PoC 초기 안정성 우선값이다.
- `MAX_MODEL_LEN=8192`: 긴 시스템 프롬프트 + 이미지 1장을 넣기 위한 보수적 시작값이다.
- `MAX_NUM_SEQS=8`: 저동시성 PoC 기준 시작점이다.
- `TENSOR_PARALLEL_SIZE=1`: 7B/8B 모델은 단일 GPU에 올린다.
- `HF_HOME`은 기본값을 강제하지 않는다. 시스템 기본 Hugging Face cache path를 사용하고, 필요할 때만 별도로 지정한다.

## 4. 모델별 설정 파일

### 4.1 UI-Venus

`${DEPLOY_VLMS_ROOT}/config/models/ui-venus.env`

```bash
MODEL_ID=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/UI-Venus-1.5-8B
SERVED_MODEL_NAME=ui-venus-1.5-8b
PORT=8001
GPU_ID=0

# 모델 카드에서 별도 template를 요구하면 여기에 경로 지정
CHAT_TEMPLATE=

TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT=image=1
```

### 4.2 MAI-UI

`${DEPLOY_VLMS_ROOT}/config/models/mai-ui.env`

```bash
MODEL_ID=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/MAI-UI-8B
SERVED_MODEL_NAME=mai-ui-8b
PORT=8002
GPU_ID=1
CHAT_TEMPLATE=
TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT=image=1
```

### 4.3 다음 단계용 UI-TARS

`${DEPLOY_VLMS_ROOT}/config/models/ui-tars.env`

```bash
MODEL_ID=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/data/models/UI-TARS-1.5-7B
SERVED_MODEL_NAME=ui-tars-1.5-7b
PORT=8003
GPU_ID=0
CHAT_TEMPLATE=
TRUST_REMOTE_CODE=1
LIMIT_MM_PER_PROMPT=image=1
```

## 5. 설정 키 설명

| 키 | 범위 | 의미 | 권장값 |
|----|------|------|--------|
| `MODEL_ID` | 모델별 | 클라우드 서버의 로컬 모델 절대경로 | `/project/.../data/models/`의 절대경로 |
| `SERVED_MODEL_NAME` | 모델별 | OpenAI API에서 사용할 모델 alias | 짧고 안정적인 이름 |
| `PORT` | 모델별 | vLLM OpenAI 서버 포트 | `8001+` |
| `GPU_ID` | 모델별 | 바인딩할 GPU 번호 | `0`, `1` |
| `CHAT_TEMPLATE` | 모델별 | 별도 Jinja template 경로 | 필요 시만 사용 |
| `TRUST_REMOTE_CODE` | 모델별 | remote code 허용 여부 | `1` |
| `LIMIT_MM_PER_PROMPT` | 모델별 | 프롬프트당 이미지 수 제한 | `image=1` |
| `HOST` | 공통 | bind host | `127.0.0.1` |
| `DTYPE` | 공통 | weight dtype | `bfloat16` |
| `GPU_MEMORY_UTILIZATION` | 공통 | KV cache 포함 전체 메모리 활용 비율 | `0.80`부터 시작 |
| `MAX_MODEL_LEN` | 공통 | 최대 컨텍스트 길이 | `8192` |
| `MAX_NUM_SEQS` | 공통 | 동시 시퀀스 수 | `8` |
| `TENSOR_PARALLEL_SIZE` | 공통 | 텐서 병렬 크기 | `1` |
| `API_KEY` | 공통 | OpenAI 호환 API 인증키 | 내부망이면 비워도 됨 |

## 6. 설정 변경 규칙

운영 중에는 아래 순서를 권장한다.

1. 공통 성능 파라미터는 `common.env`만 바꾼다.
2. 모델 교체는 `models/<name>.env`의 `MODEL_ID`만 먼저 바꾼다.
3. `PORT`와 `SERVED_MODEL_NAME`은 가급적 바꾸지 않는다.
4. 큰 변경은 기존 포트를 덮어쓰지 말고 `8004` 같은 canary 포트에서 먼저 검증한다.
5. 정상 확인 뒤에만 운영 포트로 승격한다.

## 7. 업그레이드/롤백 팁

- 모델 revision을 자주 바꿀 계획이면 `MODEL_ID`를 버전 디렉터리로 잡고, alias는 그대로 두는 편이 낫다.
- 예:

```text
/project/.../data/models/UI-Venus-1.5-8B/2026-03-10/
/project/.../data/models/UI-Venus-1.5-8B/2026-03-24/
```

- 이 경우 `ui-venus.env`의 `MODEL_ID`만 새 버전 경로로 바꾸면 된다.
- 즉시 롤백이 필요하면 이전 경로로 되돌리고 프로세스만 재시작하면 된다.

## 8. chat template 관리

GUI 특화 모델은 모델 카드에 따라 별도 chat template를 요구할 수 있다.

- template가 필요 없는 모델: `CHAT_TEMPLATE=` 빈값 유지
- template가 필요한 모델: `${DEPLOY_VLMS_ROOT}/templates/<model>.jinja`에 저장하고 `CHAT_TEMPLATE`에 경로 지정
- template 추가 검증은 반드시 운영 포트가 아니라 canary 포트에서 먼저 수행

이 규칙을 지키면, 이후 `UI-TARS`, `UI-TARS-2`, `MAI-UI-32B`로 확장할 때도 설정 체계가 무너지지 않는다.
