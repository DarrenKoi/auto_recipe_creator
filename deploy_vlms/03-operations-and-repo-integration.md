# 운영 방법 및 저장소 연동

## 1. 운영자가 실제로 자주 하는 작업

권한 제약이 있으면 자주 쓰는 작업은 수동 실행과 포트 헬스 체크 중심으로 정리하는 편이 낫다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/deploy_vlms
python scripts/start_ui_venus.py
python scripts/start_mai_ui.py
python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
ss -ltnp | grep 800
curl http://127.0.0.1:8001/v1/models
```

중단은 해당 셸에서 `Ctrl+C`로 처리하고, 백그라운드로 돌렸다면 포트 기준으로 PID를 찾아 종료하면 된다. `mai-ui`, `ui-tars`도 같은 방식으로 보면 된다.

size variant 연구는 아래처럼 generic start script를 쓰는 편이 더 단순하다.

```bash
python scripts/start_model.py ui-venus 2b
python scripts/check_vlm.py http://127.0.0.1:8102 ui-venus-2b
python scripts/start_model.py ui-venus 30b
python scripts/check_vlm.py http://127.0.0.1:8130 ui-venus-30b
```

## 2. PoC 코드와 연결하는 방법

이 저장소의 `poc/work`는 OpenAI 호환 endpoint를 아래 환경변수로 읽는다.

- `VLM_API_URL` 또는 `VLM_API_BASE_URL`
- `VLM_API_KEY`
- `VLM_MODEL_NAME`

관련 코드는 [poc/work/config.py](../poc/work/config.py), [poc/work/vlm_screen_analysis.py](../poc/work/vlm_screen_analysis.py), [poc/work/vlm_openai_client.py](../poc/work/vlm_openai_client.py)에 있다.

모델이 이미 클라우드 서버의 `data/models/` 아래에 있으므로, 이 저장소 쪽에서는 모델 다운로드를 신경 쓸 필요 없이 endpoint와 alias만 맞추면 된다.

실제 클라우드 주소 기준:

- base URL: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com`
- Flask API root: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api`

현재 코드 사실:

- `flask_api`의 기본 prefix는 `/api`다.
- VLM proxy는 `/api/vlm_serve/<service>/v1/...` 형태로 제공된다.
- 따라서 coworkers 에게는 direct port 대신 Flask API 주소를 줄 수 있다.

## 3. URL 패턴 정리

### 3.1 지금 바로 가능한 direct port 방식

이 방식은 vLLM이 각 포트에서 직접 OpenAI-compatible API를 노출한다.

| 모델 | 권장 URL |
|------|----------|
| UI-Venus | `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8001` |
| MAI-UI | `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8002` |
| UI-TARS | `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8003` |

이 저장소 클라이언트는 `/v1`를 자동으로 붙이므로 `VLM_API_URL`에는 위 값만 넣으면 된다.

### 3.2 Flask API gateway 방식

coworkers 용 공용 주소는 아래처럼 쓰면 된다.

- `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/ui-venus`
- `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/mai-ui`
- `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/ui-tars`

실제 OpenAI-compatible 경로는 아래처럼 된다.

- `/api/vlm_serve/ui-venus/v1/models`
- `/api/vlm_serve/ui-venus/v1/chat/completions`

## 4. 추천 `.env` 관리 방식

실험별 `.env`를 따로 두고, 현재 활성 파일만 `poc/work/.env`로 두는 식이 가장 단순하다.

### 4.1 UI-Venus 실험용

direct port 기준:

`poc/work/.env.ui-venus`

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8001
VLM_API_KEY=
VLM_MODEL_NAME=ui-venus-1.5-8b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

gateway 기준 예시는 아래처럼 잡을 수 있다.

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/ui-venus
VLM_API_KEY=
VLM_MODEL_NAME=ui-venus-1.5-8b
```

### 4.2 MAI-UI 실험용

`poc/work/.env.mai-ui`

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8002
VLM_API_KEY=
VLM_MODEL_NAME=mai-ui-8b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

### 4.3 UI-TARS 추가 시

`poc/work/.env.ui-tars`

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8003
VLM_API_KEY=
VLM_MODEL_NAME=ui-tars-1.5-7b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

### 4.4 size variant 실험용

예를 들어 같은 family를 `2B`, `7B`, `30B`로 비교한다면 아래처럼 별도 env를 두면 된다.

`poc/work/.env.ui-venus-2b`

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8102
VLM_API_KEY=
VLM_MODEL_NAME=ui-venus-2b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

`poc/work/.env.ui-venus-30b`

```bash
VLM_API_URL=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8130
VLM_API_KEY=
VLM_MODEL_NAME=ui-venus-30b

SAFE_MODE=true
USE_WEBP=true
MAX_IMAGE_SIZE=1280
```

## 5. 전환 절차

예를 들어 UI-Venus를 붙일 때는:

```bash
# Set poc/work/.env to the UI-Venus profile values first.
uv run python -m poc.work.list_up_tools
```

MAI-UI로 바꿀 때는:

```bash
# Set poc/work/.env to the MAI-UI profile values first.
uv run python -m poc.work.list_up_tools
```

이 저장소의 클라이언트는 base URL 끝의 `/v1`를 자동 보정하므로 `http://host:8001`만 넣어도 된다.

## 6. 운영 표준

운영 중에는 아래 기준을 고정하는 것이 좋다.

- 포트는 모델 식별자처럼 다룬다.
- `8001=ui-venus`, `8002=mai-ui`는 가능하면 계속 유지한다.
- size variant 비교는 `8102/8107/8130`처럼 family별 별도 포트대역을 두면 관리하기 쉽다.
- 모델 revision 교체는 포트 변경이 아니라 `MODEL_ID` 변경으로 처리한다.
- template 실험이나 옵션 실험은 `8004` 같은 별도 포트로 먼저 검증한다.
- 로그/결과 비교 시 `served-model-name`과 포트를 같이 기록한다.

예:

```text
endpoint=http://gpu-server:8001
model_name=ui-venus-1.5-8b
```

실제 클라우드 예시는 아래처럼 기록하는 편이 좋다.

```text
endpoint=http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8001
model_name=ui-venus-1.5-8b
```

## 7. 추천 벤치마크 순서

실험은 아래 순서를 권장한다.

1. `UI-Venus-1.5-8B`를 `8001`에 배포하고 기본 스모크 테스트
2. `MAI-UI-8B`를 `8002`에 배포하고 동일 스크린샷 세트 비교
3. `poc/work/automate_rcs_login.py` 또는 `poc/work/list_up_tools.py`에서 A/B 실행
4. 더 좋은 쪽을 기준 포트로 유지
5. 그 다음 `UI-TARS-1.5-7B`를 `8003`에 추가
6. 같은 family size 비교가 필요하면 `prepare_research_envs.py`로 `2B/7B/30B` env를 생성

## 8. 트러블슈팅 기준점

### 서버가 뜨지 않음

- `--trust-remote-code` 누락 여부 확인
- 모델 경로가 실제로 존재하는지 확인
- 현재 실행 중인 터미널 출력이나 별도 로그 파일에서 tokenizer/template 관련 에러 확인
- `UI-TARS-1.5-7B`는 `Qwen2.5-VL` 계열이므로, 현재 Python 환경에서 `transformers.models.qwen2_5_vl` import가 되는지 확인
- `UI-TARS-1.5-7B` 모델 디렉터리에 `preprocessor_config.json`, `tokenizer_config.json`, `chat_template.json`, `model.safetensors.index.json` 또는 `*.safetensors`가 실제로 있는지 확인

### 메모리 부족 또는 startup이 비정상적으로 무거움

아래 순서로 줄인다.

1. `MAX_MODEL_LEN=4096`
2. `MAX_NUM_SEQS=4`
3. `GPU_MEMORY_UTILIZATION=0.70`

초기 PoC에서는 throughput보다 안정성이 중요하므로, 무리해서 캐시를 크게 잡을 이유가 없다.

### 응답 형식이 기대와 다름

- `VLM_MODEL_NAME`이 `served-model-name`과 일치하는지 확인
- 모델 카드가 별도 chat template를 요구하는지 확인
- canary 포트에서 `CHAT_TEMPLATE` 적용 여부를 먼저 실험

### 저장소에서 연결은 되지만 품질이 낮음

- 우선 모델 자체 품질 차이인지 분리해야 하므로, 같은 프롬프트/같은 이미지/같은 `SAFE_MODE`로 A/B 비교
- 모델 교체와 프롬프트 변경을 동시에 하지 않는 것이 좋다

## 9. 나중에 확장할 때

현재 구조는 7B/8B GUI VLM 여러 개를 독립 포트로 운영하는 데 맞춰져 있다. 이후 아래 확장도 무리 없이 가능하다.

- `8003`: `UI-TARS-1.5-7B`
- `8004`: canary
- `8102/8107/8130`: `UI-Venus` size 연구용
- `8202/8207/8230`: `MAI-UI` size 연구용
- `8302/8307/8330`: `UI-TARS` size 연구용
- `8005+`: 차기 UI-Venus/MAI-UI revision
- 대형 모델(`MAI-UI-32B`, `UI-TARS-72B`)은 별도 문서로 분리하는 편이 낫다

핵심은 설정 체계를 먼저 고정해 두는 것이다. 그러면 모델이 늘어나도 운영 복잡도가 크게 증가하지 않는다.
