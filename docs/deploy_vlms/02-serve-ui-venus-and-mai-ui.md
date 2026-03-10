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

- [serve_vlm.sh](./scripts/serve_vlm.sh)
- [start_ui_venus.sh](./scripts/start_ui_venus.sh)
- [start_mai_ui.sh](./scripts/start_mai_ui.sh)
- [start_ui_tars.sh](./scripts/start_ui_tars.sh)

스크립트는 오프라인/내부망 전용 환경변수를 먼저 세팅한 뒤 `vllm serve`를 실행한다.
아래 예시는 클라우드 서버에서 `docs/deploy_vlms`로 이동한 상태를 기준으로 한다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
mkdir -p config/models templates
```

### 3.1 UI-Venus

```bash
cp scripts/common.env.example config/common.env
cp scripts/models/ui-venus.env.example config/models/ui-venus.env

./scripts/start_ui_venus.sh
```

### 3.2 MAI-UI

다른 셸에서 실행:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
cp scripts/models/mai-ui.env.example config/models/mai-ui.env

./scripts/start_mai_ui.sh
```

참고:

- 어떤 모델은 별도 template가 필요할 수 있다.
- 그런 경우 `CHAT_TEMPLATE=/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms/templates/<name>.jinja`를 설정한다.
- exact flag 이름은 서버의 `vllm 0.17` 빌드 기준으로 `vllm serve --help`에서 최종 확인한다.
- GPU 서버에서는 `docs/deploy_vlms/config` 아래 env 파일만 수정해서 쓰는 방식을 권장한다.

## 4. 기동 확인

서버에서 아래로 응답을 확인한다.

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models

# 또는
./scripts/check_vlm.sh http://127.0.0.1:8001 ui-venus-1.5-8b
./scripts/check_vlm.sh http://127.0.0.1:8002 mai-ui-8b
```

정상이면 각 포트에서 `data[0].id` 또는 유사 필드에 아래 alias가 보여야 한다.

- `ui-venus-1.5-8b`
- `mai-ui-8b`

## 5. 수동 운영 메모

권한 제약이 있으면 모델별로 별도 셸에서 시작 스크립트를 실행하고, 응답 확인 후 그대로 유지하는 방식이 가장 단순하다.

- `UI-Venus`와 `MAI-UI`는 각각 다른 셸 또는 세션에서 실행한다.
- 중단은 해당 셸에서 `Ctrl+C`로 처리한다.
- 포트 점유 프로세스를 다시 확인할 때는 `ss -ltnp | grep 800` 같은 명령으로 PID를 찾는다.
- 로그 확인은 현재 실행 중인 터미널 출력 또는 별도 리다이렉트한 로그 파일 기준으로 본다.

## 6. 다음 단계 확장

`UI-TARS-1.5-7B`를 붙일 때는 기존 포트를 건드리지 말고 `8003`으로 먼저 올린다.

예:

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms
cp scripts/models/ui-tars.env.example config/models/ui-tars.env

./scripts/start_ui_tars.sh
```

PoC 단계에서는 `UI-Venus`, `MAI-UI`, `UI-TARS`를 각각 독립 포트로 유지하는 편이 결과 비교와 회귀 추적에 좋다.
