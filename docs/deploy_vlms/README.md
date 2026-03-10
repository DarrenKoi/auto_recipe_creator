# VLM 배포 가이드

`H200 GPU 2장` 환경에서 GUI 특화 VLM을 `vLLM 0.17`로 배포하고, 이 저장소의 `poc/work` 코드와 연결하기 위한 운영 문서 모음이다.

실제 클라우드 기준 주소:

- 클라우드 base URL: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/`
- Flask API root: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api`
- 클라우드 repo root: `/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image`
- `deploy_vlms` 작업 루트: `/project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms`

주의:

- 현재 저장소의 `flask_api` 패키지는 `/api/vlm_serve/<service>/v1/...` 형태의 VLM proxy route를 제공한다.
- 즉, coworkers 용 endpoint는 `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/ui-venus` 같은 형태로 잡으면 된다.

## 권장 시작점

초기 PoC는 아래 2개만 먼저 올리는 구성을 권장한다.

| 순서 | GPU | 포트 | 모델 | served-model-name | 비고 |
|------|-----|------|------|-------------------|------|
| 1 | GPU 0 | `8001` | `UI-Venus-1.5-8B` | `ui-venus-1.5-8b` | 1차 주력 후보 |
| 2 | GPU 1 | `8002` | `MAI-UI-8B` | `mai-ui-8b` | 비교/A-B 후보 |
| 3 | GPU 0 또는 1 | `8003` | `UI-TARS-1.5-7B` | `ui-tars-1.5-7b` | 다음 단계 확장 |

지금은 VRAM을 꽉 채우는 방향보다, `모델 1개 = 포트 1개 = 서비스 1개`로 단순하게 운영하는 편이 PoC 속도와 장애 분석 측면에서 유리하다.

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

## 실행 스크립트

실행 가능한 예시 스크립트도 같이 추가했다.

- [serve_vlm.py](./scripts/serve_vlm.py)
- [check_vlm.py](./scripts/check_vlm.py)
- [common.env.example](./scripts/common.env.example)
- [ui-venus.env.example](./scripts/models/ui-venus.env.example)
- [mai-ui.env.example](./scripts/models/mai-ui.env.example)
- [ui-tars.env.example](./scripts/models/ui-tars.env.example)

이제 스크립트는 `docs/deploy_vlms` 기준으로 경로를 자동 해석하므로, 클라우드에서 아래처럼 바로 실행하면 된다.

```bash
cd /project/day/workSpace/itc-1stop-solution/itc-1stop-solution-gpu-image/docs/deploy_vlms

mkdir -p config/models templates
cp scripts/common.env.example config/common.env
cp scripts/models/ui-venus.env.example config/models/ui-venus.env
cp scripts/models/mai-ui.env.example config/models/mai-ui.env

python scripts/start_ui_venus.py
python scripts/start_mai_ui.py
python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b
python scripts/check_vlm.py http://127.0.0.1:8002 mai-ui-8b
```

이 스크립트들은 다음을 기본 전제로 둔다.

- 모델은 클라우드 서버의 `/project/.../data/models/...` 아래 로컬 절대경로에 있어야 한다.
- 설정 파일은 `docs/deploy_vlms/config/common.env`, `docs/deploy_vlms/config/models/*.env`에 둔다.
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

- direct port: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com:8001`
- flask api proxy: `http://itc-1stop-solution-gpu-image-webpp.aipp02.skhynix.com/api/vlm_serve/ui-venus`

## 빠른 요약

- 먼저 `UI-Venus-1.5-8B -> 8001`, `MAI-UI-8B -> 8002`
- 공통값은 `common.env`, 모델별 값은 `models/<name>.env`
- 수동 검증 후 같은 시작 스크립트로 재기동/운영
- PoC 전환은 `poc/work/.env`에서 `VLM_API_URL`과 `VLM_MODEL_NAME`만 바꾸면 된다
