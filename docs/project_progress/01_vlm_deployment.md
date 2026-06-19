# 01. VLM 배포·운영 (deploy_vlms + Flask Proxy)

> 목적: 오픈소스 VLM(Vision Language Model)을 조사·선정하고, 사내 HCP(GPU 서버)에 설치·운영하는
> 기반을 구축한다. 이후 모든 workflow가 이 인프라를 통해 화면을 이해한다.

근거: `docs/setup_vlms/01~05`, `deploy_vlms/`, `flask_api/vlm_serve/`.

## 1. 왜 다중 특화 모델인가 (조사·선정 배경)

단일 프런티어 LLM/VLM 하나로 모든 작업을 시키는 대신, **각 단계의 강점이 다른 오픈소스 VLM을
역할별로 분담**하는 전략을 택했다. 이유:

- 전체 화면에서 클릭 좌표를 한 번에 픽셀 정확도로 맞히기는 어렵다 → "대략 영역(coarse)"과
  "정밀 클릭(fine)"을 다른 모델로 분리.
- 텍스트 검증·OCR은 전용 OCR 모델이 더 정확하고 가볍다.
- 거대 단일 모델은 하드웨어 비용이 비현실적이다(아래 §5).

## 2. 선정 모델 5종과 역할

| 모델 | 파라미터 | Port | 역할 (route_slug) |
|------|----------|------|------------------|
| **UI-Venus-1.5-8B** | 8.3 B | 8001 | 메인 GUI grounding — 전체 화면에서 타겟 UI의 coarse bbox (`ui-venus`) |
| **MAI-UI-8B** | 8 B | 8002 | 정밀 보정 — coarse crop을 확대해 픽셀 단위 클릭점 (`mai-ui`) |
| **UI-TARS-1.5-7B** | 7.6 B | 8003 | GUI agent 대안 경로 (Qwen2.5-VL 계열, 전용 chat template 필요) (`ui-tars`) |
| **PaddleOCR-VL-1.5** | 0.9 B | 8004 | OCR 보조 — 텍스트 읽기·`Spotting:`(좌표)·표 인식 (`paddleocr-vl-1.5`) |
| **GOT-OCR-2.0-hf** | 0.58 B | 8005 | 하드 OCR fallback — 좁힌 영역 재판독 (transformers 직접 추론) (`got-ocr`) |

대표 파이프라인(workflow_1 로그인 자동화 기준):
**Capture → UI-Venus(coarse bbox) → Crop&Zoom → MAI-UI(refined point) → PaddleOCR-VL(입력 검증)**.

## 3. 사내 HCP 설치 구성

- **하드웨어**: H200 140 GiB × 2 (`common.env: GPU_TOTAL_MEMORY_GIB=140`).
- **서빙 런타임**: vLLM (BF16) — 4개 모델. GOT-OCR만 `transformers` 직접 추론.
- **GPU 배치 (co-location + auto-tune)**:
  - **GPU 0**: UI-Venus + UI-TARS, 각 `u≈0.44`(auto-tune) → ~123 GiB 사용(88%).
  - **GPU 1**: MAI-UI(`u=0.45`) + PaddleOCR-VL(`u=0.10`) + GOT-OCR(~4 GiB) → ~81 GiB 사용(58%).
- **Auto-tune 공식**: `u = ((M_gpu − M_shared)/N_models − M_proc)/M_gpu = ((140−8)/2−4)/140 ≈ 0.44`.
- **오프라인 정책**: HF live pull 금지(모델 사전 stage), telemetry/usage stats 비활성
  (`HF_HUB_OFFLINE=1`, `VLLM_DO_NOT_TRACK=1` 등), 절대 경로 사용.
- **vLLM 활용 기능**: PagedAttention(KV 캐시 효율), continuous batching, prefix caching
  (고정 프롬프트 prefix 재사용에 유리).

## 4. 운영 자동화 스크립트 (deploy_vlms/scripts)

| 스크립트 | 역할 |
|----------|------|
| `serve_vlm.py` | vLLM 진입점 — env 로드, 모델 경로 검증, 메모리 sizing, 오프라인 강제, vLLM 커맨드 구성 |
| `start_model.py` | 백그라운드 launcher — PID/로그 추적, 충돌 시 기존 인스턴스 중지 후 재기동 |
| `start_all.py` | 5개 서비스 GPU 배치대로 순차 기동 (모델 간 10초 간격) |
| `check_vlm.py` | health probe — `config/models/*.env` 스캔 후 각 포트 `/v1/models` 검증 |
| `run_got_ocr.py` / `serve_got_ocr.py` | GOT-OCR transformers 추론/서버 (vLLM 아님) |
| `stop_model.py` | 프로세스 종료·포트 정리 |

- 모델별 설정은 `config/common.env`(공통) + `config/models/<model>.env`(override)로 분리.
- UI-TARS는 Qwen2.5-VL 런타임 특수 처리: 전용 `chat_templates/ui-tars.jinja`, 필수 파일 검증.

## 5. Flask Proxy — 통합 라우팅·헬스 디스커버리

`flask_api/vlm_serve/` — 여러 모델을 하나의 OpenAI 호환 API로 묶는 proxy 계층.

- **서비스 레지스트리**: `config.py`의 `ALL_VLM_SERVICES` — 모델당 `VLMServiceEntry`
  (route_slug / display_name / model_name / upstream_port / enabled). `enabled` 플래그로 on/off 일원화.
- **proxy URL 패턴**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.
- **health 엔드포인트**: `GET /api/vlm_serve/health` — 각 upstream `/v1/models`를 probe해
  `serving` / `unreachable` / `serving_mismatch` 등으로 분류.
- **stream 강제**: UI-TARS는 일부 vLLM 버전에서 `stream=true`가 필요해 `force_stream=True`로 처리.
- 클라이언트는 모델명이 아니라 **route_slug**로 호출(예: `ui-venus`, `paddleocr-vl-1.5`) → 모델 교체에 무관.

## 6. 단일 거대 모델 대비 효율 (Kimi-K2 비교)

동일 H200 하드웨어에서 단일 프런티어 MoE 모델(Kimi-K2 계열, 1.04T 파라미터·토큰당 32B active)을
운영하는 경우와 비교.

| 지표 | 본 스택 (특화 5종) | Kimi-K2 (FP8) |
|------|-------------------|---------------|
| 필요 하드웨어 | **H200 2장** | **H200 약 8장** |
| GPU당 모델 밀도 | 2.5 모델/H200 | 0.125 모델/H200 |
| 가중치 풋프린트 | ~51 GiB | ~1,040 GiB |
| 레이턴시(짧은 JSON 출력) | ~80–150 tok/s/req | ~30–60 tok/s/req |

**핵심 메시지**: GUI grounding + OCR이라는 실제 필요 작업 표면(task surface)에 한정하면
**하드웨어 약 4배, 밀도·풋프린트 약 20배**를 절감한다. 포기하는 것은 범용 추론 능력이며,
이는 "우리가 실제로 읽어야 하는 화면"에 불필요한 범용성을 의도적으로 버린 트레이드오프다.
(근거: `docs/setup_vlms/05-resource-comparison-vs-kimi-k2.md`)

## 7. 확장성

- **6번째 서비스 추가**: `config/models/`에 `.env` + `flask_api/vlm_serve/`에 모듈 + `config.py` 등록.
  GPU 1에 ~50 GiB 헤드룸 보유.
- **multi-GPU scale-out**: UI-TARS는 data/tensor parallel 지원(`DATA_PARALLEL_SIZE` 등) — 처리량 확장 가능.
- **모델 variant**: `start_model.py ui-venus 30b` 형태의 family-size 네이밍으로 변형 모델 손쉽게 교체.
