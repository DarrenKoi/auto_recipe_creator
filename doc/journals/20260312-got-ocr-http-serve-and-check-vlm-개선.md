# 2026-03-12 GOT-OCR HTTP 서빙 및 check_vlm 개선

## 1. 진행 사항

- **`check_vlm.py` 전면 개편**: 기존에 `python check_vlm.py <base_url> <model_name>` 형태로 하나씩 확인하던 방식을 제거하고, `config/models/*.env` 파일을 자동 탐색하여 등록된 모든 VLM 서버의 생존 여부를 한 번에 확인하도록 변경
- **GOT-OCR-2.0-hf 모델 분석**: 다른 vLLM 기반 모델(8001~8004)과 달리 transformers 직접 추론 방식임을 확인. vLLM 미지원 모델이라 별도 서빙 전략 필요
- **`run_got_ocr.py` deprecated 파라미터 수정**: `torch_dtype=` → `dtype=` 변경 (transformers 최신 버전 호환)
- **`serve_got_ocr.py` 신규 작성**: GOT-OCR을 Flask HTTP 서버로 감싸서 다른 vLLM 서비스처럼 HTTP 요청으로 사용 가능하게 함 (포트 8005)
- **flask_api 프록시에 GOT-OCR 통합**: 기존 vLLM 프록시 패턴과 동일하게 `got-ocr` 서비스를 등록하여 `/api/vlm_serve/got-ocr/v1/ocr` 경로로 OCR 요청 가능
- **`got-ocr-2.0-hf.env`에 `PORT`, `SERVED_MODEL_NAME` 추가**: check_vlm.py 및 flask_api health probe에서 자동 탐지 가능하도록 설정

## 2. 수정 내용

### 수정된 파일
| 파일 경로 | 변경 내용 |
|-----------|-----------|
| `docs/deploy_vlms/scripts/check_vlm.py` | 전면 재작성 — `discover_models()`, `check_model()` 함수 기반 일괄 확인 방식으로 변경 |
| `docs/deploy_vlms/scripts/run_got_ocr.py` | `torch_dtype=` → `dtype=` 수정 (line 145) |
| `docs/deploy_vlms/config/models/got-ocr-2.0-hf.env` | `SERVED_MODEL_NAME=got-ocr-2.0-hf`, `PORT=8005` 추가 |
| `flask_api/vlm_serve/config.py` | `ALL_VLM_SERVICES`에 GOT-OCR 엔트리 추가 (port 8005, enabled) |
| `flask_api/vlm_serve/__init__.py` | `got_ocr` 모듈 import 및 `_ALL_SERVICE_BLUEPRINTS`에 등록 |

### 신규 파일
| 파일 경로 | 설명 |
|-----------|------|
| `docs/deploy_vlms/scripts/serve_got_ocr.py` | GOT-OCR Flask HTTP 서버 (모델 로드 → `/v1/models`, `/v1/ocr` 엔드포인트 제공) |
| `flask_api/vlm_serve/got_ocr.py` | flask_api 프록시 서비스 모듈 (route_slug: `got-ocr`, upstream_port: 8005) |

### 아키텍처 변경
```
serve_got_ocr.py (port 8005)     ← 모델 로드, Flask 서버
    GET  /v1/models              ← 헬스 체크 (vLLM 호환)
    POST /v1/ocr                 ← OCR 추론 (base64 이미지)
        ↑
flask_api proxy                  ← 요청 전달
    POST /api/vlm_serve/got-ocr/v1/ocr
    GET  /api/vlm_serve/got-ocr/health
```

## 3. 다음 단계

- GPU 서버에서 `serve_got_ocr.py` 실행 테스트 (모델 로드 및 OCR 추론 확인)
- `check_vlm.py` 실행하여 5개 모델(8001~8005) 일괄 상태 확인 동작 검증
- flask_api 프록시를 통한 GOT-OCR `/v1/ocr` 엔드포인트 통합 테스트

## 4. 메모리 업데이트

변경 없음
