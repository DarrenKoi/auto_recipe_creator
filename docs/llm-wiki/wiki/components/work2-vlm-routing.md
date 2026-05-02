---
tags: [component, vlm, flask-proxy, work2]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260312/20260312-vlm-proxy-pipeline.md,
  raw/journals/260312/20260312-shared-flask-vlm-config.md,
  raw/journals/260316/20260316-abort-code-server-forward-check.md,
  raw/journals/260318/260318_143312_ui-tars-1token-fix.md
]
---

# work2 VLM Routing

> `poc/work2` 자동화 스크립트가 coworker 공용 Flask proxy를 통해 VLM/OCR 서비스를 고르는 route 계층이다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)

## 왜 존재하는가? (Why)

- GUI automation script가 모델별 URL과 model name을 각자 들고 있으면 동료 PC에서 설정 차이가 커지므로, `flask_vlm.py`에 shared pipeline settings를 모아 route slug와 model name을 계산하도록 정리했다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)
- Flask API 쪽은 `/api/vlm_serve/{service}/health`, `/v1/models`, `/v1/chat/completions` 형태의 service template proxy를 제공하도록 구성되었다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- code-server forwarded URL은 browser session/cookie 의존성이 있어 script에서 안정적인 programmatic endpoint로 쓰기 어렵다고 판단했고, work2 direct probe 실험을 중단했다. (source: raw/journals/260316/20260316-abort-code-server-forward-check.md)

## 무엇인가? (What)

### 책임 범위

- `ui-venus`, `mai-ui`, `ui-tars`, `paddleocr-vl-1.5` 같은 service slug를 coworker-facing 이름으로 유지한다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- script가 사용할 Flask base URL, service proxy URL, model name, API key fallback을 계산한다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)
- live readiness 확인은 `connection_check.py`가 `/api/vlm_serve/health`와 service별 `/v1/models`를 점검하는 방식으로 분리한다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- browser-only forwarded URL 인증 재현은 이 컴포넌트의 책임이 아니다. (source: raw/journals/260316/20260316-abort-code-server-forward-check.md)
- UI-TARS 1 token 문제는 Flask proxy/SSE parsing 문제가 아니라 vLLM chat template 처리 문제로 진단되었으므로, route가 살아 있어도 model-specific serving 설정을 별도 확인해야 한다. (source: raw/journals/260318/260318_143312_ui-tars-1token-fix.md)

### 핵심 진입점

- `poc/work2/flask_vlm.py` — shared pipeline settings와 service URL/model resolver를 둔다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)
- `poc/work2/connection_check.py` — Flask health와 model route readiness를 확인한다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- `flask_api/vlm_serve/service_template.py` — upstream request/response 공통 proxy template를 담당했다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)

### 의존성

- 내부: [ocr-task-keyword-strategy.md](../concepts/ocr-task-keyword-strategy.md), [model-and-retrieval-options.md](../concepts/model-and-retrieval-options.md)
- 외부: OpenAI-compatible VLM/OCR server, Flask API gateway. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)

### 데이터 모델 / 인터페이스

```text
service slug -> Flask proxy URL -> OpenAI-compatible /v1/chat/completions
service slug -> /v1/models readiness check -> model name confirmation
```

## 어떻게 쓰는가? (How)

### 호출 예시

```powershell
uv run python poc/work2/connection_check.py
```

이 확인을 먼저 통과시킨 뒤 task script에서 필요한 service slug를 고정하는 흐름이 권장되었다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)

### 자주 쓰는 패턴

- 화면 grounding에는 primary VLM service를 쓰고, OCR-heavy 단계에는 `paddleocr-vl-1.5` service를 보조로 붙인다. (source: raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- coworker 기본 설정은 `.env` 파일 여러 개보다 `poc/work2/flask_vlm.py`의 shared settings를 먼저 본다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)
- forwarded URL 검증이 필요하면 cookie/session 인증 조건을 별도 실험으로 분리한다. (source: raw/journals/260316/20260316-abort-code-server-forward-check.md)

### 안티패턴

- browser에서 열린 forwarded URL을 Python `requests`에서 그대로 재현된다고 가정하지 않는다. (source: raw/journals/260316/20260316-abort-code-server-forward-check.md)
- script마다 service URL과 model name을 별도로 복사해 두지 않는다. (source: raw/journals/260312/20260312-shared-flask-vlm-config.md)

## 참고 자료 (References)

- 원본 메모: [20260312-vlm-proxy-pipeline.md](../../raw/journals/260312/20260312-vlm-proxy-pipeline.md)
- 원본 메모: [20260312-shared-flask-vlm-config.md](../../raw/journals/260312/20260312-shared-flask-vlm-config.md)
- 원본 메모: [20260316-abort-code-server-forward-check.md](../../raw/journals/260316/20260316-abort-code-server-forward-check.md)
- 원본 메모: [260318_143312_ui-tars-1token-fix.md](../../raw/journals/260318/260318_143312_ui-tars-1token-fix.md)
