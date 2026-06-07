---
status: accepted
---

# VLM 호출은 Flask proxy 경유를 기본으로 하고, 회사 LLM 게이트웨이만 direct 로 둔다

## 결정

`flask_vlm.py` 의 `VLMServiceEntry` 레지스트리로 모델별 연결을 정의하고, 두 가지 `connection_mode`
를 둔다.

- **`proxy`** (기본) — Flask VLM proxy 경유. UI/OCR 모델: `ui-venus-1.5-8b`(주 화면분석),
  `mai-ui-8b`(정밀 좌표), `paddleocr-vl-1.5`(OCR 확인), `got-ocr`.
  - URL 패턴: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`
  - 통합 health 발견: `GET /api/vlm_serve/health`
- **`direct`** — 회사 LLM 게이트웨이(`http://common.llm.skhynix.com/v1`)에 직접: `Kimi-K2.5`,
  `Qwen3-VL-30B-Instruct`. API 키는 `COMMON_LLM_API_KEY` env.

해석 헬퍼: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`.

## 맥락 / 이유

- 여러 VLM(ui-venus/mai-ui/paddleocr/got-ocr)을 **하나의 OpenAI 호환 인터페이스** 로 부르고 싶다.
  Flask proxy 가 라우팅·health 발견을 통합하므로 클라이언트는 slug 만 알면 된다.
- proxy 모델은 **PC별 `.env` 가 필요 없다** — 기본 엔드포인트가 하드코딩되어 있어 오피스 어느 PC 에서나
  바로 동작한다.
- 반면 회사 LLM(Kimi/Qwen)은 게이트웨이가 따로 있고 키 인증이 필요하므로 **direct** 가 자연스럽다.

## 결과 (Consequences)

- 클라이언트(`Workflow1VLMClient`)는 `service_slug` 만으로 URL/키/모델명을 해석한다
  (`resolve_*` 헬퍼). mode 에 따라 proxy URL 을 조립하거나 service URL 에 직결한다.
- 모델을 추가하려면 레지스트리에 `VLMServiceEntry` 한 줄을 더하면 된다. 비활성화는 `enabled=False`(ui-tars 처럼).
- 호출은 `temperature=0.0`, `max_tokens=4096` 기본. 이미지는 WebP(q=90) `data:` URL 첨부.
- 모든 호출은 `log_vlm_call()` 로 `vlm_calls.log` 에 감사 기록(지연·토큰·status). →
  `0003-print-logging-with-rotating-audit.md`.
- 응답 파싱은 깨진 JSON 도 견디며(`extract_json`), 스트리밍 SSE 도 처리한다.

## 운영 메모

- 등록 서비스/포트: ui-venus(8001), mai-ui(8002), ui-tars(8003, disabled), paddleocr-vl-1.5(8004),
  got-ocr(8005). 서버 측 레지스트리는 `flask_api/vlm_serve/config.py`.
- 디버그 이미지는 모델별 폴더 `debug_images/<model-slug>/` 로 분리(`resolve_debug_model_name`).
