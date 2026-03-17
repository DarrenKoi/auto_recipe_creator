# Flask VLM Coworker Setup Map

이 문서는 동료 소개용 발표를 만들 때 `docs/setup_vlms/` 안의 운영 문서를 어떤 순서로 읽으면 되는지 정리한 안내서다.
목표는 "배치 구조 -> Flask proxy 연결 -> 모델별 서빙 차이 -> 운영 제약" 순서로 읽게 만드는 것이다.

## 1. 가장 먼저 읽을 문서

1. [`README.md`](./README.md)
   전체 포트 정책, 배치 모델, coworker-facing Flask proxy 주소 정책을 한 번에 설명한다.
2. [`01-layout-and-settings.md`](./01-layout-and-settings.md)
   `deploy_vlms/config/common.env`, `config/models/*.env`, 포트와 GPU 배치 규칙을 정리한다.
3. [`03-operations-and-repo-integration.md`](./03-operations-and-repo-integration.md)
   이 저장소의 `poc/work2`와 서버 배포를 실제로 어떻게 연결하는지 보여준다.

## 2. 모델 bring-up용 문서

- [`02-serve-ui-venus-and-mai-ui.md`](./02-serve-ui-venus-and-mai-ui.md)
  초기 GUI 모델 bring-up 절차.
- [`05-ui-tars-vs-others.md`](./05-ui-tars-vs-others.md)
  `UI-TARS`가 왜 runtime 민감도가 더 높은지 설명한다.
- [`07-paddleocr-vl-1.5.md`](./07-paddleocr-vl-1.5.md)
  `PaddleOCR-VL-1.5`를 기존 `vLLM` 체계에 편입하는 방법.
- [`08-got-ocr-2.0-hf.md`](./08-got-ocr-2.0-hf.md)
  `GOT-OCR-2.0-hf`를 별도 `transformers` 경로로 다루는 이유와 테스트 절차.

## 3. 운영 제약 / 확장 문서

- [`04-offline-and-network-policy.md`](./04-offline-and-network-policy.md)
  private cloud / offline 정책을 설명할 때 참고한다.
- [`10-system-ram-and-vllm.md`](./10-system-ram-and-vllm.md)
  GPU VRAM 외에 host RAM이 왜 중요한지 설명할 때 참고한다.
- [`06-multi-size-variants.md`](./06-multi-size-variants.md)
  size variant 비교가 필요할 때만 본다.
- [`09-omniparser-v2.md`](./09-omniparser-v2.md)
  future sidecar 확장안이 필요할 때만 본다.

## 4. coworker-facing 코드 앵커

- [`poc/work2/flask_vlm.py`](../../poc/work2/flask_vlm.py)
  service slug, model name, endpoint mapping의 source of truth.
- [`poc/work2/connection_check.py`](../../poc/work2/connection_check.py)
  실제 살아 있는 서비스를 먼저 확인하는 진입점.
- [`poc/work2/vlm_client.py`](../../poc/work2/vlm_client.py)
  이미지 포함 요청을 보내는 기본 클라이언트.

## 5. 발표용 핵심 메시지

- coworkers 에게는 direct port보다 Flask proxy route를 주는 편이 단순하다.
- `ui-venus`, `mai-ui`, `ui-tars`, `paddleocr-vl-1.5`는 OpenAI-compatible `/v1/*` 흐름으로 묶인다.
- `got-ocr`는 현재 `/v1/ocr` 전용 endpoint라는 점을 분리해서 설명해야 한다.
- `poc/work2`는 서버 내부 `flask_api` 코드를 import 하지 않는 client-side 독립 경로여야 한다.
