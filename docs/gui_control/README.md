# GUI Control 가이드

`docs/gui_control/`은 현재 저장소 기준 GUI automation 운영 노트를 모아두는 위치입니다.
특히 `poc/work2/`의 Flask proxy 기반 VLM/OCR 흐름, Windows RCS 창 캡처, 안전한 read-only 분석 절차를 설명할 때 이 폴더를 기준으로 삼습니다.

이 폴더는 다음 내용을 다룹니다:

- Windows GUI 제어 전략과 도구 선택 기준
- `poc/work2/`의 current mainline 흐름
- `UI-Venus`/`UI-TARS`/direct GUI 모델과 OCR sidecar 조합 방식
- dynamic screen safety 규칙
- RCS video-to-action 장기 계획

다음 항목은 계속 `docs/setup_vlms/`를 사용합니다:

- vLLM 및 런타임 설치
- 모델 bring-up, serving, 배포 설정
- OCR/parser 서비스 배포

## 권장 읽기 순서

1. [`01-foundations-and-tooling.md`](./01-foundations-and-tooling.md)
2. [`02-grounding-hybrid-patterns.md`](./02-grounding-hybrid-patterns.md)
3. [`03-ui-venus-ocr-crop-retry.md`](./03-ui-venus-ocr-crop-retry.md)
4. [`04-dynamic-screen-safety.md`](./04-dynamic-screen-safety.md)
5. [`05-rcs-video-to-action.md`](./05-rcs-video-to-action.md)
6. [`06-view-tab-embedded-align-fail-monitoring.md`](./06-view-tab-embedded-align-fail-monitoring.md)
7. [`07-window-overlap-and-hidden-title-strategy.md`](./07-window-overlap-and-hidden-title-strategy.md)
8. [`08-workflow-engine-and-retry-strategy.md`](./08-workflow-engine-and-retry-strategy.md)

## 현재 권장 실행 순서

1. `uv run python poc/work2/connection_check.py`
2. `uv run python poc/work2/open_rcs.py`
3. `uv run python poc/work2/login_rcs.py`
4. 필요 시 `uv run python poc/work2/ocr_login_check.py`

설명:

- `connection_check.py`는 Flask `/api/vlm_serve/health` 와 각 서비스의 `/v1/models` 연결 상태를 먼저 확인합니다.
- `open_rcs.py`는 `RcsMainHD.exe` 실행만 담당합니다.
- `login_rcs.py`는 로그인 창을 찾고 캡처한 뒤, 현재 primary GUI 서비스들에 동일한 계약으로 분석을 요청해 overlay와 raw 응답을 남깁니다.
- `ocr_login_check.py`는 OCR 전용 비교 실험입니다. 텍스트 추출 품질과 위치 힌트 가능성을 점검할 때만 별도로 사용합니다.

## Repo 기준 경로

- `poc/work2/`: 현재 GUI automation 실험과 coworker용 클라이언트 경로
- `poc/work2/flask_vlm.py`: 서비스 slug, 모델명, endpoint, purpose default의 단일 source of truth
- `poc/work2/connection_check.py`: Flask health 및 서비스별 `/v1/models` 점검
- `poc/work2/vlm_client.py`: service slug 기반 OpenAI-compatible 이미지 클라이언트
- `poc/work2/open_rcs.py`: RCS 실행 전용 진입점
- `poc/work2/login_rcs.py`: 로그인 창 read-only 캡처 및 multi-model benchmark 진입점
- `poc/work2/login_benchmark.py`: 동일 screenshot을 여러 GUI 서비스에 보내는 공통 비교 로직
- `poc/work2/ocr_login_check.py`: PaddleOCR-VL/GOT-OCR 비교 스크립트
- `poc/work2/prompts/`: 로그인, 메인 탭, OCR assist, screen analysis 프롬프트 빌더
- `poc/work2/util/`: window/image/debug/time/json helper

참고:

- `login_rcs_ui_venus.py`, `login_rcs_ui_tars.py`, `*_rev2.py` 계열은 비교/실험 스크립트로 남아 있습니다.
- 현재 문서 기준 mainline entrypoint는 `login_rcs.py` 입니다.

## 운영 원칙

- `observe -> decide -> act -> verify`를 기본 루프로 유지합니다.
- 현재 `poc/work2` mainline은 read-only 분석을 먼저 강화하고, 실제 입력은 opt-in 실험으로 분리합니다.
- 서비스 선택과 endpoint 매핑은 `poc/work2/flask_vlm.py`에서만 관리합니다.
- 로컬 debug 스크린샷은 JPEG로 저장하고, VLM 요청 payload는 WebP를 우선합니다.
- 정확한 텍스트는 OCR sidecar의 책임으로, click target 선택은 GUI grounding 모델의 책임으로 분리합니다.
- direct 회사 모델과 Flask proxy 모델을 같은 benchmark 계약으로 비교하되, 연결 상태 확인과 artifact 저장 형식은 통일합니다.
