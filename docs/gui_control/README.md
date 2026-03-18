# GUI Control 가이드

`docs/gui_control/`은 이제 기존에 `docs/gui_control/`과 `docs/research/`에 나뉘어 있던 GUI automation 설계 노트를 모아두는 공식 위치입니다.

이 폴더는 다음 내용을 다룰 때 사용합니다:

- Windows GUI 제어 전략
- GUI 모델 선택 및 hybrid automation 패턴
- `UI-Venus` + OCR + crop-retry grounding 가이드
- SEM/probe 워크플로우용 dynamic screen safety 규칙
- RCS video-to-action 계획

다음 항목은 `docs/setup_vlms/`를 사용합니다:

- vLLM 설치
- 모델 bring-up 및 runtime 설정
- UI-TARS 전용 runtime 요구사항
- OCR/parser 서비스 배포

## 권장 읽기 순서

1. [`01-foundations-and-tooling.md`](./01-foundations-and-tooling.md)
2. [`02-grounding-hybrid-patterns.md`](./02-grounding-hybrid-patterns.md)
3. [`03-ui-venus-ocr-crop-retry.md`](./03-ui-venus-ocr-crop-retry.md)
4. [`04-dynamic-screen-safety.md`](./04-dynamic-screen-safety.md)
5. [`05-rcs-video-to-action.md`](./05-rcs-video-to-action.md)

## Repo 기준 경로

- `poc/work2/`: 현재 GUI automation 실험과 coworker용 클라이언트 경로
- `poc/work2/flask_vlm.py`: `work2`에서 사용하는 서비스 registry
- `poc/work2/vlm_client.py`: service slug 기반 이미지 클라이언트
- `poc/work2/login_rcs.py`: 로그인 화면 캡처 / 좌표 추출 진입점
- `poc/work2/ocr_login_check.py`: OCR 프롬프트 및 응답 점검
- `test/video_frame_parser/`: offline 비디오 파싱 및 episode 추출
- `test/vlm_input_control/`: 이전 automation, retrieval, 프롬프트 실험

## 운영 원칙

- 고정 sleep 기반 매크로보다 `observe -> decide -> act -> verify`를 우선합니다.
- 실제 사무실 환경 검증에서 동작이 꼭 필요하지 않다면 `SAFE_MODE=true`를 유지합니다.
- 로컬 debug 스크린샷은 JPEG로 저장하고, 가능하면 VLM endpoint에는 WebP를 전송합니다.
- 정확한 텍스트는 OCR의 책임으로, 목표 선택은 GUI grounding의 책임으로 분리합니다.
- 비용이 낮고 구조적인 방법에서 시작하고, 필요할 때만 비용이 높고 유연한 방법으로 단계적으로 올립니다.
