# GUI Automation 기초와 도구

이 문서는 `poc/work2/` 기준 현재 GUI automation baseline을 설명합니다.
특히 "지금 이 저장소에서 바로 돌릴 수 있는 흐름"과 "여전히 유효하지만 아직 mainline이 아닌 패턴"을 구분해서 정리합니다.

## 1. 이 저장소가 자동화하려는 대상

대상 문제는 일반적인 desktop automation이 아닙니다.
Windows 애플리케이션에서 수행되는 RCS/CD-SEM recipe 작업이며, 다음 특성이 함께 존재합니다:

- legacy Win32 계열 컨트롤
- 텍스트 비중이 높은 parameter panel
- 밀도가 높은 engineering layout
- custom drawing 되었거나 accessibility 노출이 약한 UI
- remote session 지연과 focus 불안정성

이 조합에서는 하나의 automation 방식만으로는 충분하지 않습니다.

## 2. 제어 방식

### 2.1 Object-Based Control

대표 도구:

- `pywinauto`
- `uiautomation`
- `WinAppDriver`

적합한 경우:

- 표준 text field
- accessible name 또는 automation ID가 있는 버튼
- 결정적인 dialog와 menu

취약한 경우:

- custom-rendered UI
- remote-rendered content
- accessibility 노출이 약한 화면

현재 repo 기준:

- 보조 fallback 으로는 여전히 유효합니다.
- 하지만 `poc/work2` mainline은 object API보다 screenshot 기반 확인 절차를 더 많이 사용합니다.

### 2.2 Input-Simulation Control

대표 도구:

- `pynput`

적합한 경우:

- 최종 mouse/keyboard 실행
- drag, scroll, hotkey, fallback click
- object API가 실패하는 경우의 마지막 실행 수단

취약한 경우:

- 무엇을 클릭해야 하는지 탐색하는 일
- stale screenshot 이후 지연된 action
- 안전 검증 없이 바로 실행하는 구조

현재 repo 기준:

- 실행 레이어로는 계속 유효합니다.
- 다만 현재 mainline entrypoint인 `open_rcs.py`, `login_rcs.py`는 read-only 또는 launch-only 성격이 강합니다.

### 2.3 Vision-Led Control

현재 repo에서 비교 또는 사용할 수 있는 주요 GUI 서비스:

- `ui-venus`
- `ui-tars`
- `mai-ui`
- `kimi-k2.5`
- `qwen3-vl-30b-instruct`

적합한 경우:

- screenshot 이해
- icon/tab/button grounding
- selector가 없을 때 layout reasoning
- 동일 screenshot에 대한 서비스 간 benchmark

취약한 경우:

- 정확한 텍스트의 authoritative source 역할
- verification 없이 바로 실행하는 safety-critical action
- 작은 crop의 정밀 판독을 OCR 없이 단독 처리하는 일

### 2.4 OCR 및 Parser Sidecar

현재 repo에서 쓰는 OCR 서비스:

- `paddleocr-vl-1.5`
- `got-ocr`

적합한 경우:

- 정확한 텍스트
- label/value 확인
- 어려운 crop 재판독
- text anchor 추출

취약한 경우:

- 최종 click target을 단독으로 결정하는 일
- GUI planner 역할 전체를 대신하는 일

참고:

- 현재 repo에는 `OmniParser` mainline 경로가 합쳐져 있지 않습니다.
- parser/SoM 계열은 여전히 유망하지만, 현재 문서는 "도입 후보"로 취급합니다.

## 3. 현재 저장소의 권장 스택

| Layer | Current choice | Role |
| ------ | -------------- | ---- |
| Shared registry | `poc/work2/flask_vlm.py` | 서비스 slug, 모델명, endpoint, default 역할 관리 |
| Connectivity check | `poc/work2/connection_check.py` | Flask health, proxy route, direct/proxy 연결 상태 확인 |
| Capture | `poc/work2/util/window_utils.py`, `poc/work2/util/image_utils.py` | 로그인 창 탐색, foreground, screenshot 캡처 |
| Primary GUI grounding default | `ui-venus` | 현재 `screen_analysis`, `main_tabs` 기본 서비스 |
| Alternative GUI services | `ui-tars`, `kimi-k2.5`, `qwen3-vl-30b-instruct` | 동일 계약 benchmark 및 비교 |
| Zoom-in sidecar candidate | `mai-ui` | crop-retry를 붙일 때 실험적으로 사용 |
| Text authority | `paddleocr-vl-1.5` | 현재 기본 OCR 서비스 |
| OCR fallback | `got-ocr` | crop 또는 특정 영역 재판독 |
| Client contract | `poc/work2/vlm_client.py` | service slug 기반 OpenAI-compatible 호출 |
| Debug/logging | `poc/work2/logger.py`, `poc/work2/debug_images/` | raw 응답, overlay, latency, 토큰 사용량 기록 |

현재 purpose default:

- `screen_analysis`: `ui-venus`
- `main_tabs`: `ui-venus`
- `ocr`: `paddleocr-vl-1.5`

위 기본값은 모두 `poc/work2/flask_vlm.py`에서만 바꿉니다.

## 4. 도구 선택 규칙

### 4.1 먼저 연결 상태를 확인한다

automation prompt를 디버깅하기 전에 아래 순서를 먼저 지킵니다:

1. `connection_check.py`
2. `open_rcs.py`
3. `login_rcs.py`

서비스가 안 떠 있는데 prompt를 손보는 식의 디버깅은 시간을 낭비합니다.

### 4.2 GUI grounding과 OCR은 역할을 분리한다

- 어떤 control을 눌러야 하는지: GUI grounding 서비스
- 어떤 text가 실제로 보이는지: OCR 서비스

현재 `ocr_login_check.py`도 이 원칙을 전제로 설계되어 있습니다.

### 4.3 현재 mainline은 benchmark-first다

`login_rcs.py`는 한 서비스만 맹신하지 않습니다.
현재 기본은 `primary_gui` 역할 서비스들을 같은 screenshot으로 비교하는 방식입니다.

이 구조의 장점:

- 서비스 간 성능 차이를 artifact로 바로 비교 가능
- direct 회사 모델과 Flask proxy 모델을 같은 JSON 계약으로 묶을 수 있음
- 한 모델이 불안정해도 다른 모델을 바로 비교 기준으로 삼을 수 있음

### 4.4 Crop-retry는 유효하지만 아직 기본 경로는 아니다

밀도 높은 engineering GUI에서 crop-retry는 여전히 중요합니다.
다만 현재 repo mainline은 먼저 full-screen capture + benchmark + OCR sidecar까지를 안정화하는 단계입니다.

즉:

- 개념적으로는 권장
- 구현상으로는 아직 별도 실험 또는 다음 단계 작업

## 5. 알려진 한계

### 5.1 macOS 개발, Windows 실사용

- 개발 보조 환경은 macOS입니다.
- 실제 RCS 창 탐색과 pywinauto 계열 검증은 사무실 Windows에서 확인해야 합니다.

### 5.2 Direct API와 Flask proxy가 공존한다

- `ui-venus`, `ui-tars`, `mai-ui`, OCR 서비스는 Flask proxy 경로를 사용합니다.
- `kimi-k2.5`, `qwen3-vl-30b-instruct`는 회사 direct API를 사용합니다.
- 그래서 `connection_check.py`는 direct 모델에 대해 `/models` probe를 강제하지 않고 skip 처리합니다.

### 5.3 Screenshot 이후 지연은 항상 위험하다

screenshot 캡처 자체보다 model round-trip이 더 오래 걸립니다.
특히 remote session에서는 "조금 전 화면과 지금 화면이 같다"는 가정을 두면 안 됩니다.

### 5.4 현재 mainline은 고위험 action을 다루지 않는다

현재 `open_rcs.py`는 launch-only, `login_rcs.py`는 read-only 분석 중심입니다.
이것은 한계이면서 동시에 의도된 안전장치입니다.

## 6. 핵심 Automation 규칙

- `observe -> decide -> act -> verify`를 유지합니다.
- 먼저 read-only 분석 경로를 안정화하고, 그 다음 action을 붙입니다.
- prompt, JPEG source, WebP payload, raw response, parsed JSON, overlay를 남깁니다.
- service slug, endpoint, model 선택은 개별 스크립트마다 흩뿌리지 않고 `flask_vlm.py`에만 둡니다.
- OCR 결과는 text evidence로 쓰고, 최종 click authority로 과대해석하지 않습니다.
- 고위험 작업에는 human approval을 유지합니다.

## 7. 현재 baseline이 중요한 이유

현재 저장소가 안정적으로 제공하는 baseline은 다음과 같습니다:

`registry -> connection check -> window capture -> multi-model benchmark -> artifact review`

이 baseline이 중요한 이유:

- coworker가 `poc/work2`만으로 동일한 연결 설정을 재현할 수 있음
- prompt 변경과 service 변경을 artifact 기준으로 비교할 수 있음
- action을 붙이기 전에 perception 계약을 먼저 다듬을 수 있음

## 8. 현재 Repo 기준 핵심 파일

- `poc/work2/flask_vlm.py`
- `poc/work2/connection_check.py`
- `poc/work2/vlm_client.py`
- `poc/work2/open_rcs.py`
- `poc/work2/login_rcs.py`
- `poc/work2/login_benchmark.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/prompts/prompt_login_rcs.py`
- `poc/work2/prompts/prompt_login_rcs_ui_venus.py`
- `poc/work2/prompts/prompt_login_rcs_ui_tars.py`
- `poc/work2/prompts/prompt_ocr_assist.py`
- `poc/work2/prompts/prompt_rcs_main_tabs.py`
- `poc/work2/prompts/prompt_screen_analysis.py`
- `poc/work2/util/image_utils.py`
- `poc/work2/util/window_utils.py`
