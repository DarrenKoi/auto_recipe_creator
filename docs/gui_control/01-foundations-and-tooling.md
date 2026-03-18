# GUI Automation 기초와 도구

이 문서는 기존 overview, 라이브러리 조사, capability 메모, Microsoft automation tooling 메모, CPU PoC 요약을 이 저장소의 단일 시작점으로 합친 문서입니다.

## 1. 이 저장소가 자동화하려는 대상

대상 문제는 일반적인 desktop automation이 아닙니다. Windows 애플리케이션에서 수행되는 RCS/CD-SEM recipe 작업이며, 다음 요소가 함께 존재합니다:

- legacy Win32 계열 컨트롤
- 텍스트 비중이 높은 parameter panel
- 밀도가 높은 engineering layout
- custom drawing 되었거나 일부 접근이 어려운 UI 영역
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

- custom DirectX/OpenGL surface
- remote-rendered content
- accessibility 노출이 좋지 않은 컨트롤

### 2.2 Input-Simulation Control

대표 도구:

- `pynput`

적합한 경우:

- 최종 mouse/keyboard 실행
- drag, scroll, hotkey, fallback click
- UIA/object API가 실패하는 경우

취약한 경우:

- 무엇을 클릭해야 하는지 탐색하는 일
- background-safe 실행
- 강한 타입의 control semantic 처리

### 2.3 Vision-Led Control

대표 도구:

- `UI-Venus`
- `UI-TARS`
- `MAI-UI`
- `Kimi-K2.5` 같은 외부 baseline

적합한 경우:

- screenshot 이해
- icon/tab/button grounding
- selector가 없을 때 layout reasoning

취약한 경우:

- 정확한 문자열의 authoritative source 역할
- crop 또는 OCR 지원 없이 작은 고밀도 텍스트를 읽는 일
- verification 없는 safety-critical 실행

### 2.4 OCR 및 Parser Sidecar

대표 도구:

- `PaddleOCR-VL-1.5`
- `GOT-OCR-2.0-hf`
- `OmniParser V2`

적합한 경우:

- 정확한 텍스트
- table/grid 추출
- row/label anchor 추출
- 구조화된 UI element 힌트 제공

취약한 경우:

- 단독으로 최종 clickable surface를 결정하는 일
- 밀도 높은 engineering UI에서 유일한 planner 역할을 하는 일

## 3. 이 저장소의 권장 스택

| Layer | Preferred tools | Role |
| ------ | ----------------- | ------ |
| Capture | `mss`, OS window capture helper | 안정적인 화면 캡처 |
| Primary grounding | `UI-Venus` 또는 `UI-TARS` | 전체 화면 기준 target 선택 |
| Zoom-in grounding | `MAI-UI` | 작은 target / 혼잡한 crop 재시도 |
| Text authority | `PaddleOCR-VL-1.5`, `GOT-OCR-2.0-hf` | 정확한 텍스트, spotting, 어려운 OCR fallback |
| Structured parser | `OmniParser V2` | interactable box + SoM overlay |
| Object fallback | `pywinauto`, `uiautomation` | 접근 가능한 control 접근 |
| Execution | `pynput` | click, type, drag, hotkey |

## 4. 도구 선택 규칙

### 4.1 표준 Windows 컨트롤

화면이 안정적인 control을 노출하는 것이 확실하다면 먼저 object-based 접근을 시도합니다. 보통 VLM에 점 하나를 물어보는 것보다 더 빠르고 ambiguity도 적습니다.

### 4.2 Custom Graphics 또는 밀도 높은 Engineering UI

screenshot reasoning에 OCR 또는 parser sidecar를 조합하는 방식을 우선합니다. 많은 RCS 계열 화면에서 이것이 기본 operating mode입니다.

### 4.3 최종 Click 및 Type

의사결정 단계가 끝난 뒤에는 `pynput` 또는 기존 click/type helper를 사용합니다. 실행은 perception과 분리된 별도 layer로 취급합니다.

## 5. 알려진 한계

### 5.1 권한 및 OS 경계

- elevated/UAC context는 automation을 깨뜨릴 수 있습니다.
- remote desktop에서는 focus가 흔들릴 수 있습니다.
- remote session에서는 keyboard modifier가 눌린 상태로 남을 수 있습니다.

### 5.2 Custom UI Surface

custom-rendered control에서는 accessibility 기반 도구가 완전히 실패할 수 있습니다. 이 때문에 이 저장소는 vision-first 경로를 계속 유지합니다.

### 5.3 Network 및 Runtime 지연

screenshot 캡처 자체는 빠릅니다. 주요 비용은 model inference와 remote/runtime queue입니다. 모델 round-trip에 1~3초가 걸렸다면, 그 뒤에도 화면이 그대로라고 가정하면 안 됩니다.

### 5.4 Safety

산업용 GUI automation은 일반적인 office UI scripting처럼 다루면 안 됩니다. 일부 클릭은 실제 장비 이동이나 잘못된 recipe 변경을 유발할 수 있습니다.

## 6. 핵심 Automation 규칙

- `observe -> decide -> act -> verify`를 사용합니다.
- 기본값으로 `SAFE_MODE=true`를 유지합니다.
- 프롬프트, screenshot, 응답, 최종 action 좌표를 기록합니다.
- 아주 작은 target에 대해 full-screen 한 번만 보고 판단하지 않습니다.
- semantic choice와 exact text recovery를 분리합니다.
- high-risk 작업에는 human approval을 유지합니다.

## 7. CPU Baseline이 여전히 중요한 이유

이 저장소는 이미 유용한 Tier 1 패턴을 입증했습니다:

`capture -> optimize image -> call VLM -> parse action -> execute`

이 CPU/API baseline이 여전히 중요한 이유는 다음과 같습니다:

- GPU가 없는 환경에서도 fallback이 가능함
- local serving 개선 전에 benchmark baseline을 제공함
- 더 공격적인 local stack을 배포하기 전에 프롬프트와 action schema를 안전하게 검증할 수 있음

여전히 유효한 핵심 repo 패턴:

- 로컬 debug 이미지는 JPEG
- VLM 요청 payload는 WebP
- 명시적인 `SAFE_MODE`
- downstream action code를 위한 compact JSON 출력

## 8. 현재 Repo 기준 파일

- `poc/work2/open_rcs.py`
- `poc/work2/login_rcs.py`
- `poc/work2/login_benchmark.py`
- `poc/work2/ocr_login_check.py`
- `poc/work2/util/image_utils.py`
- `poc/work2/util/window_utils.py`
