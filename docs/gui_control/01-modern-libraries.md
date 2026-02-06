# 현대적인 GUI 자동화 라이브러리

PyAutoGUI는 훌륭한 시작점이지만, 복잡한 윈도우 애플리케이션 제어에는 한계가 있습니다. 현재 업계에서 널리 쓰이는 강력한 라이브러리들을 소개합니다.

## 1. PyWinAuto (Windows 전용)
**가장 강력한 객체 기반 제어 도구**입니다. Windows의 Win32 API와 UIA(UI Automation)를 활용합니다.

- **특징:** 창 제목, 클래스 이름, 컨트롤 ID 등으로 요소를 찾습니다.
- **장점:** 창이 다른 창에 가려져 있어도 제어가 가능한 경우가 많으며, 실제 마우스를 뺏지 않고도 백그라운드에서 동작할 수 있습니다.
- **용도:** RCS(Remote Control System)와 같은 표준 Windows 애플리케이션 제어.

## 2. Pynput (크로스 플랫폼)
**저수준(Low-level) 입력 제어 및 모니터링**에 특화되어 있습니다.

- **특징:** 마우스와 키보드의 모든 입력을 가로채거나(Listener), 가상으로 발생(Controller)시킵니다.
- **장점:** 매우 가볍고 빠르며, 전역 핫키(Global Hotkeys)를 만들기에 최적입니다.
- **용도:** 정밀한 마우스 드래그 조작이나 사용자 입력 가로채기.

## 3. Playwright / Selenium (웹 전용)
웹 기반 UI를 제어할 때 사용합니다.

- **특징:** 브라우저 엔진을 직접 제어합니다.
- **용도:** 장비 제어 시스템이 웹 인터페이스를 제공할 경우 최적의 선택입니다.

## 4. BetterCam / MSS (초고속 화면 캡처)
GUI 제어의 핵심은 "현재 화면을 보는 것"입니다.

- **PyAutoGUI.screenshot():** 초당 1~2프레임 정도로 느립니다.
- **MSS:** 초당 수십 프레임 캡처가 가능하여 실시간 분석에 유리합니다.
- **BetterCam (DXCAM):** Windows의 Desktop Duplication API를 사용하여 GPU 가속 캡처를 지원합니다 (가장 빠름).

## 5. VLM 기반 제어 (Next Generation)
이미지 매칭(OpenCV) 대신 **Qwen2-VL**이나 **GPT-4o** 같은 시각 언어 모델을 사용합니다.

- **특징:** "로그인 버튼을 눌러줘"라는 명령을 받으면 모델이 화면에서 버튼의 좌표를 추론합니다.
- **장점:** UI가 조금 바뀌어도 유연하게 대응할 수 있습니다.
- **단점:** 추론 속도가 상대적으로 느리고 비용이 발생할 수 있습니다.

## 6. Microsoft 비전 기반 도구 (2025+)

### 6.1 OmniParser
**Custom Graphics UI 요소 탐지 특화 도구**입니다. DirectX/OpenGL로 렌더링된 산업용 장비 화면도 분석 가능합니다.

- **목적**: Custom Graphics UI 요소 탐지 (DirectX/OpenGL 지원)
- **접근 방식**: YOLO + Florence-2 기반 비전 파싱
- **장점**: 구조화된 JSON 출력, 빠른 속도 (0.6초/프레임), 오프라인 동작
- **단점**: GPU 필요 (최소 8GB VRAM), 초기 셋업 복잡
- **용도**: CD-SEM/VeritySEM 등 산업용 장비 화면 분석
- **라이선스**: MIT (caption) + AGPL-3.0 (detection)
- **상세 문서**: [04-microsoft-vision-tools.md](04-microsoft-vision-tools.md)

### 6.2 WinAppDriver
**Windows 애플리케이션 자동화 프레임워크**입니다. WebDriver 프로토콜 기반으로 Selenium과 유사한 API를 제공합니다.

- **목적**: Windows 애플리케이션 자동화 (UWP, WinForms, WPF, Win32)
- **접근 방식**: W3C WebDriver 프로토콜
- **장점**: Selenium 경험 활용 가능, CI/CD 통합 용이
- **용도**: PyWinAuto 대안, 테스트 자동화
- **라이선스**: MIT
- **상세 문서**: [05-microsoft-automation-ecosystem.md](05-microsoft-automation-ecosystem.md)

### 6.3 도구 선택 가이드

| 상황 | 권장 도구 | 비고 |
|------|-----------|------|
| 표준 Windows UI | PyWinAuto | 가장 빠르고 안정적 |
| Custom DirectX/OpenGL UI | **OmniParser** | Custom Graphics 문제 해결 |
| 복잡한 의미 판단 필요 | VLM (Qwen/GPT-4V) | 느리지만 유연 |
| 하이브리드 접근 | PyWinAuto + OmniParser + VLM | 최고 성공률 (95%) |

자세한 통합 패턴은 [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md)를 참조하세요.
