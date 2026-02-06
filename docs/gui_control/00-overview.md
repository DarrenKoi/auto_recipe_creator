# GUI 제어 및 자동화 개요

## 1. GUI 자동화의 두 가지 접근 방식

GUI 자동화는 크게 **좌표 기반(Coordinate-based)** 방식과 **객체 기반(Object-based)** 방식으로 나뉩니다.

### 1.1 좌표 기반 자동화 (예: PyAutoGUI)
- **원리:** 화면의 특정 (x, y) 좌표를 지정하여 마우스 클릭이나 키보드 입력을 수행합니다.
- **장점:** 매우 직관적이고 설정이 간단하며, 거의 모든 애플리케이션에 적용 가능합니다.
- **단점:** 화면 해상도, 창 크기, UI 요소의 위치 변화에 매우 취약합니다. "이미지 매칭"을 병행하지 않으면 신뢰성이 낮습니다.

### 1.2 객체 기반 자동화 (예: PyWinAuto)
- **원리:** 운영체제(Windows API, UI Automation)가 인식하는 UI 트리에서 버튼, 텍스트 박스 등 **객체(Control)**를 직접 찾아 제어합니다.
- **장점:** 창의 위치가 바뀌거나 해상도가 변해도 객체의 ID나 이름을 통해 정확한 제어가 가능합니다. "클릭" 신호를 직접 보내므로 마우스 커서를 실제로 움직이지 않고도 작업할 수 있는 경우가 많습니다.
- **단점:** 학습 곡선이 높고, 특정 OS(주로 Windows)나 특정 프레임워크(Qt, WPF 등)에 종속적일 수 있습니다.

## 2. 핵심 제어 메커니즘

- **Mouse Control:** 클릭(Left/Right/Middle), 더블 클릭, 드래그 앤 드롭, 스크롤.
- **Keyboard Control:** 단순 타이핑, 조합키(Ctrl+C, Alt+Tab), 키 누르고 있기.
- **Screen Perception:** 현재 화면 상태 확인. (전통적인 픽셀 매칭 vs 현대적인 VLM 분석)
- **State Waiting:** 특정 버튼이 활성화될 때까지 기다리거나, 특정 창이 나타날 때까지 대기하는 로직.

## 3. ARC 프로젝트에서의 역할
ARC(Auto Recipe Creator)는 **VLM(Vision Language Model)**을 통해 화면을 "이해"하고, **GUI Automation**을 통해 "행동"합니다. RCS(Remote Control System)와 같은 복잡한 산업용 소프트웨어를 제어하기 위해서는 단순 좌표 클릭 이상의 정밀한 제어 전략이 필요합니다.

## 4. 문서 구조

이 디렉토리는 GUI 자동화 도구와 전략에 대한 포괄적인 문서를 제공합니다:

### 기본 개념
- **[00-overview.md](00-overview.md)** (현재 문서) - GUI 자동화 개요
- **[01-modern-libraries.md](01-modern-libraries.md)** - 주요 라이브러리 소개 (PyWinAuto, Pynput, VLM 등)
- **[02-capabilities-and-limitations.md](02-capabilities-and-limitations.md)** - 가능한 것과 한계점
- **[03-automation-strategies.md](03-automation-strategies.md)** - RCS 자동화 전략
- **[07-cpu-based-poc.md](07-cpu-based-poc.md)** - CPU 기반 PoC 및 ROI 분석 (Tier 1 전략)

### Microsoft 도구 (2025+)
- **[04-microsoft-vision-tools.md](04-microsoft-vision-tools.md)** - OmniParser, Florence-2, Phi-4 Vision
  - **핵심 내용**: Custom DirectX/OpenGL UI 탐지 솔루션
  - **적용 대상**: CD-SEM/VeritySEM 등 산업용 장비 화면

- **[05-microsoft-automation-ecosystem.md](05-microsoft-automation-ecosystem.md)** - WinAppDriver, Python-UIAutomation, OmniTool
  - **핵심 내용**: PyWinAuto 보완 도구
  - **도구 선택 가이드** 포함

- **[06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md)** - 실전 통합 패턴
  - **핵심 내용**: PyWinAuto + OmniParser + VLM 조합
  - **코드 예시**: RCS 로그인 하이브리드 구현
  - **성능 최적화**: 캐싱, Fallback 체인, 재시도 로직

### 권장 학습 순서

**초급 (현재 ARC 프로젝트 수준)**:
1. [00-overview.md](00-overview.md) - 기본 개념 이해
2. [01-modern-libraries.md](01-modern-libraries.md) - PyWinAuto, VLM 학습
3. [07-cpu-based-poc.md](07-cpu-based-poc.md) - 기술 검증 및 ROI 이해
4. [03-automation-strategies.md](03-automation-strategies.md) - RCS 적용 방법

**중급 (Custom Graphics 문제 해결)**:
4. [02-capabilities-and-limitations.md](02-capabilities-and-limitations.md) - Section 2.3 주목
5. [04-microsoft-vision-tools.md](04-microsoft-vision-tools.md) - OmniParser 도입
6. [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md) - 통합 패턴 구현

**고급 (자율 에이전트)**:
7. [05-microsoft-automation-ecosystem.md](05-microsoft-automation-ecosystem.md) - OmniTool 패턴 연구

### 빠른 참조

| 질문 | 참조 문서 | 섹션 |
|------|-----------|------|
| PyWinAuto가 Custom 버튼을 인식 못함 | [04-microsoft-vision-tools.md](04-microsoft-vision-tools.md) | Section 1.2, 8 |
| 여러 도구를 조합하고 싶음 | [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md) | Section 1, 2 |
| VLM vs OmniParser 비교 필요 | [04-microsoft-vision-tools.md](04-microsoft-vision-tools.md) | Section 4 |
| WinAppDriver가 PyWinAuto보다 나은가? | [05-microsoft-automation-ecosystem.md](05-microsoft-automation-ecosystem.md) | Section 1.5 |
| UI 요소 위치를 캐싱하고 싶음 | [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md) | Section 3 |
| 자동화 성공률을 높이고 싶음 | [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md) | Section 4, 5 |
