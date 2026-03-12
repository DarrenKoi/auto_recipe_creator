# Microsoft 자동화 생태계

Microsoft는 GUI 자동화를 위한 다양한 오픈소스 도구를 제공합니다. 이 문서는 OmniParser를 보완하는 추가 도구들을 소개합니다.

---

## 1. WinAppDriver

### 1.1 개요

**Windows Application Driver (WinAppDriver)**는 Microsoft가 공식 제공하는 Windows 애플리케이션 자동화 프레임워크입니다.

- **GitHub**: https://github.com/microsoft/WinAppDriver
- **라이선스**: MIT
- **프로토콜**: W3C WebDriver (Selenium과 동일한 표준)
- **지원 애플리케이션**: UWP, WinForms, WPF, Classic Win32

### 1.2 핵심 특징

**WebDriver 프로토콜 기반**:
- Selenium과 동일한 API 사용
- 웹 자동화 경험을 데스크톱 앱에 적용 가능
- 크로스 플랫폼 테스트 프레임워크(Appium)와 통합 가능

**공식 Microsoft 지원**:
- Windows 10/11에서 안정적으로 동작
- UWP 앱에 대한 네이티브 지원
- Accessibility API 기반으로 PyWinAuto와 유사한 접근 방식

### 1.3 설치 및 설정

```bash
# 1. WinAppDriver 다운로드 및 설치 (Windows만 가능)
# https://github.com/microsoft/WinAppDriver/releases

# 2. Python 클라이언트 설치
pip install Appium-Python-Client

# 3. WinAppDriver 서버 실행
# WinAppDriver.exe (기본 포트: 4723)
```

### 1.4 사용 예시

```python
from appium import webdriver

# WinAppDriver 연결
desired_caps = {
    "app": "C:\\Path\\To\\YourApp.exe",
    "platformName": "Windows",
    "deviceName": "WindowsPC"
}

driver = webdriver.Remote(
    command_executor='http://127.0.0.1:4723',
    desired_capabilities=desired_caps
)

# UI 요소 찾기 및 클릭
login_button = driver.find_element_by_name("Login")
login_button.click()

# 텍스트 입력
username_field = driver.find_element_by_accessibility_id("UsernameBox")
username_field.send_keys("admin")

driver.quit()
```

### 1.5 PyWinAuto와의 차이점

| 특성 | PyWinAuto | WinAppDriver | 비고 |
|------|-----------|--------------|------|
| **프로토콜** | Python API | WebDriver (REST) | WinAppDriver는 다언어 지원 |
| **학습 곡선** | 가파름 | 완만 (Selenium 유사) | Selenium 경험자에게 유리 |
| **CI/CD 통합** | 보통 | 쉬움 | WebDriver는 CI 도구와 잘 통합됨 |
| **지원 앱** | Win32, Qt, MFC | **UWP**, WinForms, WPF | UWP는 WinAppDriver가 유리 |
| **백그라운드 실행** | 가능 (일부) | 제한적 | PyWinAuto가 더 유연 |

### 1.6 권장 사용 시나리오

- **CI/CD 파이프라인**: Jenkins/GitHub Actions에서 Windows 앱 테스트
- **UWP 앱**: Store 앱이나 Modern UI 앱 자동화
- **크로스 플랫폼 팀**: 웹 QA 팀이 데스크톱 앱도 테스트해야 할 때
- **RCS 프로젝트**: ❌ (RCS는 Classic Win32 앱이므로 PyWinAuto가 더 적합)

---

## 2. Python-UIAutomation-for-Windows

### 2.1 개요

**Python-UIAutomation-for-Windows**는 Windows UI Automation API를 Python에서 직접 사용할 수 있게 해주는 래퍼입니다.

- **GitHub**: https://github.com/yinkaisheng/Python-UIAutomation-for-Windows
- **라이선스**: Apache 2.0
- **지원 프레임워크**: MFC, WindowsForms, WPF, Modern UI, Qt, IE, Firefox, Chrome

### 2.2 PyWinAuto와의 관계

PyWinAuto도 내부적으로 UI Automation API를 사용하지만, `Python-UIAutomation-for-Windows`는 더 **Low-level API 접근**을 제공합니다.

**차이점**:
- **PyWinAuto**: High-level API, 편의 기능 많음, 추상화 레벨 높음
- **Python-UIAutomation**: Raw API 노출, 세밀한 제어 가능, 학습 곡선 더 가파름

### 2.3 사용 예시

```python
import uiautomation as auto

# 창 찾기
window = auto.WindowControl(searchDepth=1, Name="RCS Login")

# UI 트리 출력 (디버깅용)
window.GetChildren()

# 버튼 찾기 (AutomationId 사용)
button = window.ButtonControl(AutomationId="LoginButton")
button.Click()

# 텍스트 입력
edit = window.EditControl(AutomationId="ServerAddress")
edit.SendKeys("192.168.1.100")

# 요소 속성 읽기
print(f"버튼 위치: {button.BoundingRectangle}")
print(f"활성화 여부: {button.IsEnabled}")
```

### 2.4 고급 기능

**UI 트리 탐색**:
```python
# 모든 자식 요소 재귀 탐색
def print_control_tree(control, depth=0):
    print("  " * depth + f"{control.ControlTypeName}: {control.Name}")
    for child in control.GetChildren():
        print_control_tree(child, depth + 1)

print_control_tree(window)
```

**이벤트 핸들러**:
```python
# 특정 창이 나타날 때까지 대기
def on_window_open(sender, event):
    print(f"창이 열렸습니다: {event.sender.Name}")

auto.Automation.AddAutomationEventHandler(
    auto.UIA_Window_WindowOpenedEventId,
    auto.TreeScope_Subtree,
    on_window_open
)
```

### 2.5 권장 사용 시나리오

- **PyWinAuto 한계 돌파**: PyWinAuto로 접근 불가능한 요소가 있을 때
- **성능 최적화**: PyWinAuto보다 약간 빠른 Raw API 호출
- **UI 트리 디버깅**: `print_control_tree`로 전체 구조 파악
- **RCS 프로젝트**: ⚠️ PyWinAuto로 충분하지만, 특수 케이스에서 고려

---

## 3. OmniTool (OmniParser 통합 프레임워크)

### 3.1 개요

**OmniTool**은 OmniParser를 기반으로 한 Windows 11 자동화 프레임워크입니다. Computer Use 에이전트 패턴을 구현합니다.

- **GitHub**: https://github.com/microsoft/OmniParser (examples/ 디렉토리)
- **라이선스**: MIT
- **지원 LLM**: GPT-4o, DeepSeek-V3, Qwen2.5-VL, Claude Sonnet

### 3.2 Computer Use 에이전트 패턴

OmniTool은 다음 루프를 반복하여 자동화를 수행합니다:

```
1. 화면 캡처
2. OmniParser로 UI 요소 탐지
3. LLM에 현재 상태 + 목표 전달
4. LLM이 다음 액션 결정 (클릭, 타이핑, 스크롤 등)
5. 액션 실행
6. 목표 달성 여부 확인 → 1번으로 돌아가거나 종료
```

### 3.3 아키텍처

```
User Goal ("RCS에 로그인해줘")
    ↓
┌───────────────────────────────┐
│   Orchestrator (LLM)          │
│   - Task planning             │
│   - Decision making           │
│   - Error recovery            │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│   OmniParser                  │
│   - Screen → UI elements      │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│   Action Executor             │
│   - Mouse/Keyboard control    │
└───────────────────────────────┘
```

### 3.4 ARC 프로젝트와의 관계

**유사점**:
- ARC도 VLM + GUI 제어 조합 사용
- 목표: 반복 작업 자동화

**차이점**:
- **OmniTool**: 범용 Windows 자동화 (VM 제어, 파일 조작 등)
- **ARC**: CD-SEM/VeritySEM recipe 설정 특화

**참고할 점**:
- OmniTool의 에러 처리 로직
- OmniParser 결과를 LLM에 전달하는 프롬프트 구조
- 액션 실패 시 재시도 전략

### 3.5 OmniTool 코드 예시 (간소화)

```python
# OmniTool 핵심 루프 (의사 코드)

def automate_task(goal: str, max_steps: int = 20):
    for step in range(max_steps):
        # 1. 화면 캡처
        screenshot = capture_screen()

        # 2. OmniParser 분석
        ui_elements = omniparser.parse(screenshot)

        # 3. LLM 호출
        prompt = f"""
        Goal: {goal}
        Current screen has these elements:
        {json.dumps(ui_elements)}

        What should I do next?
        Respond in format: {{"action": "click", "target": "Login button"}}
        """

        response = llm.generate(prompt)
        action = parse_action(response)

        # 4. 액션 실행
        execute_action(action, ui_elements)

        # 5. 목표 달성 확인
        if check_goal_achieved(goal, screenshot):
            print("목표 달성!")
            break
```

---

## 4. 도구 선택 가이드

### 4.1 의사결정 플로우차트

```
화면이 표준 Windows UI인가?
    ├─ Yes → PyWinAuto 사용
    │         ↓
    │     실패했는가?
    │         ├─ Yes → Python-UIAutomation 시도
    │         └─ No → 완료
    │
    └─ No (Custom Graphics) → OmniParser 사용
                ↓
            복잡한 추론이 필요한가?
                ├─ Yes → OmniParser + VLM (하이브리드)
                └─ No → OmniParser만 사용
```

### 4.2 도구별 권장 사용처

| 도구 | 최적 사용처 | ARC 프로젝트 적용 |
|------|-------------|-------------------|
| **PyWinAuto** | 표준 Windows 앱 (Win32, WPF, Qt) | ✅ RCS 로그인 (1차 시도) |
| **WinAppDriver** | UWP 앱, CI/CD 통합 | ❌ RCS는 UWP 아님 |
| **Python-UIAutomation** | PyWinAuto 실패 시, 세밀한 제어 | ⚠️ 예외 케이스에만 |
| **OmniParser** | Custom Graphics, DirectX/OpenGL UI | ✅ RCS Custom 버튼 (2차 Fallback) |
| **VLM (Qwen/Claude)** | 복잡한 의미 판단, 예외 상황 | ✅ Recipe 검증, 에러 해석 |
| **OmniTool** | 범용 자동화 프레임워크 | 📚 참고용 (패턴 학습) |

### 4.3 RCS 자동화 권장 스택

**현재 (2026-02-06)**:
```
PyWinAuto (1차) → VLM (2차 Fallback)
```

**제안 (OmniParser 도입 후)**:
```
PyWinAuto (1차) → OmniParser (2차) → VLM (3차 복잡한 판단)
```

**이점**:
- PyWinAuto 성공 시: 가장 빠름 (< 0.1초)
- Custom Graphics: OmniParser로 해결 (0.6초)
- 복잡한 예외: VLM으로 추론 (2-5초)

---

## 5. 추가 Microsoft 도구 (간략 소개)

### 5.1 Accessibility Insights for Windows

- **용도**: UI Automation 트리 디버깅
- **다운로드**: https://accessibilityinsights.io/
- **사용법**: PyWinAuto/WinAppDriver 개발 시 UI 요소 구조 파악

### 5.2 Inspect.exe (Windows SDK)

- **용도**: 실시간 UI 요소 속성 확인
- **위치**: Windows SDK에 포함 (`C:\Program Files (x86)\Windows Kits\10\bin\...`)
- **사용법**: 마우스로 UI 요소 위에 올리면 AutomationId, Name 등 표시

### 5.3 UI Recorder (Power Automate Desktop)

- **용도**: GUI 작업 녹화 및 자동 코드 생성
- **한계**: 코드 품질 낮음, 복잡한 로직 불가
- **권장**: 학습용으로만 사용, 프로덕션 코드는 수동 작성

---

## 6. 오픈소스 vs 상용 솔루션

ARC 프로젝트는 **오픈소스 우선** 정책을 따릅니다. 다음 상용 도구는 제외합니다:

### 제외 도구 (비오픈소스)

- ❌ **Microsoft Power Automate Desktop**: 무료지만 소스 비공개, 제한적 API
- ❌ **UiPath**: 상용 RPA 플랫폼 (오픈소스 아님)
- ❌ **Automation Anywhere**: 상용 RPA 플랫폼
- ❌ **Microsoft Copilot Studio**: 클라우드 종속, 비용 발생

### 선택 도구 (오픈소스 & 무료)

- ✅ **OmniParser**: MIT/AGPL (오픈소스)
- ✅ **WinAppDriver**: MIT (오픈소스)
- ✅ **Python-UIAutomation**: Apache 2.0 (오픈소스)
- ✅ **PyWinAuto**: BSD (오픈소스)
- ✅ **Florence-2**: MIT (오픈소스)
- ✅ **Phi-4**: MIT (오픈소스)

---

## 7. 학습 경로 추천

### 초급 (현재 ARC 프로젝트 수준)

1. **PyWinAuto 마스터**: `poc/work/` Windows 자동화 코드 완성도 높이기
2. **VLM 통합 최적화**: 프롬프트 엔지니어링, 응답 파싱 개선
3. **에러 처리 강화**: 재시도 로직, 타임아웃 설정

### 중급 (OmniParser 도입)

1. **OmniParser 설치 및 테스트**: GPU 환경 구축
2. **하이브리드 패턴 구현**: PyWinAuto + OmniParser Fallback
3. **UI 캐싱 시스템**: 성능 최적화

### 고급 (자율 에이전트)

1. **OmniTool 패턴 연구**: Computer Use 아키텍처 이해
2. **자율 에러 복구**: LLM이 에러를 자동으로 해결
3. **멀티 태스크 병렬화**: 여러 장비 동시 제어

---

## 8. 참고 자료

- **WinAppDriver GitHub**: https://github.com/microsoft/WinAppDriver
- **Python-UIAutomation GitHub**: https://github.com/yinkaisheng/Python-UIAutomation-for-Windows
- **Accessibility Insights**: https://accessibilityinsights.io/
- **OmniTool Examples**: https://github.com/microsoft/OmniParser/tree/main/examples

---

**이전 문서**: [04-microsoft-vision-tools.md](04-microsoft-vision-tools.md) - OmniParser 및 비전 도구
**다음 문서**: [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md) - 하이브리드 자동화 패턴
