# Microsoft 비전 기반 UI 탐지 도구

## 1. OmniParser V2.0 소개

### 1.1 개요 및 핵심 기능

**OmniParser V2.0**은 Microsoft가 2025년 2월에 출시한 순수 비전 기반 GUI 파싱 시스템입니다. 기존의 객체 기반 자동화(PyWinAuto 등)가 해결하지 못했던 **Custom Graphics 문제**를 정면으로 해결합니다.

- **GitHub**: https://github.com/microsoft/OmniParser
- **라이선스**: MIT (caption 모델) + AGPL-3.0 (detection 모델)
- **핵심 기능**:
  - UI 요소 자동 탐지 (버튼, 입력 필드, 아이콘 등)
  - 바운딩 박스 + 의미론적 설명 생성
  - OCR 통합
  - Interactability 예측 (클릭 가능 여부 판단)

### 1.2 Custom Graphics 문제 해결 방법

[02-capabilities-and-limitations.md](02-capabilities-and-limitations.md)의 Section 2.3에서 언급된 바와 같이, DirectX나 OpenGL로 렌더링된 UI는 Windows UI Automation API가 인식하지 못합니다. 이는 산업용 장비 제어 소프트웨어(CD-SEM, VeritySEM 등)에서 흔히 발생하는 문제입니다.

**OmniParser의 해결책**:
- Windows API에 의존하지 않음
- 화면 이미지 자체를 분석하여 UI 요소 탐지
- Custom DirectX/OpenGL 렌더링에도 동작
- 구조화된 JSON 출력으로 프로그래밍 가능한 자동화 지원

### 1.3 기술 스택

OmniParser는 두 가지 핵심 AI 모델을 결합합니다:

1. **YOLO (Detection Model)**
   - UI 요소의 바운딩 박스 탐지
   - Icon, button, input field 등 카테고리 분류

2. **Florence-2 (Microsoft Vision Foundation Model)**
   - 탐지된 요소의 의미론적 설명 생성 (captioning)
   - OCR 기능 통합
   - 경량 모델 (0.23B/0.77B 파라미터)

### 1.4 성능 벤치마크

| 메트릭 | OmniParser V1.0 | OmniParser V2.0 | 개선폭 |
|--------|-----------------|-----------------|--------|
| 처리 속도 (A100 GPU) | 1.5초/프레임 | **0.6초/프레임** | 60% 향상 |
| Detection Precision | 88.6% | **91.6%** | +3.0% |
| Detection Recall | 82.8% | **89.4%** | +6.6% |
| Icon Caption Accuracy | 72.3% | **78.2%** | +5.9% |

*출처: OmniParser V2.0 Technical Report (2025)*

---

## 2. 핵심 기능

### 2.1 UI 요소 탐지 및 바운딩 박스 생성

OmniParser는 화면 이미지를 입력받아 다음 정보를 포함하는 UI 요소 리스트를 반환합니다:

- **바운딩 박스 좌표** (x1, y1, x2, y2)
- **요소 타입** (icon, button, input_field, text, image, dropdown 등)
- **신뢰도 점수** (0.0 ~ 1.0)

### 2.2 의미론적 요소 설명 (Icon Caption)

단순히 "버튼"이라고 분류하는 것이 아니라, Florence-2 모델을 통해 **해당 버튼이 무엇을 하는 버튼인지** 설명합니다.

예시:
- "Save button with floppy disk icon"
- "Settings gear icon"
- "Blue circular refresh button"

### 2.3 OCR 통합

텍스트가 포함된 UI 요소에 대해 자동으로 OCR을 수행하여 텍스트 내용을 추출합니다. 별도의 Tesseract나 EasyOCR 설치 불필요.

### 2.4 Interactability Prediction

각 UI 요소가 클릭 가능한지 여부를 예측합니다. 이를 통해 단순 표시용 텍스트와 실제 버튼을 구분할 수 있습니다.

---

## 3. 출력 형식 및 사용 예시

### 3.1 JSON 구조화 출력

```json
{
  "elements": [
    {
      "bbox": [120, 450, 280, 510],
      "type": "button",
      "caption": "Login button with key icon",
      "text": "Login",
      "confidence": 0.94,
      "interactable": true
    },
    {
      "bbox": [300, 200, 600, 240],
      "type": "input_field",
      "caption": "Username input box",
      "text": "",
      "confidence": 0.89,
      "interactable": true
    },
    {
      "bbox": [50, 50, 200, 100],
      "type": "icon",
      "caption": "Company logo",
      "text": "",
      "confidence": 0.97,
      "interactable": false
    }
  ],
  "screen_size": [1920, 1080],
  "processing_time": 0.62
}
```

### 3.2 ARC 프로젝트 통합 시나리오

기존 `vlm_screen_analysis.py`와 결합하여 하이브리드 접근이 가능합니다:

**시나리오 1: 빠른 UI 요소 위치 파악**
```python
# 1. OmniParser로 화면 구조 파악 (0.6초)
elements = omniparser.analyze_screen(screenshot)
login_btn = find_element_by_text(elements, "Login")

# 2. 좌표로 직접 클릭
mouse_controller.click(login_btn.center_x, login_btn.center_y)
```

**시나리오 2: VLM과 조합한 의미론적 판단**
```python
# 1. OmniParser로 구조화된 UI 정보 추출
elements = omniparser.analyze_screen(screenshot)

# 2. VLM에 구조화된 정보 제공하여 복잡한 판단 수행
vlm_prompt = f"""
다음 UI 요소들이 탐지되었습니다:
{json.dumps(elements, ensure_ascii=False)}

Recipe 설정이 올바른지 확인하고, 다음 클릭해야 할 버튼을 알려주세요.
"""

# 3. VLM의 추론 결과로 액션 결정
action = vlm_analyzer.analyze(screenshot, vlm_prompt)
```

### 3.3 RCS 화면 분석 예시

RCS(Remote Control System) 로그인 화면에서:

```python
from test.vlm_input_control import ScreenCapture
from test.vlm_input_control import MouseController

# 1. 화면 캡처
screen = ScreenCapture()
image = screen.capture_full_screen()

# 2. OmniParser 분석
elements = omniparser.analyze_screen(image)

# 3. "Server" 입력 필드 찾기
server_field = next(
    (e for e in elements if "server" in e.caption.lower() and e.type == "input_field"),
    None
)

# 4. 정확한 좌표로 클릭 및 입력
if server_field:
    mouse = MouseController()
    mouse.click(server_field.center_x, server_field.center_y)
    keyboard.type_text("192.168.1.100")
```

---

## 4. 기존 VLM 방식과 비교

| 특성 | VLM (Qwen/GPT-4V/Claude) | OmniParser | 권장 사용처 |
|------|--------------------------|------------|-------------|
| **처리 속도** | 2-5초/요청 | **0.6초/프레임** | 반복 작업: OmniParser |
| **구조화 출력** | ❌ (후처리 필요) | ✅ (즉시 사용 가능) | 프로그래밍: OmniParser |
| **복잡한 추론** | ✅ | ❌ | 의미 판단: VLM |
| **오프라인 동작** | ❌ (API 필요) | ✅ (로컬 GPU) | 보안 환경: OmniParser |
| **비용** | $0.01~0.05/요청 | 무료 (GPU 필요) | 대량 처리: OmniParser |
| **유연성** | ✅ (자연어 명령) | ❌ (UI 탐지만) | 예외 처리: VLM |
| **정확도 (좌표)** | 낮음 (±20px 오차) | **높음 (±2px)** | 정밀 클릭: OmniParser |

### 추천 하이브리드 전략

1. **OmniParser 우선 사용**: 빠르고 정확한 UI 요소 탐지
2. **VLM Fallback**: OmniParser가 요소를 찾지 못하거나, 복잡한 의미 판단이 필요한 경우
3. **결과 캐싱**: UI 레이아웃이 변하지 않으면 OmniParser 결과 재사용

자세한 통합 패턴은 [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md)를 참조하세요.

---

## 5. 설치 및 설정

### 5.1 시스템 요구사항

- **Python**: 3.10 이상
- **GPU**: CUDA 지원 GPU 필요 (최소 8GB VRAM 권장)
- **OS**: Linux, Windows, macOS (단, GPU는 Linux/Windows에서 최적)

### 5.2 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Hugging Face 모델 다운로드
python download_models.py
```

### 5.3 기본 사용법

```python
from omniparser import OmniParser

# 초기화 (모델 로딩, 첫 실행 시 수십 초 소요)
parser = OmniParser(device='cuda')

# 이미지 분석
import cv2
image = cv2.imread('screenshot.png')
results = parser.parse(image)

# 결과 출력
for element in results['elements']:
    print(f"{element['type']}: {element['caption']} at {element['bbox']}")
```

### 5.4 ARC 프로젝트 통합을 위한 requirements

OmniParser를 ARC에 통합하려면 별도 requirements 파일 작성 필요:

```bash
# requirements-omniparser.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
```

---

## 6. Florence-2 독립 사용

OmniParser의 백본 모델인 **Florence-2**는 단독으로도 사용 가능합니다.

### 6.1 Florence-2란?

- Microsoft의 경량 비전 파운데이션 모델
- 파라미터: 0.23B (Base), 0.77B (Large)
- 지원 태스크:
  - Object detection
  - Visual grounding (텍스트 → 이미지 영역 매핑)
  - OCR
  - Dense captioning
  - Region-level description

### 6.2 사용 예시

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

# 이미지 분석
image = Image.open("screenshot.png")
prompt = "<OD>"  # Object Detection 태스크

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### 6.3 OmniParser와의 관계

- **OmniParser = YOLO (detection) + Florence-2 (caption)**
- Florence-2만 사용하면 UI 요소 탐지 정확도는 낮지만, 설치가 간단함
- Custom 태스크(특정 아이콘만 찾기 등)에는 Florence-2 단독 사용도 유용

---

## 7. Phi-4 Vision (로컬 VLM 대안)

### 7.1 개요

**Phi-4-Multimodal**은 Microsoft의 경량 멀티모달 LLM입니다.

- 출시: 2026년 1월
- 크기: 14B 파라미터 (GPT-4V보다 훨씬 작음)
- 지원 모달리티: Text, Image, Audio
- 라이선스: MIT
- Hugging Face: `microsoft/phi-4-multimodal`

### 7.2 사용 시나리오

API 기반 VLM(Qwen, GPT-4V, Claude) 대신 **로컬에서 실행**하여:

- **비용 절감**: API 호출 비용 제로
- **오프라인 동작**: 인터넷 연결 불필요
- **데이터 보안**: 화면 캡처 이미지를 외부 서버로 전송하지 않음

### 7.3 제한사항

- **추론 속도**: GPU 필요 (CPU는 매우 느림)
- **정확도**: GPT-4V/Claude보다는 낮음 (특히 복잡한 UI 분석)
- **메모리**: 최소 16GB VRAM 권장

### 7.4 Phi-4 vs OmniParser

| 특성 | Phi-4 Vision | OmniParser |
|------|--------------|------------|
| **용도** | 범용 VLM (질문-답변) | UI 탐지 특화 |
| **출력** | 자연어 텍스트 | 구조화된 JSON |
| **속도** | 느림 (3-10초) | 빠름 (0.6초) |
| **권장 사용처** | 의미 판단, 추론 | UI 요소 위치 파악 |

---

## 8. CD-SEM/VeritySEM 적용 가능성

### 8.1 Current Pain Point

`automation/rcs/rcs_launcher.py`에서 현재 사용 중인 PyWinAuto는 다음 시나리오에서 실패합니다:

- **Custom DirectX 버튼**: 장비 제어 UI의 3D 렌더링 버튼
- **Embedded OpenGL 뷰어**: SEM 이미지 뷰어 내부 컨트롤
- **비표준 위젯**: Qt/WPF가 아닌 자체 구현 UI 컴포넌트

### 8.2 OmniParser 적용 방안

**Step 1: 하이브리드 아키텍처 구축**

```python
# automation/rcs/rcs_launcher.py 수정안

class RCSLauncher:
    def __init__(self, config: RCSConfig, use_omniparser: bool = True):
        self.config = config
        self.use_omniparser = use_omniparser

        if use_omniparser:
            from test.vlm_input_control.omniparser_integration import OmniParserAnalyzer
            self.omniparser = OmniParserAnalyzer()

    def click_login_button(self):
        # 1. PyWinAuto 먼저 시도
        try:
            self.app.window(title="RCS Login").Button.click()
            print("[INFO] PyWinAuto로 버튼 클릭 성공")
            return
        except Exception as e:
            print(f"[WARNING] PyWinAuto 실패: {e}")

        # 2. OmniParser로 fallback
        if self.use_omniparser:
            screenshot = ScreenCapture().capture_full_screen()
            elements = self.omniparser.analyze_screen(screenshot)

            login_btn = next(
                (e for e in elements if "login" in e.caption.lower()),
                None
            )

            if login_btn:
                MouseController().click(login_btn.center_x, login_btn.center_y)
                print("[INFO] OmniParser로 버튼 클릭 성공")
```

**Step 2: UI 레이아웃 학습**

최초 실행 시 OmniParser로 화면을 분석하고, UI 요소 위치를 캐싱:

```python
# automation/rcs/ui_cache.json (자동 생성)
{
  "login_screen": {
    "server_input": {"bbox": [300, 200, 600, 240], "last_seen": "2026-02-06T10:30:00"},
    "username_input": {"bbox": [300, 260, 600, 300], "last_seen": "2026-02-06T10:30:00"},
    "login_button": {"bbox": [400, 350, 500, 390], "last_seen": "2026-02-06T10:30:00"}
  }
}
```

**Step 3: 변화 감지 및 재학습**

화면 해시를 계산하여 UI 레이아웃 변경 감지:

```python
import hashlib

def detect_ui_change(current_screenshot, cached_hash):
    current_hash = hashlib.md5(current_screenshot.tobytes()).hexdigest()
    if current_hash != cached_hash:
        print("[INFO] UI 레이아웃 변경 감지, OmniParser 재실행")
        return True
    return False
```

### 8.3 예상 효과

- **자동화 성공률**: 70% (PyWinAuto only) → **95%** (Hybrid)
- **Custom Graphics 대응**: 불가능 → **가능**
- **유지보수 비용**: UI 변경 시 코드 수정 필요 → **자동 적응**

### 8.4 구현 우선순위

1. **High Priority**: RCS 로그인 화면 (자주 실패하는 Custom 버튼)
2. **Medium Priority**: Recipe 설정 화면 (복잡한 UI)
3. **Low Priority**: 단순 텍스트 입력 필드 (PyWinAuto로 충분)

자세한 구현 가이드는 [06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md)를 참조하세요.

---

## 9. 참고 자료

- **OmniParser GitHub**: https://github.com/microsoft/OmniParser
- **Florence-2 Model Card**: https://huggingface.co/microsoft/Florence-2-large
- **Phi-4 Documentation**: https://azure.microsoft.com/en-us/products/phi-4
- **논문**: "OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent" (arXiv:2408.00203v2, 2025)

---

**다음 문서**: [05-microsoft-automation-ecosystem.md](05-microsoft-automation-ecosystem.md) - WinAppDriver 및 보완 도구들
