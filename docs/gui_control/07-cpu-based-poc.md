# CPU 기반 자동화 PoC (Tier 1 전략)

이 문서는 GPU 인프라 투자 전, **CPU와 내부 VLM API**만을 사용하여 GUI 자동화의 실현 가능성을 검증한 PoC(Proof of Concept) 전략을 설명합니다.

## 1. 개요 및 목적

### 1.1 배경
ARC(Auto Recipe Creator) 프로젝트의 궁극적인 목표는 로컬 GPU 기반의 고속/고정밀 자동화(Tier 2)입니다. 하지만 초기 단계에서 고가용성 GPU 인프라(H100/H200) 투자를 정당화하기 위해, **현재 가용한 리소스(CPU + 내부 API)**만으로 자동화가 가능함을 입증해야 했습니다.

### 1.2 목표
- **기술 검증**: 이미지 기반 UI 분석 및 제어 파이프라인의 기술적 타당성 확인.
- **베이스라인 수립**: 현재 방식의 성능(레이턴시, 성공률)을 측정하여 GPU 도입 시의 개선 효과(ROI) 산출.
- **Tier 1 전략**: GPU가 없는 환경에서도 동작하는 "느리지만 작동하는" 백업 솔루션 확보.

## 2. 아키텍처 (Tier 1)

순수 CPU 환경에서 작동하도록 설계된 "Observe-Process-Act" 파이프라인입니다.

```mermaid
graph LR
    A[Screen Capture] -->|Image (MSS)| B(Image Optimizer)
    B -->|WebP/Resize| C{VLM API Gateway}
    C -->|Request| D[Internal API]
    D -->|Response (JSON)| E[Action Parser]
    E -->|Command| F[Input Controller]
    F -->|Mouse/Key| G[Application]
```

### 2.1 핵심 컴포넌트
1.  **Screen Capture (`mss`)**:
    - 초고속 스크린샷 캡처 (~50ms).
    - GPU 가속 없이도 CPU 부하가 적음.
2.  **Image Optimizer**:
    - **WebP 변환**: PNG 대비 파일 크기 30% 수준으로 압축하여 네트워크 대역폭 절약.
    - **Smart Resize**: 긴 쪽 기준 1920px로 리사이즈하여 API 허용량 준수 및 처리 속도 향상.
3.  **VLM API Client**:
    - **Multi-Provider Support**: Kimi 2 (Moonshot AI) 및 Qwen3-VL 지원.
    - **Smart Rate Limiting**: Provider별 요청 제한(예: Qwen 1초/1회, Kimi 3초/1회)을 클라이언트 측에서 자동 준수.
4.  **Input Controller (`pynput`)**:
    - VLM이 반환한 좌표 및 액션(클릭, 타이핑)을 물리적 입력 신호로 변환.

## 3. 구현 상세 및 최적화

### 3.1 Rate Limiting 전략
API의 호출 제한을 엄격히 준수하기 위해 데코레이터 또는 대기 로직을 구현했습니다.

```python
RATE_LIMITS = {
    VLMProvider.KIMI_2: 3.0,      # 3초에 1회
    VLMProvider.QWEN3_VL: 1.0,    # 1초에 1회
}

def _wait_for_rate_limit(self):
    elapsed = time.time() - self.last_api_call_time
    if elapsed < self.rate_limit:
        time.sleep(self.rate_limit - elapsed)
```

### 3.2 이미지 최적화
로컬 LLM API는 대용량 이미지 처리에 취약할 수 있으므로, 화질 저하를 최소화하면서 페이로드를 줄이는 것이 핵심입니다.

- **포맷**: 무손실 PNG 대신 `quality=85`의 **WebP** 사용.
- **크기**: 텍스트 가독성이 유지되는 한계선인 FHD(1920px) 수준으로 리사이즈.

## 4. 성능 벤치마크 및 ROI

PoC를 통해 측정한 현재(Tier 1) 성능과 향후 GPU 도입(Tier 2) 시의 예상 성능 비교입니다.

### 4.1 성능 비교표

| 지표 | Tier 1 (CPU + API) | Tier 2 (GPU + Local) | 개선폭 |
|------|-------------------|----------------------|--------|
| **화면 캡처** | ~50ms | ~50ms | - |
| **분석/추론** | 2000-5000ms | **600ms** (OmniParser) | **4-8배** |
| **데이터 전송** | 수백 ms (네트워크) | **0ms** (PCIe/Memory) | 즉시 |
| **총 레이턴시** | **2.5 ~ 8초** | **0.8초** | **3-10배** |
| **성공률** | 70-85% | **95%+** | 안정성 확보 |
| **비용** | API 호출당 과금 | **$0** (초기 투자 제외) | 운영비 절감 |

### 4.2 ROI (투자 대비 효과) 분석
하루 100회 자동화 태스크 수행 시:

- **시간 절약**: 일일 약 30분 → 5분으로 단축 (연간 **150시간** 절약).
- **비용 절감**: 외부/내부 API 토큰 비용 제거.
- **보안 강화**: 화면 데이터가 로컬 네트워크를 벗어나지 않음.

## 5. 결론 및 로드맵

이 PoC는 **CPU 환경에서도 기본적인 GUI 자동화가 가능함**을 증명했습니다. 하지만 상용 수준의 반응 속도와 안정성을 위해서는 **GPU 기반의 로컬 인퍼런스(Tier 2)**로의 전환이 필수적임을 데이터로 뒷받침합니다.

### 추천 마이그레이션 경로
1.  **현재**: `poc/cpu_automation_demo.py`를 활용하여 기본 로직 검증 및 데이터 수집.
2.  **과도기**: 하이브리드 패턴([06-hybrid-automation-patterns.md](06-hybrid-automation-patterns.md)) 적용 (쉬운 건 CPU, 어려운 건 API).
3.  **최종**: OmniParser 및 로컬 VLM 도입 후 API 의존성 제거.

---
**관련 코드**:
- `poc/cpu_automation_demo.py`
- `poc/README.md`
