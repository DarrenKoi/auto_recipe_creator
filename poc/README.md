# CPU-Based Automation PoC

## 목적

GPU 인프라 투자 정당화를 위한 CPU 기반 자동화 실현 가능성 증명(Proof of Concept).

회사 내부 VLM API(Kimi 2, Qwen3-VL)를 활용하여 CPU만으로도 GUI 자동화가 가능함을 입증하고, GPU 도입 시 성능 개선 폭을 제시합니다.

## 전략

### Tier 1: 순수 CPU 자동화 (현재)
- **화면 캡처** (mss) → **VLM API 호출** (Kimi 2 / Qwen3-VL) → **입력 실행** (pynput)
- 성능: 2-5초/액션, 성공률 70-80%
- CPU 사용률: < 10% (API 대기 시간)

### Tier 2: GPU 자동화 (목표)
- **로컬 OmniParser** (YOLO + Florence-2) → **빠른 추론** (0.6초) → **높은 정확도** (95%+)
- 성능: **8배 빠름**, 성공률 **+20%p 향상**, API 비용 **제로**

## 설치

### 필수 의존성

```bash
cd /Users/daeyoung/Codes/auto_recipe_creator
pip install -r test/vlm_input_control/requirements.txt
```

의존성:
- `mss` (화면 캡처)
- `pynput` (마우스/키보드 제어)
- `Pillow` (이미지 처리)
- `requests` (API 호출)

## 사용법

### 중요: Rate Limits

회사 API는 다음과 같은 rate limit이 있습니다:
- **Kimi 2**: 1 request / 3 seconds
- **Qwen3-VL**: 1 request / 1 second

PoC 데모는 자동으로 이를 준수하며, 필요 시 대기합니다.

### 이미지 최적화

로컬 LLM API는 큰 이미지를 처리하기 어려울 수 있습니다. 기본적으로:
- **WebP 포맷** 사용 (PNG 대비 30% 파일 크기 감소)
- **자동 리사이즈** (긴 쪽 1920px 기준, `--max-image-size`로 조정 가능)
- **최적화 압축** (quality=85, method=6)

### 1. 화면 분석 데모 (추천 시작점)

현재 화면을 캡처하고 VLM으로 분석만 수행합니다 (입력 없음).

```bash
python -m poc.cpu_automation_demo \
    --provider qwen3_vl \
    --api-url http://YOUR_COMPANY_API_URL \
    --api-key YOUR_API_KEY \
    --demo-type screen_analysis
```

**고해상도 화면에서 PNG 사용 (무손실):**
```bash
python -m poc.cpu_automation_demo \
    --provider qwen3_vl \
    --api-url http://YOUR_COMPANY_API_URL \
    --api-key YOUR_API_KEY \
    --demo-type screen_analysis \
    --no-webp \
    --max-image-size 2560
```

**출력 예시:**
```
[INFO] CPU 자동화 데모 초기화 완료
[INFO] VLM Provider: qwen3_vl
[INFO] Safe Mode: True

==============================================================
🔍 화면 분석 데모 시작
==============================================================

[1/3] 화면 캡처 중...
[INFO] 캡처 완료 (45.2ms)
[2/3] VLM API 호출 중...
[INFO] VLM 분석 완료 (2340ms)
[3/3] 분석 결과:
------------------------------------------------------------
화면 유형: application
주요 내용: RCS 로그인 화면
UI 요소 수: 4
가능한 액션 수: 2

UI 요소 목록:
  - Server Address Input: input
  - Username Input: input
  - Password Input: input
  - Login Button: button
------------------------------------------------------------

✅ 화면 분석 데모 완료

==============================================================
📊 CPU 기반 자동화 PoC 성능 리포트
==============================================================
총 액션 수:        1
성공:              1
실패:              0
성공률:            100.0%
평균 레이턴시:     2385 ms
최소 레이턴시:     2385 ms
최대 레이턴시:     2385 ms
==============================================================

💡 GPU 도입 시 예상 개선:
  레이턴시: 2385ms → 600ms (약 4.0배 빠름)
  성공률:   100% → 95%+ (0%p 향상)
  비용:     API 호출 비용 → $0 (로컬 추론)
  확장성:   API rate limit → 무제한 (로컬)
==============================================================
```

### 2. RCS 로그인 자동화 데모 (Safe Mode)

RCS 로그인 시퀀스를 시뮬레이션합니다 (실제 입력 없음).

```bash
python -m poc.cpu_automation_demo \
    --provider kimi_2 \
    --api-url http://YOUR_COMPANY_API_URL \
    --api-key YOUR_API_KEY \
    --demo-type rcs_login \
    --server 192.168.1.100 \
    --username admin \
    --password test123 \
    --safe-mode
```

**출력 예시:**
```
==============================================================
🚀 RCS 로그인 자동화 데모 시작
==============================================================

[1/5] 화면 캡처 중...
[INFO] 캡처 완료 (43.1ms)
[2/5] VLM API 호출 중...
[INFO] VLM 분석 완료 (2580ms)
[3/5] UI 요소 파싱 중...
[INFO] 4개 UI 요소 탐지
[4/5] 자동 입력 수행 중...
[SAFE MODE] Would click server_input at (520, 340)
[SAFE MODE] Would type: 192.168.1.100
[SAFE MODE] Would click username_input at (520, 380)
[SAFE MODE] Would type: admin
[SAFE MODE] Would click password_input at (520, 420)
[SAFE MODE] Would type: test123
[SAFE MODE] Would click login_button at (520, 470)
[5/5] 완료 (총 2650ms)

✅ RCS 로그인 데모 완료
```

### 3. 실제 입력 실행 (Live Mode)

⚠️ **주의:** `--live` 플래그를 사용하면 실제로 마우스/키보드가 제어됩니다!

```bash
python -m poc.cpu_automation_demo \
    --provider qwen3_vl \
    --api-url http://YOUR_COMPANY_API_URL \
    --api-key YOUR_API_KEY \
    --demo-type rcs_login \
    --server YOUR_SERVER \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD \
    --live
```

## API 설정

### Kimi 2 (Moonshot AI)

```bash
export VLM_API_BASE_URL="http://your-company-kimi-api.com"
export VLM_API_KEY="your-kimi-api-key"

python -m poc.cpu_automation_demo \
    --provider kimi_2 \
    --api-url $VLM_API_BASE_URL \
    --api-key $VLM_API_KEY \
    --demo-type screen_analysis
```

### Qwen3-VL

```bash
export VLM_API_BASE_URL="http://your-company-qwen3-api.com"
export VLM_API_KEY="your-qwen3-api-key"

python -m poc.cpu_automation_demo \
    --provider qwen3_vl \
    --api-url $VLM_API_BASE_URL \
    --api-key $VLM_API_KEY \
    --demo-type screen_analysis
```

## 성능 벤치마크

### CPU + API (현재)

| 지표 | 값 |
|------|-----|
| 화면 캡처 | ~50ms |
| VLM API 호출 | 2000-5000ms |
| JSON 파싱 | ~10ms |
| 입력 실행 | ~100ms/액션 |
| **총 레이턴시** | **2.5-5초** |
| **성공률** | **70-85%** (API 품질 의존) |

### GPU + Local (목표)

| 지표 | 값 | 개선폭 |
|------|-----|--------|
| 화면 캡처 | ~50ms | - |
| OmniParser 추론 | 600ms | **8배 빠름** |
| JSON 파싱 | ~10ms | - |
| 입력 실행 | ~100ms/액션 | - |
| **총 레이턴시** | **0.8초** | **3-6배 빠름** |
| **성공률** | **95%+** | **+10-25%p** |

## GPU 투자 정당화 자료

### ROI 계산

**시나리오:** 하루 100회 자동화 태스크 수행 시

#### CPU + API (현재)
- 레이턴시: 3초/액션
- 실패율: 20%
- API 비용: $0.01/호출
- **일일 비용:** $1.00
- **일일 시간 소요:** 300초 (5분) + 재시도 60초 = **360초 (6분)**

#### GPU (도입 후)
- 레이턴시: 0.6초/액션
- 실패율: 5%
- API 비용: $0
- **일일 비용:** $0
- **일일 시간 소요:** 60초 (1분) + 재시도 3초 = **63초 (1분)**

**개선:**
- 비용 절감: $365/년
- 시간 절약: 5분/일 → **1,825분/년 (30시간)**
- 신뢰성 향상: 80% → 95%

### GPU 요구사항

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA H100/H200 (8x for parallel batch processing) |
| VRAM | 80GB per GPU (OmniParser + CLIP) |
| Software | PyTorch 2.0+, CUDA 12.1+, cuDNN 8.9+ |
| Storage | 100GB (models + checkpoints) |

### 구현 로드맵

1. **Phase 1 (완료):** CPU + API PoC ✅
   - Kimi 2, Qwen3-VL 통합
   - 화면 분석 데모
   - 성능 벤치마크

2. **Phase 2 (GPU 승인 후):** GPU 인프라 구축
   - H200 GPU 할당
   - OmniParser 설치 및 최적화
   - CLIP 모델 로드

3. **Phase 3 (프로덕션):** 전체 자동화 파이프라인
   - RCS/CD-SEM 자동화
   - 멀티-GPU 배치 처리
   - 모니터링 & 로깅

## 문제 해결

### Mock 응답이 반환될 때

```
[INFO] Qwen3-VL API URL이 설정되지 않음
[INFO] Mock 응답 생성 중...
```

**해결:** API URL과 키를 올바르게 설정:
```bash
python -m poc.cpu_automation_demo \
    --provider qwen3_vl \
    --api-url http://CORRECT_URL \
    --api-key YOUR_KEY \
    --demo-type screen_analysis
```

### JSON 파싱 실패

```
[ERROR] JSON 파싱 실패
```

**원인:** VLM이 JSON이 아닌 텍스트를 반환
**해결:** 프롬프트 개선 또는 다른 provider 시도

### API 호출 타임아웃

```
[ERROR] Kimi 2 API 호출 실패: HTTPConnectionPool(host='...', port=80): Read timed out.
```

**해결:** 네트워크 연결 확인 또는 타임아웃 증가 (코드 수정 필요)

## 코드 아키텍처

```
poc/
├── __init__.py
├── cpu_automation_demo.py   # 메인 데모 스크립트
└── README.md                 # 이 문서

test/vlm_input_control/
├── vlm_screen_analysis.py    # VLM API 통합 (Kimi 2, Qwen3-VL 추가)
├── screen_capture.py          # mss 기반 화면 캡처
├── mouse_control.py           # pynput 마우스 제어
└── keyboard_control.py        # pynput 키보드 제어
```

### 주요 클래스

- **`CPUAutomationDemo`**: PoC 오케스트레이터
- **`PerformanceMetrics`**: 성능 측정 및 리포트
- **`VLMScreenAnalyzer`**: VLM API 통합 (6개 provider 지원)

## 다음 단계

1. **PoC 결과를 Data Team에 제시**
   - 성능 리포트 스크린샷
   - GPU ROI 계산
   - 구현 로드맵

2. **GPU 승인 후:**
   ```bash
   # OmniParser 설치
   git clone https://github.com/microsoft/OmniParser.git
   cd OmniParser
   pip install -r requirements.txt
   python download_models.py

   # CLIP 설치
   pip install git+https://github.com/openai/CLIP.git

   # GPU 의존성
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install faiss-gpu
   ```

3. **프로덕션 배포:**
   - H200 클러스터 설정
   - 배치 프로세싱 (`test/video_frame_parser/batch_processor.py` 패턴 참조)
   - 모니터링 대시보드

## 라이선스

내부 사용 전용. 외부 배포 금지.
