# Home Study Environment

집에서 GPU와 회사 API 없이 GUI 자동화를 학습하기 위한 환경입니다.

## 특징

- **무료 API**: Hugging Face Inference API (무료 계정)
- **GPU 불필요**: 모든 추론은 HF 서버에서 실행
- **회사 VPN 불필요**: 공개 API 사용

## 빠른 시작

### 1. HuggingFace 계정 생성 (무료)

1. https://huggingface.co/join 에서 계정 생성
2. https://huggingface.co/settings/tokens 에서 토큰 발급
3. 토큰 설정:
   ```bash
   # Windows
   set HF_TOKEN=hf_xxxx

   # Linux/Mac
   export HF_TOKEN=hf_xxxx
   ```

### 2. 의존성 설치

```bash
uv sync --extra home
```

### 3. 설정 확인

```bash
uv run python -m poc.home.test_setup
```

### 4. 데모 실행

```bash
# 전체 데모
uv run python -m poc.home.demo

# 화면 분석만
uv run python -m poc.home.demo --mode screen_analysis

# UI 요소 분석
uv run python -m poc.home.demo --mode ui_elements

# 대화형 모드 (질문하면 답변)
uv run python -m poc.home.demo --mode interactive
```

## 사용 가능한 모델

| 모델 | 용도 | 크기 | 속도 |
|------|------|------|------|
| `Qwen/Qwen2-VL-7B-Instruct` | 화면 분석 (기본) | 7B | 보통 |
| `Qwen/Qwen2-VL-2B-Instruct` | 빠른 분석 | 2B | 빠름 |
| `llava-hf/llava-1.5-7b-hf` | 대안 VLM | 7B | 보통 |
| `facebook/detr-resnet-50` | 객체 탐지 | - | 빠름 |

## Rate Limits

HuggingFace 무료 계정:
- 약 **100-300 requests/hour**
- 모델 로딩 시간 (cold start): 첫 요청 시 10-60초

PRO 계정 ($9/월):
- 8배 높은 rate limit
- 우선 접근권

## 학습 로드맵

### 1단계: 화면 이해
```bash
uv run python -m poc.home.demo --mode screen_analysis
```
- VLM이 화면을 어떻게 인식하는지 이해
- 프롬프트 엔지니어링 실습

### 2단계: UI 요소 탐지
```bash
uv run python -m poc.home.demo --mode ui_elements
```
- 클릭 가능한 요소 식별
- JSON 형식 응답 파싱

### 3단계: 대화형 분석
```bash
uv run python -m poc.home.demo --mode interactive
```
- 자유롭게 질문하며 VLM 능력 탐색
- 자동화 시나리오 구상

### 4단계: 회사에서 적용
- 집에서 배운 프롬프트를 회사 API에 적용
- Kimi 2, Qwen3-VL과 비교

## 코드 구조

```
poc/home/
├── __init__.py        # 패키지 초기화
├── hf_vlm.py          # HuggingFace VLM 통합
├── demo.py            # 데모 스크립트
├── test_setup.py      # 환경 설정 확인
└── README.md          # 이 문서
```

## 문제 해결

### "huggingface_hub not found"
```bash
uv sync --extra home
```

### "401 Unauthorized"
- HF_TOKEN이 올바르게 설정되었는지 확인
- 토큰이 만료되지 않았는지 확인

### "429 Rate Limit"
- 잠시 기다린 후 다시 시도
- PRO 계정 고려 ($9/월)

### "503 Service Unavailable"
- 모델 cold start 중 (정상)
- 10-60초 후 자동으로 준비됨

## 회사 환경 vs 집 환경

| 항목 | 회사 | 집 |
|------|------|-----|
| API | Kimi 2, Qwen3-VL | HuggingFace (무료) |
| GPU | 필요 없음 (API) | 필요 없음 (API) |
| 속도 | 2-5초/요청 | 3-10초/요청 |
| Rate Limit | 사내 정책 | ~100-300/hour |
| 비용 | 회사 부담 | 무료 |
