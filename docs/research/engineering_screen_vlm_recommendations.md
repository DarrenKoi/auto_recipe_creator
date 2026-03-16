# 복잡한 엔지니어링 화면 판독용 VLM 운영 메모 (2026-03-13)

## 목적

이 문서는 `deploy_vlms/config/`, `flask_api/vlm_serve/config.py`, 그리고 현재 저장소에서 이미 사용 중인 회사 API 모델(`Kimi-K2.5`)을 함께 기준으로, 지금 바로 비교하거나 조합할 수 있는 모델 구성을 정리한 운영 메모다.

지금 기준 핵심 질문은 아래 2개다.

1. 첫 비교는 어떤 모델끼리 하는 것이 맞는가?
2. `MAI-UI`를 zoom-in 전용 sidecar로 두고 OCR 모델과 같이 쓰는 구성이 실용적인가?

## 결론 요약

- 전체 1차 비교 세트는 `Kimi-K2.5` vs `UI-Venus-1.5-8B` vs `UI-TARS-1.5-7B`가 가장 해석하기 좋다.
- self-hosted 모델끼리의 직접 head-to-head는 `UI-Venus-1.5-8B` vs `UI-TARS-1.5-7B`로 두는 편이 깔끔하다.
- `MAI-UI-8B`는 full-screen 주력보다는 **작은 타깃 재탐색용 zoom-in sidecar**로 두는 편이 현재 구성에 더 잘 맞는다.
- OCR sidecar는 `PaddleOCR-VL-1.5`를 기본으로 두고, 아주 작은 글씨나 formatting 민감 케이스만 `GOT-OCR-2.0-hf`로 보강하는 구성이 좋다.
- `MAI-UI + PaddleOCR-VL + GOT-OCR`를 매 step 항상 다 돌리는 것은 비효율적이다. **조건부 escalation**로 묶어야 한다.
- 현재 Flask proxy 활성 기준 주력 서비스는 `UI-Venus(8001)`, `MAI-UI(8002)`, `PaddleOCR-VL-1.5(8004)`, `GOT-OCR-2.0-hf(8005)`다. `UI-TARS(8003)`는 model env는 있으나 proxy에서는 아직 비활성이다.
- `Kimi-K2.5`는 `deploy_vlms/config/`에는 없지만 회사 API 경로로 이미 사용 가능하므로, **외부 API baseline**으로 유지하는 것이 맞다.

## 1. 현재 사용 가능한 모델 구성

`deploy_vlms/config/models/*.env`, `flask_api/vlm_serve/config.py`, 그리고 저장소 내 기존 Kimi 사용 흔적(`poc/work/automate_rcs_login.py`, `poc/work/select_tool.py`, `test/vlm_input_control/vlm_screen_analysis.py`)을 합쳐 보면 현재 비교 구성은 아래와 같다.

| 구분 | served model | service slug | port | GPU | 현재 상태 | 주 용도 |
|------|--------------|--------------|------|-----|-----------|---------|
| 외부 API baseline | `Kimi-K2.5` | 회사 API | - | 외부/사내 제공 | 사용 가능 | self-hosted GUI 모델 대비 정확도/범용성 기준선 |
| GUI 주력 A | `ui-venus-1.5-8b` | `ui-venus` | `8001` | `0` | 활성 | full-screen GUI grounding 기본 후보 |
| GUI 주력 B | `ui-tars-1.5-7b` | `ui-tars` | `8003` | `0` | env 존재, proxy 비활성 | `UI-Venus`와 직접 비교할 다음 후보 |
| Zoom-in sidecar | `mai-ui-8b` | `mai-ui` | `8002` | `1` | 활성 | 작은 요소/모호한 crop 재탐색 |
| OCR 기본 sidecar | `paddleocr-vl-1.5` | `paddleocr-vl-1.5` | `8004` | `1` | 활성 | 작은 텍스트, 표, dense panel 읽기 |
| OCR fallback | `got-ocr-2.0-hf` | `got-ocr` | `8005` | `1` | 활성 | hard crop OCR, formatting 민감 케이스 |

공통 설정도 운영 해석에 중요하다.

- `common.env`는 `HOST=127.0.0.1`, `DTYPE=bfloat16`, `STRICT_OFFLINE=1` 기준이다.
- GPU 0은 `UI-Venus`와 `UI-TARS` co-location 전제를 두고 있다.
- GPU 1은 `MAI-UI + PaddleOCR-VL + GOT-OCR` 조합을 전제로 잡혀 있다.

즉, 현재 구성은 **회사 API의 Kimi를 범용 baseline으로 두고**, 로컬 GPU에서는 **GPU 0에서 coarse grounding 비교**, **GPU 1에서 zoom/OCR 보강**이라는 역할 분리가 이미 어느 정도 설계되어 있다.

## 2. 비교 순서 권장안

### 2.1 1차 비교

첫 실험은 아래처럼 가져가는 편이 맞다.

1. `Kimi-K2.5` full-screen pass
2. `UI-Venus-1.5-8B` full-screen pass
3. `UI-TARS-1.5-7B` full-screen pass
4. 같은 스크린샷 세트에서 element hit rate, click drift, retry count 비교

이 비교가 좋은 이유는 아래와 같다.

- `Kimi`를 넣으면 "현재 회사 API baseline보다 self-hosted GUI 모델이 실제로 나은가"를 바로 볼 수 있다.
- `UI-Venus`와 `UI-TARS`는 둘 다 GUI 주력 후보라 역할이 같다.
- `UI-Venus`와 `UI-TARS`는 둘 다 GPU 0 계열 모델이라 self-hosted끼리 응답 시간 비교가 공정하다.
- `MAI-UI`처럼 zoom-in 특화 역할을 섞지 않아서 결과 해석이 단순해진다.

### 2.2 2차 비교

1차에서 더 안정적인 self-hosted coarse 모델을 고른 뒤, 그 모델에 아래를 순차적으로 붙인다.

1. `MAI-UI-8B` zoom-in sidecar
2. `PaddleOCR-VL-1.5` OCR sidecar
3. `GOT-OCR-2.0-hf` fallback OCR

즉, 비교 축을 아래처럼 나누는 것이 좋다.

- 축 A: `Kimi` vs `UI-Venus` vs `UI-TARS`
- 축 B: `UI-Venus` vs `UI-TARS` self-hosted 직접 비교
- 축 C: `MAI-UI`를 붙였을 때 개선되는 failure type
- 축 D: `PaddleOCR`/`GOT-OCR`를 붙였을 때 텍스트 판독 개선량

## 3. `MAI-UI`를 zoom-in sidecar로 두는 이유

이전 메모에서도 정리했듯 `MAI-UI`는 mobile-only 모델로 보기 어렵다. 다만 현재 운영 관점에서는 아래처럼 두는 편이 더 실용적이다.

- full-screen 모든 step에 항상 태우기보다, 작은 타깃에서만 선택적으로 호출할 때 강점이 잘 살아난다.
- 현재 config상 `MAI-UI`는 GPU 1에 있고, `UI-Venus`/`UI-TARS`는 GPU 0에 있으므로 coarse-pass와 sidecar-pass를 분리하기 쉽다.
- OCR 계열과 같은 GPU에 있어도, 순차적 조건부 호출이면 운영 가능하다.

권장 역할은 아래와 같다.

1. full-screen 모델이 후보 bbox 또는 영역을 먼저 찾는다.
2. bbox가 너무 작거나 주변 요소가 과밀하면 crop을 만든다.
3. `MAI-UI-8B`로 crop/zoom-in 재탐색을 수행한다.
4. 필요하면 같은 crop에 OCR sidecar를 추가한다.

이 패턴이면 `MAI-UI`의 강점을 살리면서도, primary 비교 실험을 흐리지 않는다.

## 4. `MAI-UI + OCR sidecar` 조합은 가능한가

가능하다. 다만 **항상-on 체인**이 아니라 **조건부 escalation 체인**으로 써야 한다.

| 조건 | 우선 호출 | 기대 효과 | 주의점 |
|------|-----------|-----------|--------|
| 일반 full-screen 버튼/탭/패널 탐색 | `UI-Venus` 또는 `UI-TARS` | 빠른 coarse grounding | sidecar 불필요 |
| bbox가 작음, 주변 아이콘이 과밀함 | `MAI-UI-8B` | zoom-in 재탐색 | crop 생성 규칙이 중요 |
| 텍스트가 많고 판독이 핵심 | `PaddleOCR-VL-1.5` | dense text/OCR 보강 | bbox 자체는 GUI 모델이 더 낫다 |
| 작은 숫자/코드/포맷 보존이 중요 | `GOT-OCR-2.0-hf` | hard OCR fallback | 항상 호출하면 지연 증가 |

운영 규칙은 아래 정도가 적당하다.

1. 기본은 GUI 모델 1회 호출
2. low-confidence, small-target, crowded-toolbar면 `MAI-UI` 추가
3. text-heavy panel이면 `PaddleOCR-VL` 추가
4. `PaddleOCR-VL` 결과가 부족하거나 format 보존이 필요하면 `GOT-OCR` 추가

이렇게 하면 `MAI-UI`가 OCR sidecar들과 충돌하는 것이 아니라, **다른 failure mode를 담당하는 보조 단계**가 된다.

## 5. 현재 저장소 기준 권장 파이프라인

### 추천 1안: 비교 실험용

1. `Kimi-K2.5`, `UI-Venus-1.5-8B`, `UI-TARS-1.5-7B`를 동일 스크린샷 세트로 비교
2. self-hosted 후보 중 더 안정적인 coarse 모델을 primary로 채택
3. ambiguous case에서만 `MAI-UI-8B` zoom-in pass 추가
4. dense text crop에서는 `PaddleOCR-VL-1.5` 추가
5. 필요한 경우만 `GOT-OCR-2.0-hf` fallback

### 추천 2안: 실제 운영용

1. main UI service 후보: `UI-Venus-1.5-8B` 또는 `UI-TARS-1.5-7B`
2. Zoom-in sidecar: `MAI-UI-8B`
3. OCR default sidecar: `PaddleOCR-VL-1.5`
4. OCR fallback: `GOT-OCR-2.0-hf`
5. External baseline: `Kimi-K2.5`

현재 `poc/work2`는 단일 "primary VLM"도, purpose slot도 두지 않고 service slug 를 직접 고른다. 기본 registry 는 `poc/work2/flask_vlm.py`, 실제 호출은 `poc/work2/vlm_client.py` 기준으로 맞추면 되므로, 여기에서 `UI-TARS` 비교와 `MAI-UI` 조건부 호출만 추가하는 방식이 변경 폭이 가장 작다.

## 6. 실행 시 주의할 점

- `UI-TARS`는 `deploy_vlms/config/models/ui-tars.env`는 준비되어 있지만, `flask_api/vlm_serve/config.py`에서는 `enabled=False`다.
- 따라서 proxy 기반 A/B를 하려면 route를 활성화하거나, 우선 direct `8003` 호출로 비교해야 한다.
- `Kimi-K2.5`는 external API이므로 latency, rate limit, cost 조건을 self-hosted와 분리해서 해석해야 한다.
- `MAI-UI`, `PaddleOCR-VL`, `GOT-OCR`가 모두 GPU 1에 있으므로, sidecar는 병렬 상시 호출보다 순차적 조건부 호출이 안전하다.
- `PaddleOCR-VL`은 OCR 보강용이지 click grounding 대체용으로 보기 어렵다.

## 7. 바로 실행할 평가 세트

### 공통 시나리오

- `RCS login`
- `View/List` tab 전환
- `tool list`에서 target row 찾기
- 신규 tool window title 확인
- parameter panel 내부의 작은 text/button/icon 찾기

### 측정 지표

- target element hit rate
- click-point 오차(px)
- retry count
- step completion rate
- small-text OCR recall
- 평균 응답 시간
- sidecar escalation 발생 비율

### 권장 비교 순서

1. `Kimi` vs `UI-Venus` vs `UI-TARS`
2. self-hosted 승자 모델 + `MAI-UI`
3. self-hosted 승자 모델 + `MAI-UI` + `PaddleOCR-VL`
4. 필요한 failure case에만 `GOT-OCR` 추가

## 8. 최종 권고

- 첫 비교 세트에는 `Kimi-K2.5`도 같이 넣는 것이 맞다.
- self-hosted 직접 비교축은 `UI-Venus`와 `UI-TARS`가 맞다.
- `MAI-UI`는 primary head-to-head 대상보다 **zoom-in 특화 sidecar**로 두는 편이 좋다.
- `MAI-UI`를 OCR 모델과 함께 쓰는 것은 가능하지만, GPU 1 공유 구조상 **조건부 escalation**로 묶어야 한다.
- 현재 가장 현실적인 운영 스택은 `Kimi baseline + self-hosted GUI 주력 1개 + MAI-UI zoom-in + PaddleOCR 기본 OCR + GOT-OCR fallback`이다.

## 근거 파일

- `deploy_vlms/config/common.env`
- `deploy_vlms/config/models/ui-venus.env`
- `deploy_vlms/config/models/ui-tars.env`
- `deploy_vlms/config/models/mai-ui.env`
- `deploy_vlms/config/models/paddleocr-vl-1.5.env`
- `deploy_vlms/config/models/got-ocr-2.0-hf.env`
- `flask_api/vlm_serve/config.py`
- `poc/work2/flask_vlm.py`
- `poc/work/automate_rcs_login.py`
- `poc/work/select_tool.py`
- `test/vlm_input_control/vlm_screen_analysis.py`
