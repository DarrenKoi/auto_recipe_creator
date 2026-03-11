# 복잡한 엔지니어링 화면 판독용 VLM 추천 메모 (2026-03-11)

## 목적

이 문서는 `CD-SEM/VeritySEM/RCS`처럼 요소가 촘촘하고 텍스트가 작으며 패널 구조가 복잡한 엔지니어링 툴 화면을 읽기 위한 VLM 후보를 정리한 메모다.

핵심 질문은 아래 2개다.

1. `MAI-UI`가 모바일 전용 모델인가?
2. 현재 공개된 모델 중 복잡한 엔지니어링 화면 판독에 가장 유리한 조합은 무엇인가?

## 결론 요약

- `MAI-UI`는 **mobile-first 성향은 강하지만 mobile-only 모델은 아니다**.
- 복잡한 엔지니어링 화면 평가에는 `ScreenSpot-Pro` 같은 **professional desktop GUI grounding benchmark**가 더 중요하다.
- 현재 공개 정보 기준으로 최고 정확도는 `Holo2-235B-A22B (agentic localization)` 쪽이 앞선다.
- 상용/자체 배포 현실성까지 같이 보면 `MAI-UI-32B (+Zoom-In)`와 `UI-Venus-1.5-30B-A3B`가 가장 실무적인 상위 후보다.
- 실제 운영에서는 **GUI grounding 모델 1개만 쓰지 말고**, `zoom/crop 재탐색 + OCR 전용 보조 모델`을 같이 붙이는 편이 안전하다.

## 1. 왜 일반 GUI 벤치마크보다 `ScreenSpot-Pro`가 중요한가

복잡한 엔지니어링 화면 읽기 문제는 일반 모바일 앱이나 단순 웹보다 훨씬 까다롭다. 작은 버튼, 촘촘한 아이콘, 다중 패널, 고해상도 스크린샷, 전문 도메인 용어가 동시에 섞이기 때문이다.

`ScreenSpot-Pro`는 이런 조건에 더 가깝다.

- professional high-resolution computer use를 목표로 만든 benchmark다.
- 23개 professional application, 5개 산업군, 3개 OS를 다룬다.
- 예시 앱에 `AutoCAD`, `SolidWorks`, `Inventor`, `Vivado`, `MATLAB`, `Origin`, `Quartus` 같은 엔지니어링/분석 툴이 포함된다.
- 실제 고해상도 스크린샷과 expert annotation을 사용한다.

즉, 우리 용도에서는 `AndroidWorld` 같은 mobile navigation 점수보다 `ScreenSpot-Pro`, `OSWorld-G`, `UI-Vision` 같은 **desktop grounding 계열 지표**를 더 우선해서 봐야 한다.

## 2. `MAI-UI`는 모바일 전용인가

아니다. 다만 **공식 메시지와 데모가 모바일 쪽을 강하게 강조하는 편**이다.

`MAI-UI-8B` 공식 모델 카드는 아래를 같이 말한다.

- 모델 family는 2B, 8B, 32B, 235B-A22B까지 있다.
- mobile navigation을 강하게 강조한다.
- 동시에 grounding 결과로 `ScreenSpot-Pro`, `OSWorld-G`, `UI-Vision`을 공식 표에 포함한다.

특히 `MAI-UI-8B` 모델 카드에는 grounding 결과로 아래 수치가 직접 적혀 있다.

- `ScreenSpot-Pro`: 73.5
- `OSWorld-G`: 70.9
- `UI-Vision`: 49.2

이 수치는 family 전체 최고치 설명에 가깝고, 세부 variant별 수치는 다른 비교표를 같이 봐야 한다. 그래도 중요한 해석은 분명하다.

- `MAI-UI`는 모바일-only가 아니다.
- 오히려 **desktop grounding benchmark에서도 경쟁력 있는 GUI agent family**로 보는 것이 맞다.
- 따라서 `flask_api/vlm_serve/mai_ui.py`라는 파일명이 있다고 해서 모바일 전용 endpoint라고 해석하면 안 된다. 현재 파일은 단순 프록시 설정일 뿐이다.

## 3. 현재 추천 모델 순위

### 3.1 최고 정확도 우선

1. `Holo2-235B-A22B (Agentic Localization)`
2. `MAI-UI-32B (+Zoom-In)`
3. `UI-Venus-1.5-30B-A3B`
4. `MAI-UI-8B (+Zoom-In)`
5. `Holo2-8B (Agentic)` 또는 `UI-Venus-1.5-8B`

### 3.2 실무 배포 우선

1. `MAI-UI-32B (+Zoom-In)`
2. `UI-Venus-1.5-30B-A3B`
3. `MAI-UI-8B (+Zoom-In)`
4. `UI-Venus-1.5-8B`
5. `PaddleOCR-VL`을 보조 OCR 모델로 결합

## 4. 모델별 해석

| 모델 | 장점 | 약점/주의점 | 판단 |
|------|------|-------------|------|
| `Holo2-235B-A22B (Agentic)` | 현재 공개 수치상 최고 수준. `ScreenSpot-Pro 78.5%`, `OSWorld-G 79.0%` | `235B`급이라 운영 부담이 매우 크고, `30B/235B`는 `cc-by-nc-4.0` 기반 research-only 제약이 있다 | 정확도 ceiling 확인용 reference model |
| `MAI-UI-32B (+Zoom-In)` | `ScreenSpot-Pro 73.5%`, `OSWorld-G 70.9%`. zoom-in을 포함한 고해상도 소형 타깃 탐색이 강점 | 32B급 운영 비용 존재. mobile 브랜딩 때문에 오해받기 쉽다 | `Apache-2.0` 계열이라 실무 후보로 가장 균형이 좋다 |
| `UI-Venus-1.5-30B-A3B` | 공식 카드 기준 `ScreenSpot-Pro 69.6%`, `OSWorld-G 70.6%`, `UI-Vision 54.7%`. grounding/mobile/web을 unified하게 학습 | absolute top score는 Holo2, MAI-UI-32B보다 약간 낮다 | `Apache-2.0` 기반의 매우 강한 A/B baseline |
| `MAI-UI-8B (+Zoom-In)` | 8B 대비 강한 성능. `ScreenSpot-Pro 70.9%`, `OSWorld-G 64.2%` | 32B 대비 복잡 화면 안정성은 더 약할 가능성 | GPU 제약이 있을 때 현실적인 선택 |
| `UI-Venus-1.5-8B` | 8B인데도 `ScreenSpot-Pro 68.4%`로 강함 | 작은 텍스트/밀집 아이콘에서 대형 모델 대비 한계 가능 | 가벼운 self-hosted baseline |
| `PaddleOCR-VL` | 텍스트, 표, 수식, 차트 parsing이 강하다 | 클릭 좌표 grounding 자체는 GUI agent보다 약하다 | OCR sidecar로 거의 필수 |

## 5. 왜 단일 VLM보다 `GUI grounding + OCR sidecar` 조합이 낫나

엔지니어링 툴 화면은 아래 문제가 반복된다.

- 작은 숫자/단위/파라미터명이 많다.
- grid, table, chart, formula-like text가 섞인다.
- 패널 이름과 실제 target widget이 멀리 떨어져 있다.
- 메뉴/탭/toolbar가 과밀해서 click-point drift가 생긴다.

이 때문에 단일 GUI agent만으로는 아래 상황에서 흔들리기 쉽다.

- target text는 읽었는데 정확한 클릭 좌표를 못 잡는 경우
- 좌표는 근처까지 갔는데 인접 아이콘과 혼동하는 경우
- tiny OCR text를 놓쳐서 잘못된 panel을 읽는 경우

따라서 운영 권장 조합은 아래와 같다.

1. 1차: `MAI-UI-32B` 또는 `UI-Venus-1.5-30B-A3B`로 target panel/element 후보를 찾는다.
2. 2차: 후보 영역을 crop/zoom-in 해서 같은 모델 또는 agentic localization으로 재탐색한다.
3. 3차: 텍스트 밀집 구역은 `PaddleOCR-VL`로 보강한다.
4. 4차: 최종 클릭 전에는 bbox 중심점 대신 offset-safe click rule을 둔다.

## 6. 이 저장소 기준 추천안

현재 저장소 구조를 기준으로 하면 아래가 가장 현실적이다.

### 추천 1안

- `8001`: `UI-Venus-1.5-8B` 또는 `UI-Venus-1.5-30B-A3B`
- `8002`: `MAI-UI-8B` 또는 `MAI-UI-32B`
- `8003`: 비교용 다른 GUI 모델
- 별도 sidecar: `PaddleOCR-VL`

이 구성이 좋은 이유는 아래와 같다.

- 기존 `flask_api/vlm_serve` 구조가 모델별 proxy route 확장에 맞춰져 있다.
- `poc/work` 쪽은 이미 screenshot capture와 step-by-step action loop가 있어서 A/B test 연결이 쉽다.
- `MAI-UI`와 `UI-Venus`를 같은 스크린샷 세트로 비교하기 좋다.

### 추천 2안

최고 정확도 연구용으로는 아래처럼 보는 편이 맞다.

- reference ceiling: `Holo2-235B-A22B (Agentic)`
- production candidate A: `MAI-UI-32B (+Zoom-In)`
- production candidate B: `UI-Venus-1.5-30B-A3B`
- OCR sidecar: `PaddleOCR-VL`

즉, `Holo2`는 "얼마나 더 올라갈 수 있는가"를 보는 ceiling reference로 좋고, 실제 사내 배포는 `MAI-UI` 또는 `UI-Venus` 중심으로 가는 편이 현실적이다.

## 7. 바로 실행할 A/B 테스트 제안

### 공통 평가 세트

- `RCS login`
- `View/List` tab 전환
- `tool list`에서 target tool row 찾기
- 신규 tool window title 확인
- parameter panel 내부의 작은 text/button/icon 찾기

### 측정 지표

- target element hit rate
- click-point 오차(px)
- retry count
- step completion rate
- small-text OCR recall
- 평균 응답 시간

### 우선 비교 순서

1. `MAI-UI-8B` vs `UI-Venus-1.5-8B`
2. 더 좋은 family를 골라 `32B/30B`급으로 확장
3. 그 결과에 `PaddleOCR-VL` sidecar를 붙여 재측정
4. 필요하면 `Holo2`를 reference로 추가 비교

## 8. 최종 권고

- **질문 1 답**: `MAI-UI`는 모바일 전용이 아니다. mobile-first이지만 desktop/professional grounding benchmark도 공식적으로 다룬다.
- **질문 2 답**: 복잡한 엔지니어링 화면을 가장 잘 읽게 하려면, 현재 기준 best stack은 `MAI-UI-32B (+Zoom-In)` 또는 `UI-Venus-1.5-30B-A3B`에 `PaddleOCR-VL`을 조합하는 쪽이다.
- 순수 최고 성능 reference는 `Holo2-235B-A22B (Agentic)`다.
- 다만 우리 저장소와 운영 현실을 같이 보면 **`MAI-UI`와 `UI-Venus`를 동일 스크린샷 세트로 A/B 테스트한 뒤, OCR sidecar를 결합하는 방식**이 가장 실무적이다.

## 출처

- MAI-UI-8B model card: https://huggingface.co/Tongyi-MAI/MAI-UI-8B
- MAI-UI GitHub: https://github.com/Tongyi-MAI/MAI-UI
- MAI-UI paper: https://arxiv.org/abs/2512.22047
- UI-Venus-1.5-30B-A3B model card: https://huggingface.co/inclusionAI/UI-Venus-1.5-30B-A3B
- UI-Venus-1.5 paper: https://arxiv.org/abs/2602.09082
- Holo2-235B-A22B model card: https://huggingface.co/Hcompany/Holo2-235B-A22B
- Holo2 blog post: https://huggingface.co/blog/Hcompany/introducing-holo2-235b-a22b
- ScreenSpot-Pro GitHub: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
- ScreenSpot-Pro OpenReview: https://openreview.net/forum?id=XaKNDIAHas
- ScreenSpot-Pro dataset README: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
- PaddleOCR repository: https://github.com/PaddlePaddle/PaddleOCR
