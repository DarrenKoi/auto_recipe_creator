# Registration Verifier Arm 확장 — phase_ecc / grad_phase / MIND / RRF fuse

날짜: 2026-07-20 · 커밋: `22913b5` (main, push 완료)

## 0. 이 작업이 무엇인지 — 쉬운 말로 풀어쓴 배경

### 우리가 풀고 있는 문제

CD-SEM 장비에서 align fail이 나면, 레시피에 등록된 "align key" 사진(작은 기준 패턴)을
현재 화면(live SEM 이미지) 어디에 있는지 찾아서 그 위치로 스테이지를 옮겨야 한다.
"작은 사진을 큰 사진 안에서 찾는" 이 작업을 **template matching**이라고 부른다.

우리 매칭 파이프라인은 두 단계다:

1. **Proposer(후보 제안자)** — 화면 전체를 훑어서 "여기 같아요" 하는 후보를 8곳 정도
   점수 순으로 내놓는다. 현재 production은 3채널(edge 표현 3가지) 앙상블 + RRF다.
2. 지금까지는 proposer의 1순위를 그대로 믿었다. 그런데 골든 데이터 평가에서
   **정답이 후보 8개 안에는 있는데 1순위가 아닌 경우(rank_error)** 가 꽤 있었다.
   반도체 패턴은 비슷한 모양이 반복되기 때문에, 진짜 위치와 닮은 "가짜 후보"가
   점수를 더 받는 일이 생긴다.

이번 실험(registration verifier)은 그 rank_error를 노린다. 아이디어는 단순하다:

> 후보 8곳 각각의 주변을 잘라내서(crop), 기준 사진과 **정밀하게 다시 맞춰보고**
> "얼마나 잘 맞는가"를 다른 방식으로 재채점한 뒤 순위를 다시 매기자.

이 재채점기를 **verifier(검증자)** 라고 부른다. 전역 탐색은 하지 않고(그건 proposer 몫),
이미 나온 후보 안에서만 순위 교정 + 좌표 미세 보정을 한다. 그래서 실패해도 원래
순위로 돌아가는 안전한 구조다(전부 거부되면 기존 동작 B0로 강등).

### "registration"이란 말

의료영상·위성사진 분야 용어로, 두 이미지를 **픽셀 단위로 겹치도록 정렬**하는 것을
image registration이라 한다. "후보 위치가 대충 맞는데 몇 픽셀 어긋났다"를 찾아내는
데 딱 맞는 도구 상자라서 이 이름을 썼다.

## 1. 각 방법의 이미지 처리 원리 (이번에 추가/기존 arm 전부)

### ECC (기존, P0-A) — 밝기 지도를 미끄러뜨려 맞추기

Enhanced Correlation Coefficient. 두 이미지의 밝기 패턴이 최대한 겹치도록 이동량을
**경사 하강(gradient descent)** 식으로 조금씩 조정한다. 등산로에서 한 걸음씩 내려가듯
"조금 왼쪽으로 밀면 더 닮아지나?"를 반복해서 수렴한다. 밝기의 절대값이 아니라
상관계수(correlation)를 쓰기 때문에 전체적으로 어두워지거나 대비가 바뀌어도 견딘다.

- 장점: sub-pixel(1픽셀 이하) 정밀도.
- 약점: **capture range가 좁다** — 시작 위치가 정답에서 많이 어긋나 있으면(수 px 이상)
  엉뚱한 골짜기로 내려가 오수렴한다. 국소 최적화의 태생적 한계.

### Phase correlation (기존, P1-C) — 주파수 영역에서 이동량을 한 번에 읽기

푸리에 변환(Fourier transform)은 이미지를 "무늬의 주파수 성분"으로 분해한다.
**Fourier shift theorem**: 이미지가 평행이동하면 주파수 성분의 크기는 그대로고
**위상(phase)만 이동량에 비례해 회전**한다. 그래서 두 이미지의 위상 차이만 뽑아
역변환하면 이동량 위치에 뾰족한 피크 하나가 서고, 피크 위치가 곧 (dx, dy)다.

- 장점: 반복 없이 **한 번에, crop 전체 범위의** 이동량을 얻는다. 피크의 선명도
  (response)가 신뢰도 지표로 공짜로 나온다.
- 약점: 순수 평행이동 전용(회전·배율 못 다룸), 정밀도는 ECC보다 조금 낮음.
- Hann window: 이미지 가장자리를 부드럽게 눌러주는 창함수. FFT는 이미지가 무한히
  반복된다고 가정하므로 가장자리의 불연속이 가짜 주파수를 만드는데, 이를 막는다.

### phase_ecc (신규) — 두 방법의 약점을 서로 지우는 cascade

phase로 "대충 어디로 가야 하는지"를 전역에서 한 번에 알아내고(넓은 capture range),
그 값을 ECC의 시작점(warp 초기값)으로 넣어 sub-pixel까지 다듬는다(높은 정밀도).
망원경으로 방향을 잡고 현미경으로 초점을 맞추는 격. 합성 테스트에서 초기 오차
20px짜리 후보도 truth ±2px 안으로 복원됨을 고정했다(`_t_phase_ecc_capture_range`).

### grad_phase (신규) — 밝기 대신 "밝기의 변화율"로 phase 하기

SEM은 촬영 조건에 따라 화면 전체가 밝아지거나 대비가 뭉개지는 **contrast drift**가
있다. 이런 변화는 주로 저주파(넓게 퍼진 성분)에 실린다. Sobel 필터로 각 픽셀에서
"밝기가 얼마나 급하게 변하는가"(gradient magnitude = 윤곽선 세기 지도)를 만들면
절대 밝기·DC 성분이 소거되고 구조(edge)만 남는다. 그 위에서 phase correlation을
돌리는 것이 grad_phase다. 연구 문서 P1-C가 제안한 세 표상 중 두 번째.

### SIFT / AKAZE + RANSAC (기존, P0-B) — 특징점 대응 + 다수결 기하 검증

- **keypoint 검출**: 모서리·얼룩처럼 "어디서 봐도 알아볼 수 있는 점"을 양쪽 이미지에서
  수십 개 찾고, 각 점 주변의 생김새를 숫자 벡터(descriptor)로 요약한다.
  SIFT는 gradient 히스토그램(실수 벡터), AKAZE는 비선형 스케일 공간의 이진 벡터.
- **matching**: 벡터가 가장 닮은 점끼리 짝짓는다. Lowe ratio test로 "1등이 2등보다
  확실히 가까운" 짝만 남긴다(애매한 짝 제거).
- **RANSAC**: 짝 중에는 틀린 것(outlier)이 섞여 있다. 무작위로 몇 쌍을 뽑아 변환을
  가정하고 "이 변환에 동의하는 짝(inlier)이 몇 개냐"를 반복 투표해, 다수가 동의하는
  변환 하나를 뽑는다. 잡음에 매우 강한 다수결 원리.
- 우리는 limited-affine(이동+회전+배율)만 허용하고, 회전 ±15°/배율 0.75~1.3 밖이면
  물리적으로 불가능하다고 거부한다. inlier가 template 한 구석에만 몰려 있으면
  (coverage gate) "부분만 닮은 가짜"로 보고 거부한다.

### MIND (신규, P1-D) — "주변과 얼마나 닮았나"의 지도를 비교하기

Modality-Independent Neighbourhood Descriptor(원래 CT↔MRI 같은 서로 다른 촬영
방식의 의료영상 정합용). 각 픽셀에서 밝기 자체가 아니라 **"내 주변 8방향 패치와
나는 얼마나 닮았는가"** 라는 자기유사성(self-similarity) 패턴을 기술자로 만든다.
밝기·대비가 통째로 변해도 "구조가 자기 주변과 닮은 방식"은 유지되므로 drift에
불변에 가깝다. 이번 구현은 문서 설계 1단계 그대로 **score-only** — 이동량은 내지
않고(refined=cand) 후보 재채점·재정렬만 한다(`SCORE_ONLY_ARMS`).

### fuse (신규) — 서로 다른 verifier들의 순위 다수결 (RRF)

각 arm의 점수는 척도가 달라 직접 비교할 수 없다(ECC cc ∈ [0,1], SIFT는 inlier 수…).
그래서 점수 대신 **순위**를 합친다. Reciprocal Rank Fusion: 후보가 어떤 arm에서
r위면 1/(k+r)점을 받고, 전 arm 합산 점수로 최종 순위를 정한다(proposer 앙상블과
동일 원리, k=8). 한 방법이 자신만만하게 틀려도(false positive) 독립적인 다른
방법들이 동의하지 않으면 1위로 못 올라온다 — 연구 문서 §4.3의 안전장치.
동점은 baseline 순위 우선(보수적), 전 arm 거부 시 B0로 강등.

### 평가 지표 읽는 법

- **B0**: 기존 proposer 1순위 그대로 (기준선).
- **raw vs ref**: raw는 순위만 바꿨을 때(원좌표), ref는 순위+정밀화 좌표까지 —
  "순위 교정 효과"와 "sub-pixel 보정 효과"를 분리해 본다.
- **promote / regress**: B0가 틀렸는데 arm이 맞춤 / B0가 맞았는데 arm이 망침.
  regress가 안전성 지표(낮아야 채택 가능).
- **gt_in_topk 부분집합 표**: verifier는 정답이 후보 안에 있을 때만 도울 수 있다.
  이 표가 오르지 않으면 전체 평균 개선은 우연(버킷 구성 차이)이다.
- **err_med (신규)**: hit 행의 GT 거리 중앙값 `B0>arm` — rank와 별개로 좌표가
  실제로 몇 px 더 정확해졌는지 보는 축.

## 2. 진행 사항

1. 최근 커밋 `144c6a8`(registration verifier lab + 샘플링 드라이버, 미테스트 상태) 코드
   리뷰 — hook 계약(`align_similarity._consensus_template_ab`의 `combined_renderer` ctx),
   offset×scale 좌표 규약, tolerance 계산이 harness와 일치함을 확인. 구조적 결함 없음.
2. 합성 self-test 최초 실행 → 14/14 통과 확인. 드라이버 no_data 경로(Mac) 정상 확인.
3. 연구 문서 P1 미구현 항목 3종 + verifier 합의 융합을 구현(아래 수정 내용).
4. 확장 후 검증: lab self-test **24/24**, 신규 pytest **5/5**, 인접 드라이버 회귀
   **153 passed**, no_data 스모크 정상.
5. 사용자 지시 반영: 평가 이력이 memory에 없으면 `docs/project_progress/` 참고 —
   memory(`feedback_project_progress_docs_for_eval_history.md`)로 저장.

## 3. 수정 내용

- `poc/workflow_2/registration_lab.py`
  - `ecc_refine(init_shift=)` 파라미터 추가(warp translation 초기값).
  - 신규 arm: `phase_ecc_refine`(cascade), `grad_phase_refine`(+`_grad_mag`),
    `mind_verify`(+`_mind_descriptor`, `_zncc`; score-only).
  - `rrf_fuse_orders`(RRF 순위 융합), `SCORE_ONLY_ARMS`, `FUSE_RRF_K`, MIND 상수군.
  - SIFT/AKAZE detector `_DET_CACHE` 재사용(후보마다 재생성 제거).
  - self-test 14→24개: `_t_phase_ecc_capture_range`(20px 초기 오차 복원),
    `_t_mind_score_only`(shift=0 계약), `_t_fuse_orders`, shift/scale 테스트를 새 arm에 확장.
- `poc/workflow_2/golden_registration_eval_cond.py`
  - 기본 `ARMS` = `REG_ARM_NAMES` 7종 전체; `ALIGN_REG_FUSE`/`REG_FUSE`(기본 1, arm≥2)로
    `fuse` 의사-arm 자동 집계.
  - `_RegAccum._tally` 신설 — arm/fuse 공통 per-point 집계(중복 제거); fuse 는
    fallback 아닌 arm 순열의 RRF 합의 + ok shift-arm refined 평균 좌표.
  - `_median` + `err_b0_med_px`/`err_ref_med_px`(수집만 되고 버려지던 err 리스트 활용),
    표에 `err_med` 컬럼, summary/config에 `fuse` 기록, overlay 색상 7 arm+fuse 확장.
- `poc/workflow_2/test_golden_registration_eval_cond.py` **신규** — `_RegAccum` 합성
  e2e(실제 ensemble proposer를 합성 frame에 재실행해 hook 계약 전체 검증), `_tally`
  promote/regress 분해, 전 arm 거부 시 fuse B0 강등, `_median` 테스트. pytest 5개.
- `poc/workflow_2/docs/study/cv/align_fail_cv_methods_research_ko.md` — §4.5 구현 현황
  (2026-07-20) 추가.

## 4. 다음 단계

- **오피스에서 골든 A/B 실행**: `uv run python poc/workflow_2/golden_registration_eval_cond.py`
  (인자 없음; 기본 7 arm + fuse, 표본 40 recipe). `[DIGEST]` 한 줄만 Mac으로 보고.
- 채택 판정: gt_in_topk 부분집합 rank1 상승 + regress 낮음이 진짜 개선. rank_error
  버킷 크기가 기대 상한이므로 버킷 분류표를 먼저 볼 것.
- 유효 arm이 나오면 workflow_2에서 상수(게이트 임계) 보정 → 검증 후에만 workflow_3
  포팅(기존 전환 규칙 그대로).
- MIND가 rank-1 이득을 보일 때만 dense proposer 승격 검토(문서 중단 조건).

## 5. 메모리 업데이트

- `project_registration_verifier_lab.md` — 7 arm + fuse 확장, err_med 축, detector 캐시,
  신규 테스트 파일 반영(2026-07-20 항목 추가). MEMORY.md 인덱스 라인 갱신.
- `feedback_project_progress_docs_for_eval_history.md` **신규** — 평가 이력이 memory에
  없으면 `docs/project_progress/`(03=workflow_2 CV bench) 참고. MEMORY.md 인덱스 추가.
