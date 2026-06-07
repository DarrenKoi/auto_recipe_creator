# 평가 방법 입문 (0) — 전체 그림 + 용어 사전

> 대상 독자: CV/영상처리를 처음 접하는 사람.
> 목적: `golden_localization_eval.py` 가 쓰는 평가 용어를 **밑바닥부터** 설명한다.
> 이 시리즈를 읽고 나면 [golden_localization_office_runbook](golden_localization_office_runbook.md)
> 의 stdout 표를 혼자 해석할 수 있다.

이 문서는 시리즈의 0번(지도)이다. 순서대로 읽기를 권한다:

| # | 파일 | 한 줄 |
|---|---|---|
| 0 | **이 문서** | 전체 그림 + 용어 한눈에 |
| 1 | [eval_guide_01_classification_vs_localization_ko](eval_guide_01_classification_vs_localization_ko.md) | "분류"와 "위치추정"의 차이 — 왜 이 구분이 평가의 출발점인가 |
| 2 | [eval_guide_02_localization_metrics_ko](eval_guide_02_localization_metrics_ko.md) | rank1_hit / gt_in_topk / dist_norm — 표의 각 숫자가 뭘 재나 |
| 3 | [eval_guide_03_experiment_design_ko](eval_guide_03_experiment_design_ko.md) | 2×2 A/B, confound, paired, leakage — 왜 셀이 4개이고 어떻게 비교하나 |

---

## 1. 우리가 풀려는 문제 (1분 요약)

SEM(전자현미경) 장비는 웨이퍼 위의 **정해진 위치**를 찾아가서 측정해야 한다.
그 "정해진 위치"를 찾는 단서가 **align key**(등록해 둔 기준 이미지 조각)다.
장비가 이 위치를 못 찾으면 **align fail** 이 난다.

우리 코드(`align_key_matcher.py`)는 등록 이미지(rcp)를 들고 현재 화면(msr)을 훑어서
**"여기가 그 위치다"** 라고 좌표 하나를 찍는다. 이게 잘 되는지 **점수로 증명**하는 것이
평가(evaluation)이고, 그 스크립트가 `golden_localization_eval.py` 다.

> 핵심 한 줄: **평가 = "우리 매처가 찍은 점이 진짜 정답 위치에 떨어지나"를 숫자로 재는 일.**

---

## 2. 왜 "golden" 데이터인가

평가를 하려면 **정답(ground truth)** 이 있어야 한다. 채점하려면 답안지가 필요한 것과 같다.

- 성공(**S**) 프레임에는 장비가 실제로 측정한 자리에 **십자선(crosshair)** 이 그려져 있다.
  이 십자선이 곧 **정답 좌표**다.
- 실패(**E**) 프레임의 십자선은 정의상 *틀린 자리거나 아예 없다*. 그래서 답안지로 못 쓴다.

그래서 평가는 **항상 성공한(S-only) 데이터** = `align_images_golden/` 위에서만 한다.
이걸 "golden set"이라 부른다. ("정답을 믿을 수 있는 데이터"라는 뜻.)

자세한 이유: 시리즈 1번.

---

## 3. 용어 사전 (한 줄 정의 + 비유)

처음 보면 외계어 같은 단어들이다. 여기서 한 번에 훑고, 깊은 설명은 1·2·3번 문서로 넘긴다.

### 3-1. 평가 자체에 대한 말

| 용어 | 한 줄 정의 | 일상 비유 |
|---|---|---|
| **ground truth (GT, 정답)** | 우리가 맞혀야 할 진짜 답 | 시험 답안지 |
| **classification (분류)** | "있나/없나", "S냐 E냐" 같은 **범주**를 맞히는 일 | O/X 문제 |
| **localization (위치추정)** | 이미지 안에서 **좌표(점)** 를 맞히는 일 | 지도에 핀 꽂기 |
| **metric (지표)** | 성능을 요약한 숫자 | 점수, 등수 |
| **prediction (예측)** | 우리 코드가 내놓은 답 | 학생이 적은 답 |

### 3-2. 위치추정 지표 (시리즈 2번에서 깊게)

| 용어 | 한 줄 정의 | 직관 |
|---|---|---|
| **rank-1 hit** | 1순위로 찍은 점이 정답 근처에 떨어짐 | "한 방에 명중" |
| **rank1_hit_rate** | 전체 중 rank-1 hit 비율 | **주 판정 지표** |
| **top-N candidates** | 점수 높은 순으로 추린 후보 N개 | "정답 후보 8개 뽑기" |
| **gt_in_topk_rate** | 정답이 그 후보 N개 안에 든 비율 | "보기 안에 답이 있나" |
| **topk_not_rank1_rate** | 후보엔 있지만 1등은 아니었던 비율 | "답을 알지만 순위가 밀림" |
| **dist_norm** | 찍은 점과 정답 사이 거리(정규화) | "얼마나 빗나갔나" |
| **median / p90** | 거리 분포의 중앙값 / 상위 10% 경계 | "보통/최악에 가까운 경우" |
| **tolerance (허용오차)** | 이만큼 안에 들면 hit 으로 인정 | "정답 동그라미 반지름" |

### 3-3. 실험 설계 용어 (시리즈 3번에서 깊게)

| 용어 | 한 줄 정의 | 직관 |
|---|---|---|
| **A/B test** | 옛 방식 vs 새 방식을 같은 조건에서 비교 | 약 vs 위약 |
| **ablation (절제 실험)** | 변수를 하나씩 끄고 켜서 효과를 분리 | 재료 하나씩 빼보기 |
| **2×2 factorial** | 변수 2개 × 각 2상태 = 4조합 표 | 2단×2단 격자 |
| **confound (교란변수)** | 결과를 왜곡하는 숨은 제3의 원인 | "운동 효과? 식단 효과?" |
| **paired / unpaired** | 같은 표본끼리 비교했나 / 다른 집단인가 | 같은 사람 전후 vs 다른 사람들 |
| **delta (Δ)** | 새 − 옛, 즉 변화량/효과 크기 | "몇 점 올랐나" |
| **data leakage (정답 유출)** | 정답이 입력에 섞여 점수가 부풀려짐 | 답안지 보고 시험 |

### 3-4. 매처 내부(점수를 만드는 CV) 용어 — 참고

표의 숫자를 만들기 *전에* 매처가 하는 일. 깊은 설명은 [matcher 쪽 문서] 참고, 여기선 감만.

| 용어 | 한 줄 정의 |
|---|---|
| **template** | 비교 기준이 되는 작은 이미지 조각(등록 이미지에서 잘라냄) |
| **Chamfer matching** | 윤곽선(edge)끼리 얼마나 겹치나로 점수 매기는 방법 (픽셀 색이 아니라 *모양* 기준) |
| **distance transform** | 각 픽셀이 가장 가까운 edge 까지 몇 픽셀 떨어졌나를 담은 지도 |
| **ORB + RANSAC** | 특징점(코너 등)을 매칭해 회전/크기까지 검증하는 보조 점수 |
| **score map** | 화면 위 모든 후보 위치의 점수를 담은 2D 지도 |
| **NMS (non-max suppression)** | 점수 지도에서 봉우리(peak)만 골라 겹치는 후보 제거 |
| **align_offset** | template 중심 ≠ 정답 점일 때 둘의 차이를 보정하는 벡터 |
| **inpaint** | 이미지에서 특정 부분(여기선 십자선)을 자연스럽게 지우기 |

---

## 4. 한 장의 흐름도 — 평가가 한 프레임을 어떻게 채점하나

```
[등록 이미지 rcp]                         [현재 측정 프레임 msr (S, 십자 있음)]
      │                                              │
      │ template 만들기                               │ ① 십자선 검출 → 위치 = 정답(GT)
      │ (center 면적 crop / box 안쪽 crop)            │ ② 그 자리 inpaint 로 지움 (정답 가리기)
      ▼                                              ▼
   template ───────────► 매처(Chamfer+ORB) ◄──── 십자 지운 프레임
                              │
                              │ score map → top-N 후보 → best 1개
                              ▼
                  찍은 점 + align_offset = "예측 align point"
                              │
                              ▼
          예측 점 ↔ 정답(GT) 거리 = dist_norm
          거리 ≤ 허용오차 ?  →  YES: rank-1 hit / NO: miss
```

이 한 장이 시리즈 전체의 뼈대다. 각 단계의 "왜"가 1·2·3번 문서다.

- ①②에서 *정답은 먼저 기록하고, 입력에선 지운다* — 이게 leakage 방지(시리즈 3).
- "best 1개"가 정답 근처면 rank-1 hit, "후보 N개" 안에 있으면 gt_in_topk (시리즈 2).
- template 을 center 로 만드냐 box 로 만드냐 × 프레임을 raw 로 보냐 inpaint 로 보냐
  = 4가지 조합(2×2) 으로 각각 채점 (시리즈 3).

---

## 5. 다음 단계

→ [eval_guide_01_classification_vs_localization_ko](eval_guide_01_classification_vs_localization_ko.md)
("분류"와 "위치추정"의 차이부터. 이 스크립트가 *왜 따로 생겼는지*의 핵심.)
