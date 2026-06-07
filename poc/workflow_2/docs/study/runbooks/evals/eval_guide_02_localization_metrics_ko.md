# 평가 방법 입문 (2) — 위치추정 지표 낱낱이

> 시리즈 2번. 선행: [eval_guide_01_classification_vs_localization_ko](eval_guide_01_classification_vs_localization_ko.md)
> 이 문서의 한 줄: **표의 다섯 숫자(rank1_hit / gt_topk / topk!=1 / med_dist / p90_dist)가
> 각각 무엇을 세는지, 작은 예제로 직접 계산해 본다.**

대상 코드: `golden_localization_eval.py` 의 `_cell_stats()` (지표 계산),
`_localize()` (한 프레임 채점), `align_similarity.py` 의 `GT_TOL_NORM = 0.20`.

---

## 0. 먼저 — 한 프레임이 어떻게 채점되나 (`_localize`)

지표를 이해하려면 *한 장*이 어떻게 hit/miss 가 되는지부터 봐야 한다. 코드 흐름:

```
1) 매처가 프레임을 훑어 점수 지도를 만들고, 점수 높은 순으로 후보 N개(top-N)를 뽑는다.
   - 1등 후보의 위치 = best_xy
2) 각 후보 좌표에 align_offset 을 더해 "예측 align point" 로 환산한다.
   (template 중심과 정답 점이 다를 수 있어서. → 4절)
3) 예측 align point 와 정답(crosshair) 사이 거리를 잰다 = dist_norm.
4) dist_norm ≤ 허용오차(GT_TOL_NORM=0.20) 이면 그 후보는 "정답 근처".
   - 1등(best)이 정답 근처 → rank-1 hit
   - 1등은 아니어도 top-N 안 누군가 정답 근처 → in_topk
```

즉 한 프레임은 채점 후 다음 꼬리표를 단다:

- `hit` (bool): 1등이 정답 근처였나
- `in_topk` (bool): top-N 후보 중에 정답 근처가 있었나
- `topk_rank` (int|None): 정답 근처였던 *첫 후보의 등수* (1이면 곧 hit)
- `dist_norm` (float): 1등 예측점이 정답에서 얼마나 떨어졌나

`_cell_stats()` 는 이 꼬리표들을 한 셀(예: `box__inpaint`)의 모든 프레임에 대해 모아 비율/분포로 요약한다.

---

## 1. dist_norm — "얼마나 빗나갔나" (모든 지표의 원재료)

### 1-1. 왜 픽셀이 아니라 "정규화" 거리인가

빗나간 정도를 픽셀로 재면(예: 40px) 이미지/template 크기마다 의미가 달라진다.
작은 template 에서 40px 은 치명적, 큰 template 에선 사소할 수 있다. 그래서 **정규화**한다.

```
dist_norm = (예측점과 정답점 사이 픽셀 거리) / (template 짧은 변의 길이)
```

코드(`_localize`):
```python
short = max(1, min(tw, th))                       # template 짧은 변
dist_norm = hypot(ap - crosshair) / short         # 정규화 거리 (단위 없음)
```

- `hypot` = 유클리드 거리 = √(Δx² + Δy²) = 두 점 사이 직선 거리.
- 짧은 변으로 나누므로 **"template 한 변의 몇 배만큼 빗나갔나"** 라는 단위 없는 비율이 된다.
- 0 = 완벽 명중, 클수록 멀다.

> 비유: "10cm 빗나감"보다 "내 손 한 뼘만큼 빗나감"이 어디서나 통하는 표현인 것과 같다.
> template 크기를 "자(尺)"로 삼아 거리를 잰다.

### 1-2. 허용오차 GT_TOL_NORM = 0.20

연속 거리를 "맞았다/틀렸다"로 바꾸는 **합격선**이다.

```
dist_norm ≤ 0.20  →  hit (정답 근처로 인정)
dist_norm >  0.20  →  miss
```

0.20 = template 짧은 변의 20%. 이 안에 들어오면 실용상 "맞은 것"으로 본다.
(이 값은 cold-start 설정값이다. 실데이터로 조정될 수 있다.)

> 비유: 다트 불스아이에 "반지름 R 의 동그라미"를 그려두고, 그 안이면 명중 인정.
> R = template 의 20%.

---

## 2. rank1_hit_rate — **주 판정 지표** ("한 방에 명중률")

```python
rank1_hit_rate = (hit 인 프레임 수) / (전체 채점 프레임 수)
```

- `hit` = 1등 예측점이 허용오차 안. 즉 **매처가 한 번에 정답을 찍은 비율**.
- 생산에서 장비는 보통 1순위 위치로 바로 간다. 그래서 이게 **현실 성공률에 가장 가깝다**.

### 숫자 예제

10장을 채점했더니 1등 예측점의 `dist_norm` 이:
```
0.03, 0.08, 0.05, 0.31, 0.12, 0.45, 0.07, 0.19, 0.22, 0.10
```
허용오차 0.20 이하인 것: 0.03, 0.08, 0.05, 0.12, 0.07, 0.19, 0.10 → **7장 hit**
```
rank1_hit_rate = 7 / 10 = 0.70
```
"열 번 중 일곱 번은 한 방에 맞힌다."

---

## 3. gt_in_topk_rate — "보기 안에 정답이 있나" (proposer recall)

```python
gt_in_topk_rate = (in_topk 인 프레임 수) / (전체 프레임 수)
```

- `in_topk` = 1등이 아니더라도 **top-N 후보 중 누군가** 가 정답 근처였나.
- 매처의 후보 생성기(proposer)가 정답을 *후보로라도 떠올렸나*를 잰다. 이걸 **recall** 이라 한다.

### rank1_hit 와의 관계 — 항상 gt_in_topk ≥ rank1_hit

1등이 맞았으면(hit) 당연히 후보 안에도 있다(in_topk). 역은 아니다. 그래서:
```
gt_in_topk_rate ≥ rank1_hit_rate   (항상)
```

### 이 둘의 "갭"이 말해주는 것 — 진단의 핵심

| 패턴 | 해석 | 다음 행동 |
|---|---|---|
| rank1 높음, topk 높음 | 후보도 잘 뽑고 1등도 잘 고름 | 좋음. 그대로 |
| rank1 **낮음**, topk **높음** | 정답을 후보엔 넣는데 1등으로 못 올림 | **reranker(재정렬)로 메울 수 있는 갭** |
| rank1 낮음, topk 낮음 | 정답을 후보로조차 못 떠올림 | proposer 자체를 고쳐야 함(더 어려운 문제) |

> 비유: 객관식에서 "정답을 보기 5개 안에 넣었나(topk)" vs "그중 1번으로 골랐나(rank1)".
> 보기엔 늘 있는데 1번을 못 고르면 → "찍는 법(재정렬)"만 고치면 된다.

---

## 4. topk_not_rank1_rate — "리랭킹으로 메울 수 있던 갭의 크기"

```python
topk_not_rank1_rate = (in_topk 이면서 topk_rank > 1 인 프레임 수) / (전체 프레임 수)
```

3절의 "rank1 낮음 / topk 높음" 갭을 **숫자 하나로** 떠낸 것이다.
정답이 후보엔 있었지만(in_topk) 1등은 아니었던(rank>1) 프레임의 비율.

수치 관계(근사):
```
gt_in_topk_rate ≈ rank1_hit_rate + topk_not_rank1_rate
```
(정답이 후보에 있으면, 그건 "1등이었다(rank1_hit)" 또는 "1등은 아니었다(topk_not_rank1)" 둘 중 하나니까.)

### 예제
```
gt_in_topk_rate    = 0.90   (열에 아홉은 후보 안에 정답이 있다)
rank1_hit_rate     = 0.70   (그중 일곱만 1등이었다)
topk_not_rank1_rate= 0.20   (둘은 후보엔 있었지만 순위가 밀렸다)
```
→ "재정렬만 잘하면 이론상 +0.20 까지 rank1 을 끌어올릴 여지." (현재 리랭커는 폐기됐지만 참고 지표.)

---

## 5. median / p90 dist_norm — "명중의 정밀도"

hit 비율은 합격/불합격만 본다. 하지만 *얼마나 아슬아슬하게* 맞았는지도 중요하다.
허용오차 0.20 인데 늘 0.19 로 겨우 걸치면 위태롭고, 늘 0.03 이면 안정적이다.
그래서 거리 분포 자체를 본다.

### percentile(백분위) 개념

값들을 **작은 순서로 줄 세우고** 특정 위치의 값을 보는 것.

- **median (중앙값, p50)**: 한가운데 값. "보통 이 정도 빗나간다."
  - 평균(mean) 대신 median 을 쓰는 이유: 한두 개의 큰 실패(outlier)에 휘둘리지 않아서.
- **p90 (90백분위)**: 하위 90% 가 이 값 이하. "나쁜 쪽 10% 직전", 즉 *최악에 가까운* 경우.

코드(`_percentile`)는 정렬 후 위치를 골라 뽑는다:
```python
dists = sorted(...)               # 작은 순 정렬
median = dists[ round(0.5*(n-1)) ] # 한가운데
p90    = dists[ round(0.9*(n-1)) ] # 위쪽 10% 경계
```

### 예제
정렬된 dist_norm (10장):
```
0.03 0.05 0.07 0.08 0.10 0.12 0.19 0.22 0.31 0.45
                     ↑p50              ↑p90
```
- median ≈ 0.10 → 보통은 template 의 10% 정도만 빗나감 (양호)
- p90 ≈ 0.31 → 나쁜 경우엔 31% 까지 빗나감 (허용오차 0.20 초과 = 그런 경우는 miss)

> 읽는 법: **median 으로 평소 실력**, **p90 으로 최악 근처**를 본다. hit 비율이 같아도
> p90 이 낮은 쪽이 더 믿을 만하다(경계에서 덜 위태롭다).

---

## 6. align_offset — 이 보정을 빼먹으면 점수가 거짓말을 한다

위치추정 지표가 정확하려면 **"예측 점"을 올바른 자리로 환산**해야 한다. 여기서 함정이 하나 있다.

- 매처가 찍는 좌표 = **template 의 중심** (잘라낸 조각의 한가운데).
- recipe 가 진짜 원하는 점 = **align point** (보통 *이미지 전체의 중심*).
- 이 둘이 **다를 수 있다**. 특히 흰 box 가 이미지 중앙이 아니라 한쪽에 치우쳐 있으면.

그래서 보정 벡터를 더한다:
```
align_offset = (이미지 중심) − (template 중심)
예측 align point = (매처가 찍은 점) + align_offset
```

- `center` template: crop 중심 = 이미지 중심 → offset = (0, 0) (보정 불필요)
- `box` template: box 가 치우치면 → offset ≠ 0 (보정 필수)

**왜 중요한가:** 이 보정을 빼면 `box` 셀이 "정답까지 거리"가 아니라 "box 중심까지 거리"를
재게 되어, 새 방식이 실제보다 나쁘게/좋게 *잘못* 보인다. (실제로 2026-06-04 Codex 리뷰에서
이 누락 버그를 잡았다.) self-test 의 `RCP_B`(일부러 box 를 치우친 케이스)가 이 보정을 지키는지
감시하는 회귀 가드다.

---

## 7. GT 위생(hygiene) — 지표를 믿어도 되는지부터 본다

지표 숫자가 아무리 예뻐도, **정답(GT) 자체가 부실하면** 그 위 숫자는 의미가 없다.
그래서 표 위에 *정답 신뢰도 게이트*를 먼저 찍는다(`_summarize` 의 `hygiene`):

| 위생 항목 | 무엇을 보나 | 나쁘면 |
|---|---|---|
| `n_S` | 채점에 쓸 성공 프레임 수 | 적으면 비율이 들쭉날쭉(표본 부족) |
| `crosshair_detect_rate` | S 중 십자선(=정답)을 검출한 비율 | 낮으면 정답을 못 읽은 것 → 표본 줄고 신뢰도↓ (과거 fail-S 에서 ~0.79) |
| `mean_crosshair_conf` | 검출 자신감 평균 | 낮으면 정답 좌표 자체가 흔들림 |
| `n_E`, `n_question` | S여야 할 폴더에 섞인 E/`?` | 라벨 오염 경고 |

> 원칙: **검출률이 낮으면, 그 위에서 나온 rank1_hit 을 믿지 말고 검출기부터 고친다.**
> 답안지 절반이 안 읽히는 시험의 평균 점수를 신뢰할 수 없는 것과 같다.

---

## 8. 한 줄 치트시트

```
dist_norm           : 빗나간 거리 ÷ template 짧은 변 (작을수록 좋음, 0=완벽)
허용오차 0.20        : dist_norm 이 이 안이면 hit
rank1_hit_rate      : 1등이 hit 인 비율 ★주 판정 (높을수록 좋음)
gt_in_topk_rate     : 정답이 top-N 후보에 든 비율 (recall, ≥ rank1)
topk_not_rank1_rate : 후보엔 있지만 1등은 아닌 비율 (리랭킹 여지)
median dist_norm    : 평소 빗나간 정도 (낮을수록 좋음)
p90 dist_norm       : 최악 근처 빗나간 정도 (낮을수록 안정적)
align_offset        : template 중심 → align point 보정 (빼먹으면 box 점수 왜곡)
GT 위생             : 정답 신뢰도 게이트 — 검출률 낮으면 위 숫자 믿지 말 것
```

---

## 9. 다음 단계

→ [eval_guide_03_experiment_design_ko](eval_guide_03_experiment_design_ko.md)
(이 지표들을 **어떤 비교 구조(2×2 A/B)** 에 얹어 "새 방식이 옛 방식보다 나은가"를
판정하는지. confound, paired, data leakage 까지.)
