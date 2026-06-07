# Align 평가 지표·용어 정리 (학습용)

> 대상 코드: `poc/workflow_2/align_similarity.py`
> 목적: align point(green mark) 매칭을 **진단**할 때 콘솔에 찍히는 지표·용어
> (`in_topk`, `bACC`, `truth-forced`, `point_biserial`, `staleness` …)를 한국어로 정리.
> 관련: [recovery plan](../../journals/260529/260529_align-point-correction-recovery-plan.md),
> [match 알고리즘 intro](../algorithms/match_algorithms_intro.md)

---

## 0. 먼저 — 이 지표들은 "성능 점수"가 아니라 "원인 분리(triage) 도구"다

가장 흔한 오해부터 짚자. 아래 지표들은 *"우리 시스템이 몇 점이다"* 를 매기는 점수가 아니다.
**"매칭이 틀렸을 때 *어디가* 틀렸는지"를 좁히는 진단 도구**다.

우리가 풀려는 문제는 하나다:

> recipe 에 등록된 align key(rcp 이미지의 정중앙 영역)를, 실제 측정 이미지(msr)에서
> **같은 물리적 위치**로 다시 찾아내야 한다. 그런데 S(성공) 이미지에서조차 엉뚱한 위치가
> green mark 로 찍힌다. **왜?**

"왜"의 후보는 여러 개다 — key 자체가 유일하지 않다 / 매칭기(matcher)가 약하다 / scale 을 놓쳤다 /
reference(rcp)가 낡았다(stale) … 이걸 *눈으로* 가릴 수 없으니, **각 가설을 숫자로 분리**하려고
만든 게 이 지표들이다. 그래서 지표마다 "이 값이 이러면 → 원인은 저것" 이라는 **판정 규칙**이 붙어 있다.

> 핵심 사실(오피스 실데이터 432장으로 확인): **정답 위치(S-at-crosshair)에서조차 matcher
> median 점수가 0.62~0.74 수준이고, free 검색(free_best)은 S(0.66)보다 E(0.70)에서 오히려
> 더 높다.** 정답에서 이 정도면, free 검색이 다른 곳에서 우연히 더 높게 나와 green mark 가
> 틀린다. = **matcher 변별력(distinctiveness) 부족**이 진짜 천장.
>
> ⚠️ 이 문서의 콘솔 숫자는 **오피스 실측(432장, 2026-05-29)** 을 텍스트로 회신받은 값이다
> (`feedback_no_office_data_to_mac`). 일부는 손전사라 자릿수에 noise 가 있을 수 있으나
> 추세는 분명하다.

---

## 1. 모든 것의 기준: ground truth (S/E 라벨)과 "세 위치"

### 1.1 S/E 라벨 = ground truth

장비(tool)는 측정 결과를 스스로 S(success) / E(fail) 로 라벨링한다. 파일명 `S*` / `E*` 가 그것.
우리는 이걸 **정답(ground truth)** 으로 쓴다 — "S 면 align 이 제대로 됐어야 한다, E 면 실패다".

- **S 이미지**: 정렬 성공 → align key 가 제 위치(보통 crosshair 위치)에 있어야 한다.
- **E 이미지**: 정렬 실패 → key 를 못 찾았거나 엉뚱한 데 있다.

> ⚠️ 주의: S 라벨도 100% 믿진 않는다(`feedback_doubt_s_labels`). 도구의 self-report 라
> false-positive 가능. 그래서 라벨은 **통계의 기준**으로만 쓰고, 개별 좌표 판정엔 CV 를 쓴다.

### 1.2 "세 위치"에서 유사도를 잰다

한 장의 msr 이미지에서 rcp-center template 을 **세 군데**에 대고 점수를 잰다. 이 셋의 대비가
진단의 출발점이다 (`_process_msr`):

| 위치 | 코드 | 뜻 | S 에서 기대 | E 에서 기대 |
|---|---|---|---|---|
| `at_crosshair` | crosshair 검출 위치 ROI | 도구가 "여기가 정답"이라 본 곳 | **높음** | (E 는 crosshair 거의 없음) |
| `at_center` | 이미지 정중앙 ROI | 정렬 성공이면 key 가 중앙에 옴 | 높음 | 낮음 |
| `free_best` | **이미지 전체** free 검색 best | "key 가 *어디든* 있나?" | 높음 | 이동 복구 가능하면 높음 |

**왜 free_best 가 중요한가:** `free_best` 가 `at_crosshair`(정답)보다 높으면, matcher 가 **정답이
아닌 다른 곳에 lock-on** 했다는 직접적인 증거다. 정답 위치(at_crosshair)를 알려줘도 자유 검색이
더 높은 점수의 다른 곳을 찾아냈다는 뜻 → green mark 가 틀리는 메커니즘 그 자체다.

> 실데이터 도메인 사실: **E 이미지엔 crosshair 가 거의 없다**(`project_e_images_no_crosshair`).
> 그래서 "E = crosshair 가 틀린 위치" 유형은 사실상 없고, E 는 순수 image-matching 으로 풀어야 한다.

---

## 2. 어떤 "지표(metric)"로 유사도를 재나 — matcher / MI / NCC

같은 두 crop 의 닮음을 재는 방법이 여러 개다. 각각 강점/약점이 다르므로 **셋을 동시에** 재서
"무엇이 S/E 를 가장 잘 가르나"를 비교한다.

| 지표 | 코드 | 무엇을 보나 | drift(밝기/대비 변화)에 |
|---|---|---|---|
| **matcher score** | `_score` (Chamfer+ORB) | edge 모양(기하) 일치 | 비교적 강건, 단 edge 빈약하면 약함 |
| **MI** (mutual information) | `_mi` | 두 밝기 분포의 **상호정보량** | **강건** (밝기/대비 drift 에 둔감) |
| **NCC** (정규화 상관) | `_ncc` | pixel 값의 선형 상관 | **약함** (pixel 동일성 가정 → baseline) |

- **Chamfer / ORB**: 매칭의 주력. [match 알고리즘 intro](../algorithms/match_algorithms_intro.md) 참고.
- **MI**: 두 이미지를 겹쳐 2D 히스토그램을 만들고 "한쪽을 알면 다른 쪽의 불확실성이 얼마나 줄어드나"를
  잰다. 공정 변화로 픽셀값이 통째로 바뀌어도 *구조적 대응*이 남으면 높게 나온다. 그래서 SEM
  align 처럼 **픽셀 동일성을 가정할 수 없는**(`project_align_key_matching_constraint`) 문제에 적합.
- **NCC**: 일부러 약한 baseline 으로 둔다. "픽셀만 봐도 풀리나?"의 대조군.

> 실데이터에서 확인된 것: matcher(chamfer)는 점수면이 평평해서 변별력이 약했고(med_S 0.62 vs
> med_E 0.61, 거의 안 갈림), **MI 만 약하게 변별**(bACC 0.627)했다. 그래서 recovery plan 의
> Phase 3 에서 MI 를 **reranker**(후보 재정렬기)로 채택하는 방향이 나왔다.
>
> 👉 MI reranker 가 *구체적으로 후보를 어떻게 재정렬하는지*(멤버십 불변·순서만 변경, 순환성,
> `rerank_rank1_lift` 읽는 법)는 [MI reranker intro](mi_reranker_intro.md) 참고.

---

## 3. S/E 분리도(separation) — `bACC`, `thr`, `med_S`, `med_E`

콘솔 첫 표가 이것 (오피스 실측 432장):

```text
[INFO] S/E 분리도 (median_s 높고 median_e 낮을수록, balanced_accuracy 1.0 에 가까울수록 좋은 지표):
  metric           med_S    med_E      thr   bACC  n(S/E)
  free_best       0.6637   0.7038   0.408  0.549  234/198   ← med_E > med_S (역전!)
  at_crosshair    (S 높음)  (E 낮음)  ...    0.52   229/71    ← E 는 crosshair 거의 없음
  mi_free         0.0749   0.0365   0.0477 0.627  234/198   ← 그나마 가장 높음
  ncc_free        0.0203  -0.0006   0.0145 0.598  234/198
  free_best_box   0.7007   0.7174   0.3602 0.61   210/161   ← box-crop 도 역전
```

이 표가 답하는 질문: **"이 지표(metric)는 S 와 E 를 가를 수 있는가?"**

### 용어

- **med_S / med_E**: S 그룹 / E 그룹 점수의 **중앙값(median)**.
  좋은 지표라면 *med_S 는 높고 med_E 는 낮아야* 한다 (= 두 분포가 떨어져 있어야).
- **thr (best_threshold)**: S/E 를 가장 잘 가르는 **임계값**. 코드는 가능한 모든 임계를 훑어
  (`for t in sorted(set(s + e))`) bACC 를 최대화하는 t 를 고른다.
- **bACC (balanced accuracy, 균형 정확도)**: 이 진단의 핵심 한 줄.

### bACC 가 뭔가 — 왜 그냥 "정확도"가 아니라 "균형" 정확도인가

```text
TPR (S 를 맞춘 비율) = (점수 >= thr 인 S 개수) / (전체 S)     # 민감도(sensitivity)
TNR (E 를 맞춘 비율) = (점수 <  thr 인 E 개수) / (전체 E)     # 특이도(specificity)
bACC = 0.5 * (TPR + TNR)
```

(코드: `_separation`, line 443~)

그냥 정확도(accuracy)는 **클래스 불균형에 속는다.** 예를 들어 E 가 전체의 90%면, "전부 E"라고
찍기만 해도 정확도 90%가 나온다 — 쓸모없는데 점수만 높은 것이다. **bACC 는 S 정답률과 E 정답률을
따로 구해 평균**하므로, 한쪽만 잘 맞혀서는 점수가 오르지 않는다.

| bACC | 해석 |
|---|---|
| **1.0** | 완벽 분리. 어떤 임계로 S/E 가 깔끔히 나뉨 (self-test 합성 데이터가 이 값) |
| **0.5** | **동전 던지기.** 이 지표는 S/E 를 전혀 못 가름 (= 쓸모없음) |
| **< 0.5** | 역전. E 가 오히려 S 보다 점수가 높음 (지표가 거꾸로) |

> 실측에서 `free_best` 의 bACC 가 **0.549** — 거의 동전 던지기. 게다가 med_S(0.66) < med_E(0.70)
> 로 **방향이 살짝 역전**돼 있다(= 매칭 점수가 정렬 실패 이미지에서 더 높게 나옴). 이게
> "matcher 변별력 부족"의 가장 강한 정량적 증거다. MI(0.627)·box-crop(0.61) 만 *약하게* 변별.
> → 단일 지표로는 부족하니 후보화 + reranking 으로 가야 한다는 결론.

---

## 4. truth-forced sweep — "정답 위치에서조차 약한가?" 병목 분리

콘솔 두 번째 블록 (오피스 실측):

```text
[INFO] TRUTH-FORCED (S+crosshair 정답 위치, wide scale 0.5~1.4):
  valid/truth = 76/229  (wrong_local_peak=153)
  median chamfer: wide=0.7409  compare(≤1.0)=0.7381  scale_gain=0.0252  orb=0.0
  edge density median: tpl=0.0699  msr=0.0457
  wide_best_scale hist: {'0.5': 46, '1.0': 26, '0.6': 4}
  진단 counts: {'ok': 71, 'wrong_local_peak': 153, 'edge_problem_msr': 5}
```

(코드: `_truth_forced` line 187, `_diagnose_truth` line 169)

### 아이디어: "정답을 알려주고" 매칭을 강제로 시켜본다

평소 매칭은 "이미지 어디에 key 가 있나?"를 *모르는 채* 찾는다(free search). 그래서 틀려도
"key 자체가 약한 건지, 검색이 엉뚱한 데 끌린 건지" 구분되지 않는다.

truth-forced 는 다르다. **S 이미지의 crosshair 위치 = 정답을 이미 안다.** 그 정답 위치 주변 ROI 에서만,
**넓은 scale 대역**(0.5~1.4)으로 chamfer 를 강제로 재본다. 이렇게 하면 "검색이 틀린 곳에 갔다"는
변수를 제거하고 **순수하게 "정답 위치에서 매칭이 잘 되나?"** 만 볼 수 있다.

### 용어

- **`truth-forced` / forced**: 정답(crosshair) 위치에서 *강제로* 측정한다는 뜻. (free search 의 반대)
- **`wide_chamfer`**: 넓은 scale 대역(0.5~1.4)에서의 best chamfer.
- **`compare_chamfer`**: 생산 경로가 실제 쓰는 좁은 대역(0.6~1.0)에서의 best chamfer.
- **`scale_gain` = wide - compare**: 넓게 봤더니 점수가 *얼마나* 올랐나.
  크면 → **생산 대역이 너무 좁아 scale 을 놓쳤다**(C4 문제)는 신호.
- **`wide_best_scale_hist`**: 정답을 잡은 best scale 의 분포. `1.2`/`1.4` 에 몰리면 msr 이 rcp 보다
  확대된 배율 → 좁은 대역(≤1.0)이 놓침.
- **`truth_valid` / `wrong_local_peak`**: 정답 ROI 안에서조차 best 가 정답 좌표를 벗어났으면
  (`err_norm > 0.20`) `valid=False`, 즉 `wrong_local_peak`. **정답을 알려줘도 그 근처 다른 데가 더
  높게 나온다 = 매칭기가 근본적으로 변별 못 함.**

### diagnosis 5종 (병목 라벨)

`_diagnose_truth` 가 한 줄 진단을 붙인다. **counts 가 어디에 몰리느냐로 다음 액션이 갈린다:**

| diagnosis | 뜻 | 처방 |
|---|---|---|
| `wrong_local_peak` | 정답 위치에서도 best 가 정답을 벗어남 | matcher 변별력 보강 (top-N+reranker) |
| `template_weak` | rcp key 자체 edge 빈약 | **rcp 재등록**(key 품질 문제) |
| `scale_band_problem` | >1.0 에서만 회복됨 | **scale 대역 확장 (C4)** |
| `edge_problem_msr` | msr crop 에 edge 가 거의 없음 | Canny/전처리 조정 |
| `metric_or_reference_problem` | edge 는 있는데 chamfer 낮음 | metric 교체(MI) 또는 reference drift |

> 실측: `wrong_local_peak` **153/229 ≈ 67%** — 정답을 알려줘도 매칭이 근처 다른 데로 샜다.
> 게다가 `scale_gain` 이 0.025 로 거의 0(넓게 봐도 점수가 안 오름)이고 best scale 도 0.5/1.0 에
> 퍼져 1.2/1.4 쏠림이 없다 → **scale 문제(C4)가 아니다.** = "scale 만 고치면 되는" 문제가
> 아니라 **변별력 자체가 문제**라는 강한 증거. (`orb=0.0` 은 mark 류에 keypoint 가 거의 안
> 잡힌다는 뜻 — ORB 가 이 도메인에선 무력.)

---

## 5. gt-in-topK (proposer recall) — `in_topk`, `rank1`, `miss`

콘솔 세 번째 블록. **이 문서에서 제일 중요한 개념.** (오피스 실측)

```text
[INFO] GT-IN-TOPK (S+crosshair, top-8 chamfer 후보의 proposer recall):
  in_topk=105/229 (0.459)  rank1=60 (0.262)  miss=124  median_miss_dist_norm=0.426
  rank_hist(정답이 든 순위): {'1': 60, '2': 5, '3': 8, '4': 8, '5': 11, '6': 6, '7': 3, '8': 4}
```

(코드: `_gt_in_topk` line 256)

### 왜 이게 필요한가 — proposer vs reranker 갈림길

매칭 파이프라인을 2단계로 보면:

```text
1단계 proposer(후보 생성기): 이미지를 훑어 "여기 같은데?" 후보 top-N 개를 뽑는다.
2단계 reranker(재정렬기):    그 후보들을 다시 채점해 1등을 고른다.
```

green mark 가 틀렸을 때, **고칠 수 있는 문제와 못 고치는 문제가 다르다:**

- **정답이 후보 top-N 안에 *있는데* 순위만 밀렸다** → `in_topk=True, rank>1`
  → **reranker 로 회복 가능.** (후보 안에 정답이 있으니, MI 같은 더 똑똑한 채점으로 끌어올리면 됨)
- **정답이 후보 top-N 에 *아예 없다*** → `in_topk=False` (= miss)
  → **reranker 로 회복 불가.** 후보에 정답이 없는데 아무리 재정렬해도 소용없음.
    → **proposer(후보 생성기) 자체를 바꿔야** 한다.

이 갈림길을 모르면 헛수고한다 — 후보에 정답이 없는데(proposer 문제) MI reranker 를 붙이는 건
시간 낭비다. gt-in-topK 가 바로 이 둘을 **숫자로 가르는 계측**이다.

### 용어

- **`in_topk`**: 정답 위치가 chamfer 후보 top-N(기본 N=8) 안에 들어왔나 (True/False).
  "들어왔다"의 기준은 후보 좌표가 정답에서 `GT_TOL_NORM`(template 짧은 변의 0.20) 이내.
- **`topk_rank`**: 정답이 후보 중 **몇 등**으로 들어왔나. 1 이면 chamfer 만으로도 1등(완벽),
  `None`(miss)이면 후보에 없음.
- **`rank1`**: rank==1 인 경우 수. (chamfer 단독으로 이미 맞힌 것)
- **`miss`**: `in_topk=False` 인 경우 수.
- **`in_topk_rate` = in_topk / n = "proposer recall"**: **후보 생성기가 정답을 후보에 담아내는 비율.**
  이게 진짜 천장이다. proposer recall 이 0.70 이면, reranker 를 *아무리 완벽하게* 만들어도
  최대 70% 까지밖에 못 맞힌다(나머지 30% 는 후보에 정답이 없으니까).
- **`rank1_rate`**: chamfer 단독 정확도. reranker 없이 지금 맞히는 비율.
- **`rank_hist`**: 정답이 들어온 순위의 히스토그램. 2·3등에 많이 몰려 있으면 → reranker 여지가 큼
  (후보엔 있는데 chamfer 순위만 밀린 것이니).
- **`median_miss_dist_norm`**: miss 한 경우, 그나마 가장 가까운 후보가 정답에서 얼마나 멀었나.

### 읽는 법 (의사결정 규칙)

```text
in_topk_rate 높음(예 0.9+) + rank1_rate 낮음   →  proposer 는 잘함, chamfer 순위만 나쁨
                                                  ⇒ reranker(MI) 도입이 정답.
in_topk_rate 낮음(예 0.5)                       →  proposer 가 정답을 못 담음
                                                  ⇒ reranker 무의미, proposer(후보 생성) 교체.
```

> 실측 결과가 뼈아프다: **in_topk_rate = 0.459.** 정답이 후보 top-8 에 **절반도 못 든다.**
> 이건 MI reranker 를 *완벽하게* 만들어도 천장이 ~46% 라는 뜻 — reranker 만으로는 부족하다.
> rank_hist 를 보면 든 것 중 60/105 는 이미 rank1(chamfer 단독으로 맞힘)이고 나머지는 2~8 등에
> 얇게 퍼져 있어, reranker 로 끌어올릴 여지(rank 2~8)는 ~45 건뿐. **결론: 후보 생성기(proposer)
> 자체가 약하다 → rcp 재등록·template 개선·proposer 교체의 우선순위가 reranker 보다 높다**
> (`project_matcher_flat_chamfer_distinctiveness`). reranker 는 그 위에 얹는 보조.

---

## 6. 참조 staleness(상대 기준) — rcp 가 낡았나?

또 다른 실패 원인 후보: **rcp(등록된 reference) 자체가 낡아서**(stale) 현재 공정 모습과 다르다.
그러면 matcher 가 약한 게 아니라 *기준이 틀린* 것. 이걸 가리는 게 staleness 진단.

(코드: `_reference_quality` line 554)

### 아이디어: "rcp 가 outlier 인가"를 *상대적으로* 본다

절대 점수("rcp↔msr 유사도가 0.5다")는 함정이다 — matcher 가 약하면 정답에서도 0.5 가 나오기 때문이다.
그래서 **상대 비교**를 한다:

```text
1. 한 recipe 의 S 이미지들로 "현재 align 영역의 대표 모습" = consensus(median 이미지)를 만든다.
2. s_internal = 각 S 가 그 consensus 에 맞는 정도 (S 끼리 얼마나 뭉치나)   ← MI 로 측정
3. rcp_vs     = rcp-center 가 그 consensus 에 맞는 정도
4. relative_ratio = rcp_vs / median(s_internal)
```

같은 metric(MI)으로 재므로 **matcher 약함이 분자/분모에서 상쇄**된다. S 끼리는 잘 뭉치는데
(s_internal 높음) rcp 만 동떨어지면(rcp_vs 낮음) → ratio 가 낮음 → **rcp 가 S cluster 의 outlier =
stale**.

### 용어 / status

- **`relative_ratio`**: 낮을수록 rcp 가 S 들과 동떨어짐(=stale 의심). 1.0 근처면 rcp 도 S 만큼 맞음.
- **`s_internal_cv`** (변동계수, std/mean): S 끼리 얼마나 일관되나. 크면 S 끼리도 안 뭉쳐 consensus 를 못 믿음.
- **status 분류**:
  | status | 뜻 |
  |---|---|
  | `stale_replace` | S 끼리 뭉치는데 rcp 만 동떨어짐 → **rcp 재등록 권장** |
  | `S_inconsistent` | S 끼리도 안 뭉침(CV 큼) → consensus 불신, 판단 불가 |
  | `low_texture_inconclusive` | consensus 자체 정보량 낮음(저텍스처) → 판단 불가 |
  | `insufficient_S` | S 장수 부족(<3) → 판단 불가 |
  | `ok` | rcp 가 S cluster 안에 있음 |

> 이 임계들은 전부 cold-start(미보정)다. golden(항상 성공) 데이터셋으로 실측 calibration 예정
> (`align_success_dataset_plan.md`).

---

## 7. staleness 와 recall 을 잇기 — crosstab, tertile, point-biserial

마지막 콘솔 블록. **"staleness 가 정말 matching 실패의 원인이냐, 아니면 무관한 혼동변수냐"**
를 가린다. 두 진단(§5 gt-in-topK 와 §6 staleness)을 **교차**시킨다.

```text
[INFO] GT-IN-TOPK × 참조 staleness/ratio (원인 분리; row-weighted S+crosshair):
  status=stale_replace   n=...  in_topk_rate=...   median_ratio=...
  status=ok              n=...  in_topk_rate=...
  ratio tertiles:
    low    ratio=[..,..]  in_topk_rate=...   ← stale 후보 구간
    high   ratio=[..,..]  in_topk_rate=...   ← fresh 구간
  point_biserial(in_topk, ratio): n=... r=...   (+면 staleness 가 원인, ~0 이면 무관)
```

> 참고: 이 교차표는 staleness 판정이 된 recipe 가 충분해야 의미가 있다. 실측에서는
> `scored=71 recipes` 중 **stale=17, ok=4, 판단불가=50**(S 부족·불일치·저텍스처)으로,
> "판단 가능" 표본 자체가 적었다 → tertile/biserial 은 **golden 데이터셋 확보 후** 재측정 예정.

(코드: `_gt_topk_reference_crosstab` line 666, `_ratio_tertile` line 607, `_point_biserial...` line 644)

### 7.1 by_reference_status — status 별 recall

각 staleness status 그룹에서 proposer recall(in_topk_rate)을 따로 본다. `stale_replace` 그룹의
recall 이 `ok` 그룹보다 확연히 낮으면 → "stale 하면 매칭도 못 한다"는 연결이 보인다.

### 7.2 ratio tertile — 연속값을 3등분해서 본다

`relative_ratio`(연속값)를 정렬해 **low / mid / high 3등분(tertile)** 하고, 각 구간의 recall 을 본다.
low(가장 stale)에서 recall 이 낮고 high(fresh)에서 높으면 → **단조 관계** = staleness 가 원인이라는 패턴.

> 왜 tertile 로 자르나: status 라벨(이산)은 임계 선택에 민감하다. 연속값을 구간으로 직접 보면
> 임계와 무관하게 "ratio 가 높아질수록 recall 도 높아지나"의 *추세*를 볼 수 있다.

### 7.3 point-biserial correlation — 추세를 숫자 하나로

- **point-biserial r** = "한쪽이 **이진(binary)**, 다른 쪽이 **연속(continuous)**"인 두 변수의 상관계수.
  여기선 `in_topk`(True/False = 1/0) 와 `relative_ratio`(연속) 의 상관.
- 사실 **Pearson 상관계수와 수식이 똑같다** — 한 변수가 0/1 일 때 Pearson r 을 부르는 이름이
  point-biserial 일 뿐이다. 코드도 그냥 Pearson 공식을 쓴다(line 656~663):

  ```text
  r = cov(x, y) / (std(x) * std(y))     # x=ratio(연속), y=in_topk(0/1)
  ```

- **읽는 법** (범위 −1 ~ +1):
  | r | 해석 |
  |---|---|
  | **+ (예 +0.42)** | ratio 높을수록 in_topk 높음 → **staleness 가 recall 저하의 원인**(맞으면 rcp 재등록이 약) |
  | **≈ 0** | 둘이 무관 → staleness 는 혼동변수, recall 저하 원인은 **다른 데**(= matcher 변별력) |
  | **−** | 역방향 (이론상 이상, 데이터 의심) |

> 이게 왜 결정적이냐: §5 에서 recall 이 낮게 나와도, 그게 "rcp 가 낡아서"인지 "matcher 가
> 약해서"인지는 알 수 없다. point-biserial 이 **0 에 가까우면** → staleness 핑계를 배제하고 "원인은
> matcher 자체"라고 못박을 수 있다. 즉 **혼동변수 제거용 한 줄 통계**다.
>
> ⚠️ 상관 ≠ 인과. r 이 커도 "stale → low recall" 인과를 *증명*하진 않는다(둘 다 저텍스처 recipe
> 라서 같이 움직였을 수도). 그래서 tertile·status·biserial 을 **함께** 보고 종합 판단한다.

---

## 8. 한눈에 보는 용어 사전

| 용어 | 한 줄 정의 | 어디서 |
|---|---|---|
| **ground truth (S/E)** | 도구의 성공/실패 라벨 = 평가 기준 | 파일명 S*/E* |
| **at_crosshair / at_center / free_best** | 정답위치 / 중앙 / 전체검색best 에서 잰 유사도 | `_process_msr` |
| **matcher / MI / NCC** | 기하(Chamfer+ORB) / 상호정보 / 픽셀상관 | `_score`,`_mi`,`_ncc` |
| **med_S / med_E** | S/E 점수 중앙값 (떨어져 있어야 좋음) | `_separation` |
| **thr (best_threshold)** | S/E 를 가장 잘 가르는 임계값 | `_separation` |
| **TPR / TNR** | S 맞춘 비율 / E 맞춘 비율 | `_separation` |
| **bACC (balanced accuracy)** | 0.5*(TPR+TNR). 1=완벽, 0.5=동전 | `_separation` |
| **truth-forced** | 정답 위치에서 강제 측정 (free search 반대) | `_truth_forced` |
| **wide / compare chamfer** | 넓은(0.5~1.4) / 좁은(생산) 대역 best | `_truth_forced` |
| **scale_gain** | wide−compare. 크면 scale 대역 놓침(C4) | `_truth_forced` |
| **wrong_local_peak** | 정답 위치에서도 best 가 정답을 벗어남 | `_diagnose_truth` |
| **proposer / reranker** | 후보 생성기 / 후보 재정렬기 | (개념) |
| **in_topk** | 정답이 후보 top-N 에 들어왔나 | `_gt_in_topk` |
| **topk_rank / rank1 / miss** | 정답 순위 / 1등 / 후보에 없음 | `_gt_in_topk` |
| **in_topk_rate = proposer recall** | 후보 생성기가 정답을 담는 비율 (= 천장) | `_gt_in_topk` |
| **staleness / relative_ratio** | rcp 가 S cluster 의 outlier 인 정도(낮을수록 stale) | `_reference_quality` |
| **consensus** | S 이미지들의 median = 현재 대표 모습 | `_consensus` |
| **CV (변동계수)** | std/mean. S 끼리 일관성 | `_reference_quality` |
| **tertile** | 연속값 3등분 구간 | `_ratio_tertile` |
| **point-biserial r** | 이진×연속 상관 (=Pearson). 혼동변수 제거용 | `_point_biserial...` |

---

## 9. 전체를 하나의 의사결정 흐름으로

```text
[S/E 분리도 bACC]
   모든 metric bACC ≈ 0.5?  →  지표 자체가 S/E 못 가름 (matcher 약함 확정)
        │ MI 가 그나마 높음  →  MI 를 reranker 후보로
        ▼
[truth-forced diagnosis_counts]   (정답을 알려주고도 약한가? 어디가 병목?)
   wrong_local_peak 多   →  변별력 문제 (top-N + reranker 필요)
   scale_band_problem 多 →  scale 대역 확장(C4)
   template_weak 多      →  rcp 재등록
        ▼
[gt-in-topK in_topk_rate]         (정답이 후보에 드나? = 천장)
   높음 + rank1 낮음  →  reranker(MI) 로 회복 가능 ⇒ Phase 3 진행
   낮음              →  proposer 교체 (reranker 무의미)
        ▼
[staleness × recall: point-biserial r]   (rcp 가 낡아서? 아니면 matcher 탓?)
   r 큼(+)  →  staleness 가 원인 ⇒ rcp 재등록 우선
   r ≈ 0    →  staleness 무관 ⇒ matcher 보강에 집중
```

이 흐름의 실측 결론(432장): free_best bACC 0.549(역전) + wrong_local_peak 67% +
scale_gain≈0(C4 아님) + **proposer recall 0.459(낮음)** + MI 만 약하게 변별(0.627) →
**matcher 변별력이 진짜 병목이고, 후보 생성(proposer) 단계부터 정답을 절반밖에 못 담는다.**
그래서 우선순위는 ① rcp key 품질/재등록 + proposer 개선(top-N 후보화·distinctiveness gate),
② 그 위에 MI reranker 를 얹기 — 순이다 (recovery plan §5, Phase 2~3).

### 참고: align_point_correction 의 status 분포 (실측 432장 / 121 recipes)

위 진단의 귀결이 보정 batch 의 status 에도 그대로 나타난다:

```text
suspect_success: 208   not_distinctive: 142   low_match_both: 32
no_crosshair_draw: 244   ambiguous_modality: 4   already_aligned: 3   ok: 7
```

`ok` 가 7 건뿐 — 대부분 `not_distinctive`(변별 불가) / `suspect_success`(S 라벨인데 보정거리 큼)
로 분류돼, "모르면 보류"(reject-state 우선) 원칙대로 동작 중. 즉 지표 진단과 보정 결과가 일치한다.

---

## 부록 — 직접 돌려보기

```bash
# 실데이터 있으면 분석, 없으면 합성 self-test (Mac 에서 동작)
uv run python poc/workflow_2/align_similarity.py
```

출력: stdout 요약(위 §3~§7 표들) + `DEBUG_IMAGE_DIR/align_similarity/<ts>/{rows.jsonl, summary.json}`.
`rows.jsonl` 에 이미지 한 장당 모든 raw 값이 들어 있어, 콘솔 요약이 어떻게 집계됐는지 역추적 가능.
