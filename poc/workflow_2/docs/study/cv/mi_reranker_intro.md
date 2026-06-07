# MI reranker — "후보 재정렬"이 실제로 뭘 하나 (학습용)

> 대상 코드: `poc/workflow_2/align_similarity.py` 의 `_gt_in_topk`(MI rerank 측정) + `_consensus_template_ab`(A/B)
> 목적: green mark 정밀도를 올리려고 도입하는 **MI reranker** 가 *구체적으로 무엇을 어떻게* 하는지,
> 콘솔의 `rerank_rank1_lift` / `topk_rank_reranked` 를 어떻게 읽는지 한국어로 정리.
> 선행 문서: [평가 지표·용어 정리](align_evaluation_metrics_intro.md) (proposer/reranker, in_topk, rank1, MI 의 개념)
> 관련: [match 알고리즘 intro](../algorithms/match_algorithms_intro.md), [consensus A/B 저널](../../journals/260529/260529_152818_consensus-ab-verdict-and-next-steps.md)

---

## 0. 한 줄 요약

> **proposer(chamfer)** 가 "여기 같은데?" 후보 8개를 뽑아 준다. 그중 1등이 자주 틀린다.
> **reranker(MI)** 는 그 8개를 *다시 채점해 순서만 바꾼다.* 후보를 새로 만들지도, 버리지도 않는다.
> 정답이 후보 안에 있는데 순위만 밀린 경우를, 더 똑똑한 채점으로 1등으로 끌어올리는 게 전부다.

---

## 1. 왜 reranker 가 필요했나 (복습)

[지표 문서 §5](align_evaluation_metrics_intro.md) 에서 두 숫자를 분리했다:

- **in_topk_rate (= proposer recall):** 정답이 chamfer 후보 top-8 *안에 들어오는가*. 이게 **천장**이다.
- **rank1_rate (= 정밀도):** 그 후보 중 **1등이 정답인가**. chamfer 단독으로 지금 맞히는 비율.

오피스 A/B(test4) 에서 rcp 를 S-consensus 로 재등록하니 천장이 크게 올랐다:

| | recall(in_topk) | precision(rank1) |
|---|---|---|
| rcp | 0.436 | 0.269 |
| consensus | **0.718** | 0.538 |

그런데 **recall 0.718 vs rank1 0.538** 사이에 **0.18 의 틈**이 남는다(`topk_not_rank1`).
= "정답이 후보엔 있는데(top-8 안), chamfer 점수로는 1등이 되지 못한" 경우가 18% 다.
이 18% 가 바로 **reranker 가 노리는 회복 구간**이다 — 후보를 다시 채점해 1등으로 올리면 된다.

> 핵심 직관: chamfer 는 **후보를 만드는 데는 충분**(recall 0.718)하지만 **순서를 매기는 데는
> 약하다**(점수면이 평평 → 정답과 잡음 peak 의 점수 차가 작음). 그래서 "멤버십은 chamfer 에
> 맡기고, 순서는 다른 지표(MI)에 맡긴다"는 역할 분담이 나온다.

---

## 2. "재정렬(rerank)"의 정확한 의미 — 멤버십은 그대로, 순서만 바꾼다

이게 이 문서에서 제일 중요한 한 가지다.

```text
chamfer 후보 (점수 내림차순):     [A, B, C(정답), D, E, F, G, H]   ← C 가 3등
                                   │
                       MI 로 각 후보 다시 채점 → 내림차순 재정렬
                                   ▼
MI 재정렬 후:                     [C(정답), A, D, B, ...]            ← C 가 1등
```

- **후보 *집합*은 안 바뀐다.** A~H 그대로. 누구를 빼거나 새로 넣지 않는다.
- **순서만 바뀐다.** MI 점수로 다시 줄 세운다.

여기서 따라오는 *결정적 성질*:

> **rerank 는 `in_topk`(정답이 후보에 있나)를 절대 못 바꾼다.** 정답이 집합에 있으면 재정렬 후에도
> 있고, 없으면 없다. **바뀌는 건 `rank`(정답의 순위)뿐이다.**

그래서 코드의 self-test 가 이렇게 단언한다(`_self_test`):

```python
# 멤버십(in_topk)은 rerank 로 안 바뀐다 — rank 만 움직인다.
assert (gt["topk_rank_reranked"] is None) == (gt["topk_rank"] is None)
```

(`topk_rank == None` 이 곧 "후보에 정답 없음 = miss". 두 rank 의 None 여부가 항상 같아야 한다.)

이 불변성 덕분에 **한 숫자로 reranker 효과를 깨끗하게 잴 수 있다**:

```text
rerank_rank1_lift = (MI 재정렬 후 rank1 비율) − (chamfer rank1 비율)
```

recall(천장)은 reranker 와 무관하게 고정이므로, 이 lift 는 순수하게 "MI 가 순위 오류를 얼마나
고쳤나"만 잰다. 즉 confounder 가 없다.

---

## 3. 코드가 하는 일 (`_gt_in_topk` 안)

후보 한 묶음(`cands`)에 대해:

```python
# 1) chamfer 후보 생성 (proposer) — 집합 확정.
cands = compute_chamfer_candidates(tpl, frame_dt, scales=scales, top_n=topk)

# 2) 각 후보의 msr crop 을 떼어 template 과 MI 채점 (reranker 점수).
for c in cands:
    crop = _matched_crop(gray, c.xy, tw2, th2, c.scale)   # 후보 위치/배율에서 crop → template 크기로 resize
    mi_scores.append(_mi(tpl.raw_image, crop))

# 3) MI 내림차순으로 순서만 재배열.
order = sorted(range(len(cands)), key=lambda i: mi_scores[i], reverse=True)

# 4) 재정렬된 순서에서 정답(crosshair)의 새 순위를 다시 계산.
rr_dists = [dists[i] for i in order]
rr_rank  = next((i for i, d in enumerate(rr_dists, 1) if d <= GT_TOL_NORM), None)
```

- `_matched_crop(frame, xy, tw, th, scale)`: 후보 중심 `xy` 에서 `tw·scale × th·scale` 크기로 잘라
  template 크기 `(tw, th)` 로 리사이즈한다. MI 는 **같은 크기 두 이미지**의 밝기 분포 상호정보량이라
  크기를 맞춰야 한다.
- `_mi`: [지표 문서 §2](align_evaluation_metrics_intro.md) 의 mutual information. 밝기/대비가 통째로
  달라져도 *구조적 대응*이 남으면 높게 나온다 → SEM align 처럼 픽셀 동일성을 못 믿는 도메인에 적합.
- `GT_TOL_NORM`(=0.20): 후보가 정답으로 인정되는 거리(template 짧은 변 대비). chamfer rank 때와
  **같은 기준**을 써야 두 rank 가 apples-to-apples.

### 왜 "hard reorder"(MI 점수로 그냥 다시 정렬)인가
1차 버전은 chamfer 점수와 MI 를 섞지(blend) 않고 **MI 만으로 순서를 정한다**. 이유:
- chamfer 가 이미 멤버십 게이트 역할을 하므로, 순서 결정에서까지 chamfer 를 또 섞으면
  "MI 가 단독으로 순위 오류를 고치는가"를 *측정*할 수 없다(효과가 가려짐).
- 검증(rerank_lift) 후 필요하면 그때 `alpha·chamfer + (1−alpha)·MI` 블렌딩을 후순위로 고려한다.

---

## 4. scale 처리 — 후보의 discrete scale 을 그대로 쓴다 (지금은)

각 chamfer 후보는 자기 `scale`(0.6/0.75/0.85/1.0 중 best 하나)을 갖는다. MI crop 을 뜰 때 이 discrete
scale 을 **그대로** 쓴다. 후보별로 ±1 band 미세조정(local refine)을 *하지 않는다.* 이유:

- **순서만 정하면 되는데** 모든 후보가 같은 scale grid 를 공유하므로, discrete scale 로도 *상대 비교*엔
  충분하다(정밀 정합이 목적이 아님).
- scale 이 진짜 병목이면 그건 reranker 가 아니라 **proposer(scale 대역)를 고칠 문제**다 —
  truth-forced sweep 의 `scale_gain` 이 이미 그걸 따로 진단한다([지표 문서 §4](align_evaluation_metrics_intro.md)).
- per-candidate refine 은 MI 비용을 `top_n × scale_band` 로 키운다(프레임당, 후보당). 비싸다.

> 후속 신호: 만약 MI rank 오류가 특정 `cand.scale` 에 몰리면(예: band 경계에서만 틀림) 그때
> 국소 refine 을 추가한다. 지금은 측정부터.

---

## 5. 순환성(circularity) 체크 — "같은 consensus 로 만들고 같은 consensus 로 채점"이 반칙 아닌가?

A/B(`_consensus_template_ab`)는 S-consensus 템플릿으로 **후보를 만들고**, *같은* consensus 템플릿으로
**MI 재채점**한다. 자기가 만든 답을 자기가 채점하는 것처럼 보여 의심스러울 수 있다. 결론: **건전하다.**

- **leave-one-out(LOO):** held-out 프레임 `i` 는 consensus 만들 때 *제외*된다(`others = i 빼고`).
  그래서 "어떤 crop 을 그 crop 자신으로 채점"하는 진짜 순환은 없다.
- **MI 가 비교하는 대상:** held-out msr 의 *후보 위치 픽셀* ↔ *템플릿*. 위치/외형 비교지
  "consensus vs consensus" 가 아니다.
- **남는 편향 한 가지:** MI 가 "consensus 처럼 생긴" crop 을 전반적으로 선호할 수 있다. 그런데
  그게 바로 *검증하려는 production 동작*이다(consensus 가 새 등록 템플릿이 되기 때문). 그래서 진단용으로
  **rcp 기준 reranked 컬럼도 병렬로** 함께 찍는다(`overall_rcp_rank1_reranked_rate`).

> ⚠️ 절대 금지: held-out 의 `xhair_crop`(정답 위치 crop) *자체*를 MI reference 로 쓰는 것 —
> 그건 진짜 순환이다. 반드시 held-out 을 뺀 consensus(또는 rcp)를 reference 로 쓴다.

---

## 6. 콘솔 읽는 법

### GT-IN-TOPK 블록 (전체 S+crosshair)
```text
[INFO] GT-IN-TOPK (S+crosshair, top-8 chamfer 후보의 proposer recall):
  in_topk=.../...  rank1=... (0.262)  miss=...
  MI rerank rank1=... (0.41?)   (chamfer rank1 0.262 → MI 0.41?; in_topk 0.459 가 천장)
```
- `chamfer rank1 → MI rank1` 이 오르고 **`in_topk`(천장)에 가까울수록** reranker 가 잘 회복한 것.
- MI rank1 이 chamfer 와 거의 같으면 → MI 가 chamfer 순위를 못 고침(이 도메인에선 약함).

### CONSENSUS A/B 블록
```text
  recall  (in_topk):  rcp=0.436  consensus=0.718  lift +0.282
  precision(rank1) :  rcp=0.269  consensus=0.538  rank1_lift +0.269  topk_not_rank1=0.179
  +MI rerank rank1 :  rcp=...    consensus=...     rerank_lift +0.???  (목표: cons rank1 0.538→in_topk 0.718)
```
- **`rerank_lift`** 가 핵심. `topk_not_rank1`(0.179)을 MI 가 얼마나 메웠나.
- **판정:** `rerank_lift ≥ +0.10` 이면 MI reranker production 승격 정당화. `≈0` 이면 MI 로는
  부족 → contour 등 다른 reranker 검토.

| 결과 패턴 | 해석 | 다음 액션 |
|---|---|---|
| rerank_lift 큼(+), cons rank1 → in_topk 근접 | MI 가 순위 오류를 거의 다 회복 | **production 승격** (`compute_align_key_score` 내부에 MI reorder) |
| rerank_lift 작음(≈0) | MI 가 chamfer 순위를 못 고침 | contour/다른 reranker, 또는 proposer 강화로 |
| in_topk 자체가 낮음 | 천장이 낮음(정답이 후보에 없음) | reranker 무의미 → **proposer 교체** 우선 |

---

## 7. 다음 단계 (production 승격 시 주의)

검증되면 진단 하네스에서 production(`align_point_correction.py` 경로)으로 옮긴다. Codex 설계 verdict:

- **위치:** `compute_align_key_score` *내부*에 opt-in 으로 MI reorder. 호출부에서 나중에 재정렬하면
  `best_xy` 와 distinctiveness 필드(`second_score`/`score_gap`/`reject_reason`, 모두 `candidates[0/1]`
  파생)가 어긋나 coherence 버그가 난다.
- **DEFAULT_POLICY 불변:** 합성 smoke test 보장을 위해 기본 경로는 chamfer 순서 유지. MI reorder 는
  `STRUCTURE_POLICY`/명시 플래그에서만 활성.
- **재정렬 후 재계산:** `second_score`/`score_gap` 은 현재 chamfer #1/#2 를 가리키므로 MI reorder 후
  의미를 다시 정의. 원래 chamfer rank 는 audit/rollback 용으로 보존. **ORB 확인도 MI-best 후보에서**
  다시 돌려야 엉뚱한 후보를 재지 않는다.

---

## 부록 — 직접 돌려보기

```bash
# 실데이터 있으면 A/B 에 rerank_rank1_lift 가 찍히고, 없으면 합성 self-test (경로만 확인).
uv run python poc/workflow_2/align_similarity.py
```

self-test 출력의 `gt_topk: in_topk=True rank=1 rerank=1` 이 rerank 경로가 무오류로 돌고 멤버십을
안 바꿨다는 확인이다. 실데이터 수치(`rerank_rank1_lift`)는 오피스 재실행에서 회신받는다.
