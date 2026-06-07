# Reranker A/B 실패 분석 리포트 — MI · contour 모두 폐기

> 작성: 2026-06-02 · 대상: `align_similarity.py` rerank A/B (test1·test2), `align_key_matcher.py` mi_rerank
> 출처 결과지: `docs/console_results/260602_test1.txt`(MI), `260602_test2.txt`(contour 추가)
> 관련: [[project_matcher_flat_chamfer_distinctiveness]], `docs/align_success_dataset_plan.md`,
>        저널 `journals/260602/260602_075313_*`

---

## 0. 요약 (TL;DR)

- **무엇을 했나:** chamfer 매칭이 정답 위치를 top-K 후보엔 넣지만 1등으로 못 뽑는 갭
  (recall 0.72 vs precision 0.54)을, 후보 *순서만* 바꾸는 **reranker** 로 메우려 했다.
  두 가지를 같은 A/B 하네스로 검증: **MI**(intensity) → **contour**(형상 Hu).
- **결과:** 둘 다 실패. MI `rerank_lift = −0.013`, contour `−0.167`(baseline 아래로 떨어뜨림).
  승격 게이트(≥+0.10) 크게 미달.
- **근본 원인:** chamfer 가 surface 하는 후보들은 **패치 단위로 본질적으로 모호**하다. 같은
  프레임의 local peak 들이라 intensity 분포를 공유(→MI 무력)하고, contrast drift + 전역·불변
  형상기술자(Hu)는 미세 공간정합 정보를 버린다(→contour 가 오히려 역효과).
- **결론:** **reranker 레버 사망.** 정밀도 갭은 *순위 재정렬*이 아니라 **검색공간 축소
  (VLM-region + CV)** 또는 **proposer 교체**로만 회복 가능. 관련 production 코드(mi_rerank)도 제거.

---

## 1. 배경 — 왜 reranker 를 시도했나

align-fail 시 msr 에서 등록 align key 위치를 찾는 것이 workflow_2 의 핵심이다. 매칭 엔진은
Chamfer(+ORB) 로 top-K 후보를 생성하고 best 를 좌표로 쓴다. 그런데 실데이터(432장: S=234, E=198)
진단에서 두 가지가 드러났다.

1. **chamfer 점수면이 평평**(non-distinctive). `free_best` med_S 0.66 ≈ med_E 0.70, bACC 0.58.
   정답에서도 best peak 이 67% 빗나감(truth-forced `wrong_local_peak=153/229`).
2. **재등록(S-consensus)으로 recall 은 크게 오르나 정밀도는 갭이 남음.** consensus A/B(LOO):
   - recall(in_topk) `0.436 → 0.718` (+0.282)
   - precision(rank1) `0.269 → 0.538` (+0.269), **`topk_not_rank1 = 0.179`**
   - gt-in-topk(전역): 정답이 후보에 든 136/229 중 **rank1 은 78뿐**(58개가 rank 2~8).

→ "정답은 후보 *집합*엔 있는데 chamfer 가 1등으로 못 뽑는다." 이 갭을 메우는 자연스러운 도구가
**reranker**다: chamfer 후보 *집합*(=멤버십)은 그대로 두고, 다른 신호로 *순서만* 바꿔 정답을
1등으로 끌어올리는 것이다. 멤버십이 불변이므로 recall(in_topk)은 그대로 두면서 rank1(정밀도)만
끌어올리는 것이 기대 효과였다.

---

## 2. 가설과 근거

| reranker | 가설 | 근거 |
|---|---|---|
| **MI** (mutual information) | 후보 crop 과 template 의 MI 가 true vs decoy 를 가른다 | intensity drift 에 강건, S/E 분리도에서 **bACC 0.627 로 전 지표 1위**(약하지만 유일하게 방향 정상) |
| **contour** (Otsu+Hu) | key 의 *기하/형상* 위상이 위치를 가른다 | MI 실패 후 대안. chamfer(매끄러운 거리장)·MI(전역 intensity)가 못 보는 *공간 형상* 을 잡을 것. 도메인 원칙([[project_align_key_matching_constraint]]: 픽셀 동일성 금지, 기하/구조 매칭)과 정합 |

핵심 주의: **S/E 분리(present/absent)와 위치 reranking(true vs decoy)은 서로 다른 작업**이다. MI 가
전자에서 1위였다고 해서 후자에서도 통한다는 보장은 없다 — 이 구분이 결국 실패의 핵심이 된다(§5).

---

## 3. 실험 설계 (A/B 하네스)

`align_similarity.py` 에 reranker 검증 컬럼을 넣어 **apples-to-apples** 로 측정:

- **멤버십 게이트 = chamfer**: 후보 집합은 chamfer top-K 고정. reranker 는 *순서만* 바꾼다
  (`topk_rank_reranked`). 멤버십 불변 → in_topk 동일, rank 만 이동.
- **지표**: `rerank_rank1_lift = (rerank 후 rank1) − (chamfer rank1)`.
  consensus A/B(LOO, 재등록 검증)와 gt-in-topk(전역) 양쪽에서 측정.
- **판정 게이트**: `rerank_rank1_lift ≥ +0.10` 이면 production 승격(매처에 `mi_rerank` opt-in
  플래그를 미리 넣어두고 검증 후 활성화 예정이었음). `≈0/음수` 면 폐기.

---

## 4. 결과

| reranker | gt-topk rank1 | consensus rank1 | consensus rerank_lift | 판정 |
|---|---|---|---|---|
| chamfer (baseline) | 0.341 | 0.538 | — | — |
| **MI** (test1) | 0.336 | 0.526 | **−0.013** | ❌ 폐기 |
| **contour** (test2) | **0.214** | 0.372 | **−0.167** | ❌❌ 폐기 (더 나쁨) |

- MI: 사실상 무변(−0.013) — chamfer 순위를 거의 못 고침.
- contour: **baseline(0.341) 아래로(0.214) 떨어뜨림** = 점수가 정답과 **역상관**. decoy 가 정답보다
  형상 유사도를 더 높게 받았다는 뜻. MI 보다 훨씬 해로움.

(부수 확인 — 재등록은 reranker 와 무관하게 유효: recall +0.282, rank1 +0.269. 선명도 비율 vs S개별
edge 0.979/lap 0.959 → consensus blur 아님 = co-registration 불필요.)

---

## 5. 왜 실패했나 (근본 원인)

### 5-1. MI — "맞는 축"이 다르다
MI 는 **프레임 단위 분리**(이 프레임에 key 가 있나/없나)엔 힘이 있지만, **한 프레임 내 위치
변별**(true peak vs decoy peak)엔 무력하다.
- top-K 후보는 *같은 프레임*의 local peak 들 → intensity/texture 분포를 공유.
- 따라서 template 과의 MI 값이 후보끼리 거의 같다: **후보 간 MI 분산 ≪ 프레임 간 MI 분산.**
- MI 는 공간 배치에 둔감한 전역 통계라, "어느 위치가 정답이냐"를 가를 신호가 애초에 거의 0.

### 5-2. contour — 더 나쁜 두 가지 이유
1. **Otsu 이진화가 contrast 에 불안정.** 이 도메인의 전제 자체가 "rcp ↔ live/msr 은 밝기·대비가
   다르다"는 것이다(그래서 STRUCTURE_POLICY 가 픽셀 아닌 엣지를 쓴다). crop 마다 Otsu 임계가
   달라지면 정답 crop 의 binary mask 가 template 과 어긋나고, 엉뚱한 decoy 가 우연히 비슷하게
   이진화된다.
2. **Hu moment 가 너무 전역적 + 불변.** crop 을 7개 숫자(gross 형상)로 뭉개고 평행이동·스케일·회전
   *불변* → 정답을 가르는 바로 그 *미세 공간 정합* 정보를 버린다. MI(전체 joint 히스토그램)보다도
   위치 변별이 약하다. 결과적으로 noise 를 주입한 셈이 되어 rank1 이 baseline 아래로 떨어졌다.

### 5-3. 통합 결론 — 후보가 본질적으로 모호
intensity(MI)로도 형상(Hu)으로도 정답을 가르지 못한다는 것은, **chamfer 가 surface 하는 후보 집합이
패치 단위에서 본질적으로 모호**하다는 뜻이다. 이는 데이터 성질(공정 drift + 반복적 SEM 구조)의
문제이지, reranker 지표 선택의 문제가 아니다. → **다른 reranker 를 더 찾는 것은 무의미하다.**

---

## 6. 결론 & 함의

1. **reranker 레버 사망.** MI·contour 두 계열 독립 실패(게이트 ≥+0.10, 실측 −0.013/−0.167).
   더 짜낼 reranker 후보 없음.
2. **재등록은 별개의 큰 레버(유효).** rank1 0.27→0.54 는 *재등록* 효과. golden 데이터로
   production 화(별도 워크스트림, `align_success_dataset_plan.md`).
3. **남은 정밀도 갭의 escalation = 검색공간 축소(순위 아님):**
   - **(채택) VLM-region + CV** — VLM 이 msr 에서 align key 후보 영역(coarse box)을 좁히고 그 안에서
     CV(chamfer)가 최종 좌표. 모호 후보를 *순위*가 아니라 *공간 축소*로 가른다. 도메인 경계 유지
     (VLM=영역, CV=좌표).
   - (대안, 후순위) proposer 교체 — chamfer 후보 생성기 자체를 바꿈(contour-기반/AKAZE/multi-scale).
4. **코드 정리:** 폐기된 reranker instrumentation 제거 — `align_key_matcher.py` 의 `mi_rerank`
   플래그·`rerank_candidates_by_mi`·관련 필드, `align_similarity.py` 의 MI/contour rerank 컬럼·
   `_contour_sim`/`_hu_log`, `test_align_key_mi_rerank.py`. 음성 결과는 본 리포트로 보존.

---

## 7. 부록 — 재현 / 출처

- 결과지: `docs/console_results/260602_test1.txt`(MI), `260602_test2.txt`(contour).
- 측정 코드(제거 전 기준): `align_similarity.py` `_gt_in_topk`(`topk_rank_reranked*`),
  `_consensus_template_ab`(`rerank_rank1_lift*`).
- 데이터: fail 트리 432장(S=234, E=198), consensus A/B 는 S≥4 인 15 recipe LOO.
- 판정 게이트: `rerank_rank1_lift ≥ +0.10` → 승격. 실측 모두 음수 → 폐기.
