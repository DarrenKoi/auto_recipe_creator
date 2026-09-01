[DIGEST] 현행 production은 최근 S consensus 또는 RCP box template을 고정된 5-scale Canny/Scharr/orientation Chamfer로 각각 제안하고 RRF로 합친 뒤, 상위 8개만 NCC와 OM-MIND/SEM-ECC로 재순위한다. 67 recipe/334점의 최종 verifier 벤치에서 남은 실패 중 `pm=39`, 약 87%가 top-K 밖 recall miss였으므로 7월의 pair-ranker 우선순위는 이제 틀렸고, cond.txt 좌표가 붙은 수백만 장을 이용해 proposer 자체를 학습하는 것이 맞다. 가장 안전한 교체점은 `AlignKeyTemplate + frame -> candidates[(xy, scale, score)]` 경계이지만, 현 Canny rescore와 top-8 cap이 learned 후보를 다시 버릴 수 있으므로 learned proposal score를 보존하는 scorer/threshold 재보정까지 한 묶음으로 A/B해야 한다.

# 현행 align-key 위치 재탐색 알고리즘 감사

## 감사 범위와 7월 문서 대비 델타

이 문서는 코드와 저장소의 오피스 벤치 기록을 감사한다. Mac에는 오피스 이미지가 없어 수치를 재실행하지 않았고 아래 성능 수치는 저장된 오피스 결과다.

7월 선행 연구는 이미 top-K verifier로 ECC, SIFT/AKAZE, phase, MIND, pair-ranker, DINOv2, LoFTR/RoMa, VLM abstention을 제안했다. 당시에도 proposer miss와 rank error를 구분했지만 실행 우선순위는 ECC와 작은 pair-ranker처럼 후보 후단에 있었다 (`poc/workflow_2/docs/study/cv/align_fail_cv_methods_research_ko.md:11-17`, `poc/workflow_2/docs/study/cv/align_fail_vlm_deep_learning_addendum_ko.md:23-30`, `poc/workflow_2/docs/study/cv/align_fail_vlm_deep_learning_addendum_ko.md:126-150`).

그 뒤 판단을 바꾸는 사실은 두 가지다.

1. 최종 modality-aware verifier 벤치에서 `route_sw=0.826`, OM은 MIND 결합, SEM은 ECC 단독 전환이 채택되었고 남은 실패 중 proposer miss가 `pm=39`, 87%였다. 정답 좌표가 현 top-K에 없으므로 pair-ranker, NCC, MIND, ECC는 원리상 복구할 수 없다 (`docs/journals/260721/260721_082113_sem-ecc-라우팅-포팅-및-재등록-worklist.md:55-72`). 7월 P0-E pair-ranker를 1순위로 둔 판단은 철회한다. proposer 학습이 1순위다.
2. 7월 문서는 golden의 독립 target label이 적다는 전제로 self-supervised pretraining을 P3에 뒀다 (`poc/workflow_2/docs/study/cv/align_fail_vlm_deep_learning_addendum_ko.md:192-196`). 지금은 S 이미지마다 `cond.txt` crosshair 좌표가 붙은 수백만 장을 쓸 수 있다. supervised dense localization을 바로 학습하고 self-supervised pretraining은 라벨 부족 대책이 아니라 domain representation 초기화로 병행하면 된다. 단, recipe/equipment/time holdout은 그대로 필요하다 (`poc/workflow_2/docs/study/cv/align_fail_vlm_deep_learning_addendum_ko.md:130-136`, `poc/workflow_2/docs/study/cv/align_fail_vlm_deep_learning_addendum_ko.md:213-221`).

## 1. 현행 파이프라인 도식

```text
live paused capture
  |  production: raw frame, crosshair 제거 없음
  |  golden S eval: cond crosshair GT 추출 후 clean_image로 제거
  v
template routing
  |-- consensus 가능: 최근 S -> crosshair 중심 crop -> clean -> co-register -> median
  |                    -> blur gate -> AlignKeyTemplate
  `-- consensus 불가: RCP cond box stroke inpaint -> box inner crop
                       -> align_offset_xy 보존; cond box 불가 시 center-area crop
  v
scale band: 0.70, 0.85, 1.00, 1.20, 1.40
  v
proposer C1/C2/C3, 각 channel solo top-24
  |-- C1: gray -> CLAHE(2.0, 8x8) -> Gaussian sigma 1.0
  |         -> Canny(60,160) edge -> frame distance transform -> Chamfer
  |-- C2: raw gray Scharr magnitude -> Canny density에 맞춘 3-15% binary edge -> Chamfer
  `-- C3: raw gray Scharr edge를 8 orientation bin -> same-bin directional Chamfer
  v
RRF(k0=10), channel 간 5% short-side 또는 8 px radius cluster -> shadow top-24
  v
production cap: fused 앞 8개만 Canny Chamfer로 rescore
  v
NCC rerank: sel = 0.5 * chamfer + 0.5 * max(0, NCC)
  v
modality branch
  |-- OM: sel order + MIND order를 RRF(k=8)
  `-- SEM: ECC correlation order 단독, 전 후보 거부 시 sel order fallback
  v
선택 후보의 sel score를 공통 threshold에 적용
  |-- >= 0.6053: match
  |-- >= 0.4727: adjust
  `-- 그 미만: low
  v
best_xy = template center in live frame
  + align_offset_xy * best_scale
  -> clamp -> reposition 또는 fallback/engineer review
```

### 1.1 입력과 template 생성

- Production primary 입력은 `controller.capture()`의 paused live frame이며 현재 mode로 OM/SEM template을 고른다. 이 frame에는 별도 `clean_image` 또는 crosshair inpaint가 적용되지 않는다 (`poc/workflow_3/align/correction.py:255-264`). 반면 golden localization은 `cond.crosshair_xy`를 GT로 읽고 S frame을 `clean_image`로 정제한다 (`poc/workflow_2/golden_localization_eval_cond.py:397-420`, `poc/workflow_2/golden_localization_eval_cond.py:430-439`). consensus LOO도 crosshair-to-crosshair 가짜 lock을 막으려고 clean frame을 기본값으로 쓴다 (`poc/workflow_2/golden_consensus_eval_cond.py:101-108`).
- RCP 경로는 기본 `cond_box_crop=True`다. cond box가 유효하면 box stroke만 inpaint하고 2 px 대칭 inset한 내부를 template로 쓰며 `image center - box center`를 `align_offset_xy`로 분리 보존한다. box가 없거나 너무 작거나 멀면 면적 15% center crop으로 폴백한다 (`poc/workflow_3/align/cond_template.py:22-30`, `poc/workflow_3/align/cond_template.py:47-88`, `poc/workflow_3/align/cond_template.py:91-114`, `poc/workflow_3/align/templates.py:17-49`).
- Consensus 경로는 S 이미지의 cond crosshair를 align point로 삼아 frame을 정제한 뒤 center template 크기로 crop하고 modality별 co-registration한다 (`poc/workflow_3/align/consensus_crops.py:45-71`, `poc/workflow_3/align/consensus_crops.py:105-161`). crop median이 최소 3장이고 edge-density 비율 0.70, Laplacian-variance 비율 0.50을 모두 넘으면 consensus를 쓰고 아니면 RCP로 폴백한다 (`poc/workflow_3/align/consensus_template.py:24-28`, `poc/workflow_3/align/consensus_template.py:55-76`, `poc/workflow_3/align/consensus_resolve.py:65-79`). Consensus는 center crop이므로 offset은 0이다.
- Production은 consensus를 `<class>/<recipe>` key로 장비 구분 없이 pooling한다 (`poc/workflow_3/align/consensus_resolve.py:46-69`, `poc/workflow_3/align/consensus_crops.py:105-120`). 현재 의도된 정책이지만 tool-to-tool appearance 차이는 모델링하지 않는다.

### 1.2 공통 전처리와 scale

- `build_template`과 live frame 모두 grayscale uint8로 바꾼 뒤 CLAHE, Gaussian blur, Canny를 적용하고 frame edge의 distance transform을 만든다 (`poc/workflow_3/align/matching/engine.py:181-238`). 고정값은 CLAHE clip 2.0, tile 8x8, sigma 1.0, Canny 60/160이다 (`poc/workflow_3/align/matching/engine.py:36-47`).
- 물리 pixel scale metadata가 양쪽에 있으면 단일 비율을 쓰지만 primary correction은 명시적으로 `PAUSED_SCALES=DEFAULT_SCALES`를 넘겨 `(0.7, 0.85, 1.0, 1.2, 1.4)`만 탐색한다 (`poc/workflow_3/align/matching/engine.py:680-706`, `poc/workflow_3/align/correction.py:58-64`, `poc/workflow_3/align/correction.py:260-275`).
- 절대배율 grid fallback은 계산한 단일 scale로 같은 ensemble matcher를 호출한다 (`poc/workflow_3/align/grid_search.py:281-300`, `poc/workflow_3/align/grid_search.py:344-344`). legacy live fallback은 예외적으로 C1 `compute_align_key_score`와 broad-to-native `WIDE_SCALES`를 쓰므로 primary와 동일한 C1/C2/C3 pipeline이 아니다 (`poc/workflow_3/align/live_search.py:246-265`).

### 1.3 C1, C2, C3 proposer

| 채널 | 원리 | 현 파라미터 | 코드 근거 |
|---|---|---|---|
| C1 Canny Chamfer | template edge pixel마다 가장 가까운 frame edge까지 거리를 재 평균을 낸다. `exp(-mean_dt/10 px)`를 씌워 큰 값이 좋은 score가 되게 한다 | Canny 60/160, `DT_TAU_PX=10`, 5 scale, per-scale 및 global NMS 반경 `0.5 * template short side` | `poc/workflow_3/align/matching/engine.py:254-287`, `poc/workflow_3/align/matching/engine.py:337-379`, `poc/workflow_3/align/matching/ensemble.py:145-162` |
| C2 Scharr Chamfer | Scharr gradient magnitude 상위 비율을 binary edge로 만든 뒤 C1과 같은 Chamfer 본체 사용 | frame Canny density를 3-15%로 clamp해 template과 frame threshold를 정함 | `poc/workflow_3/align/matching/ensemble.py:16-42`, `poc/workflow_3/align/matching/ensemble.py:150-162` |
| C3 directional Chamfer | Scharr edge를 unsigned gradient 0-180도 8 bin으로 나누고 같은 방향 bin의 distance만 edge 수로 가중 평균 | 8 bins, polarity 불변, scale마다 NMS `0.5 * short side` | `poc/workflow_3/align/matching/ensemble.py:45-67`, `poc/workflow_3/align/matching/ensemble.py:70-110`, `poc/workflow_3/align/matching/ensemble.py:163-174` |

세 채널은 edge 추출만 다르고 결국 template edge가 frame edge 가까이에 있는지를 본다. 저장소 ADR도 flat Chamfer surface와 반복 SEM pattern에서 구조적 diversity가 부족하다고 명시한다 (`poc/workflow_2/docs/study/adr/0003-workflow2-offline-ensemble-lab-experiments.md:36-53`).

### 1.4 RRF, NCC, MIND/ECC, threshold, 출력

- 각 channel은 solo top-24를 만들고, channel 간 후보 center가 template short side의 5% 또는 최소 8 px 안이면 한 cluster로 합친다. rank `r`의 기여는 `1/(10+r)`이고 fused shadow pool은 24개다 (`poc/workflow_3/align/matching/ensemble.py:139-142`, `poc/workflow_3/align/matching/ensemble.py:177-222`).
- Production engine은 shadow 24개 중 앞 8개만 가져와 다시 Canny Chamfer로 점수화한다 (`poc/workflow_3/align/matching/engine.py:906-919`). 각 후보의 raw-gray NCC를 계산하고 `0.5*chamfer + 0.5*max(0,NCC)`로 초기 순위를 만든다 (`poc/workflow_3/align/matching/engine.py:921-931`).
- OM은 raw intensity가 아니라 주변 8-offset self-similarity를 표현하는 MIND-like descriptor로 후보를 재채점한다. 그다음 sel 순서와 MIND 순서를 RRF(k=8)로 합친다. MIND score가 0.10 미만이거나 flat이면 거부한다 (`poc/workflow_3/align/matching/mind_rerank.py:35-48`, `poc/workflow_3/align/matching/mind_rerank.py:101-165`, `poc/workflow_3/align/matching/engine.py:946-950`).
- SEM은 translation-only `findTransformECC`의 correlation coefficient로 후보 순서를 완전히 바꾼다. 100 iteration, epsilon `1e-5`, Gaussian filter 5, `cc<0.05`는 거부한다. shift는 버리고 기존 후보 좌표만 쓴다 (`poc/workflow_3/align/matching/mind_rerank.py:168-206`, `poc/workflow_3/align/matching/engine.py:938-945`).
- 최종 score는 선택된 후보의 위 sel 값이다. OM/SEM 공통 threshold `0.6053`과 `0.4727`로 `match/adjust/low`를 정한다. 이 값은 756 S 표본에서 Youden J와 recall 0.95 지점으로 보정되었고 0.5/0.5 NCC 가중치에 결박돼 있다 (`poc/workflow_3/align/matching/engine.py:148-159`, `poc/workflow_3/align/matching/engine.py:954-967`).
- 출력 `best_xy`는 live frame의 template center다. RCP box route는 여기에 `align_offset_xy * best_scale`을 더해 실제 align point를 만든다 (`poc/workflow_3/align/matching/engine.py:831-849`, `poc/workflow_3/align/correction.py:382-391`).

## 2. 각 채널의 가정과 SEM aperture/process drift 실패 원인

### C1: binary edge 위치가 보존된다는 가정

C1은 밝기 자체가 아니라 edge geometry를 보므로 전역 brightness/contrast 변화에는 raw NCC보다 강하다. 그러나 두 가지를 가정한다.

1. recipe와 live에서 같은 구조가 Canny 60/160을 넘어 edge로 남는다.
2. 정답의 edge set이 decoy보다 더 가까운 유일한 distance-transform basin을 만든다.

공정 drift, focus, charging, scan noise가 약한 edge를 threshold 아래로 내리거나 새 texture edge를 만들면 첫 가정이 깨진다. SEM aperture 문제에서는 key가 frame을 거의 채우고 line-line junction 주변만 위치를 구속한다. 긴 직선 구간은 선을 따라 이동해도 평균 nearest-edge distance가 거의 변하지 않는다. score surface가 평평하니 반복 pitch만큼 이동한 decoy도 같은 score를 낸다. 실제 truth-forced 결과도 229점 중 153점, 67%가 wrong local peak였다 (`poc/workflow_2/docs/study/reranker_ab_failure_analysis.md:28-36`). ADR은 세 channel이 같은 edge-distance surface를 공유한다고 결론낸다 (`poc/workflow_2/docs/study/adr/0003-workflow2-offline-ensemble-lab-experiments.md:36-47`).

### C2: gradient magnitude 상위 밀도가 구조를 안정적으로 대표한다는 가정

C2는 fixed Canny threshold 대신 frame의 Canny density에 맞춰 gradient 상위 3-15%를 고른다. 약한 edge를 더 살리는 대신 template과 live의 edge-density ordering이 같다는 가정이 붙는다. 공정 drift가 live에 texture/noise edge를 늘리면 그 noise가 density budget을 차지한다. 저텍스처 SEM은 3% 하한 때문에 실제 구조가 아닌 tie-breaking noise까지 edge로 만들 수 있다. 이 동작은 코드에 명시돼 있다 (`poc/workflow_3/align/matching/ensemble.py:21-42`). C2도 edge의 위치만 Chamfer에 넣으므로 aperture 방향의 모호성을 해소하지 못한다.

### C3: edge orientation이 decoy를 구분한다는 가정

C3는 같은 위치뿐 아니라 같은 22.5도 orientation bin의 edge가 가까워야 한다. polarity inversion에는 강하다. 그러나 line/space SEM은 대개 수평/수직 직선이 반복되어 정답과 wrong phase가 똑같은 orientation histogram을 갖는다. junction이 crop 안에서 충분히 보일 때만 C1/C2보다 구속력이 생긴다. focus나 morphology drift가 junction을 둥글게 하거나 gradient 방향을 bin 경계 건너로 바꾸면 오히려 정답 score를 깎는다. 8-bin 고정과 binary edge density는 uncertainty를 표현하지 않는다 (`poc/workflow_3/align/matching/ensemble.py:45-67`, `poc/workflow_3/align/matching/ensemble.py:84-110`).

### NCC: 후보 안에서는 raw pixel 선형 상관이 더 높다는 가정

NCC는 mean과 variance를 정규화하므로 affine brightness 변화에는 견딘다. 대신 template와 live의 국소 pixel layout이 선형 관계여야 한다. contrast polarity reversal, charging, focus blur, process morphology change에는 깨진다. Production은 음수 NCC를 0으로 잘라 polarity-reversed evidence를 완전히 버린다 (`poc/workflow_3/align/matching/engine.py:545-565`, `poc/workflow_3/align/matching/engine.py:925-930`). top-8 밖 정답은 애초에 볼 수 없다.

### OM MIND: local self-similarity가 modality drift에도 보존된다는 가정

MIND는 각 pixel이 주변 8 offset과 닮는 패턴을 비교하므로 intensity/polarity drift에 NCC보다 강하다. 그러나 반복 lattice는 정답과 decoy에서 같은 self-similarity를 만든다. flat region은 score 자체가 정의되지 않는다. 현 구현은 후보 좌표를 새로 만들지 않고 순서만 바꾸므로 proposer miss를 절대 복구하지 못한다 (`poc/workflow_3/align/matching/mind_rerank.py:13-23`, `poc/workflow_3/align/matching/mind_rerank.py:146-165`).

### SEM ECC: 후보 crop끼리 translation registration이 수렴한다는 가정

ECC는 local intensity correlation을 최적화하므로 정답 후보가 이미 capture range 안에 있고 crop 간 차이가 translation으로 설명될 때 강하다. 반복 pattern에서는 wrong phase에도 높은 local optimum이 있다. process drift가 non-linear morphology change를 만들면 correlation이 잘못된 peak에 수렴할 수 있다. Production은 ECC shift를 버리고 cc 순위만 쓰므로 새 좌표나 top-K 밖 후보를 만들지 않는다 (`poc/workflow_3/align/matching/mind_rerank.py:168-185`).

## 3. 벤치 방법론 감사

### 3.1 정답과 지표 정의

- RCP cond의 `box_ltrb`는 cursor 좌표를 image px로 바꿔 box template과 offset을 만든다. S msr cond의 `crosshair_xy`는 image px GT align point가 된다. cond가 없으면 localization driver는 crosshair detector로 폴백한다. 모든 row가 순수 cond GT인 것은 아니라는 뜻이다 (`poc/workflow_2/golden_localization_eval_cond.py:1-15`, `poc/workflow_2/golden_localization_eval_cond.py:397-420`).
- `in_topk`의 정의는 이렇다. 후보 center에 `align_offset_xy*scale`을 더해 점 하나를 얻는다. 그 점이 GT에서 template short side의 20% 이내면 hit다. `in_topk`는 그런 후보가 top-8에 하나라도 있는지이고, `rank1`은 그 첫 hit가 1번 후보인지다. 여기서 rank는 proposer의 RRF 순서이며 production의 NCC/MIND/ECC 최종 승자가 아니다 (`poc/workflow_2/align_similarity.py:301-324`, `poc/workflow_2/align_similarity.py:1037-1051`). 여러 modality가 있으면 GT를 포함한 modality를 우선하는 race를 하므로 production의 화면 mode routing보다 낙관적일 수 있다 (`poc/workflow_2/align_similarity.py:327-414`).
- Combined driver도 `topk_rank==1`과 `in_topk`를 같은 정의로 집계한다 (`poc/workflow_2/golden_combined_eval_cond.py:162-168`). 다만 consensus arm의 RCP counterfactual은 center template이고 rcp-only arm은 box template라 arm 간 RCP lift를 직접 비교할 수 없다 (`poc/workflow_2/docs/study/adr/0004-routed-combined-eval-and-s-collection.md:35-38`).

### 3.2 LOO와 표본 수

- History pool이 충분하면 과거 S로 consensus를 만들고 현재 S를 평가해 leakage가 없다. 없으면 같은 recipe/modality S 중 held-out 한 장을 빼고 나머지 최소 2장 median으로 template을 만든다 (`poc/workflow_2/align_similarity.py:936-980`).
- 널리 인용되는 consensus 결과 `in_topk 0.434 -> 0.876`, `rank1 0.318 -> 0.764`는 134 recipe, 403 S-LOO point, `min_s=3` snapshot이다 (`poc/workflow_2/docs/journals/260608/260608_163302_consensus-validated-and-productization-handoff.md:8-12`). 당시 298 recipe 중 S가 4장 이상인 recipe는 1개였다. 정확히 3장이 135개, fail-only가 151개였고 많은 LOO template이 단 2장 median이다 (`poc/workflow_2/docs/journals/260608/260608_163302_consensus-validated-and-productization-handoff.md:26-38`).
- 이후 project summary의 modality 표는 OM 135점, SEM 185점, 합계 320점에서 consensus `rank1 0.852/0.665`를 보고한다 (`docs/project_progress/03_workflow_2.md:74-96`). 최종 MIND/ECC verifier 선택은 별도 67 recipe/334점 set에서 이뤄졌다 (`docs/journals/260720/260720_163416_mind-rerank-포팅-및-커버리지-집계.md:83-100`). 이 세 snapshot은 표본과 질문이 다르므로 한 표의 같은 denominator처럼 합치면 안 된다.

### 3.3 알려진 편향과 운영 성능 방향

| 편향/불일치 | 코드 근거 | 운영 수치 영향 |
|---|---|---|
| S-LOO selection bias | 위치 metric은 crosshair가 있는 성공 S만 평가하고 E는 generic false-positive guard에만 제한 사용한다 (`poc/workflow_2/align_similarity.py:1072-1081`) | 실제 Align Fail/paused frame은 성공 S보다 drift, occlusion, wrong FOV가 심할 공산이 커서 운영 recall/rank1을 과대평가하기 쉽다. 이는 추론이다. |
| 같은 recipe 근접 frame leakage | LOO는 한 frame만 빼며 recipe/equipment/time 독립 split이 아니다 (`poc/workflow_2/align_similarity.py:936-980`) | 최근 appearance 추종이라는 production 목적에는 맞다. 다만 학습 model의 cross-recipe 일반화 성능으로 해석하면 크게 과대평가한다. |
| frame-weighted 집계 | overall은 recipe별 평균이 아니라 각 cell의 `n_S_loo`를 합산한 micro average다 (`poc/workflow_2/align_similarity.py:1082-1119`) | S가 많은 recipe가 전체 수치를 지배한다. recipe macro와 equipment macro가 없으면 deployment population에 따라 과대 또는 과소평가한다. |
| 두 장 consensus 과대표집 | sparse set 때문에 `min_s=3`, 즉 held-out 후 2장 median이 다수다 (`poc/workflow_2/golden_consensus_eval_cond.py:112-129`) | 안정된 8-10장 production history에 견주면 consensus 품질을 과소평가할 수 있다. 반대로 같은 event 근접 frame이면 독립성 부족으로 과대평가할 수 있다. 방향은 데이터 시간 간격에 달렸다. |
| GT tolerance가 template 크기 비례 | 20% short-side tolerance (`poc/workflow_2/align_similarity.py:84-88`, `poc/workflow_2/align_similarity.py:355-375`) | 큰 SEM template은 더 큰 px 오차도 hit로 인정할 수 있어 click 정확도가 부풀 수 있다. OM/SEM px tolerance를 함께 보고해야 한다. |
| modality race | GT를 포함하는 modality를 평가 후 선택 (`poc/workflow_2/align_similarity.py:337-389`) | live mode가 잘못 읽히거나 한 mode로 고정되는 오류를 숨겨 낙관적이다. Combined route의 explicit modality가 더 production에 가깝다. |
| golden clean vs production raw | LOO frame은 기본 crosshair 제거, primary capture는 raw (`poc/workflow_2/golden_consensus_eval_cond.py:101-108`, `poc/workflow_3/align/correction.py:255-264`) | 확정된 distribution mismatch다. 저장된 A/B에서는 crosshair 제거가 약 2%p 낮아 raw가 채택됐다. 그 표본이 동일하다면 clean golden headline은 production raw를 과소평가한다. 표본 동일성은 문서만으로 확정할 수 없다 (`docs/project_progress/03_workflow_2.md:147-149`). |
| recipe key 충돌의 역사 | `recipe_id` leaf만 쓰면 298 dir가 276 key로 덮였고 현재 golden consensus는 `eqp/class/recipe`로 수정됐다 (`poc/workflow_2/golden_consensus_eval_cond.py:333-338`) | 과거 leaf-key 결과는 일부 데이터 유실로 편향됐다. 현재 LOO key는 수정됐지만 production consensus pool은 class/recipe로 장비를 합치므로 두 population이 완전히 같지 않다. |
| combined eligibility granularity | `per_recipe`는 실제 `(recipe, modality)` cell인데 eligible set은 recipe string만 만든다. 한 modality가 eligible이면 해당 `rec_key`의 rcp-only 평가 전체를 건너뛴다 (`poc/workflow_2/golden_combined_eval_cond.py:492-501`, `poc/workflow_2/golden_combined_eval_cond.py:556-572`) | modality fallback 계약과 집계 granularity가 불일치한다. 빠진 sibling modality가 더 어렵다면 routed 성능을 과대평가한다. 실제 방향과 크기는 누락 cell 분포를 재집계해야 확정된다. |
| localization box offset scale 누락 | golden `_localize`는 box offset `(dx,dy)`를 best/candidate scale과 무관하게 그대로 더하지만 production과 proposer KPI는 offset에 scale을 곱한다 (`poc/workflow_2/golden_localization_eval.py:322-341`, `poc/workflow_2/align_similarity.py:367-374`, `poc/workflow_3/align/correction.py:382-387`) | scale이 1이 아닌 box arm의 align point, hit, rank가 틀린다. offset 방향과 scale에 따라 과대 또는 과소평가한다. |
| 평가 scale band drift | consensus proposer KPI는 `(0.6,0.75,0.85,1.0)`을 production band라고 주석하지만 primary production은 `(0.7,0.85,1.0,1.2,1.4)`다 (`poc/workflow_2/align_similarity.py:48-51`, `poc/workflow_3/align/correction.py:58-64`) | 작은 scale에는 유리하고 1.2/1.4 정답에는 불리한 다른 search problem이다. headline recall을 production top-K recall로 그대로 읽을 수 없다. |
| global threshold 재사용 | threshold는 OM/SEM 혼합 756 S에서 계산됐고 현재 mode별 reranker 뒤에도 공통 적용된다 (`poc/workflow_3/align/matching/engine.py:148-159`) | score calibration과 fail-frame false-positive가 과대 또는 과소로 어긋날 수 있다. mode별, fail-frame별 calibration이 필요하다. |

## 4. 결함, 미사용 경로, 고정 파라미터가 recall을 막는 지점

### 4.1 코드로 확정되는 사항

1. **Top-24를 만들고도 production은 top-8만 후단에 넘긴다.** `SOLO_TOP_K=24`, `SHADOW_N=24`지만 engine은 `ens.fused[:policy.top_n]`, 기본 8만 rescore한다 (`poc/workflow_3/align/matching/ensemble.py:139-142`, `poc/workflow_3/align/matching/engine.py:906-914`). 정답이 rank 9-24에 있으면 NCC/MIND/ECC는 그 후보를 아예 못 본다. 직접적인 recall cap이다.
2. **Learned 또는 비-Canny 후보를 넣어도 Canny로 다시 재판한다.** RRF 위치는 `_rescore_positions_to_candidates`에서 C1 Canny Chamfer로 재점수되고, 전부 0이면 `no_candidates`다 (`poc/workflow_3/align/matching/engine.py:398-432`, `poc/workflow_3/align/matching/engine.py:914-919`). C2/C3만 살린 후보는 selection에서 밀리고 future learned 후보도 같은 문제를 겪는다.
3. **Primary scale band가 고정되고 pinning은 경고만 한다.** 0.7-1.4 끝에 best가 있어도 band를 확장하지 않는다 (`poc/workflow_3/align/correction.py:58-64`, `poc/workflow_3/align/correction.py:265-276`). 실제 ratio가 밖이면 모든 channel이 정답 위치를 못 제안한다.
4. **NMS가 가까운 정답 phase를 후보 생성 단계에서 삭제할 수 있다.** 각 scale와 scale 간 global NMS 모두 template short side의 50% 사각 반경을 쓴다. 높은 decoy가 있으면 그 반경 안의 peak를 지운다 (`poc/workflow_3/align/matching/engine.py:309-371`). 삭제된 peak는 reranker가 복구할 수 없다.
5. **C3만 cross-scale NMS가 없고 RRF는 같은 channel의 중복 투표를 허용한다.** Canny/Scharr는 `_collect_candidates`의 global scale NMS를 거치지만 orientation은 scale별 peak를 한데 정렬해 자르기만 한다. `_rrf_fuse`에는 cluster당 channel별 1표 제한이 없어 C3의 같은 위치, 다른 scale 후보가 한 cluster에 여러 번 기여할 수 있다 (`poc/workflow_3/align/matching/ensemble.py:145-174`, `poc/workflow_3/align/matching/ensemble.py:177-202`). orientation 중복 peak를 과대가중하는 확정 결함이다.
6. **세 channel의 핵심 파라미터가 modality와 scale에 무관한 고정값이다.** Canny 60/160, Chamfer tau 10 px, Scharr density 3-15%, orientation 8 bins, RRF k0 10, cluster radius 5%, top-8이 OM/SEM에 공통이다 (`poc/workflow_3/align/matching/engine.py:36-47`, `poc/workflow_3/align/matching/ensemble.py:16-18`, `poc/workflow_3/align/matching/ensemble.py:45-67`, `poc/workflow_3/align/matching/ensemble.py:139-142`, `poc/workflow_3/align/matching/ensemble.py:205-222`). 현재 modality 분기는 proposer가 아니라 reranker에만 있다.
7. **Final ambiguity는 선택 scorer와 다른 signal을 본다.** NCC/MIND/ECC가 고른 후보와 무관하게 `distinctive`, `second_ratio`, `score_gap`은 Canny Chamfer top-2로 계산된다. engine docstring도 advisory라고 경고하지만 correction의 `adjust` presence gate와 optional re-register gate가 이를 사용한다 (`poc/workflow_3/align/matching/engine.py:885-889`, `poc/workflow_3/align/matching/engine.py:784-813`, `poc/workflow_3/align/correction.py:159-200`). recall 자체보다는 action routing 결함이다. 다만 true candidate가 선택돼도 fallback/review로 보낼 수 있다.
8. **반환 candidates의 순서는 최종 순위가 아니다.** `best_xy`는 NCC/MIND/ECC pick인데 `result.candidates`는 Chamfer 내림차순이다 (`poc/workflow_3/align/matching/engine.py:951-967`). downstream이 `candidates[0]`을 최종 pick으로 해석하면 잘못된다. 현 correction은 `best_xy`를 써서 안전하지만 학습 A/B와 debug consumer는 주의해야 한다.
9. **Engine 상단 설명은 stale이다.** docstring은 여전히 Chamfer+ORB `0.6/0.4`를 주 알고리즘처럼 설명하지만 ensemble production은 ORB를 계산하지 않고 `orb_inlier_ratio=0`으로 고정한다 (`poc/workflow_3/align/matching/engine.py:1-14`, `poc/workflow_3/align/matching/engine.py:954-967`). 구현 결함은 아니지만 감사와 유지보수에서 잘못된 mental model을 만든다.
10. **Legacy fallback은 다른 알고리즘이다.** primary와 grid는 ensemble을 쓰지만 degraded legacy live search는 C1+ORB만 쓴다 (`poc/workflow_3/align/grid_search.py:288-300`, `poc/workflow_3/align/live_search.py:246-265`). production 전체를 단일 C1/C2/C3 pipeline 성능으로 보고하면 안 된다.
11. **Lab 전용 경로는 production에 미사용이다.** C4 edge-NCC는 env opt-in lab channel일 뿐 기본 C1/C2/C3에 포함되지 않는다 (`poc/workflow_2/ensemble_lab.py:286-329`, `poc/workflow_2/ensemble_lab.py:387-402`). SIFT/AKAZE/phase 등 `registration_lab.py` arm도 production import가 없고 채택된 MIND/ECC만 별도 production 구현으로 포팅됐다 (`poc/workflow_2/registration_lab.py:1-38`, `poc/workflow_3/align/matching/mind_rerank.py:25-27`). Template bank 역시 bench 전용이며 ADR에서 기각됐다 (`poc/workflow_2/template_bank_lab.py:1-16`, `poc/workflow_2/docs/study/adr/0006-template-bank-matcher-rejected-fusion-exhausted.md:60-76`).

### 4.2 성능 영향이 합리적이지만 아직 추측인 사항

- **NMS 0.5 short-side가 SEM wrong phase를 과억제할 가능성**: 반복 pitch가 이 반경보다 작으면 높은 decoy 곁의 true junction peak까지 함께 사라진다. 코드는 확실하지만 실제 miss 중 이 원인의 비율은 candidate pre-NMS dump 없이는 모른다.
- **RRF spatial cluster가 scale을 무시한다. 다른 물체를 합치는 쪽인지 같은 물체를 나누는 쪽인지는 미확인**: center 5% radius만 보고 scale 차이는 cluster key에 넣지 않는다 (`poc/workflow_3/align/matching/ensemble.py:177-202`). pre/post cluster GT rank A/B가 필요하다.
- **고정 `DT_TAU_PX=10`이 scale별 score calibration을 왜곡할 가능성**: edge distance는 absolute frame px인데 scale-normalization이 없다. RRF가 channel score scale은 제거하지만 channel 내부 multi-scale rank에는 남는다.
- **장비 무관 consensus pooling이 tool-specific drift를 평균낸다. 그 결과가 blur인지 잘못된 외형인지는 미확인**: production 정책은 확인되지만 tool holdout 통계가 없어 방향과 크기는 모른다.
- **threshold가 `best_sel`에만 걸려 ECC/MIND confidence를 반영하지 않는 문제**: mode reranker가 낮은 sel 후보를 고르면 일부러 보수적 `low`가 된다 (`poc/workflow_3/align/matching/engine.py:933-957`). false action은 줄일 수 있지만 true recovery recall도 낮출 수 있다. decision confusion matrix가 필요하다.

## 5. 학습 기반 교체 경계 제안

### 결론: pair-ranker가 아니라 proposer부터, 단 scorer contract까지 함께 바꾼다

가장 맞는 1차 경계는 `AlignKeyTemplate + live gray frame + scales -> top-K candidates`다. 기존 `AlignKeyTemplate`에는 raw template, modality, physical scale, `align_offset_xy`가 이미 있다 (`poc/workflow_3/align/matching/engine.py:75-88`). downstream은 candidate의 `xy`, `scale`, `template_size`와 최종 `best_xy` 계약을 쓴다 (`poc/workflow_3/align/matching/engine.py:90-103`, `poc/workflow_3/align/matching/engine.py:841-849`). 따라서 dense learned heatmap 또는 learned correspondence가 frame 좌표의 후보 list를 내도록 만들면 correction, offset, clamp, dry-run, fallback, OK safety는 유지할 수 있다.

다만 `compute_ensemble_candidates`만 learned 구현으로 바꾸는 것은 불충분하다. 현 engine이 그 후보를 Canny로 rescore하고 top-8로 자른 뒤 NCC/MIND/ECC로 다시 고르기 때문이다. 학습 proposer가 classical edge로 약한 정답을 새로 찾는 것이 목표라면 바로 그 후보가 downstream에서 다시 밀린다. 최소 실험 seam은 다음이어야 한다.

```text
LearnedProposerResult
  candidates: [{xy, scale, proposal_score, template_size}]
  score_map/version/runtime
        |
        v
candidate scorer
  baseline arm: current Chamfer + NCC + MIND/ECC
  learned arm: learned proposal_score 보존 + optional NCC/MIND/ECC verifier
        |
        v
calibrated result
  best_xy, best_scale, decision, candidates, abstain/reject_reason
        |
        v
existing align_offset, clamp, fallback, engineer-review, dry-run safety
```

우선순위는 다음과 같다.

1. **P0 learned dense proposer**: template와 full frame을 입력해 target-center likelihood heatmap을 내고 기존 좌표계의 NMS top-24와 top-8을 모두 저장한다. loss는 cond GT 중심 heatmap 또는 offset classification/regression을 쓴다. same-frame periodic decoy는 hard negative로 준다. 기존 C1/C2/C3와 union 또는 RRF A/B를 하되 learned-only recall@8/24도 별도 보고한다.
2. **P0 supervised domain encoder, P1 self-supervised pretraining**: 수백만 cond-labeled S가 있으므로 supervised proposer가 먼저다. 같은 corpus의 S/E 및 crop을 DINO/MAE류로 pretrain한 뒤 proposer를 fine-tune하는 arm에서는 sample efficiency와 tool/time drift robustness를 비교한다. E는 target GT가 없으면 supervised positive로 쓰지 않는다.
3. **P1 learned scorer**: proposer recall이 실제로 오른 뒤에만 learned pair/cross-attention scorer를 top-24 전체에 적용해 rank1로 변환한다. 87% miss가 유지되는 동안 top-8 pair-ranker만 개선하는 것은 잘못된 목표다.
4. **전체 end-to-end `best_xy` 회귀는 보류**: coordinate contract와 abstention을 잃고 periodic scene에서 단일 평균 좌표를 낼 위험이 있다. dense heatmap으로 multiple hypotheses와 ambiguity를 보존하는 편이 현 safety router에 맞는다.

### 학습/평가 split과 acceptance gate

- split 단위는 image가 아니라 최소 recipe-disjoint이며 equipment와 time holdout을 별도 둔다. 같은 recipe의 연속 S가 train/test에 섞이면 외형 기억을 generalization으로 오판한다.
- primary KPI는 전체 및 OM/SEM별 `recall@8`, `recall@24`, `rank1`, px error다. 87% 병목을 직접 겨냥했는지 확인하려면 기존 `gt_in_topk=false` subset에서 learned proposer가 새로 건진 비율을 필수로 낸다.
- production 승격 전에는 기존 threshold를 재사용하지 않는다. learned score와 union candidate pool의 분포에서 mode별 calibration을 새로 한다. fail/paused E 또는 shadow frame으로는 high-confidence off-target와 abstention을 검증한다.
- first rollout은 offline, then office shadow다. learned output은 `best_xy`와 click에 연결하지 않고 artifact로만 저장한다. recipe/equipment/time holdout과 live paused distribution에서 개선이 재현될 때 기존 fallback/action gate 뒤에 넣는다.

### 최종 판단

7월 문서의 VLM abstention, DINOv2 descriptor, LoFTR/RoMa verifier 자체를 반복할 이유는 없다. 새 정보가 바꾸는 것은 실패 표적과 데이터 규모다. 현재 병목은 후보 안의 순위가 아니라 후보 생성이다. 무료 cond 좌표가 수백만 개라면 가장 작은 유효 실험은 현 top-K pair-ranker가 아니라 `AlignKeyTemplate`을 받아 full-frame candidate heatmap을 내는 learned proposer다. 단, 현 top-8 cap과 Canny rescore를 그대로 두면 성공을 스스로 지우므로 proposer와 candidate-score 보존 seam을 함께 평가해야 한다.
