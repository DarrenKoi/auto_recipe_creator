# 개념 설명서 — Align-key 매칭 평가에서 쓰는 용어들

> 이 문서는 주간 리포트와 벤치 digest에 반복 등장하는 용어 — **template bank, heatmap(soft-voting), RRF, baseline, digest, consensus, in_topk vs rank-1, kill-test** — 를 비전문가도 이해할 수 있게 풀어 설명한다. 기술 용어는 영어를 유지하되 개념은 한국어 prose로 설명한다.

---

## 0. 먼저, 우리가 푸는 문제 한 줄 요약

CD-SEM 장비가 wafer 위에서 **align-key**(정렬 기준 마크)를 찾아야 측정을 시작한다. 가끔 장비가 이 마크를 못 찾아 **align fail**이 난다. 우리 시스템은 화면을 캡처해 "이 큰 SEM 이미지 안에서 align-key가 정확히 *어디에* 있는가?"를 좌표로 집어내야 한다.

이게 본질적으로 **template matching**(틀 맞추기) 문제다. "이렇게 생긴 마크"라는 작은 참조 이미지(template)를 들고, 큰 화면 안을 훑어 가장 닮은 위치를 찾는다.

> 비유: *월리를 찾아라*. 작은 월리 그림(template)을 들고 큰 그림(화면) 안에서 월리가 선 정확한 한 점(align point)을 가리키는 것.

이 문서의 모든 용어는 결국 **"닮은 위치를 어떻게 더 잘 집어내느냐"**를 둘러싼 기법들이다.

---

## 1. Template (틀) — 매칭의 출발점

**Template**은 "찾고자 하는 것이 이렇게 생겼다"는 작은 참조 이미지다. 우리는 template을 어디서 얻느냐에 따라 두 종류를 쓴다.

| 종류 | 출처 | 약자 |
| --- | --- | --- |
| **rcp template** | 레시피에 *등록된* align-key 이미지 1장 | from_rcp |
| **success(S) frames** | 과거에 align이 *성공*했을 때 찍힌 실제 화면들 (여러 장) | from_msr의 S |

여기서 중요한 도메인 사실: **align-key는 픽셀이 정확히 똑같이 다시 나타나지 않는다.** 공정이 조금씩 변하기 때문이다. 그래서 "픽셀 대 픽셀 비교"가 아니라 **구조(structure)/기하(geometry)** 기반으로 닮음을 잰다. (이 프로젝트에서 NCC/SSIM 같은 순수 픽셀 일치도를 1차 지표로 금지하는 이유.)

---

## 2. Consensus (median) template — 현재 프로덕션 방식

여러 장의 S frame이 있으면, 가장 단순한 아이디어는 **"여러 장을 한 장으로 평균 내자"**다. 픽셀별 **median(중앙값)**을 취해 노이즈를 지우고 공통 구조만 남긴 **대표 한 장**을 만든다. 이걸 **consensus template**이라 부른다.

```
S1, S2, S3, ... S8   →   [median 합성]   →   consensus 한 장   →   매칭
 (개별 성공 프레임들)        (한 장으로 뭉침)        (대표 template)
```

- **장점**: 한 장만 매칭하면 되니 빠르고, 무작위 노이즈가 지워진다.
- **약점**: median이 **distinctive한 디테일까지 blur로 뭉갠다.** 미세하게 어긋난 구조들이 평균되며 흐려진다. 변별력 있는 모서리가 무뎌질 수 있다.

이 "median이 디테일을 뭉갠다"는 약점이 바로 다음 아이디어(template bank)의 출발점이다.

---

## 3. Template Bank — "한 장으로 뭉치지 말고, 여러 장을 그대로 들고 가자"

**Template bank**의 핵심 아이디어 한 줄:

> **여러 S frame을 median으로 합치지 말고(=blur 금지), N장을 *각각 선명하게(sharp)* 유지한 채 매칭한 뒤, 그 결과들을 *나중에* 합치자.**

```
consensus 방식:  S1..S8 → [먼저 median] → 1장 → 매칭          (먼저 합치고 매칭)
bank 방식:       S1..S8 → 각각 매칭 → [나중에 결과 융합]       (매칭하고 나중에 합침)
```

"합치는 순서"를 뒤로 미룬 것이 전부다. 이렇게 하면 각 멤버의 선명한 디테일이 살아 있는 채로 매칭에 참여한다. 그러면 **N개의 매칭 결과를 어떻게 하나의 답으로 합치느냐**가 새 문제로 떠오른다 — 그 합치는 방법이 바로 **heatmap**과 **RRF**다.

> 코드: `bank_build()`가 N장을 *개별* `AlignKeyTemplate`로 빌드한다(median 합치기 없음). `poc/workflow_2/template_bank_lab.py`.

---

## 4. Heatmap (soft-voting) — "약한 표도 다 더한다"

### 매칭 한 번이 만드는 것: score map

template 한 장을 큰 화면 위에서 한 칸씩 밀며 모든 위치의 "닮음 점수"를 재면, 화면과 같은 크기의 **점수 지도(score map)**가 나온다. 닮은 곳은 밝고(높은 점수), 안 닮은 곳은 어둡다.

### Heatmap 융합 = 점수 지도를 그냥 다 더하기

bank의 N개 멤버 각각이 점수 지도를 하나씩 만든다. **heatmap 융합**은 이 N개의 지도를 **픽셀별로 그냥 SUM(합산)**한 뒤, 합산 지도에서 가장 밝은 점(peak)을 정답으로 고른다.

```
멤버1 점수지도   멤버2 점수지도   멤버3 점수지도
     ░▒▓             ░░▒             ▒░░
      +               +               +        →   SUM   →   가장 밝은 한 점 = align point
```

```python
# 핵심은 이 한 줄 (template_bank_lab.py:_accumulate_heatmap)
acc[oy:oy+sh, ox:ox+sw] += score_map      # 멤버마다 점수 지도를 누적(SUM)
# 이후 acc 에서 argmax peak 추출
```

**왜 SUM이 좋을 거라 기대했나 (핵심 가설):**

진짜 align point는 *모든* 멤버에서 **약하게라도 일관되게** 밝다. 어떤 한 멤버에서는 1등이 아니더라도, 여러 멤버에서 조금씩 밝은 점이 **합산되면 강하게 살아남는다.** 반대로 어쩌다 한 멤버에서만 우연히 밝은 distractor(가짜 후보)는 합산에서 묻힌다.

> 이걸 **soft-voting**(부드러운 투표)이라 부른다. "1등만 인정"이 아니라 **모두가 가진 신뢰도 점수를 그대로 더하는** 투표라서 *soft*다. 이 방식은 "어느 멤버도 top-K 안에 못 넣은 진짜 점"(`gt_not_in_topk`)을 살려내는 걸 노린다.

---

## 5. RRF (Reciprocal Rank Fusion) — "순위만 보고 표를 모은다"

**RRF = 역순위 융합.** heatmap이 *점수값*을 더하는 것과 달리, RRF는 **점수값을 버리고 *순위(rank)*만** 본다.

각 멤버가 자기 후보들을 1등, 2등, 3등... 으로 순위 매긴다. 같은 위치에 여러 멤버가 표를 던지면, **그 위치의 순위가 높을수록 큰 표, 낮을수록 작은 표**를 받는다. 그 공식이 "순위의 역수":

```python
# template_bank_lab.py:bank_match_rrf
hit["rrf"] += 1.0 / (rrf_k + rank)     # rank=0(1등) → 큰 값, rank가 클수록 작은 값
```

`rrf_k`는 1등의 영향력이 과하게 폭발하지 않도록 눌러주는 상수(보통 60).

**heatmap vs RRF 한눈 비교:**

| | **heatmap (soft-voting)** | **RRF** |
| --- | --- | --- |
| 무엇을 더하나 | 닮음 **점수값**(연속) | 순위의 **역수**(이산) |
| 1등/꼴등 | 점수 크기 그대로 반영 | 등수만 보고, 점수 차이는 무시 |
| 강점 | 약한 신호도 살림 | 점수 스케일이 멤버마다 들쭉날쭉해도 강건 |
| 우리 역할 | **primary**(주력 가설) | **extra**(보조 — "이산 기교가 추가 가치를 주나?" 확인용) |

> 왜 RRF를 보조로만 뒀나: 프로덕션 매처가 이미 *discrete 후보 → RRF → rerank* 구조라 그 천장(0.2~0.3)에 갇혀 있었다. 같은 융합 family인 RRF-bank는 그 천장을 물려받을 공산이 크다. heatmap은 *dense 누적*이라 다른 family여서 주력으로 택했다.

---

## 6. in_topk vs rank-1 — 어느 지표를 믿어야 하나 (★ 가장 중요)

매칭이 답 후보를 여러 개(top-K, 보통 K=8) 내놓는다고 하자. 성능을 재는 두 지표:

- **in_topk**: 정답이 **상위 K개 후보 *안에* 들어 있나?** (8개 중 하나라도 맞으면 성공)
- **rank-1**: **맨 위(1등) 후보가 정답인가?** (1등이 맞아야만 성공)

```
후보: [① ② ③ ④ ⑤ ⑥ ⑦ ⑧]
       ↑                    정답이 ⑤에 있으면:
       └ rank-1은 ①만 본다     • in_topk = 성공 (8개 안에 있음)
                               • rank-1  = 실패 (1등은 ⑤가 아님)
```

**왜 이 구분이 우리 프로젝트의 핵심 교훈인가:**

실전에서 시스템은 **1등 좌표 하나를 클릭**한다. 후보 8개를 다 클릭할 수는 없다. 그러니 **실제로 출하되는 성능 = rank-1**이다.

`in_topk`는 **천장(ceiling)** — "이론상 가능한 최선"일 뿐이다. `in_topk`가 아무리 높아도 `rank-1`이 낮으면, 정답을 후보엔 넣지만 **1등으로 못 꼽는다 = 순위 매기기(ranking) 능력이 없다 = 동전 던지기.**

> **벤치 A/B에서 항상 in_topk가 아니라 rank-1을 비교하라** — 이게 이번 주 가장 비싸게 배운 규율이다. template-bank가 `in_topk`로는 consensus를 이겼지만 `rank-1 ≈ 0.5`(반반)였기에 **출하 불가** 판정이 났다.

---

## 7. Baseline · Digest · A/B — 결과를 어떻게 읽나

### Baseline (기준선)

**Baseline = 비교의 기준이 되는 "지금 방식"**이다. 새 아이디어가 좋은지는 절대 점수가 아니라 **baseline 대비 얼마나 나아졌나(lift)**로 판단한다. 우리 baseline은 두 개:

- **rcp-only**: 레시피 등록 이미지 1장만 쓰는 가장 기본 방식.
- **median-consensus**: 현재 프로덕션 방식(§2).

새 arm(heatmap/RRF)을 이 둘과 나란히 놓고 같은 골든셋에서 점수를 매겨야 "진짜 개선"인지 "그냥 다른 숫자"인지 가른다.

### Digest (요약 한 줄)

오피스 fab 데이터는 Mac으로 반출이 **금지**돼 있다. 그래서 평가를 오피스에서 돌리고, 결과를 **콘솔에 한 줄로 압축 출력**해 그 텍스트만 가져와 분석한다. 이 한 줄이 **`[DIGEST]`**다.

```
[DIGEST] om[strong 42, confirmed 0] | sem[strong 84, confirmed 0]
         └─ modality별 핵심 숫자만 압축 — 이미지 없이 텍스트로 결과 relay
```

> 이 워크플로우 전체가 "**Mac에서 blind 작성 → 오피스 git pull 실행 → digest 라인만 relay**"의 반복이다. digest는 이미지를 못 가져오는 제약을 우회하는 핵심 장치.

### A/B (대조 실험)

같은 입력(골든셋)에 **방식만 바꿔(A=baseline, B=새 arm)** 돌려 점수를 비교하는 것. 우리 골든 드라이버(`golden_consensus_eval_cond.py` 등)가 한 번 실행에 여러 arm을 나란히 평가하고 digest로 비교 결과를 뱉는다.

---

## 8. Kill-test (반대가설 죽이기)

좋은 실험은 "내 아이디어가 맞다"를 증명하려 들지 않고 **"내 아이디어가 틀릴 수 있는 가장 그럴듯한 경로를 먼저 죽인다."** 이게 **kill-test**다.

template-bank의 반대가설(H0):

> **H0**: align-key 안의 **주기적 distractor**(반복 패턴)가 S frame들 사이에서 *일관*되면, 모든 멤버가 **똑같은 엉뚱한 격자점(wrong lattice)**을 가리킨다. 그러면 합의(heatmap/RRF)가 오히려 그 **가짜를 강화**해 consensus보다 *더 나쁠* 수 있다.

kill-test는 격자 주기를 추정하고 winner를 GT 기준 버킷(`near_periodic` 등)으로 분류해 **H0가 실제로 발화하는지 직접 측정**한다. (이번엔 H0는 통과 — near_periodic이 om 0.014/sem 0.052로 낮아 distractor 강화는 일어나지 않았다.)

> 교훈: 가설이 통과(H0 기각)해도 *충분조건은 아니다*. H0는 안 터졌지만 결국 **rank-1**에서 막혔다.

---

## 9. 그래서 결론은? — template-bank는 왜 기각됐나 (ADR 0006)

이 모든 개념을 엮은 최종 판정:

1. **kill-test 통과** — distractor가 합의를 망치진 않았다(H0 기각). ✅
2. **in_topk는 consensus를 이김** — 정답을 후보 안엔 잘 넣는다. ✅
3. **그러나 rank-1 ≈ 0.5** (OM/SEM 둘 다) — **1등으로 꼽는 건 동전 던지기.** ❌

→ §6에서 봤듯 실전은 **1등 하나를 클릭**하므로 rank-1 0.5는 **출하 불가**. 게다가 heatmap·RRF·기타 3가지 융합 방법이 **모두 같은 벽**에 막혔다 → **SEM은 어떤 멤버 융합(member-fusion)으로도 풀 수 없는 "순위/변별력(ranking/distinctiveness) 문제"**라는 결론.

> **진짜 레버는 매처가 아니라 *재등록(re-registration)*** — align-key 자체를 더 변별력 있는 영역으로 다시 등록하는 것. 매처를 아무리 정교하게 만들어도, 등록된 key가 애초에 변별력이 없으면 1등을 꼽을 근거가 없다.

template-bank 코드는 `TBANK_HEATMAP=0` kill switch 뒤에 남겨 두었고(16/16 테스트 통과), `workflow_3` 프로덕션에는 **포팅하지 않았다.**

---

## 부록 — 한눈 용어집

| 용어 | 한 줄 정의 |
| --- | --- |
| **template** | "찾는 게 이렇게 생겼다"는 작은 참조 이미지 |
| **rcp / S(success) frame** | template 출처: 레시피 등록본 / 과거 성공 화면 |
| **consensus (median)** | S들을 median으로 합친 대표 한 장 (현 프로덕션, blur 약점) |
| **template bank** | S들을 합치지 않고 개별 유지 후 *나중에* 결과 융합 |
| **score map** | template를 화면 전체에 밀며 잰 위치별 닮음 점수 지도 |
| **heatmap (soft-voting)** | 멤버별 score map을 **SUM** → peak (primary, 약한 신호 살림) |
| **RRF** | 점수 버리고 **순위의 역수**를 더해 융합 (extra, 스케일에 강건) |
| **in_topk** | 정답이 상위 K 후보 안에 드나 (=천장/ceiling 지표) |
| **rank-1** | 1등 후보가 정답인가 (=실제 출하 성능, 진짜 지표) |
| **baseline** | 비교 기준 (rcp-only, median-consensus) |
| **digest** | 오피스 결과를 한 줄로 압축한 `[DIGEST]` 텍스트 (데이터 반출 우회) |
| **A/B** | 같은 입력에 방식만 바꿔 점수 비교 |
| **kill-test** | 반대가설(H0)을 먼저 죽이는 검증 |
| **re-registration** | align-key를 더 변별력 있는 영역으로 재등록 (진짜 레버) |
