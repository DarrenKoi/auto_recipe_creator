# 260602 — test1 분석: MI reranker 폐기, contour reranker 로 전환

> 세션: 2026-06-02 (오전)
> 주제: 오피스 `align_similarity.py` test1 결과 분석 → 재등록 재확인, **MI reranker 실데이터로 폐기**,
>        **왜 MI 가 안 되는지** 규명 + **contour reranker 로 교체** 결정.
> 결과지: `poc/workflow_2/docs/console_results/260602_test1.txt`
> 관련: [260529 consensus A/B verdict](../260529/260529_152818_consensus-ab-verdict-and-next-steps.md),
>        [matcher 신뢰도 저널](../260529/260529_130919_matcher-reliability-and-diagnostics.md),
>        매칭 제약 [[project_align_key_matching_constraint]], [[project_matcher_flat_chamfer_distinctiveness]]

---

## 1. 진행 사항

오피스에서 `align_similarity.py` 를 실데이터로 돌린 결과(test1)를 받아 분석했다. 지난 세션(260529)은
consensus A/B 로 **재등록 + reranker** 방향을 잡았는데, 이번 test1 은 거기에 **MI-rerank 측정 컬럼**과
**rank-1 정밀도 컬럼**이 들어간 첫 실데이터 실행이라 reranker 선택을 데이터로 판정할 수 있었다.

핵심 결론 세 가지:
1. **rcp 재등록 = 재확인.** consensus 로 recall +0.282, rank1 +0.269.
2. **co-registration(ECC) = 배제 재확인.** consensus 가 흐릿하지 않음(선명도 비율 통과).
3. **MI reranker = 폐기.** 두 측정 모두 lift 음수. → **contour reranker 로 교체**(다음 검증 대상).

그리고 사용자 요청대로 **MI 가 왜 안 되는지**(§3)를 규명하고, 메모리·권위 문서(procedure §6.1)에
이 결정을 반영했다.

---

## 2. 실험 결과 (test1 상세 해석)

### 2-1. CONSENSUS A/B (recipes=15, S(LOO)=78) — 재등록 판정

| 지표 | rcp | consensus | lift | 게이트 |
|---|---|---|---|---|
| recall (in_topk) | 0.436 | **0.718** | **+0.282** | ≥+0.10 ✅ |
| precision (rank1) | 0.269 | **0.538** | **+0.269** | 나빠지면 안 됨 ✅ |
| +MI rerank rank1 | 0.308 | **0.526** | **−0.013** | ≥+0.10 ❌ |
| 선명도 비율 vs S개별 | — | edge=**0.979** / lap=**0.959** | — | edge≥0.70·lap≥0.50 ✅ |

- **재등록 = 확정 재확인.** consensus(=S 이미지들의 합의 템플릿)를 rcp 대신 쓰면 recall 이 0.436→0.718 로
  뛴다. 이는 *전역* proposer 천장(아래 gt-topk in_topk=0.594)마저 넘는 수치 → 후보를 재배열하는 게 아니라
  **recall 천장 자체를 올린다**. rcp staleness(등록 align key 가 현재 공정과 멀어짐)가 실재함을 다시 입증.
- **co-registration = 배제.** consensus 가 개별 S 대비 흐릿하면(blur) ECC 정합이 필요한데, 선명도 비율
  edge=0.979 / lap=0.959 로 둘 다 임계 통과 → consensus 는 선명함. ECC 불필요(지난 결정 유지).
- **MI rerank = 음수.** consensus 에 MI 리랭킹을 얹으면 rank1 이 0.538 → 0.526 (**−0.013**). 회복은커녕
  미세하게 악화.

### 2-2. GT-IN-TOPK (rcp 템플릿, 전역) — proposer recall 천장 + reranker 여지

```
in_topk = 136/229 (0.594)   rank1 = 78 (0.341)   miss = 93   median_miss_dist_norm = 0.376
MI rerank rank1 = 77        (chamfer rank1 0.341 → MI 0.336;  in_topk 0.594 가 천장)
rank_hist : {1:78, 2:19, 3:11, 4:5, 5:7, 6:5, 7:7, 8:4}
```

- **reranker 여지는 실재.** 정답이 top-K 안에 든 136건 중 78건만 rank1, 나머지 **58건이 rank 2~8** →
  좋은 reranker 면 rank1 을 0.341 → 0.594(in_topk 천장) 쪽으로 끌어올릴 수 있다(상한 +0.25).
- **그러나 MI 는 그 여지를 못 먹는다.** chamfer rank1 0.341 → MI 0.336 (−0.005). consensus A/B(−0.013)와
  **방향·부호 일치** → 단일 노이즈가 아니라 MI 의 구조적 무력함.
- miss 93건은 정답이 후보에 아예 없음(median_dist 0.376, 멀다) → 이건 reranker 가 아니라 proposer 문제.

### 2-3. S/E 분리도 — "key 있음/없음" 신호로서의 지표 비교

| metric | med_S | med_E | bACC | 비고 |
|---|---|---|---|---|
| free_best | 0.6637 | 0.7038 | 0.579 | **med_E>med_S(역전)** |
| at_center | 0.0226 | nan | 0.503 | 무력(nan) |
| at_crosshair | 0.6424 | 0.6921 | 0.582 | **역전** |
| **mi_free** | **0.0749** | **0.0365** | **0.627** | 방향 정상·**1위** |
| ncc_free | 0.0203 | −0.0006 | 0.598 | 방향 정상 |
| free_best_box | 0.7007 | 0.7174 | 0.61 | 역전 |

- chamfer 계열(free_best/at_crosshair/free_best_box)은 **med_E ≥ med_S 로 역전** — S/E 를 못 가른다
  (엣지 clutter 양을 재는 셈). [[project_matcher_flat_chamfer_distinctiveness]] 재확인.
- **MI 만 방향이 맞고 bACC 1위(0.627).** 단 절대 변별력은 여전히 약함.
- E 분해: `no_crosshair=127`, `with_crosshair=71`, `recoverable_by_move=10`. fail 의 대부분(127/198)이
  crosshair 자체 미검출 → recenter 재시도로 복구 가능한 건 ~10건뿐([[project_e_images_no_crosshair]]).

### 2-4. TRUTH-FORCED — matcher 천장이 구조적임을 확인

```
valid/truth = 76/229  (wrong_local_peak = 153, 67%)
median chamfer: wide=0.7409  compare=0.7381   scale_gain=0.0252   orb=0.0   mean_dt_px=3.0
wide_best_scale hist: {1.0:26, 0.5:46, 0.6:4}     진단 counts: {ok:76, wrong_local_peak:153}
```

- 정답 위치로 **강제**해도 best peak 이 67% 는 딴 데로 샌다 → 점수면이 평평/다봉.
- `scale_gain≈0.025` → scale-band 문제 아님. `orb=0.0` → ORB 가 정답에서도 기여 0.
- 즉 matcher 천장은 scale/inpaint 가 아니라 **metric 변별력** 자체. rank1 을 0.6 이상으로 올리려면
  결국 2차 proposer 가 필요(reranker 만으로는 in_topk=0.594 천장).

---

## 3. WHY — MI 는 왜 reranker 로 안 되는가 (핵심)

> 한 줄: **MI 는 "맞는 축"이 다르다. 프레임 단위 분리(present/absent)는 되지만, 한 프레임 안에서
> 위치(true peak vs decoy)를 가르는 데는 무력하다.**

두 작업은 서로 다른 변별축을 요구한다:

1. **S/E 분리** = "이 프레임에 align key 가 있나/없나?" — 프레임 *전체*의 통계를 비교하는 이미지 단위 판정.
   S(정렬 성공·key 존재)는 template 과의 MI 가 높고 E(없음/오정렬)는 낮다 → **프레임 모집단 간** MI 차이가
   생겨 약하게 분리됨(bACC 0.627).

2. **reranking** = "한 프레임 안 top-K 후보 *위치* 중 진짜는?" — *같은 프레임 안*에서 정답 위치 crop 이
   7개 decoy 위치 crop 보다 점수가 높아야 함. 즉 **프레임 내 위치 변별**이 필요.

MI 가 (2)에서 실패하는 이유:
- top-K chamfer 후보는 **같은 프레임**의 local peak 들이다. 이미 chamfer 가 엣지 유사도로 골라낸,
  서로 비슷한 후보들. 같은 프레임이라 intensity/texture 분포를 공유 → template 과의 MI 값이
  후보끼리 거의 같다. **후보 간 MI 분산 ≪ 프레임 간(S/E) MI 분산.**
- MI 는 *공간 배치에 둔감한 전역 intensity 종속성* 척도다. 두 crop 의 히스토그램 공기여(co-occurrence)가
  같으면 픽셀이 공간적으로 뒤섞여도 MI 가 같다. 그런데 reranker 가 필요한 건 정확히 **공간 정합 품질**의
  변별 — MI 가 구조적으로 못 보는 것.
- 그래서 MI 의 변별력은 *모집단/전역 축*(present vs absent)에 살고, *프레임 내 국소 위치 축*(true vs decoy)
  에는 ~0. 실데이터가 정확히 그 모양(분리 bACC 0.627 / rerank lift −0.013·−0.005)으로 나왔다.

---

## 4. 결정 — contour reranker 로 교체 (가설, A/B 검증 대상)

- rerank 가 필요로 하는 건 key 의 **기하/형상** 변별이다. **contour 매칭**(`cv2.matchShapes` / Hu moments,
  connected-component 위상 비교)은 엣지/blob 의 *공간 배치*를 비교하므로 **위치 변별적**이다:
  정답 위치는 template 과 contour 위상이 맞고, decoy 는 엣지 밀도·intensity 가 비슷해도 contour 배치가 다르다.
- chamfer(매끄러운 거리장 → 후보는 잘 모으나 국소 peak 끼리 평평)·MI(전역 통계)가 못 가르는 부분을
  contour 의 *연결성/형상 위상*이 메운다. chamfer 는 **멤버십 게이트**로 유지, contour 는 **순서만** 결정.
- 도메인 원칙과도 정합: [[project_align_key_matching_constraint]] 가 "픽셀 동일성 금지, 기하/구조 매칭"을
  요구 → contour 는 기하/구조, MI 는 intensity 통계. 도메인이 기하 쪽을 가리킨다.

⚠️ **아직 가설.** test1 이 입증한 건 "MI 폐기"까지다. contour 는 **같은 A/B 하네스**에서
`rerank_rank1_lift ≥ +0.10` 을 통과해야 production 승격. contour 도 실패하면 → 후보가 본질적으로 모호하다는
뜻 → reranker 가 아니라 **proposer 교체 / live-search 단계 분리**로 escalation.

### green mark rank-1 로드맵 (갱신)

| 단계 | rank-1 | 상태 |
|---|---|---|
| 현재(rcp) | 0.27 | — |
| + 재등록(consensus→새 rcp) | 0.54 | **검증됨(test1 재확인)** |
| + ~~MI~~ → **contour** reranker | ~0.59(천장) 목표 | **MI 폐기, contour 검증 예정** |
| 잔여(in_topk 천장 0.59) | proposer recall 한계 | 2차 proposer / golden 확장 |

---

## 5. 수정 내용 (이번 세션, 코드 변경 없음 — 문서/메모리만)

- **결과지 적재:** `poc/workflow_2/docs/console_results/260602_test1.txt` (사용자가 오피스에서 작성).
- **메모리 갱신:** `project_matcher_flat_chamfer_distinctiveness.md` 에 **test1(2026-06-02)** 섹션 추가
  (MI 폐기 + WHY + contour 전환). reranker 줄에서 "MI/contour" → "MI 폐기, contour" 로 정정.
- **권위 문서 갱신:** `workflow_2_procedure.md` §6.1 "matcher 정밀도 결정(2026-06-02)" 신설 — 재등록 확정,
  co-reg 배제, MI 폐기 사유, contour 다음 후보, matcher 천장 구조적.
- **버퍼:** `.remember/now.md` 에 07:53 항목 추가.
- `align_similarity.py` **코드는 미변경**(working tree clean). reranker 교체는 다음 세션 구현 대상.

---

## 6. 다음 단계 (우선순위)

1. **(1순위) contour reranker A/B 컬럼 추가** — `align_similarity.py` 의 `_consensus_template_ab` /
   `_gt_in_topk` 에 MI 자리와 같은 방식으로 contour 재정렬 컬럼 추가:
   - held-out S 의 consensus top-K chamfer 후보 crop 을 `_matched_crop(...)` 으로 추출 → template 과
     contour 유사도(`cv2.matchShapes`(Hu) 또는 contour 위상 점수)로 reorder.
   - 새 컬럼: `cons_rank1_rate_reranked_contour`, `rerank_rank1_lift_contour`.
   - **설계 질문(다음 세션 시작 시 확정):** (a) contour 추출 전처리 — 이진화/threshold 방식
     (Otsu vs adaptive), 어떤 엣지/마스크에서 contour 를 딸지. (b) 점수화 — `matchShapes` 단독 vs
     Hu moment 거리 + 면적/aspect 게이트 조합. (c) chamfer 게이트 유지 + contour 순서만(하드 reorder)으로
     시작, alpha 블렌딩은 후순위.
2. **(2순위) 재등록 도구** — consensus(`<out_dir>/consensus/<recipe>__<mod>.png`)를 그대로 새 rcp 로
   교체하는 절차. `align_point_correction.py` 의 rcp 로딩 경로와 연결. golden 수집으로 S<3 recipe 커버 확대.
3. **(후속) 2차 proposer** — in_topk 천장 0.594 를 올릴 때(contour reranker 검증 후). contour/AKAZE/MI-coarse
   후보 생성기.

### 다음 오피스 실행
```bash
uv run python poc/workflow_2/align_similarity.py   # contour rerank 컬럼 추가 후 → rerank_rank1_lift_contour 확인
```

> ❓ **확인 필요(사용자):** contour reranker 전처리(이진화 방식)와 점수식(matchShapes vs Hu+게이트)을
> 다음 세션에서 같이 정할지, 아니면 내가 기본안(Otsu + matchShapes 단독, chamfer 게이트 유지)으로
> 먼저 구현해서 A/B 수치를 보고 조정할지?

---

## 7. 메모리 업데이트

- `project_matcher_flat_chamfer_distinctiveness.md` — test1 섹션 추가 완료(MI 폐기·WHY·contour 전환).
- `workflow_2_procedure.md` §6.1 신설.
- `.remember/now.md` 07:53 항목.
- `MEMORY.md` 인덱스 — 기존 `project_matcher_flat_chamfer_distinctiveness` 줄 그대로 유효(파일 내용만 갱신),
  추가 줄 불필요.
