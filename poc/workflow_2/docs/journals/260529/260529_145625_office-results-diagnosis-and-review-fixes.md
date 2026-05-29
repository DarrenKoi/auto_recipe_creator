# 260529 — 오피스 진단 결과 해석 → 재등록 레버 확정 → consensus A/B + 코드 리뷰 수정

> 세션: 2026-05-29 (오후, 컨텍스트 압축 이후)
> 주제: test1~test3 실데이터 수치 해석으로 병목 확정 → 진단 도구 보강(gt_in_topk·confounder·교차표·consensus A/B) → xhigh 코드 리뷰 14/15 수정
> 관련: [matcher 신뢰도 저널](260529_130919_matcher-reliability-and-diagnostics.md), [recovery plan](260529_align-point-correction-recovery-plan.md), [지표 학습 문서](../../study/cv/align_evaluation_metrics_intro.md), [golden dataset plan](../../align_success_dataset_plan.md)

---

## 1. 진행 사항

오피스에서 `align_similarity.py` 를 3회 돌려 텍스트로 회신받은 수치(`console_results/260529_test1~3.txt`)를 단계별로 해석하며, 매 단계 Codex(`/codex:rescue`)와 상의해 다음 측정을 추가하는 방식으로 진행했다.

### 1.1 실데이터로 확정한 병목 (test1 → test3)
- **test1**: chamfer 점수면이 평평. `free_best med_S 0.66 ≈ med_E 0.70`(bACC 0.55, 우연 수준), `truth-forced wrong_local_peak 153/229(67%)`, `scale_gain 0.025`(scale 무관), `edge_problem 5`. → 병목은 scale·inpaint·엣지부재 아니고 **matcher 변별력**. 유일 변별 레버 = `MI(0.627)`·`box-crop(0.61)`.
- **test2**: `gt_in_topk = 0.46`(miss 54%, rank1 0.26) → 정답이 chamfer top-8 후보에 절반은 아예 없음 = **proposer(후보 생성)도 병목**, MI 리랭킹 천장 46%. 참조 staleness `ok=4/71`(confounder 의심).
- **test3**(confounder 수정 후): `ok 4→9`, `low_texture=0`(저텍스처 거짓 stale 은 없었고 crop jitter 가 진짜 문제였음), 판정가능 recipe의 **65%(17/26) stale**. `point_biserial(in_topk, ratio) r=0.415(n=175)` → 참조 품질이 proposer recall 을 유의미하게 예측(분산 17%) = **참조가 주요 동인이지만 proposer 도 독립적으로 약함**.
- **결론**: **rcp 재등록이 최우선 레버** → 2차 proposer → MI 리랭커(마지막). `COMPARE_SCALES += 0.5` 는 폐기.

### 1.2 진단 도구 보강 (각 단계 Codex 합의)
1. **gt_in_topk**(proposer recall) — 정답이 chamfer top-N 후보에 드는지 → "MI 리랭킹 가능 vs proposer 교체" 갈림길 측정.
2. **staleness confounder 2건 수정** — (a) S-consensus crop 을 matcher 위치가 아닌 **crosshair 검출 위치**(scale 1.0)에서 추출(jitter 제거), (b) `MIN_CONSENSUS_SELF_MI` 저텍스처 게이트.
3. **원인 분리 교차표** `_gt_topk_reference_crosstab` — status/ratio tertile 별 in_topk recall + **point-biserial r**(ok 4개뿐이라 status 버킷 underpowered → tertile·상관으로 보강).
4. **consensus 템플릿 A/B**(`_consensus_template_ab`, LOO) — 기존 S 데이터만으로 "rcp→S-consensus 교체 시 in_topk 가 뛰나" 검증. 비순환(S crop=crosshair 위치, matcher 무관). E false-positive 가드(cons_E≥cons_S 면 generic 경고). **재등록 검증이자 프로토타입**.

### 1.3 Codex 기여
- 한국어 **지표 학습 문서** `docs/study/cv/align_evaluation_metrics_intro.md`(445줄) — 진단 콘솔 지표를 "원인 분리 도구" 관점에서 설명.

### 1.4 xhigh 코드 리뷰 (9 finder × 검증) → 15 findings → 14 수정
silently-wrong 지표 버그 위주. **Tier 1(#1~#4)은 consensus A/B 수치를 무효화**할 수 있어 office 재실행 전 필수 수정으로 분류 → 사용자 승인(전부) 후 적용.

---

## 2. 수정 내용 (`poc/workflow_2/align_similarity.py`)

### 추가(Updated/Added)
- `_gt_in_topk`(proposer recall), `_gt_topk_reference_crosstab`·`_ratio_tertile`·`_point_biserial_in_topk_vs_ratio`·`_valid_ratio_in_topk_pairs`(원인 분리), `_consensus_template_ab`(재등록 검증 A/B).
- 상수: `TOPK_CANDIDATES=STRUCTURE_POLICY.top_n`(literal→정책 파생), `GT_TOL_NORM=0.20`(독립), `AT_CROSSHAIR_ROI_FACTOR=1.2`, `MIN_CONSENSUS_SELF_MI`, `AB_MIN_S=MIN_S_FOR_CONSENSUS+1`, `AB_E_SAMPLE`.
- `summary` 키: `gt_topk`, `gt_topk_by_reference`, `consensus_ab`. 검증된 consensus 를 `<out_dir>/consensus/` 에 PNG 저장(재등록 도구가 같은 산출물 사용).
- staleness: crop 을 **crosshair-direct**(scale 1.0)로 변경, `S_inconsistent`→`low_texture`→`stale`→`ok` 순서로 판정.

### 제거(Removed) — 코드 리뷰 결과
- **`rcp_in_topk_by_name`**(basename 키 join) 완전 제거 → recipe 간 파일명 충돌로 가짜 lift 유발하던 #1. rcp baseline 을 A/B 함수 안에서 **per-recipe `rcp_tpls`**로 재계산(같은 modality·frame·`_gt_in_topk` = apples-to-apples, #2 동시 해결).
- **`s_crops` 중복 저장** 제거 → `s_frames` 단일 원천, staleness 가 modality 로 group(#12).
- `_gt_in_topk` 의 **modality 교차 pool+truncate** 제거 → **race**(modality별 best rank)로 교체. 잡음 modality 가 정답 후보를 밀어내 recall 과소평가하던 #4.
- LOO 의 **중복 `_score`**(held-out 당 2회 매칭) 제거 → 가드가 frame 당 `_score` 1회 재사용(#9).
- E 가드 루프 안의 **consensus+template 반복 빌드** 제거 → 1회 hoist(#8).
- **손수 짠 Pearson** 제거 → `np.corrcoef`(#15). `Counter` 루프내 import → 모듈 top(#11).

### 보류(Deferred)
- **#10**(공유 `frame_dt` 를 `_gt_in_topk`/`_truth_forced`/race 에 thread) — `align_key_matcher` 공개 API 변경이 필요해 생산 matcher 위험 → 별도 작업.

### 커밋(main 직접)
`b889aec`(gt_in_topk) → `0983e3a`(confounder+교차표) → `4b10bc4`(consensus A/B) → `24ed91d`(Codex 지표 문서) → `1991071`(코드 리뷰 14/15 수정). + `console_results/260529_test1~3.txt`.

### 검증 (Mac, blind)
self-test 통과. `_gt_in_topk` race=정답 modality rank1. A/B rcp baseline per-recipe 정확(rcp=align→1.0, rcp없음→0.0, 충돌 없음). 가드 cons_E<cons_S. consensus 저장 확인. corrcoef/tertile/zero-variance 가드 정상. **실데이터 full 실행은 오피스.**

---

## 3. 다음 단계

### 오피스 재실행 (코드 리뷰 수정 반영본 — A/B 수치 신뢰 가능)
```bash
uv run python poc/workflow_2/align_similarity.py
```
새 **`CONSENSUS A/B`** 섹션 회신:
- `in_topk_rate: rcp=.. consensus=.. lift=±..`
- `consensus free_best chamfer median: S=.. E=..`
- per-recipe lift 상위

**판정 기준:**
| 결과 | 의미 | 다음 작업 |
|---|---|---|
| lift 크게 + & cons_E<cons_S | 재등록이 in_topk 끌어올림(레버 확정) | **rcp 재등록**(저장된 consensus=새 rcp). golden 앞당김 |
| lift ≈ 0 | rcp 바꿔도 안 뜸 | **2차 proposer**(contour/AKAZE/MI-coarse) |
| cons_E ≈ cons_S | generic 블러 | crop **co-registration**(ECC/phase-corr) 후 재측정 |

### 후속
- crosshair v2 → production swap(오피스 E false-positive montage 눈으로 확인 후).
- **golden(항상 성공) 데이터셋 수집 — 다음 주(2026-06-01 주)**: 판단불가 45개(대부분 S<3) 해소 + staleness 임계 실측 calibration.
- #10(공유 frame_dt) 별도 리팩터로.

---

## 4. 메모리 업데이트

- `project_matcher_flat_chamfer_distinctiveness.md` 갱신(test2/3 수치: gt_in_topk 0.46, ok 9/stale 17, r=0.415, 재등록 우선) + `MEMORY.md` 인덱스 1줄 추가. 그 외 코드 구조는 이 저널과 CLAUDE.md 가 기록하므로 추가 불필요.
