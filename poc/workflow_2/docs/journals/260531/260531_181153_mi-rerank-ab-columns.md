# 260531 — MI reranker A/B 측정 컬럼 추가 (재개용 핸드오프)

> 세션: 2026-05-31
> 주제: consensus A/B 핸드오프의 "(A) MI reranker 1순위"를 착수 — Codex 설계 consult 재수령 후
> `align_similarity.py` A/B 하네스에 MI-rerank 측정 컬럼을 추가(진단→production 승격 전 검증 단계).
> 관련: [consensus A/B verdict 저널](../260529/260529_152818_consensus-ab-verdict-and-next-steps.md),
> [MI reranker intro(신규)](../../study/cv/mi_reranker_intro.md),
> [평가 지표 intro](../../study/cv/align_evaluation_metrics_intro.md)

---

## 1. 이번 구간에 한 일

### 1.1 Codex 설계 consult 재수령 (중단됐던 것)
이전 핸드오프에서 "reranker 설계 Codex consult 가 백그라운드 중단" 상태였다. `codex:setup` 확인
(ready, ChatGPT 로그인 활성) 후 4개 설계 질문을 다시 던져 verdict 를 받았다:

- **Q1 MI 단독 vs MI+contour →** *MI 단독 hard reorder 로 시작.* contour scorer 는 어느 파일에도
  미구현 + 지금 넣으면 구조 이중계산으로 "MI 가 정말 회복하나"를 가림.
- **Q2 scale ±1 refine →** *discrete `cand.scale` 그대로.* 순서만 정하면 되고 모든 후보가 같은
  scale grid 공유. scale 이 병목이면 reranker 아닌 proposer 문제(truth-forced `scale_gain` 이 진단).
  per-candidate refine 은 비용 `top_n×band`.
- **Q3 순환성 →** *LOO 구조 건전.* held-out 은 consensus 에서 제외되므로 crop-to-self 아님.
  남는 편향(MI 가 consensus-like 선호)은 곧 검증대상 production 동작 → 진단용 rcp 병렬 컬럼 유지.
  ⚠️ held-out `xhair_crop` 자체를 MI reference 로 쓰면 진짜 순환(금지).
- **Q4 production 배선 →** *`compute_align_key_score` 내부 opt-in.* 호출부 after-reorder 는
  `best_xy`/distinctiveness 필드(`candidates[0/1]` 파생)와 어긋나 coherence 버그. DEFAULT_POLICY 는
  chamfer 경로 유지(smoke test 보장). reorder 후 `second_score`/`score_gap` 재정의, ORB 도 MI-best 재실행.

### 1.2 구현 — `align_similarity.py` A/B 측정 컬럼 (진단만, production 미변경)
핵심 성질: **rerank 는 후보 *집합*(멤버십)을 안 바꾸고 *순서*만 바꾼다.** → `in_topk` 불변, `rank` 만 이동.
그래서 `rerank_rank1_lift = (MI rank1) − (chamfer rank1)` 한 숫자가 confounder 없이 reranker 효과를 잰다.

| 함수 | 변경 |
|---|---|
| `_gt_in_topk` | 각 chamfer 후보를 `_matched_crop`→`_mi` 로 채점해 내림차순 재정렬, 정답 순위 재계산. 5-tuple race(chamfer 기준 winner 의 reranked rank 보고). 반환에 `topk_rank_reranked` 추가 |
| `_consensus_template_ab` | LOO 루프에 `cons_r1_rr`/`rcp_r1_rr` 카운트. 신규: per-recipe `cons_rank1_rate_reranked`·`rerank_rank1_lift`, overall `overall_cons_rank1_reranked_rate`·`overall_rcp_rank1_reranked_rate`·`rerank_rank1_lift` |
| `_summarize` | gt_topk 에 `n_rank1_reranked`·`rank1_reranked_rate` |
| `_print_summary` | GT-IN-TOPK·A/B 블록에 MI rerank 줄 + 판정 규칙(`rerank_lift≥+0.10 → 승격`) |
| `_self_test` | `_gt_in_topk` 경로 + **멤버십 불변 단언**(`rerank None == chamfer None`) |

### 1.3 study 문서
- 신규 `docs/study/cv/mi_reranker_intro.md` — "재정렬이 실제로 뭘 하나"(멤버십 불변·순서만, 코드,
  scale 처리, 순환성, `rerank_rank1_lift` 읽는 법, production 승격 주의). 기존 metrics intro 에서 링크.

### 1.4 검증
- Mac/Windows self-test 통과: `gt_topk: in_topk=True rank=1 rerank=1` (멤버십 불변 확인).
- 이 머신엔 align_images 실데이터 0 recipes → 합성 fallback 으로 경로만 검증.
- matcher(`align_key_matcher.py`)·`DEFAULT_POLICY` 미변경 → 합성 smoke test 보장 불변.

---

## 2. 커밋 상태
- **아직 커밋 안 함.** working tree:
  - `M poc/workflow_2/align_similarity.py` (+~67 lines)
  - `M poc/workflow_2/docs/study/cv/align_evaluation_metrics_intro.md` (링크 3줄)
  - `?? poc/workflow_2/docs/study/cv/mi_reranker_intro.md` (신규)
  - `M .claude/settings.local.json` (세션 권한, 무관)
- 제안 커밋 메시지: `align_similarity: add MI-rerank rank1 columns to gt-in-topK + consensus A/B`

---

## 3. 다음 단계 (재개 시 — 우선순위 순)

### (A) 오피스에서 A/B 재실행 → `rerank_rank1_lift` 회신 (다음 1순위)
```bash
# ⚠️ cp949 콘솔 → em-dash UnicodeEncodeError. utf-8 강제 또는 파일 리다이렉트 필수.
PYTHONIOENCODING=utf-8 uv run python poc/workflow_2/align_similarity.py
```
- 볼 것: A/B 블록 `+MI rerank rank1 : ... rerank_lift +0.???`, GT-IN-TOPK 의 `MI rerank rank1`.
- **판정:** `rerank_lift ≥ +0.10` → MI reranker production 승격 정당화(`topk_not_rank1=0.179` 회복).
  `≈0` → MI 로 부족 → contour 등 다른 reranker / proposer 강화 검토.
- 결과는 `console_results/260531_*.txt` 로 저장(기존 컨벤션). 텍스트 회신받아 수치 확정.

### (B) 검증되면 production 승격 (Q4 verdict 따라)
- `compute_align_key_score` 내부에 MI reorder opt-in(STRUCTURE_POLICY/플래그). DEFAULT_POLICY 불변.
- reorder 후: `second_score`/`score_gap`/`reject_reason` 재계산, 원 chamfer rank 보존, **ORB 를 MI-best 에 재실행**.
- `align_point_correction.py` 가 쓰는 `.candidates` 가 MI 순으로 나오도록 → green mark 가 MI-best 채택.

### (C) 잔여 (이전 핸드오프에서 계속)
- **rcp 재등록 도구**(다음 주 6/01~): 저장된 `<out_dir>/consensus/<recipe>__<mod>.png` 를 새 rcp 로.
  golden 데이터셋으로 판단불가 recipe 커버리지 확대 + staleness 임계 calibration.
- **2차 proposer**(잔여 ~28% miss = recall 천장 0.718 밖): contour/AKAZE/MI-coarse 후보 생성기.
- **crosshair v2 production swap**: E false-positive(71/198) montage 육안 확인 후.
- **코드 리뷰 #10 보류**: 공유 `frame_dt` threading(`_gt_in_topk`/`_truth_forced`/race) 리팩터.

---

## 4. 열린 질문 / 주의
- (B) production 승격 시 **순환성 재확인**: 진단 A/B 는 LOO 라 비순환이지만, production 은
  단일 rcp(또는 재등록된 consensus)로 매번 매칭하므로 LOO 가 없다. "consensus 를 새 rcp 로 등록한 뒤
  그 rcp 로 MI rerank" = 등록 시점 이후엔 held-out 개념이 없어 순환 아님(rcp 는 고정 reference). OK.
- MI 비용: 후보 8개 × 프레임당 crop+히스토그램. 진단에선 무시 가능, production 실시간 경로에선
  프로파일 필요(특히 live_align_search).
