# 260601 — MI reranker production 선제구현(opt-in 플래그) + A office 실행 메모

> 세션: 2026-06-01
> 주제: 핸드오프 (B) "production 승격" 을 **opt-in 플래그로 선제구현**(A 결과 전, production 무변경).
> (A) office A/B 재실행은 이 머신에 실데이터 0 → office 에서 실행. 아래 실행 메모 참조.
> 관련: [MI rerank A/B 컬럼 저널](../260531/260531_181153_mi-rerank-ab-columns.md),
> [consensus A/B verdict](../260529/260529_152818_consensus-ab-verdict-and-next-steps.md),
> [MI reranker intro](../study/cv/mi_reranker_intro.md)

---

## 1. 이번 구간에 한 일 — (B) MI reranker production 경로 (opt-in, TDD)

핵심 성질: **chamfer = 멤버십 게이트(후보 집합 불변), MI = 순서만.** office A/B(진단)에서 쓰는
`align_similarity._mi`/`_matched_crop` 을 production matcher 에 **정확히 mirror**(수치 일치 보장).

### 변경 (`align_key_matcher.py`)
| 항목 | 내용 |
|---|---|
| `MatchPolicy.mi_rerank: bool = False` | opt-in 플래그. **DEFAULT_POLICY·STRUCTURE_POLICY 모두 False → 기존 동작·smoke test 불변** |
| `AlignKeyCandidate.chamfer_rank: int\|None` | MI 재정렬 후에도 원 chamfer 순위 보존(Q4 "원 chamfer rank 보존") |
| `_mutual_information` / `_match_crop_for_mi` | 진단 하네스와 동일한 MI·crop(footprint=raw×scale→resize→MI). **변경 시 양쪽 동기화** 주석 명시 |
| `rerank_candidates_by_mi(template, frame_gray, candidates)` | 멤버십 불변, MI 내림차순 새 list 반환, `mi_score` in-place |
| `compute_align_key_score` | distinctiveness 를 **chamfer 집합 기준으로 분리**(reorder 와 독립 → 음수 gap 방지), reorder 를 best 선택 *직전* roi-local 좌표에서 수행, best=MI-best → ORB 재실행·score·decision·best_xy 파생 |

### 설계 결정 (리뷰 포인트)
- **distinctiveness 일관성:** Q4 의 "reorder 후 second_score/score_gap 재계산" 을, *chamfer 공간*
  재계산으로 해석하면 MI 가 낮은-chamfer 후보를 1등으로 올렸을 때 `score_gap<0` 이 되어 그 케이스를
  `not_distinctive` 로 잘못 reject(= MI 가 고치려는 18% 를 도로 버림). → **distinctiveness 는 chamfer
  *집합* 모호성(가장 강한 peak vs 2nd)으로 정의**해 reorder 와 독립시켰다. best 선택만 MI 가 바꾼다.
  두 신호(어디를 고를까=MI / 집합이 모호한가=chamfer)는 직교라고 봄. **재논의 여지 있음.**
- align_similarity 의 `_mi`/`_matched_crop` 과 **중복**: A(office 실행)가 임박해 진단 하네스를 지금
  건드리지 않으려고 의도적으로 mirror(공유 리팩터는 후속). sync 주석으로 표시.

### 검증 (Mac/Windows, TDD)
- 신규 `test_align_key_mi_rerank.py` (RED→GREEN): dense-edge distractor(chamfer 포화≈1.0) vs
  degraded 진짜(저대비+blur, MI 높음) 합성 → off=distractor / **on=진짜** 로 best_xy 이동,
  멤버십 불변, `score_gap≥0`(chamfer 기준), `chamfer_rank` 보존. **PASS.**
- 회귀: `test_align_key_match` **10/10**, `test_align_key_distinctiveness` PASS,
  `align_similarity` self-test 통과(`gt_topk in_topk=True rank=1 rerank=1`). → DEFAULT 경로 byte-identical.

### 아직 안 한 것 (A 확정 후)
- **활성화(flip):** `align_point_correction.py` 가 매칭에 `STRUCTURE_POLICY` 사용(라인 743·748·795).
  A 가 `rerank_rank1_lift≥+0.10` 확정하면 **`STRUCTURE_POLICY` 에 `mi_rerank=True` 한 줄**로 production
  green mark 가 MI-best 채택. (지금은 플래그 False 라 무변경.)
- MI 실시간 비용 프로파일(후보 8개 × crop+히스토그램) — live_align_search 경로 활성화 시.

---

## 2. (A) office 실행 메모 — `rerank_rank1_lift` 회신용

이 머신엔 `poc/workflow_1/align_images/` 실데이터 0 → **office 에서 실행**해야 수치가 나온다.

```bash
# ⚠️ cp949 콘솔 → em-dash UnicodeEncodeError. PYTHONIOENCODING=utf-8 필수.
cd <repo>
PYTHONIOENCODING=utf-8 uv run python poc/workflow_2/align_similarity.py > poc/workflow_2/console_results/260601_mi_rerank.txt 2>&1
```

**볼 것 (출력 텍스트):**
- A/B 블록: `+MI rerank rank1 : ... rerank_lift +0.???`
- GT-IN-TOPK 블록: `MI rerank rank1`

**판정 규칙:**
- `rerank_lift ≥ +0.10` → MI reranker production 승격 정당화(`topk_not_rank1=0.179` 회복) →
  §1 "활성화(flip)" 수행: `STRUCTURE_POLICY.mi_rerank=True`.
- `≈0` → MI 로 부족 → contour 등 다른 reranker / 2차 proposer 검토.

결과 텍스트를 회신하면 수치 확정 후 flip 여부를 결정한다.

---

## 3. 커밋 상태
- 변경: `M align_key_matcher.py`(+~75), `?? test_align_key_mi_rerank.py`(신규), `?? 이 저널`.
- matcher 의 `DEFAULT_POLICY`/`STRUCTURE_POLICY` 동작 불변 → 기존 호출부 영향 없음(플래그 off).
- 제안 메시지: `matcher: add opt-in MI reranker (mi_rerank flag) behind DEFAULT/STRUCTURE = off`
