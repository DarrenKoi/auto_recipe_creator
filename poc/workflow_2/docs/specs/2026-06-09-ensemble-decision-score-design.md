# ensemble decision/score 정비 — 설계 (안 A)

> 2026-06-09. 선행: NCC reranker production 배선 완료(`compute_align_key_score_ensemble`,
> hit 0.422→0.608). 핸드오프 `docs/handoff/2026-06-09-decision-score-and-caller-migration.md` §3 안 A 확정.
> brainstorm 에서 설계 포인트 4개 결정(아래 §0).

## 0. 결정된 설계 포인트 (brainstorm)

1. **ORB 완전 제거** — ensemble 결정 score 식에서 ORB 삭제. selection 과 동일 신호
   `rerank_chamfer_w·chamfer + rerank_ncc_w·max(0,ncc)` 로 일원화. best 1개 ORB 계산도 폐지(비용↓).
2. **threshold 기준 = Youden J** — 새 score 분포에서 ROC (TPR−FPR) 최대점을 match_threshold 로.
   adjust_threshold 는 고-recall 점(hit 대부분 포착)으로.
3. **호출자 직접 교체** — env opt-in 플래그 없이 `compute_align_key_score` → `compute_align_key_score_ensemble`
   직접 교체(Mac→office pull 워크플로우라 git rollback 가능, No-CLI-arg 컨벤션 부합).
4. **vlm_align_key_region 이번 제외** — 4개 핵심 호출자 전환·검증 후 별도 판단.

## 1. 문제

`compute_align_key_score_ensemble` 의 **selection** 은 chamfer+NCC(검증된 0.608)인데,
**결정 score/decision** 은 여전히 공유 `_finalize_match` 의 `chamfer_weight·chamfer + orb_weight·orb`.
NCC 가 *저-chamfer 정답*(chamfer_miss 케이스 = 이 작업의 존재 이유)을 고르면 그 후보의 chamfer 가
낮아 `score` 가 낮고 → `decision="low"`. **best_xy(좌표)는 맞는데 decision/score 는 비관적.**
호출자들이 decision/score 를 하드 게이트로 쓰므로, 전환 시 방금 회복한 163개 정답이 게이트에서 다시
버려질 수 있다.

→ **결정 score 를 selection score 와 일치시키고, 새 분포에 맞춰 threshold 재캘리브.**

## 2. 아키텍처 — `_finalize_match` 파라미터화

`_finalize_match`(align_key_matcher.py:742)는 `compute_align_key_score`(baseline, 무변경 가드 10/10)와
`compute_align_key_score_ensemble` 가 **공유**한다. 내부 score 식을 무조건 바꾸면 baseline 이 깨진다.
→ 옵션 파라미터 2개를 추가하되 **default 는 기존 동작과 바이트 동일**:

```python
def _finalize_match(
    best_cand, candidates, frame, template, policy, roi_origin,
    *, chamfer_score, orb_ratio,
    score_override: float | None = None,             # 신규
    decision_thresholds: tuple[float, float] | None = None,  # 신규 (match, adjust)
) -> AlignKeyMatchResult:
    ...
    if score_override is not None:
        score = float(score_override)                # ensemble: 선택 루프의 best_sel
    else:
        score = policy.chamfer_weight * chamfer_score + policy.orb_weight * orb_ratio  # baseline 불변
    if decision_thresholds is not None:
        mt, at = decision_thresholds
    else:
        mt, at = policy.match_threshold, policy.adjust_threshold
    decision = _decision_for_score(score, match_threshold=mt, adjust_threshold=at)
    ...
```

- **baseline**: 두 인자 모두 미전달 → 기존 코드 경로 그대로. (회귀 가드 보존)
- **ensemble**: `score_override=best_sel`(선택 루프에서 이미 계산 — 재계산 0),
  `decision_thresholds=(policy.ensemble_match_threshold, policy.ensemble_adjust_threshold)`.
- overlay·result 모두 동일한 score 를 사용(일관). score 식 분기가 한 군데로 모임.

### 2.1 `_decision_for_score` 시그니처

현재 `_decision_for_score(score, policy=DEFAULT_POLICY)` → policy 의 match/adjust 를 읽음.
threshold 를 직접 받도록 keyword 화:

```python
def _decision_for_score(score, policy=DEFAULT_POLICY, *,
                        match_threshold=None, adjust_threshold=None) -> str:
    mt = match_threshold if match_threshold is not None else policy.match_threshold
    at = adjust_threshold if adjust_threshold is not None else policy.adjust_threshold
    if score >= mt: return "match"
    if score >= at: return "adjust"
    return "low"
```

호출처 2곳(`_finalize_match`, 기존 직접 호출 있으면)만 영향 — 안전.

## 3. ORB 제거 (ensemble 경로)

`compute_align_key_score_ensemble` 의 best ORB 계산 블록(현 :884-890)을 **삭제**:

```python
# 삭제 대상 (orb=0 고정으로 대체):
#   best_orb = 0.0
#   if best_cand.chamfer_score > 0.0 and btw > 0 and bth > 0:
#       crop, _ = _crop_with_padding(...)
#       best_orb, _, _ = compute_orb_inlier_ratio(...)
```

→ `_finalize_match(..., orb_ratio=0.0, score_override=best_sel, decision_thresholds=(...))`.
프레임당 `compute_orb_inlier_ratio` 1회 절약. ensemble result 의 `orb_inlier_ratio = 0.0` 고정.

⚠️ **파급**: `align_fail_correct.py:121-124` 가 `if decision=="adjust": return orb>0` 로 orb 를 2차
게이트로 씀 → ensemble 에선 orb=0 이라 항상 False. **그 호출자 전환(§5 Step 4)에서 orb 의존 제거**,
decision 만으로 판정하도록 함께 수정. (이미 인벤토리 대상)

## 4. 새 MatchPolicy 필드 + threshold 캘리브

```python
@dataclass
class MatchPolicy:
    ...
    ensemble_match_threshold: float = 0.62     # 캘리브 후 확정(placeholder)
    ensemble_adjust_threshold: float = 0.40    # 캘리브 후 확정(placeholder)
```

DEFAULT_POLICY/STRUCTURE_POLICY 모두 상속(필요 시 STRUCTURE_POLICY 에서 override).

### 4.1 캘리브 절차 (Youden J) — 구현보다 **선행**

threshold 확정값을 알아야 §2 구현을 마칠 수 있으므로 캘리브 데이터 덤프가 먼저다.

1. `reranker_rule_ab.py` 확장(또는 신규 `ensemble_threshold_calib.py`) — 전체 S 프레임마다
   NCC-selected best 의 `sel` 점수 + GT hit(err≤GT_TOL_NORM) 라벨을 `calib.jsonl` 로 덤프.
   (pool+NCC 는 이미 계산 중 — sel·hit 만 추가.)
2. **오피스 1회 실행** → jsonl 을 Mac 으로 **텍스트(숫자)** 회신. (이미지 반입 금지)
3. Mac 에서 ROC → **Youden J 점 = ensemble_match_threshold**, 고-recall 점(예: hit 95% 포착)
   = ensemble_adjust_threshold 산출. 두 값을 policy 에 하드코딩.
4. §2 구현 + threshold 확정 후 `localization_ab_ensemble` 재확인:
   `decision=="match"` 인데 miss 비율 / `decision=="low"` 인데 hit 비율이 낮은지(게이트 정합성).

## 5. 작업 순서

1. **calib 덤프 러너** (§4.1 step 1) — 오피스 실행, 숫자 회신.  *(threshold 산출)*
2. **score 정비 구현** (§2·§3·§4) — TDD. `compute_align_key_score` 무변경 가드 10/10.
   confirmed threshold 반영. code-review + codex.
3. **호출자 전환** (직접 교체, 한 번에 하나씩 + 동작 확인):
   `match_recipe_key_on_crop`(단순) → `compare_align_images` → `align_fail_correct`(orb 게이트 제거)
   → `align_point_correction`(복잡).
4. **e2e 재확인** — `localization_ab_ensemble` 로 hit 유지(≈0.608) + 게이트 정합성.

## 6. 테스트 전략

- **회귀 가드**: `uv run python poc/workflow_2/test_align_key_match.py` 10/10 (baseline 무변경).
- **신규 단위**: `_finalize_match` score_override 경로(주어지면 그 값, 없으면 chamfer+ORB),
  `_decision_for_score` threshold 오버라이드, ensemble result.score == best_sel,
  ensemble result.orb_inlier_ratio == 0.0.
- **기존**: `test_align_key_score_ensemble.py` selection 테스트 유지(score 식 변경에 맞춰 갱신).
- Mac 합성 데이터로 전부 검증 가능(골든 데이터 불요). 캘리브·e2e 만 오피스.

## 7. 비범위

- weight sweep(현 0.5/0.5; 0.608). threshold 확정 후 별도.
- `vlm_align_key_region` 전환.
- consensus 출력 정리(parked).

## 8. 제약 (불변)

CLI 인자 금지 / Korean docstring / `[INFO]·[ERROR]·[WARNING]` print / `from __future__` 금지 /
절대 임포트 / main 직접 commit·push / commit trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` / fab 이미지 Mac 반입 금지
(blind write + 오피스 run + 텍스트 digest) / CV 가 좌표·점수 결정, VLM 은 영역/타당성만 /
NCC 는 reranker 로만(primary matcher 금지).
