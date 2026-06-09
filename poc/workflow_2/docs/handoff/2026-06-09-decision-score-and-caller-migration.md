# 핸드오프 — decision/score 정비 + 호출자 전환 (NCC reranker 후속)

> 2026-06-09 작성. /compact 후 이 문서 기준으로 재개. 선행 작업은 모두 main 에 push 완료.

## 0. 지금까지 (한 문단)

ensemble proposer(3채널 RRF, pool recall@8 0.698)를 `compute_align_key_score_ensemble`에 통합했고,
candidate **selection 을 ORB → NCC reranker** 로 교체해 최종 정확도를 전환했다. e2e 검증
(`localization_ab_ensemble.py`, production 함수 호출, n≈757): **ensemble hit 0.422(baseline) → 0.608**
(+18.6pp, McNemar p≪0.0001, gain163:regress22). code-review + codex 통과. 진단 체인:
`localization_regression_diag`(회귀 3분류) → `reranker_signal_probe`(NCC 11/11 분리) →
`reranker_rule_ab`(전체 0.607). 메모리: `project_ensemble_proposer_and_consensus_race`.

핵심 코드: `align_key_matcher.py`
- `compute_align_key_score_ensemble`: selection = `rerank_chamfer_w·chamfer + rerank_ncc_w·max(0,ncc)`
  (`MatchPolicy.rerank_chamfer_w/ncc_w` 기본 0.5/0.5), best 1개에만 ORB(결정 score 용).
- NCC 프리미티브: `_ncc`, `_resize_template`, `_frame_patch`, `_candidate_ncc`.
- 결정 score/decision 은 여전히 `_finalize_match` 가 `chamfer_weight·chamfer + orb_weight·orb` 로 계산.

## 1. 블로커 — decision/score 가 NCC selection 과 어긋남

NCC 가 **저-chamfer 정답 후보**를 고르면(=chamfer_miss 케이스, 이 작업의 존재 이유) 그 후보의
chamfer 가 낮아 `score = chamfer_weight·chamfer + orb_weight·orb` 가 낮게 나오고 →
`decision="low"` 가 된다. **best_xy(좌표)는 맞는데 decision/score 는 비관적.**

→ 호출자들이 `decision`/`score` 를 **하드 게이트**로 쓰므로, 그대로 전환하면 방금 회복한
163개 정답이 게이트에서 다시 버려질 수 있다. (현재 `compute_align_key_score_ensemble` docstring 에
이 주의가 명시돼 있음.)

## 2. 호출자 인벤토리 (무엇을 게이트로 쓰나)

**전환 대상 = fallback/static-compare 경로** (live broad-scan 은 `compute_align_key_score` 유지):

| 파일 | 함수/위치 | decision/score 사용 (게이트) | 전환? |
|---|---|---|---|
| `compare_align_images.py` | :174, :200 | `_verdict_for(result.decision, result.score)` → 정적 비교 verdict | ✅ 대상 |
| `align_fail_correct.py` | :121–124 | `if decision=="match"` / `if decision=="adjust": orb>0` = key 발견 판정 | ✅ 대상 |
| `align_point_correction.py` | :1115, :1148–1149, :1221+ | free vs prior `score` 비교, `adjust_threshold` 게이트, distinctive/reject_reason | ✅ 대상(복잡) |
| `match_recipe_key_on_crop.py` | :133, :141 | `if decision=="match"` | ✅ 대상 |
| `vlm_align_key_region.py` | :376, :355–363 | distinctive/reject_reason/score_gap/score 기록·판단 | ⚠️ 검토 |
| `live_align_search.py` | :230, :240, :254, :520 | `score>=candidate_score`, `decision=="match"`, `score>=match_threshold` | ❌ 유지(live) |
| `search_align_key.py` | :163, :189, :199 | `decision=="match"`/`"adjust"` (live FOV 루프) | ❌ 유지(live) |

핵심: `align_fail_correct`·`compare_align_images`·`match_recipe_key_on_crop` 는 `decision` 문자열에
직접 의존, `align_point_correction` 은 `score` 절대값·`adjust_threshold`·distinctive 에 의존(가장 까다로움).

## 3. decision/score 정비 — 설계안 (post-compact brainstorm 에서 확정)

문제: selection(chamfer+NCC)과 score(chamfer+ORB)가 다른 신호 → 둘을 일치시켜야.

- **안 A (권장): score = selection score.** 결정 score 를 NCC selection 과 동일하게
  `rerank_chamfer_w·chamfer + rerank_ncc_w·max(0,ncc)` 로 계산. "best 를 고른 신뢰도"가 그대로
  decision 신뢰도가 됨(일관). **단 match/adjust threshold(현 0.62/0.40, chamfer+ORB 기준)를
  새 score 분포에 맞춰 재캘리브 필수** → §4.
- **안 B: ncc 를 AlignKeyMatchResult 에 노출 + 호출자가 판단.** 호출자 수정 많아짐, 비추.
- **안 C: chamfer+ORB+NCC 3-way score.** 가중 3개 + 재캘리브, 복잡.

설계 결정 포인트(brainstorm 에서):
1. ensemble 경로만 score 식을 바꿀지(=`_finalize_match` 에 분기/파라미터) vs `compute_align_key_score`
   도 영향받지 않게 격리. (원칙: 기존 함수 무변경 유지 — ensemble 전용 score 식.)
2. ORB 를 score 에서 완전히 뺄지(비용·orb_flip 무관) vs 약하게 남길지.
3. threshold 를 MatchPolicy 필드로 별도 둘지(ensemble_match_threshold 등).

## 4. threshold 재캘리브 (안 A 의 필수 하위작업)

새 score(=selection score)의 분포에서 hit/miss 를 가르는 match/adjust threshold 를 **데이터로** 정한다.
- `reranker_rule_ab.py` 를 확장(또는 신규 calibration 러너)해 전체 S 프레임의 `sel` 점수 + GT hit
  라벨(err<=tol)을 덤프 → ROC/Youden 또는 목표 precision 으로 threshold 산출.
- 이미 `reranker_rule_ab` 의 rows.jsonl 에 per-frame hit 이 있음 — sel 점수만 추가 덤프하면 됨.
- 산출된 threshold 를 ensemble 결정에 적용 후 e2e 재확인(`localization_ab_ensemble` 의 decision 분포가
  hit 와 정합하는지: decision=="match" 인데 miss / decision=="low" 인데 hit 비율).

## 5. 작업 순서 (재개 시)

1. **brainstorm** — §3 안 A 확정 + §3 설계 포인트 3개 결정 (skill: brainstorming).
2. **calibration 러너** — §4. sel 점수 + hit 덤프 → threshold 산출(오피스 1회 실행, 숫자 회신).
3. **score 정비 구현** — ensemble 전용 score 식 + threshold 필드, 기존 `compute_align_key_score`
   무변경(회귀 가드 10/10). TDD. (writing-plans → subagent-driven, code-review + codex.)
4. **호출자 전환** — 한 번에 하나씩: `match_recipe_key_on_crop`(가장 단순) → `compare_align_images`
   → `align_fail_correct` → `align_point_correction`(마지막, 복잡). 각 전환 후 동작 확인.
5. **e2e 재확인** — `localization_ab_ensemble` 로 hit 유지(≈0.608) + decision 게이트가 정답을 안
   버리는지.

## 6. 열린 질문 (brainstorm 에서 사용자에게)

- ensemble score 식에서 ORB 완전 제거 vs 잔존? (orb_flip 은 selection 문제였으니 score 엔 무해하나
  비용·일관성 측면.)
- threshold 산출 기준: 목표 precision(정답만 match) vs recall(놓치지 않기) 중 무엇 우선?
- 호출자 전환을 opt-in 플래그(env/policy)로 점진 롤아웃할지, 직접 교체할지.
- `vlm_align_key_region` 을 전환 대상에 포함할지.

## 7. 제약 (불변)

CLI 인자 금지 / Korean docstring / `[INFO]·[ERROR]·[WARNING]` print / `from __future__` 금지 /
절대 임포트 / main 직접 commit·push(Mac→office pull) / commit trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` / fab 이미지 Mac 반입 금지
(blind write + 오피스 run + 텍스트 digest) / 설계규칙: CV 가 좌표·점수 결정, VLM 은 영역/타당성만.
