# consensus 재등록 검증 성공 + productization 핸드오프

날짜: 2026-06-08 16:33
대상: `poc/workflow_2/golden_consensus_eval_cond.py` 및 consensus 재등록 레버

---

## 0. 한 줄 요약

**consensus 재등록(최근 S median 템플릿)이 PROPOSER_WALL 을 깼다.** 실데이터(cond) A/B:
in_topk **0.434 → 0.876 (+0.442)**, rank1 **0.318 → 0.764 (+0.446)** (recipes=134, S_loo=403, min_s=3).
판정: **CONSENSUS 채택 권장.** 다음 작업은 이 검증된 consensus 를 matcher/live search 의 실제
등록 template 로 승격(productization)하는 것.

---

## 1. 진행 사항 (이번 세션)

오전 저널(13:34, offset 분리 + matcher plan) 이후 consensus 평가를 끝까지 몰아 검증 완료까지 갔다.

1. **msr modality 추론 도입** — msr cond.txt 에는 `Scope` 가 없음을 사용자가 확인. 키/배율로
   modality 를 가르도록 `_msr_modality()` 구현(OM=`!OM_Brightness`/Mag<200, SEM=
   `Accelerating_voltage`/Mag>500). `_resolve_mod` = `_msr_modality(cond) or recipe_mod`.
2. **msr 경로의 죽은 Scope 코드 제거** — `_scope_label`/`_modality_of`/`scope_counts`/
   `scope_distribution` 삭제(msr 엔 Scope 없어 항상 None=dead). code-review `[]`(clean) 확인.
3. **recipe collector 정상 확인** — 초기 `recipes=1` 은 데이터가 덜 받아진 상태였고, 다 받은 뒤
   glob/rglob/`_collect_recipes` 모두 298 일치. collector 버그 아님(probe 로 확정).
4. **`recipes=1, S_loo=6` 붕괴 원인 규명** — `mod_total`=784 인데 A/B 가 recipes=1 로 붕괴.
   probe(`probe_recipe_s_counts.py`)로 **sparse 데이터**가 주원인임을 확정: 298 recipe 중
   dominant-modality S 가 ≥4 인 건 단 1개, 135개가 정확히 3장, 151개가 0장(fail-only). 부차적으로
   `recipe_id`=leaf 만이라 dict 키 충돌(298→276 고유)도 데이터 유실 유발.
5. **min_s=3 + 고유 키 적용** — `AB_MIN_S=4` 면 평가 불가 → `CONSENSUS_MIN_S`(env, 기본 3,
   바닥 3으로 clamp)로 `_consensus_template_ab(min_s=)` 호출. `_recipe_key=eqp/class/recipe`로
   by_recipe 충돌 제거. → recipes=134, S_loo=403 확보.
6. **code-review 반영** — min_s<3 무의미(LOO 바닥 fm≥3) → `_floor_min_s` clamp + 경고; 중복
   `res["min_s"]` 제거(callee 가 이미 반환). 28 tests pass.
7. **오피스 A/B 실행 → 검증 성공** — 위 0번 수치. baseline rcp 0.434 가 localization eval
   gt_in_topk 0.433 과 독립 재현 → 신뢰. 옛 천장 0.594 대비 +0.282(과거 검증 lever 와 동일).
   높은 rank1(0.764)이 "blurry generic median 아님(진짜 변별력)"을 입증.
8. **productization 통합점 매핑** — Explore 로 live_align_search/align_key_matcher/align_similarity/
   golden_consensus_eval_cond/align_fail_assets/procedure 문서를 훑어 "어디를 바꾸면 consensus 를
   rcp 대신 쓰는가"를 정리(아래 4절).

## 2. 수정 내용 (커밋)

main 에 직접 commit/push (Mac→push→office pull). 이번 세션 관련 커밋:

- `8d6dc69` consensus: infer msr modality from keys/Magnification (msr cond has NO Scope)
- `619fabe` consensus: drop dead Scope code from msr path
- `d6c23cf` probe_recipe_s_counts.py — diagnose recipes=1 (collision vs sparse)
- `18dd547` consensus: min_s=3 (configurable) + unique by_recipe key
- (code-review) consensus: clamp CONSENSUS_MIN_S to floor 3 + drop redundant res key

변경 파일: `poc/workflow_2/golden_consensus_eval_cond.py`,
`poc/workflow_2/test_golden_consensus_eval_cond.py`(28 pass),
진단 스크립트 `probe_golden_depth.py`(삭제됨), `probe_recipe_s_counts.py`(존속), `debug_cond_paths.py`.

## 3. 핵심 발견 (메모리에도 기록)

- **이 golden set 은 S 가 희박** — recipe 당 평균 ~2.6장, ≥4 단 1개·135개가 정확히 3장. consensus
  레버는 min_s=3(S=3 recipe 는 2장으로 빌드, 약하지만 rank1 로 변별력 입증) 에서만 측정 가능.
  LOO 바닥은 `len(others)>=2` → fm≥3 필요(min_s=2 는 무의미). → [[project_consensus_sparse_golden_and_recipe_id_collision]]
- **`AlignFailAssets.recipe_id` 는 recipe_name(leaf)만** — dict 키 소비자에서 충돌(유실). consensus 는
  `eqp/class/recipe` 로 국소 고유화. localization 이 "잘 되고" consensus 만 붕괴한 이유 = list vs dict 집계차.
- **전략 결론** — PROPOSER_WALL 의 갈림길(rerank vs proposer)에서 **proposer 재등록이 답**. rerank(MI/
  contour)는 이미 폐기. stale 단일 rcp key 가 병목, 최근 S median 으로 current appearance 추종이 해법.

---

## 4. 다음 단계 — productization (내일 시작) ⭐

**목표**: 검증된 consensus(최근 S median) 를 matcher/`live_align_search` 의 *실제 등록 template* 로
승격. recipe 의 S 가 ≥3 이면 consensus, 아니면 rcp center 로 폴백.

### 4-A. 통합점 (Explore 결과 요약)

| # | 무엇 | 파일 | 현재(rcp) | 바꿀 것(consensus) |
|---|---|---|---|---|
| 1 | template 빌드(프로덕션, LOO 아님) | **신규 util** | — | `_build_cond_by_recipe(assets, center_tpls)` 로 전체 S crop → `_consensus(crops)` → `build_template(version="s_consensus_prod", key_type=mod)` |
| 2 | consensus 재료 crop | `golden_consensus_eval_cond.py` L205-262 | (eval: LOO) | `_build_cond_by_recipe` 의 `s_frames[].crop` 그대로 재사용(co-reg 포함). **LOO 없이** 전부 median |
| 3 | median 빌더 | `align_similarity.py` L593 `_consensus(crops)` | — | 순수함수, 그대로 사용 |
| 4 | template→matcher | `align_key_matcher.py` `build_template`/`compute_align_key_score` | rcp template | **변경 불필요** — consensus template 도 동일 필드(raw/edge/dt) |
| 5 | template 라우팅 | `live_align_search.py` L298 `route_template(templates, mode)` | rcp dict | dict 구성 시 `consensus or rcp` 우선순위 |
| 6 | 호출부 dict | `live_align_search()` L196,227 호출자(step4/`align_fail_correct.py`/test) | `{"OM":rcp,"SEM":rcp}` | `{"OM": cons_om or rcp_om, "SEM": cons_sem or rcp_sem}` |
| 7 | blur 가드 | `align_similarity.py` L809-821 | eval 기록 | 프로덕션도 edge_ratio<0.70 or lap_ratio<0.50 면 consensus 버리고 rcp 폴백 |

### 4-B. 권장 설계 결정 (내일 확정)

1. **신규 함수** (예: `consensus_template.py` 의 `build_consensus_template(assets, modality) -> AlignKeyTemplate | None`)
   - 입력: `AlignFailAssets`(또는 recipe_dir) + modality. center_tpls 로 sizing.
   - 내부: `_build_cond_by_recipe` 로 S crop 수집 → modality 별 그룹 → `_consensus` → `build_template`.
   - **min_s 가드**: 같은 modality crop < 3 이면 `None`(폴백 신호).
   - **blur 가드**: edge/lap ratio 낮으면 `None`.
2. **소스**: 프로덕션 consensus 는 **golden S 프레임**(검증된 known-good)에서 — `align_success_dataset_plan.md`
   참조. (실시간 from_msr 의 S 는 라벨 신뢰 못 함 — [[feedback_doubt_s_labels.md]] 유의.)
3. **저장 위치**: 권장 = **in-memory 계산 + DEBUG_IMAGE_DIR 감사 저장**(eval 처럼). 사용자 recipe 폴더
   (`align_img_from_rcp`)에는 쓰지 말 것(rcp 와 혼동/race). 영구화가 필요하면 별도 `consensus_templates_s/`
   서브폴더로 — 단 사용자(다운로더)와 약속 먼저.
4. **버전 태그**: `version="s_consensus_prod"` 로 로그/overlay 에서 rcp 와 구분.

### 4-C. TDD 순서 (제안)

1. `build_consensus_template`: (a) S≥3 → AlignKeyTemplate 반환, (b) S<3 → None, (c) blur 낮음 → None,
   (d) modality 분리(om/sem 따로) 테스트. 합성 crop 으로 RED→GREEN.
2. `route_template` 또는 dict 구성부: consensus 우선·rcp 폴백 테스트.
3. 통합: `live_align_search` 가 consensus template 으로도 동작(기존 mock 으로).

### 4-D. 선택/저우선

- `CONSENSUS_MIN_S=4` 1회(강한 consensus, recipe 1개) 방향 일치 확인 — 낮은 우선순위.
- **blur 가드 수치 확인 미완** — 오피스 A/B 출력의 `edge_ratio_to_S / lap_ratio_to_S` 한 줄을 아직 못 봄.
  내일 첫 확인: ≥~0.7 이면 min_s=3 caveat 완전 해소(rank1 0.764 가 이미 강하게 시사).
- **정리**: `probe_recipe_s_counts.py`, `debug_cond_paths.py` 는 소임 완료 → 삭제 후보(사용자 확인 후).

---

## 5. 메모리 업데이트

- 신규: `project_consensus_sparse_golden_and_recipe_id_collision.md` — sparse golden(min_s=3) +
  recipe_id leaf 충돌 + **검증 결과(in_topk 0.876, rank1 0.764)** 기록. MEMORY.md 인덱스에 추가.
- 기존 `project_align_cond_files_and_coords.md` 에 msr-Scope-없음 규칙 이미 반영.

## 6. 재현/실행 메모

```bash
# 오피스 A/B 재실행
uv run python poc/workflow_2/golden_consensus_eval_cond.py
#   env: CONSENSUS_MIN_S(기본3, 바닥3), CONSENSUS_COREGISTER(기본1), ALIGN_GOLDEN_ROOT
# 테스트 (Mac, cv2 필요 → --extra dev)
uv run --extra dev pytest poc/workflow_2/test_golden_consensus_eval_cond.py -q   # 28 pass
```
