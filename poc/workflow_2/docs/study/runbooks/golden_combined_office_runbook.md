# Combined Routed Eval (OM/SEM split) — 오피스 실행 런북

> 대상 스크립트: `poc/workflow_2/golden_combined_eval_cond.py`
> 목적: production 라우팅(consensus 우선 · rcp 폴백)을 그대로 적용한 **routed 정확도** +
> **OM/SEM modality-split 증거**(실패유형 히스토그램 · per-mod Youden · split verdict)를 한 번에 낸다.
> 이 1회 실행의 verdict 가 다음 단계(Phase 2 L1 / Phase 3 L2·L3 lever)를 **게이트**한다.
> Mac 에서는 golden 이 없으면 `no_data` 로 깨끗이 빠진다 — 실판정은 오피스 실데이터로만.

> 선행 문서: 설계 `docs/superpowers/specs/2026-06-19-om-sem-modality-split-eval-design.md`,
> 플랜 `docs/superpowers/plans/2026-06-19-om-sem-modality-split-eval.md`,
> 데이터/수집 규약은 자매 런북 `golden_localization_office_runbook.md` §1 과 동일.

---

## 0. 한눈에 (TL;DR)

```
1) 설정: golden_eval_config.py 의 GOLDEN_ROOT / HISTORY_ROOT / MIN_S 확인
         (없으면 golden_eval_config.example.py 복사 후 SPLIT_* 블록 포함해 편집)
2) 실행: uv run python poc/workflow_2/golden_combined_eval_cond.py
3) 회신: 콘솔 맨 끝 [DIGEST] 한 줄 + (Step 2) SPLIT 판정 블록을 텍스트로 복사해 전달
         (전체 콘솔 대신 digest.txt 한 줄만 줘도 됨)
```

경로/임계는 전부 `golden_eval_config.py`(gitignored 실편집본)에서 읽는다. CLI 인자 없음.

---

## 1. 실행 전 설정 확인

`poc/workflow_2/golden_eval_config.py`(없으면 `golden_eval_config.example.py` 복사)에서:

| 상수 | 의미 | 비고 |
|---|---|---|
| `GOLDEN_ROOT` | align_images eval 루트(`<eqp>/<class>/<recipe>/...`) | rcp + msr S 트리 |
| `HISTORY_ROOT` | consensus 풀 루트(`<class>/<recipe>/events/<id>/S*.jpeg`) | eqp-독립, S 만 |
| `MIN_S` | consensus 최소 S 장수(바닥 3) | 미달 recipe 는 rcp-only arm 으로 |
| `SPLIT_MIN_FRAMES` / `SPLIT_MIN_RECIPES` | split 판정 게이트(표본 충분?) | 미달 modality → `insufficient` |
| `SPLIT_RANK1_GAP` / `SPLIT_RANK1_FLOOR` | OM/SEM rank1 격차 기준 | |
| `SPLIT_DOMINANCE` | 지배 실패유형 share 기준 | |

> consensus 증거를 보려면 `HISTORY_ROOT` 에 class·recipe·modality별 최근 S(권장 8~10장)가
> 있어야 한다. 비어 있으면 from_msr LOO 폴백으로만 도므로 high-S bin 이 얇아진다.

---

## 2. 실행

```bash
uv run python poc/workflow_2/golden_combined_eval_cond.py
```

레버(rcp-only arm) testbed 평가를 같이 보려면:

```bash
ALIGN_ENSEMBLE_LAB_MODE=edge_ncc uv run python poc/workflow_2/golden_combined_eval_cond.py
```

### 실행 직후 확인 — 실데이터를 읽고 있나?
- `[WARNING] golden 데이터를 찾지 못했습니다` 가 뜨면 데이터를 못 읽은 것(→ `GOLDEN_ROOT` 확인).
- 정상이면 `[INFO] (combined routed) recipe N개 → ...  [rcp matcher=...]` + min_s/clean_frame 가 찍힌다.

---

## 3. 회신할 내용 (fab-safe, 텍스트만)

**최소**: 콘솔 맨 끝 `[DIGEST]` 한 줄(또는 `digest.txt`). 이 한 줄에 3축 + modality + verdict + 증거가 다 들어있다.

```
lab=.. minS=.. consMode=hist:N/loo:M | routed r1/topk=../.. (n=..) |
cons r1/topk=../.. lift=.. (n=..,rec=..) | rcp_only r1/topk=../.. (n=..,rec=..) |
mod[OM r1/topk=../.. (n=..) SEM r1/topk=../.. (n=..)] | scaling[S=3:.. S=4:.. ..] |
verdict=<...> | evid[OM:la/pm/rm=../../.. yd=thr/J  SEM:la/pm/rm=../../.. yd=thr/J]
```

**권장 추가**: stdout 의 `=== (Step 2) 실패유형 분해` 표 + `=== (Step 2) SPLIT 판정` 블록도 함께.
산출물은 오피스 PC 에도 저장된다:
`poc/workflow_2/debug_images/golden_combined_eval_cond/<timestamp>/{summary.json, digest.txt}`.

---

## 4. verdict 가 다음 단계를 어떻게 게이트하나

`split_verdict.verdict` 값으로 갈린다:

| verdict | 의미 | 다음 행동 |
|---|---|---|
| `SPLIT` | OM/SEM rank1 격차 **있고** 지배 실패유형이 **서로 다름** | `suggested_levers` 가 가리키는 lever 만 Phase 3 로 진행 |
| `shared_tune` | 격차는 있으나 두 modality 실패유형이 같음 | modality 분리 말고 공용 reranker 튜닝 |
| `no_split` | 격차 없음 | 현행 단일 CV 정책 유지 |
| `insufficient` | 표본 게이트 미달(`insufficient_mods`) | 해당 modality 데이터 더 모아 재실행 |

- **Phase 2 (L1, per-mod Youden)** 은 verdict 와 무관하게 진행 가능(orthogonal). `evid[..yd=thr/J..]` +
  `accP`/`accM` 가 per-mod 임계 분리 근거. **단, 이건 분류(feasibility) 축이지 localization 이 아님.**
- **Phase 3 (L2/L3)** 은 `SPLIT` 일 때만, 그리고 `suggested_levers` 가 named lever 일 때만 채운다.
  - `L2_om_periodicity` → OM 이 periodic_look_alike 지배(주기 lattice 닮은꼴에 1등 뺏김).
  - `L3_sem_recall` → SEM 이 recall_miss 지배(진실이 후보 pool 에 아예 없음).
  - `om:<bucket>` / `sem:<bucket>` 같은 **중립 라벨**이 나오면 = 설계가 예상한 (modality, 실패유형)
    조합이 아니라는 뜻 → named lever 단정 말고, 그 패턴이 무슨 CV 개입을 부르는지 먼저 의논.

---

## 5. 읽을 때 주의 (이번 수정 반영, 2026-06-19)

리뷰 finding 1~3 수정으로 modality 증거가 교차오염 없이 나온다 — 해석 시 다음을 신뢰해도 된다:

- **periodic_look_alike 는 modality별로 정확하다.** dual OM/SEM recipe 에서 OM 주기성이
  SEM 실패로 새지 않는다(periodicity 를 `(recipe, modality)` 로 분리). 즉 SEM 의 periodic 지배는
  진짜 SEM 템플릿 주기성이다.
- **Youden `d_prod`/`accP` 의 비교 기준은 actuation 게이트(adjust 0.4727)** 다 — production align-point
  보정의 실제 절대 score 컷. `accM` 은 balanced(match 0.6053) 게이트 참고치(둘 다 리포트).
- **lever 라벨은 (modality, 실패유형) 쌍으로 매핑**된다. SPLIT 인데 라벨이 중립('{mod}:{bucket}')이면
  관측 패턴이 설계와 반대라는 신호 — §4 마지막 항목대로 처리.

> 데이터/수집(흰 box, crosshair, S 장수) 체크리스트는 `golden_localization_office_runbook.md` §1 과 동일.
> 오피스에서 실패하면 원인은 코드가 아니라 데이터로 좁힌다(Mac self-test/단위테스트 통과 = 코드 무결).
