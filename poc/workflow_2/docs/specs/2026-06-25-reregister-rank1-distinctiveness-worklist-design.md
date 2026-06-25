# Re-registration Phase 3 — Rank-1 Distinctiveness Worklist — Design Spec

- **Date:** 2026-06-25
- **Status:** Design (approved, pre-implementation)
- **Scope:** Offline CV bench in `poc/workflow_2` only. No `workflow_3`/production changes. Consumes an existing artifact; **does not** modify the consensus eval or add a new matching pass.
- **Related:** [reregister Phase 1 spec](2026-06-23-reregister-report-design.md) (the report this extends), [ADR 0006](../study/adr/0006-template-bank-matcher-rejected-fusion-exhausted.md) (template-bank rejection → rank-1 is the distinctiveness signal), Phase 2 (E-frame confirmation) shelved.

## 1. Context & motivation

Three matcher-fusion methods (median-consensus, soft-voting heatmap, RRF) all hit the same wall: on SEM they recover the true align point into the candidate set (in_topk ~0.92–0.94) but rank it #1 only ~0.5 of the time — a periodic distractor is a structurally equal template match (ADR 0006). The conclusion: **SEM is a ranking/distinctiveness problem, and the lever is the align key, not the matcher.** Re-registration onto a more distinctive region is the validated next move.

Phase 1 (`golden_reregister_report_cond.py`) already ranks recipes by re-registration need via S-only evidence tiers (STRONG/MEDIUM/ADVISORY) and suggests replacement whiteboxes, but its tier floors are calibration-fragile (office run #1: STRONG_FRAC_FLOOR saturated at SEM 95% / OM 52%). **Rank-1 on a recipe's own success frames is a cleaner, calibration-light distinctiveness signal** — if the matcher can't rank the true point #1 even on a clean success frame, the key is ambiguous by construction — and it is *already measured per recipe* by the consensus eval.

## 2. Goal

Join the per-recipe rank-1 already written by `golden_consensus_eval_cond.py` into the re-registration report, and emit an **engineer-facing worklist** that (a) names specific recipes, (b) ranks them worst-first, and (c) tells the engineer **what kind of fix** each needs (fresh snapshot vs new region), corroborated by the existing tier evidence and whitebox suggestion.

## 3. Data source (no new computation)

`golden_consensus_eval_cond.py` writes `summary.json` with a `per_recipe` list; each row (align_similarity.py ~line 1082) already contains:

```
{ "recipe": <rec>, "modality": "om"|"sem", "n_S_loo": n,
  "rcp_rank1_rate":  <0..1>,   # the REGISTERED rcp key's rank-1 on success frames (LOO/history)
  "cons_rank1_rate": <0..1>,   # the REGION's rank-1 (median of success crops) on success frames
  "rcp_in_topk_rate", "cons_in_topk_rate", "cons_pool_n", "mode", ... }
```

- `rcp_rank1_rate` = distinctiveness of the *registered snapshot* → low = re-register.
- `cons_rank1_rate` = distinctiveness of the *region itself* (best achievable from current crops) → low = the region is ambiguous, a fresh snapshot from the same place won't help.

This Phase reads that file; it must **not** modify the consensus eval.

## 4. Architecture — a consumer step in the reregister driver

`golden_reregister_report_cond.py` gains a join + classify + worklist step inside (or just before) the existing report assembly in `run()` (~line 1023, where `rows_by_mod` is built and `_format_report`/`_format_digest` write to `OUTPUT_ROOT`). The existing tier screening stays unchanged and becomes corroborating evidence. New pure helpers (testable on Mac) + one I/O reader.

### 4.1 Input resolution — `_load_consensus_rank1(path) -> dict`
Resolve the consensus `summary.json`:
1. `REREGISTER_CONSENSUS_SUMMARY` env (explicit path) if set; else
2. the most recent `debug_images/golden_consensus_eval_cond/<ts>/summary.json` (newest timestamp dir).

Return `{(recipe, modality): {"rcp_rank1": float, "cons_rank1": float, "n_S_loo": int, "cons_pool_n": int}}`. If no file is found, return `{}` (every recipe becomes `NO_DATA`; the report still runs — graceful degrade).

### 4.2 Fix-type classifier — `_classify_fix(rcp_rank1, cons_rank1, *, distinct_floor) -> str`
Pure, the core new logic:

| rcp_rank1 | cons_rank1 | verdict | action |
|-----------|-----------|---------|--------|
| ≥ floor | (any) | `OK` | distinctive — no action |
| < floor | ≥ floor | `FRESH_SNAPSHOT` | region fine, snapshot stale → re-register from the same region |
| < floor | < floor | `NEW_REGION` | region ambiguous → move the key; whitebox suggestion is the payload |
| `None` (no join) | `None` | `NO_DATA` | not in consensus run → rank by existing tier evidence only, marked |

`None` inputs (recipe absent from the consensus join) → `NO_DATA`. Both rates present → table above.

### 4.3 Priority — `_worklist_priority(fix_type, rcp_rank1, tier_weight) -> float`
Pure. Primary severity = `(1 - rcp_rank1)`. Multiply by a fix-type weight so the harder/higher-value fixes float up: `NEW_REGION` > `FRESH_SNAPSHOT` > `OK`. Add a small corroboration term from the existing `TIER_WEIGHT[tier]` (so a rank-1 flag that the tiers *also* flag ranks above a lone rank-1 flag). `NO_DATA` rows priced by tier weight alone and sorted below any rank-1-backed flag of equal tier. Sort worst-first.

### 4.4 Worklist output — `_format_worklist(worklist_rows) -> str` + JSON
- `reregister_worklist.txt` (human, aligned columns) and `reregister_worklist.json` (machine), both under the existing `OUTPUT_ROOT`, alongside `reregister_report.txt`/`digest.txt`.
- Columns: `rank · recipe · modality · rcp_rank1 · cons_rank1 · fix_type · suggested_whitebox · tier · priority`.
- `suggested_whitebox` reuses Phase 1's existing `_search_unique_box`/`_select_candidate` output; it is the **payload for `NEW_REGION` rows** (FRESH_SNAPSHOT rows don't need a new box).
- A one-line `[DIGEST]` extending the existing one: per-modality counts of `NEW_REGION` / `FRESH_SNAPSHOT` / `OK` / `NO_DATA` + the join-coverage number (see §5). ASCII-only (cp949).

## 5. Join-key risk (the one real implementation hazard)

The reregister driver keys recipes as `f"{a.class_name}/{a.recipe_name}"` (line 918). The consensus `per_recipe["recipe"]` (= `rec` from `_iter_recipe_modalities`) must use the **same** format for the join to land — historically there were `class/recipe` vs `eqp/class/recipe` collision/uniquification differences between drivers. **Implementation MUST, as task 1, verify the exact `rec` format the consensus driver emits and pin the join key to match** (normalize on one side if needed). Emit a loud **join-coverage line** — `[INFO] rank1-join: matched M/N report recipes to consensus rows` — so a silent key mismatch (M≈0) is impossible to miss. A near-zero match rate is a hard failure signal, not a quiet `NO_DATA` flood.

## 6. Scope, coverage, calibration

- **Both modalities.** SEM dominates the worst-first list; low-rank-1 OM keys still surface.
- **Coverage gap is explicit:** recipes the report screens but consensus didn't cover → `NO_DATA`, ranked by existing tiers, visibly marked (engineers know which rows are rank-1-backed vs tier-only).
- **One calibration lever:** `REREGISTER_DISTINCT_FLOOR` env (default `0.70`, office-tunable; follows the existing floor-knob pattern in this driver). The `OK`/`FRESH`/`NEW` split is the only threshold. The fix-type *split* between FRESH and NEW could use a separate `cons` floor later, but v1 uses the single `distinct_floor` for both (YAGNI).

## 7. Testing

- Pure helpers (`_classify_fix`, `_worklist_priority`, `_format_worklist`, and the join-merge given an in-memory summary dict) unit-tested on Mac with synthetic rows — every `_classify_fix` cell, priority ordering (NEW_REGION above FRESH above OK at equal rcp_rank1), `NO_DATA` fallback, ASCII one-line digest. New tests in `poc/workflow_2/test_reregister_report.py` (existing test file).
- `_load_consensus_rank1` tested with a tmp summary.json fixture (path resolution + empty-on-missing).
- Accuracy (which recipes actually flag) is office-gated — Mac does py_compile + the pure-logic tests + a no-data run.

## 8. Constraints (binding)

- Korean docstrings/comments; `[INFO]`/`[WARNING]` print logging (never `logging`); **ASCII-only inside `print()`/digest strings** (cp949, no em-dash).
- No `argparse`, no `from __future__`; absolute imports; workflow_2 imports workflow_3, never reverse.
- Config via env + `golden_eval_config` (edit only the tracked `golden_eval_config.example.py` + loader; never the gitignored `golden_eval_config.py`).
- Commit directly to `main` with a pathspec of exactly the touched files (no `git add -A`).

## 9. Out of scope (v1)

- Re-running or modifying the consensus eval. (If per-recipe rank-1 is ever needed for recipes the consensus set excludes, that's a future Phase — extend the consensus eval's eligibility, not this consumer.)
- A separate `cons` floor distinct from `rcp` floor (deferred; single `distinct_floor` for v1).
- Any production / workflow_3 live-detection (a different, larger thread — see the brainstorming options).
- Acting on the worklist (the engineer re-registers; this Phase only produces the prioritized, diagnosed list).

---

## 10. NEXT SESSION — start here

**One-line state:** Template-bank matcher concluded REJECTED (ADR 0006); rank-1 is the distinctiveness signal; this spec extends the Phase-1 reregister report to consume per-recipe rank-1 from the consensus eval's `summary.json` into a diagnosed, prioritized worklist. Design is approved. Implementation not started.

**To begin:** invoke `superpowers:writing-plans` against this spec to produce a TDD implementation plan, then execute it (subagent-driven-development). Suggested task ordering:

1. **Pin the join key (de-risk first).** Read what `_iter_recipe_modalities` / the consensus `per_recipe["recipe"]` actually emits vs the reregister driver's `f"{class}/{recipe}"`. Pin the join, add the `[INFO] rank1-join: matched M/N` coverage line. (Mac: read-only verification; the format is in the code.)
2. `_load_consensus_rank1(path)` — path resolution (env → newest `debug_images/golden_consensus_eval_cond/<ts>/summary.json`) + parse `per_recipe` → `{(recipe,mod): {...}}`; empty-on-missing. Test with a tmp summary.json fixture.
3. `_classify_fix` — the §4.2 table, pure, every cell tested.
4. `_worklist_priority` — §4.3, pure, ordering tested.
5. `_format_worklist` (txt + json) + the extended `[DIGEST]` — ASCII, tested.
6. Wire into `run()` (~line 1023): join → classify → prioritize → write `reregister_worklist.{txt,json}` to `OUTPUT_ROOT` next to the existing report; print the coverage line + extended digest. Guard the no-data path. `REREGISTER_DISTINCT_FLOOR` env (default 0.70) + bridge knob in `golden_eval_config.example.py` + loader.
7. Mac verify: pure-logic tests + `uv run python -c "import poc.workflow_2.golden_reregister_report_cond"` + a no-data run.

**Office run (accuracy, after implementation):**
```
# prerequisite: consensus summary.json must exist (run the consensus eval first if stale)
uv run python poc/workflow_2/golden_consensus_eval_cond.py        # writes per_recipe rank-1
uv run python poc/workflow_2/golden_reregister_report_cond.py      # consumes it -> worklist
# relay: [INFO] rank1-join coverage line, the [DIGEST] (NEW_REGION/FRESH/OK/NO_DATA counts),
#        and the top ~10 worklist rows (recipe, modality, rcp/cons rank1, fix_type)
```
**Decision after the office run:** confirm the join coverage is high (M/N near 1), then read the worklist — the `NEW_REGION` SEM rows are the highest-value re-registration targets (region itself ambiguous). Calibrate `REREGISTER_DISTINCT_FLOOR` if the OK/FRESH/NEW split looks wrong for the observed rank-1 distribution (SEM rank-1 clustered ~0.5 per ADR 0006, so 0.70 should flag most SEM keys — tune if it flags too many/few).
