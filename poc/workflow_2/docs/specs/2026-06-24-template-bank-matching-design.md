# Template-Bank Matching — Design Spec (bench experiment)

> **CONCLUDED 2026-06-25 — REJECTED (no port). See [ADR 0006](../study/adr/0006-template-bank-matcher-rejected-fusion-exhausted.md).**
> Kill-test passed (near_periodic om 0.014 / sem 0.052) and in_topk beat consensus per-modality (SEM heatmap +9.4pp, OM rrf +3.8pp), **but rank-1 — the production deliverable — was ~0.5 on both modalities** (heatmap om 0.531 / sem 0.492). SEM is a ranking/distinctiveness problem unrankable by any member-fusion (median/heatmap/RRF all hit the same ~0.5 rank-1 wall). Lever is align-key distinctiveness (re-registration), not the matcher. Code kept as a bench arm behind `TBANK_HEATMAP=0`.

- **Date:** 2026-06-24
- **Status:** CONCLUDED / REJECTED (was: Design, pre-implementation)
- **Scope:** Offline CV bench experiment in `poc/workflow_2` only. No `workflow_3`/production changes.
- **Adversarial review:** Codex rescue pass (2026-06-24) — its findings are folded in as hard
  requirements and Risks below. This spec is deliberately structured to **falsify the idea cheaply**
  before investing in a full eval arm.

---

## 1. Problem & motivation

The recipe-registered (`rcp`) align key is often non-distinctive; the production matcher
(chamfer+ORB+NCC ensemble, C1/C2/C3 proposers → RRF → NCC rerank, `poc/workflow_3/align/matching/engine.py`)
localizes weakly (best-candidate proposer scores ~0.2–0.3 even on **success** frames). SEM proposer-recall
(`gt_in_topk`) ~68% is the known bottleneck; the dominant failure is **periodic distractors inside the key
region** (ROI narrowing is structurally useless). Pixel-level NCC/SSIM is banned as a *primary* signal
(process variation).

A **median-consensus** mechanism already exists (`consensus_cv._consensus`): it stacks N aligned recent-success
(S) crops, takes `np.median` → **one fused template**, then matches it. It needs a blur guard
(`edge_ratio < 0.70`) because the median washes out distinctive structure.

## 2. Hypothesis — AND its explicit counter-hypothesis

**Hypothesis (H1):** Keeping the N S-crops **individual** (sharp) and fusing their **dense match responses by
cross-member agreement** localizes the align point better than the median (which blurs distinctive structure
and has no defence when the live frame resembles one specific past appearance).

**Primary fusion = soft-voting heatmap (dense), not RRF (discrete).** Rationale (decided 2026-06-24):
- Production *already* does discrete-candidate → RRF → rerank and is stuck at ~0.2–0.3 / 68% SEM recall; an
  RRF-bank is the same fusion family on a new input and likely inherits that ceiling. Heatmap is a genuinely
  different mechanism.
- **Recall recovery (the key argument):** RRF can only fuse candidates that reached each member's **top-K**.
  In the 0.2–0.3 regime the true point is often in **no** member's top-K — that *is* the `gt_in_topk` ~68%
  bottleneck — so RRF never sees it. Dense accumulation sums the true point's weak-but-consistent response
  across members **even when no single member ranked it**, so a weak-everywhere peak can still rise. This
  attacks `gt_not_in_topk` directly.
- Fewer free parameters (no clustering tolerance, no RRF `k`), so less room to manufacture lift (Codex #2).

RRF-bank is retained as the **extra arm** ("does discrete candidate fusion add anything over dense
accumulation?").

**Counter-hypothesis (H0, from Codex review — the thing we must rule out FIRST):** If the periodic
distractors are *consistent across S frames* (likely on a stable-process recipe), then **every** sharp member
nominates the **same** wrong lattice point, and cross-member agreement **reinforces the distractor instead of
suppressing it** — making the bank *worse* than the median, whose blur partially suppresses periodic detail.

H1 is a property of the data, not a guarantee. The experiment's first job is to decide H1 vs H0 cheaply.

## 3. Scope

- **Bench-only**, in `poc/workflow_2`. Experimental logic is a bit-parity fork (like `ensemble_lab.py`)
  that **imports** workflow_3 primitives and **never edits** workflow_3.
- Compared against two baselines already in the consensus driver: **rcp-only** and **median-consensus**.
- Primary metrics: `gt_in_topk` (proposer recall) **and** rank-1, **OM/SEM stratified**.
- The `workflow_3/align` port is a **separate, later spec, gated** on a positive + attributed result.

## 4. Architecture (units, each independently testable)

All new code lives in `poc/workflow_2`:

- **`template_bank_lab.py`** (new) — the experimental matcher fork:
  - `bank_build(crops, ...) -> list[AlignKeyTemplate]` — the N S-crops as **individual** templates
    (cond-aware crop, coregistered), i.e. `consensus_cv` source/coreg plumbing **without** the `np.median`
    collapse. Respects the same `min_s` gating.
  - `bank_match_heatmap(bank, frame, ...) -> BankResult` — the **PRIMARY** arm (soft-voting): for each member,
    compute its **dense response map** (the chamfer score map the proposer already computes) and accumulate by
    **SUM** into a common frame coordinate space across scales and members; take the global peak (peak-NMS for
    top-K). SUM = agreement semantics (a point many members weakly support rises); the only knob is the
    peak-NMS radius. Recovers true peaks that per-member top-K extraction drops (§2). Same H0 failure mode as
    any agreement method (consistent distractors add constructively) — guarded by the §7 kill-test.
  - `bank_match_rrf(bank, frame, ...) -> BankResult` — the **extra** arm ("does discrete fusion add over dense
    accumulation?"): each member runs the existing per-member proposer → top-K candidates; **collapse
    within-member** (one vote per member per cluster); **pool across members → RRF-fuse with spatial
    clustering** → **NCC-rerank** fused top candidates → winner `(xy, score)` + per-cluster member-support set.
    If this does not beat the heatmap primary, the discrete RRF machinery is not earning its complexity.
  - `BankResult` carries: winner `xy`/`score`, the fused top-K (each with member-support count), and the
    per-member raw candidates (for diagnostics).
- **bank arm in `golden_consensus_eval_cond.py`** — adds a 3rd/4th arm (RRF-bank + heatmap-control) beside
  rcp-only and median-consensus, reusing the driver's pool selection / LOO / GT / frame-clean / stratification
  / digest. Emits the kill-test report (§7) and the full A/B (§8).

The per-member proposer call MUST be **bit-parity** with the engine's existing proposer (same discipline as
Phase 2's `_free_search_best_score`), so all arms are comparable.

## 5. Data flow & no-leakage discipline

Reuse the consensus eval's **history-first + leave-one-out** discipline verbatim (it governs *which* frames
seed the bank, independent of *how* they fuse):

1. Source S frames: `HISTORY_ROOT` has ≥ `min_s` for class/recipe/modality → build from that **disjoint** pool,
   eval on the recipe's `from_msr` S (no overlap, no LOO). Else → **LOO** within `from_msr` (eval frame
   excluded from its own bank).
2. `bank_build` → `list[AlignKeyTemplate]` (no median).
3. Per eval frame (cond-cleaned, identical to consensus eval): `bank_match_rrf` and `bank_match_heatmap`.
4. Winner/candidates vs GT (crosshair px@5120 → frame px, existing convention) → `gt_in_topk`, rank-1, and the
   GT-bucket classification (§7).
5. Same frame scored by rcp-only + median-consensus arms → multi-way A/B.

## 6. Fusion specifics (the free parameters Codex flagged — pinned + swept)

**Heatmap (primary) — minimal knobs:**
- **Accumulation = SUM** across members (agreement). MAX (best-member-per-pixel) is *not* built here
  (it is the "per-member NCC-max" idea; a follow-up only if SUM underperforms).
- **Response registration:** each member's chamfer score map is placed in a common frame coordinate space
  (template-center convention, summed across the engine's scales) before accumulation — reuses the engine's
  existing cross-scale placement.
- **Only knob = peak-NMS radius** (min separation between extracted peaks); default tied to template
  short-side fraction; reported, lightly swept (±50%) to show stability. No clustering tolerance, no RRF `k`.

**RRF (extra arm) — the params Codex flagged, pinned + swept:**
- **One vote per member per cluster** (Codex #2): collapse each member's own near-duplicate candidates
  (within `cluster_tol`) so a single crop cannot double-vote.
- **Spatial clustering tolerance** `cluster_tol`: default tied to template short-side fraction; **swept** at
  ±5 / ±10 / ±20 px equivalents. Too tight splits real agreement; too loose merges true + periodic-distractor
  clusters one lattice period apart.
- **RRF constant** `k`: default 60; **swept** at 20 / 60 / 120. At large `k`, rank differences flatten and the
  fused score degenerates to "count of nearby nominations" — the reinforcement path; the sweep makes it visible.
- No RRF-arm headline number is reported without the tol × k **sensitivity table**.

## 7. Kill-test (run FIRST — decides H1 vs H0 cheaply)

Before trusting any aggregate lift, classify each fused **winner** against GT into buckets:

- `correct` — winner within GT tolerance.
- `near_periodic` — winner off GT by ~one lattice period (a consistent periodic distractor).
- `far_wrong` — winner far from GT, not a periodic offset.
- `one_member_only` — winning cluster supported by a single member.

**Kill criterion:** if, on the **heatmap (primary) arm**, `near_periodic` winners are a large fraction (and
notably larger than the median-consensus arm's), then **agreement is reinforcing distractors (H0)** → stop;
do not pursue the bank
(record the negative result; the lever is re-registration, not this matcher change). This directly answers
Codex #1 and replaces the fatal "≥2-member support" diagnostic (Codex #3): support is reported **conditioned on
GT bucket**, never as a bare count (which is circular, since fusion selects supported clusters).

The `near_periodic` test requires a per-recipe lattice-period estimate (from the template's autocorrelation /
dominant periodicity). If a recipe's period can't be estimated, its winner is bucketed `correct`/`far_wrong`
only and excluded from the periodic-reinforcement statistic (noted, not silently dropped).

## 8. Metrics & measurement rigor (hard requirements — Codex #5)

- **Both** `gt_in_topk` (recall, pre-rerank) **and** rank-1 (post-rerank), reported **separately** — a recall
  gain that dies at rerank, or vice-versa, must be visible.
- **OM/SEM stratified** always; **no pooled OM+SEM headline** (it can hide SEM regression behind OM gain).
- **`min_s` bins: 3 / 4–6 / 7+.** At `min_s=3`, LOO yields a 2-template bank ≈ median-of-2 — the mechanism is
  effectively untested, so that bin's numbers are reported but flagged as low-power.
- **Bootstrap confidence intervals** on each arm's recall/rank-1 (golden set is sparse — 298 recipes, S-thin;
  point estimates are not trustworthy). Lift is only claimed when CIs separate.
- Arms compared on **identical frames**.
- One-line `[DIGEST]` + a `digest.txt`, same relay convention as the other golden drivers.

## 9. Confounds explicitly acknowledged

- **False-S poisoning (Codex #5):** S labels are tool-reported; a false-S crop with prominent distractor
  structure poisons both the bank and the GT pool (same frames). We do **not** trust tool labels as a CV
  signal; the kill-test's `near_periodic` rate is the guard (a poisoned bank shows up as periodic
  reinforcement). Noted as an irreducible confound of the golden set, not solved here.
- **rcp-image weakness vs matcher weakness** remains unseparated (carried from the Phase 2 conclusion); this
  experiment tests a matcher-side lever and does not resolve that question.

## 10. Testing (Mac, synthetic, TDD)

- `bank_build`: N synthetic crops → N **individual** templates (not 1 median); respects `min_s`; cond-aware.
- `bank_match_heatmap` (**primary**) **positive case**: frame with a known mark + a distractor that fools
  **1 of N** members (distractors *decorrelated*) → accumulated peak lands on the true mark (H1 in miniature).
- `bank_match_heatmap` (**primary**) **recall-recovery case**: a true mark that is **rank-K+1 in every member**
  (in no member's top-K) but weakly+consistently present → the SUM heatmap peak still lands on it. This is the
  case RRF *cannot* pass, and the reason heatmap is primary (§2).
- `bank_match_heatmap` **negative case (H0)**: frame where **all** members are fooled by the **same** distractor
  → peak lands on the distractor, and the kill-test buckets it `near_periodic`. Proves the harness can *detect
  its own failure mode* rather than rubber-stamp H1.
- `bank_match_rrf` (extra) positive + H0 cases → confirms the discrete arm behaves and is comparable.
- Within-member dedup (RRF arm): a member with two near-duplicate candidates contributes **one** vote.
- RRF clustering: candidates within `cluster_tol` merge/reinforce; outside stay separate.
- Per-member proposer / response-map call is **bit-parity** with the engine primitives.
- Office-gated: the golden-set A/B accuracy (relayed as `[DIGEST]`).

## 11. Out of scope (YAGNI)

- No `workflow_3`/production changes (separate gated spec).
- No extra bank sources (history-S only; no rcp member, no synthetic augmentation).
- No threshold recalibration (recall/rank-1 first; thresholds are a port concern).
- No live/broad-scan cost work (N× match cost is a port concern, noted not solved).
- **Best-member / per-pixel MAX aggregation is not built** — heatmap (SUM, primary) + RRF (extra) only;
  MAX-heatmap / best-member is a follow-up only if SUM-heatmap underperforms.
- No per-crop distractor inpainting (a distinct idea; deferred — risks erasing the key's own internal
  structure if mis-detected).

## 12. Risk register (from Codex review)

| # | Risk | Severity | Mitigation in this spec |
|---|------|----------|-------------------------|
| 1 | Cross-member agreement reinforces consistent distractors (H0) | SERIOUS | Kill-test §7 front-loaded; `near_periodic` bucket; negative-case unit test §10 |
| 2 | RRF clustering free params manufacture lift | SERIOUS | One-vote-per-member §6; tol × k sensitivity sweep required §6 |
| 3 | "≥2-member support" diagnostic is circular/fatal | FATAL | Replaced by GT-bucket-conditioned classification §7 |
| 4 | RRF complexity not attributable vs simpler method | SERIOUS | Heatmap is now **primary**; RRF demoted to the extra arm that must *beat* heatmap to justify its complexity §4/§14 |
| 5 | Illusory lift (min_s=3 ≈ median-of-2, false-S, recall≠rank-1, OM hides SEM) | SERIOUS | min_s bins, both metrics, OM/SEM strat, bootstrap CIs §8; false-S confound §9 |

## 13. File structure

- `poc/workflow_2/template_bank_lab.py` — `bank_build`, `bank_match_rrf`, `bank_match_heatmap`, `BankResult`
  (+ within-member dedup, RRF spatial fuse, lattice-period estimate). Imports workflow_3 engine primitives;
  never edits workflow_3.
- `poc/workflow_2/test_template_bank_lab.py` — the TDD tests in §10 (incl. the H0 negative case).
- `poc/workflow_2/golden_consensus_eval_cond.py` — add bank + heatmap arms, kill-test report, stratified
  A/B with sweeps + bootstrap CIs, `[DIGEST]`.
- Config knobs bridged via `golden_eval_config` (cluster_tol, rrf_k, arm toggles) — same pattern as the
  existing bench knobs; gitignored `golden_eval_config.py` edited only via `.example.py` + loader.

## 14. Decision gate (what "done" means for THIS spec)

Run the bench. Decide one of:
- **H0 confirmed** (periodic reinforcement dominates — heatmap winners cluster `near_periodic`, ≥ median) →
  record negative result, do not port; matcher-bank is not the lever. (A clean, valuable kill.)
- **H1, heatmap wins** (the heatmap primary beats median-consensus on `gt_in_topk`/rank-1, CIs separate) →
  write the separate `workflow_3` port spec for the **heatmap** (simplest sufficient mechanism). The
  recall-recovery (`gt_not_in_topk` → recovered) magnitude is the headline.
- **H1, RRF earns its keep** (the RRF extra arm *further* beats heatmap with separated CIs) → port RRF instead;
  the discrete machinery is justified.
- **Neither beats median** → no lever here; close the thread.
