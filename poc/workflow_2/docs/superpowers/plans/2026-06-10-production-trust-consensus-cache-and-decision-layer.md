# Production-trust: consensus cache consumption + act/abstain decision layer — Plan

> **For agentic workers:** this is a strategy/phasing plan, not yet a task-by-task breakdown. Before
> implementing, run `superpowers:writing-plans` (or `agent-skills:plan` / `to-issues`) to decompose each phase
> into checkbox tasks with acceptance criteria, then use `superpowers:subagent-driven-development` /
> `executing-plans`. Develop+A/B in the `poc/workflow_2` lab, port to `poc/workflow_3` only after an office digest.

**Goal (KR):** 매칭 점수는 높지만(consensus in_topk ~0.876) 프로덕션에 못 쓰는 이유는 *신뢰도* 문제다 — 틀린 점을
자동 클릭하는 false-lock 이 사람에게 넘기는 것보다 나쁘기 때문. 평균 점수가 아니라 **operating point 의 신뢰도**
(false-lock rate, "모르겠다" 라고 abstain 하는 능력)를 올린다. 두 구멍을 메운다: (1) gather 만 되고 **소비 안 되는**
consensus 캐시를 correction 경로에 연결, (2) **lab 에 갇힌** abstain 게이트(peak-isolation AUC 0.91)를 프로덕션 3-way 결정으로.

**불변 제약(CLAUDE.md):** CLI 인자 금지 · Korean docstring · `[INFO]/[ERROR]/[WARNING]` print(+`log_work2_event`) ·
`from __future__` 금지 · 절대 임포트 · **workflow_3 는 legacy(wf1/wf2) import 금지** (= consensus build 는 workflow_3 로
이전하고 wf2 가 import) · commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Context — why this, why now

Many cycles raised the matcher score (ensemble C1/C2/C3 RRF + NCC rerank; consensus re-registration +0.442 in_topk
in lab). Scores are high but **not enough for production**. Exploration (2026-06-10) found two structural gaps:

1. **The consensus cache is gathered but never consumed.** `correct_align_fail_auto → build_templates_from_assets
   → resolve_assets_auto` reads **only the stale recipe-registered template** (`align_img_from_rcp/IMAP0001|0002`).
   The consensus **build** (`build_consensus_template` / `select_routing_templates` / `ConsensusPolicy`) lives only
   in `poc/workflow_2/consensus_template.py` and is never imported by workflow_3. The biggest lab win is
   structurally disconnected from the production loop — likely *the* reason "high scores aren't enough": the
   office matches against drift-stale templates.
2. **The abstain gate is trapped in the lab.** `peak_isolation_ratio` / `miss_predictor_stats` (AUC 0.91, Youden
   tau) live in `ensemble_lab.py`. Production `key_visibility_gate` (`align_fail_correct.py:104-126`) is **binary**
   (auto-act vs. fallback-search) — both paths attempt a correction; there is **no "ambiguous → abstain to human"**
   branch. The needed signal (`second_ratio`) is already on `AlignKeyMatchResult` from `_finalize_match`, but the
   threshold is hand-set (cold-start `max_second_ratio=0.94`), not calibrated.

**Lead track (this plan):** harden the consensus cache + build a trustworthy act/abstain/escalate decision layer.
Learned-feature (DINOv2/LoFTR) and VLM-reranker tiers are deferred to a later residual-only pass.

### Definition of "production-ready" (the target metric)

- **false-lock rate** — fraction of auto-ACT decisions where the locked point is wrong → must be near-zero.
- **trustworthy coverage** — fraction of alarms auto-acted while false-lock ≤ target (e.g. ≤1–2%).
- **abstain precision** — of abstained cases, fraction genuinely unsafe (guard against over-abstaining → everything
  to humans → no automation value).

The office A/B digest must report this act/abstain trade-off curve — **not** just in_topk.

---

## Phase 1 — Calibrated act / abstain / escalate gate (decision-layer skeleton)

Small, self-contained, reuses an already-computed signal; delivers trust value immediately (improves human routing
even under dry-run). **Recommended first.**

- **Port the abstain signal into production.** Move `peak_isolation_ratio` + `miss_predictor_stats` (Youden tau)
  logic from `poc/workflow_2/ensemble_lab.py` into a small production helper under `poc/workflow_3/vision/`. The raw
  signal (`second_ratio`) is already produced by `_finalize_match` (`align_key_matcher.py:759-843`) — what's missing
  is the *calibrated tau* and the *three-way use*.
- **Three-way `key_visibility_gate`** (`align_fail_correct.py:104-126`):
  - **ACT** — strong score AND distinctive (`second_ratio ≤ tau_act`) AND `best_scale ≥ MIN_CONFIRM_SCALE` → primary reposition + OK.
  - **ABSTAIN** — ambiguous (`second_ratio > tau_abstain`) OR score below confidence floor → do NOT click; notify engineer with candidate overlay.
  - **FALLBACK** — middle band → `live_align_search` (as today).
- **New `CorrectionOutcome.status`:** `abstained_ambiguous`, `abstained_low_confidence`. Map in
  `monitor/notify.py:notify_correction_outcome` (200-246) to distinct cube notifications — tone = "deliberately did
  NOT act, human please decide" (different from a fallback failure).
- Keep dry-run posture intact (`correction_dry_run = dry_run_requested or safe_mode`). The gate improves routing
  regardless of actuation.
- `tau_act` / `tau_abstain` start at the lab Youden value as placeholders; flagged for Phase 3 recalibration.

## Phase 2 — Consensus-cache hardening + consumption (the headline ask)

Make the gathered fresh template actually flow into correction, with quality guards. Larger lift; root-cause fix.

- **Relocate the consensus build into workflow_3** (CLAUDE.md: legacy imports from workflow_3, never reverse): move
  `build_consensus_template`, `select_routing_templates`, `ConsensusPolicy` (blur guard edge/lap ratio, `min_s`) +
  primitives (`_consensus` median, `_edge_density`, `_lap_var`) from `poc/workflow_2/consensus_template.py` /
  `align_similarity.py` into a new `poc/workflow_3/vision/consensus_template.py`. Refactor wf2 to import it
  (bit-parity so the golden eval is unaffected).
- **Consume the cache in the correction path.** Before `build_templates_from_assets`
  (`align_fail_correct.py:176-192`), add a resolver step that:
  1. Reads `ALIGN_CONSENSUS_CACHE_DIR/<eqp>/<class>/<recipe>/events/<event_id>/S*.jpeg + .S*/cond.txt`.
  2. Groups crops by modality (OM/SEM), crosshair-inpaints + co-registers, builds per-modality consensus via
     `build_consensus_template` with blur/min-S guards.
  3. Routes via `select_routing_templates(consensus_by_mod, rcp_by_mod)` → **consensus when fresh & sharp, else
     fall back to rcp**. Pass routed templates into the matcher.
- **Loop-time quality guards** (currently wf2-only): blur guard (edge-density / laplacian-variance ratio), `min_s`
  floor, co-registration verification, crosshair inpaint — enforced at gather/build time in workflow_3.
- **Cache lifecycle:** documented freshness/eviction (rolling window / TTL) + materialized per-recipe "current
  consensus" (write built templates to `…/consensus/` to avoid rebuilding every alarm). Today: unbounded
  replace-if-non-empty.
- **office_success_downloader** stays office-only (gitignored). On Mac, lock the `SuccessDownloader` Protocol +
  cond-file contract and keep `verify_success_gather.py` green. This is the activation gate.

## Phase 3 — Couple cache quality into the gate + recalibrate tau

- **Cache-aware gate:** matching against a *fresh, sharp consensus* → more permissive (act); only stale rcp →
  conservative (abstain/escalate sooner). Feed consensus availability/quality (S-count, blur ratio) into the gate.
- **Recalibrate tau on real fail-frames.** AUC-0.91 was S-LOO; the production operating tau must be re-fit on actual
  fail-frame outcomes (known caveat). Phase 2 shifts the `second_ratio` distribution, so re-fit *after* Phase 2.
  The always-on recording session already captures fail-time frames + engineer manual ops → use as the labeled set.

---

## Validation (office-digest discipline)

- Extend `poc/workflow_2/golden_consensus_eval_cond.py` to emit the **production metric**: a false-lock / coverage
  / abstain-precision curve over the tau sweep (not just in_topk). Add an A/B: **stale-rcp vs. consensus-from-cache**
  on the *act/abstain* outcome (does fresh template raise trustworthy coverage at fixed false-lock?).
- Office runs `uv run python poc/workflow_2/golden_consensus_eval_cond.py` (no args) → text digest back to Mac
  (no fab images leave the office). Mac reshapes thresholds from the digest.
- Offline tests stay green: `test_consensus_gather.py`; extend `test_align_fail_correct.py` with `abstained_*`
  paths; new `test_consensus_template.py` for the relocated build.
- Mac dry-run e2e: `SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture> uv run python
  poc/workflow_3/monitor/align_fail_monitor.py` — confirm the loop (a) builds a consensus template when a cache
  fixture is present and (b) emits `abstained_*` for a synthetically ambiguous match.

## Critical files

- `poc/workflow_3/vision/align_fail_correct.py` — `key_visibility_gate` (104-126), `build_templates_from_assets`
  (176-192), `correct_align_fail_auto` (399-442): consensus-resolver step + three-way gate + new statuses.
- `poc/workflow_3/vision/align_key_matcher.py` — `_finalize_match` (759-843), `MatchPolicy`/`STRUCTURE_POLICY`
  (120-163): source of `second_ratio`; add calibrated tau fields.
- `poc/workflow_3/vision/consensus_template.py` — **NEW**, relocated from `poc/workflow_2/consensus_template.py`
  (+ primitives from `poc/workflow_2/align_similarity.py`).
- `poc/workflow_3/vision/align_fail_assets.py` — `resolve_assets_auto` (227-277): add consensus-cache lookup.
- `poc/workflow_3/vision/consensus_gather.py` — gather writer (76-123): loop-time quality guards + lifecycle.
- `poc/workflow_3/monitor/notify.py` — `notify_correction_outcome` (200-246): map `abstained_*`.
- `poc/workflow_3/__init__.py` — `ALIGN_CONSENSUS_CACHE_DIR` (73-77).
- `poc/workflow_2/golden_consensus_eval_cond.py` + `ensemble_lab.py` — extend digest with false-lock/coverage curve.
- `poc/workflow_3/config.py` — `Workflow3Settings`: tau / consensus-routing toggles (env-overridable).

## Out of scope (deferred residual-only tiers)

- Learned-feature proposer channel (DINOv2/LoFTR, ~11–25M params, server-side Flask or local ONNX-CPU) — orthogonal
  signal for "answer not in top-K" structural misses. Revisit only after the cache+gate ceiling is measured.
- VLM multi-image re-ranker (Qwen3-VL-30B, validated in `poc/workflow_2/vlm_align_key_region.py`) — for
  "mis-ranked but present" cases, gated by the abstain flag.

## Open decisions (resolve during execution)

- Phase ordering: Phase 1 first (fast trust value) vs. Phase 2 first (root fix) vs. parallel. Recommended: Phase 1 → 2 → 3.
- Consensus routing policy: consensus-*replaces*-rcp vs. consensus-and-rcp-*both-as-candidates*.
- Freshness/TTL window: how many days / how many S events define "current".

---

_Related: spec/plan `2026-06-10-consensus-gather-in-loop(-design).md` (gather landed; build + consumption = this plan)._
