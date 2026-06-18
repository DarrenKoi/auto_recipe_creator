# Drop `align_img_from_msr` from the production loop

**Date:** 2026-06-18
**Status:** Approved (design)
**Scope:** `poc/workflow_3/` production loop (`align_fail_monitor`, `align_fail_monitor_only_check`)

## Problem

The production loop downloads `align_img_from_msr` (the measurement trajectory: `S*` success / `E*` fail frames) on the critical path via a **synchronous, blocking** office fetch (`rcp_msr_gather.gather_rcp_msr` → `office_rcp_msr_downloader.download_rcp_msr`) before each cycle reads assets.

But the runtime correction/feasibility code **never consumes msr**:

- `correct_align_fail_auto` (`align/correction.py`) captures its own **live SEM frame** (`controller.capture()`) and matches a **consensus or rcp** template *into* that frame. `best_xy` is a coordinate in the live image; the align point comes from the template's `align_offset_xy` (`align/templates.py:34`, from the rcp cond box) plus scale-band matching.
- `mark_align_feasibility` (`align/diagnostics/feasibility_check.py`) reads rcp templates only.
- `assets.py` loads `from_msr` / `current_sem`, but those fields are consumed **only** by offline diagnostics (`compare_align_images`, `crosshair_detect`, `align_point_correction`).

So msr is a blocking input the loop pays for and discards. Since the consensus method (success images + rcp) is now the template path, msr can be removed from production.

### Why the live crop can't replace the template

The live SEM frame is the **search space** (haystack), not the source of the align point. At align-fail time the live frame is an `E`-class view with **no crosshair** — it does not encode where the align point is. The align point is only knowable by locating the **registered key (rcp)** or a **past-success key (consensus, which carries the crosshair)** inside the live frame and applying its offset. Therefore the reference template stays; only the unused msr fetch is removed.

## Goal

Production loop (`align_fail_monitor` + `align_fail_monitor_only_check`) stops downloading `align_img_from_msr`. Runtime template inputs become **consensus (preferred) → rcp (fallback)**. msr remains fetchable **on demand** for the offline benchmark (workflow_2 / workflow_3 eval), not on the loop's critical path.

## Decision: (A) decouple code, keep collection capability

Confirmed with user:
- Remove the **runtime** dependency on msr (the blocking download + the misleading missing-asset warning).
- **Keep** an on-demand path to fetch msr, because offline benchmark work on workflow_2/_3 continues.
- The S-sparsity risk (consensus may not build for a recipe → rcp-only fallback) is **accepted**: it is the same floor as today, no regression. When the system is not confident, take **no action** and let engineers handle it (existing `engineer_review` gate / cube notify).

## Changes

### 1. Contract + gather — `monitor/rcp_msr_gather.py` (git-tracked)
- Add `include_msr: bool = False` to the `RcpMsrDownloader.download_rcp_msr` Protocol and thread it through `gather_rcp_msr(eqp_id, recipe_id, settings, *, include_msr=False)`.
- Production default = **rcp-only**. The two production callers (`align_fail_monitor.py:352`, `align_fail_monitor_only_check.py:163`) keep calling `gather_rcp_msr(...)` with the default and thus stop requesting msr.
- Update the module docstring: rcp is the runtime input; msr is offline-bench-only.

### 2. Office reference impl — `monitor/temp_office_rcp_msr_downloader.py`
- Honor `include_msr`: skip the `download_align_images_from_msr` calls (lines ~105, ~112) when `include_msr` is False.
- Document that the **real** gitignored `office_rcp_msr_downloader.py` needs the same guard applied **at the office** (cannot be edited from the Mac dev machine). Until that edit lands, the office downloader still fetches msr — the contract is in place but the speed-up only realizes after the office edit.

### 3. `align/assets.py`
- Remove `current_sem` from the missing-asset WARNING loop (lines 218-221) so an absent msr tree at runtime is silent. `recipe_om` / `recipe_sem` keep warning (runtime-essential).
- Keep loading `from_msr` / `current_sem` when present (empty tuple / `None` otherwise) so offline diagnostics still work on bench data. No signature change to `resolve_assets`.

### 4. Offline-bench fetch — new `monitor/fetch_msr_offline.py`
- Standalone, no-CLI-args, `uv run python poc/workflow_3/monitor/fetch_msr_offline.py`.
- Calls the downloader with `include_msr=True` for one configured `EQP_ID` / `RECIPE_ID` (env / hardcoded defaults, per repo convention) to deliberately expand the golden set.
- Best-effort, prints `[INFO]`/`[WARNING]`, never required by the loop.

### 5. Tests + docs
- `monitor/test_rcp_msr_gather.py`: assert the production call requests **rcp-only** (`include_msr=False`) and that `include_msr=True` requests msr. Existing gate tests (enabled flag / empty recipe_id / downloader absent) keep passing.
- `monitor/test_occupied_popup.py:152` stubs `gather_rcp_msr` — unaffected.
- Docs: update `config.py:93`, `__init__.py:61` comments, and `CLAUDE.md` (filesystem contract + env notes) to state msr is no longer a runtime input (consensus → rcp), and that `fetch_msr_offline.py` is the bench fetch.

## Out of scope / unchanged
- Matching pipeline: `correction.py`, `feasibility_check.py`, `templates.py` — already match consensus/rcp into the live crop.
- Consensus modules (`consensus_*`, `success_gather`) — already the primary template path; no new wiring.
- Offline diagnostics that read msr — unchanged; they operate on the existing/bench tree.

## Risk & mitigation
- **Consensus not built for a recipe (S-sparse).** Falls back to rcp-only template — identical to current floor, no regression. When confidence is low, the loop already escalates to engineer review rather than acting.
- **Office module lag.** The git-tracked contract requests rcp-only, but the real office downloader must be edited at the office to actually skip msr I/O. Until then: correct behavior, no speed-up yet (msr still downloaded but still ignored). No functional risk.

## Acceptance
- Production loop no longer requests msr (verified via `test_rcp_msr_gather.py` and the rcp-only call sites).
- No `current_sem` missing-asset warning during runtime when msr is absent.
- `fetch_msr_offline.py` fetches rcp+msr when run manually.
- Offline diagnostics still function on a tree that contains msr.
