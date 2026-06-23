# Re-registration Phase 2 — E-frame Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Phase 1's S-only latent-risk candidates to `E_CONFIRMED` when the align-key free-search best-score collapses from success (S) to fail (E) frames.

**Architecture:** A post-pass inside the existing driver `poc/workflow_2/golden_reregister_report_cond.py`, run after Phase 1 builds rows (tiers assigned) and before ranking. For each flagged row it reuses the S best-scores Phase 1 already computed, runs the same proposer on E frames (no GT — E frames have no crosshair), and applies a high-S-premise collapse rule. Phase 1 logic is untouched; the post-pass only overrides the tier.

**Tech Stack:** Python 3.10+, numpy, OpenCV (via the workflow_3 match engine), pytest, uv.

**Spec:** `poc/workflow_2/docs/specs/2026-06-23-reregister-phase2-eframe-confirmation-design.md`

---

## ⏸ RESUME / HANDOFF (paused 2026-06-23 EOD — continue tomorrow)

**Status:** Spec + this plan are DONE and committed. **Phase 2 code is NOT started** — 0/6 tasks
implemented. Nothing in `golden_reregister_report_cond.py` has Phase 2 changes yet.

**Exact next action to resume:** decide execution mode, then start at **Task 1**.
- Recommended: **Subagent-Driven Development** (we used it for Phase 1). To resume, invoke
  `superpowers:subagent-driven-development` against THIS plan file; it dispatches a fresh subagent
  per task (1→6), reviews each, then a whole-branch review. Tasks 1-5 verify on Mac
  (`uv run pytest poc/workflow_2/test_reregister_report.py`); Task 6 accuracy is office-only.
- Alternative: inline execution (executing-plans) with checkpoints.
- A `.superpowers/sdd/progress.md` ledger (gitignored) will track task completion if SDD is used —
  check it first on resume; tasks marked complete there are DONE, do not re-dispatch.

**Git state:** branch `main`, HEAD = `f3274e1`, everything pushed (Mac→push→office pull workflow).
Phase 2 doc commits: spec `99d5833`, plan `f3274e1`. Resume by `git pull` at the office first.

**What already shipped this session (Phase 1 box-fidelity thread — COMPLETE):**
`f7501fb` off-center offset fix → `463426b` tight scale band → `b7cb567` tol widen 0.20→0.30
(+ `47cda7e`/`6aa23af`/`ca31403` diagnostics, `7284572` fast-mode cap, `8dbe880` config bridge).
Result: `w_sugg 0→1`. Knobs `REREGISTER_MAX_RECIPES`, `REREGISTER_FIDELITY_SCALES`,
`REREGISTER_GT_TOL_NORM` all bridge via `golden_eval_config`. Journal:
`docs/journals/260623/260623_155946_reregister-phase1-box-fidelity-debug.md`. Memory:
`project_reregister_report_phase1.md`.

**Open loose end (NOT blocking Phase 2):** a full uncapped office run of the box-fidelity changes
is still pending — `w_sugg` was confirmed `1` only on a 20-recipe fast-mode cap. When convenient run
`REREGISTER_MAX_RECIPES=0 uv run python poc/workflow_2/golden_reregister_report_cond.py` and relay
the `[DIGEST]` for the full 67-recipe set.

**Phase 2 calibration (after Task 6 lands):** first office run prints
`[INFO] e_confirm on: S_FLOOR=.. E_FLOOR=.. COLLAPSE_MARGIN=..` + a `[DIGEST]` with `confirmed C`
per modality. Relay those + 1-2 sample `E_CONFIRMED` rows (`s_rep->e_rep(n_e=K)`), then tune
`S_FLOOR/E_FLOOR/COLLAPSE_MARGIN` via env (code unchanged) using the observed S/E score spread.
Defaults (0.60/0.50/0.15) are starting guesses since matcher scores cluster ~0.6.

**Hard reminders for whoever resumes:** office data cannot come to Mac (write blind, relay digest
text — never ask for sample images); commit directly to `main` with **pathspec** (only the files a
task touches; no `add -A`); ASCII-only `print()` (cp949 console); never edit the gitignored
`golden_eval_config.py` (edit `.example.py` + loader). Full rules in Global Constraints below.

---

## Global Constraints

- Korean docstrings throughout; print-based logging with `[INFO]`/`[WARNING]` prefixes (never the `logging` module).
- **ASCII-only** in any `print(...)` string (office console is cp949); no em-dash (U+2014). Docstrings may use any text.
- No `argparse`/CLI flags; config via env + `golden_eval_config`. No `from __future__` imports.
- Absolute imports `from poc.workflow_3...`; workflow_2 imports workflow_3, never the reverse.
- `golden_eval_config.py` is gitignored office scratch — edit only the tracked `golden_eval_config.example.py` + `golden_eval_config_loader.py`.
- Aggregation is **median**. `E_CONFIRMED` is the top tier and **overrides** the Phase 1 tier (never computed inside `_evidence_tier`).
- Phase 2 thresholds are env-configurable AND bridged via `golden_eval_config` (same pattern as `REREGISTER_ACCEPT_MARGIN`).
- E free-search uses `COMPARE_SCALES` (center-crop localization, same as `_gt_in_topk`), NOT `_FIDELITY_SCALES`.
- E best-score proposer call must stay bit-parity with `_gt_in_topk`'s proposer call (`align_similarity.py:348,361`).

## File Structure

- `poc/workflow_2/golden_reregister_report_cond.py` — all new helpers + the run() post-pass (single driver, mirrors Phase 1).
- `poc/workflow_2/test_reregister_report.py` — new unit tests (extends the existing file).
- `poc/workflow_2/golden_eval_config.example.py` + `golden_eval_config_loader.py` — 4 new threshold/toggle knobs.

**Tasks 1-5 are Mac-testable (pure helpers + engine-backed synthetic + formatting). Task 6 is office-run driver glue.**
Sequencing: 1 → 2 → 3 → 4 → 5 → 6.

---

## Task 1: Config knobs

**Files:**
- Modify: `poc/workflow_2/golden_eval_config.example.py` (after the `REREGISTER_FIDELITY_SCALES` / `REREGISTER_GT_TOL_NORM` block)
- Modify: `poc/workflow_2/golden_eval_config_loader.py` (import block + `seed_env`)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces: env vars `REREGISTER_E_CONFIRM` (default "1"), `REREGISTER_S_FLOOR` ("0.60"), `REREGISTER_E_FLOOR` ("0.50"), `REREGISTER_COLLAPSE_MARGIN` ("0.15"), bridged from `golden_eval_config` via `setdefault` (real env wins).

- [ ] **Step 1: Write the failing test**

Add to `poc/workflow_2/test_reregister_report.py`:

```python
def test_seed_env_bridges_e_confirm_defaults(monkeypatch):
    for k in ("REREGISTER_E_CONFIRM", "REREGISTER_S_FLOOR", "REREGISTER_E_FLOOR",
              "REREGISTER_COLLAPSE_MARGIN"):
        monkeypatch.delenv(k, raising=False)
    cfg.seed_env()
    assert os.environ["REREGISTER_E_CONFIRM"] == "1"
    assert os.environ["REREGISTER_S_FLOOR"] == "0.6"
    assert os.environ["REREGISTER_E_FLOOR"] == "0.5"
    assert os.environ["REREGISTER_COLLAPSE_MARGIN"] == "0.15"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py::test_seed_env_bridges_e_confirm_defaults -q`
Expected: FAIL (KeyError — keys not yet bridged).

- [ ] **Step 3: Implement**

In `golden_eval_config.example.py`, after the `REREGISTER_GT_TOL_NORM = 0.30` line:

```python
# === Phase 2: E-frame confirmation (golden_reregister_report_cond) ===
# 1 이면 flagged recipe 를 E(fail) 프레임 score-collapse 로 confirm, 0 이면 Phase 1 만.
REREGISTER_E_CONFIRM = 1
# collapse 규칙 임계(점수 ~0.6 압축 분포 기준 출발점, office 보정 대상).
REREGISTER_S_FLOOR = 0.6           # S best-score median 이 이 이상이어야 'collapse' 전제 성립.
REREGISTER_E_FLOOR = 0.5           # E best-score median 이 이 밑이면 (delta 작아도) collapse.
REREGISTER_COLLAPSE_MARGIN = 0.15  # S_rep - E_rep 이 이 이상이면 collapse.
```

In `golden_eval_config_loader.py`, extend the inner import try/except (after the `REREGISTER_GT_TOL_NORM` import block):

```python
    try:
        from poc.workflow_2.golden_eval_config import (
            REREGISTER_E_CONFIRM, REREGISTER_S_FLOOR, REREGISTER_E_FLOOR,
            REREGISTER_COLLAPSE_MARGIN,
        )
    except ImportError:   # 구버전 config(Phase 2 knob 없음) 하위호환.
        REREGISTER_E_CONFIRM = 1
        REREGISTER_S_FLOOR, REREGISTER_E_FLOOR, REREGISTER_COLLAPSE_MARGIN = 0.6, 0.5, 0.15
```

In the OUTER `except ImportError` (file absent) defaults block, append:

```python
    REREGISTER_E_CONFIRM = 1
    REREGISTER_S_FLOOR, REREGISTER_E_FLOOR, REREGISTER_COLLAPSE_MARGIN = 0.6, 0.5, 0.15
```

In `seed_env()`, after the `REREGISTER_GT_TOL_NORM` block:

```python
    os.environ.setdefault("REREGISTER_E_CONFIRM", str(REREGISTER_E_CONFIRM))
    os.environ.setdefault("REREGISTER_S_FLOOR", str(REREGISTER_S_FLOOR))
    os.environ.setdefault("REREGISTER_E_FLOOR", str(REREGISTER_E_FLOOR))
    os.environ.setdefault("REREGISTER_COLLAPSE_MARGIN", str(REREGISTER_COLLAPSE_MARGIN))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py::test_seed_env_bridges_e_confirm_defaults -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config_loader.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): Phase 2 config knobs (E-confirm + collapse thresholds)"
```

---

## Task 2: `_e_confirm` rule + `E_CONFIRMED` tier weight + thresholds

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (`TIER_WEIGHT` at line 29; new consts + `_e_confirm` near the tier helpers ~line 118-135)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: env from Task 1.
- Produces: `S_FLOOR`, `E_FLOOR`, `COLLAPSE_MARGIN`, `E_CONFIRM_ON` module consts; `_e_confirm(s_rep, e_rep) -> bool`; `TIER_WEIGHT["E_CONFIRMED"] = 3.0`.

- [ ] **Step 1: Write the failing test**

```python
def test_e_confirm_rule_all_branches():
    import poc.workflow_2.golden_reregister_report_cond as r
    # high-S premise met + delta collapse -> confirm.
    assert r._e_confirm(0.80, 0.60) is True           # delta 0.20 >= 0.15
    # high-S premise met + E below floor (delta small) -> confirm.
    assert r._e_confirm(0.62, 0.49) is True            # delta 0.13 < 0.15 but e <= 0.50
    # high-S premise met but E holds up -> no collapse.
    assert r._e_confirm(0.80, 0.70) is False           # delta 0.10 < 0.15 and e > 0.50
    # low S (no premise) -> never confirm even if E tiny.
    assert r._e_confirm(0.55, 0.10) is False           # s < 0.60
    # missing reps -> no confirm.
    assert r._e_confirm(None, 0.10) is False
    assert r._e_confirm(0.90, None) is False

def test_e_confirmed_tier_is_top_weight():
    import poc.workflow_2.golden_reregister_report_cond as r
    assert r.TIER_WEIGHT["E_CONFIRMED"] > r.TIER_WEIGHT["STRONG"]
    assert r._risk_score("E_CONFIRMED", 0.2) > r._risk_score("STRONG", 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "e_confirm_rule or e_confirmed_tier"`
Expected: FAIL (`_e_confirm` undefined; `E_CONFIRMED` not in `TIER_WEIGHT`).

- [ ] **Step 3: Implement**

Replace `TIER_WEIGHT` at line 29:

```python
TIER_WEIGHT = {"E_CONFIRMED": 3.0, "STRONG": 2.0, "MEDIUM": 1.0, "ADVISORY": 0.3, "NONE": 0.0}
```

Add consts near the other env-read consts (after `ACCEPT_MARGIN`, before `TIER_WEIGHT` or just after it):

```python
# Phase 2 (E-frame confirmation) 임계 — office 보정 대상(점수 ~0.6 압축 분포).
E_CONFIRM_ON = os.getenv("REREGISTER_E_CONFIRM", "1") != "0"
S_FLOOR = float(os.getenv("REREGISTER_S_FLOOR", "0.60"))
E_FLOOR = float(os.getenv("REREGISTER_E_FLOOR", "0.50"))
COLLAPSE_MARGIN = float(os.getenv("REREGISTER_COLLAPSE_MARGIN", "0.15"))
```

Add `_e_confirm` next to `_risk_score` (~line 136):

```python
def _e_confirm(s_rep, e_rep):
    """score collapse S->E 판정. high-S premise + (delta 또는 E-floor). 순수 함수.

    s_rep/e_rep: free-search best-score 의 modality-row median(없으면 None).
    """
    if s_rep is None or e_rep is None:
        return False
    if s_rep < S_FLOOR:                       # high-S premise: 애초에 findable 했어야 collapse.
        return False
    return (s_rep - e_rep >= COLLAPSE_MARGIN) or (e_rep <= E_FLOOR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "e_confirm_rule or e_confirmed_tier"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): _e_confirm rule + E_CONFIRMED top tier weight"
```

---

## Task 3: `_median` + `_s_rep_score` (pure aggregation)

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (new pure helpers near `_mean` ~line 187)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces: `_median(xs) -> float | None`; `_s_rep_score(frame_results) -> float | None` (median of per-frame `cand_scores[0]`).
- Consumes (data shape): each `frame_results` item is a dict from `_gt_in_topk` carrying `cand_scores` (descending proposer scores), e.g. `{"cand_scores": [0.71, 0.65, ...], ...}`.

- [ ] **Step 1: Write the failing test**

```python
def test_median_odd_even_and_empty():
    import poc.workflow_2.golden_reregister_report_cond as r
    assert r._median([0.2, 0.9, 0.5]) == 0.5            # odd
    assert r._median([0.2, 0.8]) == 0.5                  # even -> mean of two
    assert r._median([]) is None

def test_s_rep_score_uses_best_per_frame_median():
    import poc.workflow_2.golden_reregister_report_cond as r
    frame_results = [
        {"cand_scores": [0.80, 0.4]},
        {"cand_scores": [0.60, 0.3]},
        {"cand_scores": []},          # 점수 없는 프레임은 skip.
    ]
    assert r._s_rep_score(frame_results) == 0.70          # median(0.80, 0.60)
    assert r._s_rep_score([]) is None
    assert r._s_rep_score([{"cand_scores": []}]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "median_odd or s_rep_score"`
Expected: FAIL (`_median` / `_s_rep_score` undefined).

- [ ] **Step 3: Implement**

Add near `_mean` (~line 187):

```python
def _median(xs):
    """리스트 median. 짝수 길이는 가운데 두 값 평균. 빈 리스트는 None."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def _s_rep_score(frame_results):
    """S frame_results 의 프레임별 best proposer 점수(cand_scores[0])의 median. 없으면 None."""
    scores = []
    for fr in frame_results:
        cs = fr.get("cand_scores") or []
        if cs:
            scores.append(float(cs[0]))
    return _median(scores)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "median_odd or s_rep_score"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): _median + _s_rep_score pure aggregation"
```

---

## Task 4: `_free_search_best_score` + `_e_rep_score` (engine-backed)

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (extend the `align_similarity` import at line 544; add `preprocess_for_matching` import; new helpers near `_self_match_ratio` ~line 680)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `_median` (Task 3); `center_tpl` is an `AlignKeyTemplate` from `build_template`.
- Produces: `_free_search_best_score(center_tpl, gray) -> float | None` (None on exception/empty candidates); `_e_rep_score(center_tpl, e_frames) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
def test_free_search_best_score_localizes_mark():
    import numpy as np
    import poc.workflow_2.golden_reregister_report_cond as r
    from poc.workflow_3.align.matching.engine import build_template
    img = _frame_with_offset_mark()                      # 기존 합성 헬퍼(저텍스처+고유 마크).
    tpl = build_template(img.copy(), recipe_id="e", version="e", key_type="om")
    score = r._free_search_best_score(tpl, img)          # 자기 자신에 free-search -> 강한 점수.
    assert score is not None and score > 0.0

def test_e_rep_score_median_and_empty():
    import poc.workflow_2.golden_reregister_report_cond as r
    from poc.workflow_3.align.matching.engine import build_template
    img = _frame_with_offset_mark()
    tpl = build_template(img.copy(), recipe_id="e", version="e", key_type="om")
    one = r._free_search_best_score(tpl, img)
    assert r._e_rep_score(tpl, [img, img]) == one        # 동일 프레임 2개 median = 단일 점수.
    assert r._e_rep_score(tpl, []) is None               # E 없음 -> None.
```

(`_frame_with_offset_mark` already exists in the test file from the Phase 1 fidelity work.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "free_search_best_score or e_rep_score_median"`
Expected: FAIL (`_free_search_best_score` / `_e_rep_score` undefined).

- [ ] **Step 3: Implement**

Extend the import at line 544:

```python
from poc.workflow_2.align_similarity import (
    _build_templates, _gt_in_topk, COMPARE_SCALES,
    _propose_topk, USE_ENSEMBLE_PROPOSER, TOPK_CANDIDATES,
)
```

Add `preprocess_for_matching` to the engine import block (near `build_template`):

```python
from poc.workflow_3.align.matching.engine import preprocess_for_matching
```

Add helpers near `_self_match_ratio` (~line 680):

```python
def _free_search_best_score(center_tpl, gray):
    """center 템플릿 free-search 최고 proposer 점수. 예외/후보없음(artifact)이면 None.

    낮은 점수는 collapse 증거이므로 절대 None/0 으로 버리지 않는다(후보가 있으면 float).
    proposer 호출은 _gt_in_topk 와 bit-parity (align_similarity.py:348,361) — 변경 시 함께.
    E free-search 는 center-crop localization 이라 COMPARE_SCALES (NOT _FIDELITY_SCALES).
    """
    try:
        frame_dt = None if USE_ENSEMBLE_PROPOSER else preprocess_for_matching(gray)[1]
        cands = _propose_topk(center_tpl, gray, frame_dt, scales=COMPARE_SCALES, topk=TOPK_CANDIDATES)
    except Exception:
        return None
    if not cands:
        return None
    return float(max(c.score for c in cands))


def _e_rep_score(center_tpl, e_frames):
    """E 프레임별 best score(None 제외)의 median. 사용가능 점수 0개면 None."""
    scores = [s for s in (_free_search_best_score(center_tpl, g) for g in e_frames) if s is not None]
    return _median(scores)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "free_search_best_score or e_rep_score_median"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): _free_search_best_score + _e_rep_score (E proposer, bit-parity)"
```

---

## Task 5: DIGEST / report / banner for `E_CONFIRMED`

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (`_format_digest` ~line 172, `_format_report` ~line 151)
- Test: `poc/workflow_2/test_reregister_report.py` (add new + update existing digest/report assertions)

**Interfaces:**
- Consumes (row fields): `e_confirmed` (bool), `s_rep`/`e_rep` (float|None), `n_e` (int) — set by Task 6; tests here supply them directly.
- Produces: digest gains `confirmed N` per modality; report gains `s_rep->e_rep(n_e=K)` column + an E-confirmation note line.

- [ ] **Step 1: Write the failing test**

```python
def test_digest_includes_confirmed_count():
    import poc.workflow_2.golden_reregister_report_cond as r
    rows_by_mod = {"om": [
        {"recipe": "c/a", "tier": "E_CONFIRMED", "e_confirmed": True, "suggestion": "none"},
        {"recipe": "c/b", "tier": "STRONG", "e_confirmed": False, "suggestion": "box(1,2,3,4)"},
    ], "sem": []}
    d = r._format_digest(rows_by_mod)
    assert "confirmed 1" in d
    assert d == d.encode("ascii", "replace").decode("ascii")   # ASCII only.

def test_report_shows_e_columns_and_note():
    import poc.workflow_2.golden_reregister_report_cond as r
    rows_by_mod = {"om": [
        {"recipe": "c/a", "tier": "E_CONFIRMED", "e_confirmed": True,
         "strong_fail_frac": 0.5, "worst_disp": 0.9, "msr_peak_tail": 0.1,
         "self_ratio": 0.2, "advisory_confidence": "ok", "n_s": 3,
         "suggestion": "none", "sugg_self": None, "sugg_fidelity": None,
         "s_rep": 0.80, "e_rep": 0.55, "n_e": 2},
    ], "sem": []}
    out = r._format_report(rows_by_mod)
    assert "0.800->0.550" in out          # s_rep->e_rep column.
    assert "n_e=2" in out
    assert "E_CONFIRMED rows" in out      # confirmation note line present.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "digest_includes_confirmed or report_shows_e_columns"`
Expected: FAIL (no `confirmed` count, no s_rep->e_rep column / note).

- [ ] **Step 3: Implement**

In `_format_digest`, inside the per-mod loop, add the confirmed count and include it in the part string:

```python
        strong = sum(1 for r in rows if r["tier"] == "STRONG")
        confirmed = sum(1 for r in rows if r.get("e_confirmed"))
        w_sugg = sum(1 for r in rows if str(r.get("suggestion", "none")).startswith("box"))
        top = ",".join(r["recipe"] for r in rows[:2]) or "-"
        parts.append(f"{mod}[screened {len(rows)}, strong {strong}, confirmed {confirmed}, "
                     f"w_sugg {w_sugg}, top {top}]")
```

In `_format_report`: add the `s_rep->e_rep` column to the header and each row, and append a note line when any row is confirmed. Replace the `cols` line and the row-append, and add the note before `return`:

```python
    cols = ("rank recipe tier strong_fail worst_disp msr_tail self_ratio(conf) "
            "n_s s_rep->e_rep(n_e) suggestion sugg_self/fid")
    ...
        for i, r in enumerate(rows, 1):
            lines.append(" ".join([
                str(i), r["recipe"], r["tier"], _fmt_num(r["strong_fail_frac"]),
                _fmt_num(r["worst_disp"]), _fmt_num(r["msr_peak_tail"]),
                f"{_fmt_num(r['self_ratio'])}({r.get('advisory_confidence','ok')})",
                str(r["n_s"]),
                f"{_fmt_num(r.get('s_rep'))}->{_fmt_num(r.get('e_rep'))}(n_e={r.get('n_e', 0)})",
                r.get("suggestion", "none"),
                f"{_fmt_num(r.get('sugg_self'))}/{_fmt_num(r.get('sugg_fidelity'))}",
            ]))
        lines.append("")
    if any(rr.get("e_confirmed") for rs in rows_by_mod.values() for rr in rs):
        lines.append("E_CONFIRMED rows have fail-frame (E) score-collapse confirmation; "
                     "others remain S-only latent.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run new tests + the existing digest/report tests (they assert the format)**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "digest or report"`
Expected: the two new tests PASS. If `test_digest_is_ascii_one_line_per_pipe` or `test_report_is_ascii_and_has_banner_and_rows` now fail because they assert on the old column text, update those assertions to match the new format (the digest is still one `|`-joined line; the report still has the banner + rows). Re-run until green.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): digest confirmed-count + report s_rep->e_rep column + note"
```

---

## Task 6: `_load_e_frames` + run() post-pass integration (office-run)

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (`_load_e_frames` near `_load_s_frames` ~line 698; `_recipe_row` row dict ~line 834-842; `run()` post-pass after the rows build loop ~line 865)
- Test: `poc/workflow_2/test_reregister_report.py` (no-data smoke); full-suite green; accuracy is office-gated.

**Interfaces:**
- Consumes: `_load_s_frames` pattern; `iter_msr_images`, `load_gray`, `load_cond`, `_tool_label`, `_route_modality_for_mod` (all already imported/defined); `_s_rep_score`, `_e_rep_score`, `_e_confirm`, `_risk_score`, `E_CONFIRM_ON`, `S_FLOOR`, `E_FLOOR`, `COLLAPSE_MARGIN` (Tasks 2-4).
- Produces: `_load_e_frames(assets, modality) -> list[np.ndarray]`; row fields `e_confirmed`/`s_rep`/`e_rep`/`n_e`/`_frame_results`; tier override to `E_CONFIRMED` in `run()`.

- [ ] **Step 1: Write the failing test (no-data smoke — keeps the office-gated path import-safe)**

```python
def test_run_no_data_still_returns_warning_with_e_confirm(monkeypatch, tmp_path):
    import poc.workflow_2.golden_reregister_report_cond as r
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))   # 빈 루트.
    monkeypatch.setenv("REREGISTER_E_CONFIRM", "1")
    assert r.run() == "[WARNING] no_data"
```

- [ ] **Step 2: Run test to verify it fails or errors**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py::test_run_no_data_still_returns_warning_with_e_confirm -q`
Expected: PASS already if no syntax error (run() returns no_data on empty root); this is the regression guard that the post-pass code (added in Step 3) doesn't break the no-data path. Proceed to Step 3, then re-run.

- [ ] **Step 3: Implement**

Add `_load_e_frames` right after `_load_s_frames` (~line 750):

```python
def _load_e_frames(assets, modality):
    """recipe 의 from_msr E(fail) 프레임을 modality 필터해 raw gray 리스트로 반환.

    E 는 crosshair/GT 가 없으므로 inpaint/clean 없이 raw gray 그대로 쓴다. modality 배정은
    S 와 동일하게 cond 기반(_route_modality_for_mod) — cond/ modality 미상 프레임은 skip.
    """
    from poc.workflow_3.align.assets import load_gray

    available_mods = {modality}
    result = []
    for msr_path in iter_msr_images(assets):
        if _tool_label(msr_path.name) != "E":
            continue   # S 프레임 제외(Phase 1 이 소비).
        try:
            gray_raw = load_gray(msr_path)
        except Exception as exc:
            print(f"[WARNING] msr(E) 로드 실패 {msr_path.name}: {exc}")
            continue
        cond = load_cond(msr_path)
        if _route_modality_for_mod(cond, available_mods, modality) is None:
            continue   # modality 미상/불일치 skip.
        result.append(gray_raw)
    return result
```

In `_recipe_row`, change the returned row dict to add Phase 2 default fields + keep `_frame_results`:

```python
    return {
        "recipe": f"{assets.class_name}/{assets.recipe_name}",
        "modality": modality, "tier": tier, "risk_score": _risk_score(tier, sev),
        "strong_fail_frac": strong["strong_fail_frac"], "worst_disp": strong["worst_disp"],
        "msr_peak_tail": medium["msr_peak_tail"], "self_ratio": self_ratio_val,
        "advisory_confidence": conf, "n_s": strong["n_s"],
        "suggestion": "none", "sugg_self": None, "sugg_fidelity": None,
        "e_confirmed": False, "s_rep": None, "e_rep": None, "n_e": 0,
        "_assets": assets, "_center": center_tpls, "_box": box_tpls,
        "_s_frames": s_frames, "_frame_results": frame_results,
    }
```

In `run()`, immediately after the `rows_by_mod` build loop (after line 865, before the C2 box-suggest block at line 867):

```python
    # Phase 2: E-frame confirmation post-pass (flagged row 만, upgrade-only).
    if E_CONFIRM_ON:
        print(f"[INFO] e_confirm on: S_FLOOR={S_FLOOR} E_FLOOR={E_FLOOR} "
              f"COLLAPSE_MARGIN={COLLAPSE_MARGIN}")
        for mod in rows_by_mod:
            for row in rows_by_mod[mod]:
                if row["tier"] == "NONE":
                    continue
                center_tpl = row["_center"].get(mod)
                if center_tpl is None:
                    continue
                s_rep = _s_rep_score(row.get("_frame_results", []))
                e_frames = _load_e_frames(row["_assets"], mod)
                e_rep = _e_rep_score(center_tpl, e_frames)
                row["s_rep"], row["e_rep"], row["n_e"] = s_rep, e_rep, len(e_frames)
                if _e_confirm(s_rep, e_rep):
                    row["e_confirmed"] = True
                    row["tier"] = "E_CONFIRMED"
                    sev = max(0.0, s_rep - e_rep)   # collapse 클수록 E_CONFIRMED 내 상위.
                    row["risk_score"] = _risk_score("E_CONFIRMED", sev)
```

- [ ] **Step 4: Run no-data smoke + full suite**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q`
Expected: all PASS (the no-data smoke confirms the post-pass doesn't break the empty-root path; engine-backed tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): _load_e_frames + run() E-confirm post-pass (Phase 2)"
```

---

## Office run (accuracy verification — Mac cannot do)

```
# golden_eval_config.py: GOLDEN_ROOT=<align_images_golden>, REREGISTER_E_CONFIRM=1
# (fast A/B) REREGISTER_MAX_RECIPES=20
uv run python poc/workflow_2/golden_reregister_report_cond.py
# -> [INFO] e_confirm on: S_FLOOR=.. E_FLOOR=.. COLLAPSE_MARGIN=..
# -> [DIGEST] reregister(S-only): om[screened N, strong S, confirmed C, w_sugg W, top ..] | sem[..]
# -> debug_images/golden_reregister_report_cond/{reregister_report.txt, digest.txt}
```

Relay: the `[INFO] e_confirm` line + the `[DIGEST]` line + 1-2 sample `E_CONFIRMED` rows (`s_rep->e_rep(n_e=K)`). Calibrate `S_FLOOR`/`E_FLOOR`/`COLLAPSE_MARGIN` from the S/E score spread (env A/B, code unchanged). Judgement: `E_CONFIRMED` = re-registration confirmed (top priority); `STRONG`/`MEDIUM`/`ADVISORY` remain S-only latent.
