# Re-registration Phase 3 — Rank-1 Distinctiveness Worklist — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `golden_reregister_report_cond.py` to join the per-recipe rank-1 already written by the consensus eval into the re-registration report, emitting an engineer-facing **worklist** (`reregister_worklist.{txt,json}`) that names recipes worst-first and tells the engineer *what kind of fix* each needs (FRESH_SNAPSHOT vs NEW_REGION).

**Architecture:** A pure consumer step inside `golden_reregister_report_cond.run()`. It reads the consensus driver's `summary.json` (`per_recipe` rows — already contain `rcp_rank1_rate` / `cons_rank1_rate`), normalizes the join key (consensus triplet `eqp/class/recipe` → reregister doublet `class/recipe`), classifies each recipe's fix type against a single distinctiveness floor, prioritizes worst-first, and writes the worklist next to the existing report. **No change to the consensus eval and no new CV computation.** New logic is pure helpers (Mac-testable); only path resolution touches I/O.

**Tech Stack:** Python 3.10+, stdlib only (`json`, `os`, `pathlib`). Tests are pytest-style in the existing `poc/workflow_2/test_reregister_report.py` (run with `uv run pytest`).

## Global Constraints

- Korean docstrings/comments. (CLAUDE.md)
- `[INFO]` / `[WARNING]` / `[ERROR]` print-based logging; never the `logging` module. (CLAUDE.md)
- **ASCII-only inside `print()` and any written digest/worklist string** — office console is cp949; no em-dash (U+2014). (CLAUDE.md)
- No `argparse` / CLI flags; no `from __future__` imports. (CLAUDE.md)
- Absolute imports `from poc.workflow_2.xxx import ...`; workflow_2 imports workflow_3, never the reverse. (CLAUDE.md)
- Config via env + `golden_eval_config`. Edit **only** the tracked `golden_eval_config.example.py` + `golden_eval_config_loader.py`; **never** the gitignored `golden_eval_config.py`. (memory)
- Commit directly to `main`, staging only this plan's touched files via explicit pathspec (no `git add -A` / `commit -a`); verify scope with `git show --stat`. (memory: pathspec commits for concurrent edits)
- Compare **rank-1**, not in_topk, in any A/B reasoning — in_topk is a ceiling; rank-1 is what ships. (memory)

## Confirmed facts (read before starting — these pin the implementation)

- **Consensus key** (`golden_consensus_eval_cond.py:333-338`): `_recipe_key(assets)` = `f"{assets.eqp_id}/{assets.class_name}/{assets.recipe_id}"` where `recipe_id` is the leaf recipe name. The `per_recipe` rows in `summary.json` carry this **triplet** in the `"recipe"` field.
- **Reregister key** (`golden_reregister_report_cond.py:918`): row `"recipe"` = `f"{a.class_name}/{a.recipe_name}"` — a **doublet**. The join normalizes the consensus triplet down to this doublet (drop the leading eqp segment).
- **Per-recipe row fields** (`align_similarity.py:1082-1093`): `{"recipe": <triplet>, "modality": "om"|"sem", "n_S_loo": int, "cons_pool_n": int, "rcp_rank1_rate": float, "cons_rank1_rate": float, ...}`.
- **Reregister row fields** available at the wire-in point (`_format_report`, lines 180-188): `recipe`, `modality` (the key of `rows_by_mod`), `tier`, `risk_score`, `worst_disp`, `suggestion` (the whitebox payload string, `"none"` when absent), `sugg_self`, `sugg_fidelity`.
- **TIER_WEIGHT** (line 36): `{"E_CONFIRMED": 3.0, "STRONG": 2.0, "MEDIUM": 1.0, "ADVISORY": 0.3, "NONE": 0.0}`.
- **`run()` write tail** (lines 1101-1104): `rows_by_mod` is already ranked (`_rank_rows`, lines 1088-1089) before `_format_report` / `_format_digest` write to `OUTPUT_ROOT`. The new worklist step inserts here.
- **OUTPUT_ROOT** (line 729): `DEBUG_IMAGE_DIR / "golden_reregister_report_cond"`. Consensus summaries live at `DEBUG_IMAGE_DIR / "golden_consensus_eval_cond" / <ts> / "summary.json"`.
- **Tests** are pytest-style (`def test_*`, `monkeypatch`, `tmp_path`) in `poc/workflow_2/test_reregister_report.py`; run via `uv run pytest poc/workflow_2/test_reregister_report.py`.
- **Config bridge** pattern (`golden_eval_config_loader.py:97-114`): each `REREGISTER_*` knob is `os.environ.setdefault("REREGISTER_X", str(REREGISTER_X))` inside `seed_env()`, with an import-fallback default near line 72.

## Calibration features folded in from the 06-25 spec review (sharpen output, not structure)

- **FRESH_SNAPSHOT hint** (review finding 1): `cons_rank1` is a *median* over crops, but the fix is a *single* re-snapshot, so FRESH_SNAPSHOT can over-promise. The worklist therefore tags FRESH_SNAPSHOT rows with a hint that enabling the already-implemented consensus-live-correction may be a cheaper/better fix than re-registration. (Implemented in Task 5.)
- **Rank-1 histogram** (review finding 2): a single `distinct_floor` does two jobs; calibrating it needs the *distribution*, not the mean. The run prints a per-modality rank-1 histogram so the first office run can tune `REREGISTER_DISTINCT_FLOOR`. (Implemented in Task 5/6.)

---

### Task 1: Join-key normalizer — `_normalize_consensus_key`

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (add pure helper in the "순수 헬퍼" block, after `_rank_rows`, ~line 160)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces: `_normalize_consensus_key(rec: str) -> str` — maps a consensus `per_recipe["recipe"]` value to the reregister doublet `class/recipe`. A triplet `eqp/class/recipe` returns its last two segments joined by `/`; a value already a doublet (or shorter) returns unchanged (stripped). Used by `_build_rank1_lookup` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
def test_normalize_consensus_key_triplet_to_doublet():
    # consensus 키는 eqp/class/recipe 트리플렛 → reregister 의 class/recipe 더블렛으로.
    assert rr._normalize_consensus_key("EQP01/CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_doublet_passthrough():
    # 이미 더블렛이면 그대로(정규화 멱등).
    assert rr._normalize_consensus_key("CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_extra_segments_keeps_last_two():
    # 혹시 4단 이상이어도 마지막 두 세그먼트(class/recipe)만.
    assert rr._normalize_consensus_key("F/EQP01/CLSA/REC1") == "CLSA/REC1"


def test_normalize_consensus_key_strips_whitespace():
    assert rr._normalize_consensus_key("  EQP01/CLSA/REC1  ") == "CLSA/REC1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k normalize_consensus_key -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_normalize_consensus_key'`

- [ ] **Step 3: Write minimal implementation**

Add after `_rank_rows` (~line 160):

```python
def _normalize_consensus_key(rec):
    """consensus per_recipe['recipe'](eqp/class/recipe 트리플렛)를 reregister 의
    class/recipe 더블렛으로 정규화한다. 조인 키 정렬용 — 마지막 두 세그먼트만 취한다.

    consensus 는 _recipe_key(assets)=eqp/class/recipe 로 키를 잡고(장비 leaf 충돌 방지),
    reregister 는 class/recipe 로 잡으므로 eqp 접두를 떼야 조인이 붙는다. 이미 더블렛이면
    멱등(그대로), 세그먼트가 더 많아도 class/recipe 두 개만 남긴다.
    """
    parts = [p for p in str(rec).strip().split("/") if p != ""]
    if len(parts) <= 2:
        return "/".join(parts)
    return "/".join(parts[-2:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k normalize_consensus_key -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): pin rank1 join key (consensus triplet -> reregister doublet)"
```

---

### Task 2: Consensus rank-1 lookup — `_build_rank1_lookup` (pure) + `_load_consensus_rank1` (I/O)

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (pure helper after Task 1's helper; I/O reader near the other `_load_*` readers, ~line 773)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `_normalize_consensus_key` (Task 1).
- Produces:
  - `_build_rank1_lookup(per_recipe: list) -> dict` — pure. Maps `(doublet_recipe, modality) -> {"rcp_rank1": float, "cons_rank1": float, "n_S_loo": int, "cons_pool_n": int}`. On collision (two triplets → same doublet+modality) keeps the **worst** (lowest `rcp_rank1`) — conservative for a re-registration worklist — and the running module counter `_LAST_JOIN_COLLISIONS` records how many rows collapsed (read by Task 6's coverage line). Rows missing `rcp_rank1_rate`/`cons_rank1_rate`/`modality` are skipped.
  - `_load_consensus_rank1(path=None) -> dict` — I/O. Resolves the consensus `summary.json` (explicit `path` arg → `REREGISTER_CONSENSUS_SUMMARY` env → newest `DEBUG_IMAGE_DIR/golden_consensus_eval_cond/<ts>/summary.json`), reads `per_recipe`, returns `_build_rank1_lookup(...)`. Returns `{}` (no raise) when no file is found or JSON is unreadable. Used by Task 6's wire-in.

- [ ] **Step 1: Write the failing test**

```python
def test_build_rank1_lookup_basic_and_key_normalization():
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5, "n_S_loo": 6, "cons_pool_n": 8},
        {"recipe": "EQP01/CLSA/REC1", "modality": "om",
         "rcp_rank1_rate": 0.9, "cons_rank1_rate": 0.95, "n_S_loo": 4, "cons_pool_n": 4},
    ]
    lk = rr._build_rank1_lookup(per_recipe)
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.4
    assert lk[("CLSA/REC1", "sem")]["cons_rank1"] == 0.5
    assert lk[("CLSA/REC1", "om")]["rcp_rank1"] == 0.9


def test_build_rank1_lookup_collision_keeps_worst():
    # 두 장비의 같은 class/recipe·modality 가 더블렛으로 충돌 -> 최저 rcp_rank1(보수적) 유지.
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.6, "cons_rank1_rate": 0.6, "n_S_loo": 5, "cons_pool_n": 5},
        {"recipe": "EQP02/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.3, "cons_rank1_rate": 0.4, "n_S_loo": 5, "cons_pool_n": 5},
    ]
    lk = rr._build_rank1_lookup(per_recipe)
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.3


def test_build_rank1_lookup_skips_incomplete_rows():
    per_recipe = [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem"},  # no rates
        {"recipe": "EQP01/CLSA/REC2", "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5},  # no modality
    ]
    assert rr._build_rank1_lookup(per_recipe) == {}


def test_load_consensus_rank1_empty_on_missing(tmp_path):
    missing = tmp_path / "nope" / "summary.json"
    assert rr._load_consensus_rank1(str(missing)) == {}


def test_load_consensus_rank1_reads_fixture(tmp_path):
    import json
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps({"per_recipe": [
        {"recipe": "EQP01/CLSA/REC1", "modality": "sem",
         "rcp_rank1_rate": 0.4, "cons_rank1_rate": 0.5, "n_S_loo": 6, "cons_pool_n": 8},
    ]}), encoding="utf-8")
    lk = rr._load_consensus_rank1(str(summ))
    assert lk[("CLSA/REC1", "sem")]["rcp_rank1"] == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k "rank1_lookup or load_consensus_rank1" -v`
Expected: FAIL with `AttributeError: ... has no attribute '_build_rank1_lookup'`

- [ ] **Step 3: Write minimal implementation**

Add the module counter near the top constants (after `TIER_WEIGHT`, ~line 37):

```python
# 조인 충돌(같은 class/recipe·modality 가 여러 장비에서 collapse)된 consensus row 수 — 커버리지 로그용.
_LAST_JOIN_COLLISIONS = 0
```

Add the pure helper after `_normalize_consensus_key`:

```python
def _build_rank1_lookup(per_recipe):
    """consensus per_recipe 리스트 -> {(class/recipe, modality): {rcp_rank1, cons_rank1,
    n_S_loo, cons_pool_n}}. 순수 함수(테스트용). 키는 _normalize_consensus_key 로 더블렛 정규화.

    충돌(두 장비의 같은 class/recipe·modality 가 더블렛으로 합쳐짐)은 최저 rcp_rank1(가장
    변별력 낮은 쪽)을 유지한다 — 재등록 worklist 는 보수적으로 flag 하는 게 안전. 합쳐진 수는
    모듈 카운터 _LAST_JOIN_COLLISIONS 에 기록(커버리지 라인에서 노출).
    rcp_rank1_rate/cons_rank1_rate/modality 가 없는 row 는 건너뛴다.
    """
    global _LAST_JOIN_COLLISIONS
    _LAST_JOIN_COLLISIONS = 0
    lookup = {}
    for row in per_recipe:
        mod = row.get("modality")
        rcp = row.get("rcp_rank1_rate")
        cons = row.get("cons_rank1_rate")
        if mod is None or rcp is None or cons is None:
            continue
        key = (_normalize_consensus_key(row.get("recipe", "")), mod)
        rec = {
            "rcp_rank1": float(rcp),
            "cons_rank1": float(cons),
            "n_S_loo": int(row.get("n_S_loo", 0)),
            "cons_pool_n": int(row.get("cons_pool_n", 0)),
        }
        if key in lookup:
            _LAST_JOIN_COLLISIONS += 1
            if rec["rcp_rank1"] < lookup[key]["rcp_rank1"]:
                lookup[key] = rec        # 최저(worst) 유지.
        else:
            lookup[key] = rec
    return lookup
```

Add the I/O reader near the other `_load_*` readers (~line 773):

```python
def _resolve_consensus_summary(path=None):
    """consensus summary.json 경로 해석: 명시 인자 -> REREGISTER_CONSENSUS_SUMMARY env ->
    DEBUG_IMAGE_DIR/golden_consensus_eval_cond/<ts>/summary.json 중 최신 ts. 없으면 None.
    """
    if path:
        p = Path(path)
        return p if p.exists() else None
    env = os.getenv("REREGISTER_CONSENSUS_SUMMARY")
    if env:
        p = Path(env)
        return p if p.exists() else None
    base = DEBUG_IMAGE_DIR / "golden_consensus_eval_cond"
    if not base.is_dir():
        return None
    cands = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for d in cands:
        s = d / "summary.json"
        if s.exists():
            return s
    return None


def _load_consensus_rank1(path=None):
    """consensus summary.json 을 읽어 (class/recipe, modality) -> rank1 dict 로. 파일 부재/
    파싱 실패 시 빈 dict(graceful degrade — 모든 recipe NO_DATA). consensus eval 무수정.
    """
    summ = _resolve_consensus_summary(path)
    if summ is None:
        print("[WARNING] rank1-join: consensus summary.json not found (all recipes -> NO_DATA)")
        return {}
    try:
        data = json.loads(summ.read_text(encoding="utf-8"))
    except Exception as e:   # noqa: BLE001 - 파싱 실패는 graceful degrade.
        print(f"[WARNING] rank1-join: failed to read {summ} ({e}); NO_DATA fallback")
        return {}
    return _build_rank1_lookup(data.get("per_recipe", []))
```

**Import note (verified):** `os` is imported (line 9) and `Path` is imported (line 701, in scope for the I/O helpers added at ~773). But `json` is **not** imported anywhere in this module — add `import json` to the top import block (after `import os`, line 9). Both `_load_consensus_rank1` (this task) and the `run()` wire-in (Task 6) need it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k "rank1_lookup or load_consensus_rank1" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): load+build consensus rank1 lookup with collision handling"
```

---

### Task 3: Fix-type classifier — `_classify_fix`

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (pure helper block)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces: `_classify_fix(rcp_rank1, cons_rank1, *, distinct_floor) -> str` — pure. Returns one of `"OK"`, `"FRESH_SNAPSHOT"`, `"NEW_REGION"`, `"NO_DATA"`. `None` for `rcp_rank1` (recipe absent from the consensus join) → `"NO_DATA"`. Used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
def test_classify_fix_ok_when_rcp_distinctive():
    assert rr._classify_fix(0.8, 0.4, distinct_floor=0.7) == "OK"   # rcp >= floor -> OK (cons 무시)


def test_classify_fix_fresh_snapshot_when_region_fine():
    assert rr._classify_fix(0.5, 0.8, distinct_floor=0.7) == "FRESH_SNAPSHOT"


def test_classify_fix_new_region_when_region_ambiguous():
    assert rr._classify_fix(0.4, 0.5, distinct_floor=0.7) == "NEW_REGION"


def test_classify_fix_no_data_when_rcp_none():
    assert rr._classify_fix(None, None, distinct_floor=0.7) == "NO_DATA"
    assert rr._classify_fix(None, 0.9, distinct_floor=0.7) == "NO_DATA"


def test_classify_fix_floor_is_inclusive():
    # rcp_rank1 == floor 이면 OK(>=). cons == floor 이면 FRESH(>=).
    assert rr._classify_fix(0.7, 0.0, distinct_floor=0.7) == "OK"
    assert rr._classify_fix(0.6, 0.7, distinct_floor=0.7) == "FRESH_SNAPSHOT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k classify_fix -v`
Expected: FAIL with `AttributeError: ... has no attribute '_classify_fix'`

- [ ] **Step 3: Write minimal implementation**

```python
def _classify_fix(rcp_rank1, cons_rank1, *, distinct_floor):
    """등록 key 와 영역의 rank-1 변별력으로 fix 유형을 분류한다(순수). spec 4.2 표.

    rcp_rank1 = 등록 snapshot 의 success-frame rank-1(낮으면 재등록), cons_rank1 = 영역(중앙값
    consensus)의 rank-1(낮으면 영역 자체가 모호 -> 같은 자리 재촬영 무용). 둘 다 floor 이상 needed.
    rcp_rank1=None(조인 미스) -> NO_DATA.
    """
    if rcp_rank1 is None:
        return "NO_DATA"
    if rcp_rank1 >= distinct_floor:
        return "OK"
    if cons_rank1 is not None and cons_rank1 >= distinct_floor:
        return "FRESH_SNAPSHOT"
    return "NEW_REGION"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k classify_fix -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): fix-type classifier (OK/FRESH_SNAPSHOT/NEW_REGION/NO_DATA)"
```

---

### Task 4: Worklist priority — `_worklist_priority`

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (add `FIX_WEIGHT` constant near `TIER_WEIGHT`; pure helper)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces:
  - `FIX_WEIGHT = {"NEW_REGION": 3.0, "FRESH_SNAPSHOT": 2.0, "OK": 0.0, "NO_DATA": 0.0}` — module constant.
  - `_worklist_priority(fix_type, rcp_rank1, tier_weight) -> float` — pure. For a classified row (`rcp_rank1` not `None`): `FIX_WEIGHT[fix_type] * (1 - rcp_rank1) + tier_weight + _RANK1_BACKED_BONUS`. For `NO_DATA` (`rcp_rank1 is None`): `tier_weight` alone (no bonus), so it sorts below any rank-1-backed flag of equal tier. Used by Task 5 sorting.

- [ ] **Step 1: Write the failing test**

```python
def test_worklist_priority_new_region_above_fresh_above_ok_at_equal_rcp():
    # 같은 rcp_rank1·tier 에서 NEW_REGION > FRESH_SNAPSHOT > OK.
    tw = rr.TIER_WEIGHT["STRONG"]
    p_new = rr._worklist_priority("NEW_REGION", 0.5, tw)
    p_fresh = rr._worklist_priority("FRESH_SNAPSHOT", 0.5, tw)
    p_ok = rr._worklist_priority("OK", 0.5, tw)
    assert p_new > p_fresh > p_ok


def test_worklist_priority_lower_rcp_rank1_ranks_higher():
    tw = rr.TIER_WEIGHT["NONE"]
    assert rr._worklist_priority("NEW_REGION", 0.2, tw) > rr._worklist_priority("NEW_REGION", 0.6, tw)


def test_worklist_priority_no_data_below_equal_tier_backed_flag():
    # NO_DAT(rcp_rank1=None) 은 같은 tier 의 rank-1-backed flag 보다 아래.
    tw = rr.TIER_WEIGHT["MEDIUM"]
    p_nodata = rr._worklist_priority("NO_DATA", None, tw)
    p_ok_backed = rr._worklist_priority("OK", 0.9, tw)   # rank-1-backed, 같은 tier
    assert p_ok_backed > p_nodata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k worklist_priority -v`
Expected: FAIL with `AttributeError: ... has no attribute '_worklist_priority'`

- [ ] **Step 3: Write minimal implementation**

Add the constant after `TIER_WEIGHT` (~line 37):

```python
# fix 유형 가중(worklist 정렬) — 어려운/고가치 fix 가 위로. _worklist_priority 에서 사용.
FIX_WEIGHT = {"NEW_REGION": 3.0, "FRESH_SNAPSHOT": 2.0, "OK": 0.0, "NO_DATA": 0.0}
# rank-1-backed row 가 같은 tier 의 NO_DATA 보다 위에 오도록 하는 미세 보너스.
_RANK1_BACKED_BONUS = 0.001
```

Add the pure helper:

```python
def _worklist_priority(fix_type, rcp_rank1, tier_weight):
    """worklist 정렬 우선순위(순수). 1차 severity = (1 - rcp_rank1) * fix 가중 + tier 코로보레이션.

    rcp_rank1=None(NO_DATA) 은 tier_weight 단독으로 산정 -> 같은 tier 의 rank-1-backed flag
    아래로 내려간다(_RANK1_BACKED_BONUS 차이). 클수록 worst -> worklist 위.
    """
    if rcp_rank1 is None:
        return float(tier_weight)
    severity = 1.0 - float(rcp_rank1)
    return FIX_WEIGHT.get(fix_type, 0.0) * severity + float(tier_weight) + _RANK1_BACKED_BONUS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k worklist_priority -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): worklist priority (fix-weight x severity + tier corroboration)"
```

---

### Task 5: Worklist assembly + format + histogram — `_worklist_rows`, `_format_worklist`, `_rank1_histogram`

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (assembly + format helpers in the format block)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `_classify_fix`, `_worklist_priority`, `FIX_WEIGHT`, `TIER_WEIGHT` (Tasks 3-4); reregister row dicts (`recipe`, `tier`, `suggestion`).
- Produces:
  - `_worklist_rows(rows_by_mod: dict, rank1_lookup: dict, *, distinct_floor) -> list[dict]` — pure. For each row in each modality, joins on `(row["recipe"], mod)`, classifies, prices, and builds a worklist row: `{rank?, recipe, modality, rcp_rank1, cons_rank1, fix_type, suggested_whitebox, tier, priority, hint}`. `suggested_whitebox` = `row.get("suggestion", "none")` (payload for `NEW_REGION`). `hint` = the consensus-live-correction note for `FRESH_SNAPSHOT`, else `""`. Sorted worst-first by `priority` desc. Returns ALL rows (caller filters OK for the TXT body).
  - `_format_worklist(worklist_rows: list) -> str` — pure, ASCII. Aligned-column TXT; body lists `NEW_REGION` / `FRESH_SNAPSHOT` / `NO_DATA` worst-first (OK excluded from the body, summarized in the count line). One column header line. ASCII only.
  - `_rank1_histogram(rank1_lookup: dict) -> str` — pure, ASCII. Per-modality 10-bucket (`0.0-0.1` ... `0.9-1.0`) counts of `rcp_rank1`, one line per modality, for floor calibration.

- [ ] **Step 1: Write the failing test**

```python
def test_worklist_rows_joins_classifies_and_sorts(_distinct=0.7):
    rows_by_mod = {
        "sem": [
            {"recipe": "CLSA/REC1", "tier": "STRONG", "suggestion": "box=10,10,40,40"},
            {"recipe": "CLSA/REC2", "tier": "NONE", "suggestion": "none"},
        ],
        "om": [
            {"recipe": "CLSA/REC3", "tier": "MEDIUM", "suggestion": "none"},
        ],
    }
    lookup = {
        ("CLSA/REC1", "sem"): {"rcp_rank1": 0.3, "cons_rank1": 0.4, "n_S_loo": 6, "cons_pool_n": 8},
        ("CLSA/REC2", "sem"): {"rcp_rank1": 0.5, "cons_rank1": 0.9, "n_S_loo": 5, "cons_pool_n": 5},
        # REC3 (om) 없음 -> NO_DATA
    }
    wl = rr._worklist_rows(rows_by_mod, lookup, distinct_floor=_distinct)
    by_rec = {(w["recipe"], w["modality"]): w for w in wl}
    assert by_rec[("CLSA/REC1", "sem")]["fix_type"] == "NEW_REGION"
    assert by_rec[("CLSA/REC1", "sem")]["suggested_whitebox"] == "box=10,10,40,40"
    assert by_rec[("CLSA/REC2", "sem")]["fix_type"] == "FRESH_SNAPSHOT"
    assert by_rec[("CLSA/REC3", "om")]["fix_type"] == "NO_DATA"
    # worst-first: NEW_REGION(REC1) 이 FRESH(REC2) 보다 앞.
    order = [(w["recipe"], w["modality"]) for w in wl]
    assert order.index(("CLSA/REC1", "sem")) < order.index(("CLSA/REC2", "sem"))


def test_worklist_rows_fresh_snapshot_has_live_correction_hint():
    rows_by_mod = {"om": [{"recipe": "CLSA/REC1", "tier": "NONE", "suggestion": "none"}]}
    lookup = {("CLSA/REC1", "om"): {"rcp_rank1": 0.5, "cons_rank1": 0.9,
                                    "n_S_loo": 4, "cons_pool_n": 4}}
    wl = rr._worklist_rows(rows_by_mod, lookup, distinct_floor=0.7)
    assert "live-correction" in wl[0]["hint"]


def test_format_worklist_is_ascii_and_excludes_ok_body():
    rows = [
        {"recipe": "CLSA/REC1", "modality": "sem", "rcp_rank1": 0.3, "cons_rank1": 0.4,
         "fix_type": "NEW_REGION", "suggested_whitebox": "box=1,2,3,4", "tier": "STRONG",
         "priority": 2.1, "hint": ""},
        {"recipe": "CLSA/REC9", "modality": "om", "rcp_rank1": 0.95, "cons_rank1": 0.95,
         "fix_type": "OK", "suggested_whitebox": "none", "tier": "NONE",
         "priority": 0.0, "hint": ""},
    ]
    txt = rr._format_worklist(rows)
    assert txt.isascii()
    assert "NEW_REGION" in txt and "CLSA/REC1" in txt
    assert "CLSA/REC9" not in txt          # OK 는 body 제외
    assert "—" not in txt             # em-dash 금지


def test_rank1_histogram_ascii_and_per_modality():
    lookup = {
        ("CLSA/REC1", "sem"): {"rcp_rank1": 0.05, "cons_rank1": 0.4, "n_S_loo": 6, "cons_pool_n": 8},
        ("CLSA/REC2", "sem"): {"rcp_rank1": 0.55, "cons_rank1": 0.5, "n_S_loo": 5, "cons_pool_n": 5},
        ("CLSA/REC3", "om"):  {"rcp_rank1": 0.95, "cons_rank1": 0.9, "n_S_loo": 4, "cons_pool_n": 4},
    }
    h = rr._rank1_histogram(lookup)
    assert h.isascii()
    assert "sem" in h and "om" in h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k "worklist_rows or format_worklist or rank1_histogram" -v`
Expected: FAIL with `AttributeError: ... has no attribute '_worklist_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
# FRESH_SNAPSHOT row 안내: cons_rank1 은 중앙값이라 단일 재촬영은 그 denoising 을 못 받는다.
# 런타임 consensus-live-correction(이미 구현)을 켜는 편이 더 싸고 나은 fix 일 수 있음.
_FRESH_HINT = "region ok by median; enabling consensus-live-correction may beat a single re-snapshot"


def _worklist_rows(rows_by_mod, rank1_lookup, *, distinct_floor):
    """rows_by_mod 를 consensus rank1 과 조인 -> fix 유형 분류 + 우선순위 -> worst-first 리스트(순수).

    각 worklist row: recipe, modality, rcp_rank1, cons_rank1, fix_type, suggested_whitebox,
    tier, priority, hint. suggested_whitebox 는 NEW_REGION 의 payload(Phase1 의 row['suggestion']).
    FRESH_SNAPSHOT 은 consensus-live-correction hint 부여. OK 포함(호출측이 TXT body 에서 제외).
    """
    out = []
    for mod, rows in rows_by_mod.items():
        for r in rows:
            info = rank1_lookup.get((r["recipe"], mod))
            rcp_r1 = info["rcp_rank1"] if info else None
            cons_r1 = info["cons_rank1"] if info else None
            fix = _classify_fix(rcp_r1, cons_r1, distinct_floor=distinct_floor)
            tw = TIER_WEIGHT.get(r.get("tier", "NONE"), 0.0)
            out.append({
                "recipe": r["recipe"],
                "modality": mod,
                "rcp_rank1": rcp_r1,
                "cons_rank1": cons_r1,
                "fix_type": fix,
                "suggested_whitebox": r.get("suggestion", "none"),
                "tier": r.get("tier", "NONE"),
                "priority": _worklist_priority(fix, rcp_r1, tw),
                "hint": _FRESH_HINT if fix == "FRESH_SNAPSHOT" else "",
            })
    out.sort(key=lambda w: w["priority"], reverse=True)
    return out


def _format_worklist(worklist_rows):
    """worklist 를 ASCII 정렬 테이블 텍스트로. body 는 NEW_REGION/FRESH_SNAPSHOT/NO_DATA
    worst-first(OK 는 카운트 줄에만). em-dash 금지(cp949).
    """
    body = [w for w in worklist_rows if w["fix_type"] != "OK"]
    n_ok = sum(1 for w in worklist_rows if w["fix_type"] == "OK")
    lines = ["=== Re-registration worklist (rank-1 distinctiveness) ==="]
    counts = {}
    for w in worklist_rows:
        counts[w["fix_type"]] = counts.get(w["fix_type"], 0) + 1
    lines.append("counts: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append("rank recipe modality rcp_r1 cons_r1 fix_type tier whitebox priority hint")
    for i, w in enumerate(body, 1):
        lines.append(" ".join([
            str(i), w["recipe"], w["modality"],
            _fmt_num(w["rcp_rank1"]), _fmt_num(w["cons_rank1"]),
            w["fix_type"], w["tier"], w["suggested_whitebox"],
            f"{w['priority']:.3f}", (w["hint"] or "-"),
        ]))
    lines.append(f"(OK rows omitted from body: {n_ok})")
    return "\n".join(lines) + "\n"


def _rank1_histogram(rank1_lookup):
    """rcp_rank1 분포를 modality별 10구간 ASCII 히스토그램으로 — distinct_floor 보정용(순수)."""
    buckets = {"om": [0] * 10, "sem": [0] * 10}
    for (_rec, mod), info in rank1_lookup.items():
        if mod not in buckets:
            buckets[mod] = [0] * 10
        idx = min(9, max(0, int(float(info["rcp_rank1"]) * 10)))
        buckets[mod][idx] += 1
    lines = ["rank1-hist (rcp_rank1, buckets 0.0..1.0):"]
    for mod in sorted(buckets):
        lines.append(f"  {mod}: " + " ".join(str(c) for c in buckets[mod]))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k "worklist_rows or format_worklist or rank1_histogram" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): worklist assembly+format+rank1 histogram (live-correction hint)"
```

---

### Task 6: Wire into `run()` + `REREGISTER_DISTINCT_FLOOR` env + config bridge + verify

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py` (module const + `run()` write tail ~line 1102)
- Modify: `poc/workflow_2/golden_eval_config.example.py` (new knob in the reregister block)
- Modify: `poc/workflow_2/golden_eval_config_loader.py` (`seed_env` bridge + import-fallback default)
- Test: `poc/workflow_2/test_reregister_report.py` (config-bridge test + a no-data wire-in smoke test)

**Interfaces:**
- Consumes: `_load_consensus_rank1`, `_worklist_rows`, `_format_worklist`, `_rank1_histogram`, `_LAST_JOIN_COLLISIONS` (Tasks 2-5).
- Produces: `DISTINCT_FLOOR` module constant (`float(os.getenv("REREGISTER_DISTINCT_FLOOR", "0.70"))`); `run()` writes `reregister_worklist.txt` + `reregister_worklist.json` to `OUTPUT_ROOT` and prints the `[INFO] rank1-join: matched M/N` coverage line + histogram + extended digest.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_env_bridges_distinct_floor(monkeypatch):
    monkeypatch.delenv("REREGISTER_DISTINCT_FLOOR", raising=False)
    cfg.seed_env()
    assert os.environ["REREGISTER_DISTINCT_FLOOR"] == "0.7"


def test_seed_env_respects_existing_distinct_floor(monkeypatch):
    monkeypatch.setenv("REREGISTER_DISTINCT_FLOOR", "0.55")
    cfg.seed_env()
    assert os.environ["REREGISTER_DISTINCT_FLOOR"] == "0.55"


def test_join_coverage_line_format():
    # M/N 커버리지 문자열은 ASCII 한 줄.
    line = rr._join_coverage_line(3, 5, collisions=1)
    assert line.isascii()
    assert "3/5" in line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -k "distinct_floor or join_coverage" -v`
Expected: FAIL (`KeyError: 'REREGISTER_DISTINCT_FLOOR'` / `AttributeError: ... '_join_coverage_line'`)

- [ ] **Step 3: Write the implementation**

In `golden_reregister_report_cond.py`, add the module const near the other `REREGISTER_*` floors (~line 34):

```python
# rank-1 변별력 floor — rcp_rank1/cons_rank1 의 OK/FRESH/NEW 분기 임계(office 보정 대상).
# SEM rank-1 이 ~0.5 군집(ADR 0006)이라 0.70 이면 대부분 SEM key 가 flag 된다. 히스토그램 보고 튜닝.
DISTINCT_FLOOR = float(os.getenv("REREGISTER_DISTINCT_FLOOR", "0.70"))
```

Add the coverage-line helper (pure, in the format block):

```python
def _join_coverage_line(matched, total, *, collisions=0):
    """rank1 조인 커버리지 한 줄(ASCII). matched/total + 충돌 수. M~0 이면 키 불일치 경보."""
    return f"[INFO] rank1-join: matched {matched}/{total} report recipes to consensus rows (collisions={collisions})"
```

Wire into `run()` right after `_format_report` is written (after line 1102, before the digest write). The rows are already ranked here:

```python
    # ---- Phase 3: rank-1 distinctiveness worklist (consensus summary.json 소비) ----
    rank1_lookup = _load_consensus_rank1()
    worklist = _worklist_rows(rows_by_mod, rank1_lookup, distinct_floor=DISTINCT_FLOOR)
    total_recipes = sum(len(rs) for rs in rows_by_mod.values())
    matched = sum(1 for w in worklist if w["rcp_rank1"] is not None)
    print(_join_coverage_line(matched, total_recipes, collisions=_LAST_JOIN_COLLISIONS))
    print(f"[INFO] distinct_floor={DISTINCT_FLOOR} (env REREGISTER_DISTINCT_FLOOR)")
    print(_rank1_histogram(rank1_lookup))
    (OUTPUT_ROOT / "reregister_worklist.txt").write_text(_format_worklist(worklist), encoding="utf-8")
    (OUTPUT_ROOT / "reregister_worklist.json").write_text(
        json.dumps(worklist, ensure_ascii=False, indent=2), encoding="utf-8")
    _wl_counts = {}
    for w in worklist:
        _wl_counts[w["fix_type"]] = _wl_counts.get(w["fix_type"], 0) + 1
    print("[DIGEST] worklist " + " ".join(f"{k}={v}" for k, v in sorted(_wl_counts.items()))
          + f" join={matched}/{total_recipes}")
```

In `golden_eval_config.example.py`, add to the `re-registration 리포트` block (after `REREGISTER_GT_TOL_NORM`):

```python
# rank-1 변별력 floor(Phase 3 worklist). rcp_rank1>=floor=OK, <floor 이고 cons_rank1>=floor=
# FRESH_SNAPSHOT, 둘 다 <floor=NEW_REGION. SEM ~0.5 군집이라 0.70 이 대부분 SEM flag. office 보정.
REREGISTER_DISTINCT_FLOOR = 0.70
```

In `golden_eval_config_loader.py`: add `REREGISTER_DISTINCT_FLOOR` to the import near line 45 (alongside `REREGISTER_MAX_RECIPES`), its fallback near line 73, and the `setdefault` bridge near line 105:

```python
# (import block, ~line 45) — extend the existing REREGISTER import:
        from poc.workflow_2.golden_eval_config import REREGISTER_DISTINCT_FLOOR
    except ImportError:
        REREGISTER_DISTINCT_FLOOR = 0.70
# (fallback block, ~line 73):
    REREGISTER_DISTINCT_FLOOR = 0.70
# (seed_env, after the REREGISTER_GT_TOL_NORM setdefault ~line 105):
    os.environ.setdefault("REREGISTER_DISTINCT_FLOOR", str(REREGISTER_DISTINCT_FLOOR))
```

> Match the existing try/except import grouping in the loader — add `REREGISTER_DISTINCT_FLOOR` to whichever `from ... import` group keeps the import/fallback/seed three places consistent (see lines 45-48, 73, 101-105 for the pattern).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -v`
Expected: PASS (all — the full file, including pre-existing tests)

- [ ] **Step 5: Mac verification (no office data — import + no-data run)**

```bash
uv run python -c "import poc.workflow_2.golden_reregister_report_cond"   # py import OK
uv run python -c "import poc.workflow_2.golden_eval_config_loader as c; c.seed_env(); import os; print(os.environ['REREGISTER_DISTINCT_FLOOR'])"
```
Expected: imports clean; prints `0.7`. (A full `run()` needs the align_images tree + a consensus summary — office-gated; on Mac with no consensus summary the join degrades to all-`NO_DATA` with the `[WARNING] rank1-join: ... not found` line, which is the correct graceful-degrade path.)

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config_loader.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(reregister): wire rank-1 worklist into run() + REREGISTER_DISTINCT_FLOOR knob"
```

---

## Office run (accuracy — after implementation, office PC only)

```bash
# prerequisite: consensus summary.json must exist (run the consensus eval first if stale)
uv run python poc/workflow_2/golden_consensus_eval_cond.py        # writes per_recipe rank-1
uv run python poc/workflow_2/golden_reregister_report_cond.py     # consumes it -> worklist
```
Relay back: the `[INFO] rank1-join: matched M/N` coverage line (must be near 1 — M~0 means the join key broke), the `rank1-hist` lines, the `[DIGEST] worklist ...` counts, and the top ~10 `reregister_worklist.txt` rows. **Decision:** confirm join coverage high → read the `NEW_REGION` SEM rows (highest-value re-registration targets) → calibrate `REREGISTER_DISTINCT_FLOOR` against the observed rank-1 histogram if the OK/FRESH/NEW split looks wrong (SEM rank-1 clustered ~0.5 per ADR 0006, so 0.70 should flag most SEM keys).

---

## Self-Review

**Spec coverage:**
- §3 data source (read `per_recipe`, no consensus-eval change) → Task 2 (`_load_consensus_rank1` reads, never writes).
- §4.1 input resolution (env → newest ts) → Task 2 `_resolve_consensus_summary`.
- §4.2 fix-type table (all 4 cells incl. `None`→NO_DATA) → Task 3 `_classify_fix`.
- §4.3 priority (fix-weight × severity + tier; NO_DATA below equal-tier backed) → Task 4 `_worklist_priority`.
- §4.4 worklist output (txt+json, columns, whitebox payload, extended digest) → Task 5 + Task 6 wire-in.
- §5 join-key risk (pin format + loud coverage line) → Task 1 + Task 6 `_join_coverage_line` + `_LAST_JOIN_COLLISIONS`.
- §6 calibration lever `REREGISTER_DISTINCT_FLOOR` default 0.70 → Task 6 const + bridge.
- §7 testing (pure helpers + tmp fixture + no-data) → Tasks 1-6 tests + Task 6 Step 5.
- §8 constraints → Global Constraints block.
- Review finding 1 (FRESH over-promise) → Task 5 `_FRESH_HINT`.
- Review finding 2 (histogram for calibration) → Task 5 `_rank1_histogram` + Task 6 print.

**Beyond-spec decision (documented):** the **collision** created by normalizing triplet→doublet (two tools, same recipe) — not in the spec — is handled in Task 2 (keep worst rcp_rank1, count collisions, surface in the coverage line). This is the one real implementation hazard §5 only half-named.

**Type consistency:** `rcp_rank1`/`cons_rank1` are `float|None` everywhere; `_classify_fix`/`_worklist_priority` accept `None`; lookup keys are `(doublet_str, modality_str)` tuples consistently in Tasks 2 and 5; `suggested_whitebox` reads `row["suggestion"]` (the confirmed Phase-1 field name).
