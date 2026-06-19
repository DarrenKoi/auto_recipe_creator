# OM/SEM modality-split eval (Phase 1: 증거 eval + verdict) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** combined eval(`golden_combined_eval_cond.py`)에 per-modality(OM/SEM) 실패유형 히스토그램 + per-modality Youden 분리 + split verdict 를 추가해, 오피스 1회 실행으로 "OM/SEM CV 정책을 쪼갤지"가 증거 기반으로 떨어지게 한다.

**Architecture:** 전부 combined 드라이버의 **순수 후처리 헬퍼** + run() glue 추가. `_localize`·매칭 수학·consensus LOO 경로는 불변(bit-drift 0). periodic 신호는 기존 `ensemble_lab.template_periodicity`(recipe-level) 재사용, consensus arm 실패유형은 per_recipe rate 에서 재구성, Youden 은 per-frame score 가 있는 rcp_only arm 셀에서 측정. 판정 임계는 `golden_eval_config` 명명 상수.

**Tech Stack:** Python 3.10+, numpy(이미 의존), pytest(`test_golden_combined_eval_cond.py` 확장). No new deps. No CLI(env/상수만). 선행 spec: `poc/workflow_2/docs/superpowers/specs/2026-06-19-om-sem-modality-split-eval-design.md`.

## Global Constraints (CLAUDE.md)

- No `argparse`/CLI flags. No `from __future__` imports. Korean docstrings.
- Print-based logging `[INFO]`/`[WARNING]`/`[ERROR]`; em-dash(U+2014) 금지(office cp949). 단 이 드라이버는 `print` 만 씀.
- Absolute imports `from poc.workflow_2.xxx import ...`.
- Git: commit directly to `main`; stage only this plan's files via pathspec(no `git add -A`); verify with `git show --stat`.
- Commit footer (every commit):
  ```text
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- **Modify** `poc/workflow_2/golden_combined_eval_cond.py` — 신규 순수 헬퍼(Task 1-3) + run()/report/digest glue(Task 4).
- **Modify** `poc/workflow_2/test_golden_combined_eval_cond.py` — 합성 row 단위테스트 확장(Task 1-3).
- **Modify** `poc/workflow_2/golden_eval_config.example.py` — SPLIT_* 상수 블록(Task 3).

> `golden_eval_config.py`(실편집, gitignored)는 사용자가 오피스에서 example 블록을 복사해 튜닝한다. 코드에는 import-with-fallback 기본값이 박혀 있어 파일이 없거나 구버전이어도 동작한다.

---

### Task 1: 실패유형 히스토그램 헬퍼 (양 arm)

**Files:**

- Modify: `poc/workflow_2/golden_combined_eval_cond.py` (신규 헬퍼; 기존 `_consensus_by_modality` 아래에 추가)
- Test: `poc/workflow_2/test_golden_combined_eval_cond.py` (확장)

**Interfaces:**

- Produces:
  - `_classify_cell(cell) -> str` ∈ {"rank1_hit","look_alike","recall_miss"}
  - `_with_shares(acc) -> dict` (counts dict → {n, <type>:{n,share}})
  - `_failure_hist_by_modality(cells) -> {mod: shares}` (rcp_only 셀; `cell["periodic"]` 로 look_alike 세분)
  - `_failure_hist_from_rates(per_recipe) -> {mod: shares}` (consensus rate 재구성; `row["periodic"]`)
  - `_merge_failure_hists(*hists) -> shares`
  - `_routed_failure_hist(cons_hist, rcp_hist) -> {mod: shares}`

- [ ] **Step 1: Write the failing tests**

`poc/workflow_2/test_golden_combined_eval_cond.py` 의 맨 끝(`if __name__` 블록 없으니 파일 끝)에 추가:

```python
# --- 실패유형 히스토그램 (Step 2: OM/SEM split 증거) ---

def test_classify_cell_three_types():
    assert gcc._classify_cell({"topk_rank": 1, "in_topk": True}) == "rank1_hit"
    assert gcc._classify_cell({"topk_rank": 3, "in_topk": True}) == "look_alike"
    assert gcc._classify_cell({"topk_rank": None, "in_topk": False}) == "recall_miss"


def test_failure_hist_by_modality_counts_and_periodic():
    cells = [
        {"mod": "om", "topk_rank": 1, "in_topk": True, "periodic": False},
        {"mod": "om", "topk_rank": 2, "in_topk": True, "periodic": True},   # periodic look_alike
        {"mod": "om", "topk_rank": 4, "in_topk": True, "periodic": False},  # non-periodic look_alike
        {"mod": "OM", "topk_rank": None, "in_topk": False, "periodic": True},  # recall_miss (periodic 무관)
    ]
    h = gcc._failure_hist_by_modality(cells)["om"]
    assert h["n"] == 4
    assert h["rank1_hit"]["n"] == 1
    assert h["look_alike"]["n"] == 2
    assert h["recall_miss"]["n"] == 1
    assert h["periodic_look_alike"]["n"] == 1          # topk_rank 2 만(rank4 는 non-periodic)
    assert h["look_alike"]["share"] == 0.5


def test_failure_hist_from_rates_reconstructs_counts():
    # n=10, in_topk=0.8(=8 in pool), rank1=0.5(=5 rank1) → recall_miss=2, look_alike=3, rank1_hit=5.
    rows = [{"modality": "sem", "n_S_loo": 10, "cons_in_topk_rate": 0.8,
             "cons_rank1_rate": 0.5, "periodic": False}]
    h = gcc._failure_hist_from_rates(rows)["sem"]
    assert h["recall_miss"]["n"] == 2
    assert h["look_alike"]["n"] == 3
    assert h["rank1_hit"]["n"] == 5
    assert h["periodic_look_alike"]["n"] == 0


def test_failure_hist_from_rates_periodic_tags_lookalike():
    rows = [{"modality": "om", "n_S_loo": 10, "cons_in_topk_rate": 0.8,
             "cons_rank1_rate": 0.5, "periodic": True}]
    h = gcc._failure_hist_from_rates(rows)["om"]
    assert h["periodic_look_alike"]["n"] == 3          # periodic recipe → 전체 look_alike 가 periodic.


def test_merge_failure_hists_sums_and_reshare():
    a = gcc._with_shares({"rank1_hit": 1, "look_alike": 1, "recall_miss": 0,
                          "periodic_look_alike": 1, "n": 2})
    b = gcc._with_shares({"rank1_hit": 0, "look_alike": 2, "recall_miss": 2,
                          "periodic_look_alike": 0, "n": 4})
    m = gcc._merge_failure_hists(a, b)
    assert m["n"] == 6
    assert m["look_alike"]["n"] == 3
    assert m["recall_miss"]["n"] == 2
    assert m["periodic_look_alike"]["n"] == 1
    assert m["look_alike"]["share"] == 0.5


def test_routed_failure_hist_unions_modalities():
    cons = {"om": gcc._with_shares({"rank1_hit": 5, "look_alike": 0, "recall_miss": 0,
                                    "periodic_look_alike": 0, "n": 5})}
    rcp = {"sem": gcc._with_shares({"rank1_hit": 0, "look_alike": 0, "recall_miss": 3,
                                    "periodic_look_alike": 0, "n": 3})}
    out = gcc._routed_failure_hist(cons, rcp)
    assert out["om"]["n"] == 5 and out["sem"]["n"] == 3
    assert out["sem"]["recall_miss"]["share"] == 1.0
```

- [ ] **Step 2: Run the tests to verify they FAIL**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "failure_hist or classify_cell or merge_failure or routed_failure" -q`
Expected: FAIL with `AttributeError: module 'poc.workflow_2.golden_combined_eval_cond' has no attribute '_classify_cell'` (헬퍼 미정의).

- [ ] **Step 3: Implement the helpers**

`poc/workflow_2/golden_combined_eval_cond.py` 에서 `_consensus_by_modality` 함수 정의 **바로 위**(현재 line 175 `def _consensus_by_modality` 직전)에 추가:

```python
_FAILURE_TYPES = ("rank1_hit", "look_alike", "recall_miss", "periodic_look_alike")


def _classify_cell(cell):
    """per-frame 셀 → 실패유형. topk_rank==1=rank1_hit, in_topk(=rank 2..k)=look_alike, 그 외=recall_miss."""
    if cell.get("topk_rank") == 1:
        return "rank1_hit"
    if cell.get("in_topk"):
        return "look_alike"
    return "recall_miss"


def _with_shares(acc):
    """counts dict(키 _FAILURE_TYPES + 'n') → {n, <type>:{n, share}}. share = n_type/n."""
    n = int(acc.get("n", 0))
    out = {"n": n}
    for t in _FAILURE_TYPES:
        c = int(acc.get(t, 0))
        out[t] = {"n": c, "share": round(c / n, 3) if n else None}
    return out


def _failure_hist_by_modality(cells):
    """rcp_only per-frame 셀 list → {mod: _with_shares}. periodic_look_alike = look_alike & cell['periodic']."""
    by_mod = {}
    for c in cells:
        mod = (c.get("mod") or "unknown").lower()
        acc = by_mod.setdefault(mod, {t: 0 for t in _FAILURE_TYPES} | {"n": 0})
        t = _classify_cell(c)
        acc[t] += 1
        acc["n"] += 1
        if t == "look_alike" and c.get("periodic"):
            acc["periodic_look_alike"] += 1
    return {mod: _with_shares(acc) for mod, acc in by_mod.items()}


def _failure_hist_from_rates(per_recipe):
    """consensus per_recipe rows → {mod: _with_shares}. per-frame 없이 rate 에서 재구성.

    recall_miss = round(n*(1-in_topk)), look_alike = round(n*(in_topk-rank1)),
    rank1_hit = n - 둘(음수 클램프). periodic recipe(row['periodic'])면 look_alike 전부 periodic.
    """
    by_mod = {}
    for r in per_recipe:
        mod = (r.get("modality") or "unknown").lower()
        n = int(r["n_S_loo"])
        topk = float(r["cons_in_topk_rate"])
        rank1 = float(r["cons_rank1_rate"])
        recall_miss = max(0, int(round(n * (1.0 - topk))))
        look_alike = max(0, int(round(n * (topk - rank1))))
        rank1_hit = max(0, n - recall_miss - look_alike)
        acc = by_mod.setdefault(mod, {t: 0 for t in _FAILURE_TYPES} | {"n": 0})
        acc["rank1_hit"] += rank1_hit
        acc["look_alike"] += look_alike
        acc["recall_miss"] += recall_miss
        acc["n"] += n
        if r.get("periodic"):
            acc["periodic_look_alike"] += look_alike
    return {mod: _with_shares(acc) for mod, acc in by_mod.items()}


def _merge_failure_hists(*hists):
    """여러 _with_shares dict 합산 → 새 _with_shares (counts 합산 후 share 재계산)."""
    acc = {t: 0 for t in _FAILURE_TYPES} | {"n": 0}
    for h in hists:
        if not h:
            continue
        for t in _FAILURE_TYPES:
            acc[t] += int(h.get(t, {}).get("n", 0))
        acc["n"] += int(h.get("n", 0))
    return _with_shares(acc)


def _routed_failure_hist(cons_hist, rcp_hist):
    """modality 별 routed 실패히스토그램 = consensus arm + rcp_only arm 합산."""
    return {mod: _merge_failure_hists(cons_hist.get(mod), rcp_hist.get(mod))
            for mod in set(cons_hist) | set(rcp_hist)}
```

> 주: `{t: 0 for t in _FAILURE_TYPES} | {"n": 0}` 는 dict union(3.9+). 이 repo 는 3.10+ 라 안전.

- [ ] **Step 4: Run the tests to verify they PASS**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "failure_hist or classify_cell or merge_failure or routed_failure" -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_combined_eval_cond.py poc/workflow_2/test_golden_combined_eval_cond.py
git commit -m "feat(workflow_2): per-modality failure-mode histogram helpers (both arms)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat HEAD | head -8
```

---

### Task 2: per-modality Youden 분리 헬퍼 (rcp_only arm)

**Files:**

- Modify: `poc/workflow_2/golden_combined_eval_cond.py` (Task 1 헬퍼 아래에 추가)
- Test: `poc/workflow_2/test_golden_combined_eval_cond.py` (확장)

**Interfaces:**

- Produces:
  - `_youden_threshold(samples) -> {thr, J, tpr, fpr, n_pos, n_neg}` (samples = `[(score, hit_bool)]`)
  - `_youden_by_modality(cells) -> {mod: youden}` (rcp_only 셀의 score+hit)

- [ ] **Step 1: Write the failing tests**

파일 끝에 추가:

```python
# --- per-modality Youden 분리 (분류 축; L1 증거) ---

def test_youden_threshold_finds_separating_point():
    # hit(pos) 점수 {0.7,0.8,0.9}, miss(neg) {0.1,0.2,0.3} → 임계 0.7 에서 J=1.0.
    samples = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False), (0.1, False)]
    y = gcc._youden_threshold(samples)
    assert y["thr"] == 0.7
    assert y["J"] == 1.0
    assert y["tpr"] == 1.0 and y["fpr"] == 0.0
    assert y["n_pos"] == 3 and y["n_neg"] == 3


def test_youden_threshold_none_when_one_class_empty():
    y = gcc._youden_threshold([(0.5, True), (0.6, True)])
    assert y == {"thr": None, "J": None, "tpr": None, "fpr": None, "n_pos": 2, "n_neg": 0}


def test_youden_by_modality_splits_and_skips_missing_score():
    cells = [
        {"mod": "om", "score": 0.9, "hit": True},
        {"mod": "om", "score": 0.2, "hit": False},
        {"mod": "sem", "score": 0.8, "hit": True},
        {"mod": "sem", "score": 0.1, "hit": False},
        {"mod": "sem", "score": None, "hit": True},   # score 없으면 skip
    ]
    out = gcc._youden_by_modality(cells)
    assert out["om"]["n_pos"] == 1 and out["om"]["n_neg"] == 1
    assert out["sem"]["n_pos"] == 1 and out["sem"]["n_neg"] == 1   # None score 제외
```

- [ ] **Step 2: Run the tests to verify they FAIL**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "youden" -q`
Expected: FAIL with `AttributeError: ... has no attribute '_youden_threshold'`.

- [ ] **Step 3: Implement the helpers**

Task 1 의 `_routed_failure_hist` **아래**에 추가:

```python
def _youden_threshold(samples):
    """[(score, hit_bool)] → Youden 최적 임계. hit=True 가 positive(임계 이상이면 맞다고 예측).

    임계 후보 = 등장한 score 들. J = TPR - FPR 최대점. 한 클래스라도 비면 None.
    """
    pos = [s for s, h in samples if h]
    neg = [s for s, h in samples if not h]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"thr": None, "J": None, "tpr": None, "fpr": None, "n_pos": n_pos, "n_neg": n_neg}
    best = None
    for thr in sorted({s for s, _ in samples}):
        tpr = sum(1 for s in pos if s >= thr) / n_pos
        fpr = sum(1 for s in neg if s >= thr) / n_neg
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, thr, tpr, fpr)
    j, thr, tpr, fpr = best
    return {"thr": round(thr, 4), "J": round(j, 4),
            "tpr": round(tpr, 3), "fpr": round(fpr, 3), "n_pos": n_pos, "n_neg": n_neg}


def _youden_by_modality(cells):
    """rcp_only 셀 → {mod: _youden_threshold}. score(None 아님)+hit 있는 셀만."""
    by_mod = {}
    for c in cells:
        if c.get("score") is None or "hit" not in c:
            continue
        by_mod.setdefault((c.get("mod") or "unknown").lower(), []).append(
            (float(c["score"]), bool(c["hit"])))
    return {mod: _youden_threshold(s) for mod, s in by_mod.items()}
```

- [ ] **Step 4: Run the tests to verify they PASS**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "youden" -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_combined_eval_cond.py poc/workflow_2/test_golden_combined_eval_cond.py
git commit -m "feat(workflow_2): per-modality Youden separability helper (rcp_only arm)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat HEAD | head -8
```

---

### Task 3: split verdict + dominant + n_recipes + config 상수

**Files:**

- Modify: `poc/workflow_2/golden_combined_eval_cond.py` (헬퍼 + config import-with-fallback)
- Modify: `poc/workflow_2/golden_eval_config.example.py` (SPLIT_* 블록)
- Test: `poc/workflow_2/test_golden_combined_eval_cond.py` (확장)

**Interfaces:**

- Produces:
  - `_dominant_failure(fail, dominance) -> str | None` ∈ {"periodic_look_alike","other_look_alike","recall_miss",None}
  - `_recipes_by_modality(per_recipe, rcp_only_cells) -> {mod: int}`
  - `_split_verdict(fail_om, fail_sem, rate_om, rate_sem, cfg) -> dict`
  - `_SPLIT_CFG` (module dict from config 상수 or 기본값)
  - `golden_eval_config.example.py`: `SPLIT_MIN_FRAMES/SPLIT_MIN_RECIPES/SPLIT_RANK1_GAP/SPLIT_RANK1_FLOOR/SPLIT_DOMINANCE`

- [ ] **Step 1: Add SPLIT_* constants to the config template**

`poc/workflow_2/golden_eval_config.example.py` 의 맨 끝(현재 `MIN_S = None` 다음 줄)에 추가:

```python

# === OM/SEM split 판정 임계 (golden_combined_eval_cond 전용; 오피스에서 데이터 보고 튜닝) ===
# 이 블록은 combined 드라이버만 읽는다(env 브리지 X). 실편집 golden_eval_config.py 에 복사해 조정.
SPLIT_MIN_FRAMES = 30      # modality당 최소 채점 프레임(미달 → verdict=insufficient)
SPLIT_MIN_RECIPES = 5      # modality당 최소 recipe
SPLIT_RANK1_GAP = 0.10     # |rank1(OM)-rank1(SEM)| 이 이상이면 split 후보(10pp)
SPLIT_RANK1_FLOOR = 0.70   # 약한 쪽 routed rank1 이 이 밑이면 split 후보
SPLIT_DOMINANCE = 0.40     # 지배 실패유형 최소 비중(총 실패 중)
```

- [ ] **Step 2: Write the failing tests**

`poc/workflow_2/test_golden_combined_eval_cond.py` 파일 끝에 추가:

```python
# --- split verdict (판정 규칙) ---

_CFG = {"SPLIT_MIN_FRAMES": 30, "SPLIT_MIN_RECIPES": 5,
        "SPLIT_RANK1_GAP": 0.10, "SPLIT_RANK1_FLOOR": 0.70, "SPLIT_DOMINANCE": 0.40}


def _hist(rank1_hit=0, look_alike=0, recall_miss=0, periodic_look_alike=0):
    n = rank1_hit + look_alike + recall_miss
    return gcc._with_shares({"rank1_hit": rank1_hit, "look_alike": look_alike,
                             "recall_miss": recall_miss,
                             "periodic_look_alike": periodic_look_alike, "n": n})


def _rate(n_frames, rank1_rate, n_recipes):
    return {"n_frames": n_frames, "rank1_rate": rank1_rate, "n_recipes": n_recipes}


def test_dominant_failure_periodic_lookalike():
    h = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)   # 실패 8 전부 periodic
    assert gcc._dominant_failure(h, 0.40) == "periodic_look_alike"


def test_dominant_failure_recall_miss():
    h = _hist(rank1_hit=2, recall_miss=8)
    assert gcc._dominant_failure(h, 0.40) == "recall_miss"


def test_dominant_failure_none_when_no_failures():
    assert gcc._dominant_failure(_hist(rank1_hit=5), 0.40) is None


def test_dominant_failure_none_when_below_dominance():
    # 실패 9 = periodic 3 + other_look_alike 3(=look_alike6-periodic3) + recall 3 → 각 share 1/3 < 0.4 → None.
    h = _hist(rank1_hit=0, look_alike=6, recall_miss=3, periodic_look_alike=3)
    assert gcc._dominant_failure(h, 0.40) is None


def test_split_verdict_split_when_gap_and_divergent():
    om = _hist(rank1_hit=2, look_alike=8, periodic_look_alike=8)    # OM: periodic 지배
    sem = _hist(rank1_hit=2, recall_miss=8)                         # SEM: recall 지배
    v = gcc._split_verdict(om, sem, _rate(40, 0.60, 6), _rate(40, 0.85, 6), _CFG)  # gap 0.25
    assert v["verdict"] == "SPLIT"
    assert v["dominant_om"] == "periodic_look_alike"
    assert v["dominant_sem"] == "recall_miss"
    assert "L2_om_periodicity" in v["suggested_levers"]
    assert "L3_sem_recall" in v["suggested_levers"]


def test_split_verdict_shared_tune_when_gap_no_divergence():
    om = _hist(rank1_hit=2, recall_miss=8)
    sem = _hist(rank1_hit=2, recall_miss=8)                         # 같은 실패유형
    v = gcc._split_verdict(om, sem, _rate(40, 0.55, 6), _rate(40, 0.85, 6), _CFG)
    assert v["verdict"] == "shared_tune"
    assert v["suggested_levers"] == []


def test_split_verdict_no_split_when_no_gap():
    om = _hist(rank1_hit=8, periodic_look_alike=2, look_alike=2)
    sem = _hist(rank1_hit=8, recall_miss=2)
    v = gcc._split_verdict(om, sem, _rate(40, 0.82, 6), _rate(40, 0.85, 6), _CFG)  # gap 0.03, floor ok
    assert v["verdict"] == "no_split"


def test_split_verdict_insufficient_when_thin():
    om = _hist(rank1_hit=2, recall_miss=8)
    sem = _hist(rank1_hit=2, recall_miss=8)
    v = gcc._split_verdict(om, sem, _rate(10, 0.5, 2), _rate(40, 0.85, 6), _CFG)   # OM n_frames<30
    assert v["verdict"] == "insufficient"
    assert "om" in v["insufficient_mods"]


def test_recipes_by_modality_counts_distinct():
    per_recipe = [{"modality": "om", "recipe": "A"}, {"modality": "om", "recipe": "B"}]
    cells = [{"mod": "om", "rec_key": "B"}, {"mod": "sem", "rec_key": "C"}]   # B 중복, C 신규
    out = gcc._recipes_by_modality(per_recipe, cells)
    assert out["om"] == 2 and out["sem"] == 1
```

- [ ] **Step 3: Run the tests to verify they FAIL**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "dominant or split_verdict or recipes_by_modality" -q`
Expected: FAIL with `AttributeError: ... has no attribute '_dominant_failure'`.

- [ ] **Step 4: Implement helpers + config import**

`poc/workflow_2/golden_combined_eval_cond.py` 상단의 `OUTPUT_ROOT = ...`(현재 line 62) **아래**에 config import-with-fallback 추가:

```python
# split 판정 임계 — golden_eval_config(실편집/gitignored)에서 읽고, 없으면 기본값(example 과 동일).
try:
    from poc.workflow_2.golden_eval_config import (
        SPLIT_MIN_FRAMES, SPLIT_MIN_RECIPES, SPLIT_RANK1_GAP, SPLIT_RANK1_FLOOR, SPLIT_DOMINANCE,
    )
except ImportError:   # 파일 부재/구버전 — 기본값 폴백.
    SPLIT_MIN_FRAMES, SPLIT_MIN_RECIPES = 30, 5
    SPLIT_RANK1_GAP, SPLIT_RANK1_FLOOR, SPLIT_DOMINANCE = 0.10, 0.70, 0.40

_SPLIT_CFG = {
    "SPLIT_MIN_FRAMES": SPLIT_MIN_FRAMES, "SPLIT_MIN_RECIPES": SPLIT_MIN_RECIPES,
    "SPLIT_RANK1_GAP": SPLIT_RANK1_GAP, "SPLIT_RANK1_FLOOR": SPLIT_RANK1_FLOOR,
    "SPLIT_DOMINANCE": SPLIT_DOMINANCE,
}
```

그리고 Task 2 의 `_youden_by_modality` **아래**에 헬퍼 추가:

```python
_LEVER_BY_BUCKET = {
    "periodic_look_alike": "L2_om_periodicity",
    "recall_miss": "L3_sem_recall",
    "other_look_alike": "shared_rerank",
}


def _dominant_failure(fail, dominance):
    """_with_shares dict → 지배 실패 bucket 또는 None. 분모 = 총 실패(look_alike+recall_miss).

    bucket: periodic_look_alike / other_look_alike(= look_alike-periodic) / recall_miss.
    최대 bucket share >= dominance 면 그 bucket, 아니면 None(지배 없음).
    """
    la = int(fail.get("look_alike", {}).get("n", 0))
    pla = int(fail.get("periodic_look_alike", {}).get("n", 0))
    rm = int(fail.get("recall_miss", {}).get("n", 0))
    buckets = {"periodic_look_alike": pla, "other_look_alike": max(0, la - pla), "recall_miss": rm}
    total_fail = sum(buckets.values())
    if total_fail == 0:
        return None
    top = max(buckets, key=buckets.get)
    return top if buckets[top] / total_fail >= dominance else None


def _recipes_by_modality(per_recipe, rcp_only_cells):
    """modality 별 기여 recipe 수(중복 제거). consensus=row['recipe'], rcp_only=cell['rec_key']."""
    by_mod = {}
    for r in per_recipe:
        by_mod.setdefault((r.get("modality") or "unknown").lower(), set()).add(r.get("recipe"))
    for c in rcp_only_cells:
        by_mod.setdefault((c.get("mod") or "unknown").lower(), set()).add(c.get("rec_key"))
    return {mod: len(s) for mod, s in by_mod.items()}


def _split_verdict(fail_om, fail_sem, rate_om, rate_sem, cfg):
    """OM/SEM 실패히스토그램(routed) + rate → verdict.

    게이트: n_frames>=MIN_FRAMES AND n_recipes>=MIN_RECIPES (미달 → insufficient).
    격차: |r1차| >= RANK1_GAP OR min(r1) < RANK1_FLOOR.
    분기: 두 modality 지배 실패유형이 서로 다름.
    verdict: 격차&분기 → SPLIT(+lever), 격차만 → shared_tune, 그 외 → no_split.
    """
    insufficient = [name for name, rate in (("om", rate_om), ("sem", rate_sem))
                    if rate.get("n_frames", 0) < cfg["SPLIT_MIN_FRAMES"]
                    or rate.get("n_recipes", 0) < cfg["SPLIT_MIN_RECIPES"]]
    if insufficient:
        return {"verdict": "insufficient", "insufficient_mods": insufficient,
                "dominant_om": None, "dominant_sem": None, "suggested_levers": []}

    r_om = rate_om.get("rank1_rate") or 0.0
    r_sem = rate_sem.get("rank1_rate") or 0.0
    gap = abs(r_om - r_sem) >= cfg["SPLIT_RANK1_GAP"] or min(r_om, r_sem) < cfg["SPLIT_RANK1_FLOOR"]
    dom_om = _dominant_failure(fail_om, cfg["SPLIT_DOMINANCE"])
    dom_sem = _dominant_failure(fail_sem, cfg["SPLIT_DOMINANCE"])
    divergent = dom_om is not None and dom_sem is not None and dom_om != dom_sem

    if gap and divergent:
        verdict = "SPLIT"
        levers = sorted({_LEVER_BY_BUCKET.get(dom_om), _LEVER_BY_BUCKET.get(dom_sem)} - {None})
    elif gap:
        verdict, levers = "shared_tune", []
    else:
        verdict, levers = "no_split", []
    return {"verdict": verdict, "insufficient_mods": [],
            "dominant_om": dom_om, "dominant_sem": dom_sem, "suggested_levers": levers}
```

- [ ] **Step 5: Run the tests to verify they PASS**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -k "dominant or split_verdict or recipes_by_modality" -q`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/golden_combined_eval_cond.py poc/workflow_2/test_golden_combined_eval_cond.py poc/workflow_2/golden_eval_config.example.py
git commit -m "feat(workflow_2): split verdict + dominant-failure + SPLIT_* config constants

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat HEAD | head -8
```

---

### Task 4: run() 배선 + report + digest + summary 키

**Files:**

- Modify: `poc/workflow_2/golden_combined_eval_cond.py` (`run`, `_print_report`, `_digest_line`)

**Interfaces:**

- Consumes: Task 1-3 헬퍼, `ensemble_lab.template_periodicity` / `PERIODICITY_TAU`.
- Produces: `summary["by_modality"]["failure_modes"]` / `["youden"]`, `summary["split_verdict"]`; report 표 + digest 의 `verdict=...` 토큰.

> 단위테스트 없음(golden 필요) — 합성 row 헬퍼 테스트(Task 1-3) 전부 PASS + Mac `no_data` 실행 exit 0 + py_compile 로 검증(Task 5).

- [ ] **Step 1: Add the template_periodicity import**

`poc/workflow_2/golden_combined_eval_cond.py` 의 import 블록(현재 line 58 `from poc.workflow_2 import golden_consensus_eval_cond as gce` 다음 줄)에 추가:

```python
from poc.workflow_2.ensemble_lab import template_periodicity, PERIODICITY_TAU
```

- [ ] **Step 2: Compute periodic_by_key in the template-build loop**

`run()` 의 step 1 루프(현재 line 253-268 `for assets in recipes:` 블록)에서, `tpls_by_key[rec_key] = (assets, center_tpls, box_tpls)` **바로 위**에 periodic 계산을 추가하고, 루프 전에 dict 를 초기화한다.

`by_recipe = {}` / `tpls_by_key = {}` 선언 줄(현재 line 252-253) 다음에 추가:

```python
    periodic_by_key = {}              # rec_key -> bool (등록 key 가 주기/대칭 = template_periodicity > tau)
```

그리고 같은 루프 안 `rec_key = gce._recipe_key(assets)` 다음 줄에 추가:

```python
        try:
            rec_periodicity = max(
                (template_periodicity(t.raw_image) for t, _off in center_tpls.values()
                 if t is not None), default=0.0)
        except Exception as exc:
            print(f"[WARNING] periodicity 계산 실패 {assets.recipe_id}: {exc}")
            rec_periodicity = 0.0
        periodic_by_key[rec_key] = rec_periodicity > PERIODICITY_TAU
```

- [ ] **Step 3: Tag rcp_only cells with periodic + rec_key; join periodic into per_recipe**

`run()` 의 step 3 루프(현재 line 281-289)를 아래로 교체:

```python
    # 3) rcp-only arm — eligible 이 아닌 모든 recipe 의 rcp box localization.
    rcp_only_cells = []
    n_rcp_only_recipes = 0
    for rec_key, (assets, center_tpls, box_tpls) in tpls_by_key.items():
        if rec_key in eligible_keys:
            continue
        cells = _score_rcp_only(assets, center_tpls, box_tpls)
        if cells:
            is_periodic = periodic_by_key.get(rec_key, False)
            for c in cells:                       # 실패유형 히스토그램용 태깅(순수 헬퍼는 이 필드만 읽음).
                c["periodic"] = is_periodic
                c["rec_key"] = rec_key
            n_rcp_only_recipes += 1
            rcp_only_cells.extend(cells)
    rcp_only = _arm_rates(rcp_only_cells)

    # consensus per_recipe row 에 periodic(recipe-level) join — _failure_hist_from_rates 가 읽음.
    for r in per_recipe:
        r["periodic"] = periodic_by_key.get(r["recipe"], False)
```

- [ ] **Step 4: Assemble the new summary sections**

`run()` 의 step 4(현재 line 296-299, `cons_by_mod`/`rcp_by_mod`/`routed_by_mod` 계산) **다음**, `summary = {` 딕셔너리 만들기 **전**에 추가:

```python
    # Step 2: 실패유형 히스토그램(양 arm + routed) + per-mod Youden + split verdict.
    cons_fail = _failure_hist_from_rates(per_recipe)
    rcp_fail = _failure_hist_by_modality(rcp_only_cells)
    routed_fail = _routed_failure_hist(cons_fail, rcp_fail)
    youden_by_mod = _youden_by_modality(rcp_only_cells)
    n_rec_by_mod = _recipes_by_modality(per_recipe, rcp_only_cells)

    def _mod_rate(mod):
        rt = routed_by_mod.get(mod, {})
        return {"n_frames": rt.get("n_frames", 0), "rank1_rate": rt.get("rank1_rate"),
                "n_recipes": n_rec_by_mod.get(mod, 0)}

    empty_hist = _with_shares({"n": 0})
    split_verdict = _split_verdict(
        routed_fail.get("om", empty_hist), routed_fail.get("sem", empty_hist),
        _mod_rate("om"), _mod_rate("sem"), _SPLIT_CFG)
```

그리고 `summary` 딕셔너리의 `"by_modality": {...}` 항목(현재 line 323-327)에 두 키를 추가하고, 딕셔너리 끝에 `split_verdict` 를 추가한다. `"by_modality"` 블록을 아래로 교체:

```python
        "by_modality": {                            # Step 1: OM vs SEM 층화(split 여부 판단 근거).
            "consensus": cons_by_mod,
            "rcp_only": rcp_by_mod,
            "routed": routed_by_mod,
            "failure_modes": {                      # Step 2: 실패유형 분해(어떻게 깨지나).
                "consensus": cons_fail, "rcp_only": rcp_fail, "routed": routed_fail,
            },
            "youden": youden_by_mod,                # Step 2: per-mod 임계 분리(L1 증거; rcp_only arm).
        },
        "split_verdict": split_verdict,             # Step 2: SPLIT/shared_tune/no_split/insufficient.
```

- [ ] **Step 5: Add the verdict + failure-mode table to the report**

`_print_report` 의 마지막(현재 line 421 `print("    * OM(저배율...")` 다음)에 추가:

```python

    fm = (summary.get("by_modality") or {}).get("failure_modes", {}).get("routed", {})
    yd = (summary.get("by_modality") or {}).get("youden", {})
    print("\n[INFO] === (Step 2) 실패유형 분해 (routed; 'rank1 이 왜 낮나') ===")
    print(f"    {'mod':<6} {'n':>5} {'rank1_hit':>10} {'look_alike':>11} "
          f"{'periodic_la':>12} {'recall_miss':>12}   youden(thr/J,n+/-)")
    for mod in ("om", "sem"):
        h = fm.get(mod, {})
        y = yd.get(mod, {})
        if not h:
            continue
        def _sh(t):
            return h.get(t, {}).get("share")
        ys = (f"{y.get('thr')}/{y.get('J')} ({y.get('n_pos')}+/{y.get('n_neg')}-)"
              if y.get("thr") is not None else "-")
        print(f"    {mod.upper():<6} {h.get('n', 0):>5} {str(_sh('rank1_hit')):>10} "
              f"{str(_sh('look_alike')):>11} {str(_sh('periodic_look_alike')):>12} "
              f"{str(_sh('recall_miss')):>12}   {ys}")
    print("    * periodic_la 지배=OM 주기억제(L2), recall_miss 지배=SEM recall proposer(L3).")

    sv = summary.get("split_verdict") or {}
    print(f"\n[INFO] === (Step 2) SPLIT 판정 === verdict={sv.get('verdict', '-')}  "
          f"OM 지배={sv.get('dominant_om')}  SEM 지배={sv.get('dominant_sem')}  "
          f"권장 lever={sv.get('suggested_levers') or '-'}"
          + (f"  (insufficient: {sv.get('insufficient_mods')})"
             if sv.get('verdict') == 'insufficient' else ""))
```

- [ ] **Step 6: Add verdict token to the digest line**

`_digest_line` 의 `return (...)` 직전(현재 line 364)에 추가:

```python
    sv = summary.get("split_verdict") or {}
    verdict_tok = (f"verdict={sv.get('verdict', '-')}"
                   + (f"(om={sv.get('dominant_om')},sem={sv.get('dominant_sem')})"
                      if sv.get('verdict') == 'SPLIT' else ""))
```

그리고 `return (...)` 의 마지막 `f"scaling[{scaling}]"` 줄을 아래로 교체(맨 끝에 verdict 토큰 append — 기존 digest 테스트의 substring 은 보존):

```python
        f"scaling[{scaling}] | "
        f"{verdict_tok}"
```

- [ ] **Step 7: Verify existing digest/helper tests still pass**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -q`
Expected: all passed (기존 digest 테스트는 append 라 substring 보존; `_summary()` fixture 는 `split_verdict` 없어 `.get` 으로 `verdict=-` 토큰만 붙음).

- [ ] **Step 8: Commit**

```bash
git add poc/workflow_2/golden_combined_eval_cond.py
git commit -m "feat(workflow_2): wire failure-mode + youden + split verdict into combined eval run/report/digest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git show --stat HEAD | head -8
```

---

### Task 5: 최종 검증 (Mac, golden 없음)

**Files:** none (검증만).

- [ ] **Step 1: py_compile + 전체 헬퍼 테스트**

Run:
```bash
uv run python -m py_compile poc/workflow_2/golden_combined_eval_cond.py poc/workflow_2/golden_eval_config.example.py
uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -q
```
Expected: py_compile 무출력(성공), pytest 전부 passed(기존 + 신규 ~24개).

- [ ] **Step 2: 인접 회귀 가드 (ensemble_lab + localization 테스트)**

Run: `uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py poc/workflow_2/test_ensemble_lab.py poc/workflow_2/test_golden_localization_eval_cond.py -q`
Expected: all passed (ADR 0004 의 73 passed + 신규). `_localize`/매칭 미변경이라 회귀 없어야 함.

- [ ] **Step 3: no_data 경로 (Mac, golden 부재) 가 깨끗이 빠지는지**

Run: `uv run python poc/workflow_2/golden_combined_eval_cond.py`
Expected: `[WARNING] golden 데이터를 찾지 못했습니다: ...` 후 exit 0 (no_data). traceback 없음. (오피스에서만 실제 verdict 수치가 나온다.)

- [ ] **Step 4: 새 헬퍼가 summary 에 실제로 꽂히는지 스모크(합성 1-recipe)**

Run:
```bash
uv run python -c "
import poc.workflow_2.golden_combined_eval_cond as g
# 합성 summary 조립 경로만 점검: 헬퍼 chain 이 verdict 까지 도달하나.
om = g._with_shares({'rank1_hit':2,'look_alike':8,'recall_miss':0,'periodic_look_alike':8,'n':10})
sem = g._with_shares({'rank1_hit':2,'look_alike':0,'recall_miss':8,'periodic_look_alike':0,'n':10})
v = g._split_verdict(om, sem, {'n_frames':40,'rank1_rate':0.6,'n_recipes':6},
                     {'n_frames':40,'rank1_rate':0.85,'n_recipes':6}, g._SPLIT_CFG)
print('[SMOKE]', v['verdict'], v['suggested_levers'])
assert v['verdict']=='SPLIT'
print('[PASS] verdict chain ok')
"
```
Expected: `[SMOKE] SPLIT [...]` then `[PASS] verdict chain ok`.

---

## Final verification

- [ ] **전체 테스트 한 번 더 + no_data 실행**

Run:
```bash
uv run pytest poc/workflow_2/test_golden_combined_eval_cond.py -q
uv run python poc/workflow_2/golden_combined_eval_cond.py
```
Expected: pytest all passed, 드라이버 no_data exit 0.

- [ ] **오피스 실행 안내(코드 아님, 핸드오프 메모)**

오피스에서 `golden_eval_config.py` 에 SPLIT_* 블록(Task 3 Step 1)을 복사한 뒤
`uv run python poc/workflow_2/golden_combined_eval_cond.py` → `[DIGEST]` 줄의 `verdict=...` +
`(Step 2)` 표로 split 판정. SPLIT 이면 권장 lever(L2/L3)가 Phase 3 대상. 이 plan 은 거기서 끝(Phase 1).

---

## Self-Review

**Spec coverage:**

- spec §4.1 (실패유형 히스토그램 양 arm) → Task 1. ✓
- spec §4.2 (periodic = template_periodicity 재사용, recipe-level join) → Task 4 Step 2-3 + Task 1 의 periodic 분기. ✓
- spec §4.3 (per-mod Youden, rcp_only arm) → Task 2. ✓
- spec §4.4 (split verdict + config 상수) → Task 3. ✓
- spec §4.5 (summary/report/digest 출력) → Task 4. ✓
- spec §6 Phase 1 (eval+verdict, Mac 합성테스트→오피스 실행) → Task 1-5. ✓
- spec §8 (`_localize`/매칭 불변, template_periodicity import 재사용, 신규 헬퍼 목록) → Task 4 Step 1 import + 헬퍼 추가만, 매칭 경로 미변경. ✓
- spec §6 Phase 2/3 (L1/L2/L3 구현) → **이 plan 범위 밖**(증거 게이트, 별도 plan). ✓

**Placeholder scan:** TBD/TODO/"적절히 처리" 없음. 모든 코드 블록 완전. ✓

**Type consistency:**

- `_with_shares` 입력 = counts dict(키 `_FAILURE_TYPES` + `"n"`), 출력 = {n, <type>:{n,share}} — Task 1/3 일관 사용.
- 셀 필드 `topk_rank`/`in_topk`/`mod`/`score`/`hit`/`periodic`/`rec_key` — `_localize` 반환(score/hit/topk_rank/in_topk/mod) + run() 태깅(periodic/rec_key) 일치.
- per_recipe row 키 `modality`/`n_S_loo`/`cons_in_topk_rate`/`cons_rank1_rate`/`recipe`/`periodic` — `_consensus_template_ab` 산출(테스트 fixture 확인) + run() join(periodic) 일치.
- `_split_verdict` 의 rate dict 키 `n_frames`/`rank1_rate`/`n_recipes` — run() `_mod_rate` 출력과 일치.
- `_SPLIT_CFG` 키 5개 — config import + `_split_verdict`/`_dominant_failure` 참조 일치.
