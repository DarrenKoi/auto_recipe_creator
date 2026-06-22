# Re-registration Priority Ranking Report — Implementation Plan (Phase 1, S-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 골든 align_images 셋을 오프라인 스캔해 recipe 별 align-key 재등록 필요도를 modality 별로 worst-first
랭킹하고, flagged recipe 에 교체 whitebox 후보를 제안하는 workflow_2 벤치 드라이버를 만든다.

**Architecture:** 신규 `golden_reregister_report_cond.py` 가 production 엔진(`compute_align_key_score_ensemble`)과
기존 free-search 머신(`_gt_in_topk`)을 재사용해 recipe·modality 별 3-tier 증거를 산출 → 연속(threshold-free)
랭킹 → 박스 제안. **순수 로직(집계·tier·랭킹·제안 선택·포맷)은 같은 파일 내 I/O-분리 헬퍼**로 두고 합성
데이터로 단위 테스트, I/O·매칭 글루는 office 골든에서만 정확도 검증(Mac 은 py_compile + no_data).

**Tech Stack:** Python 3.10+, OpenCV, numpy. `poc.workflow_3.align` 엔진/템플릿/clean. 설정은
`golden_eval_config.py`(gitignore) + `seed_env()` 브리지.

**Spec:** `poc/workflow_2/docs/specs/2026-06-23-reregister-report-design.md` (읽고 시작할 것).

## Global Constraints

- **No CLI args** — 설정은 `golden_eval_config.py` 상수 + env(seed_env)만. `uv run python <script>.py` 로 실행.
- **Korean docstrings**, `[INFO]/[WARNING]/[ERROR]` print 로깅(logging 모듈 금지).
- **print/DIGEST/배너 문자열은 ASCII 만** — em-dash(U+2014) 금지(cp949 콘솔). 구분자는 `-`/`:`/`->`.
- **Absolute imports** `from poc.workflow_3...` / `from poc.workflow_2...` (workflow_2→workflow_3, 역방향 금지).
- **production(workflow_3) 코드 무수정** — 신규 코드는 `poc/workflow_2/` 안에만.
- **생성 산출물은 gitignored `debug_images/<driver>/` 로** — workflow_2 루트(git-tracked)에 쓰지 않는다.
- **debug 이미지는 JPEG**(이 파이프라인에 VLM 없음 → WebP 비대상).
- **연속 랭킹** — live τ(0.98/0.6053)는 참조선 표기만, tier 경계 아님. tier 경계는 자체 절대 floor(MSR/SELF_FLOOR).
- 모든 신호는 [0,1] 유계 raw 값 — **cohort 정규화(min-max) 금지**(1-recipe/동값에서 0/0, rank-relative 오류).

---

## File Structure

- **Create** `poc/workflow_2/golden_reregister_report_cond.py` — 드라이버 + 순수 헬퍼(명확히 구획). module-level
  `seed_env()` + module consts. `run()` 은 `__main__` 에서만 실행.
- **Create** `poc/workflow_2/test_reregister_report.py` — 순수 헬퍼 + 합성-이미지 테스트(드라이버 모듈에서 import).
- **Modify** `poc/workflow_2/golden_eval_config.example.py` — `REREGISTER_*` 추가.
- **Modify** `poc/workflow_2/golden_eval_config_loader.py` — `seed_env()` 에 `REREGISTER_*` 브리지.

**Module consts** (드라이버 상단; 오피스 보정 대상): `MSR_FLOOR=0.85`, `SELF_FLOOR=0.85`,
`EXCL_RADIUS_FOOTPRINTS=1.0`, `SUGG_SCALES=(0.8,1.0,1.25)`, `SUGG_STRIDE_RATIO=0.25`, `SPLIT_MIN_S=4`,
`ACCEPT_MARGIN=0.05`(env `REREGISTER_ACCEPT_MARGIN` 있으면 우선), `TIER_WEIGHT={"STRONG":2.0,"MEDIUM":1.0,"ADVISORY":0.3,"NONE":0.0}`.

**Tasks 2-5 = C1/C2 순수 헬퍼(Mac TDD). Task 1 = config. Tasks 6-7 = 드라이버 글루(office-run).**
시퀀싱: 1 → 2 → 3 → 4 → 6 (C1 완결·테스트) → 5 → 7 (C2).

---

## Task 1: Config knobs

**Files:**
- Modify: `poc/workflow_2/golden_eval_config.example.py`
- Modify: `poc/workflow_2/golden_eval_config_loader.py`
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Produces: env vars `REREGISTER_BOX_SUGGEST`(기본 "1"), `REREGISTER_TOPN`("0"), `REREGISTER_ACCEPT_MARGIN`(미설정 가능)
  를 `seed_env()` 가 `os.environ.setdefault` 로 브리지.

- [ ] **Step 1: Write the failing test**

`poc/workflow_2/test_reregister_report.py` (신규):

```python
"""reregister 리포트 순수 헬퍼 + config 브리지 테스트."""
import os
from poc.workflow_2 import golden_eval_config_loader as cfg


def test_seed_env_bridges_reregister_defaults():
    # 기존 값 격리
    for k in ("REREGISTER_BOX_SUGGEST", "REREGISTER_TOPN"):
        os.environ.pop(k, None)
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "1"
    assert os.environ["REREGISTER_TOPN"] == "0"


def test_seed_env_respects_existing_reregister(monkeypatch):
    monkeypatch.setenv("REREGISTER_BOX_SUGGEST", "0")
    cfg.seed_env()
    assert os.environ["REREGISTER_BOX_SUGGEST"] == "0"  # OS env 우선
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q`
Expected: FAIL (`KeyError: 'REREGISTER_BOX_SUGGEST'` — loader 가 아직 안 브리지).

- [ ] **Step 3: Add config template entries**

`golden_eval_config.example.py` 끝에 추가:

```python
# === re-registration 리포트 (golden_reregister_report_cond) ===
# 1 이면 박스 제안(C2)까지, 0 이면 랭킹 리포트(C1)만.
REREGISTER_BOX_SUGGEST = 1
# DIGEST/overlay 상위 N 제한(0=무제한).
REREGISTER_TOPN = 0
```

- [ ] **Step 4: Bridge in loader**

`golden_eval_config_loader.py` 의 `seed_env()` 안, 기존 `os.environ.setdefault(...)` 블록 옆에 추가
(기존 패턴 그대로 — 값은 str 로):

```python
    os.environ.setdefault("REREGISTER_BOX_SUGGEST", str(getattr(_cfg, "REREGISTER_BOX_SUGGEST", 1)))
    os.environ.setdefault("REREGISTER_TOPN", str(getattr(_cfg, "REREGISTER_TOPN", 0)))
    if getattr(_cfg, "REREGISTER_ACCEPT_MARGIN", None) is not None:
        os.environ.setdefault("REREGISTER_ACCEPT_MARGIN", str(_cfg.REREGISTER_ACCEPT_MARGIN))
```

(`_cfg` 는 loader 가 이미 import 한 config 모듈 별칭 — 기존 코드의 이름을 따른다. 다르면 그 이름 사용.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config_loader.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister report config knobs"
```

---

## Task 2: Evidence aggregation helpers (STRONG / MEDIUM / self_ratio)

**Files:**
- Create: `poc/workflow_2/golden_reregister_report_cond.py` (헤더 + consts + 이 헬퍼들)
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `_gt_in_topk` 반환 dict 형식 `{topk_rank, in_topk, best_cand_dist_norm, peak_ratio, ...}`.
- Produces:
  - `_aggregate_strong(frame_results: list) -> dict` → `{"strong_fail_frac": float, "worst_disp": float, "n_s": int}`
  - `_aggregate_medium(frame_results: list) -> dict` → `{"msr_peak_tail": float, "n_s": int}`
  - `_self_ratio(cands: list, best_xy, excl_radius_px: float) -> float`

- [ ] **Step 1: Write the failing tests**

`test_reregister_report.py` 에 추가:

```python
from poc.workflow_2 import golden_reregister_report_cond as rr


def test_aggregate_strong_counts_off_target_and_missing():
    # 3 프레임: rank1(ok) / rank3(off) / None=in_topk False(missing) → fail 2/3.
    frames = [
        {"in_topk": True, "topk_rank": 1, "best_cand_dist_norm": 0.05},
        {"in_topk": True, "topk_rank": 3, "best_cand_dist_norm": 0.4},
        {"in_topk": False, "topk_rank": None, "best_cand_dist_norm": 0.9},
    ]
    out = rr._aggregate_strong(frames)
    assert out["n_s"] == 3
    assert abs(out["strong_fail_frac"] - 2 / 3) < 1e-9
    assert out["worst_disp"] == 0.9


def test_aggregate_strong_all_clean():
    frames = [{"in_topk": True, "topk_rank": 1, "best_cand_dist_norm": 0.02}]
    assert rr._aggregate_strong(frames)["strong_fail_frac"] == 0.0


def test_aggregate_medium_uses_max_tail_and_zero_for_missing():
    # peak_ratio None(후보<2)은 0 으로 반영, tail=max.
    frames = [{"peak_ratio": 0.7}, {"peak_ratio": None}, {"peak_ratio": 0.93}]
    out = rr._aggregate_medium(frames)
    assert out["msr_peak_tail"] == 0.93
    assert out["n_s"] == 3


def test_self_ratio_excludes_trivial_peak():
    # cands: 자기-peak(원점 score 1.0) + 근접 sidelobe(제외돼야) + 먼 look-alike 0.6.
    class C:
        def __init__(self, xy, score):
            self.xy, self.score = xy, score
    cands = [C((100, 100), 1.0), C((104, 100), 0.95), C((400, 400), 0.6)]
    # excl 10px → sidelobe(거리4) 제외, 먼 look-alike 생존 → 0.6/1.0.
    assert abs(rr._self_ratio(cands, (100, 100), 10.0) - 0.6) < 1e-9


def test_self_ratio_unique_when_no_survivor():
    class C:
        def __init__(self, xy, score):
            self.xy, self.score = xy, score
    cands = [C((100, 100), 1.0), C((103, 100), 0.9)]  # 둘 다 excl 안 → 생존 0.
    assert rr._self_ratio(cands, (100, 100), 10.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "aggregate or self_ratio"`
Expected: FAIL (`ModuleNotFoundError`/`AttributeError` — 드라이버·헬퍼 없음).

- [ ] **Step 3: Create the driver module header + consts + these helpers**

`poc/workflow_2/golden_reregister_report_cond.py` (신규):

```python
"""re-registration 우선순위 랭킹 리포트 (Phase 1, S-only risk screening).

골든 셋을 스캔해 recipe·modality 별 재등록 필요도를 3-tier 증거로 산출·랭킹하고, flagged recipe 에
교체 whitebox 후보를 제안한다. 순수 로직(이 파일의 _ 헬퍼)은 I/O 와 분리해 합성 데이터로 단위 테스트한다.
정확도 숫자는 오피스 골든에서만(Mac 은 py_compile + no_data).

spec: poc/workflow_2/docs/specs/2026-06-23-reregister-report-design.md
"""
import os

from poc.workflow_2 import golden_eval_config_loader
golden_eval_config_loader.seed_env()  # gce import 전 env 브리지 (다른 드라이버와 동일).

import numpy as np

# ---- module consts (오피스 보정 대상) ----
MSR_FLOOR = 0.85               # peak_ratio tail 이 이 이상이면 MEDIUM.
SELF_FLOOR = 0.85              # self_ratio 가 이 이상이면 ADVISORY(OM 만).
EXCL_RADIUS_FOOTPRINTS = 1.0   # self-match 제외존 = 이 배수 × max(tw,th).
SUGG_SCALES = (0.8, 1.0, 1.25)
SUGG_STRIDE_RATIO = 0.25
SPLIT_MIN_S = 4
ACCEPT_MARGIN = float(os.getenv("REREGISTER_ACCEPT_MARGIN", "0.05"))
TIER_WEIGHT = {"STRONG": 2.0, "MEDIUM": 1.0, "ADVISORY": 0.3, "NONE": 0.0}

SURVIVORSHIP_BANNER = (
    "S-only latent-risk screening: candidates among historically-successful "
    "recipes, NOT a confirmed fail list. E-frame confirmation = Phase 2."
)


# ====================================================================
# 순수 헬퍼 — 증거 집계 (I/O 없음, 합성 데이터로 테스트).
# ====================================================================
def _aggregate_strong(frame_results):
    """STRONG: 무가드 free-search 가 진짜 점을 못 고른 S 프레임 비율 + worst 변위.

    frame_results: 프레임별 `_gt_in_topk` 반환 dict 리스트(None 프레임은 호출부에서 제외).
    fail = in_topk=False 또는 topk_rank>1.
    """
    n = len(frame_results)
    if n == 0:
        return {"strong_fail_frac": 0.0, "worst_disp": 0.0, "n_s": 0}
    fails = sum(
        1 for f in frame_results
        if (not f.get("in_topk")) or (f.get("topk_rank") or 1) > 1
    )
    worst = max((f.get("best_cand_dist_norm") or 0.0) for f in frame_results)
    return {"strong_fail_frac": fails / n, "worst_disp": float(worst), "n_s": n}


def _aggregate_medium(frame_results):
    """MEDIUM: peak_ratio(top2/top1) worst-case(max). None(후보<2)은 0(모호 아님)."""
    n = len(frame_results)
    tail = max((f.get("peak_ratio") or 0.0) for f in frame_results) if n else 0.0
    return {"msr_peak_tail": float(tail), "n_s": n}


def _self_ratio(cands, best_xy, excl_radius_px):
    """rcp self-match 의 변별도: 자기-peak 제외존 밖 최강 look-alike / 자기-peak.

    cands: score 내림차순 후보(.xy, .score). best_xy = 자기-peak 위치(=cands[0].xy).
    제외존 밖 생존 후보가 없으면 0.0(완전 변별). best score 0 가드.
    """
    if not cands:
        return 0.0
    best_score = float(cands[0].score) or 0.0
    if best_score <= 0:
        return 0.0
    bx, by = best_xy
    for c in cands[1:]:
        if float(np.hypot(c.xy[0] - bx, c.xy[1] - by)) > excl_radius_px:
            return float(c.score) / best_score
    return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "aggregate or self_ratio"`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister evidence aggregation helpers"
```

---

## Task 3: Tiering + risk score + ranking

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py`
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `MSR_FLOOR`, `SELF_FLOOR`, `TIER_WEIGHT`.
- Produces:
  - `_evidence_tier(modality, strong_fail_frac, msr_peak_tail, self_ratio) -> tuple[str, float]` → `(tier, severity)`
  - `_risk_score(tier, severity) -> float`
  - `_rank_rows(rows: list[dict]) -> list[dict]` (한 modality 의 row 들을 risk desc 정렬)

- [ ] **Step 1: Write the failing tests**

```python
def test_tier_strong_when_free_search_fails():
    tier, sev = rr._evidence_tier("sem", 0.5, 0.99, 0.99)
    assert tier == "STRONG" and sev == 0.5


def test_tier_medium_on_msr_tail():
    tier, sev = rr._evidence_tier("sem", 0.0, 0.90, 0.99)
    assert tier == "MEDIUM" and sev == 0.90


def test_tier_advisory_only_for_om():
    tier, _ = rr._evidence_tier("om", 0.0, 0.10, 0.90)
    assert tier == "ADVISORY"


def test_sem_self_never_surfaces():
    # SEM self-match 가 높아도(near-degenerate) MEDIUM/ADVISORY 로 안 뜸 → NONE.
    tier, _ = rr._evidence_tier("sem", 0.0, 0.10, 0.99)
    assert tier == "NONE"


def test_tier_none_below_floors():
    assert rr._evidence_tier("om", 0.0, 0.10, 0.10)[0] == "NONE"


def test_risk_score_orders_tiers():
    assert (rr._risk_score("STRONG", 0.0) > rr._risk_score("MEDIUM", 0.99)
            > rr._risk_score("ADVISORY", 0.99) > rr._risk_score("NONE", 0.99))


def test_rank_rows_desc_with_disp_tiebreak():
    rows = [
        {"recipe": "a", "risk_score": 2.5, "worst_disp": 0.3},
        {"recipe": "b", "risk_score": 2.5, "worst_disp": 0.9},  # 동점 → worst_disp 큰 게 위.
        {"recipe": "c", "risk_score": 1.2, "worst_disp": 0.9},
    ]
    ranked = rr._rank_rows(rows)
    assert [r["recipe"] for r in ranked] == ["b", "a", "c"]


def test_rank_rows_single_and_equal_safe():
    # 1-recipe / 동값 cohort 에서 예외·div 없이 동작(min-max 제거 회귀 가드).
    assert rr._rank_rows([{"recipe": "x", "risk_score": 1.0, "worst_disp": 0.0}])[0]["recipe"] == "x"
    rr._rank_rows([{"recipe": "p", "risk_score": 1.0, "worst_disp": 0.0},
                   {"recipe": "q", "risk_score": 1.0, "worst_disp": 0.0}])  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "tier or risk or rank"`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

드라이버에 추가:

```python
# ====================================================================
# 순수 헬퍼 — tier / risk / 랭킹.
# ====================================================================
def _evidence_tier(modality, strong_fail_frac, msr_peak_tail, self_ratio):
    """가장 강한 증거 1개로 tier 결정(상관 축 이중계수 회피). raw 절대 floor 경계.

    SEM self-match 는 near-degenerate 라 단독 tier 를 만들지 않는다(ADVISORY 는 OM 만).
    반환 (tier, severity) — severity 는 tier 내 정렬 키(raw, 정규화 없음).
    """
    if strong_fail_frac > 0:
        return "STRONG", float(strong_fail_frac)
    if msr_peak_tail >= MSR_FLOOR:
        return "MEDIUM", float(msr_peak_tail)
    if modality == "om" and self_ratio >= SELF_FLOOR:
        return "ADVISORY", float(self_ratio)
    return "NONE", 0.0


def _risk_score(tier, severity):
    """tier 가중 + tier 내 raw severity. cohort 통계 없음(1-recipe/동값 안전)."""
    return TIER_WEIGHT[tier] + float(severity)


def _rank_rows(rows):
    """한 modality row 들을 risk_score desc 정렬, 동점은 worst_disp desc tiebreak."""
    return sorted(rows, key=lambda r: (r["risk_score"], r.get("worst_disp", 0.0)), reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "tier or risk or rank"`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister tiering + risk score + ranking"
```

---

## Task 4: Report + digest + banner formatting

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py`
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: ranked row dicts with keys `recipe, tier, strong_fail_frac, worst_disp, msr_peak_tail,
  self_ratio, advisory_confidence, n_s, suggestion`(`"box(...)" | "none" | "insufficient"`),
  `sugg_self, sugg_fidelity`(없으면 None).
- Produces: `_format_report(rows_by_mod: dict) -> str`, `_format_digest(rows_by_mod: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def _sample_rows():
    return {
        "om": [
            {"recipe": "L/r1", "tier": "STRONG", "strong_fail_frac": 0.5, "worst_disp": 0.8,
             "msr_peak_tail": 0.99, "self_ratio": 0.9, "advisory_confidence": "ok", "n_s": 6,
             "risk_score": 2.5, "suggestion": "box(10,10,40,40)", "sugg_self": 0.5, "sugg_fidelity": 0.7},
        ],
        "sem": [
            {"recipe": "L/r2", "tier": "MEDIUM", "strong_fail_frac": 0.0, "worst_disp": 0.2,
             "msr_peak_tail": 0.92, "self_ratio": 0.99, "advisory_confidence": "low", "n_s": 3,
             "risk_score": 1.92, "suggestion": "insufficient", "sugg_self": None, "sugg_fidelity": None},
        ],
    }


def test_report_is_ascii_and_has_banner_and_rows():
    text = rr._format_report(_sample_rows())
    text.encode("ascii")  # em-dash 등 비-ASCII 있으면 raise.
    assert rr.SURVIVORSHIP_BANNER in text
    assert "L/r1" in text and "L/r2" in text
    assert "STRONG" in text and "MEDIUM" in text


def test_digest_is_ascii_one_line_per_pipe():
    d = rr._format_digest(_sample_rows())
    d.encode("ascii")
    assert d.startswith("[DIGEST] reregister(S-only):")
    assert "om[" in d and "sem[" in d and "|" in d


def test_banner_has_no_emdash():
    assert "—" not in rr.SURVIVORSHIP_BANNER
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "report or digest or banner"`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

```python
# ====================================================================
# 순수 헬퍼 — 포맷(ASCII only).
# ====================================================================
def _fmt_num(x):
    return "-" if x is None else f"{float(x):.3f}"


def _format_report(rows_by_mod):
    """modality 별 worst-first 테이블 텍스트(ASCII). 헤더에 survivorship 배너."""
    lines = ["=== Re-registration priority (S-only screening) ===", SURVIVORSHIP_BANNER, ""]
    cols = ("rank recipe tier strong_fail worst_disp msr_tail self_ratio(conf) "
            "n_s suggestion sugg_self/fid")
    for mod in ("om", "sem"):
        rows = rows_by_mod.get(mod, [])
        lines.append(f"-- {mod.upper()} ({len(rows)} screened) --")
        lines.append(cols)
        for i, r in enumerate(rows, 1):
            lines.append(" ".join([
                str(i), r["recipe"], r["tier"], _fmt_num(r["strong_fail_frac"]),
                _fmt_num(r["worst_disp"]), _fmt_num(r["msr_peak_tail"]),
                f"{_fmt_num(r['self_ratio'])}({r.get('advisory_confidence','ok')})",
                str(r["n_s"]), r.get("suggestion", "none"),
                f"{_fmt_num(r.get('sugg_self'))}/{_fmt_num(r.get('sugg_fidelity'))}",
            ]))
        lines.append("")
    return "\n".join(lines)


def _format_digest(rows_by_mod):
    """1줄 DIGEST(ASCII). modality 별 screened/strong/w_sugg + top recipe 2개."""
    parts = []
    for mod in ("om", "sem"):
        rows = rows_by_mod.get(mod, [])
        strong = sum(1 for r in rows if r["tier"] == "STRONG")
        w_sugg = sum(1 for r in rows if str(r.get("suggestion", "none")).startswith("box"))
        top = ",".join(r["recipe"] for r in rows[:2]) or "-"
        parts.append(f"{mod}[screened {len(rows)}, strong {strong}, w_sugg {w_sugg}, top {top}]")
    return "[DIGEST] reregister(S-only): " + " | ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "report or digest or banner"`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister report + digest formatting"
```

---

## Task 5: Box-suggestion pure helpers

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py`
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: `SUGG_SCALES`, `SUGG_STRIDE_RATIO`, `SPLIT_MIN_S`, `ACCEPT_MARGIN`.
- Produces:
  - `_split_frames(frame_keys, *, split_min_s=SPLIT_MIN_S) -> tuple | None`
  - `_iter_candidate_boxes(img_w, img_h, base_box, *, scales=SUGG_SCALES, stride_ratio=SUGG_STRIDE_RATIO) -> list`
  - `_mean(xs) -> float`
  - `_select_candidate(cand_metrics, baseline) -> dict | None`
  - `_accept_candidate(cand, baseline, *, accept_margin=ACCEPT_MARGIN) -> bool`
  - `_box_overlap_ratio(box, region) -> float`
  - `_dodge_guard(cand_overlap, base_overlap, val_delta, *, accept_margin=ACCEPT_MARGIN) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
def test_split_frames_insufficient():
    assert rr._split_frames(["a", "b", "c"], split_min_s=4) is None


def test_split_frames_deterministic_halves():
    sel, val = rr._split_frames(["a", "b", "c", "d", "e"], split_min_s=4)
    assert sel == ["a", "c", "e"] and val == ["b", "d"]  # even-idx select, odd-idx validate.


def test_iter_candidate_boxes_within_bounds():
    boxes = rr._iter_candidate_boxes(200, 200, (80, 80, 120, 120))
    assert boxes  # 비어있지 않음.
    for (l, t, r, b) in boxes:
        assert 0 <= l < r <= 200 and 0 <= t < b <= 200


def test_select_candidate_gates_on_baseline_fidelity():
    baseline = {"self_ratio": 0.95, "sel_fidelities": [0.6, 0.6]}  # mean 0.6.
    cands = [
        {"box": (0, 0, 10, 10), "self_ratio": 0.3, "sel_fidelities": [0.4, 0.4]},  # fid<baseline → 탈락.
        {"box": (1, 1, 11, 11), "self_ratio": 0.5, "sel_fidelities": [0.7, 0.7]},  # 통과, self 0.5.
        {"box": (2, 2, 12, 12), "self_ratio": 0.4, "sel_fidelities": [0.65, 0.65]},  # 통과, self 0.4(최저).
    ]
    pick = rr._select_candidate(cands, baseline)
    assert pick["box"] == (2, 2, 12, 12)


def test_select_candidate_none_when_all_fail_gate():
    baseline = {"self_ratio": 0.9, "sel_fidelities": [0.8, 0.8]}
    cands = [{"box": (0, 0, 1, 1), "self_ratio": 0.1, "sel_fidelities": [0.5, 0.5]}]
    assert rr._select_candidate(cands, baseline) is None


def test_accept_candidate_requires_both_margins():
    baseline = {"self_ratio": 0.95, "val_fidelities": [0.6, 0.6]}
    good = {"self_ratio": 0.4, "val_fidelities": [0.7, 0.7]}  # fid +0.1, self -0.55 → accept.
    assert rr._accept_candidate(good, baseline) is True
    weak_fid = {"self_ratio": 0.4, "val_fidelities": [0.61, 0.61]}  # fid +0.01 < margin.
    assert rr._accept_candidate(weak_fid, baseline) is False
    weak_self = {"self_ratio": 0.93, "val_fidelities": [0.7, 0.7]}  # self -0.02 < margin.
    assert rr._accept_candidate(weak_self, baseline) is False


def test_box_overlap_ratio():
    # box (0,0,10,10) area100; region (5,5,15,15) intersect (5,5,10,10) area25 → 0.25.
    assert abs(rr._box_overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)) - 0.25) < 1e-9
    assert rr._box_overlap_ratio((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_dodge_guard_rejects_overlap_avoidance_near_margin():
    # 현재 overlap 0.5, 후보 0.0(급감) + val_delta 가 margin 부근(0.05) → reject.
    assert rr._dodge_guard(0.0, 0.5, 0.05) is True
    # 후보가 충분히 이기면(val_delta 큼) overlap 급감이어도 통과(가짜 아님).
    assert rr._dodge_guard(0.0, 0.5, 0.5) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "split or candidate or overlap or dodge"`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

```python
# ====================================================================
# 순수 헬퍼 — 박스 제안 (C2).
# ====================================================================
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _split_frames(frame_keys, *, split_min_s=SPLIT_MIN_S):
    """held-out 분할. < split_min_s 면 None(insufficient). even-idx=select, odd-idx=validate."""
    if len(frame_keys) < split_min_s:
        return None
    select = [k for i, k in enumerate(frame_keys) if i % 2 == 0]
    validate = [k for i, k in enumerate(frame_keys) if i % 2 == 1]
    return select, validate


def _iter_candidate_boxes(img_w, img_h, base_box, *, scales=SUGG_SCALES, stride_ratio=SUGG_STRIDE_RATIO):
    """엔지니어 박스 크기 × scales 윈도를 stride 로 슬라이드. 이미지 경계 내 박스만."""
    bl, bt, br, bb = base_box
    bw, bh = br - bl, bb - bt
    short = max(1, min(bw, bh))
    stride = max(1, int(round(stride_ratio * short)))
    out = []
    for s in scales:
        w, h = max(1, int(round(bw * s))), max(1, int(round(bh * s)))
        if w >= img_w or h >= img_h:
            continue
        for t in range(0, img_h - h + 1, stride):
            for l in range(0, img_w - w + 1, stride):
                out.append((l, t, l + w, t + h))
    return out


def _select_candidate(cand_metrics, baseline):
    """select-half: mean fidelity >= baseline mean fidelity 인 후보 중 최저 self_ratio. 없으면 None."""
    base_fid = _mean(baseline["sel_fidelities"])
    passing = [c for c in cand_metrics if _mean(c["sel_fidelities"]) >= base_fid]
    if not passing:
        return None
    return min(passing, key=lambda c: c["self_ratio"])


def _accept_candidate(cand, baseline, *, accept_margin=ACCEPT_MARGIN):
    """validate-half: mean paired fidelity delta >= margin AND self_ratio 개선 >= margin."""
    fid_delta = _mean(cand["val_fidelities"]) - _mean(baseline["val_fidelities"])
    self_gain = baseline["self_ratio"] - cand["self_ratio"]
    return fid_delta >= accept_margin and self_gain >= accept_margin


def _box_overlap_ratio(box, region):
    """box 가 region(=removal mask 사각형)과 겹치는 비율 = 교집합/ box 면적."""
    l, t, r, b = box
    rl, rt, rr_, rb = region
    iw = max(0, min(r, rr_) - max(l, rl))
    ih = max(0, min(b, rb) - max(t, rt))
    area = max(1, (r - l) * (b - t))
    return (iw * ih) / area


def _dodge_guard(cand_overlap, base_overlap, val_delta, *, accept_margin=ACCEPT_MARGIN):
    """True=REJECT. 후보가 overlap 급감(현재 대비)으로만 이득(val_delta 가 margin 부근)이면 가짜 이득."""
    avoids = cand_overlap < base_overlap - 0.2
    marginal = val_delta < 2 * accept_margin
    return avoids and marginal
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "split or candidate or overlap or dodge"`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister box-suggestion pure helpers"
```

---

## Task 6: C1 driver integration (frame load + match pass + run)

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py`
- Test: `poc/workflow_2/test_reregister_report.py` (no_data 경로만; 정확도는 office)

**Interfaces:**
- Consumes: Task 2-4 헬퍼; `_build_templates`, `_gt_in_topk`, `compute_align_key_score_ensemble`,
  `clean_align_image.clean_image`, `resolve_assets_auto`, `DEBUG_IMAGE_DIR`.
- Produces: `_recipe_row(assets, modality) -> dict | None`, `run() -> str`.

**읽고 시작:** `poc/workflow_2/golden_localization_eval_cond.py` 의 `_process_msr_cond`(프레임 modality 배정 +
clean + cond GT 환산 패턴), `poc/workflow_3/align/clean_align_image.py`(`clean_image`/`build_removal_mask`),
`poc/workflow_2/golden_combined_eval_cond.py`(`DEBUG_IMAGE_DIR`/`OUTPUT_ROOT`/no_data 종료 패턴).

- [ ] **Step 1: Write the no_data test**

```python
def test_run_no_data_returns_warning(monkeypatch, tmp_path):
    # 빈 골든 루트 → no_data 경로(예외 없이 경고 문자열).
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))
    out = rr.run()
    assert "no_data" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k no_data`
Expected: FAIL (`AttributeError: run`).

- [ ] **Step 3: Implement frame load + match pass + run**

`_build_templates`/`_gt_in_topk` 는 `from poc.workflow_2.align_similarity import _build_templates, _gt_in_topk`.
`compute_align_key_score_ensemble`, `STRUCTURE_POLICY`/`PAUSED_SCALES` 는 engine 에서. `clean_image`/
`build_removal_mask` 는 `from poc.workflow_3.align.clean_align_image import clean_image, build_removal_mask`.
`DEBUG_IMAGE_DIR` 는 `from poc.workflow_2 import DEBUG_IMAGE_DIR`.

핵심 글루(드라이버에 추가; 좌표/모듈 세부는 위 "읽고 시작" 파일의 기존 패턴을 그대로 따른다):

```python
import cv2
from pathlib import Path
from poc.workflow_2.align_similarity import _build_templates, _gt_in_topk
from poc.workflow_3.align.matching.engine import (
    compute_align_key_score_ensemble, STRUCTURE_POLICY, PAUSED_SCALES,
)
from poc.workflow_2 import DEBUG_IMAGE_DIR

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "golden_reregister_report_cond"


def _self_match_ratio(box_tpl):
    """rcp box 템플릿을 자기 raw 이미지에 매칭 → exclusion-zone self_ratio + degenerate 판정.

    반환 (self_ratio, confidence). 템플릿이 이미지를 거의 채우면 confidence='low'(SEM near-degenerate).
    """
    img = box_tpl.raw_image
    th, tw = img.shape[:2]
    res = compute_align_key_score_ensemble(box_tpl, img, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY)
    cands = list(res.candidates)
    if not cands:
        return 0.0, "low"
    excl = EXCL_RADIUS_FOOTPRINTS * max(tw, th)
    ratio = _self_ratio(cands, cands[0].xy, excl)
    # raw image 가 작아 슬라이드 여지 부족하면 near-degenerate.
    conf = "low" if (tw >= img.shape[1] * 0.9 and th >= img.shape[0] * 0.9) else "ok"
    return ratio, conf


def _recipe_row(assets, modality):
    """한 recipe·modality 의 C1 증거 row. S 프레임/템플릿 없으면 None.

    S 프레임 로딩(clean + 프레임 GT)·modality 배정은 _process_msr_cond 패턴을 따른다.
    각 S 프레임: rcp center 템플릿으로 _gt_in_topk → STRONG/MEDIUM 재료.
    self_ratio: box 템플릿 self-match(ADVISORY).
    """
    center_tpls, box_tpls = _build_templates(assets)
    if center_tpls.get(modality) is None:
        return None
    # S 프레임 (gray_clean, gt_xy) 리스트 — 기존 패턴으로 로딩(modality 별).
    s_frames = _load_s_frames(assets, modality)   # 아래 헬퍼.
    if not s_frames:
        return None
    frame_results = []
    for gray, gt_xy in s_frames:
        r = _gt_in_topk(gray, gt_xy, {modality: center_tpls[modality]})
        if r is not None:
            frame_results.append(r)
    if not frame_results:
        return None
    strong = _aggregate_strong(frame_results)
    medium = _aggregate_medium(frame_results)
    self_ratio, conf = (0.0, "ok")
    if box_tpls.get(modality) is not None:
        self_ratio, conf = _self_match_ratio(box_tpls[modality])
    tier, sev = _evidence_tier(modality, strong["strong_fail_frac"], medium["msr_peak_tail"], self_ratio)
    return {
        "recipe": f"{assets.class_name}/{assets.recipe_name}",
        "modality": modality, "tier": tier, "risk_score": _risk_score(tier, sev),
        "strong_fail_frac": strong["strong_fail_frac"], "worst_disp": strong["worst_disp"],
        "msr_peak_tail": medium["msr_peak_tail"], "self_ratio": self_ratio,
        "advisory_confidence": conf, "n_s": strong["n_s"],
        "suggestion": "none", "sugg_self": None, "sugg_fidelity": None,
        "_assets": assets, "_center": center_tpls, "_box": box_tpls, "_s_frames": s_frames,
    }
```

`_load_s_frames(assets, modality)` 는 `_process_msr_cond` 의 프레임 modality 배정 + `clean_image` 호출 +
cond.txt crosshair px@5120 → 프레임 px 환산을 떼어낸 thin wrapper 로 구현(그 함수의 좌표 변환을 그대로 재사용).
`run()`:

```python
def run():
    """골든 루트 walk → recipe·modality 별 row → 랭킹 → 리포트/DIGEST 파일. 반환 = DIGEST(또는 no_data 경고)."""
    root = Path(os.getenv("ALIGN_GOLDEN_ROOT", "")).expanduser()
    recipes = _walk_recipes(root)   # resolve_assets_auto 기반; 기존 드라이버 walk 패턴 재사용.
    if not recipes:
        print("[WARNING] no_data: ALIGN_GOLDEN_ROOT empty or unset")
        return "[WARNING] no_data"
    rows_by_mod = {"om": [], "sem": []}
    for assets in recipes:
        for mod in ("om", "sem"):
            row = _recipe_row(assets, mod)
            if row is not None:
                rows_by_mod[mod].append(row)
    # C2(박스 제안)는 Task 7 에서 flagged row 에 채운다(REREGISTER_BOX_SUGGEST=1).
    for mod in rows_by_mod:
        rows_by_mod[mod] = _rank_rows(rows_by_mod[mod])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reregister_report.txt").write_text(_format_report(rows_by_mod), encoding="utf-8")
    digest = _format_digest(rows_by_mod)
    (OUTPUT_ROOT / "digest.txt").write_text(digest, encoding="utf-8")
    print(digest)
    return digest


if __name__ == "__main__":
    run()
```

`_walk_recipes(root)` 는 기존 골든 드라이버의 recipe 순회(glob `<eqp>/<class>/<recipe>`)를 따르고 각 recipe 에
`resolve_assets_auto(...)` 로 `assets` 를 만든다. 비면 `[]`.

- [ ] **Step 4: Run no_data test + py_compile**

Run: `uv run python -m py_compile poc/workflow_2/golden_reregister_report_cond.py`
Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q`
Run: `ALIGN_GOLDEN_ROOT=/tmp/empty uv run python poc/workflow_2/golden_reregister_report_cond.py`
Expected: py_compile OK; 모든 테스트 PASS; 드라이버 실행 시 `[WARNING] no_data` + exit 0.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister C1 driver (screening report + ranking)"
```

---

## Task 7: C2 box-suggestion integration + synthetic-image tests + overlay

**Files:**
- Modify: `poc/workflow_2/golden_reregister_report_cond.py`
- Test: `poc/workflow_2/test_reregister_report.py`

**Interfaces:**
- Consumes: Task 5 헬퍼 + `_recipe_row` row(`_box`/`_s_frames`/`_assets`), `build_removal_mask`.
- Produces: `_suggest_for_row(row) -> None`(row 의 `suggestion`/`sugg_self`/`sugg_fidelity` 채움),
  `_render_overlay(row) -> Path | None`.

- [ ] **Step 1: Write synthetic-image tests (real engine)**

```python
import numpy as np


def _periodic_img(w=240, h=240, period=24):
    g = np.zeros((h, w), np.uint8)
    g[:, ::period] = 255
    g[::period, :] = 255
    return g


def test_suggestion_finds_unique_patch_over_periodic():
    # 주기 배경 + 한 곳에 비주기 고유 마크 → 검색이 그 영역 박스를 찾음(self_ratio 낮음).
    img = _periodic_img()
    img[112:128, 112:128] = 180  # 고유 블록.
    base = (10, 10, 30, 30)  # 주기 영역의 엔지니어 박스(모호).
    found = rr._search_unique_box(img, base)   # 헬퍼: 최저 self_ratio 후보 박스 반환.
    fl, ft, fr_, fb = found["box"]
    # 고유 마크 영역 근처를 골라야(주기 영역 base 보다 self_ratio 낮음).
    assert found["self_ratio"] < 0.9


def test_suggestion_all_periodic_returns_none_distinctive():
    img = _periodic_img()
    base = (10, 10, 34, 34)
    found = rr._search_unique_box(img, base)
    # 전부 주기 → 어떤 박스도 충분히 변별 안 됨 → 호출부가 "no distinctive sub-region" 으로 처리.
    assert found is None or found["self_ratio"] >= 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q -k "unique or periodic"`
Expected: FAIL (`AttributeError: _search_unique_box`).

- [ ] **Step 3: Implement search + suggest + overlay**

```python
from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.clean_align_image import build_removal_mask  # mask 사각형 유도용(또는 GT rect).


def _passes_texture(patch, *, min_edge=0.02, min_lap=5.0):
    """blank/저텍스처 패치 skip. _edge_density/_lap_var 재사용."""
    from poc.workflow_2.align_similarity import _edge_density, _lap_var
    if patch.size == 0:
        return False
    return _edge_density(patch) >= min_edge and _lap_var(patch) >= min_lap


def _patch_self_ratio(img, box):
    l, t, r, b = box
    patch = img[t:b, l:r]
    if not _passes_texture(patch):
        return 1.0   # 저텍스처 → 변별 무의미(최대 모호 취급, 선택 안 됨).
    tpl = build_template(patch.copy(), recipe_id="sugg", version="sugg", key_type="om")
    res = compute_align_key_score_ensemble(tpl, img, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY)
    cands = list(res.candidates)
    if not cands:
        return 1.0
    excl = EXCL_RADIUS_FOOTPRINTS * max(r - l, b - t)
    return _self_ratio(cands, cands[0].xy, excl)


def _search_unique_box(img, base_box):
    """후보 박스 중 self_ratio 최저(가장 변별)를 반환. {box, self_ratio} 또는 None(후보 없음)."""
    h, w = img.shape[:2]
    best = None
    for box in _iter_candidate_boxes(w, h, base_box):
        sr = _patch_self_ratio(img, box)
        if best is None or sr < best["self_ratio"]:
            best = {"box": box, "self_ratio": sr}
    return best
```

`_suggest_for_row(row)` 는 `REREGISTER_BOX_SUGGEST=1` 일 때 flagged(tier != NONE) row 에 대해:
1. `_split_frames(row["_s_frames keys"])` → None 이면 `row["suggestion"]="insufficient"` 후 return.
2. select-half 에서 `_search_unique_box` + 후보별 select fidelity(후보 박스를 select 프레임 GT 위치에 매칭한
   `.score`) + baseline(현재 box) 동일 측정 → `_select_candidate`.
3. validate-half 에서 후보·baseline 재측정 → `_accept_candidate` + `_dodge_guard`(removal mask 사각형은
   `build_removal_mask` 또는 GT crosshair rect 로). 채택이면 `row["suggestion"]=f"box{box}"`, `sugg_self`,
   `sugg_fidelity`(validate mean) 채움; 아니면 `"no distinctive sub-region"`.
`_render_overlay(row)` 는 현재 박스(자홍)+후보 박스(초록)를 rcp 이미지에 그려
`OUTPUT_ROOT/<recipe>_reregister.jpg` 로 **JPEG** 저장(`cv2.imwrite(..., [int(cv2.IMWRITE_JPEG_QUALITY), 90])`).
`run()` 의 랭킹 직전에 `if os.getenv("REREGISTER_BOX_SUGGEST","1") != "0": for row in flagged: _suggest_for_row(row)`
+ 채택 row `_render_overlay` 호출을 추가한다(REREGISTER_TOPN 적용).

- [ ] **Step 4: Run tests + py_compile + no_data**

Run: `uv run python -m py_compile poc/workflow_2/golden_reregister_report_cond.py`
Run: `uv run pytest poc/workflow_2/test_reregister_report.py -q`
Run: `ALIGN_GOLDEN_ROOT=/tmp/empty uv run python poc/workflow_2/golden_reregister_report_cond.py`
Expected: py_compile OK; 전체 테스트 PASS; no_data 경고 + exit 0.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_reregister_report_cond.py poc/workflow_2/test_reregister_report.py
git commit -m "feat(workflow_2): reregister C2 box-suggestion + overlay"
```

---

## Office run (정확도 검증 — Mac 불가)

```text
# golden_eval_config.py: GOLDEN_ROOT=<align_images_golden>, REREGISTER_BOX_SUGGEST=1
uv run python poc/workflow_2/golden_reregister_report_cond.py
# → debug_images/golden_reregister_report_cond/{reregister_report.txt, digest.txt} + overlays
# [DIGEST] 한 줄 회신. 판정: STRONG tier = 재등록 1순위. 제안 박스는 engineer 검토 후보.
```
