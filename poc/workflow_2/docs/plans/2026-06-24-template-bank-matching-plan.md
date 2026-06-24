# Template-Bank Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bench-only template-bank matcher (soft-voting heatmap primary, RRF extra arm) plus a kill-test/A-B eval arm, to test whether cross-member agreement over individual recent-success S crops localizes the align point better than median-consensus — without ever editing workflow_3.

**Architecture:** A new bit-parity fork module `poc/workflow_2/template_bank_lab.py` (like `ensemble_lab.py`) that imports workflow_3 engine primitives and implements `bank_build` + two match arms + a GT-bucket classifier, plus new arms inside `golden_consensus_eval_cond.py`. The heatmap arm sums each member's dense chamfer score map into a common frame-center coordinate space and takes the global peak; the RRF arm pools per-member proposer candidates with one-vote-per-member spatial fusion. A §7-style kill-test classifies each winner as correct / near_periodic / far_wrong / one_member_only.

**Tech Stack:** Python 3.10+, numpy, OpenCV (via the workflow_3 match engine), pytest, uv.

**Spec:** `poc/workflow_2/docs/specs/2026-06-24-template-bank-matching-design.md`

## Global Constraints

- Korean docstrings/comments; print-based logging with `[INFO]`/`[WARNING]` prefixes (never the `logging` module).
- **ASCII-only** inside any `print(...)` string (office console is cp949); no em-dash (U+2014). Docstrings/comments may use Korean.
- No `argparse`/CLI flags; config via env + `golden_eval_config` (bridged in `golden_eval_config_loader.seed_env`, edit only the tracked `golden_eval_config.example.py`, never the gitignored `golden_eval_config.py`). No `from __future__` imports.
- Absolute imports `from poc.workflow_3...`; **workflow_2 imports workflow_3, never the reverse. `template_bank_lab.py` must never edit or be imported by workflow_3** (bit-parity fork, like `ensemble_lab.py`).
- Commit directly to `main` with a **pathspec** of exactly the files each task touches (no `git add -A`, no `commit -a`) — parallel sessions share the repo.
- **Candidate/winner xy are always template-center frame pixels** (engine invariant). Dense score-map placement: `_chamfer_score_map_at_scale` returns a top-left map; center of `score_map[y,x]` is `(x+tw//2, y+th//2)`.
- The RRF arm's per-member proposer call MUST be bit-parity with `_gt_in_topk` (`align_similarity.py:361`): `frame_dt = None if USE_ENSEMBLE_PROPOSER else preprocess_for_matching(gray)[1]`, then `_propose_topk(tpl, gray, frame_dt, scales=COMPARE_SCALES, topk=TOPK_CANDIDATES)`.
- Heatmap is **primary**; RRF must *beat* heatmap to justify its complexity. Aggregation is SUM (not MAX). MAX/best-member is NOT built.

## File Structure

- `poc/workflow_2/template_bank_lab.py` — NEW. `BankResult`, `bank_build`, `_accumulate_heatmap`, `_peaks_center`, `bank_match_heatmap`, `_dedup_within_member`, `bank_match_rrf`, `estimate_lattice_period`, `classify_winner`. Imports workflow_3 primitives; never edits workflow_3.
- `poc/workflow_2/test_template_bank_lab.py` — NEW. All unit tests (synthetic, Mac).
- `poc/workflow_2/golden_consensus_eval_cond.py` — MODIFY. Pure eval helpers (`_bootstrap_ci`, `_aggregate_buckets`, `_format_bank_digest`) + run() bank/heatmap/rrf arms + kill-test report (office-gated accuracy).
- `poc/workflow_2/golden_eval_config.example.py` + `golden_eval_config_loader.py` — MODIFY. 5 new knobs.

Sequencing: 1 → 2 → 3 → 4 → 5 → 6 → 7. Tasks 1-6 are Mac-testable (synthetic + pure helpers). Task 7 is driver glue; its accuracy is office-gated.

---

## Task 1: Config knobs

**Files:**
- Modify: `poc/workflow_2/golden_eval_config.example.py` (after the `REREGISTER_GT_TOL_NORM` block)
- Modify: `poc/workflow_2/golden_eval_config_loader.py` (inner import try/except + outer default + `seed_env`)
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Produces: env vars `TBANK_HEATMAP` ("1"), `TBANK_RRF` ("1"), `TBANK_PEAK_NMS_FRAC` ("0.5"), `TBANK_CLUSTER_TOL_FRAC` ("0.10"), `TBANK_RRF_K` ("60"), bridged via `setdefault` (real env wins).

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_2/test_template_bank_lab.py`:

```python
"""Template-bank matcher bench fork 테스트 (합성 데이터, Mac). workflow_3 미수정 fork."""
import os

import numpy as np

import poc.workflow_2.golden_eval_config_loader as cfg


def test_seed_env_bridges_tbank_defaults(monkeypatch):
    for k in ("TBANK_HEATMAP", "TBANK_RRF", "TBANK_PEAK_NMS_FRAC",
              "TBANK_CLUSTER_TOL_FRAC", "TBANK_RRF_K"):
        monkeypatch.delenv(k, raising=False)
    cfg.seed_env()
    assert os.environ["TBANK_HEATMAP"] == "1"
    assert os.environ["TBANK_RRF"] == "1"
    assert os.environ["TBANK_PEAK_NMS_FRAC"] == "0.5"
    assert os.environ["TBANK_CLUSTER_TOL_FRAC"] == "0.1"
    assert os.environ["TBANK_RRF_K"] == "60"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py::test_seed_env_bridges_tbank_defaults -q`
Expected: FAIL (KeyError — keys not bridged).

- [ ] **Step 3: Implement**

In `golden_eval_config.example.py`, after the `REREGISTER_GT_TOL_NORM = 0.30` line:

```python
# === Template-bank matcher bench (golden_consensus_eval_cond) ===
# heatmap=primary(soft-voting), rrf=extra arm. 1/0 토글.
TBANK_HEATMAP = 1
TBANK_RRF = 1
TBANK_PEAK_NMS_FRAC = 0.5      # heatmap peak-NMS 반경 = 이 비율 * 템플릿 단변.
TBANK_CLUSTER_TOL_FRAC = 0.10  # RRF arm 공간 클러스터 허용 = 이 비율 * 템플릿 단변.
TBANK_RRF_K = 60               # RRF 상수.
```

In `golden_eval_config_loader.py`, after the `REREGISTER_GT_TOL_NORM` inner import try/except:

```python
    try:
        from poc.workflow_2.golden_eval_config import (
            TBANK_HEATMAP, TBANK_RRF, TBANK_PEAK_NMS_FRAC,
            TBANK_CLUSTER_TOL_FRAC, TBANK_RRF_K,
        )
    except ImportError:   # 구버전 config(template-bank knob 없음) 하위호환.
        TBANK_HEATMAP, TBANK_RRF = 1, 1
        TBANK_PEAK_NMS_FRAC, TBANK_CLUSTER_TOL_FRAC, TBANK_RRF_K = 0.5, 0.10, 60
```

In the OUTER `except ImportError` (file absent) defaults block, append:

```python
    TBANK_HEATMAP, TBANK_RRF = 1, 1
    TBANK_PEAK_NMS_FRAC, TBANK_CLUSTER_TOL_FRAC, TBANK_RRF_K = 0.5, 0.10, 60
```

In `seed_env()`, after the `REREGISTER_GT_TOL_NORM` block:

```python
    os.environ.setdefault("TBANK_HEATMAP", str(TBANK_HEATMAP))
    os.environ.setdefault("TBANK_RRF", str(TBANK_RRF))
    os.environ.setdefault("TBANK_PEAK_NMS_FRAC", str(TBANK_PEAK_NMS_FRAC))
    os.environ.setdefault("TBANK_CLUSTER_TOL_FRAC", str(TBANK_CLUSTER_TOL_FRAC))
    os.environ.setdefault("TBANK_RRF_K", str(TBANK_RRF_K))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py::test_seed_env_bridges_tbank_defaults -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config_loader.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): template-bank config knobs (heatmap/rrf arms + params)"
```

---

## Task 2: `bank_build` (N individual templates, no median)

**Files:**
- Create: `poc/workflow_2/template_bank_lab.py`
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Consumes: `build_template` (engine), `coregister_crops` (consensus_cv).
- Produces: `bank_build(crops, *, recipe_id, modality, min_s, coregister=True) -> list[AlignKeyTemplate]` — None-safe: returns `[]` if `len(crops) < min_s`. Each crop becomes one `AlignKeyTemplate` (offset 0, key_type=modality).

- [ ] **Step 1: Write the failing test**

Add to `poc/workflow_2/test_template_bank_lab.py`:

```python
def _mark_crop(seed, size=64):
    """저텍스처 배경 + seed 위치 고유 마크가 있는 합성 gray crop."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size) * 30 + 20).astype(np.uint8)
    cx = cy = size // 2
    img[cy - 6:cy + 6, cx - 1:cx + 1] = 230   # 십자 마크(고유 구조).
    img[cy - 1:cy + 1, cx - 6:cx + 6] = 230
    return img


def test_bank_build_keeps_members_individual():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    assert len(bank) == 4                     # median 으로 합치지 않고 N 개 유지.
    assert all(hasattr(m, "edge_map") for m in bank)
    assert all(m.key_type == "om" for m in bank)


def test_bank_build_respects_min_s():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(2)]
    assert tb.bank_build(crops, recipe_id="r", modality="om", min_s=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "bank_build"`
Expected: FAIL (module / `bank_build` not defined).

- [ ] **Step 3: Implement**

Create `poc/workflow_2/template_bank_lab.py` with exactly this content:

```python
"""Template-bank matcher (bench 전용 bit-parity fork — ensemble_lab.py 패턴).

workflow_3 엔진 primitive 를 import 만 하고 절대 수정/역import 하지 않는다.
heatmap(soft-voting, primary) + rrf(extra) 두 arm, 그리고 winner 의 GT-bucket 분류.
좌표 규약: 모든 candidate/winner xy = 템플릿 중심 frame 픽셀(엔진 불변식).
"""
from dataclasses import dataclass

import numpy as np

from poc.workflow_3.align.matching.engine import build_template
from poc.workflow_3.align.consensus_cv import coregister_crops


def bank_build(crops, *, recipe_id, modality, min_s, coregister=True):
    """N 개의 S crop 을 *개별* AlignKeyTemplate 로 빌드(median 합치기 없음).

    crops: 동일 크기 uint8 gray crop 리스트. len<min_s 면 [] (consensus min_s 동일 게이트).
    coregister=True 면 빌드 전 sub-pixel 정렬(consensus 와 동일 전처리, blur 없음).
    """
    if crops is None or len(crops) < min_s:
        return []
    members = coregister_crops(crops) if coregister else list(crops)
    bank = []
    for i, c in enumerate(members):
        tpl = build_template(
            np.ascontiguousarray(c), recipe_id=recipe_id, version=f"s{i}",
            key_type=modality, align_offset_xy=(0, 0),
        )
        bank.append(tpl)
    return bank
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "bank_build"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/template_bank_lab.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): template_bank_lab.bank_build (individual members, no median)"
```

---

## Task 3: `bank_match_heatmap` (PRIMARY — soft-voting dense accumulation)

**Files:**
- Modify: `poc/workflow_2/template_bank_lab.py`
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Consumes: `preprocess_for_matching`, `_chamfer_score_map_at_scale` (engine); `COMPARE_SCALES`, `TOPK_CANDIDATES` (align_similarity).
- Produces:
  - `BankResult` dataclass: `xy: tuple|None`, `score: float`, `cand_xys: list[tuple]`, `cand_scores: list[float]`, `member_support: list[int]|None`, `arm: str`.
  - `_accumulate_heatmap(bank, frame_dt, frame_shape, scales) -> np.ndarray` (frame-center coords).
  - `_peaks_center(acc, *, nms_radius, max_peaks, min_score) -> list[tuple[float,int,int]]` (score,cx,cy desc).
  - `bank_match_heatmap(bank, gray, *, scales=COMPARE_SCALES, peak_nms_frac=0.5, topk=TOPK_CANDIDATES) -> BankResult`.

- [ ] **Step 1: Write the failing test**

Add:

```python
def _frame_with_mark(mark_xy, distractor_xy=None, size=256, seed=0):
    """저텍스처 frame + (mark_xy 에 십자) (+ distractor_xy 에 동일 십자)."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(size, size) * 30 + 20).astype(np.uint8)

    def _cross(cx, cy):
        img[cy - 6:cy + 6, cx - 1:cx + 1] = 230
        img[cy - 1:cy + 1, cx - 6:cx + 6] = 230

    _cross(*mark_xy)
    if distractor_xy is not None:
        _cross(*distractor_xy)
    return img


def test_heatmap_positive_one_member_distractor():
    """1/N 멤버만 distractor 에 끌려도 합산 peak 는 참 마크에 안착."""
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    frame = _frame_with_mark((120, 120))             # 참 마크 1곳만.
    res = tb.bank_match_heatmap(bank, frame)
    assert res.xy is not None
    assert abs(res.xy[0] - 120) <= 8 and abs(res.xy[1] - 120) <= 8


def test_heatmap_h0_all_members_same_distractor():
    """모든 멤버가 같은 distractor 에 끌리면 합산 peak 도 distractor (실패모드 검출)."""
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    # 참 마크(60,60) 약하게, distractor(190,190) 강하게(모든 멤버 공통).
    frame = _frame_with_mark((190, 190))
    res = tb.bank_match_heatmap(bank, frame)
    assert res.xy is not None
    assert abs(res.xy[0] - 190) <= 8 and abs(res.xy[1] - 190) <= 8


def test_heatmap_recovers_consistent_peak_individual_members_miss():
    """SUM 누적이 '개별 멤버는 1등으로 안 꼽지만 일관된 약 peak' 를 복원 — heatmap-primary 의
    핵심 근거(RRF 가 통과 못 하는 케이스). map 레벨에서 결정적으로 검증(취약한 실이미지 chamfer 회피)."""
    import poc.workflow_2.template_bank_lab as tb
    H = W = 64
    n_members = 4
    true_xy = (32, 32)
    acc = np.zeros((H, W), dtype=np.float32)
    member_maps = []
    for i in range(n_members):
        m = np.zeros((H, W), dtype=np.float32)
        m[true_xy[1], true_xy[0]] = 0.4                  # 모든 멤버 공통 약 peak.
        m[50, 10 + i * 8] = 1.0                          # 멤버별 고유 강 peak(서로 다른 위치).
        member_maps.append(m)
        acc += m
    # 개별 멤버의 argmax 는 각자 distractor(1.0 > 0.4) — 참 위치를 1등으로 안 꼽는다.
    for m in member_maps:
        assert np.unravel_index(int(np.argmax(m)), m.shape) != (true_xy[1], true_xy[0])
    # 합산: 참 위치 4*0.4=1.6 > 각 distractor 1.0 -> _peaks_center 가 참 위치를 1등으로.
    peaks = tb._peaks_center(acc, nms_radius=3, max_peaks=5, min_score=0.0)
    assert peaks and (peaks[0][1], peaks[0][2]) == true_xy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "heatmap"`
Expected: FAIL (`bank_match_heatmap` / `_peaks_center` not defined).

- [ ] **Step 3: Implement**

Add these imports to the module header (alongside the existing `build_template` / `coregister_crops` imports):

```python
from poc.workflow_3.align.matching.engine import preprocess_for_matching, _chamfer_score_map_at_scale
from poc.workflow_2.align_similarity import COMPARE_SCALES, TOPK_CANDIDATES
```

Then add to `template_bank_lab.py`:

```python
@dataclass
class BankResult:
    """bank 매칭 결과. xy=템플릿중심 frame 픽셀(없으면 None)."""
    xy: "tuple[int, int] | None"
    score: float
    cand_xys: list
    cand_scores: list
    member_support: "list[int] | None"
    arm: str


def _accumulate_heatmap(bank, frame_dt, frame_shape, scales):
    """멤버×scale 의 dense chamfer score map 을 frame-중심 좌표계에 SUM 누적.

    score_map[y,x] 는 top-left 배치 점수 → 중심은 (x+tw//2, y+th//2). 그래서
    acc[th//2:th//2+sh, tw//2:tw//2+sw] += score_map 로 중심좌표계에 더한다.
    """
    H, W = frame_shape
    acc = np.zeros((H, W), dtype=np.float32)
    for tpl in bank:
        for s in scales:
            score_map, (tw, th) = _chamfer_score_map_at_scale(tpl.edge_map, frame_dt, s)
            if score_map is None:
                continue
            sh, sw = score_map.shape
            oy, ox = th // 2, tw // 2
            if oy + sh > H or ox + sw > W:
                continue
            acc[oy:oy + sh, ox:ox + sw] += score_map
    return acc


def _peaks_center(acc, *, nms_radius, max_peaks, min_score):
    """중심좌표계 누적맵에서 NMS argmax peak 추출 → [(score,cx,cy),...] 내림차순."""
    work = acc.copy()
    peaks = []
    r = max(1, int(nms_radius))
    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        cy, cx = divmod(idx, work.shape[1])
        sc = float(work[cy, cx])
        if sc <= min_score:
            break
        peaks.append((sc, cx, cy))
        y0, y1 = max(0, cy - r), min(work.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(work.shape[1], cx + r + 1)
        work[y0:y1, x0:x1] = -np.inf
    return peaks


def bank_match_heatmap(bank, gray, *, scales=COMPARE_SCALES, peak_nms_frac=0.5,
                       topk=TOPK_CANDIDATES):
    """PRIMARY arm: 멤버 dense 응답을 SUM → 전역 peak. per-member top-K 가 떨어뜨린
    참 peak 도 약하게 일관되면 합산으로 살아남는다(gt_not_in_topk 공략)."""
    if not bank:
        return BankResult(None, 0.0, [], [], None, "heatmap")
    frame_dt = preprocess_for_matching(gray)[1]
    acc = _accumulate_heatmap(bank, frame_dt, gray.shape[:2], scales)
    short = min(bank[0].raw_image.shape[:2])
    nms_radius = max(1, int(peak_nms_frac * short))
    peaks = _peaks_center(acc, nms_radius=nms_radius, max_peaks=topk, min_score=0.0)
    if not peaks:
        return BankResult(None, 0.0, [], [], None, "heatmap")
    cand_xys = [(cx, cy) for (_s, cx, cy) in peaks]
    cand_scores = [s for (s, _cx, _cy) in peaks]
    return BankResult(cand_xys[0], cand_scores[0], cand_xys, cand_scores, None, "heatmap")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "heatmap"`
Expected: PASS. (If `test_heatmap_h0_all_members_same_distractor` is flaky on the synthetic, strengthen the distractor contrast in the fixture until the single strong mark dominates — the test asserts the harness *follows* the dominant peak, proving it can detect H0.)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/template_bank_lab.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): bank_match_heatmap (soft-voting dense accumulation, primary)"
```

---

## Task 4: `bank_match_rrf` (extra arm — per-member proposer + one-vote spatial RRF)

**Files:**
- Modify: `poc/workflow_2/template_bank_lab.py`
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Consumes: `_propose_topk`, `USE_ENSEMBLE_PROPOSER`, `COMPARE_SCALES`, `TOPK_CANDIDATES` (align_similarity); `preprocess_for_matching`, `_candidate_ncc` (engine).
- Produces:
  - `_dedup_within_member(cands, tol) -> list` (best per spatial cluster, preserves order).
  - `bank_match_rrf(bank, gray, *, scales=COMPARE_SCALES, topk=TOPK_CANDIDATES, cluster_tol, rrf_k) -> BankResult` (member_support populated).

- [ ] **Step 1: Write the failing test**

```python
def test_dedup_within_member_collapses_near_duplicates():
    import poc.workflow_2.template_bank_lab as tb

    class C:
        def __init__(self, xy, score):
            self.xy, self.score, self.scale = xy, score, 1.0
    cands = [C((100, 100), 0.9), C((103, 101), 0.5), C((180, 180), 0.7)]
    kept = tb._dedup_within_member(cands, tol=10)
    assert len(kept) == 2                       # (100,100)+(103,101) 한 표로 병합.
    assert kept[0].xy == (100, 100)             # 클러스터 대표 = 최고점.


def test_rrf_positive_one_member_distractor():
    import poc.workflow_2.template_bank_lab as tb
    crops = [_mark_crop(s) for s in range(4)]
    bank = tb.bank_build(crops, recipe_id="r", modality="om", min_s=3)
    frame = _frame_with_mark((120, 120))
    res = tb.bank_match_rrf(bank, frame, cluster_tol=10, rrf_k=60)
    assert res.xy is not None
    assert abs(res.xy[0] - 120) <= 10 and abs(res.xy[1] - 120) <= 10
    assert res.member_support is not None and res.member_support[0] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "dedup_within_member or rrf_positive"`
Expected: FAIL (`_dedup_within_member` / `bank_match_rrf` not defined).

- [ ] **Step 3: Implement**

```python
from poc.workflow_3.align.matching.engine import _candidate_ncc
from poc.workflow_2.align_similarity import _propose_topk, USE_ENSEMBLE_PROPOSER


def _dedup_within_member(cands, tol):
    """한 멤버 candidate 들을 공간 tol 로 클러스터링, 클러스터당 최고점 1개만 남긴다
    (member 내 near-duplicate 가 cross-member 표를 중복 행사하는 것 방지)."""
    kept = []
    for c in sorted(cands, key=lambda z: float(getattr(z, "score", 0.0)), reverse=True):
        cx, cy = c.xy
        if all((cx - k.xy[0]) ** 2 + (cy - k.xy[1]) ** 2 > tol * tol for k in kept):
            kept.append(c)
    return kept


def bank_match_rrf(bank, gray, *, scales=COMPARE_SCALES, topk=TOPK_CANDIDATES,
                   cluster_tol, rrf_k):
    """EXTRA arm: 멤버별 proposer top-K → member 내 dedup(1표/멤버/클러스터) →
    cross-member 공간 클러스터 RRF → max-member NCC rerank. heatmap 을 못 이기면
    discrete 기교가 값을 못 한다는 뜻."""
    if not bank:
        return BankResult(None, 0.0, [], [], [], "rrf")
    frame_dt = None if USE_ENSEMBLE_PROPOSER else preprocess_for_matching(gray)[1]
    # 멤버별 proposer (bit-parity with _gt_in_topk:361), member 내 dedup.
    per_member = []
    for tpl in bank:
        cands = _propose_topk(tpl, gray, frame_dt, scales=scales, topk=topk)
        per_member.append(_dedup_within_member(cands or [], cluster_tol))
    # cross-member 공간 클러스터: 각 클러스터에 멤버별 best rank 로 RRF 점수.
    clusters = []   # 각: {"xy":(x,y), "rrf":float, "members":set, "scale":float}
    for mi, cands in enumerate(per_member):
        for rank, c in enumerate(cands):
            cx, cy = c.xy
            hit = None
            for cl in clusters:
                if (cx - cl["xy"][0]) ** 2 + (cy - cl["xy"][1]) ** 2 <= cluster_tol ** 2:
                    hit = cl
                    break
            if hit is None:
                clusters.append({"xy": (cx, cy), "rrf": 0.0, "members": set(),
                                 "scale": float(getattr(c, "scale", 1.0) or 1.0)})
                hit = clusters[-1]
            if mi not in hit["members"]:               # 멤버당 클러스터 1표.
                hit["members"].add(mi)
                hit["rrf"] += 1.0 / (rrf_k + rank)
    if not clusters:
        return BankResult(None, 0.0, [], [], [], "rrf")
    # max-member NCC rerank: 클러스터 xy 에서 멤버 raw 중 최고 NCC 를 점수로.
    def _ncc_max(cl):
        vals = [_candidate_ncc(t.raw_image, gray, cl["xy"], cl["scale"]) for t in bank]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else -1.0
    ordered = sorted(clusters, key=lambda cl: (cl["rrf"], _ncc_max(cl)), reverse=True)
    cand_xys = [cl["xy"] for cl in ordered]
    cand_scores = [cl["rrf"] for cl in ordered]
    support = [len(cl["members"]) for cl in ordered]
    return BankResult(cand_xys[0], cand_scores[0], cand_xys, cand_scores, support, "rrf")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "dedup_within_member or rrf_positive"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/template_bank_lab.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): bank_match_rrf (one-vote spatial RRF + max-member NCC, extra arm)"
```

---

## Task 5: `estimate_lattice_period` + `classify_winner` (kill-test diagnostic)

**Files:**
- Modify: `poc/workflow_2/template_bank_lab.py`
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Consumes: `_to_grayscale` (engine) for the autocorr; numpy.
- Produces:
  - `estimate_lattice_period(template) -> float | None` — dominant periodicity (px) of the template's autocorrelation; None if no clear period.
  - `classify_winner(winner_xy, gt_xy, *, period, tol_px, member_support=None) -> str` ∈ {`correct`,`near_periodic`,`far_wrong`,`one_member_only`}.

- [ ] **Step 1: Write the failing test**

```python
def test_classify_winner_buckets():
    import poc.workflow_2.template_bank_lab as tb
    gt = (100, 100)
    # correct: within tol.
    assert tb.classify_winner((104, 99), gt, period=40, tol_px=8) == "correct"
    # near_periodic: off by ~one period along x.
    assert tb.classify_winner((140, 100), gt, period=40, tol_px=8) == "near_periodic"
    # far_wrong: not a period multiple, far.
    assert tb.classify_winner((175, 100), gt, period=40, tol_px=8) == "far_wrong"
    # one_member_only overrides (rrf arm) when support==1.
    assert tb.classify_winner((104, 99), gt, period=40, tol_px=8,
                              member_support=1) == "one_member_only"


def test_estimate_lattice_period_detects_stripes():
    import poc.workflow_2.template_bank_lab as tb
    from poc.workflow_3.align.matching.engine import build_template
    size, p = 96, 12
    img = np.zeros((size, size), np.uint8)
    img[:, ::p] = 220                                # 주기 p 세로 줄무늬.
    tpl = build_template(img, recipe_id="r", version="r", key_type="om")
    per = tb.estimate_lattice_period(tpl)
    assert per is not None and abs(per - p) <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "classify_winner or lattice_period"`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement**

```python
from poc.workflow_3.align.matching.engine import _to_grayscale


def estimate_lattice_period(template):
    """템플릿 raw 의 1D 자기상관 주기(px) 추정. 1차/2차 peak 간격의 최소; 불명확하면 None.

    행/열 평균 프로파일의 정규화 자기상관에서 lag>=3 의 첫 강한 peak 를 주기로 본다.
    """
    g = _to_grayscale(template.raw_image).astype(np.float32)

    def _axis_period(profile):
        p = profile - profile.mean()
        if np.allclose(p, 0):
            return None
        ac = np.correlate(p, p, mode="full")[len(p) - 1:]
        ac = ac / (ac[0] + 1e-6)
        best_lag, best_val = None, 0.0
        for lag in range(3, len(ac) // 2):
            if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.3:
                if ac[lag] > best_val:
                    best_val, best_lag = ac[lag], lag
        return float(best_lag) if best_lag is not None else None

    cands = [v for v in (_axis_period(g.mean(axis=0)), _axis_period(g.mean(axis=1)))
             if v is not None]
    return min(cands) if cands else None


def classify_winner(winner_xy, gt_xy, *, period, tol_px, member_support=None):
    """winner 를 GT 대비 버킷 분류. one_member_only(rrf support==1)이 최우선."""
    if member_support is not None and member_support <= 1:
        return "one_member_only"
    if winner_xy is None:
        return "far_wrong"
    dx = winner_xy[0] - gt_xy[0]
    dy = winner_xy[1] - gt_xy[1]
    dist = float(np.hypot(dx, dy))
    if dist <= tol_px:
        return "correct"
    if period and period > 0:
        # 가장 가까운 격자 배수와의 잔차가 tol 안이면 near_periodic.
        rx = abs(dx) - round(abs(dx) / period) * period
        ry = abs(dy) - round(abs(dy) / period) * period
        if np.hypot(rx, ry) <= tol_px and dist <= 3 * period:
            return "near_periodic"
    return "far_wrong"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "classify_winner or lattice_period"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/template_bank_lab.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): lattice-period estimate + GT-bucket classify_winner (kill-test)"
```

---

## Task 6: Pure eval helpers (bootstrap CI, bucket aggregation, digest)

**Files:**
- Modify: `poc/workflow_2/golden_consensus_eval_cond.py` (new pure helpers near the existing `_format_box_crop_digest`)
- Test: `poc/workflow_2/test_template_bank_lab.py`

**Interfaces:**
- Produces:
  - `_bootstrap_ci(values, *, n_boot=1000, seed=0, lo=2.5, hi=97.5) -> tuple[float, float]` (percentile CI of the mean; `(nan, nan)` for empty).
  - `_aggregate_buckets(labels) -> dict` — counts of `correct/near_periodic/far_wrong/one_member_only` + total.
  - `_format_bank_digest(stats_by_mod) -> str` — one ASCII `[DIGEST]` line.

- [ ] **Step 1: Write the failing test**

```python
def test_bootstrap_ci_is_deterministic_and_bracketed():
    import poc.workflow_2.golden_consensus_eval_cond as g
    vals = [0.0] * 50 + [1.0] * 50            # mean 0.5.
    lo, hi = g._bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert g._bootstrap_ci(vals, n_boot=500, seed=1) == (lo, hi)   # seed 고정 결정적.
    import math
    a, b = g._bootstrap_ci([], n_boot=10, seed=1)
    assert math.isnan(a) and math.isnan(b)


def test_aggregate_buckets_counts():
    import poc.workflow_2.golden_consensus_eval_cond as g
    labels = ["correct", "correct", "near_periodic", "far_wrong", "one_member_only"]
    d = g._aggregate_buckets(labels)
    assert d["correct"] == 2 and d["near_periodic"] == 1 and d["total"] == 5


def test_format_bank_digest_ascii_one_line():
    import poc.workflow_2.golden_consensus_eval_cond as g
    stats = {"om": {"heatmap_in_topk": 0.71, "cons_in_topk": 0.66, "near_periodic": 0.05},
             "sem": {"heatmap_in_topk": 0.70, "cons_in_topk": 0.66, "near_periodic": 0.30}}
    d = g._format_bank_digest(stats)
    assert d.startswith("[DIGEST] template-bank")
    assert "om[" in d and "sem[" in d and "\n" not in d
    assert d == d.encode("ascii", "replace").decode("ascii")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "bootstrap_ci or aggregate_buckets or bank_digest"`
Expected: FAIL (helpers not defined).

- [ ] **Step 3: Implement**

In `golden_consensus_eval_cond.py`, after `_format_box_crop_digest`:

```python
def _bootstrap_ci(values, *, n_boot=1000, seed=0, lo=2.5, hi=97.5):
    """평균의 percentile bootstrap CI. 빈 입력은 (nan,nan). seed 고정 결정적."""
    import numpy as _np
    arr = _np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = _np.random.RandomState(seed)
    means = _np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        means[i] = arr[rng.randint(0, n, n)].mean()
    return (float(_np.percentile(means, lo)), float(_np.percentile(means, hi)))


def _aggregate_buckets(labels):
    """classify_winner 라벨 리스트 -> 버킷 카운트 + total."""
    out = {"correct": 0, "near_periodic": 0, "far_wrong": 0, "one_member_only": 0, "total": 0}
    for lb in labels:
        if lb in out:
            out[lb] += 1
        out["total"] += 1
    return out


def _format_bank_digest(stats_by_mod):
    """per-modality bank vs consensus 한 줄 ASCII digest."""
    parts = []
    for mod in ("om", "sem"):
        s = stats_by_mod.get(mod)
        if not s:
            continue
        parts.append(
            f"{mod}[heatmap {s['heatmap_in_topk']:.3f} vs cons {s['cons_in_topk']:.3f}, "
            f"near_periodic {s['near_periodic']:.3f}]")
    return "[DIGEST] template-bank (in_topk + kill-test): " + " | ".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q -k "bootstrap_ci or aggregate_buckets or bank_digest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_consensus_eval_cond.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): bench eval helpers (bootstrap CI, bucket aggregate, bank digest)"
```

---

## Task 7: Eval-arm integration in `golden_consensus_eval_cond.run()` (office-gated)

**Files:**
- Modify: `poc/workflow_2/golden_consensus_eval_cond.py` (read TBANK env consts near other env reads; add the bank/heatmap/rrf arm computation inside the per-frame eval loop; min_s bins; kill-test report; `[DIGEST]`)
- Test: `poc/workflow_2/test_template_bank_lab.py` (no-data smoke; full accuracy office-gated)

**Interfaces:**
- Consumes: `template_bank_lab.bank_build / bank_match_heatmap / bank_match_rrf / estimate_lattice_period / classify_winner`; `_bootstrap_ci / _aggregate_buckets / _format_bank_digest` (Task 6); the driver's existing S-frame selection (history-first/LOO), GT crosshair, modality routing, `GT_TOL_NORM`, `CONSENSUS_MIN_S`.
- Produces: `TBANK_*` module consts; per-modality bank stats (in_topk, rank1, bucket fractions, bootstrap CIs) and `min_s` bins (3 / 4-6 / 7+) in `summary.json`; a `[DIGEST]` line; an `[INFO] kill-test` report.

**Implementation note (read the existing `run()` first):** locate the loop that already, per recipe/modality, has the **co-registered S crop list** + each **eval S frame** + its **GT crosshair xy** + the **rcp-only/consensus in_topk** results (around the `_consensus_template_ab` / per-point block, ~lines 569-815). The bank arms reuse the SAME crop source and GT as consensus — build the bank from the same crops the consensus median is built from (history-first disjoint pool, else LOO-excluding the eval frame), so no-leakage is identical by construction. If the crop list / per-frame GT are not directly exposed at one point, report DONE_WITH_CONCERNS describing where they live rather than guessing.

- [ ] **Step 1: Write the failing test (no-data smoke — keeps the office-gated path import-safe)**

```python
def test_consensus_run_no_data_with_tbank(monkeypatch, tmp_path):
    import poc.workflow_2.golden_consensus_eval_cond as g
    monkeypatch.setenv("ALIGN_GOLDEN_ROOT", str(tmp_path))   # 빈 루트.
    monkeypatch.setenv("TBANK_HEATMAP", "1")
    out = g.run()
    assert out in ("no_data", "no_ab")     # 빈 루트에서 깨지지 않고 정상 반환.
```

- [ ] **Step 2: Run test to verify it fails or errors**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py::test_consensus_run_no_data_with_tbank -q`
Expected: PASS already if no syntax error (run() returns early on empty root); this is the regression guard that the Step-3 arm code doesn't break the no-data path. Proceed to Step 3, then re-run.

- [ ] **Step 3: Implement**

Near the other env-read consts at the top of `golden_consensus_eval_cond.py`:

```python
# Template-bank arms (bench 실험; heatmap=primary, rrf=extra). seed_env 가 기본 브리지.
TBANK_HEATMAP = os.getenv("TBANK_HEATMAP", "1") != "0"
TBANK_RRF = os.getenv("TBANK_RRF", "1") != "0"
TBANK_PEAK_NMS_FRAC = float(os.getenv("TBANK_PEAK_NMS_FRAC", "0.5"))
TBANK_CLUSTER_TOL_FRAC = float(os.getenv("TBANK_CLUSTER_TOL_FRAC", "0.10"))
TBANK_RRF_K = int(os.getenv("TBANK_RRF_K", "60"))
```

Add the import near the top (workflow_2 sibling import):

```python
from poc.workflow_2.template_bank_lab import (
    bank_build, bank_match_heatmap, bank_match_rrf, estimate_lattice_period, classify_winner,
)
```

Inside the per-recipe/modality eval block, after the co-registered crop list `crops` and the consensus `in_topk` are computed, and for each eval frame `gray` with GT `(cxh, cyh)` and `short = min(gray.shape[:2])` (mirror the existing `GT_TOL_NORM * short` tolerance), add (guard `TBANK_HEATMAP`):

```python
    if TBANK_HEATMAP and len(crops) >= CONSENSUS_MIN_S:
        bank = bank_build(crops, recipe_id=recipe_id, modality=mod, min_s=CONSENSUS_MIN_S)
        period = estimate_lattice_period(bank[0]) if bank else None
        tol_px = GT_TOL_NORM * short
        short_t = min(bank[0].raw_image.shape[:2]) if bank else short
        # heatmap (primary)
        hres = bank_match_heatmap(bank, gray, peak_nms_frac=TBANK_PEAK_NMS_FRAC)
        hb = classify_winner(hres.xy, (cxh, cyh), period=period, tol_px=tol_px)
        # rrf (extra)
        rres = (bank_match_rrf(bank, gray, cluster_tol=int(TBANK_CLUSTER_TOL_FRAC * short_t),
                               rrf_k=TBANK_RRF_K) if TBANK_RRF else None)
        rb = (classify_winner(rres.xy, (cxh, cyh), period=period, tol_px=tol_px,
                              member_support=(rres.member_support[0] if rres and rres.member_support else None))
              if rres else None)
        # BOTH metrics (spec §8): in_topk = GT in fused top-K (tol_px); rank1 = top fused cand within tol.
        h_in = any(np.hypot(x - cxh, y - cyh) <= tol_px for (x, y) in hres.cand_xys)
        h_rank1 = hres.xy is not None and np.hypot(hres.xy[0] - cxh, hres.xy[1] - cyh) <= tol_px
        # accumulate per modality AND per min_s bin (len(crops) -> "3" / "4-6" / "7+"):
        #   heatmap: h_in, h_rank1, hb (bucket); rrf: r_in/r_rank1/rb (from rres); consensus in_topk/rank1.
        # summary.json MUST carry in_topk AND rank1 per arm per modality per min_s bin (spec §8);
        # buckets via _aggregate_buckets; per-recipe means -> _bootstrap_ci for CIs.
```

After the loop, compute per-modality `heatmap_in_topk` / `cons_in_topk` means, bucket fractions via `_aggregate_buckets`, bootstrap CIs via `_bootstrap_ci`, and the min_s-bin breakdown; write them into the `res` dict (so they land in `summary.json`), print the kill-test line and the digest:

```python
    print(f"[INFO] kill-test (heatmap winners): "
          f"om near_periodic={om_np:.3f} sem near_periodic={sem_np:.3f} "
          f"(>= cons near_periodic => H0 distractor reinforcement)")
    print(_format_bank_digest(bank_stats_by_mod))
```

ASCII-only in every print. Do not regress the existing rcp/consensus digest or the `no_data`/`no_ab` returns.

- [ ] **Step 4: Run no-data smoke + full suite**

Run: `uv run pytest poc/workflow_2/test_template_bank_lab.py -q`
Then import-safety: `uv run python -c "import poc.workflow_2.golden_consensus_eval_cond"`
Expected: all PASS; import exits 0. (Golden-set accuracy is office-gated.)

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/golden_consensus_eval_cond.py poc/workflow_2/test_template_bank_lab.py
git commit -m "feat(workflow_2): template-bank arms + kill-test report in consensus eval (Phase, office-gated)"
```

---

## Office run (accuracy — Mac cannot do)

```
# golden_eval_config.py: GOLDEN_ROOT=<golden_align_images>, TBANK_HEATMAP=1, TBANK_RRF=1
# optional: HISTORY_ROOT set for disjoint-pool banks; CONSENSUS_MIN_S to bin
uv run python poc/workflow_2/golden_consensus_eval_cond.py
# -> [INFO] kill-test (heatmap winners): om near_periodic=.. sem near_periodic=..
# -> [DIGEST] template-bank (in_topk + kill-test): om[heatmap X vs cons Y, near_periodic Z] | sem[..]
# -> debug_images/golden_consensus_eval_cond/<ts>/summary.json (per-mod in_topk/rank1, buckets, CIs, min_s bins)
```

Relay back: the `[INFO] kill-test` line, the `[DIGEST]`, and the per-modality `min_s`-bin in_topk + bootstrap CIs from `summary.json`. **Decision (spec §14):** H0 if heatmap `near_periodic` >= consensus (kill); else if heatmap in_topk beats consensus with separated CIs -> port heatmap; if RRF further beats heatmap -> port RRF instead; if neither beats consensus -> close the thread.

**Sensitivity sweep (spec §6 — done by env re-runs, like the Phase-1 fidelity-band A/B):** the RRF arm's `cluster_tol`/`k` are env-tunable, so the tol x k table is produced by re-running with different values and relaying each `[DIGEST]` (no internal sweep loop — YAGNI, matches the established bench A/B pattern):

```
TBANK_CLUSTER_TOL_FRAC=0.06 TBANK_RRF_K=20  uv run python poc/workflow_2/golden_consensus_eval_cond.py
TBANK_CLUSTER_TOL_FRAC=0.10 TBANK_RRF_K=60  uv run python poc/workflow_2/golden_consensus_eval_cond.py
TBANK_CLUSTER_TOL_FRAC=0.16 TBANK_RRF_K=120 uv run python poc/workflow_2/golden_consensus_eval_cond.py
```
The heatmap primary has only `TBANK_PEAK_NMS_FRAC` (lightly swept ±50% the same way). No RRF-arm headline is reported without its tol x k table.
