# Ensemble Proposer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** single rcp box template 경로의 proposer recall(gt_in_topk)을 다중 구조 채널 ensemble proposer로 끌어올린다.

**Architecture:** 기존 multi-scale chamfer 엔진을 edge map만 바꿔 3채널(C1 Canny-DT, C2 Scharr gradient, C3 orientation-binned directional)로 확장하고, RRF(순위 기반, 스케일 무관)로 융합해 top-8 + shadow-24 후보를 낸다. proposer recall만 격리한 A/B를 `golden_localization_eval_cond` 재사용으로 측정한다.

**Tech Stack:** Python, OpenCV(cv2), NumPy, pytest. 모듈: `poc/workflow_2/align_key_matcher.py`(확장), 신규 `ensemble_proposer.py`, `proposer_recall_ab.py`.

**Spec:** `poc/workflow_2/docs/specs/2026-06-09-ensemble-proposer-design.md`

**실행 규약:** Korean docstring, `[INFO]/[WARNING]` print, no `__future__`, 절대 import(`from poc.workflow_2.xxx import ...`), no CLI args. 테스트는 `uv run python -m pytest <path> -q`.

---

## File Structure

- **Modify** `poc/workflow_2/align_key_matcher.py` — `_mean_dt_map_at_scale` 추출 + `_collect_candidates` 추출(채널 무관 후보 생성). C1 동작 보존.
- **Create** `poc/workflow_2/ensemble_proposer.py` — C2 Scharr/C3 directional edge 추출 + directional chamfer + `compute_ensemble_candidates`(RRF 융합 + per-channel solo).
- **Create** `poc/workflow_2/test_ensemble_proposer.py` — 채널·directional·RRF 합성 단위테스트.
- **Create** `poc/workflow_2/proposer_recall_ab.py` — recall@N + attribution 순수 헬퍼 + A/B 러너(localization eval 재사용).
- **Create** `poc/workflow_2/test_proposer_recall_ab.py` — recall@N/attribution 순수 헬퍼 테스트.

---

## Task 1: `_mean_dt_map_at_scale` / `_collect_candidates` 추출 (리팩터, 동작 보존)

C2/C3가 chamfer 엔진을 edge map만 바꿔 재사용하려면, (a) exp 적용 전 mean_dt map과 (b) edge→후보 생성 본체를 분리해야 한다. C1 동작은 불변(회귀 가드).

**Files:**
- Modify: `poc/workflow_2/align_key_matcher.py` (`_chamfer_score_map_at_scale` ~230, `compute_chamfer_candidates` ~303)
- Test: `poc/workflow_2/test_align_key_match.py` (기존 10/10 회귀)

- [ ] **Step 1: 회귀 기준 확인 (리팩터 전 green)**

Run: `uv run python -m pytest poc/workflow_2/test_align_key_match.py -q`
Expected: PASS (10 passed) — 리팩터 후에도 동일해야 함.

- [ ] **Step 2: `_mean_dt_map_at_scale` 추출**

`align_key_matcher.py`의 `_chamfer_score_map_at_scale`를 아래로 교체(내부에서 mean_dt map을 만드는 부분을 분리):

```python
def _mean_dt_map_at_scale(template_edges, frame_dt, scale):
    """단일 스케일의 *mean_dt map* (exp 적용 전). 작을수록 좋음.

    반환 (mean_dt_map | None, (tw, th)). 매칭 불가(템플릿이 프레임보다 큼 / edge 없음) → (None, ...).
    """
    edges_scaled = _scaled_edges(template_edges, scale)
    th, tw = edges_scaled.shape[:2]
    fh, fw = frame_dt.shape[:2]
    if th >= fh or tw >= fw:
        return None, (tw, th)
    template_mask = (edges_scaled > 0).astype(np.float32)
    edge_count = float(template_mask.sum())
    if edge_count <= 0:
        return None, (tw, th)
    result = cv2.matchTemplate(frame_dt, template_mask, cv2.TM_CCORR)
    return (result / edge_count).astype(np.float32), (tw, th)


def _chamfer_score_map_at_scale(template_edges, frame_dt, scale):
    """단일 스케일 score map = exp(-mean_dt/DT_TAU_PX). (mean_dt map wrapper — 동작 보존)."""
    mean_dt_map, (tw, th) = _mean_dt_map_at_scale(template_edges, frame_dt, scale)
    if mean_dt_map is None:
        return None, (tw, th)
    return np.exp(-mean_dt_map / DT_TAU_PX).astype(np.float32), (tw, th)
```

- [ ] **Step 3: `_collect_candidates` 추출 (compute_chamfer_candidates 본체)**

`compute_chamfer_candidates`를 아래로 교체(edge map을 직접 받는 본체 + 얇은 wrapper):

```python
def _collect_candidates(template_edges, frame_dt, *, scales=DEFAULT_SCALES,
                        top_n=8, nms_radius_ratio=0.5, min_score=0.0):
    """edge map + frame_dt → multi-scale chamfer NMS top-N 후보 (chamfer 내림차순).

    채널 무관 — C1(canny)/C2(scharr) 가 같은 본체를 edge map 만 바꿔 호출한다.
    """
    collected = []  # (score, cx, cy, scale, tw, th)
    for scale in scales:
        score_map, (tw, th) = _chamfer_score_map_at_scale(template_edges, frame_dt, scale)
        if score_map is None:
            continue
        nms_r = max(4, int(min(tw, th) * nms_radius_ratio))
        for s, cx, cy in _extract_peaks(
            score_map, tw, th, max_peaks=top_n, min_score=min_score, nms_radius=nms_r,
        ):
            collected.append((s, cx, cy, scale, tw, th))
    collected.sort(key=lambda t: t[0], reverse=True)
    kept = []
    for item in collected:
        s, cx, cy, scale, tw, th = item
        merge_r = max(4, int(min(tw, th) * nms_radius_ratio))
        if any((abs(cx - k[1]) <= merge_r and abs(cy - k[2]) <= merge_r) for k in kept):
            continue
        kept.append(item)
        if len(kept) >= top_n:
            break
    return [
        AlignKeyCandidate(score=float(s), chamfer_score=float(s), xy=(cx, cy),
                          scale=float(scale), template_size=(tw, th))
        for (s, cx, cy, scale, tw, th) in kept
    ]


def compute_chamfer_candidates(template, frame_dt, *, scales=DEFAULT_SCALES,
                               top_n=8, nms_radius_ratio=0.5, min_score=0.0):
    """multi-scale Chamfer NMS top-N 후보 (C1 = canny edge map). _collect_candidates wrapper."""
    return _collect_candidates(
        template.edge_map, frame_dt, scales=scales, top_n=top_n,
        nms_radius_ratio=nms_radius_ratio, min_score=min_score)
```

- [ ] **Step 4: 회귀 통과 확인**

Run: `uv run python -m pytest poc/workflow_2/test_align_key_match.py poc/workflow_2/test_golden_localization_eval_cond.py -q`
Expected: PASS (변경 없음 — 동작 보존 리팩터).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/align_key_matcher.py
git commit -m "workflow_2(matcher): extract _mean_dt_map_at_scale/_collect_candidates (no behavior change) — ensemble 준비"
```

---

## Task 2: C2 Scharr gradient 채널

Canny가 버리는 약한/저대비 구조를 잡는 두 번째 채널. gradient magnitude를 **C1 foreground 밀도에 맞춘 percentile**로 이진화.

**Files:**
- Create: `poc/workflow_2/ensemble_proposer.py`
- Test: `poc/workflow_2/test_ensemble_proposer.py`

- [ ] **Step 1: 실패 테스트 작성**

`test_ensemble_proposer.py`:

```python
"""ensemble proposer(C2 Scharr·C3 directional·RRF) 합성 단위테스트 — Mac 실행 가능."""
import numpy as np
import cv2

from poc.workflow_2 import ensemble_proposer as ep


def _square_img(size=200, box=(70, 70, 60, 60), bg=110, edge=230):
    """배경 위 사각 윤곽 하나 — gradient/edge 채널이 윤곽을 잡는지 본다."""
    img = np.full((size, size), bg, np.uint8)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 2)
    return img


def test_scharr_edges_density_matched_to_canny():
    img = _square_img()
    canny = cv2.Canny(cv2.GaussianBlur(img, (0, 0), 1.0), 60, 160)
    r_c1 = float((canny > 0).mean())
    edges = ep._scharr_edges(img, r_c1)
    assert edges.dtype == np.uint8 and edges.shape == img.shape
    r_c2 = float((edges > 0).mean())
    # 밀도가 C1 근처(±5%p 절대)로 맞춰져야 동등 비교 가능.
    assert abs(r_c2 - r_c1) <= 0.05, (r_c1, r_c2)


def test_scharr_edges_clamp_low_density():
    # 거의 균일한 이미지 → r_c1 매우 작아도 하한 3% clamp 로 edge 가 생긴다.
    img = np.full((200, 200), 110, np.uint8)
    edges = ep._scharr_edges(img, 0.0)
    assert float((edges > 0).mean()) > 0.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -q`
Expected: FAIL — `No module named 'poc.workflow_2.ensemble_proposer'`.

- [ ] **Step 3: `ensemble_proposer.py` + `_scharr_edges` 구현**

```python
"""다중 구조 채널 ensemble proposer (C1 Canny + C2 Scharr + C3 orientation-binned).

기존 multi-scale chamfer 엔진(align_key_matcher)을 edge map 만 바꿔 재사용한다.
RRF(순위 기반, 스케일 무관)로 채널 후보를 융합해 top-N + shadow pool 을 낸다.
설계: docs/specs/2026-06-09-ensemble-proposer-design.md.
"""
import cv2
import numpy as np

from poc.workflow_2.align_key_matcher import (
    AlignKeyCandidate, DEFAULT_SCALES, DT_TAU_PX, _collect_candidates,
    _to_grayscale, preprocess_for_matching,
)

# C2: gradient magnitude foreground 밀도를 C1 에 맞춘다(3~15% clamp).
SCHARR_R_MIN = 0.03
SCHARR_R_MAX = 0.15


def _scharr_edges(image, r_c1):
    """Scharr gradient magnitude 를 C1 밀도 매칭 percentile 로 이진화한 edge map(uint8 0/255).

    threshold = (1 - r) 백분위, r = clamp(r_c1, 3%~15%). Otsu 대신 밀도 매칭 →
    C1(Canny) 과 foreground ratio 를 맞춰 채널별 mean_dt 스케일을 동등하게.
    """
    gray = _to_grayscale(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    r = float(min(SCHARR_R_MAX, max(SCHARR_R_MIN, r_c1)))
    thr = float(np.percentile(mag, 100.0 * (1.0 - r)))
    edges = (mag >= thr).astype(np.uint8) * 255
    return edges
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/ensemble_proposer.py poc/workflow_2/test_ensemble_proposer.py
git commit -m "workflow_2(proposer): C2 Scharr gradient channel (density-matched percentile)"
```

---

## Task 3: C3 orientation-binned directional chamfer

방향 일치까지 요구해 점수면을 뾰족하게 — proposer_wall 직격. 8 bin, 0–180° half-angle, bin별 edge-count weighted mean dt.

**Files:**
- Modify: `poc/workflow_2/ensemble_proposer.py`
- Test: `poc/workflow_2/test_ensemble_proposer.py`

- [ ] **Step 1: 실패 테스트 추가**

`test_ensemble_proposer.py`에 추가:

```python
def test_orientation_bin_edges_shape_and_count():
    img = _square_img()
    bins = ep._orientation_bin_edges(img, n_bins=8)
    assert len(bins) == 8
    assert all(b.shape == img.shape and b.dtype == np.uint8 for b in bins)
    # 사각 윤곽 → 가로/세로 방향 bin 에 edge 가 몰린다(전체 edge 합 > 0).
    assert sum(int((b > 0).sum()) for b in bins) > 0


def test_directional_chamfer_peak_at_true_location():
    # template = 사각 윤곽 crop, frame 에 같은 윤곽을 (dx,dy) 평행이동 배치 →
    # directional chamfer score map 의 peak 가 그 위치 근처여야 한다.
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    smap, (tw, th) = ep._directional_chamfer_score_map(tpl, frame, scale=1.0, n_bins=8)
    assert smap is not None
    y, x = np.unravel_index(int(np.argmax(smap)), smap.shape)
    cx, cy = x + tw // 2, y + th // 2
    # true 중심 ≈ (120+30, 90+30) = (150,120). 허용 8px.
    assert abs(cx - 150) <= 8 and abs(cy - 120) <= 8, (cx, cy)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -k "orientation or directional" -q`
Expected: FAIL — `_orientation_bin_edges` / `_directional_chamfer_score_map` 없음.

- [ ] **Step 3: 구현 추가**

`ensemble_proposer.py`에 추가:

```python
N_ORIENT_BINS = 8


def _orientation_bin_edges(image, n_bins=N_ORIENT_BINS, r_c1=None):
    """edge 픽셀을 gradient 방향(0~180° half-angle) n_bins 로 나눈 binary map 리스트.

    edge 위치는 C2 와 동일 밀도 매칭(_scharr_edges). 각 edge 픽셀을 unsigned gradient
    각도(0~180)로 bin 분류 → bin 별 0/255 map. polarity 불변(SEM/OM 밝기 반전 강건).
    """
    gray = _to_grayscale(image)
    if r_c1 is None:
        canny = preprocess_for_matching(gray)[0]
        r_c1 = float((canny > 0).mean())
    edges = _scharr_edges(gray, r_c1) > 0
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0          # 0~180 half-angle.
    bin_idx = np.minimum((ang / (180.0 / n_bins)).astype(np.int32), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = (edges & (bin_idx == b)).astype(np.uint8) * 255
        out.append(m)
    return out


def _directional_chamfer_score_map(template_gray, frame_gray, *, scale, n_bins=N_ORIENT_BINS):
    """방향 분할 chamfer score map = exp(-weighted_mean_dt/DT_TAU_PX).

    bin 별: same-bin frame edge 의 DT 위에 same-bin template edge 를 슬라이드한 mean_dt.
    bin 들을 template bin edge-count 가중 평균(weighted mean) → exp. min/sum 대신 weighted
    mean: 한 방향만 맞아도 과대평가(min)·edge 많은 bin 지배(sum) 회피, 기존 평균거리 규약 일관.
    """
    from poc.workflow_2.align_key_matcher import _scaled_edges
    t_bins = _orientation_bin_edges(template_gray, n_bins)
    f_bins = _orientation_bin_edges(frame_gray, n_bins)
    num = None        # Σ_bin (edge_count_b * mean_dt_map_b)
    den = 0.0         # Σ_bin edge_count_b
    out_size = None
    for tb, fb in zip(t_bins, f_bins):
        tb_s = _scaled_edges(tb, scale)
        th, tw = tb_s.shape[:2]
        fh, fw = fb.shape[:2]
        if th >= fh or tw >= fw:
            return None, (tw, th)
        mask = (tb_s > 0).astype(np.float32)
        cnt = float(mask.sum())
        if cnt <= 0:
            continue
        f_dt = cv2.distanceTransform(cv2.bitwise_not(fb), cv2.DIST_L2, 5).astype(np.float32)
        mean_dt = cv2.matchTemplate(f_dt, mask, cv2.TM_CCORR) / cnt
        num = mean_dt * cnt if num is None else num + mean_dt * cnt
        den += cnt
        out_size = (tw, th)
    if num is None or den <= 0:
        return None, (out_size or (0, 0))
    weighted_mean_dt = num / den
    return np.exp(-weighted_mean_dt / DT_TAU_PX).astype(np.float32), out_size
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -k "orientation or directional" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/ensemble_proposer.py poc/workflow_2/test_ensemble_proposer.py
git commit -m "workflow_2(proposer): C3 orientation-binned directional chamfer (8 bins, weighted mean dt)"
```

---

## Task 4: RRF 융합 + `compute_ensemble_candidates`

3채널 solo 후보를 RRF(스케일 무관 순위)로 융합 → top-8 + shadow-24, per-channel solo 보존(attribution).

**Files:**
- Modify: `poc/workflow_2/ensemble_proposer.py`
- Test: `poc/workflow_2/test_ensemble_proposer.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_rrf_fuse_is_scale_free_rank_based():
    # 채널 A 점수 스케일이 100배 커도 RRF 는 순위만 보므로 결과가 스케일에 불변.
    A = [ep._Cand(xy=(10, 10), score=900.0), ep._Cand(xy=(50, 50), score=100.0)]
    B = [ep._Cand(xy=(10, 10), score=9.0), ep._Cand(xy=(80, 80), score=1.0)]
    fused = ep._rrf_fuse([A, B], k0=10, match_radius=5, top_n=3)
    # (10,10) 은 두 채널 모두 rank1 → 최상위.
    assert fused[0].xy == (10, 10)
    A2 = [ep._Cand(xy=(10, 10), score=9.0), ep._Cand(xy=(50, 50), score=1.0)]
    fused2 = ep._rrf_fuse([A2, B], k0=10, match_radius=5, top_n=3)
    assert fused2[0].xy == (10, 10)   # 점수 스케일 바뀌어도 순위 동일.


def test_ensemble_candidates_returns_topn_and_shadow():
    frame = _square_img(size=240, box=(120, 90, 60, 60))
    tpl = _square_img(size=80, box=(10, 10, 60, 60))
    res = ep.compute_ensemble_candidates(tpl, frame, top_n=8, shadow_n=24)
    assert len(res.fused) <= 24 and len(res.fused) >= 1
    assert res.top_n_count == 8
    assert set(res.solo.keys()) == {"canny", "scharr", "orient"}
    # 진짜 위치(≈150,120)가 fused 후보 중에 있다.
    assert any(abs(c.xy[0] - 150) <= 8 and abs(c.xy[1] - 120) <= 8 for c in res.fused)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -k "rrf or ensemble_candidates" -q`
Expected: FAIL — `_Cand`/`_rrf_fuse`/`compute_ensemble_candidates` 없음.

- [ ] **Step 3: 구현 추가**

```python
from dataclasses import dataclass, field


@dataclass
class _Cand:
    """채널/융합 후보 — xy(template 중심, frame 좌표) + score(+scale)."""
    xy: tuple
    score: float
    scale: float = 1.0


@dataclass
class EnsembleResult:
    """ensemble 결과 — fused(RRF 정렬, shadow_n 까지) + top_n_count + per-channel solo."""
    fused: list                       # list[_Cand] RRF 내림차순
    top_n_count: int
    solo: dict = field(default_factory=dict)   # {"canny"|"scharr"|"orient": list[_Cand]}


# RRF/NMS 상수 (cold-start, 오피스 sweep 으로 보정 — spec §6).
RRF_K0 = 10
SOLO_TOP_K = 24
SHADOW_N = 24


def _channel_solo_candidates(template_gray, frame_gray, channel, *, scales=DEFAULT_SCALES,
                             top_k=SOLO_TOP_K):
    """채널 한 개의 solo 후보(top_k). channel: 'canny'|'scharr'|'orient'."""
    g = _to_grayscale(template_gray)
    f = _to_grayscale(frame_gray)
    if channel in ("canny", "scharr"):
        if channel == "canny":
            t_edges, _ = preprocess_for_matching(g)
            _, f_dt = preprocess_for_matching(f)
        else:
            r_c1 = float((preprocess_for_matching(f)[0] > 0).mean())
            t_edges = _scharr_edges(g, r_c1)
            f_edges = _scharr_edges(f, r_c1)
            f_dt = cv2.distanceTransform(cv2.bitwise_not(f_edges), cv2.DIST_L2, 5).astype(np.float32)
        cands = _collect_candidates(t_edges, f_dt, scales=scales, top_n=top_k)
        return [_Cand(xy=c.xy, score=c.chamfer_score, scale=c.scale) for c in cands]
    # orient: per-scale directional score map → peaks.
    from poc.workflow_2.align_key_matcher import _extract_peaks
    collected = []
    for scale in scales:
        smap, (tw, th) = _directional_chamfer_score_map(g, f, scale=scale)
        if smap is None:
            continue
        nms_r = max(4, int(min(tw, th) * 0.5))
        for s, cx, cy in _extract_peaks(smap, tw, th, max_peaks=top_k, min_score=0.0, nms_radius=nms_r):
            collected.append(_Cand(xy=(cx, cy), score=float(s), scale=float(scale)))
    collected.sort(key=lambda c: c.score, reverse=True)
    return collected[:top_k]


def _rrf_fuse(channel_lists, *, k0=RRF_K0, match_radius=8, top_n=SHADOW_N):
    """채널별 후보 리스트를 RRF 로 융합. fused(c) = Σ_채널 1/(k0 + rank).

    채널 간 후보는 center 거리 <= match_radius 면 동일 후보로 묶는다(가장 높은 score member 가 대표).
    반환 list[_Cand] (fused score 내림차순, top_n 까지). 스케일 무관(순위 기반).
    """
    clusters = []  # {"xy", "score"(대표 chamfer), "rrf"}
    for ch_list in channel_lists:
        ranked = sorted(ch_list, key=lambda c: c.score, reverse=True)
        for rank, cand in enumerate(ranked, 1):
            hit = next((cl for cl in clusters
                        if abs(cl["xy"][0] - cand.xy[0]) <= match_radius
                        and abs(cl["xy"][1] - cand.xy[1]) <= match_radius), None)
            contrib = 1.0 / (k0 + rank)
            if hit is None:
                clusters.append({"xy": cand.xy, "score": cand.score, "rrf": contrib})
            else:
                hit["rrf"] += contrib
                if cand.score > hit["score"]:
                    hit["xy"], hit["score"] = cand.xy, cand.score
    clusters.sort(key=lambda cl: cl["rrf"], reverse=True)
    return [_Cand(xy=cl["xy"], score=cl["rrf"]) for cl in clusters[:top_n]]


def compute_ensemble_candidates(template_gray, frame_gray, *, top_n=8, shadow_n=SHADOW_N,
                                k0=RRF_K0, scales=DEFAULT_SCALES):
    """3채널 ensemble proposer → EnsembleResult.

    각 채널 solo top-K → RRF 융합(shadow_n 까지) + per-channel solo 보존(attribution).
    fused 의 앞 top_n 이 KPI 후보, 나머지는 shadow(진단). match/NMS 반경은 template 짧은 변 비례.
    """
    th, tw = _to_grayscale(template_gray).shape[:2]
    short = max(1, min(tw, th))
    match_r = max(8, int(0.05 * short))
    solo = {ch: _channel_solo_candidates(template_gray, frame_gray, ch, scales=scales)
            for ch in ("canny", "scharr", "orient")}
    fused = _rrf_fuse(list(solo.values()), k0=k0, match_radius=match_r, top_n=shadow_n)
    return EnsembleResult(fused=fused, top_n_count=top_n, solo=solo)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python -m pytest poc/workflow_2/test_ensemble_proposer.py -q`
Expected: PASS (전체 6 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/ensemble_proposer.py poc/workflow_2/test_ensemble_proposer.py
git commit -m "workflow_2(proposer): RRF fusion + compute_ensemble_candidates (top-8 + shadow-24 + per-channel solo)"
```

---

## Task 5: proposer-recall A/B 러너

baseline(C1) vs ensemble 의 proposer recall@{8,16,24}를 `golden_localization_eval_cond` 재사용으로 정직하게 A/B. 순수 헬퍼(recall@N/attribution)부터 TDD.

**Files:**
- Create: `poc/workflow_2/proposer_recall_ab.py`
- Test: `poc/workflow_2/test_proposer_recall_ab.py`

- [ ] **Step 1: 실패 테스트 작성 (순수 헬퍼)**

`test_proposer_recall_ab.py`:

```python
"""proposer recall@N / attribution 순수 헬퍼 테스트 — Mac 실행 가능."""
import poc.workflow_2.proposer_recall_ab as pab
from poc.workflow_2.ensemble_proposer import _Cand


def test_gt_rank_in_candidates_hits_within_tol():
    # GT=(100,100), offset=(0,0), short=200, tol=0.20 → 허용 40px. 2번째 후보가 hit.
    cands = [_Cand(xy=(180, 180), score=0.9), _Cand(xy=(115, 100), score=0.8)]
    rank = pab._gt_rank(cands, gt_xy=(100, 100), offset=(0, 0), short=200, tol=0.20)
    assert rank == 2


def test_gt_rank_none_when_all_far():
    cands = [_Cand(xy=(10, 10), score=0.9)]
    assert pab._gt_rank(cands, gt_xy=(200, 200), offset=(0, 0), short=200, tol=0.20) is None


def test_recall_at_counts_rank_within_n():
    ranks = [1, 3, None, 9, 16]
    # @8 → rank<=8 인 것: 1,3 → 2/5=0.4
    assert pab._recall_at(ranks, 8) == 0.4
    # @16 → 1,3,9,16 → 4/5=0.8
    assert pab._recall_at(ranks, 16) == 0.8


def test_offset_applied_before_compare():
    # offset (40,0) 적용 시 후보 (60,100)+offset=(100,100)=GT → hit.
    cands = [_Cand(xy=(60, 100), score=0.9)]
    assert pab._gt_rank(cands, gt_xy=(100, 100), offset=(40, 0), short=200, tol=0.20) == 1
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python -m pytest poc/workflow_2/test_proposer_recall_ab.py -q`
Expected: FAIL — `No module named 'poc.workflow_2.proposer_recall_ab'`.

- [ ] **Step 3: 순수 헬퍼 구현**

`proposer_recall_ab.py`:

```python
"""proposer recall A/B — baseline(C1 chamfer) vs ensemble 의 후보 recall@{8,16,24}.

proposer recall 만 격리: 후보 xy+align_offset 이 GT(cond crosshair) 허용오차 내인지(=membership)만
본다. final score/decision·reranker 재정렬 금지(proposer/reranker 섞임 방지). modality 라우팅·
box template·cond GT 는 golden_localization_eval_cond 재사용. 설계: docs/specs/2026-06-09-...md.
실행(오피스): uv run python poc/workflow_2/proposer_recall_ab.py
"""
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import math

RECALL_NS = (8, 16, 24)


def _gt_rank(cands, *, gt_xy, offset, short, tol):
    """후보 리스트(정렬됨)에서 (xy+offset) 이 GT 허용오차(tol·short) 내인 첫 1-base rank. 없으면 None."""
    dx, dy = offset
    lim = tol * short
    for i, c in enumerate(cands, 1):
        ax, ay = c.xy[0] + dx, c.xy[1] + dy
        if math.hypot(ax - gt_xy[0], ay - gt_xy[1]) <= lim:
            return i
    return None


def _recall_at(ranks, n):
    """rank 리스트(None=miss)에서 rank<=n 비율."""
    if not ranks:
        return 0.0
    return round(sum(1 for r in ranks if r is not None and r <= n) / len(ranks), 3)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python -m pytest poc/workflow_2/test_proposer_recall_ab.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/proposer_recall_ab.py poc/workflow_2/test_proposer_recall_ab.py
git commit -m "workflow_2(proposer): recall@N / gt_rank pure helpers (proposer-recall A/B)"
```

---

## Task 6: A/B 러너 본체 (오피스 실행)

순수 헬퍼 위에 frame 순회 + baseline/ensemble 후보 생성 + 표 출력. localization eval 재사용.

**Files:**
- Modify: `poc/workflow_2/proposer_recall_ab.py`

- [ ] **Step 1: 러너 구현 추가**

`proposer_recall_ab.py`에 추가 (localization eval 재사용 — modality 라우팅·box template·cond GT):

```python
import json
from pathlib import Path

import numpy as np

from poc.workflow_2 import DEBUG_IMAGE_DIR
from poc.workflow_2.align_fail_assets import iter_msr_images, load_gray
from poc.workflow_2.align_key_matcher import (
    DEFAULT_SCALES, _collect_candidates, _to_grayscale, preprocess_for_matching,
)
from poc.workflow_2.align_similarity import COMPARE_SCALES, GT_TOL_NORM
from poc.workflow_2.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_2.cond_file import load_cond
from poc.workflow_2.ensemble_proposer import _Cand, compute_ensemble_candidates
from poc.workflow_2 import golden_localization_eval as gle
import poc.workflow_2.golden_localization_eval_cond as glec
from poc.workflow_2.align_point_correction import _tool_label
from poc.workflow_1.util.time_utils import make_timestamp_tag

OUTPUT_ROOT = DEBUG_IMAGE_DIR / "proposer_recall_ab"


def _baseline_candidates(template_gray, frame_gray, *, top_k=24):
    """C1-only(canny) 후보 top_k — ensemble 비교 기준."""
    t_edges, _ = preprocess_for_matching(_to_grayscale(template_gray))
    _, f_dt = preprocess_for_matching(_to_grayscale(frame_gray))
    cands = _collect_candidates(t_edges, f_dt, scales=COMPARE_SCALES, top_n=top_k)
    return [_Cand(xy=c.xy, score=c.chamfer_score, scale=c.scale) for c in cands]


def run():
    """baseline vs ensemble proposer recall A/B (인자 없음). 반환 success|no_data."""
    root_env = os.getenv("ALIGN_GOLDEN_ROOT")
    root = Path(root_env) if root_env else glec.GOLDEN_ROOT
    recipes = gle._collect_recipes(root) if root.is_dir() else []
    if not recipes:
        print(f"[ERROR] golden 데이터 없음: {root}")
        return "no_data"
    out_dir = OUTPUT_ROOT / make_timestamp_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ranks, ens_ranks = [], []
    solo_ranks = {"canny": [], "scharr": [], "orient": []}
    for assets in recipes:
        if assets is None:
            continue
        try:
            _center, box_tpls = glec._build_offset_templates_cond(assets)
        except Exception as exc:
            print(f"[WARNING] template 빌드 실패 {assets.recipe_id}: {exc}")
            continue
        available = {m for m, v in box_tpls.items() if v is not None}
        if not available:
            continue
        for p in iter_msr_images(assets):
            if _tool_label(p.name) != "S":
                continue
            cond = load_cond(p)
            routed = glec._route_modality(cond, available)
            if routed is None or box_tpls.get(routed) is None:
                continue
            tpl, (dx, dy) = box_tpls[routed]
            if not (cond and cond.crosshair_xy is not None):
                continue
            gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
            gt_xy = (int(round(gx)), int(round(gy)))
            try:
                gray_raw = load_gray(p)
            except Exception:
                continue
            frame = clean_image(gray_raw, cond)        # crosshair 제거(box__inpaint 경로와 동일).
            t_gray = tpl.raw_image
            short = max(1, min(t_gray.shape[0], t_gray.shape[1]))
            base = _baseline_candidates(t_gray, frame)
            ens = compute_ensemble_candidates(t_gray, frame)
            base_ranks.append(_gt_rank(base, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))
            ens_ranks.append(_gt_rank(ens.fused, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))
            for ch, lst in ens.solo.items():
                solo_ranks[ch].append(_gt_rank(lst, gt_xy=gt_xy, offset=(dx, dy), short=short, tol=GT_TOL_NORM))

    n = len(base_ranks)
    if not n:
        print("[ERROR] 처리된 S 프레임 없음.")
        return "no_data"
    summary = {"n": n, "GT_TOL_NORM": GT_TOL_NORM,
               "baseline": {f"recall@{k}": _recall_at(base_ranks, k) for k in RECALL_NS},
               "ensemble": {f"recall@{k}": _recall_at(ens_ranks, k) for k in RECALL_NS},
               "solo": {ch: {f"recall@{k}": _recall_at(r, k) for k in RECALL_NS}
                        for ch, r in solo_ranks.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n[INFO] === proposer recall A/B (S {n}장, tol={GT_TOL_NORM}) ===")
    print(f"  {'variant':<12} {'recall@8':>9} {'recall@16':>10} {'recall@24':>10}")
    for name, d in (("baseline(C1)", summary["baseline"]), ("ensemble", summary["ensemble"])):
        print(f"  {name:<12} {d['recall@8']:>9} {d['recall@16']:>10} {d['recall@24']:>10}")
    for ch, d in summary["solo"].items():
        print(f"  solo:{ch:<7} {d['recall@8']:>9} {d['recall@16']:>10} {d['recall@24']:>10}")
    lift8 = round(summary["ensemble"]["recall@8"] - summary["baseline"]["recall@8"], 3)
    lift24 = round(summary["ensemble"]["recall@24"] - summary["baseline"]["recall@24"], 3)
    print(f"\n  >>> budgeted recall@8 lift = {lift8:+}  | shadow recall@24 lift = {lift24:+}")
    print("      @24↑·@8↑ = ensemble 효과 / @24↑·@8≈ = fusion 부족 / @24≈ = 채널 무효")
    print(f"[INFO] 완료: {out_dir}")
    return "success"


if __name__ == "__main__":
    raise SystemExit(0 if run() == "success" else 1)
```

- [ ] **Step 2: import/구문 sanity (Mac, 데이터 없어도 import 검증)**

Run: `uv run python -c "import poc.workflow_2.proposer_recall_ab as m; print('import OK', hasattr(m,'run'))"`
Expected: `import OK True`.

- [ ] **Step 3: 빈 데이터 경로 확인 (Mac)**

Run: `uv run python poc/workflow_2/proposer_recall_ab.py`
Expected: `[ERROR] golden 데이터 없음: ...` + 종료코드 1 (Mac엔 golden 데이터 없음 — 정상).

- [ ] **Step 4: 전체 회귀**

Run: `uv run python -m pytest poc/workflow_2/ -q`
Expected: PASS (기존 102 + 신규 단위테스트 모두 통과).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/proposer_recall_ab.py
git commit -m "workflow_2(proposer): proposer-recall A/B runner (baseline C1 vs ensemble, recall@8/16/24 + attribution)"
```

---

## Task 7: 오피스 A/B 실행 + 판정 (수동)

**Files:** 없음 (실행/관찰).

- [ ] **Step 1: 오피스에서 실행**

Run: `uv run python poc/workflow_2/proposer_recall_ab.py`
관찰: `recall@8/16/24` 표 (baseline vs ensemble vs solo:canny/scharr/orient).

- [ ] **Step 2: 판정 기록 (digest 로 회신)**

- `budgeted recall@8 lift`: ensemble이 0.557(현 box__inpaint gt_in_topk) 대비 +면 채택 후보.
- shadow `recall@24`: ↑면 proposer 다양성 확보(채널 추가 효과), ≈면 채널 무효.
- solo 채널별 recall: 어느 채널(scharr/orient)이 GT를 가장 많이 넣나 → 다음 iteration 우선순위.
- 실패모드 점검: FM1(directional bin jitter) — orient solo가 유독 낮으면 ±1 bin tolerance 재측정 필요. FM2(clutter) — scharr solo가 높은데 fused가 안 오르면 false-consensus 의심.

---

## Self-Review

**Spec coverage:**
- C1/C2/C3 채널 → Task 1(리팩터)/2(C2)/3(C3). ✅
- RRF 융합(k0=10, 매칭반경 0.05·short, post-NMS) → Task 4. ✅ (post-fusion global NMS는 `_rrf_fuse`의 match_radius 클러스터링이 중복 병합 역할 — spec의 0.25·short 별도 NMS는 클러스터 대표 1개라 사실상 충족; 오피스 결과로 과병합 보이면 Task 7 후속에서 분리 NMS 추가).
- shadow@24 + per-channel solo → Task 4(EnsembleResult.solo)/Task 6. ✅
- recall@8/16/24, GT_TOL 0.20, offset 적용, rerank 재정렬 금지 → Task 5/6. ✅
- 채널 attribution(solo top-24 credit) → Task 6(solo_ranks). ✅
- 실패모드 FM1/FM2 점검 → Task 7. ✅

**Placeholder scan:** 모든 step에 실제 코드/명령/기대출력 있음. "적절히 처리" 류 없음. ✅

**Type consistency:** `_Cand`(xy,score,scale), `EnsembleResult`(fused,top_n_count,solo), `_collect_candidates`(Task1 정의)→Task4/6 사용, `_gt_rank`/`_recall_at`(Task5)→Task6 사용. `compute_ensemble_candidates` 시그니처 일관. ✅

**미세 갭:** spec의 "post-fusion global NMS 0.25·short"는 Task4에서 RRF 클러스터링(match_radius)으로 흡수 — 명시적 별도 NMS는 생략(YAGNI). 과병합 징후 시 Task7 후 추가. 기록함.
