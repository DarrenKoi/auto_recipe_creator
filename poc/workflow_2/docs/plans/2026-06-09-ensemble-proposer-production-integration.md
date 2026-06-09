# Ensemble Proposer 프로덕션 통합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ensemble proposer(recall@8 0.557→0.698)를 생산 매칭에 연결하는 신규 `compute_align_key_score_ensemble`를 추가하되, 기존 `compute_align_key_score`는 출력 비트 동일하게 보존한다.

**Architecture:** `compute_align_key_score`의 전처리 블록과 best-선택-이후 블록을 behavior-preserving하게 두 공유 헬퍼(`_prepare_match_inputs`, `_finalize_match` + `_no_candidate_result`)로 추출한다. 신규 함수는 ensemble proposer → chamfer rescore(`_rescore_positions_to_candidates`) → ORB pool-rerank → 공유 finalize 순으로 동작한다. 기존 `test_align_key_match.py` 10/10이 추출의 회귀 가드.

**Tech Stack:** Python, OpenCV(cv2), numpy. 기존 `poc/workflow_2/align_key_matcher.py` + `ensemble_proposer.py`. 테스트는 pytest(합성 데이터, Mac).

**제약(전 작업 공통):** CLI 인자 금지 / Korean docstring / `[INFO]·[ERROR]·[WARNING]` print(로깅 모듈 금지) / `from __future__` 금지 / 절대 임포트(`from poc.workflow_2.xxx`) / main 직접 commit(브랜치 없음) / commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**비범위(YAGNI):** 호출자 전환(`compare_align_images` 등 한 줄 교체), 오피스 e2e, live broad-scan 통합, 캐싱/병렬화.

---

## File Structure

- **Modify** `poc/workflow_2/align_key_matcher.py`
  - 추출: `_prepare_match_inputs`(전처리), `_no_candidate_result`(후보 0 결과), `_finalize_match`(distinctiveness+decision+overlay+result).
  - 리팩터: `compute_align_key_score`가 위 헬퍼를 호출하도록(동작 불변).
  - 신규: `_rescore_positions_to_candidates`(rescore), `compute_align_key_score_ensemble`(통합 진입점).
- **Create** `poc/workflow_2/test_align_key_score_ensemble.py` — rescore/rerank/통합 단위테스트.
- **회귀 가드** `poc/workflow_2/test_align_key_match.py`(기존, 무수정) — 추출 후에도 통과.

추출 대상 원본 라인(작업 시점 기준): 전처리 `compute_align_key_score:564-629`, 후보 0 `636-649`, finalize `651-715`.

---

### Task 1: 전처리 블록을 `_prepare_match_inputs`로 추출 (behavior-preserving)

**Files:**
- Modify: `poc/workflow_2/align_key_matcher.py` (신규 헬퍼 + `compute_align_key_score` head 교체)
- Test(regression): `poc/workflow_2/test_align_key_match.py` (무수정)

- [ ] **Step 1: 기존 회귀 테스트가 통과하는 baseline 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_match.py -v`
Expected: PASS (기존 10케이스). 이 결과가 추출의 기준선.

- [ ] **Step 2: `_prepare_match_inputs` 헬퍼 추가**

`compute_align_key_score` 정의(현재 543행) **바로 위**에 추가. 본문은 현재 `compute_align_key_score`의 564~629행을 그대로 옮긴 것(동작 불변).

```python
def _prepare_match_inputs(
    template: AlignKeyTemplate,
    frame: np.ndarray,
    *,
    frame_nm_per_pixel: float | None,
    roi_hint: tuple[int, int, int, int] | None,
    scales: tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], tuple[int, int]]:
    """매칭 전처리 — gray + scale 해석 + ROI crop + frame_dt.

    반환 (gray_frame, frame_dt, scales, roi_origin). compute_align_key_score 와
    compute_align_key_score_ensemble 가 공유한다. 추출 전 동작과 동일.
    """
    gray_frame = _to_grayscale(frame)

    # 스케일 결정 — ROI 검증의 최소 크롭 크기 산출에 필요하므로 먼저.
    if scales is not None:
        if not scales:
            raise ValueError("scales override must be a non-empty tuple")
        if any(s <= 0 for s in scales):
            raise ValueError(f"all scales must be positive, got {scales}")
        scales = tuple(float(s) for s in scales)
    elif (
        template.nm_per_pixel is not None
        and frame_nm_per_pixel is not None
        and frame_nm_per_pixel > 0
    ):
        single_scale = template.nm_per_pixel / frame_nm_per_pixel
        if single_scale <= 0:
            raise ValueError(
                f"resolved scale must be positive, got {single_scale} "
                f"(template.nm_per_pixel={template.nm_per_pixel}, "
                f"frame_nm_per_pixel={frame_nm_per_pixel})"
            )
        scales = (float(single_scale),)
    else:
        scales = DEFAULT_SCALES

    # 가능한 최소 템플릿 크기 (모든 스케일 중 최소).
    th0, tw0 = template.edge_map.shape[:2]
    min_scale = min(scales)
    min_th = max(8, int(round(th0 * min_scale)))
    min_tw = max(8, int(round(tw0 * min_scale)))

    roi_origin = (0, 0)
    if roi_hint is not None:
        if not (isinstance(roi_hint, tuple) and len(roi_hint) == 4):
            raise ValueError(
                f"roi_hint must be a 4-tuple (x, y, w, h), got {roi_hint!r}"
            )
        rx, ry, rw, rh = (int(v) for v in roi_hint)
        if rw <= 0 or rh <= 0:
            raise ValueError(
                f"roi_hint width/height must be positive, got w={rw}, h={rh}"
            )
        fh, fw = gray_frame.shape[:2]
        x0 = max(0, rx)
        y0 = max(0, ry)
        x1 = min(fw, rx + rw)
        y1 = min(fh, ry + rh)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"roi_hint {(rx, ry, rw, rh)} does not intersect frame "
                f"of size {(fw, fh)}"
            )
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_w <= min_tw or crop_h <= min_th:
            raise ValueError(
                f"roi_hint crop {(crop_w, crop_h)} is smaller than the "
                f"smallest scaled template {(min_tw, min_th)} "
                f"(min_scale={min_scale:.3f}); widen the ROI or skip the hint"
            )
        gray_frame = gray_frame[y0:y1, x0:x1].copy()
        roi_origin = (x0, y0)

    _frame_edges, frame_dt = preprocess_for_matching(gray_frame)
    return gray_frame, frame_dt, scales, roi_origin
```

- [ ] **Step 3: `compute_align_key_score` head를 헬퍼 호출로 교체**

`compute_align_key_score` 본문에서 `gray_frame = _to_grayscale(frame)`(564행)부터 `_frame_edges, frame_dt = preprocess_for_matching(gray_frame)`(629행)까지를 아래 한 줄로 교체. 그 아래 `candidates = compute_chamfer_candidates(...)`부터는 그대로 둔다.

```python
    gray_frame, frame_dt, scales, roi_origin = _prepare_match_inputs(
        template,
        frame,
        frame_nm_per_pixel=frame_nm_per_pixel,
        roi_hint=roi_hint,
        scales=scales,
    )
```

- [ ] **Step 4: 회귀 테스트로 동작 불변 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_match.py -v`
Expected: PASS — Step 1과 동일한 케이스 수/결과. 한 건이라도 달라지면 추출이 동작을 바꾼 것.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/align_key_matcher.py
git commit -m "refactor(workflow_2): extract _prepare_match_inputs from compute_align_key_score (behavior-preserving)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_no_candidate_result` + `_finalize_match` 추출 (behavior-preserving)

**Files:**
- Modify: `poc/workflow_2/align_key_matcher.py`
- Test(regression): `poc/workflow_2/test_align_key_match.py` (무수정)

- [ ] **Step 1: `_no_candidate_result` 헬퍼 추가**

`compute_align_key_score` 정의 위(Task 1 헬퍼 근처)에 추가. 본문은 현재 636~649행과 동일.

```python
def _no_candidate_result(
    frame: np.ndarray,
    frame_dt: np.ndarray,
    template: AlignKeyTemplate,
    roi_origin: tuple[int, int],
) -> AlignKeyMatchResult:
    """후보 0개 — 기존 동작(중앙, 점수 0, reject_reason='no_candidates')을 보존."""
    fh, fw = frame_dt.shape[:2]
    center = (fw // 2 + roi_origin[0], fh // 2 + roi_origin[1])
    t_h, t_w = template.edge_map.shape[:2]
    overlay = _render_overlay(
        frame, cx=center[0], cy=center[1], tw=t_w, th=t_h,
        decision="low", score=0.0, chamfer=0.0, orb=0.0, scale=1.0,
    )
    return AlignKeyMatchResult(
        score=0.0, chamfer_score=0.0, orb_inlier_ratio=0.0,
        best_xy=center, best_scale=1.0, decision="low", debug_overlay=overlay,
        candidates=[], reject_reason="no_candidates", distinctive=False,
    )
```

- [ ] **Step 2: `_finalize_match` 헬퍼 추가**

본문은 현재 651~715행과 동일하되, `best`/`chamfer_score`/`orb_ratio`를 인자로 받는다(호출부가 best 선택·ORB 실행 방식만 다르고 마감은 공유).

```python
def _finalize_match(
    best_cand: AlignKeyCandidate,
    candidates: list,
    frame: np.ndarray,
    template: AlignKeyTemplate,
    policy: MatchPolicy,
    roi_origin: tuple[int, int],
    *,
    chamfer_score: float,
    orb_ratio: float,
) -> AlignKeyMatchResult:
    """best 선택 이후 공유 마감 — distinctiveness + score/decision + overlay + result.

    distinctiveness 는 *chamfer 집합* 기준(가장 강한 chamfer peak 이 2nd 대비 유일한가).
    candidates 의 xy 는 roi-local 이며 여기서 roi_origin 을 가산해 절대좌표로 만든다.
    """
    chamfer_sorted = sorted(candidates, key=lambda c: c.chamfer_score, reverse=True)
    ch_best = chamfer_sorted[0]
    ch_second = chamfer_sorted[1] if len(chamfer_sorted) > 1 else None
    second_score = float(ch_second.chamfer_score) if ch_second is not None else None
    score_gap = (
        float(ch_best.chamfer_score - ch_second.chamfer_score) if ch_second is not None else None
    )
    second_ratio = (
        float(ch_second.chamfer_score / ch_best.chamfer_score)
        if ch_second is not None and ch_best.chamfer_score > 0 else None
    )
    distinctive = True
    reject_reason: str | None = None
    if ch_second is not None:
        if (score_gap is not None and score_gap < policy.min_distinct_gap) or (
            second_ratio is not None and second_ratio > policy.max_second_ratio
        ):
            distinctive = False
            reject_reason = "not_distinctive"

    cx, cy = best_cand.xy
    best_scale = best_cand.scale
    tw, th = best_cand.template_size
    best_cand.orb_inlier_ratio = float(orb_ratio)

    score = policy.chamfer_weight * chamfer_score + policy.orb_weight * orb_ratio
    decision = _decision_for_score(score, policy)

    # 후보 좌표를 roi 절대 좌표로 환산 (best_xy 의미는 기존과 동일).
    abs_xy = (cx + roi_origin[0], cy + roi_origin[1])
    for c in candidates:
        c.xy = (c.xy[0] + roi_origin[0], c.xy[1] + roi_origin[1])

    overlay = _render_overlay(
        frame, cx=abs_xy[0], cy=abs_xy[1], tw=tw, th=th,
        decision=decision, score=score, chamfer=chamfer_score, orb=orb_ratio, scale=best_scale,
    )

    return AlignKeyMatchResult(
        score=float(score),
        chamfer_score=float(chamfer_score),
        orb_inlier_ratio=float(orb_ratio),
        best_xy=abs_xy,
        best_scale=float(best_scale),
        decision=decision,
        debug_overlay=overlay,
        candidates=candidates,
        second_score=second_score,
        score_gap=score_gap,
        second_ratio=second_ratio,
        distinctive=distinctive,
        reject_reason=reject_reason,
    )
```

- [ ] **Step 3: `compute_align_key_score` 꼬리를 헬퍼 호출로 교체**

현재 636~649행(후보 0 처리)을 교체:

```python
    if not candidates:
        return _no_candidate_result(frame, frame_dt, template, roi_origin)
```

현재 651~715행(distinctiveness ~ return) 전체를 교체. best 선택 + ORB(best 1개)만 남기고 마감은 `_finalize_match`에 위임:

```python
    best = candidates[0]
    cx, cy = best.xy
    tw, th = best.template_size
    chamfer_score = best.chamfer_score

    # ORB: best 위치를 중심으로 한 윈도우 vs 템플릿.
    orb_ratio = 0.0
    if chamfer_score > 0.0 and tw > 0 and th > 0:
        crop, _crop_origin = _crop_with_padding(gray_frame, cx, cy, tw, th, pad=1.6)
        orb_ratio, _n_inliers, _n_matches = compute_orb_inlier_ratio(
            template.raw_image, crop
        )

    return _finalize_match(
        best, candidates, frame, template, policy, roi_origin,
        chamfer_score=chamfer_score, orb_ratio=orb_ratio,
    )
```

- [ ] **Step 4: 회귀 테스트로 동작 불변 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_match.py -v`
Expected: PASS — Task 1 Step 1과 동일한 케이스 수/결과.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/align_key_matcher.py
git commit -m "refactor(workflow_2): extract _finalize_match/_no_candidate_result (behavior-preserving)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: rescore 헬퍼 `_rescore_positions_to_candidates`

ensemble proposer가 준 위치(center xy, scale)에 chamfer score를 부여해 기존 후보 자료형(`AlignKeyCandidate`)으로 환원한다.

**Files:**
- Modify: `poc/workflow_2/align_key_matcher.py`
- Create: `poc/workflow_2/test_align_key_score_ensemble.py`

- [ ] **Step 1: 실패 테스트 작성 — rescore가 직접 score map 룩업과 일치**

`poc/workflow_2/test_align_key_score_ensemble.py` 생성:

```python
"""compute_align_key_score_ensemble + rescore/rerank 단위테스트 (합성 데이터, Mac)."""
import numpy as np

from poc.workflow_2 import align_key_matcher as akm
from poc.workflow_2.align_key_matcher import (
    AlignKeyCandidate,
    _chamfer_score_map_at_scale,
    _rescore_positions_to_candidates,
    build_template,
    compute_align_key_score_ensemble,
)


def _synthetic_template_and_frame():
    """프레임 중앙에 박힌 사각형 패턴 + 동일 패턴 템플릿(합성)."""
    rng = np.random.RandomState(0)
    frame = (rng.rand(240, 320) * 40).astype(np.uint8)
    # 뚜렷한 구조: 중앙 사각형 테두리.
    cy, cx = 120, 160
    frame[cy - 30:cy + 30, cx - 30] = 255
    frame[cy - 30:cy + 30, cx + 30] = 255
    frame[cy - 30, cx - 30:cx + 30] = 255
    frame[cy + 30, cx - 30:cx + 30] = 255
    tmpl_img = frame[cy - 40:cy + 40, cx - 40:cx + 40].copy()
    template = build_template(tmpl_img, recipe_id="t", version="v")
    return template, frame, (cx, cy)


def test_rescore_matches_direct_score_map_lookup():
    template, frame, _ = _synthetic_template_and_frame()
    _edges, frame_dt = akm.preprocess_for_matching(frame)
    scale = 1.0
    score_map, (tw, th) = _chamfer_score_map_at_scale(template.edge_map, frame_dt, scale)
    # 맵 유효 영역 안의 임의 center 위치.
    x0, y0 = 50, 40
    cx, cy = x0 + tw // 2, y0 + th // 2
    expected = float(score_map[y0, x0])

    cands = _rescore_positions_to_candidates(template, frame_dt, [((cx, cy), scale)])

    assert len(cands) == 1
    assert isinstance(cands[0], AlignKeyCandidate)
    assert cands[0].xy == (cx, cy)
    assert cands[0].template_size == (tw, th)
    assert abs(cands[0].chamfer_score - expected) < 1e-6
    assert cands[0].score == cands[0].chamfer_score


def test_rescore_out_of_bounds_scores_zero():
    template, frame, _ = _synthetic_template_and_frame()
    _edges, frame_dt = akm.preprocess_for_matching(frame)
    cands = _rescore_positions_to_candidates(template, frame_dt, [((-999, -999), 1.0)])
    assert len(cands) == 1
    assert cands[0].chamfer_score == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_score_ensemble.py::test_rescore_matches_direct_score_map_lookup -v`
Expected: FAIL — `ImportError: cannot import name '_rescore_positions_to_candidates'` (및 `compute_align_key_score_ensemble`).

- [ ] **Step 3: `_rescore_positions_to_candidates` 구현**

`align_key_matcher.py`에 `compute_chamfer_candidates` 근처(후보 관련 함수 모음)에 추가:

```python
def _rescore_positions_to_candidates(
    template: AlignKeyTemplate,
    frame_dt: np.ndarray,
    positions: list,
) -> list:
    """ensemble 위치 [(center_xy, scale)] → chamfer rescore 된 AlignKeyCandidate 리스트.

    scale 별 chamfer score map 을 1회 계산(캐시)하고, 각 center 를 top-left 로 환산해
    score_map 룩업한다. 맵 밖/매칭 불가 → chamfer_score=0.0. 입력 순서(=RRF 순위) 보존.
    """
    score_map_cache: dict = {}
    out: list = []
    for (cx, cy), scale in positions:
        scale = float(scale)
        if scale not in score_map_cache:
            score_map_cache[scale] = _chamfer_score_map_at_scale(
                template.edge_map, frame_dt, scale
            )
        score_map, (tw, th) = score_map_cache[scale]
        chamfer = 0.0
        if score_map is not None:
            x0 = int(cx) - tw // 2
            y0 = int(cy) - th // 2
            if 0 <= y0 < score_map.shape[0] and 0 <= x0 < score_map.shape[1]:
                chamfer = float(score_map[y0, x0])
        out.append(
            AlignKeyCandidate(
                score=chamfer,
                chamfer_score=chamfer,
                xy=(int(cx), int(cy)),
                scale=scale,
                template_size=(tw, th),
            )
        )
    return out
```

- [ ] **Step 4: rescore 테스트 통과 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_score_ensemble.py -k rescore -v`
Expected: PASS (2 케이스). `compute_align_key_score_ensemble` 미구현이라 다른 테스트는 아직 collect 실패할 수 있으니 `-k rescore`로 격리.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_2/align_key_matcher.py poc/workflow_2/test_align_key_score_ensemble.py
git commit -m "feat(workflow_2): _rescore_positions_to_candidates — chamfer rescore of ensemble positions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `compute_align_key_score_ensemble` (proposer→rescore→ORB pool-rerank→finalize)

**Files:**
- Modify: `poc/workflow_2/align_key_matcher.py`
- Modify: `poc/workflow_2/test_align_key_score_ensemble.py`

- [ ] **Step 1: 실패 테스트 작성 — 통합 결과 형태 + no-candidate + pool-rerank**

`test_align_key_score_ensemble.py`에 추가(상단 import는 Task 3에서 이미 `compute_align_key_score_ensemble` 포함):

```python
def test_ensemble_returns_valid_result_shape():
    template, frame, (cx, cy) = _synthetic_template_and_frame()
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
    assert isinstance(result, akm.AlignKeyMatchResult)
    assert 0 <= result.best_xy[0] < frame.shape[1]
    assert 0 <= result.best_xy[1] < frame.shape[0]
    assert result.decision in ("match", "adjust", "low")
    assert isinstance(result.candidates, list)


def test_ensemble_no_candidates_returns_reject(monkeypatch):
    template, frame, _ = _synthetic_template_and_frame()

    class _Empty:
        fused = []

    monkeypatch.setattr(akm, "compute_ensemble_candidates", lambda *a, **k: _Empty())
    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,))
    assert result.reject_reason == "no_candidates"
    assert result.candidates == []
    assert result.score == 0.0


def test_ensemble_pool_rerank_prefers_orb_favored(monkeypatch):
    """ORB 가 chamfer-비최상 후보를 우세하게 만들면 best_xy 가 그 후보가 되어야."""
    template, frame, _ = _synthetic_template_and_frame()

    # 두 위치를 강제로 반환 — 첫째는 chamfer 우세, 둘째는 ORB 우세가 되도록.
    pos_a = (80, 70)
    pos_b = (200, 150)

    class _Cand:
        def __init__(self, xy):
            self.xy = xy
            self.scale = 1.0

    class _Ens:
        fused = [_Cand(pos_a), _Cand(pos_b)]

    monkeypatch.setattr(akm, "compute_ensemble_candidates", lambda *a, **k: _Ens())

    # rescore: 둘 다 chamfer>0 이어야 ORB 게이트 통과. pos_a 가 chamfer 더 높게.
    def _fake_rescore(tpl, fdt, positions):
        out = []
        for (cx, cy), scale in positions:
            ch = 0.9 if (cx, cy) == pos_a else 0.5
            out.append(akm.AlignKeyCandidate(
                score=ch, chamfer_score=ch, xy=(cx, cy), scale=1.0, template_size=(40, 40)))
        return out

    monkeypatch.setattr(akm, "_rescore_positions_to_candidates", _fake_rescore)

    # ORB: pos_b 만 높은 inlier → combined(0.8*0.5 + 0.2*1.0=0.6) > pos_a(0.8*0.9+0.2*0=0.72)? 아님.
    # orb_weight 를 키워 pos_b 가 이기도록 STRUCTURE_POLICY 대신 커스텀.
    from poc.workflow_2.align_key_matcher import MatchPolicy
    policy = MatchPolicy(chamfer_weight=0.2, orb_weight=0.8)

    def _fake_orb(tmpl_img, crop, **k):
        # crop 위치를 못 보니, 호출 순서가 아니라 내용으로 구분 불가 → 좌표 기반 패치 대신
        # _crop_with_padding 를 패치해서 어느 후보인지 식별.
        return (0.0, 0, 0)

    # 더 단순하게: _crop_with_padding 를 패치해 center 를 태그로 흘려보내고 ORB 가 그걸 읽게.
    def _fake_crop(frame_img, cx, cy, tw, th, *, pad=1.5):
        return np.full((4, 4), 1 if (cx, cy) == pos_b else 0, dtype=np.uint8), (0, 0)

    def _orb_by_tag(tmpl_img, crop, **k):
        return (1.0, 10, 10) if int(crop[0, 0]) == 1 else (0.0, 0, 0)

    monkeypatch.setattr(akm, "_crop_with_padding", _fake_crop)
    monkeypatch.setattr(akm, "compute_orb_inlier_ratio", _orb_by_tag)

    result = compute_align_key_score_ensemble(template, frame, scales=(1.0,), policy=policy)
    # combined: pos_a=0.2*0.9+0.8*0=0.18, pos_b=0.2*0.5+0.8*1.0=0.90 → pos_b 승.
    assert result.best_xy == pos_b


def test_ensemble_rejects_invalid_scales():
    template, frame, _ = _synthetic_template_and_frame()
    try:
        compute_align_key_score_ensemble(template, frame, scales=())
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty scales")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_score_ensemble.py -k ensemble -v`
Expected: FAIL — `AttributeError`/`ImportError`: `compute_align_key_score_ensemble` 미정의.

- [ ] **Step 3: `compute_align_key_score_ensemble` 구현**

`compute_align_key_score` 정의 **바로 아래**에 추가. `ensemble_proposer` import는 파일 상단 import 블록에 추가(`from poc.workflow_2.ensemble_proposer import compute_ensemble_candidates`). 모듈 함수 참조는 monkeypatch 가능하도록 `akm.compute_ensemble_candidates`/`_rescore_positions_to_candidates`/`_crop_with_padding`/`compute_orb_inlier_ratio`가 모듈 전역에서 호출되게 한다(직접 이름 호출 = 동일 모듈 전역 참조라 monkeypatch 반영됨).

```python
def compute_align_key_score_ensemble(
    template: AlignKeyTemplate,
    frame: np.ndarray,
    *,
    frame_nm_per_pixel: float | None = None,
    roi_hint: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] | None = None,
    policy: MatchPolicy = DEFAULT_POLICY,
) -> AlignKeyMatchResult:
    """ensemble proposer 기반 매칭 — compute_align_key_score 와 동일 시그니처/결과 형태.

    proposer(3채널 RRF, recall 향상) → chamfer rescore → ORB pool-rerank → 공유 finalize.
    A/B 가 잰 recall@N(진실이 후보 집합에 듦)을 최종 픽으로 전환하려면 pool 전체를 verifier
    (chamfer+ORB)로 rerank 해야 한다(설계: docs/specs/2026-06-09-ensemble-proposer-
    production-integration-design.md). 프레임당 비용↑(ORB×top_n + ensemble ~1s) 이므로
    fallback/static-compare 경로 전용 — live broad-scan 은 compute_align_key_score 유지.
    """
    gray_frame, frame_dt, scales, roi_origin = _prepare_match_inputs(
        template,
        frame,
        frame_nm_per_pixel=frame_nm_per_pixel,
        roi_hint=roi_hint,
        scales=scales,
    )

    ens = compute_ensemble_candidates(
        template.raw_image, gray_frame, scales=scales, top_n=policy.top_n
    )
    positions = [(c.xy, c.scale) for c in ens.fused]
    candidates = _rescore_positions_to_candidates(template, frame_dt, positions)
    if not candidates:
        return _no_candidate_result(frame, frame_dt, template, roi_origin)

    # verifier-rerank: top_n 후보에 ORB → combined = chamfer_w*chamfer + orb_w*orb → argmax.
    # 이 단계가 proposer recall 을 최종 픽으로 전환한다(RRF-top 단독 채택 금지).
    best_cand = candidates[0]
    best_combined = -1.0
    best_orb = 0.0
    for cand in candidates[:policy.top_n]:
        cx, cy = cand.xy
        tw, th = cand.template_size
        chamfer = cand.chamfer_score
        orb = 0.0
        if chamfer > 0.0 and tw > 0 and th > 0:
            crop, _crop_origin = _crop_with_padding(gray_frame, cx, cy, tw, th, pad=1.6)
            orb, _n_inliers, _n_matches = compute_orb_inlier_ratio(template.raw_image, crop)
        combined = policy.chamfer_weight * chamfer + policy.orb_weight * orb
        if combined > best_combined:
            best_combined = combined
            best_cand = cand
            best_orb = orb

    # 반환 candidates 는 chamfer 내림차순(AlignKeyCandidate.score 계약). best 는 별도 추적.
    candidates_sorted = sorted(candidates, key=lambda c: c.chamfer_score, reverse=True)
    return _finalize_match(
        best_cand, candidates_sorted, frame, template, policy, roi_origin,
        chamfer_score=best_cand.chamfer_score, orb_ratio=best_orb,
    )
```

- [ ] **Step 4: 통합 테스트 통과 확인**

Run: `uv run pytest poc/workflow_2/test_align_key_score_ensemble.py -v`
Expected: PASS (rescore 2 + ensemble 4 = 6 케이스).

- [ ] **Step 5: workflow_2 전체 회귀 + ensemble proposer 테스트 동반 확인**

Run: `uv run pytest poc/workflow_2/ -v`
Expected: PASS — 기존 전 테스트 + 신규 6케이스. (기존 `test_align_key_match.py` 포함 회귀 없음.)

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_2/align_key_matcher.py poc/workflow_2/test_align_key_score_ensemble.py
git commit -m "feat(workflow_2): compute_align_key_score_ensemble — proposer→rescore→ORB pool-rerank→finalize

ensemble proposer(recall@8 0.698) 생산 통합 진입점. 기존 compute_align_key_score 무변경.
fallback/static-compare 전용(프레임당 비용↑); 호출자 전환·오피스 e2e 는 별도.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (작성자 체크)

**Spec coverage:**
- 신규 `compute_align_key_score_ensemble`(동일 시그니처) → Task 4. ✓
- proposer→rescore→ORB pool-rerank→finalize → Task 3(rescore) + Task 4(rerank/finalize 호출). ✓
- 공유 `_finalize_match` 추출 + 기존 무변경/비트동일 → Task 1·2(+회귀 가드). ✓
- rescore center→top-left 룩업, 맵밖 0 → Task 3. ✓
- no-candidate 경로 공유 → Task 2(`_no_candidate_result`) + Task 4 호출. ✓
- scales/ROI override 에러 보존 → `_prepare_match_inputs`(Task 1) + Task 4 테스트. ✓
- 테스트 1~6(spec) → Task 3·4 테스트로 커버. ✓
- 비범위(호출자 전환/e2e/live/캐싱) → plan에서 제외 명시. ✓

**Type consistency:** `AlignKeyCandidate(score, chamfer_score, xy, scale, template_size, orb_inlier_ratio)` / `AlignKeyMatchResult` 필드 / `MatchPolicy(chamfer_weight, orb_weight, top_n)` / `_chamfer_score_map_at_scale → (score_map|None, (tw,th))` / `compute_orb_inlier_ratio → (ratio, inliers, matches)` / `compute_ensemble_candidates(...).fused` / `_Cand.xy,.scale` — 전부 실제 코드와 일치. `_finalize_match` 시그니처(best_cand, candidates, frame, template, policy, roi_origin, *, chamfer_score, orb_ratio)는 Task 2 정의와 Task 2·4 호출이 일치. ✓

**Placeholder scan:** 모든 코드 step에 완전한 코드. TBD/TODO 없음. ✓
