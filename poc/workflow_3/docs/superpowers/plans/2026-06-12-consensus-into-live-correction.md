# Consensus 템플릿 → 실시간 보정 경로 투입 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 검증된 consensus(최근 S median) 템플릿을 workflow_3 실시간 align-fail 보정의 등록 template 으로 승격하되, 부족·blur·실패 시 항상 기존 rcp key 로 안전 폴백한다.

**Architecture:** Approach A(resolver 모듈). bench(workflow_2)에서 검증된 순수 build/gate/select 로직을 bit-parity 로 포팅하고, workflow_3 event-cache 레이아웃을 읽는 adapter + cold-cache bounded sync 를 포함한 resolver 를 새로 만든다. matcher(`matching/engine.py`)와 rcp 머티리얼라이즈(`templates.py`)는 불변 — consensus template 은 rcp 와 동일한 `AlignKeyTemplate` 필드를 갖는다.

**Tech Stack:** Python 3.10+, OpenCV(cv2), numpy, uv. 테스트는 `uv run python <test>.py`(기존 align 스모크 테스트 규약) + 합성/monkeypatch.

**Spec:** `poc/workflow_3/docs/superpowers/specs/2026-06-12-consensus-into-live-correction-design.md`

---

## File Structure

**신규(3):**
- `poc/workflow_3/align/consensus_cv.py` — 포팅된 순수 CV 프리미티브: median(`_consensus`), 선명도(`_edge_density`/`_lap_var`), crop 기하(`_matched_crop`), co-registration(`_align_to_ref`/`coregister_crops`) + 상수. 의존: cv2/np 만.
- `poc/workflow_3/align/consensus_template.py` — 포팅된 build/gate/select: `ConsensusPolicy`/`ConsensusResult`/`_sharpness_ratio`/`build_consensus_template`/`select_routing_templates`/`CONSENSUS_VERSION`. 순수, I/O 없음.
- `poc/workflow_3/align/consensus_crops.py` — event-cache adapter: `_cond_crosshair_xy`/`_cond_consensus_crop`/`_resolve_mod`/`_precrop_drop_reason`/`build_center_tpls_for_sizing`/`load_coregistered_crops`.
- `poc/workflow_3/align/consensus_resolve.py` — `resolve_templates(assets, config, eqp_id)` 오케스트레이션(crops→build→select + cold-cache sync).

**수정(5):**
- `poc/workflow_3/config.py` — consensus 노브 + `gather_max_events` 5→8.
- `poc/workflow_3/align/consensus_gather.py` — TTL freshness + 원자적 swap + non-empty=이미지 수.
- `poc/workflow_3/monitor/success_gather.py` — `wait_for_gather` 추가.
- `poc/workflow_3/align/correction.py` — `CorrectionConfig` consensus 필드 + 449행 swap.
- `poc/workflow_3/monitor/cycle.py` — `_exec_run_correction` 가 settings→CorrectionConfig 전달.

**테스트(신규):** 각 신규 모듈 옆 `test_*.py`(기존 align 규약: `uv run python ...test_*.py`).

---

## Task 1: consensus_cv.py — 포팅된 순수 CV 프리미티브

**Files:**
- Create: `poc/workflow_3/align/consensus_cv.py`
- Test: `poc/workflow_3/align/test_consensus_cv.py`

bench 원본(bit-parity, 재구현 금지): `_consensus`/`_edge_density`/`_lap_var` = `poc/workflow_2/align_similarity.py`; `_matched_crop` = `poc/workflow_2/align_similarity.py:144`; `_align_to_ref`/`coregister_crops`/상수 = `poc/workflow_2/golden_consensus_eval_cond.py:147-171,92-93`.

- [ ] **Step 1: 실패 테스트 작성**

```python
# poc/workflow_3/align/test_consensus_cv.py
"""consensus_cv 포팅 프리미티브 스모크 — bench 와 bit-parity 동작 확인."""
import numpy as np

from poc.workflow_3.align.consensus_cv import (
    _consensus, _edge_density, _lap_var, _matched_crop, coregister_crops,
)


def test_consensus_median_of_uint8():
    a = np.zeros((8, 8), np.uint8)
    b = np.full((8, 8), 100, np.uint8)
    c = np.full((8, 8), 200, np.uint8)
    out = _consensus([a, b, c])
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 100  # median(0,100,200)=100


def test_sharpness_metrics_nonnegative():
    g = (np.random.RandomState(0).rand(32, 32) * 255).astype(np.uint8)
    assert _edge_density(g) >= 0.0
    assert _lap_var(g) >= 0.0
    assert _edge_density(np.zeros((0, 0), np.uint8)) == 0.0  # 빈 입력 가드


def test_matched_crop_resizes_to_template_size():
    frame = np.zeros((100, 120), np.uint8)
    crop = _matched_crop(frame, (60, 50), tw=20, th=16, scale=1.0)
    assert crop is not None and crop.shape == (16, 20)


def test_matched_crop_returns_none_when_too_small():
    frame = np.zeros((100, 120), np.uint8)
    assert _matched_crop(frame, (0, 0), tw=20, th=16, scale=0.01) is None


def test_coregister_passthrough_under_two():
    one = [np.zeros((8, 8), np.uint8)]
    assert coregister_crops(one) is one  # <2 면 그대로


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception:
            failed += 1; print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_cv.py`
Expected: FAIL — `ModuleNotFoundError: poc.workflow_3.align.consensus_cv`.

- [ ] **Step 3: consensus_cv.py 작성(bench 코드 verbatim 포팅)**

```python
# poc/workflow_3/align/consensus_cv.py
"""consensus 빌드용 순수 CV 프리미티브 — bench(workflow_2)에서 bit-parity 포팅.

median(_consensus)·선명도(_edge_density/_lap_var)·crop 기하(_matched_crop)·
co-registration(_align_to_ref/coregister_crops)은 검증된 bench 로직 그대로다
(재구현 금지 — 표류 시 bit-parity/검증 수치 +0.442 가 깨진다). bench 출처:
align_similarity.py(_consensus/_edge_density/_lap_var/_matched_crop),
golden_consensus_eval_cond.py(_align_to_ref/coregister_crops/상수).
"""

import cv2
import numpy as np

# co-registration 상수 (golden_consensus_eval_cond.py:92-93).
COREG_ITERS = 2                 # ref median 다듬으며 원본 재정렬(보간 누적 방지).
COREG_MAX_SHIFT_FRAC = 0.3      # 추정 shift 가 변의 이 비율 초과면 spurious → 정렬 생략.


def _consensus(crops: list) -> np.ndarray:
    """동일 크기 gray crop 들의 median 이미지 = '현재 align 영역의 대표 모습'."""
    stack = np.stack([c.astype(np.float32) for c in crops])
    return np.median(stack, axis=0).astype(np.uint8)


def _edge_density(gray: np.ndarray) -> float:
    """Canny(60,160) edge 픽셀 비율 — matcher 전처리와 동일 임계."""
    if gray is None or gray.size == 0:
        return 0.0
    e = cv2.Canny(gray, 60, 160)
    return float((e > 0).mean())


def _lap_var(gray: np.ndarray) -> float:
    """Laplacian 분산 — 선명도 지표. median consensus blur 여부 확인용."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _matched_crop(frame: np.ndarray, center_xy, tw: int, th: int, scale: float):
    """center 위치/스케일에서 crop 을 떼어 template 크기로 리사이즈. 너무 작으면 None."""
    cw = max(1, int(round(tw * scale)))
    ch = max(1, int(round(th * scale)))
    cx, cy = center_xy
    x0 = max(0, int(cx - cw // 2))
    y0 = max(0, int(cy - ch // 2))
    x1 = min(frame.shape[1], x0 + cw)
    y1 = min(frame.shape[0], y0 + ch)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)


def _align_to_ref(img, ref):
    """img 를 ref 에 sub-pixel 평행이동 정렬(phase correlation). 과도 shift 면 원본 반환."""
    h, w = ref.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(img.astype(np.float32), ref.astype(np.float32), win)
    if abs(dx) > COREG_MAX_SHIFT_FRAC * w or abs(dy) > COREG_MAX_SHIFT_FRAC * h:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def coregister_crops(crops):
    """crop 들을 공통 reference(다듬어진 median)에 sub-pixel 정렬해 median 을 또렷하게.

    매 iter ref=median(현재 정렬본)으로 갱신하되, 정렬은 항상 *원본*에서 한 번만 적용해
    보간 blur 누적을 막는다. crop 2장 미만이면 그대로 반환.
    """
    if len(crops) < 2:
        return crops
    aligned = list(crops)
    for _ in range(COREG_ITERS):
        ref = np.median(np.stack([a.astype(np.float32) for a in aligned]), 0).astype(np.uint8)
        aligned = [_align_to_ref(c, ref) for c in crops]
    return aligned
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_cv.py`
Expected: `5/5 pass`.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/align/consensus_cv.py poc/workflow_3/align/test_consensus_cv.py
git commit -m "feat(workflow_3): port consensus CV primitives (median/sharpness/crop/coreg) bit-parity from bench"
```

---

## Task 2: consensus_template.py — 포팅된 build/gate/select

**Files:**
- Create: `poc/workflow_3/align/consensus_template.py`
- Test: `poc/workflow_3/align/test_consensus_template.py`

bench 원본: `poc/workflow_2/consensus_template.py`(전체). 차이는 **import 만** — bench 는 `from poc.workflow_2.align_similarity import _consensus, _edge_density, _lap_var` 인데, workflow_3 는 workflow_2 를 import 하면 안 되므로(DAG 위반) Task 1 의 `consensus_cv` 에서 가져온다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# poc/workflow_3/align/test_consensus_template.py
"""consensus_template build/gate/select 스모크 — 게이트 폴백 신호 + 라우팅 조립."""
import numpy as np

from poc.workflow_3.align.consensus_template import (
    DEFAULT_CONSENSUS_POLICY, ConsensusPolicy,
    build_consensus_template, select_routing_templates,
)


def _sharp_crops(n, size=64):
    """edge 풍부한 또렷한 동일 패턴 crop n장(blur 가드 통과용)."""
    rng = np.random.RandomState(1)
    base = (rng.rand(size, size) * 255).astype(np.uint8)
    return [base.copy() for _ in range(n)]


def test_insufficient_s_returns_none():
    res = build_consensus_template(_sharp_crops(2), recipe_id="E/c/r", modality="sem",
                                   policy=ConsensusPolicy(min_s=4))
    assert res.template is None and res.reason == "insufficient_s" and res.n_crops == 2


def test_enough_sharp_builds_template():
    res = build_consensus_template(_sharp_crops(5), recipe_id="E/c/r", modality="sem",
                                   policy=ConsensusPolicy(min_s=4))
    assert res.reason == "ok" and res.template is not None
    assert res.template.key_type == "sem"


def test_blurry_consensus_rejected():
    # 서로 크게 어긋난 노이즈 → median 이 흐려 edge/lap 비율 < 임계.
    rng = np.random.RandomState(2)
    crops = [(rng.rand(64, 64) * 255).astype(np.uint8) for _ in range(5)]
    res = build_consensus_template(crops, recipe_id="E/c/r", modality="om",
                                   policy=DEFAULT_CONSENSUS_POLICY)
    assert res.template is None and res.reason == "blurry"


def test_select_prefers_consensus_else_rcp():
    class T:  # AlignKeyTemplate 자리 대역(select 는 객체 정체성만 본다).
        pass
    cons_om, rcp_om, rcp_sem = T(), T(), T()
    out = select_routing_templates({"om": cons_om, "sem": None},
                                   {"om": rcp_om, "sem": rcp_sem})
    assert out["OM"] is cons_om   # consensus 우선
    assert out["SEM"] is rcp_sem  # consensus None → rcp 폴백


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except Exception:
            failed += 1; print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_template.py`
Expected: FAIL — `ModuleNotFoundError: poc.workflow_3.align.consensus_template`.

- [ ] **Step 3: consensus_template.py 작성(bench verbatim, import 만 workflow_3 로)**

```python
# poc/workflow_3/align/consensus_template.py
"""consensus(최근 S median) template 빌더 — 검증된 레버의 프로덕션 진입점.

bench(poc/workflow_2/consensus_template.py)에서 bit-parity 포팅. 책임 = **게이트**:
consensus 가 신뢰 불가(부족/blur)면 template=None 을 돌려 호출부가 rcp 로 폴백하게 한다
(consensus or rcp). 즉 어떤 사유든 최악 = 검증된 rcp 베이스라인 → 회귀 위험 0.
검증 근거: cond A/B in_topk 0.434→0.876, rank1 0.318→0.764 (저널 260608_163302).

입력 crop 규약: **한 modality**의, 이미 정제(crosshair 제거)·crosshair 중심 crop·
co-registration 까지 끝난 동일 크기 gray 배열들(= consensus_crops.load_coregistered_crops 출력).
"""

import statistics
from dataclasses import dataclass

from poc.workflow_3.align.matching.engine import AlignKeyTemplate, build_template
from poc.workflow_3.align.consensus_cv import _consensus, _edge_density, _lap_var

CONSENSUS_VERSION = "s_consensus_prod"   # 로그/overlay 에서 rcp 와 구분.


@dataclass(frozen=True)
class ConsensusPolicy:
    """consensus 게이트 임계. blur 임계(0.70/0.50)는 golden 실측 확정값."""

    min_s: int = 3                # 같은 modality 최소 S crop 장수(resolver 가 settings 로 주입).
    edge_ratio_min: float = 0.70  # consensus edge density / 개별 crop median. 미만이면 흐림.
    lap_ratio_min: float = 0.50   # consensus Laplacian 분산 비율. 미만이면 흐림.


DEFAULT_CONSENSUS_POLICY = ConsensusPolicy()


@dataclass
class ConsensusResult:
    """build_consensus_template 결과 + audit. template None = rcp 폴백 신호."""

    template: AlignKeyTemplate | None
    modality: str
    n_crops: int
    edge_ratio: float | None
    lap_ratio: float | None
    reason: str                          # "ok" | "insufficient_s" | "blurry".


def _sharpness_ratio(metric_fn, consensus, crops):
    """consensus 선명도 ÷ 개별 crop 선명도 median. 분모 0(featureless)이면 None."""
    c_val = metric_fn(consensus)
    s_med = statistics.median([metric_fn(c) for c in crops])
    if s_med <= 0:
        return None
    return round(c_val / s_med, 4)


def build_consensus_template(crops, *, recipe_id, modality,
                             policy=DEFAULT_CONSENSUS_POLICY):
    """정제·정렬된 같은 modality S crop 들로 consensus template 을 짓는다(게이트 포함).

    template 이 None 이면 호출부는 rcp center 로 폴백한다.
    """
    n = len(crops)
    if n < policy.min_s:
        return ConsensusResult(None, modality, n, None, None, "insufficient_s")

    consensus = _consensus(crops)
    edge_ratio = _sharpness_ratio(_edge_density, consensus, crops)
    lap_ratio = _sharpness_ratio(_lap_var, consensus, crops)

    edge_bad = edge_ratio is None or edge_ratio < policy.edge_ratio_min
    lap_bad = lap_ratio is None or lap_ratio < policy.lap_ratio_min
    if edge_bad or lap_bad:
        return ConsensusResult(None, modality, n, edge_ratio, lap_ratio, "blurry")

    template = build_template(consensus, recipe_id=recipe_id,
                              version=CONSENSUS_VERSION, key_type=modality)
    return ConsensusResult(template, modality, n, edge_ratio, lap_ratio, "ok")


# modality(빌더 규약) → route_template 의 키 규약.
_MOD_TO_ROUTE_KEY = {"om": "OM", "sem": "SEM"}


def select_routing_templates(consensus_by_mod, rcp_by_mod):
    """route_template 에 넘길 dict 을 consensus 우선·rcp 폴백으로 조립.

    Args:
        consensus_by_mod: {'om'|'sem': AlignKeyTemplate | None}. None = 게이트 폴백 신호.
        rcp_by_mod:       {'om'|'sem': AlignKeyTemplate | None}. 베이스라인 center template.
    Returns:
        {'OM'|'SEM': AlignKeyTemplate} — route_template(templates, mode) 에 바로 사용.
    """
    out = {}
    for mod, route_key in _MOD_TO_ROUTE_KEY.items():
        chosen = consensus_by_mod.get(mod) or rcp_by_mod.get(mod)
        if chosen is not None:
            out[route_key] = chosen
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_template.py`
Expected: `4/4 pass`.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/align/consensus_template.py poc/workflow_3/align/test_consensus_template.py
git commit -m "feat(workflow_3): port consensus build/gate/select bit-parity (rcp fallback on insufficient/blurry)"
```

---

## Task 3: consensus_crops.py — event-cache adapter

**Files:**
- Create: `poc/workflow_3/align/consensus_crops.py`
- Test: `poc/workflow_3/align/test_consensus_crops.py`

bench 원본: `_resolve_mod`/`_cond_crosshair_xy`/`_cond_consensus_crop`/`_precrop_drop_reason` = `golden_consensus_eval_cond.py:185-243`. 이미 workflow_3 에 있는 것 재사용: `clean_image`/`cursor_to_image`/`OVERSAMPLE`(`align/clean_align_image.py`), `msr_modality`(`align/cond_file.py`), `load_cond`(`align/cond_file.py`), `load_gray`(`align/assets.py`), `_matched_crop`/`coregister_crops`(Task 1), `build_templates_from_assets`(`align/templates.py`), `_events_dir_for`(`align/consensus_gather.py`).

**바이트-패리티 주의(Codex#3):** modality 는 `msr_modality(cond) or recipe_mod`(= bench `_resolve_mod`), crop sizing 은 box-crop 이 아닌 **center crop**(소문자 om/sem)으로 한다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# poc/workflow_3/align/test_consensus_crops.py
"""consensus_crops adapter 스모크 — modality resolve 폴백 + crop/coreg + drop 집계.

load_cond/load_gray 를 monkeypatch 해 cond.txt 형식 결합 없이 adapter 로직만 검증한다.
"""
import sys, types
import numpy as np

import poc.workflow_3.align.consensus_crops as cc


class _FakeCond:
    def __init__(self, xy):
        self.crosshair_xy = xy  # cursor frame ×10 좌표(없으면 None)


class _FakeTpl:
    def __init__(self, w, h):
        self.raw_image = np.zeros((h, w), np.uint8)
        self.align_offset_xy = (0, 0)


def _patch(monkeypatch_map):
    """모듈 전역을 임시 교체하고 원복 함수 반환."""
    saved = {k: getattr(cc, k) for k in monkeypatch_map}
    for k, v in monkeypatch_map.items():
        setattr(cc, k, v)
    def restore():
        for k, v in saved.items():
            setattr(cc, k, v)
    return restore


def _events(tmp, n, prefix="20260612_0900"):
    """events/<id>/S1.jpeg 더미 파일 n개 생성(내용은 monkeypatch 가 가로채므로 빈 파일)."""
    ev = tmp / "E1" / "c" / "r" / "events"
    ev.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = ev / f"{prefix}{i:02d}_r_lot"
        d.mkdir()
        (d / "S1.jpeg").write_bytes(b"x")
    return tmp


def test_resolve_mod_falls_back_to_recipe_mod():
    assert cc._resolve_mod(_FakeCond((10, 10)), "sem") in ("om", "sem")  # 키 추론 or 폴백
    # msr_modality 가 None 을 줄 때 recipe_mod 폴백 보장:
    restore = _patch({"msr_modality": lambda cond: None})
    try:
        assert cc._resolve_mod(_FakeCond((1, 1)), "sem") == "sem"
        assert cc._resolve_mod(_FakeCond((1, 1)), None) is None
    finally:
        restore()


def test_load_coregistered_crops_groups_and_caps(tmp_path):
    gray = (np.random.RandomState(0).rand(200, 200) * 255).astype(np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _FakeCond((100, 100)),     # 중앙 crosshair
        "clean_image": lambda g, cond: g,                  # 정제 no-op (테스트)
        "cursor_to_image": lambda xy, oversample=10: (xy[0] / 10.0, xy[1] / 10.0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 6)
        center = {"sem": (_FakeTpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=4)
        assert set(out) == {"sem"}
        assert len(out["sem"]) == 4               # cap=max_events
        assert out["sem"][0].shape == (32, 40)    # center tpl 크기로 crop
    finally:
        restore()


def test_missing_crosshair_dropped(tmp_path):
    gray = np.zeros((200, 200), np.uint8)
    restore = _patch({
        "load_gray": lambda p: gray,
        "load_cond": lambda p: _FakeCond(None),   # crosshair 없음 → drop
        "clean_image": lambda g, cond: g,
        "cursor_to_image": lambda xy, oversample=10: (0, 0),
        "msr_modality": lambda cond: "sem",
    })
    try:
        _events(tmp_path, 4)
        center = {"sem": (_FakeTpl(40, 32), (0, 0))}
        out = cc.load_coregistered_crops(tmp_path, "E1", "c/r", center, max_events=8)
        assert out.get("sem", []) == []           # 전부 drop
    finally:
        restore()


if __name__ == "__main__":
    import tempfile, pathlib, traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1; print(f"[FAIL] {name}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_crops.py`
Expected: FAIL — `ModuleNotFoundError: poc.workflow_3.align.consensus_crops`.

- [ ] **Step 3: consensus_crops.py 작성**

```python
# poc/workflow_3/align/consensus_crops.py
"""event-cache(align_consensus_cache) → consensus 재료 crop 어댑터.

bench `_build_cond_by_recipe`(golden_consensus_eval_cond.py)의 cache 판. msr/LOO 레이아웃
대신 align_consensus_cache/<eqp>/<class>/<recipe>/events/<event_id>/S* 를 읽는다.
modality 분류는 bench 와 동일하게 `_resolve_mod = msr_modality(cond) or recipe_mod`(단일-modality
recipe 가 silently drop 되지 않게), crop 은 clean→crosshair중심→center tpl 크기 고정,
modality 별 co-registration. 이름(load_cond/load_gray/clean_image/cursor_to_image/msr_modality)은
테스트에서 monkeypatch 하므로 모듈 전역 import 로 둔다.
"""

from collections import Counter, defaultdict
from pathlib import Path

from poc.workflow_3.align.assets import load_gray
from poc.workflow_3.align.clean_align_image import OVERSAMPLE, clean_image, cursor_to_image
from poc.workflow_3.align.cond_file import load_cond, msr_modality
from poc.workflow_3.align.consensus_cv import _matched_crop, coregister_crops
from poc.workflow_3.align.consensus_gather import _events_dir_for
from poc.workflow_3.align.templates import build_templates_from_assets

# rcp 라우팅 키(대문자) → consensus 빌더/center 키(소문자).
_ROUTE_TO_MOD = {"OM": "om", "SEM": "sem"}


def build_center_tpls_for_sizing(assets):
    """consensus crop sizing 전용 center template — 소문자 om/sem, (tpl, offset).

    box-crop 이 아닌 **center-area crop**(cond_box_crop=False)이라 bench center_tpls 와
    동일 기하(crop 크기 = template.raw_image 크기). 런타임 rcp 라우팅 template(대문자,
    box-crop 가능)과 별개로 만든다(Codex#3). 없으면 그 modality 는 빠진다.
    """
    rcp_center = build_templates_from_assets(assets, cond_box_crop=False)  # {"OM":tpl,"SEM":tpl}
    out = {}
    for route_key, tpl in rcp_center.items():
        mod = _ROUTE_TO_MOD.get(route_key)
        if mod and tpl is not None:
            out[mod] = (tpl, (0, 0))
    return out


def _cond_crosshair_xy(cond):
    """cond.crosshair_xy(cursor frame, ×10) → 이미지 px (x, y). 없으면 None."""
    if cond is None or cond.crosshair_xy is None:
        return None
    gx, gy = cursor_to_image(cond.crosshair_xy, OVERSAMPLE)
    return (int(round(gx)), int(round(gy)))


def _cond_consensus_crop(gray, cond, size_wh):
    """crosshair(=align point) 중심·고정 size 의 정제된(crosshair 제거) crop. 없으면 None."""
    xy = _cond_crosshair_xy(cond)
    if xy is None:
        return None
    cleaned = clean_image(gray, cond)        # crosshair(+box) 제거 후 자른다.
    w, h = size_wh
    return _matched_crop(cleaned, xy, w, h, 1.0)


def _resolve_mod(cond, recipe_mod):
    """msr 프레임 routing modality: msr 키/배율 추론 → recipe rcp modality 폴백."""
    return msr_modality(cond) or recipe_mod


def _precrop_drop_reason(cond, xy, mod, has_tpl):
    """S 프레임이 crop 이전 단계에서 빠지는 사유(없으면 None=채택)."""
    if cond is None:
        return "missing_cond"
    if xy is None:
        return "missing_crosshair"
    if mod is None:
        return "missing_modality"
    if not has_tpl:
        return "no_template"
    return None


def _iter_event_s_images(events_dir, max_events):
    """events/ 의 최신 max_events event 의 S* 이미지 경로를 yield(시각 prefix=정렬=시간)."""
    if not events_dir.is_dir():
        return
    try:
        event_dirs = sorted([d for d in events_dir.iterdir() if d.is_dir()])
    except OSError:
        return
    for ev in event_dirs[-max_events:]:          # 최신 우선 cap.
        for img in sorted(ev.glob("S*")):
            if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                yield img


def load_coregistered_crops(cache_root, eqp_id, cache_key, center_tpls, *, max_events):
    """event-cache 의 S 이미지를 modality 별 co-registered crop 리스트로 만든다.

    Args:
        cache_root: ALIGN_CONSENSUS_CACHE_DIR(또는 테스트 temp).
        eqp_id: 장비 id.
        cache_key: "<class>/<recipe>" (gather 가 쓴 키와 동일 — leaf 금지).
        center_tpls: {'om'|'sem': (center_tpl, offset)} sizing 전용.
        max_events: 최신 event 캡(= settings.gather_max_events).
    Returns:
        {'om'|'sem': [gray_crop, ...]} — 빌더 입력. 비면 빈 dict/빈 리스트.
    """
    events_dir = _events_dir_for(eqp_id, cache_key, cache_root)
    rcp_mods = [m for m, v in center_tpls.items() if v is not None]
    recipe_mod = rcp_mods[0] if len(rcp_mods) == 1 else None

    by_mod = defaultdict(list)
    drop_counts = Counter()
    for p in _iter_event_s_images(events_dir, max_events):
        try:
            cond = load_cond(p)
        except Exception:
            cond = None
        mod = _resolve_mod(cond, recipe_mod) if cond is not None else None
        xy = _cond_crosshair_xy(cond)
        tpl_item = center_tpls.get(mod) or next(
            (t for t in center_tpls.values() if t is not None), None)
        reason = _precrop_drop_reason(cond, xy, mod, tpl_item is not None)
        if reason:
            drop_counts[reason] += 1
            continue
        tpl = tpl_item[0]
        size_wh = (tpl.raw_image.shape[1], tpl.raw_image.shape[0])
        try:
            gray = load_gray(p)
        except Exception:
            drop_counts["load_failed"] += 1
            continue
        crop = _cond_consensus_crop(gray, cond, size_wh)
        if crop is None:
            drop_counts["crop_failed"] += 1
            continue
        by_mod[mod].append(crop)

    # modality 별 co-registration(외형 달라 섞으면 안 됨).
    out = {}
    for mod, crops in by_mod.items():
        out[mod] = list(coregister_crops(crops))
    if drop_counts:
        print(f"[INFO] consensus crops drop: {dict(drop_counts)} (cache_key={cache_key})")
    return out
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_crops.py`
Expected: `3/3 pass`.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/align/consensus_crops.py poc/workflow_3/align/test_consensus_crops.py
git commit -m "feat(workflow_3): event-cache->crops adapter (resolve_mod fallback, center-tpl crop, coreg)"
```

---

## Task 4: config.py — consensus 노브 추가

**Files:**
- Modify: `poc/workflow_3/config.py`

기존 `gather_max_events: int = 5` 및 `env_int("ALIGN_FAIL_GATHER_MAX_EVENTS", 5)` 를 8 로 올리고, consensus 5필드를 추가한다. `consensus_gather.GATHER_MAX_EVENTS` 도 8 로 동기(Task 5 에서).

- [ ] **Step 1: 실패 테스트 작성**

```python
# poc/workflow_3/test_config_consensus.py (임시 검증 — Task 9 에서 제거 가능)
from poc.workflow_3.config import load_workflow3_settings


def test_consensus_defaults():
    s = load_workflow3_settings()
    assert s.consensus_enabled is True
    assert s.consensus_min_s == 4
    assert s.gather_max_events == 8
    assert s.consensus_sync_timeout_sec == 8.0
    assert s.consensus_refresh_ttl_sec == 21600


if __name__ == "__main__":
    test_consensus_defaults(); print("1/1 pass")
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/test_config_consensus.py`
Expected: FAIL — `AttributeError: 'Workflow3Settings' object has no attribute 'consensus_enabled'`.

- [ ] **Step 3: config.py 수정**

`gather_max_events` 기본값 필드를 8 로:
```python
    gather_max_events: int = 8  # align/consensus_gather.py 의 GATHER_MAX_EVENTS 와 동일 값 유지.
```
바로 아래(같은 dataclass 필드 블록)에 추가:
```python
    consensus_enabled: bool = True            # consensus 라우팅 마스터 토글(off → 순수 rcp).
    consensus_min_s: int = 4                  # modality별 build·신뢰 최소 S(floor 3).
    consensus_sync_timeout_sec: float = 8.0   # cold-cache bounded 대기(초).
    consensus_refresh_ttl_sec: int = 21600    # gather 재fetch TTL(초, 6h).
```
`load_workflow3_settings()` 의 env 매핑 블록에서 `gather_max_events` 줄을 8 로 바꾸고 4줄 추가
(주변의 `env_flag`/`env_int`/`env_float` 헬퍼 사용 패턴 그대로):
```python
        gather_max_events=env_int("ALIGN_FAIL_GATHER_MAX_EVENTS", 8),
        consensus_enabled=env_flag("ALIGN_FAIL_CONSENSUS", default=True),
        consensus_min_s=env_int("ALIGN_FAIL_CONSENSUS_MIN_S", 4),
        consensus_sync_timeout_sec=env_float("ALIGN_FAIL_CONSENSUS_SYNC_TIMEOUT", 8.0),
        consensus_refresh_ttl_sec=env_int("ALIGN_FAIL_CONSENSUS_REFRESH_TTL", 21600),
```
주: `env_float` 헬퍼가 없으면 `config.py` 상단 헬퍼 영역에 `env_int` 와 같은 꼴로 추가
(`def env_float(name, default): v=os.environ.get(name,"").strip(); return float(v) if v else default`).
`consensus_min_s` 는 floor 3 보장 위해 매핑 직후 `max(3, ...)` clamp 를 적용한다.

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/test_config_consensus.py`
Expected: `1/1 pass`.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/config.py poc/workflow_3/test_config_consensus.py
git commit -m "feat(workflow_3): consensus settings (enabled/min_s/sync_timeout/refresh_ttl) + gather_max_events 5->8"
```

---

## Task 5: consensus_gather.py — TTL freshness + 원자적 swap + non-empty=이미지 수

**Files:**
- Modify: `poc/workflow_3/align/consensus_gather.py`
- Test: `poc/workflow_3/align/test_consensus_gather.py` (기존 파일에 케이스 추가)

3가지(Codex#2): (a) `refresh_ttl_sec` 미경과면 다운로드 skip(`reason="fresh"`); (b) 교체를 events/ 가 항상 유효하도록 `events.old` 경유; (c) non-empty 판정을 `n_images>=1` 로. `GATHER_MAX_EVENTS` 5→8.

- [ ] **Step 1: 실패 테스트 작성(기존 test 파일 말미에 추가)**

```python
# poc/workflow_3/align/test_consensus_gather.py 에 추가
import time as _time
from pathlib import Path as _Path

from poc.workflow_3.align.consensus_gather import gather_success_images, _events_dir_for


class _StubEvent:
    def __init__(self, n_images):
        self.event_id = "20260612_090000_r_lot"
        self.image_paths = [_Path(f"S{i}.jpeg") for i in range(n_images)]
        self.cond_paths = []


class _Downloader:
    def __init__(self, events):
        self.events = events
        self.calls = 0
    def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
        self.calls += 1
        for ev in self.events:
            d = _Path(dest_dir) / ev.event_id
            d.mkdir(parents=True, exist_ok=True)
            for ip in ev.image_paths:
                (d / ip.name).write_bytes(b"x")
        return self.events


def test_zero_image_event_preserves_old_cache(tmp_path):
    # 기존 캐시 1장 존재.
    ev_dir = _events_dir_for("E1", "c/r", tmp_path)
    (ev_dir / "old").mkdir(parents=True)
    (ev_dir / "old" / "S0.jpeg").write_bytes(b"x")
    dl = _Downloader([_StubEvent(0)])           # 이미지 0장 event.
    res = gather_success_images("E1", "c/r", downloader=dl, cache_root=tmp_path,
                                refresh_ttl_sec=0)
    assert res.reason == "empty"
    assert (ev_dir / "old" / "S0.jpeg").exists()  # 옛 캐시 보존.


def test_ttl_skips_download(tmp_path):
    ev_dir = _events_dir_for("E1", "c/r", tmp_path)
    (ev_dir / "e1").mkdir(parents=True)
    (ev_dir / "e1" / "S0.jpeg").write_bytes(b"x")
    dl = _Downloader([_StubEvent(2)])
    res = gather_success_images("E1", "c/r", downloader=dl, cache_root=tmp_path,
                                refresh_ttl_sec=3600)  # 방금 만든 캐시 → TTL 내.
    assert res.reason == "fresh" and dl.calls == 0


if __name__ == "__main__":
    import tempfile, traceback, sys
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and "tmp_path" in v.__code__.co_varnames]
    # (기존 test 본문의 러너가 있으면 그쪽을 쓰고, 여기 추가분만 별도로 돌려도 됨)
    failed = 0
    for name, fn in fns:
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(_Path(d)); print(f"[PASS] {name}")
        except Exception:
            failed += 1; print(f"[FAIL] {name}"); traceback.print_exc()
    print(f"{len(fns)-failed}/{len(fns)} pass"); sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_gather.py`
Expected: FAIL — `gather_success_images() got an unexpected keyword argument 'refresh_ttl_sec'`.

- [ ] **Step 3: consensus_gather.py 수정**

`GATHER_MAX_EVENTS = 5` → `GATHER_MAX_EVENTS = 8`.

`gather_success_images` 시그니처에 `refresh_ttl_sec: float = 0` 추가하고, `recipe_id` 가드 직후 TTL 가드 삽입:
```python
def gather_success_images(eqp_id, recipe_id, *, downloader,
                          max_events=GATHER_MAX_EVENTS,
                          cache_root=ALIGN_CONSENSUS_CACHE_DIR,
                          refresh_ttl_sec=0) -> GatherResult:
    events_dir = _events_dir_for(eqp_id, recipe_id, cache_root)
    if not recipe_id:
        return GatherResult(eqp_id, recipe_id, events_dir, 0, 0, "skipped")

    # TTL freshness: 최근 새로고침 이내면 다운로드 skip(기존 캐시 재사용).
    if refresh_ttl_sec and events_dir.is_dir():
        try:
            age = time.time() - events_dir.stat().st_mtime
        except OSError:
            age = None
        if age is not None and age < refresh_ttl_sec:
            n_events, n_images = _count_events(events_dir)
            return GatherResult(eqp_id, recipe_id, events_dir, n_events, n_images, "fresh")
```
파일 상단에 `import time` 추가(없으면). `_count_events` 헬퍼(원자 swap 전 이미지 수 카운트)를 추가:
```python
def _count_events(events_dir) -> tuple:
    """events/ 의 (event 수, S 이미지 수). 없으면 (0,0)."""
    n_events = n_images = 0
    try:
        for ev in events_dir.iterdir():
            if not ev.is_dir():
                continue
            imgs = [p for p in ev.glob("S*")
                    if p.is_file() and p.suffix.lower() in _S_IMAGE_EXTS]
            if imgs:
                n_events += 1
                n_images += len(imgs)
    except OSError:
        pass
    return n_events, n_images
```
non-empty 판정과 원자 swap 부 교체 — 기존 `n_events == 0` 분기와 swap 블록을:
```python
        staged = staged or []
        n_events = len(staged)
        n_images = sum(len(ev.image_paths) for ev in staged)

        if n_images == 0:                       # 이미지 0장이면 옛 캐시 보존(Codex#2).
            shutil.rmtree(staging_dir, ignore_errors=True)
            return GatherResult(eqp_id, recipe_id, events_dir, 0, 0, "empty")

        # 원자적 교체 — 항상 유효한 events/ 유지: staging→events.new→events, 기존은 events.old 비킴.
        events_dir.parent.mkdir(parents=True, exist_ok=True)
        new_dir = events_dir.parent / ".events_new"
        old_dir = events_dir.parent / ".events_old"
        for d in (new_dir, old_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        staging_dir.replace(new_dir)            # staging → .events_new
        if events_dir.exists():
            events_dir.replace(old_dir)         # 기존 events → .events_old (짧은 부재 창)
        new_dir.replace(events_dir)             # .events_new → events
        shutil.rmtree(old_dir, ignore_errors=True)
```
주: 완전한 무중단 rename 은 동일 디렉터리 엔트리라 OS 보장이 제한적이지만, 위 순서는 `rmtree(events)`
선행(현행)보다 events/ 부재 창을 rename 1회로 최소화한다. adapter(Task 3 `_iter_event_s_images`)는
`events_dir.is_dir()` False / `OSError` 를 0 crop 으로 안전 처리하므로 그 창에서도 rcp 폴백된다.

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_gather.py`
Expected: 신규 2케이스 + 기존 케이스 모두 PASS.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/align/consensus_gather.py poc/workflow_3/align/test_consensus_gather.py
git commit -m "feat(workflow_3): gather TTL freshness + atomic events swap + non-empty=image-count"
```

---

## Task 6: success_gather.py — wait_for_gather

**Files:**
- Modify: `poc/workflow_3/monitor/success_gather.py`
- Test: `poc/workflow_3/monitor/test_success_gather.py` (기존에 추가)

lock-safe(Codex#1): `_IN_FLIGHT_LOCK` 안에서 thread 스냅샷만 뜨고 lock 을 푼 뒤 `join`. lock 쥔 채
join 하거나 `gather_success_async`(같은 lock) 재호출 금지. TTL 도 async 경로에 적용.

- [ ] **Step 1: 실패 테스트 작성(기존 test 파일에 추가)**

```python
# poc/workflow_3/monitor/test_success_gather.py 에 추가
import threading, time
import poc.workflow_3.monitor.success_gather as sg


def test_wait_for_gather_joins_inflight():
    done = {"v": False}
    def _slow():
        time.sleep(0.05); done["v"] = True
    t = threading.Thread(target=_slow); t.start()
    with sg._IN_FLIGHT_LOCK:
        sg._IN_FLIGHT[("E1", "c/r")] = t
    ok = sg.wait_for_gather("E1", "c/r", timeout=1.0)
    assert done["v"] is True and isinstance(ok, bool)


def test_wait_for_gather_timeout_returns_false_fast():
    def _hang():
        time.sleep(5)
    t = threading.Thread(target=_hang, daemon=True); t.start()
    with sg._IN_FLIGHT_LOCK:
        sg._IN_FLIGHT[("E2", "c/r")] = t
    start = time.time()
    ok = sg.wait_for_gather("E2", "c/r", timeout=0.1)
    assert ok is False and (time.time() - start) < 1.0   # lock 안 쥐고 join → 빠른 반환
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/monitor/test_success_gather.py`
Expected: FAIL — `module 'success_gather' has no attribute 'wait_for_gather'`.

- [ ] **Step 3: success_gather.py 수정**

`gather_success_async` 의 `gather_success_images(...)` 호출에 TTL 인자 전달:
```python
                result = gather_success_images(
                    eqp_id, recipe_id,
                    downloader=_DOWNLOADER,
                    max_events=settings.gather_max_events,
                    refresh_ttl_sec=settings.consensus_refresh_ttl_sec,
                )
```
파일 말미(`__all__` 위)에 추가:
```python
def _cache_has_min_events(eqp_id, recipe_id) -> bool:
    """events/ 에 S 이미지가 1장 이상 있나(채워졌는지 거친 판정)."""
    from poc.workflow_3.align.consensus_gather import count_staged_events
    n_events, _ = count_staged_events(eqp_id, recipe_id)
    return n_events > 0


def wait_for_gather(eqp_id, recipe_id, timeout) -> bool:
    """진행 중인 gather thread 를 bounded join 한 뒤 캐시가 채워졌는지 반환한다.

    lock 안에서는 thread 스냅샷만 — join 은 lock 밖에서(데드락 방지). 알람 시점에 이미
    async fire 됐으면 그 thread 를 join(중복 fetch 회피). 없고 캐시도 비면 1회 fire 후 join.
    반환 bool = join 후 events/ 에 S 가 있나(resolver 는 True 일 때만 crop 재로드).
    """
    if not recipe_id or not DOWNLOADER_AVAILABLE:
        return _cache_has_min_events(eqp_id, recipe_id)

    key = (eqp_id, recipe_id)
    with _IN_FLIGHT_LOCK:
        thread = _IN_FLIGHT.get(key)
        if thread is not None and not thread.is_alive():
            thread = None
    if thread is None and not _cache_has_min_events(eqp_id, recipe_id):
        # 알람 fire 가 없었거나 이미 끝났는데 캐시가 비어 있음 → 1회 fire(내부에서 thread 등록).
        from poc.workflow_3.config import load_workflow3_settings
        thread = gather_success_async(eqp_id, recipe_id, load_workflow3_settings())

    if thread is not None:
        thread.join(timeout)
    return _cache_has_min_events(eqp_id, recipe_id)
```
`__all__` 에 `"wait_for_gather"` 추가.

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/monitor/test_success_gather.py`
Expected: 신규 2케이스 + 기존 PASS.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/monitor/success_gather.py poc/workflow_3/monitor/test_success_gather.py
git commit -m "feat(workflow_3): wait_for_gather (lock-safe bounded join of in-flight gather)"
```

---

## Task 7: consensus_resolve.py — resolve_templates 오케스트레이션

**Files:**
- Create: `poc/workflow_3/align/consensus_resolve.py`
- Test: `poc/workflow_3/align/test_consensus_resolve.py`

`resolve_templates(assets, *, eqp_id, consensus_enabled, min_s, max_events, sync_timeout_sec, cond_box_crop)` → `{"OM":tpl,"SEM":tpl}`. cache_key=`class/recipe`, 결과는 `ConsensusResult.template` 만 select 로(Codex#4), cold-cache 시 `wait_for_gather` 가 True 일 때만 1회 재로드(Codex#1).

- [ ] **Step 1: 실패 테스트 작성**

```python
# poc/workflow_3/align/test_consensus_resolve.py
"""resolve_templates 오케스트레이션 — 킬스위치/consensus 채택/insufficient 폴백/cold-sync."""
import sys, types
import poc.workflow_3.align.consensus_resolve as cr


class _Assets:
    eqp_id = "E1"; class_name = "c"; recipe_name = "r"


class _Tpl:
    def __init__(self, tag): self.tag = tag


def _patch(d):
    saved = {k: getattr(cr, k) for k in d}
    for k, v in d.items(): setattr(cr, k, v)
    return lambda: [setattr(cr, k, v) for k, v in saved.items()]


def test_killswitch_returns_rcp():
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"OM": _Tpl("rcp_om")},
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=False,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["OM"].tag == "rcp_om"
    finally:
        restore()


def test_consensus_adopted_when_enough():
    cons_tpl = _Tpl("cons_sem")
    class _Res:  template = cons_tpl; reason = "ok"; modality = "sem"; n_crops = 5; edge_ratio=1.0; lap_ratio=1.0
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"SEM": _Tpl("rcp_sem")},
        "build_center_tpls_for_sizing": lambda assets: {"sem": (_Tpl("center"), (0, 0))},
        "load_coregistered_crops": lambda *a, **k: {"sem": [object()] * 5},
        "build_consensus_template": lambda crops, *, recipe_id, modality, policy: _Res(),
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=True,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["SEM"].tag == "cons_sem"   # consensus 채택
    finally:
        restore()


def test_insufficient_falls_back_to_rcp_without_sync_when_warm():
    class _Res:  template = None; reason = "insufficient_s"; modality = "sem"; n_crops = 1; edge_ratio=None; lap_ratio=None
    calls = {"wait": 0}
    def _wait(*a, **k): calls["wait"] += 1; return False
    restore = _patch({
        "build_templates_from_assets": lambda assets, cond_box_crop: {"SEM": _Tpl("rcp_sem")},
        "build_center_tpls_for_sizing": lambda assets: {"sem": (_Tpl("center"), (0, 0))},
        # 캐시에 1장이라도 있으면 cold 아님 → sync 안 함:
        "load_coregistered_crops": lambda *a, **k: {"sem": [object()]},
        "build_consensus_template": lambda crops, *, recipe_id, modality, policy: _Res(),
        "wait_for_gather": _wait,
    })
    try:
        out = cr.resolve_templates(_Assets(), eqp_id="E1", consensus_enabled=True,
                                   min_s=4, max_events=8, sync_timeout_sec=8.0, cond_box_crop=True)
        assert out["SEM"].tag == "rcp_sem"
        assert calls["wait"] == 0            # warm(>=1 crop) → cold-sync 미발동
    finally:
        restore()


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try: fn(); print(f"[PASS] {fn.__name__}")
        except Exception: failed += 1; print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} pass"); sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_resolve.py`
Expected: FAIL — `ModuleNotFoundError: poc.workflow_3.align.consensus_resolve`.

- [ ] **Step 3: consensus_resolve.py 작성**

```python
# poc/workflow_3/align/consensus_resolve.py
"""resolve_templates — 실시간 보정용 라우팅 template(consensus 우선·rcp 폴백) 조립.

correction.correct_align_fail_auto 가 build_templates_from_assets 대신 이걸 호출한다.
어떤 실패(부족/blur/sync timeout/예외)든 해당 modality 는 rcp 로 강등 — 회귀 위험 0.
cache_key 는 반드시 "<class>/<recipe>"(gather 가 쓴 키) — assets.recipe_id(leaf)는 금지(Codex#5).
"""

from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR
from poc.workflow_3.align.templates import build_templates_from_assets
from poc.workflow_3.align.consensus_crops import (
    build_center_tpls_for_sizing, load_coregistered_crops,
)
from poc.workflow_3.align.consensus_template import (
    ConsensusPolicy, build_consensus_template, select_routing_templates,
)
from poc.workflow_3.monitor.success_gather import wait_for_gather

LOG_COMPONENT = "consensus_resolve"


def resolve_templates(assets, *, eqp_id, consensus_enabled, min_s, max_events,
                      sync_timeout_sec, cond_box_crop,
                      cache_root=ALIGN_CONSENSUS_CACHE_DIR):
    """{'OM'|'SEM': AlignKeyTemplate} 반환. consensus 신뢰 가능 modality 만 consensus, 그외 rcp."""
    rcp_route = build_templates_from_assets(assets, cond_box_crop=cond_box_crop)  # 런타임 라우팅(대문자)
    rcp_by_mod = {}                       # 소문자 키로 select 입력 정규화.
    for route_key, tpl in rcp_route.items():
        rcp_by_mod[route_key.lower()] = tpl
    if not consensus_enabled:
        print("[INFO] consensus 비활성(killswitch) → rcp 라우팅")
        return rcp_route

    cache_key = f"{assets.class_name}/{assets.recipe_name}"   # gather 가 쓴 키와 동일(leaf 금지).
    try:
        center_tpls = build_center_tpls_for_sizing(assets)
    except Exception as exc:
        print(f"[WARNING] consensus center tpl 실패 → rcp: {exc}")
        return rcp_route

    crops_by_mod = _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events)

    # cold-cache: 어떤 modality 도 min_s 를 못 채우고 캐시가 비면 1회 bounded sync 후 재로드.
    enough = any(len(c) >= min_s for c in crops_by_mod.values())
    if not enough:
        if wait_for_gather(eqp_id, cache_key, sync_timeout_sec) is True:   # bool 분기(Codex#1)
            crops_by_mod = _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events)
        # False(timeout/실패) → 재로드 없이 그대로 진행(아래서 insufficient → rcp).

    policy = ConsensusPolicy(min_s=min_s)
    cons_by_mod = {}
    for mod, crops in crops_by_mod.items():
        try:
            res = build_consensus_template(crops, recipe_id=cache_key, modality=mod, policy=policy)
        except Exception as exc:
            print(f"[WARNING] consensus build 예외({mod}) → rcp: {exc}")
            continue
        reason = res.reason if res.template is not None else f"rcp:{res.reason}"
        print(f"[INFO] consensus[{mod}] n={res.n_crops} edge={res.edge_ratio} "
              f"lap={res.lap_ratio} → {'consensus' if res.template is not None else reason}")
        if res.template is not None:
            cons_by_mod[mod] = res.template      # ConsensusResult.template 만(Codex#4)

    return select_routing_templates(cons_by_mod, rcp_by_mod)


def _safe_load(cache_root, eqp_id, cache_key, center_tpls, max_events):
    """load_coregistered_crops 의 예외/부재를 빈 dict 으로 흡수(rcp 폴백 보장)."""
    try:
        return load_coregistered_crops(cache_root, eqp_id, cache_key, center_tpls,
                                       max_events=max_events)
    except Exception as exc:
        print(f"[WARNING] consensus crops 로드 예외 → rcp: {exc}")
        return {}
```

- [ ] **Step 4: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_consensus_resolve.py`
Expected: `3/3 pass`.

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_3/align/consensus_resolve.py poc/workflow_3/align/test_consensus_resolve.py
git commit -m "feat(workflow_3): resolve_templates — consensus-or-rcp routing with cold-cache bounded sync"
```

---

## Task 8: correction.py + cycle.py 통합(call-site swap)

**Files:**
- Modify: `poc/workflow_3/align/correction.py` (CorrectionConfig 필드 + 449행 swap)
- Modify: `poc/workflow_3/monitor/cycle.py` (`_exec_run_correction` 가 settings 값 전달)
- Test: `poc/workflow_3/align/test_correction.py` (기존에 통합 케이스 추가)

- [ ] **Step 1: 실패 테스트 작성(기존 test_correction.py 에 추가)**

```python
# poc/workflow_3/align/test_correction.py 에 추가
import poc.workflow_3.align.correction as corr


def test_correct_auto_uses_resolver(monkeypatch=None):
    """correct_align_fail_auto 가 resolve_templates 를 호출해 라우팅 dict 을 받는다."""
    called = {"resolve": 0}

    class _A:
        eqp_id = "E1"; class_name = "c"; recipe_name = "r"; recipe_dir = "/x"
    def _resolve(assets, **kw):
        called["resolve"] += 1
        from poc.workflow_3.align.matching.test_engine import make_synthetic_template
        from poc.workflow_3.align.matching.engine import build_template
        t = build_template(make_synthetic_template(key_type="box"),
                           recipe_id="r", version="v", key_type="sem")
        return {"SEM": t}

    orig_assets = corr.resolve_assets_auto
    orig_resolve = getattr(corr, "resolve_templates", None)
    corr.resolve_assets_auto = lambda **k: _A()
    corr.resolve_templates = _resolve
    try:
        # correct_align_fail 은 controller 가 필요 → dry-run mock 데모 컨트롤러 사용.
        monitor, _ = corr._make_primary_demo(key_in_view=True)
        out = corr.correct_align_fail_auto(monitor, dry_run=True,
                                           eqp_id="E1", recipe_name="c/r")
        assert called["resolve"] == 1
        assert out.status in ("corrected", "escalated_no_ok", "ok_detect_error",
                              "fallback_corrected", "fallback_escalated")
    finally:
        corr.resolve_assets_auto = orig_assets
        if orig_resolve is not None:
            corr.resolve_templates = orig_resolve
```

- [ ] **Step 2: 실패 확인**

Run: `uv run python poc/workflow_3/align/test_correction.py`
Expected: FAIL — `module 'correction' has no attribute 'resolve_templates'`(아직 import 안 함).

- [ ] **Step 3: correction.py 수정**

import 추가:
```python
from poc.workflow_3.align.consensus_resolve import resolve_templates
```
`CorrectionConfig` dataclass 에 필드 추가(기존 `cond_box_crop` 옆):
```python
    consensus_enabled: bool = True
    consensus_min_s: int = 4
    consensus_max_events: int = 8
    consensus_sync_timeout_sec: float = 8.0
```
449행 교체:
```python
    templates = resolve_templates(
        assets,
        eqp_id=eqp_id,
        consensus_enabled=config.consensus_enabled,
        min_s=config.consensus_min_s,
        max_events=config.consensus_max_events,
        sync_timeout_sec=config.consensus_sync_timeout_sec,
        cond_box_crop=config.cond_box_crop,
    )
```
(`if not templates:` 폴백 블록은 그대로 — resolver 가 빈 dict 이면 동일하게 no_assets 처리.)

- [ ] **Step 4: cycle.py 수정 — settings 값을 CorrectionConfig 로**

`_exec_run_correction` 의 `CorrectionConfig(...)` 에 추가:
```python
            config=CorrectionConfig(
                reregister_ratio_threshold=settings.reregister_second_ratio_threshold,
                cond_box_crop=settings.cond_box_crop,
                consensus_enabled=settings.consensus_enabled,
                consensus_min_s=settings.consensus_min_s,
                consensus_max_events=settings.gather_max_events,
                consensus_sync_timeout_sec=settings.consensus_sync_timeout_sec,
            ),
```

- [ ] **Step 5: 통과 확인**

Run: `uv run python poc/workflow_3/align/test_correction.py`
Expected: 신규 케이스 + 기존 전부 PASS.

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_3/align/correction.py poc/workflow_3/monitor/cycle.py poc/workflow_3/align/test_correction.py
git commit -m "feat(workflow_3): route correction through resolve_templates (consensus-preferred, rcp fallback)"
```

---

## Task 9: 전체 회귀 + 정리

**Files:**
- Modify/Delete: `poc/workflow_3/test_config_consensus.py`(Task 4 임시 — 유지하거나 제거)

- [ ] **Step 1: align 스모크 전체 실행**

```bash
uv run python poc/workflow_3/align/test_consensus_cv.py
uv run python poc/workflow_3/align/test_consensus_template.py
uv run python poc/workflow_3/align/test_consensus_crops.py
uv run python poc/workflow_3/align/test_consensus_resolve.py
uv run python poc/workflow_3/align/test_consensus_gather.py
uv run python poc/workflow_3/align/test_correction.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/align/matching/test_engine.py
```
Expected: 전부 pass(특히 `test_engine` 은 matcher 불변 회귀 확인).

- [ ] **Step 2: 개발 PC dry-run 루프 1알람 replay(소켓/RCS 없이 import·경로 무결성)**

```bash
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  ALIGN_FAIL_CONSENSUS=1 uv run python poc/workflow_3/monitor/align_fail_monitor.py
```
Expected: import 에러 없이 시작, downloader 부재면 `[INFO] success downloader 없음...` + consensus 휴면(rcp 폴백) 로그. (RCS 미존재로 사이클은 `rcs_unavailable` 이어도 정상.)

- [ ] **Step 3: README/CLAUDE.md 메모(선택)**

`poc/workflow_3/README.md` 의 consensus 섹션에 "live correction 에 consensus 라우팅 투입(ALIGN_FAIL_CONSENSUS, min_s=4); office downloader 가 채워야 활성" 1줄 추가. `align/__init__.py` 에 신규 모듈 export 가 관례면 추가(아니면 생략 — `__all__` optional).

- [ ] **Step 4: 커밋**

```bash
git add poc/workflow_3/README.md
git commit -m "docs(workflow_3): note consensus live-correction routing + activation condition"
```

---

## Self-Review 체크 결과

- **Spec 커버리지:** §2 아키텍처=Task1-8, §3 데이터흐름=Task7, §4-A 포팅=Task2, §4-B adapter=Task3, §4-C resolver+sync=Task6·7, §4-D TTL/원자swap=Task5, §5 downloader 계약=비범위(문서만), §6 config=Task4, §7 에러강등=Task3·5·7(예외→rcp), §8 테스트=각 Task, §9 롤아웃=Task9. 누락 없음.
- **Placeholder:** 모든 코드 step 에 실제 코드 포함. 포팅은 bench 출처 명시 + 코드 inline(verbatim). TBD 없음.
- **타입 일관성:** `ConsensusResult`/`ConsensusPolicy`/`build_consensus_template`/`select_routing_templates`/`load_coregistered_crops`/`build_center_tpls_for_sizing`/`resolve_templates`/`wait_for_gather`/`gather_success_images(refresh_ttl_sec=)`/`CorrectionConfig` 필드명 = 전 Task 동일. cache_key=`class/recipe` 일관. select 입력은 항상 `AlignKeyTemplate|None`(ConsensusResult 아님).
