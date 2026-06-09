# C4 Phase 0+1 — ensemble lab(workflow_2) + RRF robustness + periodicity (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ensemble 실험을 workflow_2 에 격리한 lab 모듈로 옮기고(production 엔진 무수정), 그 위에서 RRF 대표-선택 robustness(Phase 0)와 template 주기성 점수(Phase 1)를 얹는다.

**Architecture:** **개발 위치 규약(사용자 확정 2026-06-10)** — ensemble 개선은 workflow_2 에서 실험·측정(drivers: `golden_localization_eval_cond.py`, `golden_consensus_eval_cond.py`), 검증되면 workflow_3/vision 으로 포팅. 따라서 신규 `poc/workflow_2/ensemble_lab.py` 가 채널 solo·primitive 는 workflow_3 에서 **import 재사용**하고, *실험 부분만*(rescore 대표·`template_periodicity`·[Phase2] C4 채널) 포크한다. 기본 인자는 workflow_3 production ensemble 과 **bit-parity**(검증 완료: fused 24개 동일).

**Tech Stack:** Python 3.10+, OpenCV, NumPy, pytest(`uv run pytest`). 절대 import. Korean docstring, print-based logging. Mac 합성 + import smoke 로 검증(office 데이터 불요).

**Spec:** `poc/workflow_2/docs/specs/2026-06-10-c4-context-distinctiveness-channel-design.md` (§5 Phase 0, §6.2 Phase 1, §10 개발 위치).

**범위:** Phase 0 + Phase 1 만. Phase 2(C4 context-patch NCC 채널) 및 workflow_3 포팅은 검증 게이트 통과 후 별도 plan.

---

## File Structure

| 파일 | 책임 | 변경 |
|---|---|---|
| `poc/workflow_2/ensemble_lab.py` | **신규** ensemble 실험장 — 포크된 융합 + 주기성 | Task 1(`rrf_fuse`,`compute_ensemble_candidates`), Task 2(`template_periodicity`) |
| `poc/workflow_2/test_ensemble_lab.py` | **신규** lab 단위 테스트(pytest) | Task 1·2 테스트 |
| `poc/workflow_2/golden_consensus_eval_cond.py` | consensus 평가/계측(driver) | Task 3(`template_periodic_rate` 집계) |
| `poc/workflow_3/vision/*` | production 엔진 | **무수정**(import 재사용만; 포팅은 검증 후 별도) |

**Baseline (시작 전):** `uv run pytest poc/workflow_3/vision/test_ensemble_proposer.py -q` → `10 passed`(엔진 무회귀 기준선).

---

## Task 1 (Phase 0): `ensemble_lab.py` 포크 + `rescore_fn` 대표 선택

**Files:**
- Create: `poc/workflow_2/ensemble_lab.py`
- Test: `poc/workflow_2/test_ensemble_lab.py`

근거: 실험을 production 엔진과 분리(개발 위치 규약). 채널 solo 는 재사용하고 융합만 포크해 (a) workflow_3 ensemble 과 parity 인 baseline 확보, (b) C4(이종 NCC-isolation 점수) 합류 시 대표를 공통 yardstick 으로 고르는 `rescore_fn` 추가(Codex #3).

- [ ] **Step 1: 실패 테스트 작성** — `poc/workflow_2/test_ensemble_lab.py` 신규 생성

```python
import numpy as np
import cv2

from poc.workflow_2 import ensemble_lab as lab
from poc.workflow_3.vision import ensemble_proposer as ep


def _sq(size=200, box=(70, 70, 60, 60), bg=110, edge=230):
    img = np.full((size, size), bg, np.uint8)
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 2)
    return img


def test_lab_ensemble_parity_with_workflow3():
    # lab 기본 채널/무 rescore = workflow_3 production ensemble 과 bit-parity (실험 baseline).
    frame = _sq(240, (120, 90, 60, 60))
    tpl = _sq(80, (10, 10, 60, 60))
    a = ep.compute_ensemble_candidates(tpl, frame, top_n=8, shadow_n=24)
    b = lab.compute_ensemble_candidates(tpl, frame, top_n=8, shadow_n=24)
    pa = [(c.xy, round(c.score, 6), c.scale) for c in a.fused]
    pb = [(c.xy, round(c.score, 6), c.scale) for c in b.fused]
    assert pa == pb


def test_lab_rrf_fuse_rescore_fn_overrides_raw_representative():
    # 한 클러스터 두 멤버: raw score 는 A(20,20) 우세지만 yardstick 은 B(22,22) 선호 → 대표 B (Codex #3).
    A = [ep._Cand(xy=(20, 20), score=9.0, scale=0.6)]
    B = [ep._Cand(xy=(22, 22), score=1.0, scale=1.2)]

    def rescore(xy, scale):
        return 100.0 if xy == (22, 22) else 0.0

    fused = lab.rrf_fuse([A, B], k0=10, match_radius=5, top_n=1, rescore_fn=rescore)
    assert fused[0].xy == (22, 22) and fused[0].scale == 1.2


def test_lab_rrf_fuse_default_representative_unchanged():
    # rescore_fn 없으면 raw score 최댓값 멤버가 대표 — workflow_3 동작 보존.
    A = [ep._Cand(xy=(20, 20), score=9.0, scale=0.6)]
    B = [ep._Cand(xy=(22, 22), score=1.0, scale=1.2)]
    fused = lab.rrf_fuse([A, B], k0=10, match_radius=5, top_n=1)
    assert fused[0].xy == (20, 20) and fused[0].scale == 0.6
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_2/test_ensemble_lab.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'poc.workflow_2.ensemble_lab'`

- [ ] **Step 3: `ensemble_lab.py` 구현** — 신규 파일 전체

```python
"""ensemble 개선 실험장 (workflow_2). production 엔진(workflow_3/vision)을 건드리지 않고
새 융합/주기성/[Phase2] 채널을 시험한다. 검증되면 workflow_3 으로 포팅한다.

drivers: golden_localization_eval_cond.py, golden_consensus_eval_cond.py.
재사용: 채널 solo 후보·primitive 는 workflow_3.ensemble_proposer 에서 import. 실험 부분
(rescore 대표 선택, template_periodicity, [Phase2] C4 'context' 채널)만 여기 포크한다.
기본 인자에서 compute_ensemble_candidates 는 workflow_3 ensemble 과 bit-parity.

실행 (인자 없음): uv run pytest poc/workflow_2/test_ensemble_lab.py
"""
import cv2
import numpy as np

from poc.workflow_3.vision.ensemble_proposer import (
    EnsembleResult, RRF_K0, SHADOW_N, _Cand, _channel_solo_candidates,
)
from poc.workflow_3.vision.align_key_matcher import DEFAULT_SCALES, _to_grayscale


def rrf_fuse(channel_lists, *, k0=RRF_K0, match_radius=8, top_n=SHADOW_N, rescore_fn=None):
    """RRF 융합(포크). fused(c) = Σ_채널 1/(k0 + rank). 채널 간 후보는 center 거리 <=
    match_radius(Chebyshev) 면 동일 후보로 묶는다. 반환 list[_Cand](fused 내림차순, top_n).

    대표(xy/scale) 선택:
      - rescore_fn 없음(기본): 클러스터 멤버 중 raw score 최댓값 — workflow_3 _rrf_fuse 동작 보존.
      - rescore_fn 있음: 멤버 위치를 공통 yardstick rescore_fn(xy, scale)->float 로 재평가해 대표
        선택. 이종 채널(C4 NCC-isolation) 합류 시 raw score 비교 불가 문제 해소(Codex 리뷰 #3).
        rescore 실패(예외)/전무는 raw 대표 폴백.
    """
    clusters = []  # {"xy","score","scale","rrf","members":[(xy, score, scale)]}
    for ch_list in channel_lists:
        ranked = sorted(ch_list, key=lambda c: c.score, reverse=True)
        for rank, cand in enumerate(ranked, 1):
            hit = next((cl for cl in clusters
                        if abs(cl["xy"][0] - cand.xy[0]) <= match_radius
                        and abs(cl["xy"][1] - cand.xy[1]) <= match_radius), None)
            contrib = 1.0 / (k0 + rank)
            if hit is None:
                clusters.append({"xy": cand.xy, "score": cand.score, "scale": cand.scale,
                                 "rrf": contrib,
                                 "members": [(cand.xy, cand.score, cand.scale)]})
            else:
                hit["rrf"] += contrib
                hit["members"].append((cand.xy, cand.score, cand.scale))
                if cand.score > hit["score"]:           # raw 대표 추적(무 rescore/폴백 경로).
                    hit["xy"], hit["score"], hit["scale"] = cand.xy, cand.score, cand.scale
    if rescore_fn is not None:
        for cl in clusters:
            best = None  # (rescore, xy, scale)
            for (xy, _s, scale) in cl["members"]:
                try:
                    rs = float(rescore_fn(xy, scale))
                except Exception:
                    continue
                if best is None or rs > best[0]:
                    best = (rs, xy, scale)
            if best is not None:
                cl["xy"], cl["scale"] = best[1], best[2]   # 대표를 공통 yardstick 으로 교체.
    clusters.sort(key=lambda cl: cl["rrf"], reverse=True)
    return [_Cand(xy=cl["xy"], score=cl["rrf"], scale=cl["scale"]) for cl in clusters[:top_n]]


def compute_ensemble_candidates(template_gray, frame_gray, *,
                                channels=("canny", "scharr", "orient"),
                                top_n=8, shadow_n=SHADOW_N, k0=RRF_K0,
                                scales=DEFAULT_SCALES, rescore_fn=None):
    """lab ensemble — channel 선택 + rescore 대표. 기본 인자는 workflow_3
    compute_ensemble_candidates 와 bit-parity(parity 테스트로 고정).
    """
    th, tw = _to_grayscale(template_gray).shape[:2]
    short = max(1, min(tw, th))
    match_r = max(8, int(0.05 * short))
    solo = {ch: _channel_solo_candidates(template_gray, frame_gray, ch, scales=scales)
            for ch in channels}
    fused = rrf_fuse(list(solo.values()), k0=k0, match_radius=match_r,
                     top_n=shadow_n, rescore_fn=rescore_fn)
    return EnsembleResult(fused=fused, top_n_count=top_n, solo=solo)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_2/test_ensemble_lab.py -q`
Expected: `3 passed`. (parity + rescore override + default 보존)

- [ ] **Step 5: 엔진 무회귀 확인** — workflow_3 손 안 댔는지 검증

Run: `uv run pytest poc/workflow_3/vision/test_ensemble_proposer.py -q`
Expected: `10 passed` (변동 없음 — lab 은 import 만).

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_2/ensemble_lab.py poc/workflow_2/test_ensemble_lab.py
git commit -m "$(printf 'workflow_2(ensemble lab Phase 0): production 무수정 포크 + rescore_fn 대표\n\nensemble 실험을 workflow_2 ensemble_lab 으로 격리(개발 위치 규약): 채널 solo 는\nworkflow_3 재사용, 융합만 포크. 기본 인자 bit-parity(parity 테스트). rescore_fn 으로\n이종 채널 대표를 공통 yardstick 선택(Codex #3). 3 passed, 엔진 10 passed 무회귀.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 2 (Phase 1): `template_periodicity()` — 내재 주기성 점수

**Files:**
- Modify: `poc/workflow_2/ensemble_lab.py` (상수 + 함수 추가)
- Test: `poc/workflow_2/test_ensemble_lab.py`

근거: NCC-isolation 만으론 주기성을 못 본다(Codex #1). template 2D autocorrelation off-center peak 로 내재 주기성을 0..1 로 잰다. Mac 합성 검증값: grating 1.0 / contact-array 1.0 / unique-blob 0.361 / flat 0.0 → τ=0.5 가 주기 vs 유일을 가른다.

- [ ] **Step 1: 실패 테스트 작성** — `test_ensemble_lab.py` 끝에 추가

```python
def test_template_periodicity_high_on_grating():
    g = np.zeros((120, 120), np.uint8)
    g[:, ::20] = 255                      # 주기 20px 수직 줄무늬
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    assert lab.template_periodicity(g) > lab.PERIODICITY_TAU


def test_template_periodicity_high_on_contact_array():
    a = np.full((120, 120), 110, np.uint8)
    for yy in range(12, 120, 24):
        for xx in range(12, 120, 24):
            cv2.circle(a, (xx, yy), 5, 230, -1)   # 주기 24px dot 격자
    assert lab.template_periodicity(a) > lab.PERIODICITY_TAU


def test_template_periodicity_low_on_unique_blob():
    b = np.full((120, 120), 110, np.uint8)
    cv2.circle(b, (60, 60), 12, 230, -1)          # 단일 유일 블롭
    assert lab.template_periodicity(b) < lab.PERIODICITY_TAU


def test_template_periodicity_zero_on_flat():
    f = np.full((100, 100), 128, np.uint8)        # 무특징 grey
    assert lab.template_periodicity(f) == 0.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_2/test_ensemble_lab.py::test_template_periodicity_zero_on_flat -v`
Expected: FAIL — `AttributeError: module 'poc.workflow_2.ensemble_lab' has no attribute 'template_periodicity'`

- [ ] **Step 3: 상수 + 함수 구현** — `ensemble_lab.py` 의 import 블록 **아래**(첫 `def rrf_fuse` 위)에 추가

```python
# Phase 1: template 내재 주기성(autocorrelation off-center peak). cold-start, 오피스 보정 예정.
PERIODICITY_EXCL_FRAC = 0.10   # zero-lag 제외 중심 반경 = min(h,w) 의 이 비율.
PERIODICITY_TAU = 0.5          # 이 이상이면 template_periodic(=재등록 후보). 합성 검증으로 선택.


def template_periodicity(template_gray):
    """template 의 내재 주기성 점수 [0,1] — 2D autocorrelation 의 off-center peak 높이.

    높을수록 반복 패턴(grating/array) → 어떤 위치도 유일하지 않음(= align key 모호, 재등록 후보).
    0=유일/무특징. scale 무관(상대 비율). NCC-isolation 이 못 보는 주기성을 보강(Codex 리뷰 #1).
    """
    g = _to_grayscale(template_gray).astype(np.float32)
    g = g - g.mean()
    if g.std() < 1e-6:
        return 0.0                                   # 무특징 grey → 주기성 정의 안 됨.
    F = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))   # 2D autocorrelation.
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    peak0 = ac[cy, cx]
    if peak0 <= 0:
        return 0.0
    ac = ac / peak0                                  # zero-lag = 1.0 정규화.
    r = max(1, int(PERIODICITY_EXCL_FRAC * min(h, w)))
    yy, xx = np.ogrid[:h, :w]
    outside = (yy - cy) ** 2 + (xx - cx) ** 2 > r * r
    if not outside.any():
        return 0.0
    return float(np.clip(ac[outside].max(), 0.0, 1.0))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest poc/workflow_2/test_ensemble_lab.py -q`
Expected: `7 passed` (Task 1 의 3 + 신규 4).

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_2/ensemble_lab.py poc/workflow_2/test_ensemble_lab.py
git commit -m "$(printf 'workflow_2(ensemble lab Phase 1): template_periodicity 내재 주기성 점수\n\n2D FFT autocorrelation off-center peak [0,1]. 합성: grating/array 1.0,\nunique-blob 0.361, flat 0.0 (tau=0.5). NCC-isolation 이 못 보는 주기성 보강\n(Codex 리뷰 #1) + 재등록 후보 신호. 7 passed.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## Task 3 (Phase 1): golden 평가에 `template_periodic_rate` 집계

**Files:**
- Modify: `poc/workflow_2/golden_consensus_eval_cond.py` (import, recipe 루프, summary)

근거: Phase 1 의 독립적 가치 — 현재 등록된 align key 중 몇 %가 내재적으로 주기적(=재등록 후보)인지 golden set 에서 측정. C4 없이도 actionable. eval 은 office 데이터 필요라 Mac 에선 import smoke 로만 검증(실측은 오피스).

- [ ] **Step 1: import 추가** — `golden_consensus_eval_cond.py` 의 `from poc.workflow_3.vision.align_point_correction import _tool_label` 줄 **위**에 삽입

```python
from poc.workflow_2.ensemble_lab import PERIODICITY_TAU, template_periodicity
```

- [ ] **Step 2: 루프 앞 리스트 초기화** — `run()` 의 `by_recipe = {}` 블록을 교체

기존:
```python
    by_recipe = {}
    mod_total = Counter()
    drop_total = Counter()
```
변경:
```python
    by_recipe = {}
    mod_total = Counter()
    drop_total = Counter()
    periodicities = []   # [(rec_key, periodicity)] — Phase 1 재등록 후보 신호.
```

- [ ] **Step 3: recipe 루프에서 periodicity 수집** — `rec_key = _recipe_key(assets)` 줄 **아래**, `if n_s:` **위**에 삽입

기존:
```python
        rec_key = _recipe_key(assets)          # eqp/class/recipe 고유 키 — leaf 충돌 방지.
        if n_s:
            by_recipe[rec_key] = entry
```
변경:
```python
        rec_key = _recipe_key(assets)          # eqp/class/recipe 고유 키 — leaf 충돌 방지.
        # template 내재 주기성(Phase 1): 등록 key 가 반복패턴이면 어떤 위치도 유일하지 않음
        # → 재등록 후보. modality 중 max(가장 주기적인 쪽)로 recipe 를 대표.
        rec_periodicity = max(
            (template_periodicity(t.raw_image) for t, _off in center_tpls.values()
             if t is not None), default=0.0)
        periodicities.append((rec_key, round(rec_periodicity, 3)))
        if n_s:
            by_recipe[rec_key] = entry
```

- [ ] **Step 4: 집계·출력** — modality 출력 블록 `print(f"\n[INFO] === msr S 해결된 modality...` **앞**에 삽입

```python
    # === template 내재 주기성(Phase 1): 재등록 후보 비율 ===
    n_periodic = sum(1 for _k, p in periodicities if p > PERIODICITY_TAU)
    n_tpl = len(periodicities)
    periodic_rate = round(n_periodic / n_tpl, 3) if n_tpl else 0.0
    worst = sorted(periodicities, key=lambda kp: kp[1], reverse=True)
    print(f"\n[INFO] === template 주기성(재등록 후보, tau={PERIODICITY_TAU}) === "
          f"periodic {n_periodic}/{n_tpl} (rate={periodic_rate}) — 상위: "
          + ", ".join(f"{k}={p}" for k, p in worst[:5]))
```

- [ ] **Step 5: summary 키 추가** — `res["proposer"] = ...` 줄 **아래**에 삽입

```python
    res["template_periodic_rate"] = periodic_rate              # Phase 1 재등록 후보 비율.
    res["template_periodicities"] = dict(periodicities)        # recipe 별 주기성(재등록 우선순위).
```

- [ ] **Step 6: import smoke 검증 (Mac)**

Run: `uv run python poc/workflow_2/golden_consensus_eval_cond.py 2>&1 | tail -5`
Expected: office 데이터 없으므로 `[ERROR] golden 데이터를 찾지 못했습니다: ... (env ALIGN_GOLDEN_ROOT).` 한 줄로 종료. **Traceback/ImportError 없음**(import·문법 무오류). 실제 rate 는 오피스 run.

- [ ] **Step 7: 커밋**

```bash
git add poc/workflow_2/golden_consensus_eval_cond.py
git commit -m "$(printf 'workflow_2(consensus Phase 1): template_periodic_rate 집계\n\n등록 align key 중 내재 주기적(template_periodicity>tau)인 비율 = 재등록 후보\n신호. recipe 별 주기성 summary.json 기록(재등록 우선순위). lab.template_periodicity\n재사용. C4 없이도 actionable.\n\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')"
```

---

## 검증 게이트 (Phase 2 진입 전, 오피스)

1. `uv run pytest poc/workflow_2/test_ensemble_lab.py -q` → `7 passed`.
2. `uv run pytest poc/workflow_3/vision/test_ensemble_proposer.py -q` → `10 passed` (엔진 무회귀).
3. `uv run python poc/workflow_2/golden_consensus_eval_cond.py` → `template_periodic_rate` 확인.
   - rate 유의미(예: ≥0.2)면 재등록 레버 실재 → Phase 2 trust-weight 토대.
   - rate≈0 이면 골든셋 key 가 주기적이지 않음 → C4 주기성 페널티 효용 재검토.
4. 결과 보고 **Phase 2(C4 context-patch NCC) plan 착수 여부** 결정(spec §4 실행 결정).
   Phase 2 는 lab 에 `'context'` 채널 추가 + golden_localization_eval_cond/golden_consensus_eval_cond
   양쪽 ablation. workflow_3 포팅은 그 다음 단계.

---

## Self-Review

- **Spec 커버리지**: §10 개발 위치(workflow_2 lab)→Task 1 파일 배치; §5 대표 robustness→Task 1 `rescore_fn`; §6.2 periodicity→Task 2; "재등록 신호"→Task 3. per-channel weight·C4 채널·chamfer 생존 게이트·bin 계측은 §6.3/§7 = Phase 2(의도적 제외, 실행 결정 일치).
- **Placeholder 스캔**: 없음 — 모든 step 에 실제 코드/명령/기대출력.
- **타입 일관성**: `rrf_fuse(..., rescore_fn=None)`·`compute_ensemble_candidates(..., channels=, rescore_fn=)`·`template_periodicity(np.ndarray)->float`·`PERIODICITY_TAU` 가 Task 1/2 정의와 Task 3 사용에서 동일. `center_tpls.values()` = `(tpl, offset)` 튜플(기존 `_build_cond_by_recipe` 와 동일) → Task 3 `for t, _off in center_tpls.values()` 정합. lab import 심볼(`_Cand`,`_channel_solo_candidates`,`EnsembleResult`,`RRF_K0`,`SHADOW_N`,`_to_grayscale`,`DEFAULT_SCALES`)은 workflow_3 에 실재(parity 실행으로 확인).
- **production 무수정**: Task 1·2 는 workflow_2 만; Task 1 Step 5 가 엔진 10 passed 로 무회귀 가드. workflow_3 포팅은 검증 게이트 후 별도 plan.
