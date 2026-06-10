# Tier 1.1 box-crop localization 검증 — 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `golden_localization_eval_cond.py` 에 miss-distance bin 층화 리포트를 더해, cond-box-crop+offset 이 far/very-far 구조적 miss 를 center-crop 대비 살리는지 office 1회 실행으로 게이트한다.

**Architecture:** 리포팅 전용 확장 — matcher / `_localize` / offset 계산은 무수정(bit-parity). 순수 함수 3개(bin 분류 + arm 집계 + 리포트)를 `all_rows` 후처리로 추가하고, `run()` 에 `FORCE_ENSEMBLE` 강제와 `summary["binned"]` 출력을 배선한다. 모든 신규 로직은 Mac 합성 입력으로 TDD; 실제 숫자는 office 가 채운다.

**Tech Stack:** Python 3.10+, numpy, pytest, `uv`. 대상 파일은 `poc/workflow_2/` lab(`golden_localization_eval_cond.py` = `glec`, `golden_localization_eval.py` = `gle`).

**설계 근거:** `poc/workflow_2/docs/superpowers/specs/2026-06-11-box-crop-localization-validation-design.md`

---

## 앙상블 알고리즘은 무엇을 바꾸나? (핵심: 알고리즘 자체는 0)

이 검증 작업은 **앙상블 매칭 알고리즘을 한 줄도 바꾸지 않는다**(bit-parity 규칙). 정리하면:

**바꾸지 않는 것 — 앙상블 알고리즘 내부 전부 그대로:**
- C1/C2/C3 proposer (Canny / Scharr / orientation-binned chamfer)
- RRF fusion (k0=10, top-N)
- NCC reranker (selection = chamfer 0.5 + ncc 0.5)
- Youden 보정 임계 (match 0.6053 / adjust 0.4727)
- scale band, distinctiveness gate, 전처리(CLAHE/Canny) 등
→ `compute_align_key_score_ensemble` 와 `ensemble_proposer.py` 는 **무수정**. 손대면 bit-parity 위반.

**바꾸는 것 — 앙상블의 *사용 방식* 딱 하나:**
- `_apply_matcher_default()` 가 `ALIGN_USE_ENSEMBLE` 환경변수 기본값을 `1` 로 채운다
  (`os.environ.setdefault`). 그러면 eval 의 `_matcher_for_eval()` 가 baseline
  (`compute_align_key_score`) 대신 **기존** ensemble(`compute_align_key_score_ensemble`)을 호출한다.
- 즉 "어떤 매처를 부를지" 선택만 바뀐다. 알고리즘은 그대로. `ALIGN_USE_ENSEMBLE=0` 으로
  baseline 비교도 여전히 가능(escape hatch).

**그럼 box-crop 은 앙상블과 어떻게 엮이나? — 알고리즘이 아니라 *입력 template* 의 변화:**
- box-crop 은 `_build_offset_templates_cond`(이미 lab 에 존재, 무수정)가 rcp 를
  `cond.box_ltrb` 로 crop 한 `box` template 과 center-area crop 한 `center` template
  **두 벌**을 만드는 것뿐이다. 이 template 생성 로직도 이번에 안 바꾼다.
- 검증은 **동일한 ensemble 매처**에 (a) center template, (b) box template 을 각각 먹여
  localization(gt_in_topk/rank1)이 어떻게 달라지나를 miss-distance bin 별로 잰다.
  알고리즘은 상수, 입력만 A/B.
- 측정하는 가설은 "앙상블을 고치자"가 아니라 **"앙상블에 더 변별력 있는 template
  (box-cropped unique region)을 주면 far/very-far localization 이 오르나?"** 이다.

요약: 이번 작업의 코드 변경은 **전부 (1) 리포팅 추가 + (2) 매처 선택 토글** 이며, 앙상블
알고리즘과 box-crop template 생성 로직은 **둘 다 기존 코드를 그대로 호출**한다. 알고리즘
개선은 이 게이트를 통과한 *뒤* production 포팅 단계(별도 spec)에서, 그것도 알고리즘이 아니라
template 준비 경로를 바꾸는 식으로만 일어난다.

---

## File Structure

- **Modify** `poc/workflow_2/golden_localization_eval_cond.py`
  - bin 상수(`DISP_BINS`/`RESCUE_MULT`/`BIN_FRAME`) + 순수 함수(`_bin_label`/`displacement_bin`/`rescue_bin`/`_arm_rates`/`_binned_localization_report`) + 출력(`_print_binned_report`) + `FORCE_ENSEMBLE`/`_apply_matcher_default`.
  - `_process_msr_cond` 의 row 에 `frame_hw` 1줄 추가. `run()` 에 배선.
- **Modify** `poc/workflow_2/test_golden_localization_eval_cond.py`
  - 신규 순수 함수용 pytest 케이스 추가(합성 입력).

전부 한 파일 쌍 안의 추가라 새 파일은 없다. 측정 로직(`gle._localize`/`_summarize`/`_matcher_for_eval`)은 건드리지 않는다.

---

## Task 1: bin 분류 헬퍼 + 상수

**Files:**
- Modify: `poc/workflow_2/golden_localization_eval_cond.py`
- Test: `poc/workflow_2/test_golden_localization_eval_cond.py`

- [ ] **Step 1: 실패 테스트 작성**

`test_golden_localization_eval_cond.py` 끝에 추가:

```python
# --- Tier 1.1: miss-distance bin 분류 (순수) ---

def test_displacement_bin_boundaries():
    # frame 512x512 → center (256,256), short=512. norm = |GT-center|/512.
    fhw = (512, 512)
    assert glec.displacement_bin((296, 256), fhw) == "near"      # 40/512=0.078
    assert glec.displacement_bin((333, 256), fhw) == "mid"       # 77/512=0.150
    assert glec.displacement_bin((386, 256), fhw) == "far"       # 130/512=0.254
    assert glec.displacement_bin((456, 256), fhw) == "veryfar"   # 200/512=0.391


def test_rescue_bin_boundaries():
    tol = glec.gle.GT_TOL_NORM   # center dist_norm 을 tol 배수로 줘 경계 검증.
    assert glec.rescue_bin(0.5 * tol) == "hit"
    assert glec.rescue_bin(1.5 * tol) == "near"
    assert glec.rescue_bin(3.0 * tol) == "far"
    assert glec.rescue_bin(5.0 * tol) == "veryfar"
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py::test_displacement_bin_boundaries poc/workflow_2/test_golden_localization_eval_cond.py::test_rescue_bin_boundaries -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'displacement_bin'`.

- [ ] **Step 3: 최소 구현**

`golden_localization_eval_cond.py` 의 기존 `OFFSET_WARN/OFFSET_SKIP` 상수 근처에 추가:

```python
# --- Tier 1.1 검증: miss-distance bin 층화 (리포팅 전용; 경계는 1차 결과 후 재조정 가능) ---
# A: GT 가 frame 중심에서 떨어진 정도(frame 짧은변 비율) = 구조적 displacement.
DISP_BINS = ((0.10, "near"), (0.20, "mid"), (0.35, "far"))   # 그 외 → "veryfar".
# B: center-crop arm 이 GT 에서 빗나간 거리(GT_TOL_NORM 배수) = rescue framing.
RESCUE_MULT = ((1.0, "hit"), (2.0, "near"), (4.0, "far"))    # 그 외 → "veryfar".
BIN_FRAME = "inpaint"   # 층화는 clean(inpaint) cell 로 — lever_verdict(box__inpaint)와 일관.


def _bin_label(value, edges, over_label):
    """value 를 오름차순 (경계, 라벨) edges 로 분류. 어느 경계도 안 넘으면 over_label."""
    for edge, label in edges:
        if value < edge:
            return label
    return over_label


def displacement_bin(gt_xy, frame_hw):
    """GT(정렬점)가 frame 중심에서 얼마나 떨어졌나 → near/mid/far/veryfar.

    norm = |GT - frame_center| / frame 짧은변. '구조적 displacement' 의 직접 척도.
    """
    h, w = frame_hw
    short = max(1, min(int(w), int(h)))
    norm = float(np.hypot(gt_xy[0] - w / 2.0, gt_xy[1] - h / 2.0) / short)
    return _bin_label(norm, DISP_BINS, "veryfar")


def rescue_bin(center_dist_norm):
    """center-crop arm 이 GT 에서 얼마나 빗나갔나 → hit/near/far/veryfar (GT_TOL_NORM 배수)."""
    return _bin_label(float(center_dist_norm) / gle.GT_TOL_NORM, RESCUE_MULT, "veryfar")
```

> `np` 와 `gle` 는 이미 이 모듈에서 import 되어 있다(파일 상단 확인). 없으면 추가.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py::test_displacement_bin_boundaries poc/workflow_2/test_golden_localization_eval_cond.py::test_rescue_bin_boundaries -q`
Expected: PASS (2 passed).

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_2/golden_localization_eval_cond.py poc/workflow_2/test_golden_localization_eval_cond.py
git commit -m "test(wf2): Tier 1.1 bin 분류 헬퍼(displacement/rescue) + 경계 테스트"
```

---

## Task 2: arm 집계 + bin 리포트

**Files:**
- Modify: `poc/workflow_2/golden_localization_eval_cond.py`
- Test: `poc/workflow_2/test_golden_localization_eval_cond.py`

- [ ] **Step 1: 실패 테스트 작성**

`test_golden_localization_eval_cond.py` 에 추가. 합성 `_localize` 결과/row 헬퍼 + 집계·결손 검증:

```python
# --- Tier 1.1: bin × arm 집계 ---

def _loc(in_topk, hit, dist_norm=0.0):
    """합성 _localize 결과(집계가 쓰는 키만)."""
    return {"in_topk": in_topk, "hit": hit, "dist_norm": dist_norm,
            "topk_rank": 1 if hit else (2 if in_topk else None),
            "align_xy": (0, 0), "mod": "om", "score": 0.5}


def _row(center, box, gt_xy, frame_hw, label="S"):
    cells = {}
    if center is not None:
        cells["center__inpaint"] = center
    if box is not None:
        cells["box__inpaint"] = box
    return {"label": label, "crosshair_xy": gt_xy, "frame_hw": frame_hw, "cells": cells}


def test_binned_report_displacement_aggregates():
    fhw = (512, 512)   # 두 행 모두 near (norm<0.10).
    rows = [
        _row(_loc(in_topk=True, hit=True),  _loc(in_topk=True, hit=False), (296, 256), fhw),
        _row(_loc(in_topk=False, hit=False), _loc(in_topk=True, hit=True),  (300, 256), fhw),
    ]
    near = glec._binned_localization_report(rows)["by_displacement"]["near"]
    assert near["center"] == {"n": 2, "gt_in_topk": 0.5, "rank1": 0.5}
    assert near["box"] == {"n": 2, "gt_in_topk": 1.0, "rank1": 0.5}


def test_binned_report_rescue_uses_center_distnorm():
    tol = glec.gle.GT_TOL_NORM
    fhw = (512, 512)
    rows = [
        _row(_loc(True, True, dist_norm=0.5 * tol), _loc(True, False), (256, 256), fhw),  # hit bin
        _row(_loc(False, False, dist_norm=3.0 * tol), _loc(True, True), (256, 256), fhw), # far bin
    ]
    by_miss = glec._binned_localization_report(rows)["by_center_miss"]
    assert by_miss["hit"]["box"] == {"n": 1, "gt_in_topk": 1.0, "rank1": 0.0}
    assert by_miss["far"]["box"] == {"n": 1, "gt_in_topk": 1.0, "rank1": 1.0}


def test_binned_report_skips_missing_box_and_nonS():
    fhw = (512, 512)
    rows = [
        _row(_loc(True, True), None, (296, 256), fhw),            # box 결손 → box 분모서 제외.
        _row(_loc(True, True), _loc(True, True), (296, 256), fhw, label="E"),  # 비-S → 무시.
    ]
    near = glec._binned_localization_report(rows)["by_displacement"]["near"]
    assert near["center"]["n"] == 1
    assert near["box"] == {"n": 0, "gt_in_topk": None, "rank1": None}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -k binned_report -q`
Expected: FAIL — `AttributeError: ... '_binned_localization_report'`.

- [ ] **Step 3: 최소 구현**

Task 1 헬퍼 아래에 추가:

```python
def _arm_rates(vals):
    """_localize 결과 표본 → {n, gt_in_topk, rank1}. _cell_stats 의 bin 표용 부분집합."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "gt_in_topk": None, "rank1": None}
    return {
        "n": n,
        "gt_in_topk": round(sum(1 for v in vals if v["in_topk"]) / n, 3),
        "rank1": round(sum(1 for v in vals if v["hit"]) / n, 3),
    }


def _binned_localization_report(all_rows):
    """S row 들을 두 방식으로 층화 → bin × arm(center/box) 의 gt_in_topk/rank1 집계.

    A by_displacement: GT-from-frame-center(구조적 displacement). frame_hw 필요.
    B by_center_miss : center-crop arm 의 dist_norm(rescue; box 가 center 실패를 건지나).
    둘 다 inpaint cell 기준. matcher 재실행 없이 row 후처리만. frame_hw 결손 행은
    A 에서 제외하고 경고 카운트만 올린다(조용한 누락 방지).
    """
    cc, bc = f"center__{BIN_FRAME}", f"box__{BIN_FRAME}"
    disp = {b: {"center": [], "box": []} for b in ("near", "mid", "far", "veryfar")}
    resc = {b: {"center": [], "box": []} for b in ("hit", "near", "far", "veryfar")}
    n_no_frame = 0
    for r in all_rows:
        if r.get("label") != "S":
            continue
        cells = r.get("cells", {})
        center, box = cells.get(cc), cells.get(bc)
        gt, fhw = r.get("crosshair_xy"), r.get("frame_hw")
        # A: displacement (GT + frame 크기).
        if gt is not None and fhw is not None:
            b = displacement_bin(gt, fhw)
            if center is not None:
                disp[b]["center"].append(center)
            if box is not None:
                disp[b]["box"].append(box)
        elif gt is not None and center is not None:
            n_no_frame += 1
        # B: rescue (center cell 의 dist_norm).
        if center is not None:
            b = rescue_bin(center["dist_norm"])
            resc[b]["center"].append(center)
            if box is not None:
                resc[b]["box"].append(box)

    def _roll(binmap, order):
        return {b: {arm: _arm_rates(binmap[b][arm]) for arm in ("center", "box")}
                for b in order}

    return {
        "frame": BIN_FRAME,
        "n_no_frame_hw": n_no_frame,
        "by_displacement": _roll(disp, ("near", "mid", "far", "veryfar")),
        "by_center_miss": _roll(resc, ("hit", "near", "far", "veryfar")),
    }
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -k binned_report -q`
Expected: PASS (3 passed).

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_2/golden_localization_eval_cond.py poc/workflow_2/test_golden_localization_eval_cond.py
git commit -m "test(wf2): Tier 1.1 _binned_localization_report(displacement+rescue) 집계"
```

---

## Task 3: FORCE_ENSEMBLE 강제(테스트 가능 헬퍼)

**Files:**
- Modify: `poc/workflow_2/golden_localization_eval_cond.py`
- Test: `poc/workflow_2/test_golden_localization_eval_cond.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# --- Tier 1.1: ensemble 매처 hardcode (setdefault escape hatch) ---

def test_apply_matcher_default_forces_ensemble(monkeypatch):
    import poc.workflow_2.golden_localization_eval as gle
    import poc.workflow_3.vision.align_key_matcher as akm
    monkeypatch.delenv("ALIGN_USE_ENSEMBLE", raising=False)
    glec._apply_matcher_default()                       # 미설정 → ensemble 로 채움.
    assert gle._matcher_for_eval() is akm.compute_align_key_score_ensemble


def test_apply_matcher_default_respects_explicit_off(monkeypatch):
    import poc.workflow_2.golden_localization_eval as gle
    import poc.workflow_3.vision.align_key_matcher as akm
    monkeypatch.setenv("ALIGN_USE_ENSEMBLE", "0")       # 명시적 0 → 유지(escape hatch).
    glec._apply_matcher_default()
    assert gle._matcher_for_eval() is akm.compute_align_key_score
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -k apply_matcher_default -q`
Expected: FAIL — `AttributeError: ... '_apply_matcher_default'`.

- [ ] **Step 3: 최소 구현**

파일 상단 import 에 `os` 가 없으면 추가(`import os`). bin 상수 근처에:

```python
FORCE_ENSEMBLE = True   # Tier 1.1 검증은 항상 production ensemble 매처로.


def _apply_matcher_default():
    """FORCE_ENSEMBLE 이면 ALIGN_USE_ENSEMBLE 기본값을 1 로 채운다(명시적 0 은 존중)."""
    if FORCE_ENSEMBLE:
        os.environ.setdefault("ALIGN_USE_ENSEMBLE", "1")
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -k apply_matcher_default -q`
Expected: PASS (2 passed).

- [ ] **Step 5: 커밋**

```bash
git add poc/workflow_2/golden_localization_eval_cond.py poc/workflow_2/test_golden_localization_eval_cond.py
git commit -m "test(wf2): Tier 1.1 FORCE_ENSEMBLE _apply_matcher_default(setdefault escape hatch)"
```

---

## Task 4: run() 배선 + frame_hw 캡처 + 출력 (glue, office 검증)

**Files:**
- Modify: `poc/workflow_2/golden_localization_eval_cond.py`

> 이 Task 는 통합 glue(파일 I/O·전체 파이프라인 의존)라 Mac 단위테스트 대상이 아니다 — Task 1~3 의 순수 함수가 동작을 보증하고, 실제 표는 office 실행이 검증한다. (기존 테스트도 `run()`/`_process_msr_cond` 는 단위테스트하지 않는다.)

- [ ] **Step 1: frame_hw 를 row 에 기록**

`_process_msr_cond` 에서 `gray_raw` 로드 직후 만드는 `row = {...}` dict 에 한 줄 추가:

```python
        "modality": None, "modality_skip": None,
        "cells": {}, "overlay": None,
        "frame_hw": [int(gray_raw.shape[0]), int(gray_raw.shape[1])],   # Tier 1.1 displacement bin 용.
```

- [ ] **Step 2: run() 시작에서 ensemble 강제**

`run()` 본문 맨 앞(매처 모드 읽기 `matcher_mode = gle._matcher_for_eval().__name__` *이전*)에:

```python
    _apply_matcher_default()   # Tier 1.1: ensemble 기본(ALIGN_USE_ENSEMBLE=0 으로 끌 수 있음).
```

- [ ] **Step 3: summary 에 binned 리포트 싣고 출력**

`summary["align_offset_diag"] = _offset_diag_cond(offset_records)` 다음 줄에:

```python
    summary["binned"] = _binned_localization_report(all_rows)   # Tier 1.1 bin×arm 게이트.
```

`(out_dir / "summary.json").write_text(...)` *이후*, 기존 offset 진단 출력 부근에 표 출력 추가:

```python
    _print_binned_report(summary["binned"])
```

- [ ] **Step 4: `_print_binned_report` 구현**

Task 2 함수 아래에:

```python
def _print_binned_report(binned):
    """bin × arm(center/box) 표를 콘솔로 — office digest 가 그대로 베껴쓸 형식."""
    print("\n" + "=" * 64)
    print(f"[INFO] === Tier 1.1 box-crop 게이트 (frame={binned['frame']}) ===")
    if binned.get("n_no_frame_hw"):
        print(f"  [WARNING] frame_hw 없는 S 행 {binned['n_no_frame_hw']}개 → displacement 표서 제외.")
    for title, key, order in (
        ("[A] by structural displacement (GT-from-center)", "by_displacement",
         ("near", "mid", "far", "veryfar")),
        ("[B] by center-arm miss (rescue)", "by_center_miss",
         ("hit", "near", "far", "veryfar")),
    ):
        print(f"\n  {title}")
        print(f"    {'bin':<9} {'center gt_in_topk/rank1':<26} {'box gt_in_topk/rank1':<24} n(c/b)")
        for b in order:
            c, x = binned[key][b]["center"], binned[key][b]["box"]
            print(f"    {b:<9} {str(c['gt_in_topk'])+'/'+str(c['rank1']):<26} "
                  f"{str(x['gt_in_topk'])+'/'+str(x['rank1']):<24} {c['n']}/{x['n']}")
    print("=" * 64)
```

- [ ] **Step 5: import sanity + 전체 테스트 그린**

모듈이 깨지지 않고 전 테스트가 통과하는지 확인:

Run: `uv run python -c "import poc.workflow_2.golden_localization_eval_cond as glec; print('import OK', glec.FORCE_ENSEMBLE, glec.BIN_FRAME)"`
Expected: `import OK True inpaint`

Run: `uv run pytest poc/workflow_2/test_golden_localization_eval_cond.py -q`
Expected: PASS (기존 + 신규 전부 green).

- [ ] **Step 6: 커밋**

```bash
git add poc/workflow_2/golden_localization_eval_cond.py
git commit -m "feat(wf2): Tier 1.1 run() 배선 - frame_hw 캡처 + FORCE_ENSEMBLE + binned 표 출력"
```

---

## Handoff: office 실행 (Mac 아님)

Mac 작업 완료 후 push → office pull → 1회 실행:

```bash
uv run python poc/workflow_2/golden_localization_eval_cond.py
```

콘솔의 `Tier 1.1 box-crop 게이트` 표 + `summary.json["binned"]` 를 digest 로 회신
(spec §5 양식, `260608_133422` placeholder 채움). 판정(spec §6): far+veryfar 에서
box `gt_in_topk` 가 center +0.05 이상이면 production 포팅 green-light, 아니면 1.1 중단.

---

## Self-Review

- **Spec coverage:** §3.1 FORCE_ENSEMBLE→Task 3, §3.2 frame_hw→Task 4 Step1, §3.3 bin 리포트→Task 1·2, §3.4 임계→Task 1 상수, §4 테스트→Task 1~3, §5 digest 출력→Task 4 Step4, §6 결정→Handoff. 전부 매핑됨.
- **Placeholder scan:** 코드 블록 전부 실제 코드. digest 표의 빈칸은 office 산출물 양식(의도).
- **Type 일관성:** `_arm_rates` 반환 `{n, gt_in_topk, rank1}` 를 테스트·`_print_binned_report` 가 동일 키로 사용. `displacement_bin(gt_xy, frame_hw)`·`rescue_bin(center_dist_norm)` 시그니처가 호출부와 일치. cell 키 `center__inpaint`/`box__inpaint` = `BIN_FRAME` 합성과 일치.
