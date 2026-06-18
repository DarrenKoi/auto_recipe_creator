# Drop `align_img_from_msr` from the production loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the production loop (`align_fail_monitor`, `align_fail_monitor_only_check`) from downloading `align_img_from_msr`; keep an on-demand msr fetch for the offline benchmark.

**Architecture:** Add an `include_msr` flag to the `RcpMsrDownloader` contract and `gather_rcp_msr`, defaulting the production gather to **rcp-only**. The office reference downloader honors the flag (real office module edited at the office). Silence the now-expected missing-`current_sem` warning at runtime. Add a standalone script that fetches msr (`include_msr=True`) for bench expansion. The matching pipeline (consensus → rcp into the live crop) is unchanged.

**Tech Stack:** Python 3.10+, no external deps added. Self-test scripts (`[PASS]`/`[FAIL]` prints, run via `uv run python`) per repo convention — **not** pytest for `monitor/`.

## Global Constraints

- No `argparse` / CLI flags — config via env or hardcoded defaults; scripts run with just `uv run python <script>.py`. (CLAUDE.md)
- No `from __future__` imports. (CLAUDE.md)
- Print-based logging: `[INFO]` / `[WARNING]` / `[ERROR]` prefixes; no `logging` module in these files; no em-dash (U+2014) inside `print()` strings. (CLAUDE.md)
- Korean docstrings. (CLAUDE.md)
- Absolute imports `from poc.workflow_3.xxx import ...`. (CLAUDE.md)
- Git: commit directly to `main`; stage only this plan's files via pathspec (no `git add -A` / `commit -a`); verify scope with `git show --stat`. (memory: pathspec commits for concurrent edits)
- Commit message footer (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX
  ```

---

### Task 1: `include_msr` flag on the contract + gather (rcp-only production default)

**Files:**
- Modify: `poc/workflow_3/monitor/rcp_msr_gather.py` (Protocol `download_rcp_msr`, `gather_rcp_msr`, docstrings)
- Test: `poc/workflow_3/monitor/test_rcp_msr_gather.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `RcpMsrDownloader.download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr: bool = True) -> int`
  - `gather_rcp_msr(eqp_id, recipe_id, settings: Workflow3Settings, *, include_msr: bool = False) -> bool`
  - Production callers (`align_fail_monitor.py:352`, `align_fail_monitor_only_check.py:163`) keep calling `gather_rcp_msr(eqp_id, info["recipe_id"], settings)` with the default → rcp-only. **No edit at those call sites.**

- [ ] **Step 1: Update the two fakes in the test to accept and record `include_msr`**

In `poc/workflow_3/monitor/test_rcp_msr_gather.py`, change `_RecordingDownloader.download_rcp_msr` (currently line 27):

```python
    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True):
        self.calls.append({
            "eqp_id": eqp_id, "recipe_id": recipe_id,
            "dest_dir": dest_dir, "include_msr": include_msr,
        })
        return 4  # 받은 이미지 수(rcp 2 + msr 2 가정).
```

And the `_Boom` fake inside `test_swallows_downloader_exception` (currently line 85):

```python
    class _Boom:
        def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True):
            raise RuntimeError("FTP timeout")
```

- [ ] **Step 2: Add two new test functions**

Append to `poc/workflow_3/monitor/test_rcp_msr_gather.py` (before `def main()`):

```python
def test_production_requests_rcp_only():
    """기본(프로덕션) 호출은 include_msr=False 로 rcp 만 받는다."""
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    rcp_msr_gather.gather_rcp_msr("EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True))
    ok = len(fake.calls) == 1 and fake.calls[0]["include_msr"] is False
    print(f"[{'PASS' if ok else 'FAIL'}] production_requests_rcp_only: "
          f"include_msr={fake.calls[0]['include_msr'] if fake.calls else '-'}")
    return ok


def test_include_msr_propagates():
    """include_msr=True 를 주면 downloader 까지 그대로 전달된다(오프라인 벤치용)."""
    fake = _RecordingDownloader()
    rcp_msr_gather._DOWNLOADER = fake
    rcp_msr_gather.RCP_MSR_DOWNLOADER_AVAILABLE = True
    rcp_msr_gather.gather_rcp_msr(
        "EQP1", "CLS/RCP", _settings(rcp_msr_gather_enabled=True), include_msr=True
    )
    ok = len(fake.calls) == 1 and fake.calls[0]["include_msr"] is True
    print(f"[{'PASS' if ok else 'FAIL'}] include_msr_propagates: "
          f"include_msr={fake.calls[0]['include_msr'] if fake.calls else '-'}")
    return ok
```

Add both to the `results` list in `main()`:

```python
    results = [
        test_fires_when_enabled(),
        test_skips_when_disabled(),
        test_skips_without_recipe(),
        test_skips_without_downloader(),
        test_swallows_downloader_exception(),
        test_loader_canonical(),
        test_production_requests_rcp_only(),
        test_include_msr_propagates(),
    ]
```

- [ ] **Step 3: Run the test to verify the new cases FAIL**

Run: `uv run python poc/workflow_3/monitor/test_rcp_msr_gather.py`
Expected: `[FAIL] production_requests_rcp_only` and `[FAIL] include_msr_propagates` (gather does not yet pass/accept `include_msr`; `KeyError: 'include_msr'` is also acceptable as the failure signal). The other 6 still `[PASS]`.

- [ ] **Step 4: Implement `include_msr` in `gather_rcp_msr` and the Protocol**

In `poc/workflow_3/monitor/rcp_msr_gather.py`, update the Protocol method (currently line 35):

```python
    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr: bool = True) -> int:
        """eqp_id + recipe_id('<class>/<recipe>') 의 align_img_from_rcp 를 dest_dir 아래에
        office MES 와 동일한 레이아웃으로 쓰고, 쓴 이미지 총개수를 반환한다.

        include_msr=True 일 때만 align_img_from_msr(측정 궤적, S*/E* + 숨김폴더 cond)도 함께
        받는다. 프로덕션 루프는 msr 을 소비하지 않으므로 include_msr=False(rcp 만)로 부른다.
        오프라인 벤치(golden set 확장)에서만 include_msr=True 로 부른다.

        dest_dir 는 호출부가 넘기는 recipe leaf 경로(`ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe>`).
        받을 게 없으면 0 을 반환한다."""
        ...
```

Update `gather_rcp_msr` signature + the downloader call (currently lines 76, 91). Replace the function header through the download call:

```python
def gather_rcp_msr(
    eqp_id, recipe_id, settings: Workflow3Settings, *, include_msr: bool = False
) -> bool:
    """recipe 의 rcp 입력 이미지를 align_images 트리로 **동기** 다운로드한다.

    프로덕션 기본은 rcp 만(include_msr=False) — 보정/feasibility 는 라이브 캡처 프레임에
    consensus/rcp 템플릿을 매칭하므로 msr 을 소비하지 않는다. msr 은 오프라인 벤치에서만
    필요해 include_msr=True 로 명시할 때 받는다(fetch_msr_offline.py).

    rcp_msr_gather_enabled off / recipe_id 없음 / downloader 부재면 아무것도 안 하고 False.
    다운로드가 (예외 없이) 끝나면 True. 예외는 삼키고(best-effort) False 를 반환해
    모니터 루프가 죽지 않게 한다.

    cycle 직전에 호출해 assets 읽기 전 디스크 적재를 보장하는 게 핵심이다.
    """
    if not settings.rcp_msr_gather_enabled or not recipe_id or not RCP_MSR_DOWNLOADER_AVAILABLE:
        return False

    # recipe_id = '<class>/<recipe>' 라 ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe> 로 3단 중첩.
    dest_dir = ALIGN_IMAGES_DIR / eqp_id / recipe_id
    try:
        n_images = _DOWNLOADER.download_rcp_msr(
            eqp_id, recipe_id, dest_dir=dest_dir, include_msr=include_msr
        )
        kind = "rcp+msr" if include_msr else "rcp"
        print(f"[INFO] {kind} 다운로드 완료: EQP_ID={eqp_id} recipe={recipe_id} "
              f"images={n_images} dest={dest_dir}")
        return True
    except Exception as exc:
        print(f"[WARNING] rcp/msr 다운로드 예외: EQP_ID={eqp_id} recipe={recipe_id} error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="gather_error", level="warning",
            eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
        )
        return False
```

Also update the module docstring header (currently lines 3-11) to state rcp is the runtime input and msr is offline-bench-only:

```python
"""rcp(+선택적 msr) 입력 이미지 office 다운로드 접점 + 동기 fetch (monitor glue).

align_img_from_rcp(등록 align key)는 보정/점검의 런타임 입력이다. 보정/feasibility 는
라이브 캡처 프레임에 consensus(우선)/rcp(폴백) 템플릿을 매칭하며, align_img_from_msr
(측정 궤적)은 런타임에서 소비하지 않는다. 따라서 프로덕션 gather 는 rcp 만 받는다
(include_msr=False 기본). msr 은 오프라인 벤치(golden set 확장)에서만 fetch_msr_offline.py
로 include_msr=True 로 받는다.

기본 계약은 office MES 가 align_images 트리에 직접 적재하는 것이지만, MES 출력을 그
트리로 받지 못하는 환경에서는 office_rcp_msr_downloader 가 알람 시점에 그 트리로 내려받는다.

success_gather(consensus S 이미지)와 달리 **동기(blocking)** 다. rcp 는 cycle 이
assets(feasibility/보정)를 읽기 *전에* 반드시 디스크에 있어야 하므로, async 로 fire 하면
feasibility 가 빈 트리를 읽어 '보정 불가' 오판을 낼 수 있다. 따라서 cycle 직전에 받아
완료를 보장한다.

office 모듈 부재(개발 PC)·예외 시 조용히 skip 해 모니터 루프를 죽이지 않는다. 
office_rcp_msr_downloader 는 정위치(poc.workflow_3.monitor)에서 로드한다.
"""
```

- [ ] **Step 5: Run the full test, verify all PASS**

Run: `uv run python poc/workflow_3/monitor/test_rcp_msr_gather.py`
Expected: `[INFO] 8/8 cases passed`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/monitor/rcp_msr_gather.py poc/workflow_3/monitor/test_rcp_msr_gather.py
git commit -m "feat(workflow_3): rcp-only production gather (include_msr flag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX"
git show --stat HEAD | head -15
```

---

### Task 2: Office reference downloader honors `include_msr`

**Files:**
- Modify: `poc/workflow_3/monitor/temp_office_rcp_msr_downloader.py` (`download_rcp_msr` signature + skip msr when False; usage notes)

**Interfaces:**
- Consumes: `RcpMsrDownloader.download_rcp_msr(..., include_msr=...)` contract from Task 1.
- Produces: reference implementation that skips msr I/O when `include_msr=False`. The real gitignored `office_rcp_msr_downloader.py` is edited at the office to match.

> Note: this file is a `temp_`-prefixed template; it is **not** collected by pytest and cannot run `download_rcp_msr` on Mac (office fns absent → `RuntimeError`). Verification here is import-safety + visual review of the guard.

- [ ] **Step 1: Update `download_rcp_msr` to accept `include_msr` and skip msr when False**

In `poc/workflow_3/monitor/temp_office_rcp_msr_downloader.py`, replace the method (currently lines 60-91):

```python
    def download_rcp_msr(self, eqp_id, recipe_id, *, dest_dir, include_msr=True) -> int:
        """eqp_id + recipe_id('<class>/<recipe>') 의 rcp(+선택적 msr) 이미지를 받고 총개수를 반환.

        include_msr=True 일 때만 측정 궤적(msr)도 받는다. 프로덕션 루프는 msr 을 소비하지
        않으므로 rcp 만(include_msr=False) 받아 동기 다운로드 지연을 줄인다. msr 은 오프라인
        벤치(fetch_msr_offline.py)에서만 include_msr=True 로 받는다.

        Case 1(내부 경로 계산): download_align_images_from_rcp/msr 가 align_images 트리에
        직접 적재하므로 dest_dir 는 검증용으로만 쓴다(쓰는 곳==읽는 곳 점검).

        전제: 두 함수는 idempotent(이미 있으면 FTP skip, 기존 경로 반환) - 모듈 docstring
        의 [중요] 참고. send_cube_align_fail_info 도 같은 함수를 embed 용으로 부르므로,
        가드가 없으면 알람당 같은 이미지를 두 번 받는다.
        """
        if not _OFFICE_FNS_AVAILABLE:
            raise RuntimeError(
                "office_rich_notify 의 download_align_images_from_rcp/msr 를 찾을 수 없습니다 "
                "(개발 PC). 오피스 PC 에서만 동작합니다."
            )

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Case 1 - 함수가 (eqp_id, recipe_id) 로 내부에서 경로를 계산해 적재.
        rcp_imgs, rcp_conds = download_align_images_from_rcp(eqp_id, recipe_id)
        # msr 은 프로덕션에서 미사용 - include_msr=True(오프라인 벤치)일 때만 받는다.
        if include_msr:
            msr_imgs, msr_conds = self._download_msr_with_retry(eqp_id, recipe_id)
        else:
            msr_imgs, msr_conds = [], []

        # '쓰는 곳 != 읽는 곳' 조기 발견 - 이 다운로더가 막으려는 바로 그 버그 클래스.
        self._warn_if_outside(list(rcp_imgs) + list(msr_imgs), dest_dir)

        n_images = len(rcp_imgs) + len(msr_imgs)
        print(f"[INFO] rcp/msr 다운로드: rcp={len(rcp_imgs)}장 "
              f"msr={len(msr_imgs)}장 (cond rcp={len(rcp_conds)}/msr={len(msr_conds)}) -> {dest_dir}")
        return n_images
```

- [ ] **Step 2: Add an office-edit note to the module docstring**

In the same file, append to the module docstring (after the existing usage block, before the closing `"""` at line 30):

```
[프로덕션 = rcp 만] download_rcp_msr(include_msr=False) 가 기본이라 알람마다 rcp 만
받는다. msr 은 오프라인 벤치(fetch_msr_offline.py)가 include_msr=True 로 부를 때만 받는다.
오피스 PC 의 실제 office_rcp_msr_downloader.py 에도 동일하게 include_msr 가드를 적용해야
프로덕션에서 msr FTP I/O 가 실제로 빠진다(이 temp_ 견본만 고치면 git 추적용일 뿐 동작 안 함).
```

- [ ] **Step 3: Verify the template still imports on Mac**

Run: `uv run python -c "import poc.workflow_3.monitor.temp_office_rcp_msr_downloader as m; print('[PASS] import ok; include_msr default =', m.RcpMsrDownloader.download_rcp_msr.__defaults__)"`
Expected: prints `[PASS] import ok; include_msr default = (True,)` (no exception; office fns guarded by the existing `try/except ImportError`).

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/monitor/temp_office_rcp_msr_downloader.py
git commit -m "feat(workflow_3): office rcp/msr downloader honors include_msr (rcp-only prod)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX"
git show --stat HEAD | head -10
```

---

### Task 3: Silence the runtime `current_sem` missing-asset warning

**Files:**
- Modify: `poc/workflow_3/align/assets.py:215-223` (drop `current_sem` from the warning loop)
- Test: `poc/workflow_3/align/test_assets_no_msr.py` (create)

**Interfaces:**
- Consumes: `resolve_assets(recipe_dir) -> AlignFailAssets`, `RCP_OM_STEM`, `RCP_SEM_STEM`, `FROM_RCP_DIRNAME` (existing).
- Produces: `resolve_assets` no longer prints a `current_sem` `[WARNING]` when msr is absent; `recipe_om` / `recipe_sem` warnings unchanged.

- [ ] **Step 1: Write the failing self-test**

Create `poc/workflow_3/align/test_assets_no_msr.py`:

```python
"""resolve_assets 가 msr 부재 시 current_sem 경고를 내지 않는지 검증하는 self-test.

프로덕션은 더 이상 align_img_from_msr 을 받지 않으므로(런타임 미소비), msr 트리가 비어도
'current_sem 못 찾음' 경고는 노이즈다. recipe_om/recipe_sem 경고는 그대로여야 한다.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/align/test_assets_no_msr.py
"""

import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from poc.workflow_3.align import FROM_RCP_DIRNAME, RCP_OM_STEM, RCP_SEM_STEM
from poc.workflow_3.align.assets import resolve_assets


def _make_recipe_dir(base: Path, *, with_rcp: bool) -> Path:
    """<base>/EQP/CLS/RCP 레이아웃을 만들고 (옵션) rcp 더미 이미지 2장을 둔다."""
    recipe_dir = base / "EQP" / "CLS" / "RCP"
    if with_rcp:
        rcp = recipe_dir / FROM_RCP_DIRNAME
        rcp.mkdir(parents=True, exist_ok=True)
        (rcp / f"{RCP_OM_STEM}.jpg").write_bytes(b"x")
        (rcp / f"{RCP_SEM_STEM}.jpg").write_bytes(b"x")
    else:
        recipe_dir.mkdir(parents=True, exist_ok=True)
    return recipe_dir


def test_no_current_sem_warning_when_msr_absent():
    """rcp 만 있고 msr 이 없으면 current_sem 경고가 없어야 한다."""
    with tempfile.TemporaryDirectory() as tmp:
        recipe_dir = _make_recipe_dir(Path(tmp), with_rcp=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            assets = resolve_assets(recipe_dir)
        out = buf.getvalue()
    ok = (
        "current_sem" not in out.replace("current_sem =", "")  # 경고 라벨 부재
        and "[WARNING] current_sem" not in out
        and assets.current_sem is None
        and assets.recipe_om is not None
    )
    print(f"[{'PASS' if ok else 'FAIL'}] no_current_sem_warning_when_msr_absent: "
          f"current_sem_in_out={'[WARNING] current_sem' in out}")
    return ok


def test_recipe_warning_still_fires_when_rcp_absent():
    """rcp 도 없으면 recipe_om/recipe_sem 경고는 그대로 나와야 한다(회귀 가드)."""
    with tempfile.TemporaryDirectory() as tmp:
        recipe_dir = _make_recipe_dir(Path(tmp), with_rcp=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            resolve_assets(recipe_dir)
        out = buf.getvalue()
    ok = "[WARNING] recipe_om" in out and "[WARNING] recipe_sem" in out
    print(f"[{'PASS' if ok else 'FAIL'}] recipe_warning_still_fires_when_rcp_absent: "
          f"recipe_om_warn={'[WARNING] recipe_om' in out}")
    return ok


def main():
    print("[INFO] assets no-msr self-test 시작")
    results = [
        test_no_current_sem_warning_when_msr_absent(),
        test_recipe_warning_still_fires_when_rcp_absent(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the test to verify the first case FAILS**

Run: `uv run python poc/workflow_3/align/test_assets_no_msr.py`
Expected: `[FAIL] no_current_sem_warning_when_msr_absent` (current code prints `[WARNING] current_sem 이미지를 찾지 못했습니다`); `[PASS] recipe_warning_still_fires_when_rcp_absent`.

- [ ] **Step 3: Drop `current_sem` from the warning loop in `assets.py`**

In `poc/workflow_3/align/assets.py`, replace the warning loop (currently lines 215-223):

```python
    # current_sem(from_msr) 은 런타임 미소비라 부재해도 경고하지 않는다 - 오프라인
    # 진단에서만 쓰며, 거기서는 호출부가 None 여부를 직접 확인한다.
    for label, path in (
        ("recipe_om", assets.recipe_om),
        ("recipe_sem", assets.recipe_sem),
    ):
        if path is None:
            print(f"[WARNING] {label} 이미지를 찾지 못했습니다: {recipe_dir}")
        else:
            print(f"[INFO] {label} = {path.name}")
    if assets.current_sem is not None:
        print(f"[INFO] current_sem = {assets.current_sem.name}")
    return assets
```

- [ ] **Step 4: Run the test, verify both PASS**

Run: `uv run python poc/workflow_3/align/test_assets_no_msr.py`
Expected: `[INFO] 2/2 cases passed`, exit 0.

- [ ] **Step 5: Run an existing assets consumer smoke test to confirm no regression**

Run: `uv run python poc/workflow_3/align/diagnostics/compare_align_images.py`
Expected: runs to completion (falls back to synthetic self-test if no assets present); no traceback. The `current_sem`-absent path still handled by its own `is not None` checks (compare_align_images.py:259).

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/align/assets.py poc/workflow_3/align/test_assets_no_msr.py
git commit -m "fix(workflow_3): no current_sem warning when msr absent (runtime no longer reads msr)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX"
git show --stat HEAD | head -10
```

---

### Task 4: Standalone offline msr fetch script

**Files:**
- Create: `poc/workflow_3/monitor/fetch_msr_offline.py`

**Interfaces:**
- Consumes: `gather_rcp_msr(eqp_id, recipe_id, settings, *, include_msr=True)` (Task 1), `load_workflow3_settings()`, env `ALIGN_EQP_ID` / `ALIGN_RECIPE_NAME` (recipe_id = `<class>/<recipe>`).
- Produces: a `uv run`-able script that fetches rcp+msr for one configured recipe into the `align_images` tree for bench expansion.

> No automated test: this is a thin glue script whose only real action requires the office downloader. It is verified by the gate-path print on a dev PC (downloader absent → `[WARNING] ... 다운로드 안 됨`).

- [ ] **Step 1: Create the script**

Create `poc/workflow_3/monitor/fetch_msr_offline.py`:

```python
"""오프라인 벤치(golden set 확장)용 rcp+msr 동기 다운로드 스크립트.

프로덕션 루프는 align_img_from_msr 을 받지 않는다(런타임 미소비). 하지만 workflow_2/_3
오프라인 벤치(golden localization/consensus eval)는 측정 궤적(S*/E*)을 정답 근거로 쓰므로,
golden set 을 새 recipe 로 넓힐 때 이 스크립트로 그 recipe 의 rcp+msr 을 받아둔다.

설정은 env 로만 받는다(CLAUDE.md: argparse 금지):
  - ALIGN_EQP_ID       : 장비 ID (필수)
  - ALIGN_RECIPE_NAME  : '<class>/<recipe>' 형태의 recipe_id (필수)
office 다운로더(office_rcp_msr_downloader)가 있어야 실제로 받는다 - 개발 PC 에서는
게이트에서 막혀 [WARNING] 후 종료한다(루프와 동일 best-effort 철학).

    uv run python poc/workflow_3/monitor/fetch_msr_offline.py
"""

import os

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor.rcp_msr_gather import gather_rcp_msr


def main():
    eqp_id = os.environ.get("ALIGN_EQP_ID", "").strip()
    recipe_id = os.environ.get("ALIGN_RECIPE_NAME", "").strip()
    if not eqp_id or not recipe_id:
        print("[ERROR] ALIGN_EQP_ID 와 ALIGN_RECIPE_NAME('<class>/<recipe>') 를 env 로 지정하세요.")
        return 1

    settings = load_workflow3_settings()
    print(f"[INFO] 오프라인 msr fetch: EQP_ID={eqp_id} recipe={recipe_id} (include_msr=True)")
    ok = gather_rcp_msr(eqp_id, recipe_id, settings, include_msr=True)
    if ok:
        print("[INFO] rcp+msr 다운로드 완료.")
        return 0
    print("[WARNING] 다운로드 안 됨 (downloader 부재/게이트 off/예외). 콘솔 로그 확인.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the env-missing guard on a dev PC**

Run: `uv run python poc/workflow_3/monitor/fetch_msr_offline.py`
Expected: `[ERROR] ALIGN_EQP_ID 와 ALIGN_RECIPE_NAME ...`, exit 1 (no env set).

- [ ] **Step 3: Verify the downloader-absent path returns cleanly**

Run: `ALIGN_EQP_ID=EQP1 ALIGN_RECIPE_NAME=CLS/RCP uv run python poc/workflow_3/monitor/fetch_msr_offline.py`
Expected: prints the `[INFO] 오프라인 msr fetch...` line then `[WARNING] 다운로드 안 됨 ...`, exit 1 (dev PC has no office downloader; `gather_rcp_msr` returns False at the gate).

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/monitor/fetch_msr_offline.py
git commit -m "feat(workflow_3): fetch_msr_offline.py — on-demand rcp+msr for offline bench

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX"
git show --stat HEAD | head -10
```

---

### Task 5: Documentation updates

**Files:**
- Modify: `poc/workflow_3/config.py:92-97` (rcp/msr gather comment)
- Modify: `poc/workflow_3/__init__.py:59-65` (layout comment)
- Modify: `CLAUDE.md` (filesystem contract note)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the `config.py` gather comment**

In `poc/workflow_3/config.py`, replace lines 92-96 (the comment block above `rcp_msr_gather_enabled`):

```python
    # --- rcp 입력 이미지 office 다운로드 ---
    # align_img_from_rcp(등록 align key)는 보정/점검의 런타임 입력이다. 보정/feasibility 는
    # 라이브 캡처 프레임에 consensus(우선)/rcp(폴백) 템플릿을 매칭하며, align_img_from_msr
    # (측정 궤적)은 런타임에서 소비하지 않으므로 프로덕션 gather 는 rcp 만 받는다(rcp-only).
    # msr 은 오프라인 벤치에서만 fetch_msr_offline.py 로 받는다. 기본 계약은 office MES 가
    # align_images 트리에 직접 적재하는 것이지만, 못 받는 환경에선 office_rcp_msr_downloader
    # 가 알람 시점에 동기로 내려받는다(cycle 이 assets 읽기 전 디스크 적재 보장).
```

- [ ] **Step 2: Update the `__init__.py` layout comment**

In `poc/workflow_3/__init__.py`, replace lines 59-61:

```python
# 오피스 MES 가 align fail 시 생성하는 이미지 루트. align fail 핸들러는 여기에
# captured_img_from_rcs(녹화 포함)를 함께 적재하고, align.assets 가 읽는다.
#   align_images/<eqp_id>/<class>/<recipe>/{align_img_from_rcp, align_img_from_msr, captured_img_from_rcs}
# 런타임은 align_img_from_rcp(+consensus)만 소비한다. align_img_from_msr 은 오프라인
# 벤치 전용이며 프로덕션 루프는 받지 않는다(fetch_msr_offline.py 로 필요 시 수동 적재).
```

- [ ] **Step 3: Update `CLAUDE.md` filesystem contract note**

In `CLAUDE.md`, find the filesystem-contract block listing `align_img_from_msr/` (the `measurement trajectory (E = fail) (office MES)` line) and append a note immediately after that code block:

```markdown
- **Runtime no longer consumes `align_img_from_msr`** (2026-06-18): correction/feasibility match consensus(preferred)/rcp(fallback) templates into the live capture, so the production loop (`align_fail_monitor`, `align_fail_monitor_only_check`) downloads **rcp only** (`gather_rcp_msr(..., include_msr=False)`). msr is offline-bench-only — fetch it on demand with `poc/workflow_3/monitor/fetch_msr_offline.py` (`include_msr=True`).
```

- [ ] **Step 4: Sanity-check imports still load after comment edits**

Run: `uv run python -c "import poc.workflow_3, poc.workflow_3.config; print('[PASS] imports ok')"`
Expected: `[PASS] imports ok` (comment-only edits, no behavior change).

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/__init__.py CLAUDE.md
git commit -m "docs(workflow_3): msr is offline-bench-only; production gather is rcp-only

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H2kzF2iWiSCK59c68ELHTX"
git show --stat HEAD | head -10
```

---

## Final verification

- [ ] **Run the two new/changed self-tests together**

Run:
```bash
uv run python poc/workflow_3/monitor/test_rcp_msr_gather.py
uv run python poc/workflow_3/align/test_assets_no_msr.py
```
Expected: `8/8` and `2/2` cases passed.

- [ ] **Confirm production call sites are untouched and still default to rcp-only**

Run: `grep -n "gather_rcp_msr(eqp_id" poc/workflow_3/monitor/align_fail_monitor.py poc/workflow_3/monitor/align_fail_monitor_only_check.py`
Expected: both call `gather_rcp_msr(eqp_id, info["recipe_id"], settings)` with no `include_msr` arg (→ rcp-only).

---

## Self-Review

**Spec coverage:**
- Spec §Changes 1 (contract + gather, rcp-only default) → Task 1. ✓
- Spec §Changes 2 (office reference honors include_msr + office-edit note) → Task 2. ✓
- Spec §Changes 3 (assets.py silence current_sem warning) → Task 3. ✓
- Spec §Changes 4 (offline-bench fetch script) → Task 4. ✓
- Spec §Changes 5 (tests + docs) → Task 1/Task 3 tests + Task 5 docs. ✓
- Spec §Out of scope (matching pipeline, consensus, live-crop) → not modified. ✓
- Spec §Risk (S-sparse → rcp fallback; office lag) → preserved; no code asserts otherwise. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N"; all code blocks are complete. ✓

**Type consistency:** `download_rcp_msr(..., *, dest_dir, include_msr: bool = True)` and `gather_rcp_msr(..., *, include_msr: bool = False)` are used identically across Tasks 1, 2, 4 and both fakes in the test. `resolve_assets`, `RCP_OM_STEM`, `RCP_SEM_STEM`, `FROM_RCP_DIRNAME` match `assets.py` / `align/__init__.py`. ✓
