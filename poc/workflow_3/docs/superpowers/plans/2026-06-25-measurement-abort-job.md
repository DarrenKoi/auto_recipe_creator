# Add a measurement-abort job to the production loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second MES-alarm job to the production loop — abort a measurement run when MES fires a consecutive-fail threshold alarm — sharing the existing single while loop and serialized RCS GUI. Ships notify-only behind a double actuation gate; the align-fail path is untouched.

**Architecture:** One process, one `monitor_loop`. The same `source.poll()` feeds a new optional `filter_measurement_fail`; matching rows dispatch through `process_abort_rows` → `run_abort_cycle`, a sibling of `run_alarm_cycle` that reuses RCS-ready/connect/window-wait/teardown and adds one new step: locate (VLM) + click (gated) a Stop/Abort button and confirm. Detection is stateless (MES owns the streak). See `../specs/2026-06-25-measurement-abort-job-design.md`.

**Tech Stack:** Python 3.10+, no new external deps. Self-test scripts (`[PASS]`/`[FAIL]` prints, run via `uv run python`) per repo convention — **not** pytest for `monitor/`.

## Global Constraints

- No `argparse` / CLI flags — config via env or hardcoded defaults; scripts run with just `uv run python <script>.py`. (CLAUDE.md)
- No `from __future__` imports. (CLAUDE.md)
- Print-based logging: `[INFO]` / `[WARNING]` / `[ERROR]` prefixes; no `logging` module in these files; no em-dash (U+2014) inside `print()` strings (office console is cp949). (CLAUDE.md)
- Korean docstrings. (CLAUDE.md)
- Absolute imports `from poc.workflow_3.xxx import ...`. (CLAUDE.md)
- VLM client takes a **route_slug** (`"ui-venus"`), not a model name (`"ui-venus-1.5-8b"`). (memory)
- Actuation double-gate: a real Abort click requires `SAFE_MODE=0` **and** `abort_action_dry_run=0`, mirroring `correction_dry_run`. (spec §Staged enablement)
- Git: commit directly to `main`; stage only this plan's files via pathspec (no `git add -A` / `commit -a`); verify scope with `git show --stat`. (memory: pathspec commits for concurrent edits)
- Commit message footer (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R
  ```

---

### Task 1: Config — `MEAS_FAIL_*` settings + abort double-gate

**Files:**
- Modify: `poc/workflow_3/config.py` (`Workflow3Settings` fields + `load_workflow3_settings` env overrides)
- Test: `poc/workflow_3/test_config_meas_fail.py` (create)

**Interfaces:**
- Produces (new `Workflow3Settings` fields):
  - `meas_fail_abort_enabled: bool = True`
  - `meas_fail_alid: str = ""`
  - `abort_action_dry_run: bool = True`
  - `abort_button_vlm_service: str = "ui-venus"`
- Consumes: existing `env_flag` / `env_int` / `env_float` / `_env_str` helpers, `base.safe_mode`.

- [ ] **Step 1: Write the failing self-test**

Create `poc/workflow_3/test_config_meas_fail.py`:

```python
"""measurement-abort 설정 로딩 self-test.

새 MEAS_FAIL_* env 가 Workflow3Settings 에 반영되는지, abort 클릭 이중 게이트가
correction_dry_run 과 동일하게 SAFE_MODE 에 의해 강제 dry-run 되는지 검증한다.

    uv run python poc/workflow_3/test_config_meas_fail.py
"""

import os
from contextlib import contextmanager

from poc.workflow_3.config import load_workflow3_settings


@contextmanager
def _env(**kv):
    """env 임시 설정 후 원복."""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults():
    """기본값: 잡 on, ALID 미설정(빈 문자열), 클릭은 dry-run."""
    with _env(SAFE_MODE="1", MEAS_FAIL_ABORT_ENABLED=None, MEAS_FAIL_ALID=None,
              MEAS_FAIL_ABORT_DRY_RUN=None, MEAS_FAIL_ABORT_BUTTON_SERVICE=None):
        s = load_workflow3_settings()
    ok = (s.meas_fail_abort_enabled is True and s.meas_fail_alid == ""
          and s.abort_action_dry_run is True
          and s.abort_button_vlm_service == "ui-venus")
    print(f"[{'PASS' if ok else 'FAIL'}] defaults: enabled={s.meas_fail_abort_enabled} "
          f"alid={s.meas_fail_alid!r} dry_run={s.abort_action_dry_run}")
    return ok


def test_safe_mode_forces_dry_run():
    """SAFE_MODE=1 이면 MEAS_FAIL_ABORT_DRY_RUN=0 을 줘도 dry-run 강제."""
    with _env(SAFE_MODE="1", MEAS_FAIL_ABORT_DRY_RUN="0"):
        s = load_workflow3_settings()
    ok = s.abort_action_dry_run is True
    print(f"[{'PASS' if ok else 'FAIL'}] safe_mode_forces_dry_run: dry_run={s.abort_action_dry_run}")
    return ok


def test_armed_when_safe_off_and_flag_zero():
    """SAFE_MODE=0 + MEAS_FAIL_ABORT_DRY_RUN=0 일 때만 클릭 무장(dry_run=False)."""
    with _env(SAFE_MODE="0", MEAS_FAIL_ABORT_DRY_RUN="0"):
        s = load_workflow3_settings()
    ok = s.abort_action_dry_run is False
    print(f"[{'PASS' if ok else 'FAIL'}] armed_when_safe_off_and_flag_zero: dry_run={s.abort_action_dry_run}")
    return ok


def test_env_overrides():
    """MEAS_FAIL_ALID / ENABLED / SERVICE env 반영."""
    with _env(MEAS_FAIL_ABORT_ENABLED="0", MEAS_FAIL_ALID="9012",
              MEAS_FAIL_ABORT_BUTTON_SERVICE="mai-ui"):
        s = load_workflow3_settings()
    ok = (s.meas_fail_abort_enabled is False and s.meas_fail_alid == "9012"
          and s.abort_button_vlm_service == "mai-ui")
    print(f"[{'PASS' if ok else 'FAIL'}] env_overrides: enabled={s.meas_fail_abort_enabled} "
          f"alid={s.meas_fail_alid!r} service={s.abort_button_vlm_service}")
    return ok


def main():
    print("[INFO] meas_fail config self-test 시작")
    results = [
        test_defaults(),
        test_safe_mode_forces_dry_run(),
        test_armed_when_safe_off_and_flag_zero(),
        test_env_overrides(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `uv run python poc/workflow_3/test_config_meas_fail.py`
Expected: `AttributeError` / `TypeError` (fields don't exist yet) — that is the red signal. All four cases fail to construct.

- [ ] **Step 3: Add the four fields to `Workflow3Settings`**

In `poc/workflow_3/config.py`, after the `cond_box_crop` field (line 173), add:

```python

    # --- 측정 실패 abort 잡 (MES 임계 알람 기반) ---
    # MES 가 'N회 연속 측정 실패' 임계 알람을 쏘면, 그 tool 에 접속해 측정을 abort 한다
    # (align fail 과 같은 단일 루프/단일 GUI 를 공유, 직렬 처리). 검출은 무상태(스트릭은
    # MES 가 셈). abort 클릭은 보정과 동일한 이중 게이트(SAFE_MODE off + dry-run 0)이며
    # 기본은 notify-only(검출+증거 캡처+엔지니어 cube, 클릭은 dry-run).
    meas_fail_abort_enabled: bool = True       # 잡 마스터 토글(검출+알림).
    meas_fail_alid: str = ""                   # 임계 알람 ALID(오피스 확인 필요, 빈값=replay 비활성).
    abort_action_dry_run: bool = True          # 클릭 게이트. 실제 abort 는 SAFE_MODE off + 0 일 때만.
    abort_button_vlm_service: str = "ui-venus" # Abort 버튼 locator route_slug(모델명 아님).
```

- [ ] **Step 4: Add env overrides to `load_workflow3_settings`**

In `poc/workflow_3/config.py`, alongside the existing `correction_dry_run` derivation (after line 182), add the abort gate derivation:

```python
    # abort 클릭 이중 게이트: SAFE_MODE 켜지면 env 무관 dry-run(보정과 동일 패턴).
    abort_dry_run_requested = env_flag("MEAS_FAIL_ABORT_DRY_RUN", default=True)
    abort_action_dry_run = abort_dry_run_requested or base.safe_mode
```

Then add to the `Workflow3Settings(...)` constructor call (after `cond_box_crop=...`, line 247):

```python
        meas_fail_abort_enabled=env_flag("MEAS_FAIL_ABORT_ENABLED", default=True),
        meas_fail_alid=_env_str("MEAS_FAIL_ALID", ""),
        abort_action_dry_run=abort_action_dry_run,
        abort_button_vlm_service=_env_str("MEAS_FAIL_ABORT_BUTTON_SERVICE", "ui-venus"),
```

- [ ] **Step 5: Run the test, verify all PASS**

Run: `uv run python poc/workflow_3/test_config_meas_fail.py`
Expected: `[INFO] 4/4 cases passed`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/test_config_meas_fail.py
git commit -m "feat(workflow_3): MEAS_FAIL_* abort settings + double gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 2: Detector — `AlarmSource.filter_measurement_fail` (optional, graceful)

**Files:**
- Modify: `poc/workflow_3/monitor/alarm_source.py`
- Test: `poc/workflow_3/monitor/test_alarm_source_meas.py` (create)

**Interfaces:**
- Produces:
  - `AlarmSource.filter_measurement_fail(rows)` — returns measurement-fail rows, or an empty `DataFrame` when no measurement filter is wired (job self-disables).
  - `AlarmSource(kind, poll_fn, filter_fn, *, meas_filter_fn=None, available=True)` — new optional kwarg.
  - `_replay_filter_measurement_fail(rows, alid)` — replay fixture filter keyed on `MEAS_FAIL_ALID`.
- Consumes: office `filter_measurement_fail` declared as an **optional** attr in `load_office_integration`.

- [ ] **Step 1: Write the failing self-test**

Create `poc/workflow_3/monitor/test_alarm_source_meas.py`:

```python
"""AlarmSource 의 measurement-fail 필터 self-test.

- meas_filter 가 없으면 filter_measurement_fail 은 빈 결과(잡 자가 비활성).
- meas_filter 가 있으면 ALID 로 분리해 align(9006) 과 섞이지 않는다.

    uv run python poc/workflow_3/monitor/test_alarm_source_meas.py
"""

import pandas as pd

from poc.workflow_3.monitor.alarm_source import (
    AlarmSource,
    _replay_filter_align_fail,
    _replay_filter_measurement_fail,
)


def _rows():
    return pd.DataFrame(
        [
            {"EQP_ID": "EQP1", "ALID": "9006"},   # align fail
            {"EQP_ID": "EQP2", "ALID": "9012"},   # measurement fail (예시 ALID)
            {"EQP_ID": "EQP3", "ALID": "9999"},   # 무관
        ]
    )


def test_no_meas_filter_returns_empty():
    """meas_filter 미주입이면 filter_measurement_fail 은 빈 결과."""
    src = AlarmSource("x", _rows, _replay_filter_align_fail)
    out = src.filter_measurement_fail(_rows())
    ok = out is not None and len(out) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] no_meas_filter_returns_empty: n={0 if out is None else len(out)}")
    return ok


def test_meas_filter_splits_by_alid():
    """meas_filter 주입 시 ALID 로 align/meas 가 분리된다."""
    src = AlarmSource(
        "x", _rows, _replay_filter_align_fail,
        meas_filter_fn=lambda r: _replay_filter_measurement_fail(r, "9012"),
    )
    align = src.filter_align_fail(_rows())
    meas = src.filter_measurement_fail(_rows())
    ok = (len(align) == 1 and align.iloc[0]["EQP_ID"] == "EQP1"
          and len(meas) == 1 and meas.iloc[0]["EQP_ID"] == "EQP2")
    print(f"[{'PASS' if ok else 'FAIL'}] meas_filter_splits_by_alid: "
          f"align={len(align)} meas={len(meas)}")
    return ok


def test_replay_meas_empty_alid_passes_nothing():
    """ALID 빈 문자열이면(미설정) measurement 매칭 0건(replay 비활성)."""
    out = _replay_filter_measurement_fail(_rows(), "")
    ok = out is not None and len(out) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] replay_meas_empty_alid_passes_nothing: n={len(out)}")
    return ok


def main():
    print("[INFO] alarm_source meas self-test 시작")
    results = [
        test_no_meas_filter_returns_empty(),
        test_meas_filter_splits_by_alid(),
        test_replay_meas_empty_alid_passes_nothing(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `uv run python poc/workflow_3/monitor/test_alarm_source_meas.py`
Expected: `ImportError` (`_replay_filter_measurement_fail` / `meas_filter_fn` don't exist).

- [ ] **Step 3: Implement in `alarm_source.py`**

Add `meas_filter_fn` to `AlarmSource.__init__` and the new method. Replace the class body (lines 21-40):

```python
class AlarmSource:
    """poll() 로 알람 DataFrame 을 돌려주는 공급자."""

    def __init__(self, kind: str, poll_fn, filter_fn, *, meas_filter_fn=None, available: bool = True):
        self.kind = kind
        self.available = available
        self._poll_fn = poll_fn
        self._filter_fn = filter_fn
        self._meas_filter_fn = meas_filter_fn

    def poll(self):
        """전체 CD-SEM 알람 rows 를 반환한다 (DataFrame 또는 None)."""
        if not self.available:
            return None
        return self._poll_fn()

    def filter_align_fail(self, rows):
        """알람 rows 에서 align fail(ALID=9006) 만 남긴다."""
        if rows is None:
            return None
        return self._filter_fn(rows)

    def filter_measurement_fail(self, rows):
        """알람 rows 에서 측정 실패 임계 알람만 남긴다.

        meas_filter 가 안 걸려 있으면(오피스 모듈 미제공/ALID 미설정) 빈 DataFrame 을
        돌려줘 abort 잡을 자가 비활성한다 — align 경로에는 영향 없다.
        """
        import pandas as pd

        if self._meas_filter_fn is None or rows is None:
            return pd.DataFrame()
        return self._meas_filter_fn(rows)
```

Add the replay filter after `_replay_filter_align_fail` (line 58):

```python
def _replay_filter_measurement_fail(rows: "pd.DataFrame", alid: str) -> "pd.DataFrame":
    """replay rows 에서 측정 실패 임계 ALID 만 남긴다.

    alid 가 빈 문자열(미설정)이거나 ALID 컬럼이 없으면 0건(잡 비활성).
    """
    alid = (alid or "").strip()
    if not alid or rows is None or rows.empty or "ALID" not in rows.columns:
        return rows.iloc[0:0] if rows is not None and hasattr(rows, "iloc") else pd.DataFrame()
    mask = rows["ALID"].astype(str).str.strip() == alid
    return rows[mask].reset_index(drop=True)
```

In `_load_office_module`, declare the measurement filter as **optional** (does not gate availability). Replace the function (lines 43-50):

```python
def _load_office_module():
    """office_align_fail_alarm 을 정위치에서 찾는다. 없으면 None.

    filter_measurement_fail 은 선택 attr 이라 required 에 넣지 않는다 — 없으면
    abort 잡만 비활성되고 align 경로는 그대로 동작한다.
    """
    integration = load_office_integration(
        "office_align_fail_alarm",
        "poc.workflow_3.monitor.office_align_fail_alarm",
        required_attrs=("get_cdsem_alarms", "filter_align_fail"),
    )
    return integration.module if integration.available else None
```

Wire both filters into `load_alarm_source`. Replace the body (lines 82-100):

```python
def load_alarm_source(kind: str = "office") -> AlarmSource:
    """설정된 종류의 AlarmSource 를 만든다. office 모듈이 없으면 replay→비활성 폴백.

    measurement-fail 필터는 선택이다 — office 모듈에 filter_measurement_fail 가 있을
    때만(또는 replay 의 MEAS_FAIL_ALID 가 설정됐을 때만) abort 잡이 활성된다.
    """
    meas_alid = os.environ.get("MEAS_FAIL_ALID", "").strip()
    if kind == "replay":
        csv_path = os.environ.get("ALIGN_FAIL_REPLAY_CSV", "").strip()
        if csv_path and os.path.isfile(csv_path):
            replay = _ReplaySource(csv_path)
            meas_fn = (lambda r: _replay_filter_measurement_fail(r, meas_alid)) if meas_alid else None
            return AlarmSource(
                "replay", replay.poll, _replay_filter_align_fail, meas_filter_fn=meas_fn
            )
        print(f"[WARNING] ALIGN_FAIL_REPLAY_CSV 가 없거나 파일이 아님: {csv_path!r} - 알람 비활성")
        return AlarmSource("disabled", lambda: None, lambda rows: rows, available=False)

    module = _load_office_module()
    if module is not None:
        meas_fn = getattr(module, "filter_measurement_fail", None)
        if meas_fn is None:
            print("[INFO] office 모듈에 filter_measurement_fail 없음 - 측정 abort 잡 비활성(align 만 동작).")
        return AlarmSource(
            "office", module.get_cdsem_alarms, module.filter_align_fail, meas_filter_fn=meas_fn
        )

    print(
        "[WARNING] office_align_fail_alarm 모듈을 찾지 못함 - 알람 폴링 비활성. "
        "개발 PC 에서는 ALIGN_FAIL_ALARM_SOURCE=replay + ALIGN_FAIL_REPLAY_CSV 를 쓰세요."
    )
    return AlarmSource("disabled", lambda: None, lambda rows: rows, available=False)
```

Update `__all__` (line 103):

```python
__all__ = ["AlarmSource", "load_alarm_source"]
```

(no change needed — internal helpers are imported by the test by name.)

- [ ] **Step 4: Run the test, verify all PASS**

Run: `uv run python poc/workflow_3/monitor/test_alarm_source_meas.py`
Expected: `[INFO] 3/3 cases passed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/alarm_source.py poc/workflow_3/monitor/test_alarm_source_meas.py
git commit -m "feat(workflow_3): optional measurement-fail alarm filter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 3: Abort-button locator — `monitor/abort_button.py`

**Files:**
- Create: `poc/workflow_3/monitor/abort_button.py`
- Test: `poc/workflow_3/monitor/test_abort_button.py` (create — parse-only, no live VLM)

**Interfaces:**
- Produces:
  - `locate_abort_button(*, frame_bgr, client) -> tuple[int, int] | None`
  - `locate_abort_confirm(*, frame_bgr, client) -> tuple[int, int] | None`
- Consumes: `encode_image_webp`, `bbox_to_pixels`, `bbox_center`, `extract_json`, `Workflow1VLMClient` (identical to `align/ok_button.py`).

> The VLM call cannot run on a dev PC without Flask. The test injects a fake client returning a canned JSON string, exercising the parse/scale path only — the same seam `ok_button.py` would use if it had a test.

- [ ] **Step 1: Create the locator (mirror `align/ok_button.py`)**

Create `poc/workflow_3/monitor/abort_button.py`:

```python
"""측정 abort(Stop/중지) 버튼을 VLM 으로 찾는 locator.

measurement-abort 잡(`monitor/cycle.run_abort_cycle`)의 마지막 단계에서 쓴다: MES 가
'N회 연속 측정 실패' 임계 알람을 쏜 tool 에 접속한 뒤, 진행 중인 측정을 멈추는 Stop/Abort
버튼을 눌러야 한다. 버튼은 RCS tool 창 위의 컨트롤이라 좌표가 **screen 절대 픽셀**이다.

설계 경계: VLM 은 버튼 *영역*만 식별한다. abort 여부는 MES 가 이미 결정했다(임계 알람).
여기서는 어떤 UI 버튼을 누를지 위치만 찾는다. align/ok_button.py 와 동일한 헬퍼/패턴을 쓴다.
"""

import cv2
import numpy as np
from PIL import Image

from poc.workflow_3.util.image_utils import encode_image_webp
from poc.workflow_3.util.json_utils import bbox_center, bbox_to_pixels, extract_json
from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient


def _abort_button_system_prompt() -> str:
    """Abort/Stop 버튼 탐지 시스템 프롬프트."""
    return (
        "You analyse a screenshot of a CD-SEM / VeritySEM metrology tool that is RUNNING "
        "a measurement recipe. The operator needs to STOP / ABORT the in-progress "
        "measurement run because too many points have failed.\n"
        "Locate the button that STOPS or ABORTS the running measurement. It is a "
        "clickable control, usually labelled 'Stop', 'Abort', '중지', or '정지'. Do NOT "
        "return a 'Pause', 'Cancel' on an unrelated dialog, the window close (X) button, "
        "menu items, or the SEM image itself.\n"
        "Return strict JSON only. If no such Stop/Abort button is clearly visible, say so "
        "rather than guessing."
    )


def _abort_button_user_prompt() -> str:
    """Abort 버튼 탐지 사용자 프롬프트(엄격한 JSON 스키마)."""
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "abort_button_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "abort_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "abort_button_bbox must tightly enclose the Stop/Abort button only. "
        "If none is clearly visible, set abort_button_visible=false, abort_button_bbox=null."
    )


def _frame_to_webp_b64(frame_bgr: np.ndarray) -> tuple[str, int, int]:
    """grayscale/BGR/BGRA numpy 프레임을 WebP base64 로 인코딩. 반환 (b64, w, h)."""
    if frame_bgr.ndim == 2:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        rgb = cv2.cvtColor(frame_bgr[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"지원하지 않는 프레임 shape: {frame_bgr.shape}")
    image = Image.fromarray(rgb)
    return encode_image_webp(image, quality=90)


def _locate(frame_bgr, client, system_prompt, user_prompt, visible_key, bbox_key):
    """공통 locate: 프레임→VLM→bbox→screen 중심 좌표. 없으면 None."""
    image_b64, w, h = _frame_to_webp_b64(frame_bgr)
    response = client.chat_with_image_b64(
        image_b64=image_b64,
        system_message=system_prompt,
        user_text=user_prompt,
        image_mime="image/webp",
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if parsed.get(visible_key) is not True:
        return None
    bbox_px = bbox_to_pixels(parsed.get(bbox_key), w, h, parsed.get("coord_system"))
    if bbox_px is None:
        return None
    center = bbox_center(bbox_px)
    return int(center["x"]), int(center["y"])


def locate_abort_button(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """전체 화면 프레임에서 Stop/Abort 버튼 중심의 SCREEN 픽셀 좌표를 반환(없으면 None)."""
    return _locate(
        frame_bgr, client, _abort_button_system_prompt(), _abort_button_user_prompt(),
        "abort_button_visible", "abort_button_bbox",
    )


def _confirm_system_prompt() -> str:
    return (
        "A confirmation dialog has appeared asking the operator to confirm STOPPING / "
        "ABORTING the measurement run (e.g. 'Abort this run?', '측정을 중지하시겠습니까?').\n"
        "Locate the button that CONFIRMS the abort - usually 'Yes', 'OK', or '확인'. Do NOT "
        "return 'No', 'Cancel', or '취소'.\n"
        "Return strict JSON only. If no such confirm button is clearly visible, say so."
    )


def _confirm_user_prompt() -> str:
    return (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "abort_button_visible": true,\n'
        '  "coord_system": "relative_1000",\n'
        '  "abort_button_bbox": {"left": 0, "top": 0, "right": 0, "bottom": 0},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "abort_button_bbox must tightly enclose the Yes/OK/확인 confirm button only. "
        "If none is clearly visible, set abort_button_visible=false, abort_button_bbox=null."
    )


def locate_abort_confirm(*, frame_bgr: np.ndarray, client: Workflow1VLMClient) -> tuple[int, int] | None:
    """abort 확인 다이얼로그의 Yes/확인 버튼 중심 SCREEN 좌표를 반환(없으면 None)."""
    return _locate(
        frame_bgr, client, _confirm_system_prompt(), _confirm_user_prompt(),
        "abort_button_visible", "abort_button_bbox",
    )
```

- [ ] **Step 2: Create the parse-only self-test**

Create `poc/workflow_3/monitor/test_abort_button.py`:

```python
"""abort_button locator 파싱/스케일 self-test (라이브 VLM 불필요).

가짜 client 가 캔드 JSON 을 돌려주게 해 bbox→screen 좌표 변환 경로만 검증한다.

    uv run python poc/workflow_3/monitor/test_abort_button.py
"""

import numpy as np

from poc.workflow_3.monitor.abort_button import locate_abort_button


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """chat_with_image_b64 에 캔드 응답을 돌려주는 가짜 VLM client."""

    def __init__(self, text):
        self._text = text

    def chat_with_image_b64(self, **kwargs):
        return _FakeResp(self._text)


def test_visible_center_relative_1000():
    """relative_1000 bbox 중심이 프레임 픽셀로 환산된다(1000x500 프레임 가정)."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient(
        '{"abort_button_visible": true, "coord_system": "relative_1000", '
        '"abort_button_bbox": {"left": 400, "top": 800, "right": 600, "bottom": 900}, '
        '"confidence": 0.9}'
    )
    xy = locate_abort_button(frame_bgr=frame, client=client)
    # center rel (500, 850) -> px (500, 425)
    ok = xy is not None and abs(xy[0] - 500) <= 2 and abs(xy[1] - 425) <= 2
    print(f"[{'PASS' if ok else 'FAIL'}] visible_center_relative_1000: xy={xy}")
    return ok


def test_not_visible_returns_none():
    """abort_button_visible=false 면 None."""
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    client = _FakeClient('{"abort_button_visible": false, "abort_button_bbox": null}')
    xy = locate_abort_button(frame_bgr=frame, client=client)
    ok = xy is None
    print(f"[{'PASS' if ok else 'FAIL'}] not_visible_returns_none: xy={xy}")
    return ok


def main():
    print("[INFO] abort_button self-test 시작")
    results = [
        test_visible_center_relative_1000(),
        test_not_visible_returns_none(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the test, verify all PASS**

Run: `uv run python poc/workflow_3/monitor/test_abort_button.py`
Expected: `[INFO] 2/2 cases passed`, exit 0. (If `bbox_to_pixels`' coord-system heuristic shifts the center by >2px, widen the tolerance to match `json_utils` behavior — verify against `align/ok_button.py`'s contract, do not change the util.)

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/monitor/abort_button.py poc/workflow_3/monitor/test_abort_button.py
git commit -m "feat(workflow_3): VLM Stop/Abort button locator (mirrors ok_button)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 4: Abort cycle — `run_abort_cycle` + `_exec_abort_measurement` in `cycle.py`

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py` (new step builder, executor, cycle fn)
- Test: `poc/workflow_3/monitor/test_abort_cycle.py` (create — dry-run, no RCS)

**Interfaces:**
- Produces:
  - `build_abort_steps(eqp_id) -> list[WorkflowStep]`
  - `_exec_abort_measurement(step, context, settings) -> StepResult`
  - `run_abort_cycle(eqp_id, recipe_id, settings, *, tag=None) -> CycleResult`
- Consumes: existing `_exec_ensure_rcs_ready`, `_exec_close_alert_popup`, `_exec_connect_tool`, `_exec_wait_tool_window`, `_exec_capture_screen`, `WorkflowRunner`, `close_tool`, `close_alert_window`; new `locate_abort_button` / `locate_abort_confirm` (Task 3); `notify_abort_outcome` (Task 5).

**Outcome contract (`CycleResult.outcome_status`):** `aborted` | `abort_dry_run` | `abort_button_not_found` | `abort_error`.

- [ ] **Step 1: Add the abort step builder + executor map entry**

In `poc/workflow_3/monitor/cycle.py`, after `build_check_steps` / `_CHECK_STEP_EXECUTORS` (line 729), add:

```python
def build_abort_steps(eqp_id: str) -> list[WorkflowStep]:
    """측정 abort 사이클 step — 접속 → 증거 캡처 → Stop/Abort 버튼 클릭(게이트)."""
    return [
        WorkflowStep(
            step_id="ensure_rcs_ready", step_type="recover",
            target_description="RCS 메인 창 확보(전면화/재실행+재로그인)",
            success_criteria=_ctx_set("rcs_main_window"),
        ),
        WorkflowStep(
            step_id="close_alert_popup", step_type="cleanup",
            target_description="감지 알림 팝업 닫기(screenshot 오염 방지)",
        ),
        WorkflowStep(
            step_id="connect_tool", step_type="action",
            target_description=f"List 탭에서 tool 더블클릭: {eqp_id}",
            depends_on=["ensure_rcs_ready"],
        ),
        WorkflowStep(
            step_id="wait_tool_window", step_type="detect",
            target_description="Remote Monitoring 창 대기",
            depends_on=["connect_tool"], success_criteria=_ctx_set("tool_window"),
        ),
        WorkflowStep(
            step_id="capture_screen", step_type="action",
            target_description="abort 전 증거 화면 1장 캡처",
            depends_on=["wait_tool_window"], success_criteria=_ctx_set("capture_path"),
        ),
        WorkflowStep(
            step_id="abort_measurement", step_type="action",
            target_description="Stop/Abort 버튼 클릭으로 측정 중단(이중 게이트)",
            depends_on=["capture_screen"],
        ),
    ]
```

- [ ] **Step 2: Add the `_exec_abort_measurement` executor**

In the same file, after `_exec_capture_screen` (line 683), add the new executor. It locates the button, and clicks only when armed:

```python
def _exec_abort_measurement(step, context, settings: Workflow3Settings) -> StepResult:
    """Stop/Abort 버튼 locate + (무장 시) 클릭 + 확인. 이중 게이트로 보호한다.

    실제 클릭은 SAFE_MODE off **이고** abort_action_dry_run=False 일 때만. 그 외에는
    locate 좌표를 [DRY-RUN] 으로 로깅하고 클릭하지 않는다(notify-only 검증 경로).
    """
    started_at = time.time()
    eqp_id = context["eqp_id"]
    tool_window = context.get("tool_window")
    if tool_window is None:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_no_window", error_message="tool 창 없음 - abort 생략",
        )

    client = None
    try:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.abort_button_vlm_service, timeout_sec=15.0)
    except Exception as exc:
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_vlm_error", error_message=f"{type(exc).__name__}: {exc}",
        )

    from poc.workflow_3.monitor.abort_button import locate_abort_button, locate_abort_confirm

    import numpy as np

    frame = np.array(capture_window(tool_window))
    xy = locate_abort_button(frame_bgr=frame, client=client)
    if xy is None:
        context["abort_outcome"] = "abort_button_not_found"
        print(f"[WARNING] Abort 버튼을 찾지 못함 - 엔지니어 직접 처리 (EQP_ID={eqp_id})")
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_button_not_found", error_message="Stop/Abort 버튼 미검출",
        )

    armed = settings.action_enabled and not settings.abort_action_dry_run
    if not armed:
        context["abort_outcome"] = "abort_dry_run"
        print(f"[INFO] [DRY-RUN] Abort 버튼 검출 screen=({xy[0]},{xy[1]}) - 클릭 생략 "
              f"(SAFE_MODE/abort_dry_run 게이트). EQP_ID={eqp_id}")
        return _make_result(step, "success", started_at, settings)

    # --- 무장 상태: 클릭 + 확인 다이얼로그 ---
    try:
        click_at_screen({"x": xy[0], "y": xy[1]}, "abort_button", action_enabled=True)
        time.sleep(_CHECK_CAPTURE_SETTLE_SEC)
        confirm_frame = np.array(capture_window(tool_window))
        cxy = locate_abort_confirm(frame_bgr=confirm_frame, client=client)
        if cxy is not None:
            click_at_screen({"x": cxy[0], "y": cxy[1]}, "abort_confirm", action_enabled=True)
        context["abort_outcome"] = "aborted"
        print(f"[INFO] 측정 abort 실행: EQP_ID={eqp_id} button=({xy[0]},{xy[1]}) "
              f"confirm={cxy}")
    except Exception as exc:
        context["abort_outcome"] = "abort_error"
        return _make_result(
            step, "failed", started_at, settings,
            failure_class="abort_click_error", error_message=f"{type(exc).__name__}: {exc}",
        )
    return _make_result(step, "success", started_at, settings)


_ABORT_STEP_EXECUTORS = {
    "ensure_rcs_ready": _exec_ensure_rcs_ready,
    "close_alert_popup": _exec_close_alert_popup,
    "connect_tool": _exec_connect_tool,
    "wait_tool_window": _exec_wait_tool_window,
    "capture_screen": _exec_capture_screen,
    "abort_measurement": _exec_abort_measurement,
}
```

- [ ] **Step 3: Add `run_abort_cycle`**

After `run_alarm_cycle` (line 650), add the sibling cycle. It reuses the runner + guaranteed teardown, with no recording / no engineer-watch:

```python
def run_abort_cycle(
    eqp_id: str,
    recipe_id: str,
    settings: Workflow3Settings,
    *,
    tag: str | None = None,
) -> CycleResult:
    """측정 abort 알람 1건 사이클 — 접속 → 증거 캡처 → Stop/Abort 클릭(게이트) → 닫기.

    run_alarm_cycle 의 형제. 녹화/engineer-watch 는 없다(abort 가 곧 행동). step 실패로
    runner 가 중단돼도 cube 알림·tool 닫기·팝업 backstop 은 finally 가 보장한다.
    """
    tag = tag or make_timestamp_tag()
    result = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag)

    if not RCS_MODULES_AVAILABLE:
        result.run_status = "rcs_unavailable"
        result.notes.append("RCS 모듈 비활성 - 감지/로그만")
        notify_abort_outcome(eqp_id, recipe_id, None, enabled=settings.rich_notify_enabled)
        return result

    context: dict = {"eqp_id": eqp_id, "recipe_id": recipe_id, "tag": tag}
    runner = WorkflowRunner(
        settings, workflow_name=f"measurement_abort_{eqp_id}",
        log_name="work2", component_name=LOG_COMPONENT,
    )

    def executor(step, step_context):
        return _ABORT_STEP_EXECUTORS[step.step_id](step, step_context, settings)

    input_blocked = False
    try:
        if _should_block_input(settings):
            input_blocked = block_input(True, debug_label=f"measurement_abort {eqp_id}")
        run = runner.run(build_abort_steps(eqp_id), context, executor)
        result.run_status = run.status
        result.run_dir = str(run.run_dir or "")
        for step_result in run.step_results:
            if step_result.status == "failed":
                result.failed_step = step_result.step_id
                result.failure_class = step_result.failure_class or ""
                break

        result.outcome_status = context.get("abort_outcome", "")
        if context.get("capture_path") is not None:
            result.outcome_path = str(context["capture_path"])
        notify_abort_outcome(
            eqp_id, recipe_id, result.outcome_status or None,
            capture_path=result.outcome_path, enabled=settings.rich_notify_enabled,
        )
    except Exception as exc:
        result.run_status = "error"
        result.notes.append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] abort 사이클 예외: EQP_ID={eqp_id}, error={exc}")
        log_work2_event(
            component=LOG_COMPONENT, message="abort_cycle_error", level="error",
            eqp_id=eqp_id, error=str(exc),
        )
    finally:
        if input_blocked:
            block_input(False, debug_label=f"measurement_abort {eqp_id}")
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            try:
                close_tool(eqp_id)
            except Exception as exc:
                print(f"[WARNING] tool 창 닫기 실패: {exc}")
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return result
```

Add the `notify_abort_outcome` import to the existing notify import (line 33):

```python
from poc.workflow_3.monitor.notify import (
    close_alert_window,
    notify_abort_outcome,
    notify_correction_outcome,
)
```

- [ ] **Step 4: Create the dry-run self-test**

Create `poc/workflow_3/monitor/test_abort_cycle.py`:

```python
"""run_abort_cycle dry-run self-test (RCS 불필요).

RCS 모듈이 없는 개발 PC 에서 run_abort_cycle 이 rcs_unavailable 로 안전 종료하고
cube notify 를 호출하는지, CycleResult 형태가 맞는지 검증한다.

    uv run python poc/workflow_3/monitor/test_abort_cycle.py
"""

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import cycle as cycle_mod
from poc.workflow_3.monitor.cycle import run_abort_cycle


def test_rcs_unavailable_safe_exit():
    """RCS 모듈 비활성이면 rcs_unavailable 로 끝나고 notify 가 1회 불린다."""
    calls = []
    orig_notify = cycle_mod.notify_abort_outcome
    orig_avail = cycle_mod.RCS_MODULES_AVAILABLE
    cycle_mod.notify_abort_outcome = lambda *a, **k: calls.append((a, k))
    cycle_mod.RCS_MODULES_AVAILABLE = False
    try:
        result = run_abort_cycle("EQP1", "CLS/RCP", load_workflow3_settings(), tag="t0")
    finally:
        cycle_mod.notify_abort_outcome = orig_notify
        cycle_mod.RCS_MODULES_AVAILABLE = orig_avail
    ok = result.run_status == "rcs_unavailable" and len(calls) == 1 and result.eqp_id == "EQP1"
    print(f"[{'PASS' if ok else 'FAIL'}] rcs_unavailable_safe_exit: "
          f"status={result.run_status} notify_calls={len(calls)}")
    return ok


def main():
    print("[INFO] run_abort_cycle self-test 시작")
    results = [test_rcs_unavailable_safe_exit()]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the test, verify PASS**

Run: `uv run python poc/workflow_3/monitor/test_abort_cycle.py`
Expected: `[INFO] 1/1 cases passed`. (Depends on Task 5's `notify_abort_outcome` existing; if running tasks strictly in order, land Task 5's notify stub first or accept an ImportError until Task 5 — see note in Task 5 Step 1.)

- [ ] **Step 6: Commit**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_abort_cycle.py
git commit -m "feat(workflow_3): run_abort_cycle + gated Stop/Abort executor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 5: Notify — `notify_abort_outcome`

> **Ordering note:** `run_abort_cycle` (Task 4) imports `notify_abort_outcome`. If implementing strictly task-by-task, do **Task 5 Step 1** (add the function) *before* running Task 4's test. The plan lists it after Task 4 only because the cycle is the conceptual centerpiece; the dependency runs the other way.

**Files:**
- Modify: `poc/workflow_3/monitor/notify.py` (add `notify_abort_outcome`)
- Test: covered by Task 4's `test_abort_cycle.py` (notify call counted).

**Interfaces:**
- Produces: `notify_abort_outcome(eqp_id, recipe_id, outcome, *, capture_path="", enabled=True) -> None`
- Consumes: the same office rich-notify adapter `notify_correction_outcome` uses.

- [ ] **Step 1: Read the existing `notify_correction_outcome` and mirror it**

Open `poc/workflow_3/monitor/notify.py`, locate `notify_correction_outcome`, and add a sibling that summarizes the abort. Mirror its office-adapter gating and `enabled` short-circuit exactly; only the message text differs:

```python
def notify_abort_outcome(
    eqp_id: str,
    recipe_id: str,
    outcome,
    *,
    capture_path: str = "",
    enabled: bool = True,
) -> None:
    """측정 abort 결과를 cube rich notification 으로 보낸다(correction 알림의 형제).

    outcome 은 문자열 status('aborted'|'abort_dry_run'|'abort_button_not_found'|
    'abort_error') 또는 None(rcs 비활성). enabled=False 면 아무 것도 하지 않는다.
    notify_correction_outcome 과 동일한 office 어댑터 게이팅을 따른다.
    """
    if not enabled:
        return
    status = outcome or "unknown"
    summary = {
        "aborted": "측정을 자동 중단했습니다(연속 측정 실패 임계 도달).",
        "abort_dry_run": "측정 중단 대상 감지(notify-only) - 엔지니어 확인 필요.",
        "abort_button_not_found": "Stop/Abort 버튼 미검출 - 엔지니어 직접 중단 필요.",
        "abort_error": "측정 중단 시도 중 오류 - 엔지니어 직접 확인 필요.",
        "unknown": "측정 실패 abort 잡 결과 미상 - 엔지니어 확인 필요.",
    }.get(status, "측정 실패 abort 잡 - 엔지니어 확인 필요.")
    # (office rich-notify 어댑터 호출 — notify_correction_outcome 과 동일한 로딩/게이트
    #  패턴을 따른다. 어댑터 부재(개발 PC)면 [INFO] 로그만.)
    print(f"[INFO] abort notify: EQP_ID={eqp_id} recipe={recipe_id} "
          f"status={status} msg={summary} capture={capture_path}")
    # TODO(office): notify_correction_outcome 이 쓰는 동일 어댑터로 cube 발송을 연결.
    #   여기서는 dev-PC 안전을 위해 [INFO] 로그로만 두고, 오피스에서 어댑터를 붙인다.
```

> Match the real office-adapter call used by `notify_correction_outcome` (read it first). Keep the dev-PC-safe `[INFO]` fallback so the function never raises when the office module is absent.

- [ ] **Step 2: Verify import + Task 4 test now passes**

Run:
```bash
uv run python -c "from poc.workflow_3.monitor.notify import notify_abort_outcome; print('[PASS] import ok')"
uv run python poc/workflow_3/monitor/test_abort_cycle.py
```
Expected: `[PASS] import ok`, then `[INFO] 1/1 cases passed`.

- [ ] **Step 3: Commit**

```bash
git add poc/workflow_3/monitor/notify.py
git commit -m "feat(workflow_3): notify_abort_outcome cube summary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 6: Loop integration — `process_abort_rows` + `monitor_loop` wiring + manifest

**Files:**
- Modify: `poc/workflow_3/monitor/align_fail_monitor.py`
- Test: `poc/workflow_3/monitor/test_process_abort_rows.py` (create)

**Interfaces:**
- Produces:
  - `process_abort_rows(meas_fails, aborted_tools, settings, abort_cooldown=None) -> int`
  - `append_abort_manifest(info, cycle)` → `measurement_abort_cycles.csv`
- Consumes: `run_abort_cycle` (Task 4), existing `_collapse_rows_by_tool`, `_alarm_time_to_tag`, `filter_rows_within_window`, `_OCCUPIED_FAILURE_CLASSES`.

- [ ] **Step 1: Write the failing self-test (edge-trigger + gating)**

Create `poc/workflow_3/monitor/test_process_abort_rows.py`:

```python
"""process_abort_rows edge-trigger / 게이트 self-test (run_abort_cycle 모킹).

- meas_fail_abort_enabled=False 면 사이클 안 돈다.
- 새 EQP_ID 마다 1회만 abort 사이클(중복 알람 무시), 해제되면 재처리 가능.

    uv run python poc/workflow_3/monitor/test_process_abort_rows.py
"""

import pandas as pd

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult


def _settings(**over):
    s = load_workflow3_settings()
    return s.__class__(**{**s.to_snapshot() if hasattr(s, "to_snapshot") else s.__dict__, **over}) \
        if False else _replace(s, **over)


def _replace(s, **over):
    """frozen dataclass 치환 헬퍼."""
    import dataclasses
    return dataclasses.replace(s, **over)


def _rows(*eqps):
    return pd.DataFrame([{"EQP_ID": e, "ALID": "9012", "UTC9": "", "RECIPE_ID": "CLS/RCP"} for e in eqps])


def test_disabled_skips(monkey_calls=None):
    """잡 비활성이면 사이클 0회."""
    calls = []
    orig = afm.run_alarm_cycle  # noqa
    afm_run = afm.__dict__.get("run_abort_cycle")
    afm.run_abort_cycle = lambda *a, **k: calls.append(a) or CycleResult("E", "R", "t")
    try:
        n = afm.process_abort_rows(_rows("EQP1"), set(), _replace(load_workflow3_settings(),
                                   meas_fail_abort_enabled=False), {})
    finally:
        afm.run_abort_cycle = afm_run
    ok = n == 0 and len(calls) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] disabled_skips: handled={n} calls={len(calls)}")
    return ok


def test_edge_trigger_once_per_tool():
    """같은 EQP_ID 는 active 인 동안 1회만 처리."""
    calls = []
    afm_run = afm.__dict__.get("run_abort_cycle")
    afm.run_abort_cycle = lambda eqp, rcp, st, **k: (calls.append(eqp) or CycleResult(eqp, rcp, "t"))
    active = set()
    try:
        s = _replace(load_workflow3_settings(), meas_fail_abort_enabled=True, cycle_enabled=True)
        n1 = afm.process_abort_rows(_rows("EQP1"), active, s, {})
        n2 = afm.process_abort_rows(_rows("EQP1"), active, s, {})  # 중복 - 무시
    finally:
        afm.run_abort_cycle = afm_run
    ok = n1 == 1 and n2 == 0 and calls == ["EQP1"]
    print(f"[{'PASS' if ok else 'FAIL'}] edge_trigger_once_per_tool: n1={n1} n2={n2} calls={calls}")
    return ok


def main():
    print("[INFO] process_abort_rows self-test 시작")
    results = [test_disabled_skips(), test_edge_trigger_once_per_tool()]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

> Simplify the helper: if `_settings` proves awkward, just use `dataclasses.replace(load_workflow3_settings(), **over)` inline. The test only needs `meas_fail_abort_enabled` and `cycle_enabled` toggled.

- [ ] **Step 2: Run the test, verify it FAILS**

Run: `uv run python poc/workflow_3/monitor/test_process_abort_rows.py`
Expected: `AttributeError: module ... has no attribute 'process_abort_rows'`.

- [ ] **Step 3: Implement `process_abort_rows` + `append_abort_manifest`**

In `poc/workflow_3/monitor/align_fail_monitor.py`, add the manifest path constant near `CYCLE_MANIFEST_PATH` (line 45):

```python
ABORT_MANIFEST_PATH = LOG_DIR / "measurement_abort_cycles.csv"
```

Add the import of `run_abort_cycle` to the cycle import (line 34):

```python
from poc.workflow_3.monitor.cycle import CycleResult, run_abort_cycle, run_alarm_cycle
```

Add `append_abort_manifest` (reuse `CYCLE_MANIFEST_COLUMNS`) after `append_cycle_manifest` (line 239):

```python
def append_abort_manifest(info: dict, cycle: CycleResult) -> None:
    """측정 abort 사이클 1건을 measurement_abort_cycles.csv 에 한 줄 누적한다.

    align manifest 와 같은 컬럼/형식(CycleResult)을 재사용한다. 기록 실패는 삼킨다.
    """
    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ABORT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_header = not ABORT_MANIFEST_PATH.exists() or ABORT_MANIFEST_PATH.stat().st_size == 0
        with ABORT_MANIFEST_PATH.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if write_header:
                writer.writerow(CYCLE_MANIFEST_COLUMNS)
            writer.writerow([
                detected_at, cycle.eqp_id, cycle.recipe_id, info["alid"], info["utc9"],
                info["alarm_name"], cycle.run_status, cycle.failed_step, cycle.failure_class,
                cycle.outcome_status, cycle.outcome_path, cycle.key_decision, cycle.best_xy,
                cycle.frame_count, cycle.recording_dir, cycle.run_dir,
            ])
        print(f"[INFO] abort manifest 기록 -> {ABORT_MANIFEST_PATH} "
              f"(EQP_ID={cycle.eqp_id}, outcome={cycle.outcome_status or '-'})")
    except Exception as exc:
        print(f"[WARNING] abort manifest 기록 실패: {exc}")
```

Add `process_abort_rows` after `process_fail_rows` (line 380):

```python
def process_abort_rows(
    meas_fails,
    aborted_tools: set[str],
    settings: Workflow3Settings,
    abort_cooldown: dict | None = None,
) -> int:
    """측정 실패 임계 알람을 EQP_ID 기준 edge-trigger 로 처리해 abort 사이클을 돈다.

    process_fail_rows 의 얇은 형제: 팝업/gather/correction 없이 run_abort_cycle 만 돌린다.
    잡 비활성(meas_fail_abort_enabled=False)이면 아무 것도 하지 않는다. 점유(select)로
    포기하면 active 에 넣지 않고 cooldown 등록(align 과 동일).
    """
    if not settings.meas_fail_abort_enabled:
        return 0
    if abort_cooldown is None:
        abort_cooldown = {}
    by_tool = _collapse_rows_by_tool(meas_fails)
    current_tools = set(by_tool.keys())

    now = time.time()
    for eqp_id in list(abort_cooldown):
        if eqp_id not in current_tools or now >= abort_cooldown[eqp_id]:
            del abort_cooldown[eqp_id]
    cooling = current_tools & set(abort_cooldown)

    new_tools = current_tools - aborted_tools - cooling
    cleared = aborted_tools - current_tools
    for eqp_id in sorted(cleared):
        print(f"[INFO] 측정 실패 알람 해제: EQP_ID={eqp_id}")
    aborted_tools.difference_update(cleared)

    handled = 0
    for eqp_id in sorted(new_tools):
        info = by_tool[eqp_id]
        print(f"[WARNING] 측정 실패 임계 감지: EQP_ID={eqp_id}, ALID={info['alid']}, "
              f"RECIPE_ID={info['recipe_id']}, 시각={info['alarm_time']}")
        append_alarm_record(
            eqp_id, str(info["alarm_time"] or ""), info["alarm_name"], info["alid"],
            recipe_id=info["recipe_id"], operation_desc=info["operation_desc"],
            lot_type_cd=info["lot_type_cd"],
        )
        if settings.cycle_enabled:
            cycle = run_abort_cycle(eqp_id, info["recipe_id"], settings,
                                    tag=_alarm_time_to_tag(info["utc9"]))
        else:
            cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
            cycle.run_status = "cycle_disabled"
        append_abort_manifest(info, cycle)

        if cycle.failure_class in _OCCUPIED_FAILURE_CLASSES:
            abort_cooldown[eqp_id] = time.time() + settings.occupied_retry_cooldown_sec
            print(f"[INFO] EQP_ID={eqp_id} 점유 추정 - active 미등록, "
                  f"{settings.occupied_retry_cooldown_sec:.0f}s 후 재시도")
        else:
            aborted_tools.add(eqp_id)
        handled += 1
    return handled
```

- [ ] **Step 4: Run the test, verify PASS**

Run: `uv run python poc/workflow_3/monitor/test_process_abort_rows.py`
Expected: `[INFO] 2/2 cases passed`.

- [ ] **Step 5: Wire into `monitor_loop`**

In `monitor_loop`, add the abort dedup state next to `active_tools` (after line 397):

```python
    aborted_tools: set[str] = set()  # 측정 실패 abort edge-trigger 상태.
    abort_cooldown: dict = {}         # 점유로 포기한 abort 대상 재시도 유예.
```

In the poll body, after the align `fails` are computed and processed, add the measurement-fail branch. Replace the `else:` block that calls `process_fail_rows` (lines 432-439) so both jobs dispatch from the same tick:

```python
            else:
                idle_logged = False
                count = process_fail_rows(fails, active_tools, settings, occupied_cooldown)
                if count == 0:
                    print(
                        f"[INFO] {datetime.now().strftime('%H:%M:%S')} - "
                        f"신규 없음 (활성 {len(active_tools)}대 유지)"
                    )

            # 측정 실패 abort 잡 — 같은 alarms 스트림을 두 번째 필터로 본다(단일 GUI 직렬).
            if settings.meas_fail_abort_enabled and alarms is not None:
                meas = source.filter_measurement_fail(alarms)
                meas = filter_rows_within_window(meas, settings.detection_window_sec)
                if not _alarm_rows_empty(meas):
                    process_abort_rows(meas, aborted_tools, settings, abort_cooldown)
                elif aborted_tools:
                    for eqp_id in sorted(aborted_tools):
                        print(f"[INFO] 측정 실패 알람 해제: EQP_ID={eqp_id}")
                    aborted_tools.clear()
```

> Note: the align branch is `if/else` on align emptiness. The new abort block sits **outside** that `else` (same indentation as the `if _alarm_rows_empty(fails):` / `else:` pair) so abort is evaluated every tick regardless of align state. Keep `alarms` in scope (it already is — `alarms = source.poll()` at line 420).

Update the startup banner (after line 408) to report the abort job state:

```python
    print(
        f"[INFO] 측정 실패 abort 잡: {'on' if settings.meas_fail_abort_enabled else 'off'}"
        f"{' (notify-only, dry-run)' if settings.meas_fail_abort_enabled and settings.abort_action_dry_run else ''}"
        f"{' [ARMED]' if settings.meas_fail_abort_enabled and not settings.abort_action_dry_run else ''}"
        f", ALID={settings.meas_fail_alid or '미설정'}"
    )
```

- [ ] **Step 6: Verify the replay dry-run drives both jobs end-to-end**

Create a tiny fixture and run the loop once in replay. Use an existing align replay CSV if present; otherwise:

```bash
printf 'EQP_ID,ALID,RECIPE_ID\nEQPX,9012,CLS/RCP\n' > /tmp/meas_replay.csv
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=/tmp/meas_replay.csv \
  MEAS_FAIL_ALID=9012 ALIGN_FAIL_POLL_SEC=2 \
  timeout 8 uv run python poc/workflow_3/monitor/align_fail_monitor.py
```
Expected: startup banner shows `측정 실패 abort 잡: on (notify-only, dry-run), ALID=9012`; the replay row flows to `[WARNING] 측정 실패 임계 감지: EQP_ID=EQPX`; `run_abort_cycle` reports `rcs_unavailable` (no RCS on dev PC) and a `notify_abort_outcome` `[INFO]` line; an `measurement_abort_cycles.csv` row is written. No traceback. (Ctrl+C / timeout ends it.)

- [ ] **Step 7: Confirm the align path is unchanged**

Run: `uv run python poc/workflow_3/monitor/test_success_gather.py` and re-run any existing monitor self-tests touched. Expected: still pass; `process_fail_rows` and its call site are unmodified (only an additional sibling branch was added).

- [ ] **Step 8: Commit**

```bash
git add poc/workflow_3/monitor/align_fail_monitor.py poc/workflow_3/monitor/test_process_abort_rows.py
git commit -m "feat(workflow_3): dispatch measurement-abort job from the main loop

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

### Task 7: Docs — README + CLAUDE.md

**Files:**
- Modify: `poc/workflow_3/README.md` (loop description + `MEAS_FAIL_*` env table + office checklist)
- Modify: `CLAUDE.md` (two-job loop note + abort double-gate)

**Interfaces:** none (docs only).

- [ ] **Step 1: README — document the second job**
  - Add to the loop description that the loop now handles two MES alarm classes (align fail + measurement-fail abort) sharing one serialized GUI.
  - Add a `MEAS_FAIL_*` env table: `MEAS_FAIL_ABORT_ENABLED` (1), `MEAS_FAIL_ALID` ("" — office-confirmed), `MEAS_FAIL_ABORT_DRY_RUN` (1 — notify-only default), `MEAS_FAIL_ABORT_BUTTON_SERVICE` (ui-venus).
  - Add to the office checklist: provide `filter_measurement_fail` in `office_align_fail_alarm`; confirm `MEAS_FAIL_ALID`; calibrate the Stop/Abort button; arm (`SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0`) only after dry-run verification.

- [ ] **Step 2: CLAUDE.md — `monitor/` description + env note**
  - In the `monitor/` bullet, note `run_abort_cycle` / `process_abort_rows` / `abort_button.py` as the measurement-abort job (second MES alarm class, shares the serialized GUI, double-gated, notify-only default).
  - Add the `MEAS_FAIL_*` flags to the "Recently added env flags" section, mirroring the existing entries' style, including the double-gate (`SAFE_MODE=0` + `MEAS_FAIL_ABORT_DRY_RUN=0`).

- [ ] **Step 3: Sanity-check imports still load**

Run: `uv run python -c "import poc.workflow_3.monitor.align_fail_monitor, poc.workflow_3.monitor.cycle; print('[PASS] imports ok')"`
Expected: `[PASS] imports ok`.

- [ ] **Step 4: Commit**

```bash
git add poc/workflow_3/README.md CLAUDE.md
git commit -m "docs(workflow_3): document measurement-abort job + MEAS_FAIL_* env

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MKXKY55Jv81LPR2TiNKT9R"
git show --stat HEAD | head -10
```

---

## Final verification

- [ ] **Run all new self-tests together**

```bash
uv run python poc/workflow_3/test_config_meas_fail.py
uv run python poc/workflow_3/monitor/test_alarm_source_meas.py
uv run python poc/workflow_3/monitor/test_abort_button.py
uv run python poc/workflow_3/monitor/test_abort_cycle.py
uv run python poc/workflow_3/monitor/test_process_abort_rows.py
```
Expected: 4/4, 3/3, 2/2, 1/1, 2/2 — all pass.

- [ ] **Confirm the align path is untouched**

Run: `git diff --stat main -- poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/align_fail_monitor.py`
Expected: only **additions** (new functions/branch); `run_alarm_cycle`, `process_fail_rows`, and the align `process_fail_rows(...)` call site bodies are unchanged.

- [ ] **Replay dry-run drives both jobs** (Task 6 Step 6) — banner shows abort job on/notify-only, abort manifest row written, no traceback.

---

## Self-Review

**Spec coverage:**
- Spec §Changes 1 (detector: optional `filter_measurement_fail` + replay + optional office attr) → Task 2. ✓
- Spec §Changes 2 (`MEAS_FAIL_*` config + double gate) → Task 1. ✓
- Spec §Changes 3 (abort-button locator, mirrors ok_button, VLM region-only) → Task 3. ✓
- Spec §Changes 4 (abort cycle: reuse steps + new gated executor, no recording/watch) → Task 4. ✓
- Spec §Changes 5 (`notify_abort_outcome`) → Task 5. ✓
- Spec §Changes 6 (loop integration: `process_abort_rows` + `monitor_loop` + manifest) → Task 6. ✓
- Spec §Changes 7 (README + CLAUDE.md) → Task 7. ✓
- Spec §Staged enablement (notify-only default, double gate) → Task 1 (`abort_action_dry_run` default True + SAFE_MODE force) + Task 4 (armed check). ✓
- Spec §Out of scope (align path untouched; no streak tracking; no preemption) → asserted in Final verification + no align edits. ✓

**Placeholder scan:** One intentional `TODO(office)` in Task 5 (the real cube adapter is gitignored/office-only — the dev-PC `[INFO]` fallback is complete and correct). No other TBD/incomplete blocks.

**Ordering hazard flagged:** Task 4 imports Task 5's `notify_abort_outcome`; the note in Task 5 + Task 4 Step 5 directs landing the notify function first. ✓

**Type consistency:** `run_abort_cycle(...) -> CycleResult`, `process_abort_rows(...) -> int`, `locate_abort_button(*, frame_bgr, client) -> tuple[int,int]|None`, `notify_abort_outcome(eqp_id, recipe_id, outcome, *, capture_path="", enabled=True)` are used identically across tasks and tests. `CycleResult` columns reused for the abort manifest. ✓

**Convention compliance:** no argparse; Korean docstrings; `[INFO]/[WARNING]/[ERROR]` prints (no `logging`); no em-dash in `print()` strings; absolute imports; route_slug (not model name) for VLM service; pathspec commits. ✓
