# Loop Failure-Path Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the workflow_3 real-time loop survive injected faults — teardown always completes, a broken tool cannot starve the others, and no office call can stall the monitor forever.

**Architecture:** A new `monitor/teardown.py` provides `run_teardown(steps)`, which runs each labelled teardown step under its own guard and always continues. Each of the three cycles builds its teardown list in a named `_*_teardown_steps()` function so the "unblock input first" ordering is assertable without an RCS stack. The monitor loop gains a failure cooldown (reusing the existing occupied-cooldown dict) and a per-tool guard; `gather_rcp_msr` gains an optional bound using `success_gather`'s daemon-thread + bounded-join idiom.

**Tech Stack:** Python 3.10+, stdlib `threading` only. No new dependencies. Tests are plain scripts run with `uv run python <path>` (the `monitor/` convention), not pytest.

**Spec:** `poc/workflow_3/docs/superpowers/specs/2026-08-06-loop-failure-path-hardening-design.md`

## Global Constraints

- **Korean docstrings** throughout — every new function and module gets one.
- **Print-based logging** with `[INFO]` / `[ERROR]` / `[WARNING]` prefixes. Never the `logging` module (exception: `poc/workflow_3/logger.py` audit trail, which you call via `log_work2_event`).
- **No em-dash (U+2014) inside `print()` strings** — the office console is cp949 and cannot encode it. Docstrings and comments may use it.
- **No `from __future__ import annotations`** or any `__future__` import.
- **No CLI arguments** — no `argparse`, no flags. All configuration via `Workflow3Settings` fields read from `ALIGN_FAIL_*` env vars.
- **Absolute imports** within `poc/`: `from poc.workflow_3.xxx import ...`. workflow_3 never imports workflow_3e; workflow_3e imports workflow_3 one-way.
- **Tests run directly**, not under pytest: each test file ends with an `if __name__ == "__main__":` block calling every test function, and each test prints `[OK] <test_name>` on success.
- **Commit directly to `main`** (no branches), and **commit with explicit pathspecs** — never `git add -A` or `git commit -a`, because parallel sessions edit this repo concurrently. Verify scope with `git show --stat` after each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `poc/workflow_3/monitor/teardown.py` | **Create.** `run_teardown(steps, *, label)` — the guarded-teardown primitive. One function, no imports from `cycle.py`. | 1 |
| `poc/workflow_3/monitor/test_teardown.py` | **Create.** Helper unit tests + the ordering assertions for all three cycles. | 1, 2, 3, 4 |
| `poc/workflow_3/monitor/cycle.py` | **Modify.** Add `_teardown_steps()` (alarm cycle) and `_check_teardown_steps()` (check-only); both `finally` blocks call `run_teardown`. | 2, 3 |
| `poc/workflow_3e/abort_cycle.py` | **Modify.** Add `_abort_teardown_steps()`; `finally` calls `run_teardown`. | 4 |
| `poc/workflow_3/config.py` | **Modify.** Two new `Workflow3Settings` fields + their env reads. | 5, 7 |
| `poc/workflow_3/workflow_3_config.example.py` | **Modify.** Two new scratch-config constants. | 5, 7 |
| `poc/workflow_3/workflow_3_config_loader.py` | **Modify.** Two new `_CONST_TO_ENV` rows. | 5, 7 |
| `poc/workflow_3/monitor/align_fail_monitor.py` | **Modify.** Failure cooldown + per-tool guard in `process_fail_rows`. | 5 |
| `poc/workflow_3/monitor/test_failure_cooldown.py` | **Create.** Cooldown + per-tool-guard tests for both monitors. | 5, 6 |
| `poc/workflow_3/monitor/align_fail_monitor_only_check.py` | **Modify.** Same cooldown + guard in its own `process_fail_rows` copy. | 6 |
| `poc/workflow_3/monitor/rcp_msr_gather.py` | **Modify.** Optional `timeout_sec` bound + in-flight guard. | 7 |
| `poc/workflow_3/monitor/test_rcp_gather_timeout.py` | **Create.** Timeout + in-flight-guard tests. | 7 |
| `CLAUDE.md` | **Modify.** Document the two new env flags. | 8 |

**Task order matters:** Task 1 produces the primitive that Tasks 2-4 consume. Task 5 establishes the cooldown pattern that Task 6 mirrors. Task 7 is independent of 1-6. Task 8 is documentation only.

---

### Task 1: The guarded-teardown primitive

**Files:**
- Create: `poc/workflow_3/monitor/teardown.py`
- Test: `poc/workflow_3/monitor/test_teardown.py`

**Interfaces:**
- Consumes: nothing (leaf module — imports only stdlib).
- Produces: `run_teardown(steps: list[tuple[str, Callable[[], None]]], *, label: str = "") -> list[tuple[str, str]]`. Returns a list of `(step_name, error_string)` for steps that raised, in execution order. Returns `[]` when everything succeeded. **Never raises.**

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/monitor/test_teardown.py`:

```python
"""guarded teardown 헬퍼 + 세 사이클의 teardown 순서 불변식 테스트.

RCS/Windows 없이 도는 단위 테스트다. 사이클 전체는 Mac 에서 RCS_MODULES_AVAILABLE
이 False 라 조기 반환하므로, teardown 목록을 만드는 함수(_teardown_steps 계열)만
직접 호출해 순서를 검사한다.

`uv run python poc/workflow_3/monitor/test_teardown.py` 로 직접 실행.
"""

from poc.workflow_3.monitor.teardown import run_teardown


def test_raising_step_does_not_block_later_steps():
    """한 단계가 던져도 뒤 단계는 반드시 실행된다 - teardown 의 핵심 계약."""
    calls = []

    def _boom():
        calls.append("boom")
        raise RuntimeError("terminate failed")

    failures = run_teardown([
        ("first", lambda: calls.append("first")),
        ("boom", _boom),
        ("last", lambda: calls.append("last")),
    ])
    assert calls == ["first", "boom", "last"], calls
    assert [n for n, _ in failures] == ["boom"], failures
    print("[OK] test_raising_step_does_not_block_later_steps")


def test_failures_returned_in_order_with_names():
    """실패는 (이름, 오류문자열) 로, 실행 순서대로 반환된다."""
    failures = run_teardown([
        ("a", lambda: (_ for _ in ()).throw(ValueError("va"))),
        ("b", lambda: None),
        ("c", lambda: (_ for _ in ()).throw(KeyError("kc"))),
    ])
    assert [n for n, _ in failures] == ["a", "c"], failures
    assert "va" in failures[0][1], failures[0][1]
    assert "ValueError" in failures[0][1], failures[0][1]
    print("[OK] test_failures_returned_in_order_with_names")


def test_helper_never_raises():
    """모든 단계가 던져도 헬퍼 자체는 절대 예외를 올리지 않는다."""
    failures = run_teardown([
        ("x", lambda: (_ for _ in ()).throw(RuntimeError("x"))),
        ("y", lambda: (_ for _ in ()).throw(RuntimeError("y"))),
    ], label="unit")
    assert len(failures) == 2, failures
    print("[OK] test_helper_never_raises")


def test_empty_list_is_noop():
    assert run_teardown([]) == []
    print("[OK] test_empty_list_is_noop")


if __name__ == "__main__":
    test_raising_step_does_not_block_later_steps()
    test_failures_returned_in_order_with_names()
    test_helper_never_raises()
    test_empty_list_is_noop()
    print("\n[OK] teardown 헬퍼 테스트 통과")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_teardown.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'poc.workflow_3.monitor.teardown'`

- [ ] **Step 3: Write minimal implementation**

Create `poc/workflow_3/monitor/teardown.py`:

```python
"""사이클 teardown 을 단계별로 독립 보호해 실행하는 헬퍼.

teardown 의 계약은 "가능한 만큼 정리한다" 이다. 한 단계가 실패했다고 뒤 단계
(특히 사용자 입력 차단 해제, tool 창 닫기)를 건너뛰면 장비와 엔지니어가 잠긴 채
남는다. 그래서 각 단계를 개별 try 로 감싸고 무조건 다음으로 넘어간다.

**순서 규약**: 모든 teardown 목록의 **첫 단계는 사용자 입력 차단 해제**여야 한다.
뒤 단계가 전부 실패해도 엔지니어의 마우스/키보드는 풀려 있어야 하기 때문이다.
이 규약은 test_teardown.py 가 세 사이클 모두에 대해 검사한다.
"""


def run_teardown(steps, *, label=""):
    """teardown 단계를 순서대로 실행하되, 각 단계를 독립 보호한다.

    steps: (이름, 인자없는 callable) 목록. 이름은 로그/반환값 식별자다.
    label: 로그에 붙일 사이클 식별 문자열(예: "align_fail_cycle EQP1").
    반환: 실패한 (이름, "예외타입: 메시지") 목록 — 성공만이면 빈 목록.
    이 함수 자체는 어떤 경우에도 예외를 올리지 않는다.
    """
    failures = []
    suffix = f" [{label}]" if label else ""
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            failures.append((name, detail))
            print(f"[WARNING] teardown 단계 실패({name}){suffix}: {detail}")
    return failures


__all__ = ["run_teardown"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python poc/workflow_3/monitor/test_teardown.py`
Expected: PASS — four `[OK]` lines then `[OK] teardown 헬퍼 테스트 통과`

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/teardown.py poc/workflow_3/monitor/test_teardown.py
git commit -m "feat(workflow_3): guarded teardown 헬퍼 - 단계 실패가 뒤를 막지 않게

teardown 계약은 '가능한 만큼 정리'다. 각 단계를 독립 try 로 감싸 한 단계가
던져도 입력 해제/tool 닫기가 반드시 실행되게 한다. 실패는 (이름, 오류)로
반환해 호출부가 result.notes 에 남길 수 있다."
git show --stat --oneline HEAD
```

---

### Task 2: Adopt in `run_alarm_cycle`

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py` (add `_teardown_steps`, rewrite `finally` at `:630-648`)
- Test: `poc/workflow_3/monitor/test_teardown.py` (append)

**Interfaces:**
- Consumes: `run_teardown(steps, *, label)` from Task 1.
- Produces: `_teardown_steps(eqp_id, context, result, settings, *, input_blocked, recording) -> list[tuple[str, Callable]]`. First element is always `("input_unblock", ...)`. Tasks 3 and 4 mirror this shape with different names.

**Design note the implementer must preserve:** every step is *always* present in the returned list, with its precondition checked *inside* the closure. Do not filter steps out when a precondition fails — a constant-length, constant-order list is what makes the ordering test meaningful.

- [ ] **Step 1: Write the failing test**

Append to `poc/workflow_3/monitor/test_teardown.py` (before the `if __name__` block):

```python
def test_alarm_cycle_teardown_unblocks_input_first():
    """run_alarm_cycle 의 teardown 첫 단계는 반드시 입력 해제여야 한다.

    뒤 단계(녹화 중지/tool 닫기/팝업)가 전부 실패해도 엔지니어의 물리 입력은
    풀려 있어야 한다. 이것이 F1(check-only 입력 잠금) 계열의 회귀 테스트다.
    """
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.cycle import CycleResult, _teardown_steps

    settings = load_workflow3_settings()
    result = CycleResult(eqp_id="EQP1", recipe_id="C/R", tag="t")
    steps = _teardown_steps(
        "EQP1", {}, result, settings, input_blocked=True, recording=None
    )
    assert steps[0][0] == "input_unblock", [n for n, _ in steps]
    names = [n for n, _ in steps]
    assert names == ["input_unblock", "recording_stop", "close_tool", "close_alert"], names
    print("[OK] test_alarm_cycle_teardown_unblocks_input_first")
```

And add to the `if __name__ == "__main__":` block, before the final print:

```python
    test_alarm_cycle_teardown_unblocks_input_first()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_teardown.py`
Expected: FAIL with `ImportError: cannot import name '_teardown_steps' from 'poc.workflow_3.monitor.cycle'`

- [ ] **Step 3: Write minimal implementation**

In `poc/workflow_3/monitor/cycle.py`, add this import near the other `monitor` imports at the top of the file:

```python
from poc.workflow_3.monitor.teardown import run_teardown
```

Then add `_teardown_steps` immediately above `def run_alarm_cycle(`:

```python
def _teardown_steps(eqp_id, context, result, settings, *, input_blocked, recording):
    """알람 사이클 teardown 단계 목록 - 순서가 계약이다.

    첫 단계는 **항상** 입력 해제다: 뒤 단계가 전부 실패해도 엔지니어의 물리
    마우스/키보드는 풀려 있어야 한다. 각 단계의 전제조건은 목록에서 빼지 않고
    클로저 **안에서** 판정한다 - 목록 길이/순서를 일정하게 유지해야 순서 규약을
    테스트로 검사할 수 있다.
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"align_fail_cycle {eqp_id}")

    def _stop_recording():
        # stop() 과 결과 필드 갱신을 함께 감싼다 - 여기서 실패하면 두 필드가
        # 조용히 비는 대신 note 가 남고 나머지 teardown 은 계속된다.
        sess = recording if recording is not None else context.get("recording")
        if sess is None:
            return
        frames = sess.stop("cycle_teardown")
        result.recording_dir = str(sess.out_dir)
        result.frame_count = len(frames)

    def _close_tool():
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)

    def _close_alert():
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return [
        ("input_unblock", _unblock),
        ("recording_stop", _stop_recording),
        ("close_tool", _close_tool),
        ("close_alert", _close_alert),
    ]
```

Now replace the entire `finally:` block of `run_alarm_cycle` (currently `cycle.py:630-648`) with:

```python
    finally:
        # teardown 은 run_teardown 이 단계별로 보호한다 - 한 단계가 던져도 입력
        # 해제/tool 닫기/팝업 backstop 은 반드시 실행된다.
        failures = run_teardown(
            _teardown_steps(
                eqp_id, context, result, settings,
                input_blocked=input_blocked, recording=recording,
            ),
            label=f"align_fail_cycle {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)
```

Delete the now-dead `input_blocked = False` assignment that followed the old unblock call inside `finally`; the local is not read after the `finally` block.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_teardown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
```
Expected: both PASS. `test_occupied_popup.py` must be unchanged and still green — it exercises `process_fail_rows`, which calls `run_alarm_cycle`.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_teardown.py
git commit -m "refactor(workflow_3): run_alarm_cycle teardown 을 run_teardown 으로

finally 블록이 _teardown_steps 목록 + run_teardown 호출로 축약된다. 단계
실패는 result.notes 에 teardown_failed:<단계> 로 남아 cycle manifest 에 실린다.
순서 불변식(첫 단계=입력 해제)을 테스트로 고정."
git show --stat --oneline HEAD
```

---

### Task 3: Adopt in `run_check_only_cycle` (fixes F1)

**Files:**
- Modify: `poc/workflow_3/monitor/cycle.py` (add `_check_teardown_steps`, rewrite `finally` at `:1641-1652`)
- Test: `poc/workflow_3/monitor/test_teardown.py` (append)

**Interfaces:**
- Consumes: `run_teardown` (Task 1); mirrors `_teardown_steps` (Task 2).
- Produces: `_check_teardown_steps(eqp_id, context, settings, *, input_blocked) -> list[tuple[str, Callable]]`. No `recording_stop` step — the check-only cycle does not record.

**This is the task that fixes F1.** The current order is `close_tool` → `close_alert_window` (unguarded) → `block_input(False)`, so a throw in `close_alert_window` leaves the engineer's mouse and keyboard blocked. The fix is the reordering, not merely the guarding.

- [ ] **Step 1: Write the failing test**

Append to `poc/workflow_3/monitor/test_teardown.py` (before the `if __name__` block):

```python
def test_check_only_teardown_unblocks_input_first():
    """F1 회귀 테스트 - check-only teardown 이 입력 해제를 마지막에 두면 안 된다.

    기존 순서는 close_tool -> close_alert_window(미보호) -> block_input(False) 라,
    close_alert_window 가 던지면 엔지니어 입력이 잠긴 채 남았다.
    """
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor.cycle import _check_teardown_steps

    settings = load_workflow3_settings()
    steps = _check_teardown_steps("EQP1", {}, settings, input_blocked=True)
    names = [n for n, _ in steps]
    assert names[0] == "input_unblock", names
    assert names == ["input_unblock", "close_tool", "close_alert"], names
    print("[OK] test_check_only_teardown_unblocks_input_first")


def test_check_only_teardown_survives_failing_close_alert():
    """close_alert 가 던져도 앞선 입력 해제는 이미 실행됐고, 헬퍼는 안 던진다."""
    from poc.workflow_3.monitor.teardown import run_teardown

    calls = []
    failures = run_teardown([
        ("input_unblock", lambda: calls.append("unblock")),
        ("close_tool", lambda: calls.append("close_tool")),
        ("close_alert", lambda: (_ for _ in ()).throw(OSError("popup gone"))),
    ])
    assert calls == ["unblock", "close_tool"], calls
    assert [n for n, _ in failures] == ["close_alert"], failures
    print("[OK] test_check_only_teardown_survives_failing_close_alert")
```

And add both to the `if __name__ == "__main__":` block:

```python
    test_check_only_teardown_unblocks_input_first()
    test_check_only_teardown_survives_failing_close_alert()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_teardown.py`
Expected: FAIL with `ImportError: cannot import name '_check_teardown_steps' from 'poc.workflow_3.monitor.cycle'`

- [ ] **Step 3: Write minimal implementation**

In `poc/workflow_3/monitor/cycle.py`, add `_check_teardown_steps` immediately above `def run_check_only_cycle(`:

```python
def _check_teardown_steps(eqp_id, context, settings, *, input_blocked):
    """점검(check-only) 사이클 teardown 단계 목록 - 녹화가 없어 3단계다.

    첫 단계는 **항상** 입력 해제다. 과거 이 사이클만 해제를 마지막에 둬서,
    close_alert_window 가 던지면 엔지니어 입력이 잠긴 채 남는 결함이 있었다
    (F1). 순서는 test_teardown.py 가 검사한다.
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"align_fail_check {eqp_id}")

    def _close_tool():
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)

    def _close_alert():
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return [
        ("input_unblock", _unblock),
        ("close_tool", _close_tool),
        ("close_alert", _close_alert),
    ]
```

Replace the entire `finally:` block of `run_check_only_cycle` (currently `cycle.py:1641-1652`) with:

```python
    finally:
        # 입력 해제를 **첫 단계**로 올린다 - 과거엔 close_alert_window 뒤에 있어
        # 그게 던지면 엔지니어 입력이 잠긴 채 남았다(F1).
        failures = run_teardown(
            _check_teardown_steps(eqp_id, context, settings, input_blocked=input_blocked),
            label=f"align_fail_check {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)
```

Note the behavior change this makes deliberate: input is now released *before* `close_tool` runs, whereas the old code held the block through the close. That is the intended fix — the automated GUI phase is over once the runner has returned, and holding the block through teardown is what created the lockout risk.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_teardown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/cycle.py poc/workflow_3/monitor/test_teardown.py
git commit -m "fix(workflow_3): check-only teardown 이 엔지니어 입력을 잠근 채 끝날 수 있던 결함

기존 순서 close_tool -> close_alert_window(미보호) -> block_input(False) 에서
close_alert_window 가 던지면 해제가 실행되지 않아 물리 마우스/키보드가 잠긴 채
남았다(탈출은 Ctrl+Alt+Del 뿐). 해제를 첫 단계로 올리고 run_teardown 으로
단계별 보호. 회귀 테스트로 순서 고정."
git show --stat --oneline HEAD
```

---

### Task 4: Adopt in `workflow_3e` abort cycle (fixes F4 there)

**Files:**
- Modify: `poc/workflow_3e/abort_cycle.py` (add `_abort_teardown_steps`, rewrite `finally` at `:217-225`)
- Test: `poc/workflow_3/monitor/test_teardown.py` (append)

**Interfaces:**
- Consumes: `run_teardown` from `poc.workflow_3.monitor.teardown` (workflow_3e imports workflow_3 one-way — this direction is correct).
- Produces: `_abort_teardown_steps(eqp_id, context, settings, *, input_blocked) -> list[tuple[str, Callable]]`.

This cycle already unblocks first, so it has no F1 lockout. What it has is F4 — the same copy-pasted shape whose "teardown always runs" property nothing enforces.

- [ ] **Step 1: Write the failing test**

Append to `poc/workflow_3/monitor/test_teardown.py` (before the `if __name__` block):

```python
def test_abort_cycle_teardown_unblocks_input_first():
    """workflow_3e abort 사이클도 같은 순서 규약을 따른다(F4 - 복제된 형태 방지)."""
    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3e.abort_cycle import _abort_teardown_steps

    settings = load_workflow3_settings()
    steps = _abort_teardown_steps("EQP1", {}, settings, input_blocked=True)
    names = [n for n, _ in steps]
    assert names == ["input_unblock", "close_tool", "close_alert"], names
    print("[OK] test_abort_cycle_teardown_unblocks_input_first")
```

And add to the `if __name__ == "__main__":` block:

```python
    test_abort_cycle_teardown_unblocks_input_first()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_teardown.py`
Expected: FAIL with `ImportError: cannot import name '_abort_teardown_steps' from 'poc.workflow_3e.abort_cycle'`

- [ ] **Step 3: Write minimal implementation**

In `poc/workflow_3e/abort_cycle.py`, add near the existing `from poc.workflow_3.monitor.notify import close_alert_window` import:

```python
from poc.workflow_3.monitor.teardown import run_teardown
```

Add `_abort_teardown_steps` immediately above `def run_abort_cycle(`:

```python
def _abort_teardown_steps(eqp_id, context, settings, *, input_blocked):
    """abort 사이클 teardown 단계 목록 - workflow_3 의 두 사이클과 같은 규약.

    첫 단계는 **항상** 입력 해제. 전제조건은 목록에서 빼지 않고 클로저 안에서
    판정한다(목록 길이/순서 고정 -> 순서 테스트 가능).
    """

    def _unblock():
        if input_blocked:
            block_input(False, debug_label=f"measurement_abort {eqp_id}")

    def _close_tool():
        if context.get("tool_window") is not None and CLOSE_TOOL_AVAILABLE:
            close_tool(eqp_id)

    def _close_alert():
        close_alert_window(timeout_sec=settings.alert_close_timeout_sec)

    return [
        ("input_unblock", _unblock),
        ("close_tool", _close_tool),
        ("close_alert", _close_alert),
    ]
```

Replace the `finally:` block at `abort_cycle.py:217-225` with:

```python
    finally:
        failures = run_teardown(
            _abort_teardown_steps(eqp_id, context, settings, input_blocked=input_blocked),
            label=f"measurement_abort {eqp_id}",
        )
        result.notes.extend(f"teardown_failed:{n}: {e}" for n, e in failures)
```

Before writing that last line, confirm the abort cycle's result object has a `notes` list. Run:

```bash
grep -n "notes" poc/workflow_3e/abort_cycle.py | head
```

If it reuses `CycleResult` (which has `notes: list[str]`), keep the line. If its result type has no `notes` field, drop the `result.notes.extend(...)` line — `run_teardown` already prints each failure, so the audit trail is not lost.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_teardown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3e/abort_cycle.py poc/workflow_3/monitor/test_teardown.py
git commit -m "refactor(workflow_3e): abort 사이클 teardown 을 run_teardown 으로 통일

세 사이클(alarm/check-only/abort)이 같은 teardown 규약을 공유한다. 복제된
finally 형태가 각자 드리프트하던 F4 를 구조로 닫는다."
git show --stat --oneline HEAD
```

---

### Task 5: Failure cooldown + per-tool guard in the production monitor

**Files:**
- Modify: `poc/workflow_3/config.py` (add field ~line 59, add env read ~line 201)
- Modify: `poc/workflow_3/workflow_3_config.example.py`
- Modify: `poc/workflow_3/workflow_3_config_loader.py`
- Modify: `poc/workflow_3/monitor/align_fail_monitor.py:321-378`
- Test: `poc/workflow_3/monitor/test_failure_cooldown.py`

**Interfaces:**
- Consumes: existing `occupied_cooldown` dict (`dict[str, float]`, eqp_id → expiry epoch), `CycleResult` from `monitor/cycle.py`.
- Produces: `Workflow3Settings.failure_retry_cooldown_sec: float`; module-level `_cycle_failed(cycle) -> bool` in `align_fail_monitor.py`, reused verbatim by Task 6.

**Two facts verified in the codebase that this task depends on — do not re-derive:**
1. `_exec_run_correction` (`cycle.py:468-475`) returns `"success"` regardless of `outcome.status`, and only sets `failed_step`/`correction_error` on an exception. So a **correction fallback never trips this cooldown.**
2. `CycleResult.run_status` defaults to `"not_run"` and `failed_step` to `""` (`cycle.py:120-129`). The existing `test_occupied_popup.py` fake cycle sets neither, so it stays green.

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/monitor/test_failure_cooldown.py`:

```python
"""실패 cooldown + tool 별 가드 스모크 테스트 (F2/F5).

RCS/office 없이 도는 단위 테스트다. process_fail_rows 의 의존성을 stub 으로
바꾸고 CycleResult 를 직접 만들어 분기를 검사한다.

`uv run python poc/workflow_3/monitor/test_failure_cooldown.py` 로 직접 실행.
"""

import time

from poc.workflow_3.monitor import align_fail_monitor as afm
from poc.workflow_3.monitor.cycle import CycleResult


def _swap(state, module, name, fn):
    """module.name 을 fn 으로 교체하고 원복용으로 저장."""
    state[(module, name)] = getattr(module, name)
    setattr(module, name, fn)


def _restore(state):
    for (module, name), orig in state.items():
        setattr(module, name, orig)


def _stub_deps(state, module, cycle_fn, cycle_attr="run_alarm_cycle"):
    _swap(state, module, "append_alarm_record", lambda *a, **k: None)
    _swap(state, module, "notify_align_fail_popup", lambda *a, **k: None)
    _swap(state, module, "gather_success_async", lambda *a, **k: None)
    _swap(state, module, "gather_rcp_msr", lambda *a, **k: None)
    _swap(state, module, "append_cycle_manifest", lambda *a, **k: None)
    _swap(state, module, cycle_attr, cycle_fn)


def _cycle_returning(**fields):
    def _fake(eqp_id, recipe_id, settings, tag=None):
        r = CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "")
        for k, v in fields.items():
            setattr(r, k, v)
        return r
    return _fake


def test_error_cycle_registers_cooldown_and_skips_active():
    """run_status='error' 면 cooldown 등록 + active 미등록(재시도는 만료 후)."""
    state = {}
    _stub_deps(state, afm, _cycle_returning(run_status="error"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP1", "recipe_id": "C/R"}]

        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP1" not in active, active
        assert "EQP1" in cooldown, cooldown

        # cooldown 중에는 재처리하지 않는다.
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 0, n

        # 만료되면 다시 시도한다.
        cooldown["EQP1"] = time.time() - 1
        n = afm.process_fail_rows(fails, active, settings, cooldown)
        assert n == 1, n
    finally:
        _restore(state)
    print("[OK] test_error_cycle_registers_cooldown_and_skips_active")


def test_aborted_cycle_with_failed_step_registers_cooldown():
    """runner 중단(failed_step 세팅)도 실패로 보고 cooldown 을 건다."""
    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="aborted", failed_step="wait_tool_window"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        afm.process_fail_rows(
            [{"eqp_id": "EQP2", "recipe_id": "C/R"}], active, settings, cooldown)
        assert "EQP2" in cooldown and "EQP2" not in active
    finally:
        _restore(state)
    print("[OK] test_aborted_cycle_with_failed_step_registers_cooldown")


def test_correction_fallback_does_not_register_cooldown():
    """정상 수행 + fallback outcome 은 실패가 아니다 - 과잉 트리거 방지.

    _exec_run_correction 은 outcome.status 와 무관하게 success 를 반환하므로
    (cycle.py:468-475) fallback 은 run_status='completed' 로 온다.
    """
    state = {}
    _stub_deps(state, afm, _cycle_returning(
        run_status="completed", outcome_status="fallback_live_search"))
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        afm.process_fail_rows(
            [{"eqp_id": "EQP3", "recipe_id": "C/R"}], active, settings, cooldown)
        assert "EQP3" in active, active
        assert "EQP3" not in cooldown, cooldown
    finally:
        _restore(state)
    print("[OK] test_correction_fallback_does_not_register_cooldown")


def test_raising_tool_does_not_skip_remaining_tools():
    """tool 1대가 던져도 같은 poll 의 나머지 tool 은 처리된다(F5) + 던진 쪽은 cooldown."""
    state = {}
    seen = []

    def _cycle(eqp_id, recipe_id, settings, tag=None):
        seen.append(eqp_id)
        if eqp_id == "EQP_BAD":
            raise RuntimeError("boom")
        return CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "",
                           run_status="completed")

    _stub_deps(state, afm, _cycle)
    try:
        settings = afm.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP_BAD", "recipe_id": "C/R"},
                 {"eqp_id": "EQP_OK", "recipe_id": "C/R"}]
        afm.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP_OK" in seen, seen
        assert "EQP_OK" in active, active
        assert "EQP_BAD" in cooldown, cooldown
        assert "EQP_BAD" not in active, active
    finally:
        _restore(state)
    print("[OK] test_raising_tool_does_not_skip_remaining_tools")


if __name__ == "__main__":
    test_error_cycle_registers_cooldown_and_skips_active()
    test_aborted_cycle_with_failed_step_registers_cooldown()
    test_correction_fallback_does_not_register_cooldown()
    test_raising_tool_does_not_skip_remaining_tools()
    print("\n[OK] 실패 cooldown / tool 가드 테스트 통과")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_failure_cooldown.py`
Expected: FAIL — the first test fails at `assert "EQP1" in cooldown` because no failure cooldown exists yet.

- [ ] **Step 3: Write minimal implementation**

**3a.** In `poc/workflow_3/config.py`, add the field directly after `occupied_retry_cooldown_sec` (line 59):

```python
    # 점유 외 사유로 사이클이 실패한 tool 의 재시도 유예(초). 없으면 매 poll 재시도해
    # 직렬화된 단일 RCS 커서를 독점하고 다른 알람을 굶긴다(F2).
    failure_retry_cooldown_sec: float = 300.0
```

And the env read directly after the `occupied_retry_cooldown_sec=` line (~201):

```python
        failure_retry_cooldown_sec=env_float("ALIGN_FAIL_FAILURE_COOLDOWN_SEC", 300.0),
```

**3b.** In `poc/workflow_3/workflow_3_config.example.py`, next to the other cycle toggles:

```python
FAILURE_COOLDOWN_SEC = None  # 실패 tool 재시도 유예(초). None=기본 300.
```

**3c.** In `poc/workflow_3/workflow_3_config_loader.py`, add to `_CONST_TO_ENV` under the `# [1]` or a new `# [7] 재시도 정책` group:

```python
    # [7] 재시도 정책
    ("FAILURE_COOLDOWN_SEC", "ALIGN_FAIL_FAILURE_COOLDOWN_SEC"),
```

**3d.** In `poc/workflow_3/monitor/align_fail_monitor.py`, add this helper above `process_fail_rows`:

```python
def _cycle_failed(cycle) -> bool:
    """사이클이 정상 완료되지 못했는지 - 실패 cooldown 트리거 판정.

    True: 예외로 끝났거나(run_status='error') runner 가 step 실패로 중단(failed_step).
    False: 정상 수행. **correction fallback 은 실패가 아니다** - _exec_run_correction
    은 outcome.status 와 무관하게 success 를 반환하므로(cycle.py:468-475) fallback 은
    run_status='completed' 로 온다. 엔지니어 인계는 정상 경로이며 이미 알람 해제까지
    active_tools 에 머문다.
    """
    return cycle.run_status == "error" or bool(cycle.failed_step)
```

Then restructure the per-tool loop body. Replace the body of `for eqp_id in sorted(new_tools):` (currently lines 321-378) so the whole per-tool block is guarded and the failure branch is added:

```python
    for eqp_id in sorted(new_tools):
        try:
            info = by_tool[eqp_id]
            alarm_time = str(info["alarm_time"] or "")

            print(
                f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
                f"ALID={info['alid']}, RECIPE_ID={info['recipe_id']}, "
                f"LOT_TYPE={info['lot_type_cd']}, 시각={alarm_time}"
            )
            append_alarm_record(
                eqp_id, alarm_time, info["alarm_name"], info["alid"],
                recipe_id=info["recipe_id"],
                operation_desc=info["operation_desc"],
                lot_type_cd=info["lot_type_cd"],
            )
            if settings.popup_enabled:
                notify_align_fail_popup(
                    eqp_id, alarm_time, info["alarm_name"],
                    recipe_id=info["recipe_id"],
                    operation_desc=info["operation_desc"],
                    lot_type_cd=info["lot_type_cd"],
                    timeout_sec=settings.popup_timeout_sec,
                )

            # consensus 재료 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
            gather_success_async(eqp_id, info["recipe_id"], settings)

            # rcp 1차 입력 — 사이클이 assets 를 읽기 전에 **동기**(bounded) 다운로드.
            gather_rcp_msr(eqp_id, info["recipe_id"], settings)

            # 알람별 사이클 — RECIPE_ID 유무와 무관하게 접속+녹화는 수행.
            if settings.cycle_enabled:
                cycle = run_alarm_cycle(
                    eqp_id,
                    info["recipe_id"],
                    settings,
                    tag=_alarm_time_to_tag(info["utc9"]),
                )
            else:
                cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
                cycle.run_status = "cycle_disabled"

            append_cycle_manifest(info, cycle)

            # 점유(select)로 포기: active 에 넣지 않고 cooldown 등록 → 만료 후 재시도.
            if cycle.failure_class in _OCCUPIED_FAILURE_CLASSES:
                occupied_cooldown[eqp_id] = time.time() + settings.occupied_retry_cooldown_sec
                print(
                    f"[INFO] EQP_ID={eqp_id} 점유(select) 추정 - active 미등록, "
                    f"{settings.occupied_retry_cooldown_sec:.0f}s 후 재시도"
                )
            elif _cycle_failed(cycle):
                # 실패 tool 을 매 poll 재시도하면 단일 RCS 커서를 독점한다(F2).
                occupied_cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
                print(
                    f"[WARNING] EQP_ID={eqp_id} 사이클 실패(status={cycle.run_status}, "
                    f"step={cycle.failed_step or '-'}) - active 미등록, "
                    f"{settings.failure_retry_cooldown_sec:.0f}s 후 재시도"
                )
            else:
                active_tools.add(eqp_id)
            newly_handled += 1
        except Exception as exc:
            # tool 1대의 예외가 같은 poll 의 나머지 tool 을 건너뛰게 하면 안 된다(F5).
            # 던진 tool 은 cooldown 에 넣어 다음 poll 에 같은 예외를 반복하지 않게 한다.
            occupied_cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
            print(f"[ERROR] EQP_ID={eqp_id} 처리 예외 - 나머지 tool 계속: {exc}")
            log_work2_event(
                component=LOG_COMPONENT, message="tool_process_error", level="error",
                eqp_id=eqp_id, error=str(exc),
            )
```

Before writing the `log_work2_event` call, confirm `log_work2_event` and `LOG_COMPONENT` are already imported in `align_fail_monitor.py`:

```bash
grep -n "log_work2_event\|LOG_COMPONENT" poc/workflow_3/monitor/align_fail_monitor.py | head -3
```

If either is missing, add the import `from poc.workflow_3.logger import log_work2_event` and define `LOG_COMPONENT = "align_fail_monitor"` near the top, matching the pattern in `monitor/cycle.py`.

Finally, update the `process_fail_rows` docstring to say the cooldown dict now covers both occupied and failed tools.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
uv run python poc/workflow_3/monitor/test_teardown.py
```
Expected: all three PASS. `test_occupied_popup.py` passing unmodified is the check that the cooldown dict's value type was not churned.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/workflow_3_config.example.py \
        poc/workflow_3/workflow_3_config_loader.py \
        poc/workflow_3/monitor/align_fail_monitor.py \
        poc/workflow_3/monitor/test_failure_cooldown.py
git commit -m "feat(workflow_3): 실패 tool cooldown + tool 별 가드 (F2/F5)

F2: 점유 외 사유로 실패한 tool 도 cooldown 을 받는다. 없으면 매 poll 재시도해
직렬 단일 RCS 커서를 독점하고 다른 알람을 굶겼다. 기존 occupied_cooldown dict 를
그대로 재사용(값 타입 float 유지 - test_occupied_popup 무수정 통과).
트리거는 run_status='error' 또는 failed_step - correction fallback 은 정상 경로라
제외(_exec_run_correction 은 outcome 과 무관하게 success 반환).

F5: tool 처리 본문을 try 로 감싸 1대의 예외가 같은 poll 의 나머지를 건너뛰지
않게 한다. 던진 tool 도 cooldown.

신규 env ALIGN_FAIL_FAILURE_COOLDOWN_SEC (기본 300)."
git show --stat --oneline HEAD
```

---

### Task 6: Same cooldown + guard in the check-only monitor

**Files:**
- Modify: `poc/workflow_3/monitor/align_fail_monitor_only_check.py:114` (its own `process_fail_rows` copy) and its call site at `:306`
- Test: `poc/workflow_3/monitor/test_failure_cooldown.py` (append)

**Interfaces:**
- Consumes: the `_cycle_failed` pattern from Task 5 (copy it into this module — do not import a private name across monitors; the two `process_fail_rows` copies are deliberately independent).
- Produces: `process_fail_rows(fails, active_tools, settings, cooldown=None)` — a **new fourth parameter** on the check-only copy, defaulting to `None` so the existing call at `:306` keeps working before it is updated in this task.

- [ ] **Step 1: Write the failing test**

Append to `poc/workflow_3/monitor/test_failure_cooldown.py` (before the `if __name__` block):

```python
def test_check_only_monitor_registers_failure_cooldown():
    """check-only 모니터도 같은 규약 - 실패 tool 은 cooldown, 나머지는 계속(F2/F5)."""
    from poc.workflow_3.monitor import align_fail_monitor_only_check as afmc

    state = {}
    seen = []

    def _cycle(eqp_id, recipe_id, settings, tag=None):
        seen.append(eqp_id)
        if eqp_id == "EQP_BAD":
            raise RuntimeError("boom")
        return CycleResult(eqp_id=eqp_id, recipe_id=recipe_id, tag=tag or "",
                           run_status="completed")

    _stub_deps(state, afmc, _cycle, cycle_attr="run_check_only_cycle")
    try:
        settings = afmc.load_workflow3_settings()
        active, cooldown = set(), {}
        fails = [{"eqp_id": "EQP_BAD", "recipe_id": "C/R"},
                 {"eqp_id": "EQP_OK", "recipe_id": "C/R"}]
        afmc.process_fail_rows(fails, active, settings, cooldown)
        assert "EQP_OK" in seen and "EQP_OK" in active, (seen, active)
        assert "EQP_BAD" in cooldown and "EQP_BAD" not in active, (cooldown, active)
    finally:
        _restore(state)
    print("[OK] test_check_only_monitor_registers_failure_cooldown")
```

Add to the `if __name__ == "__main__":` block:

```python
    test_check_only_monitor_registers_failure_cooldown()
```

The check-only module additionally calls `send_detection_notify_async`, which the production one does not. Make `_stub_deps` tolerant of per-module differences by replacing its body with:

```python
def _stub_deps(state, module, cycle_fn, cycle_attr="run_alarm_cycle"):
    for name in ("append_alarm_record", "notify_align_fail_popup",
                 "send_detection_notify_async", "gather_success_async",
                 "gather_rcp_msr", "append_cycle_manifest"):
        if hasattr(module, name):
            _swap(state, module, name, lambda *a, **k: None)
    _swap(state, module, cycle_attr, cycle_fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_failure_cooldown.py`
Expected: FAIL — `process_fail_rows() takes 3 positional arguments but 4 were given`, or the cooldown assertion fails.

- [ ] **Step 3: Write minimal implementation**

**3a.** Add this helper above `process_fail_rows` in `poc/workflow_3/monitor/align_fail_monitor_only_check.py`. It is a deliberate copy of the production one — the two `process_fail_rows` implementations are independent by design, and importing a private name across monitors would couple them:

```python
def _cycle_failed(cycle) -> bool:
    """사이클이 정상 완료되지 못했는지 - 실패 cooldown 트리거 판정.

    True: 예외로 끝났거나(run_status='error') runner 가 step 실패로 중단(failed_step).
    점유(select) 팝업도 step 실패로 오므로 여기 포함된다 - 점검 모니터는 production
    처럼 점유를 따로 분기하지 않고 같은 cooldown 으로 처리한다.
    False: 정상 수행. correction 계열 fallback 은 실패가 아니다(점검 사이클은 보정
    actuation 자체가 없다).
    """
    return cycle.run_status == "error" or bool(cycle.failed_step)
```

**3b.** Replace the whole `process_fail_rows` (currently lines 114-193) with:

```python
def process_fail_rows(
    fails,
    active_tools: set[str],
    settings: Workflow3Settings,
    cooldown: dict | None = None,
) -> int:
    """EQP_ID 기준 edge-triggered 로 신규 알람마다 점검 사이클을 수행한다.

    production `align_fail_monitor.process_fail_rows` 와 동일한 edge-trigger 규약
    이되, 알람별 사이클만 `run_check_only_cycle`(접속 → 캡처 → 닫기)로 바뀐다.

    `cooldown` 은 {eqp_id: 재시도 가능 epoch} - 사이클이 실패한 tool 을 매 poll
    재시도하면 직렬화된 단일 RCS 커서를 독점해 다른 알람을 굶긴다(F2). tool 1대의
    예외가 같은 poll 의 나머지를 건너뛰지 않게 본문은 tool 별로 보호한다(F5).

    `active_tools`/`cooldown` 은 in-place 로 갱신된다. 새로 처리한 개수를 반환.
    """
    if cooldown is None:
        cooldown = {}
    by_tool = _collapse_rows_by_tool(fails)
    current_tools = set(by_tool.keys())

    # cooldown 만료/알람해제 정리 → 남은 것은 이번 poll 에서 건너뛴다.
    now = time.time()
    for eqp_id in list(cooldown):
        if eqp_id not in current_tools or now >= cooldown[eqp_id]:
            del cooldown[eqp_id]
    cooling = current_tools & set(cooldown)
    for eqp_id in sorted(cooling):
        print(f"[INFO] EQP_ID={eqp_id} cooldown 중 - 이번 poll 재시도 건너뜀")

    new_tools = current_tools - active_tools - cooling
    cleared_tools = active_tools - current_tools

    for eqp_id in sorted(cleared_tools):
        print(f"[INFO] Align Fail 해제: EQP_ID={eqp_id}")
    active_tools.difference_update(cleared_tools)

    newly_handled = 0
    for eqp_id in sorted(new_tools):
        try:
            info = by_tool[eqp_id]
            alarm_time = str(info["alarm_time"] or "")

            print(
                f"[WARNING] Align Fail 감지: EQP_ID={eqp_id}, "
                f"ALID={info['alid']}, RECIPE_ID={info['recipe_id']}, "
                f"LOT_TYPE={info['lot_type_cd']}, 시각={alarm_time}"
            )
            append_alarm_record(
                eqp_id, alarm_time, info["alarm_name"], info["alid"],
                recipe_id=info["recipe_id"],
                operation_desc=info["operation_desc"],
                lot_type_cd=info["lot_type_cd"],
            )
            if settings.popup_enabled:
                notify_align_fail_popup(
                    eqp_id, alarm_time, info["alarm_name"],
                    recipe_id=info["recipe_id"],
                    operation_desc=info["operation_desc"],
                    lot_type_cd=info["lot_type_cd"],
                    timeout_sec=settings.popup_timeout_sec,
                )

            # 감지 시점 cube rich notification — 점검 모드는 보정 actuation 이 없어
            # CorrectionOutcome 을 만들지 않으므로 detection-time 변형을 쓴다.
            send_detection_notify_async(
                eqp_id, info["recipe_id"], enabled=settings.rich_notify_enabled,
            )

            # 과거 데이터 수집 — recipe 최근 성공(S) 이미지 stage (비차단 best-effort).
            gather_success_async(eqp_id, info["recipe_id"], settings)

            # rcp 1차 입력 — cycle 이 assets(feasibility)를 읽기 전에 **동기** 다운로드.
            # (Task 7 이 여기에 timeout_sec= 를 추가한다.)
            gather_rcp_msr(eqp_id, info["recipe_id"], settings)

            # 점검 전용 사이클 — 접속 → 첫 화면 1장 캡처 → tool 닫기 (보정/녹화 없음).
            if settings.cycle_enabled:
                cycle = run_check_only_cycle(
                    eqp_id,
                    info["recipe_id"],
                    settings,
                    tag=_alarm_time_to_tag(info["utc9"]),
                )
            else:
                cycle = CycleResult(eqp_id=eqp_id, recipe_id=info["recipe_id"], tag="")
                cycle.run_status = "cycle_disabled"

            append_cycle_manifest(info, cycle)

            if _cycle_failed(cycle):
                cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
                print(
                    f"[WARNING] EQP_ID={eqp_id} 점검 사이클 실패(status={cycle.run_status}, "
                    f"step={cycle.failed_step or '-'}) - active 미등록, "
                    f"{settings.failure_retry_cooldown_sec:.0f}s 후 재시도"
                )
            else:
                active_tools.add(eqp_id)
            newly_handled += 1
        except Exception as exc:
            # tool 1대의 예외가 같은 poll 의 나머지 tool 을 건너뛰게 하면 안 된다(F5).
            cooldown[eqp_id] = time.time() + settings.failure_retry_cooldown_sec
            print(f"[ERROR] EQP_ID={eqp_id} 처리 예외 - 나머지 tool 계속: {exc}")

    return newly_handled
```

The `gather_rcp_msr` call above deliberately has **no** `timeout_sec` yet — that parameter does not exist until Task 7, which adds it at both monitor call sites in its Step 3e.

**3c.** Confirm `time` is imported in this module (the pruning block needs it):

```bash
grep -n "^import time\|^import " poc/workflow_3/monitor/align_fail_monitor_only_check.py | head -5
```

If absent, add `import time` with the other stdlib imports.

**3d.** Update the call site. Find the loop setup and the call:

```bash
grep -n "active_tools\|process_fail_rows" poc/workflow_3/monitor/align_fail_monitor_only_check.py
```

Next to where `active_tools: set[str] = set()` is initialized in the monitor loop, add:

```python
    cooldown: dict = {}  # {eqp_id: 재시도 가능 epoch} — 사이클 실패로 쉬는 tool.
```

and change the call at `:306` to:

```python
                count = process_fail_rows(fails, active_tools, settings, cooldown)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/monitor/align_fail_monitor_only_check.py \
        poc/workflow_3/monitor/test_failure_cooldown.py
git commit -m "feat(workflow_3): check-only 모니터에도 실패 cooldown + tool 가드

check-only 는 자체 process_fail_rows 사본을 갖고 cooldown 인자조차 없었다.
오피스에서 무인 실행되므로 production 과 같은 규약을 적용한다(F2/F5)."
git show --stat --oneline HEAD
```

---

### Task 7: Bound the synchronous office download

**Files:**
- Modify: `poc/workflow_3/config.py` (field + env read)
- Modify: `poc/workflow_3/workflow_3_config.example.py`, `poc/workflow_3/workflow_3_config_loader.py`
- Modify: `poc/workflow_3/monitor/rcp_msr_gather.py`
- Modify: `poc/workflow_3/monitor/align_fail_monitor.py` (pass the timeout), `poc/workflow_3/monitor/align_fail_monitor_only_check.py` (if it calls `gather_rcp_msr`)
- Test: `poc/workflow_3/monitor/test_rcp_gather_timeout.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-6.
- Produces: `gather_rcp_msr(eqp_id, recipe_id, settings, *, include_msr=False, timeout_sec=None) -> bool`. `timeout_sec=None` means **wait indefinitely** (current behavior, preserved for the offline bench). `Workflow3Settings.rcp_gather_timeout_sec: float`.

**Critical constraint:** `poc/workflow_3/monitor/fetch_msr_offline.py:31` calls `gather_rcp_msr(..., include_msr=True)` for the offline bench, where a multi-minute download is legitimate. That call must keep its unbounded behavior — which is why `timeout_sec` defaults to `None` rather than to the setting.

- [ ] **Step 1: Write the failing test**

Create `poc/workflow_3/monitor/test_rcp_gather_timeout.py`:

```python
"""rcp 동기 다운로드의 bounded 대기 + in-flight 가드 테스트 (F3).

office downloader 없이 도는 단위 테스트다 - 모듈의 _DOWNLOADER 를 stub 으로
바꿔 느린/정상 다운로드를 흉내낸다.

`uv run python poc/workflow_3/monitor/test_rcp_gather_timeout.py` 로 직접 실행.
"""

import threading
import time
from types import SimpleNamespace

from poc.workflow_3.config import load_workflow3_settings
from poc.workflow_3.monitor import rcp_msr_gather as rmg


def _install_downloader(fn):
    """모듈의 downloader 를 교체하고 원복 함수를 돌려준다."""
    orig_dl = rmg._DOWNLOADER
    orig_flag = rmg.RCP_MSR_DOWNLOADER_AVAILABLE
    rmg._DOWNLOADER = SimpleNamespace(download_rcp_msr=fn)
    rmg.RCP_MSR_DOWNLOADER_AVAILABLE = True

    def _restore():
        rmg._DOWNLOADER = orig_dl
        rmg.RCP_MSR_DOWNLOADER_AVAILABLE = orig_flag
        with rmg._IN_FLIGHT_LOCK:
            rmg._IN_FLIGHT.clear()
    return _restore


def test_slow_download_returns_within_bound():
    """timeout 을 넘긴 다운로드는 bound 안에서 False 로 돌아온다(무한 정지 금지)."""
    def _slow(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        time.sleep(5.0)
        return 1

    restore = _install_downloader(_slow)
    try:
        settings = load_workflow3_settings()
        started = time.time()
        ok = rmg.gather_rcp_msr("EQP1", "C/R1", settings, timeout_sec=0.5)
        elapsed = time.time() - started
        assert ok is False, ok
        assert elapsed < 3.0, elapsed      # 5초 다운로드에 묶이지 않았다.
    finally:
        restore()
    print("[OK] test_slow_download_returns_within_bound")


def test_fast_download_returns_true():
    """timeout 안에 끝나면 True."""
    def _fast(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        return 3

    restore = _install_downloader(_fast)
    try:
        settings = load_workflow3_settings()
        assert rmg.gather_rcp_msr("EQP2", "C/R2", settings, timeout_sec=5.0) is True
    finally:
        restore()
    print("[OK] test_fast_download_returns_true")


def test_in_flight_guard_skips_concurrent_same_recipe():
    """같은 recipe 의 gather 가 아직 도는 중이면 새 gather 를 fire 하지 않는다.

    timeout 만 넣고 이 가드가 없으면, 시간 초과된 스레드가 계속 쓰는 동안 새
    스레드가 같은 dest_dir 에 겹쳐 써서 '보이는 정지'가 '조용한 부분읽기 경쟁'
    으로 바뀐다.
    """
    calls = []
    release = threading.Event()

    def _blocking(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        calls.append(eqp_id)
        release.wait(5.0)
        return 1

    restore = _install_downloader(_blocking)
    try:
        settings = load_workflow3_settings()
        # 1차: timeout 으로 포기하지만 스레드는 계속 돈다.
        rmg.gather_rcp_msr("EQP3", "C/R3", settings, timeout_sec=0.3)
        assert calls == ["EQP3"], calls
        # 2차: 같은 recipe - 아직 진행 중이므로 새 fire 없음.
        rmg.gather_rcp_msr("EQP3", "C/R3", settings, timeout_sec=0.3)
        assert calls == ["EQP3"], calls
    finally:
        release.set()
        time.sleep(0.1)
        restore()
    print("[OK] test_in_flight_guard_skips_concurrent_same_recipe")


def test_exception_in_download_returns_false():
    """다운로드 예외는 삼키고 False - 모니터 루프를 죽이지 않는다(기존 계약 유지)."""
    def _boom(eqp_id, recipe_id, *, dest_dir, include_msr=False):
        raise RuntimeError("network down")

    restore = _install_downloader(_boom)
    try:
        settings = load_workflow3_settings()
        assert rmg.gather_rcp_msr("EQP4", "C/R4", settings, timeout_sec=5.0) is False
    finally:
        restore()
    print("[OK] test_exception_in_download_returns_false")


if __name__ == "__main__":
    test_slow_download_returns_within_bound()
    test_fast_download_returns_true()
    test_in_flight_guard_skips_concurrent_same_recipe()
    test_exception_in_download_returns_false()
    print("\n[OK] rcp gather timeout / in-flight 가드 테스트 통과")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python poc/workflow_3/monitor/test_rcp_gather_timeout.py`
Expected: FAIL with `AttributeError: module ... has no attribute '_IN_FLIGHT_LOCK'` (raised in `_install_downloader`'s restore path), or `TypeError: gather_rcp_msr() got an unexpected keyword argument 'timeout_sec'`.

- [ ] **Step 3: Write minimal implementation**

**3a.** In `poc/workflow_3/config.py`, add the field after `rcp_msr_gather_enabled` (line 99):

```python
    # 동기 rcp 다운로드 대기 상한(초). office 호출이 걸려도 모니터가 무한 정지하지
    # 않게 한다(F3). 초과 시 받은 만큼으로 진행 - assets 부분/부재 가능성 있음.
    rcp_gather_timeout_sec: float = 60.0
```

And the env read after the `rcp_msr_gather_enabled=` line (~210):

```python
        rcp_gather_timeout_sec=env_float("ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC", 60.0),
```

**3b.** In `workflow_3_config.example.py`, next to `GATHER_RCP_MSR`:

```python
RCP_GATHER_TIMEOUT_SEC = None  # rcp 동기 다운로드 대기 상한(초). None=기본 60.
```

**3c.** In `workflow_3_config_loader.py`, add to the `# [6]` group in `_CONST_TO_ENV`:

```python
    ("RCP_GATHER_TIMEOUT_SEC", "ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC"),
```

**3d.** In `poc/workflow_3/monitor/rcp_msr_gather.py`, add `import threading` at the top with the other imports, and these module-level globals next to `_DOWNLOADER`:

```python
# recipe_id -> Thread. 진행 중인 gather 가 있으면 새로 fire 하지 않는다 - timeout 으로
# 포기한 스레드가 같은 dest_dir 에 계속 쓰는 동안 새 스레드가 겹쳐 쓰면 부분읽기
# 경쟁이 난다(success_gather 와 같은 규약).
_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT: dict = {}
```

Replace the body of `gather_rcp_msr` with:

```python
def gather_rcp_msr(
    eqp_id, recipe_id, settings: Workflow3Settings, *,
    include_msr: bool = False, timeout_sec=None,
) -> bool:
    """recipe 의 rcp 입력 이미지를 align_images 트리로 **동기** 다운로드한다.

    동기 계약은 유지된다 - cycle 이 assets(feasibility/보정)를 읽기 전에 디스크
    적재를 보장해야 하기 때문이다. 다만 대기는 bounded 다: daemon thread 로 돌리고
    timeout_sec 만큼만 join 한다(success_gather.wait_for_gather 와 같은 관용구).

    timeout_sec=None 이면 무한 대기(기존 동작) - 오프라인 벤치 fetch_msr_offline.py
    는 수 분짜리 msr 다운로드가 정상이라 상한을 두지 않는다. 모니터는
    settings.rcp_gather_timeout_sec 를 넘긴다.

    반환 True = 시간 안에 예외 없이 끝남. False = 게이트 미충족/예외/시간 초과.
    시간 초과 시 스레드는 계속 돌지만 루프는 진행한다 - assets 가 없거나 부분일 수
    있어 feasibility 가 '보정 불가' 오판을 낼 수 있다(알람 1건의 bounded 오답이
    전체 루프 무한 정지보다 낫다는 판단).
    """
    if not settings.rcp_msr_gather_enabled or not recipe_id or not RCP_MSR_DOWNLOADER_AVAILABLE:
        return False

    # recipe_id = '<class>/<recipe>' 라 ALIGN_IMAGES_DIR/<eqp>/<class>/<recipe> 로 3단 중첩.
    dest_dir = ALIGN_IMAGES_DIR / eqp_id / recipe_id
    outcome = {"ok": False}

    def _run():
        try:
            n_images = _DOWNLOADER.download_rcp_msr(
                eqp_id, recipe_id, dest_dir=dest_dir, include_msr=include_msr
            )
            kind = "rcp+msr" if include_msr else "rcp"
            print(f"[INFO] {kind} 다운로드 완료: EQP_ID={eqp_id} recipe={recipe_id} "
                  f"images={n_images} dest={dest_dir}")
            outcome["ok"] = True
        except Exception as exc:
            print(f"[WARNING] rcp/msr 다운로드 예외: EQP_ID={eqp_id} recipe={recipe_id} error={exc}")
            log_work2_event(
                component=LOG_COMPONENT, message="gather_error", level="warning",
                eqp_id=eqp_id, recipe_id=recipe_id, error=str(exc),
            )

    key = recipe_id   # eqp 무관 - 같은 recipe 는 같은 dest 하위를 쓴다.
    with _IN_FLIGHT_LOCK:
        dead = [k for k, t in _IN_FLIGHT.items() if not t.is_alive()]
        for k in dead:
            del _IN_FLIGHT[k]
        if key in _IN_FLIGHT and _IN_FLIGHT[key].is_alive():
            print(f"[INFO] rcp gather 이미 진행 중(skip): EQP_ID={eqp_id} recipe={recipe_id}")
            return False
        thread = threading.Thread(target=_run, daemon=True)
        _IN_FLIGHT[key] = thread
        # start 도 lock 안에서 - 등록과 시작 사이 틈에 다른 호출자의 prune 이
        # 미시작 thread 를 지우고 중복 fire 하는 창을 닫는다.
        thread.start()

    thread.join(timeout_sec)
    if thread.is_alive():
        print(f"[WARNING] rcp 다운로드 시간 초과({timeout_sec}s) - 받은 만큼으로 진행: "
              f"EQP_ID={eqp_id} recipe={recipe_id}")
        return False
    return outcome["ok"]
```

**3e.** In `poc/workflow_3/monitor/align_fail_monitor.py`, pass the bound at the call site inside the per-tool block:

```python
            gather_rcp_msr(eqp_id, info["recipe_id"], settings,
                           timeout_sec=settings.rcp_gather_timeout_sec)
```

And the same change in `poc/workflow_3/monitor/align_fail_monitor_only_check.py`, where Task 6 left the call without a timeout:

```python
            gather_rcp_msr(eqp_id, info["recipe_id"], settings,
                           timeout_sec=settings.rcp_gather_timeout_sec)
```

Leave `fetch_msr_offline.py:31` **unchanged** — it must stay unbounded (a multi-minute `include_msr=True` bench download is legitimate, and `timeout_sec` defaults to `None`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python poc/workflow_3/monitor/test_rcp_gather_timeout.py
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add poc/workflow_3/config.py poc/workflow_3/workflow_3_config.example.py \
        poc/workflow_3/workflow_3_config_loader.py \
        poc/workflow_3/monitor/rcp_msr_gather.py \
        poc/workflow_3/monitor/align_fail_monitor.py \
        poc/workflow_3/monitor/align_fail_monitor_only_check.py \
        poc/workflow_3/monitor/test_rcp_gather_timeout.py
git commit -m "feat(workflow_3): rcp 동기 다운로드에 bounded 대기 + in-flight 가드 (F3)

동기 계약은 유지(cycle 이 assets 읽기 전 적재 보장)하되 대기만 bounded 로:
daemon thread + join(timeout), success_gather.wait_for_gather 와 같은 관용구.
timeout_sec=None 은 무한 대기라 오프라인 fetch_msr_offline.py 는 무영향.

in-flight 가드 동반 필수 - timeout 만 넣으면 포기한 스레드가 계속 쓰는 동안
새 스레드가 겹쳐 써서 '보이는 정지'가 '조용한 부분읽기 경쟁'으로 바뀐다.

신규 env ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC (기본 60)."
git show --stat --oneline HEAD
```

---

### Task 8: Document the new env flags

**Files:**
- Modify: `CLAUDE.md` (the "Recently added env flags" list)

**Interfaces:**
- Consumes: the two settings fields from Tasks 5 and 7.
- Produces: nothing code-facing.

- [ ] **Step 1: Add the entry**

In `CLAUDE.md`, in the bulleted "Recently added env flags" list, add one bullet:

```markdown
- **Loop failure-path hardening** (teardown always completes; a failing tool cannot starve the others; no office call stalls the monitor): `ALIGN_FAIL_FAILURE_COOLDOWN_SEC` (300 — a tool whose cycle ends in `run_status=error` or an aborted step is parked for this long, same mechanism as the occupied cooldown; a correction *fallback* is not a failure and does not trigger it), `ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC` (60 — bound on the synchronous `gather_rcp_msr` download; the call keeps its synchronous contract but waits on a daemon thread with a bounded join, and an in-flight guard prevents a timed-out download from racing a new one. `fetch_msr_offline.py` passes `timeout_sec=None` and stays unbounded). Teardown for all three cycles (`run_alarm_cycle`, `run_check_only_cycle`, `workflow_3e.run_abort_cycle`) goes through `monitor/teardown.run_teardown`, which guards each step independently; the ordering rule is that **input unblock is always the first step**, asserted by `monitor/test_teardown.py`.
```

- [ ] **Step 2: Verify the full suite**

```bash
uv run python poc/workflow_3/monitor/test_teardown.py
uv run python poc/workflow_3/monitor/test_failure_cooldown.py
uv run python poc/workflow_3/monitor/test_rcp_gather_timeout.py
uv run python poc/workflow_3/monitor/test_occupied_popup.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/align/test_correction.py
```
Expected: all PASS. The last three are pre-existing suites that must not regress.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: 루프 하드닝 env 2종 + teardown 순서 규약 문서화"
git show --stat --oneline HEAD
```

---

## Office verification checklist (not code — for the next office session)

These tests prove control flow survives injected faults. They cannot prove real RCS behavior. At the office, confirm:

1. **Teardown order fix is invisible in the happy path** — run `align_fail_monitor_only_check.py` once against a real tool; the tool window must still close and no popup must linger. The only behavioral change is that user input is released slightly earlier.
2. **`ALIGN_FAIL_BLOCK_INPUT=1` + a forced failure** — if you can induce a check-only cycle failure with block-input on, confirm the mouse and keyboard are usable immediately afterward. This is the F1 fix, and it is the one thing no Mac test can demonstrate.
3. **Gather timeout is not too tight** — watch the `[INFO] rcp 다운로드 완료` lines for a few alarms and compare with the 60s default. Raise `RCP_GATHER_TIMEOUT_SEC` if real downloads run close to it, since a timeout produces partial assets and a possible wrong "보정 불가".
4. **Failure cooldown is not too aggressive** — if a tool is parked for 300s, confirm from the console that it was a genuine cycle failure and not a normal fallback.
