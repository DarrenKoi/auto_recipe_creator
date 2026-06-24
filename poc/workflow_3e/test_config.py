"""Workflow3eSettings / load_workflow3e_settings self-test.

새 MEAS_FAIL_* env 반영 + abort 클릭 이중 게이트(SAFE_MODE 강제 dry-run)를 검증한다.

    uv run python poc/workflow_3e/test_config.py
"""

import os
from contextlib import contextmanager

from poc.workflow_3e.config import load_workflow3e_settings


@contextmanager
def _env(**kv):
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
    """기본값: 잡 on, ALID 빈값, 클릭 dry-run, locator slug ui-venus."""
    with _env(SAFE_MODE="1", MEAS_FAIL_ABORT_ENABLED=None, MEAS_FAIL_ALID=None,
              MEAS_FAIL_ABORT_DRY_RUN=None, MEAS_FAIL_ABORT_BUTTON_SERVICE=None):
        s = load_workflow3e_settings()
    ok = (s.meas_fail_abort_enabled is True and s.meas_fail_alid == ""
          and s.abort_action_dry_run is True and s.abort_button_vlm_service == "ui-venus")
    print(f"[{'PASS' if ok else 'FAIL'}] defaults: enabled={s.meas_fail_abort_enabled} "
          f"alid={s.meas_fail_alid!r} dry_run={s.abort_action_dry_run}")
    return ok


def test_safe_mode_forces_dry_run():
    """SAFE_MODE=1 이면 MEAS_FAIL_ABORT_DRY_RUN=0 을 줘도 dry-run 강제."""
    with _env(SAFE_MODE="1", MEAS_FAIL_ABORT_DRY_RUN="0"):
        s = load_workflow3e_settings()
    ok = s.abort_action_dry_run is True
    print(f"[{'PASS' if ok else 'FAIL'}] safe_mode_forces_dry_run: dry_run={s.abort_action_dry_run}")
    return ok


def test_armed_when_safe_off_and_flag_zero():
    """SAFE_MODE=0 + MEAS_FAIL_ABORT_DRY_RUN=0 일 때만 클릭 무장(dry_run=False)."""
    with _env(SAFE_MODE="0", MEAS_FAIL_ABORT_DRY_RUN="0"):
        s = load_workflow3e_settings()
    ok = s.abort_action_dry_run is False
    print(f"[{'PASS' if ok else 'FAIL'}] armed_when_safe_off_and_flag_zero: dry_run={s.abort_action_dry_run}")
    return ok


def test_env_overrides_and_base_fields_present():
    """MEAS_FAIL_* env 반영 + base(Workflow3Settings) 필드도 그대로 살아 있다."""
    with _env(MEAS_FAIL_ABORT_ENABLED="0", MEAS_FAIL_ALID="9012",
              MEAS_FAIL_ABORT_BUTTON_SERVICE="mai-ui"):
        s = load_workflow3e_settings()
    ok = (s.meas_fail_abort_enabled is False and s.meas_fail_alid == "9012"
          and s.abort_button_vlm_service == "mai-ui"
          # base 필드 sanity — 상속이 깨지지 않았는지.
          and hasattr(s, "poll_interval_sec") and hasattr(s, "correction_dry_run"))
    print(f"[{'PASS' if ok else 'FAIL'}] env_overrides_and_base_fields_present: "
          f"alid={s.meas_fail_alid!r} poll={getattr(s, 'poll_interval_sec', None)}")
    return ok


def main():
    print("[INFO] workflow_3e config self-test 시작")
    results = [
        test_defaults(),
        test_safe_mode_forces_dry_run(),
        test_armed_when_safe_off_and_flag_zero(),
        test_env_overrides_and_base_fields_present(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
