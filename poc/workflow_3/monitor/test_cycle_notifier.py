"""CycleNotifier self-test — 알람 1건당 cube '정확히 1회' + 지연 watchdog 검증.

사이클 본문과 finally 가 같은 notifier 를 두 번 부르는 구조(예외 경로에서도 침묵하지
않게)라서, 중복 발송이 없다는 것이 핵심 불변식이다. watchdog 은 timer_factory 를
주입해 실제 시간을 기다리지 않고 결정적으로 검증한다.

CLAUDE.md 규칙: argparse 미사용, [OK] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/monitor/test_cycle_notifier.py
"""

from poc.workflow_3.monitor import notify
from poc.workflow_3.align.correction import CorrectionOutcome


class FakeTimer:
    """threading.Timer 대역 — start/cancel 만 기록하고 fire() 로 수동 발화한다."""

    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.started = False
        self.cancelled = False
        self.daemon = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        """만료 시점 재현 — cancel 여부와 무관하게 콜백을 호출한다.

        실제 Timer 도 cancel 과 발화가 경합하면 콜백이 돌 수 있으므로, 게이트가
        스스로 중복을 막는지 보려면 취소된 타이머도 발화시켜 봐야 한다.
        """
        self.function()


def _swap(state, module, name, value):
    state.setdefault(module, {})[name] = getattr(module, name)
    setattr(module, name, value)


def _restore(state):
    for module, saved in state.items():
        for name, value in saved.items():
            setattr(module, name, value)


def _outcome(status):
    return CorrectionOutcome(
        status=status, path="primary", key_decision="match",
        best_xy=(1, 2), ok_screen_xy=None, fallback=None,
    )


def _patch_sinks(state):
    """outcome/progress 두 발송 경로를 기록용 fake 로 교체하고 (outcomes, progress) 반환."""
    outcomes = []
    progress = []
    _swap(state, notify, "notify_correction_outcome",
          lambda *a, **k: outcomes.append((a, k)))
    _swap(state, notify, "send_progress_notify",
          lambda *a, **k: progress.append((a, k)))
    return outcomes, progress


def test_outcome_sent_once_even_if_called_twice():
    """본문 + finally 이중 호출 → cube 는 1회만(중복 알림 없음)."""
    state = {}
    outcomes, _ = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        first = notifier.notify_outcome(_outcome("awaiting_engineer_ok"))
        second = notifier.notify_outcome(None)
        assert first is True, first
        assert second is False, second
        assert len(outcomes) == 1, outcomes
        # 발송된 것은 첫 호출(실제 outcome)이어야 한다 - finally 의 None 이 덮으면 안 된다.
        assert outcomes[0][0][2].status == "awaiting_engineer_ok", outcomes
    finally:
        _restore(state)
    print("[OK] test_outcome_sent_once_even_if_called_twice")


def test_finally_notifies_when_body_never_did():
    """예외로 본문이 발송을 못 했으면 finally 호출이 실제로 나간다(침묵 금지)."""
    state = {}
    outcomes, _ = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        sent = notifier.notify_outcome(None)
        assert sent is True, sent
        assert len(outcomes) == 1, outcomes
        assert outcomes[0][0][2] is None, outcomes
    finally:
        _restore(state)
    print("[OK] test_finally_notifies_when_body_never_did")


def test_watchdog_fires_progress_when_cycle_is_slow():
    """지연 임계까지 결과가 없으면 '진행 중' 고지가 1회 나간다(무한 침묵 금지)."""
    state = {}
    outcomes, progress = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        notifier.start_watchdog(90.0)
        assert notifier._timer.interval == 90.0, notifier._timer.interval
        notifier._timer.fire()
        assert len(progress) == 1, progress
        # 진행 고지는 결과 알림이 아니다 - 결과는 아직 0건이어야 한다.
        assert outcomes == [], outcomes
        assert progress[0][0][0] == "EQP1", progress
    finally:
        _restore(state)
    print("[OK] test_watchdog_fires_progress_when_cycle_is_slow")


def test_watchdog_silent_when_outcome_arrives_first():
    """결과가 먼저 나오면 watchdog 은 취소되고, 늦게 발화해도 진행 고지는 없다."""
    state = {}
    outcomes, progress = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        notifier.start_watchdog(90.0)
        timer = notifier._timer
        notifier.notify_outcome(_outcome("corrected"))
        assert timer.cancelled is True
        # cancel 과 발화가 경합한 경우까지 - 게이트가 스스로 막아야 한다.
        timer.fire()
        assert progress == [], progress
        assert len(outcomes) == 1, outcomes
    finally:
        _restore(state)
    print("[OK] test_watchdog_silent_when_outcome_arrives_first")


def test_watchdog_disabled_when_delay_not_positive():
    """delay<=0 이면 watchdog 자체를 걸지 않는다(끄는 스위치)."""
    state = {}
    _, progress = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        notifier.start_watchdog(0.0)
        assert notifier._timer is None, notifier._timer
        assert progress == [], progress
    finally:
        _restore(state)
    print("[OK] test_watchdog_disabled_when_delay_not_positive")


def test_watchdog_progress_fires_only_once():
    """타이머가 어떤 이유로 두 번 발화해도 진행 고지는 1회다."""
    state = {}
    _, progress = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier("EQP1", "CLS/RCP", timer_factory=FakeTimer)
        notifier.start_watchdog(90.0)
        notifier._timer.fire()
        notifier._timer.fire()
        assert len(progress) == 1, progress
    finally:
        _restore(state)
    print("[OK] test_watchdog_progress_fires_only_once")


def test_watchdog_respects_disabled_notifier():
    """rich_notify 비활성(enabled=False) 이면 진행 고지도 나가지 않는다."""
    state = {}
    _, progress = _patch_sinks(state)
    try:
        notifier = notify.CycleNotifier(
            "EQP1", "CLS/RCP", enabled=False, timer_factory=FakeTimer,
        )
        notifier.start_watchdog(90.0)
        assert notifier._timer is None, notifier._timer
        assert progress == [], progress
    finally:
        _restore(state)
    print("[OK] test_watchdog_respects_disabled_notifier")


def test_settings_expose_notify_delay():
    """지연 임계는 env 로 튜닝 가능해야 한다(기본 90s, 0=끄기)."""
    import os

    from poc.workflow_3.config import load_workflow3_settings

    saved = os.environ.pop("ALIGN_FAIL_NOTIFY_DELAY_SEC", None)
    try:
        assert load_workflow3_settings().notify_delay_sec == 90.0
        os.environ["ALIGN_FAIL_NOTIFY_DELAY_SEC"] = "45"
        assert load_workflow3_settings().notify_delay_sec == 45.0
    finally:
        os.environ.pop("ALIGN_FAIL_NOTIFY_DELAY_SEC", None)
        if saved is not None:
            os.environ["ALIGN_FAIL_NOTIFY_DELAY_SEC"] = saved
    print("[OK] test_settings_expose_notify_delay")


def main() -> int:
    print("[INFO] CycleNotifier self-test 시작")
    tests = [
        test_outcome_sent_once_even_if_called_twice,
        test_finally_notifies_when_body_never_did,
        test_watchdog_fires_progress_when_cycle_is_slow,
        test_watchdog_silent_when_outcome_arrives_first,
        test_watchdog_disabled_when_delay_not_positive,
        test_watchdog_progress_fires_only_once,
        test_watchdog_respects_disabled_notifier,
        test_settings_expose_notify_delay,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    print(f"[INFO] {len(tests) - failed}/{len(tests)} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
