"""notify 합성 self-test — 모호 키 재등록 권고 surface 검증(office/실장비 없이).

build_outcome_summary 의 재등록 권고 줄과 notify_correction_outcome 의 분기
(corrected_but_ambiguous audit / cube 생략)를 모듈 전역 monkeypatch 로 검증한다.

CLAUDE.md 규칙: argparse 미사용, [PASS]/[FAIL] print, Mac 에서 그대로 실행.
    uv run python poc/workflow_3/monitor/test_notify.py
"""

from poc.workflow_3.monitor import notify
from poc.workflow_3.monitor.notify import (
    build_outcome_summary,
    notify_correction_outcome,
)
from poc.workflow_3.align.correction import CorrectionOutcome


def _outcome(status, *, second_ratio=None, distinctive=True):
    """검증용 CorrectionOutcome(모호도 필드만 가변)."""
    return CorrectionOutcome(
        status=status, path="primary", key_decision="match",
        best_xy=(1, 2), ok_screen_xy=None, fallback=None,
        second_ratio=second_ratio, distinctive=distinctive,
    )


def _patch_events():
    """notify.log_work2_event 를 기록용 fake 로 교체하고 events 리스트를 돌려준다."""
    events = []
    notify.log_work2_event = lambda **kw: events.append(kw)
    return events


def _patch_cube_recorder():
    """_SEND_CUBE_FN 을 호출 기록용 fake 로 교체(+available True)하고 calls 를 돌려준다."""
    calls = []
    notify._SEND_CUBE_FN = lambda *a, **k: calls.append((a, k))
    notify.RICH_NOTIFY_AVAILABLE = True
    return calls


def test_summary_recommends_when_ambiguous() -> bool:
    """second_ratio>임계 → second_ratio 값 + '재등록 권장(모호 키)' 둘 다 포함."""
    s = build_outcome_summary(_outcome("corrected", second_ratio=0.991),
                              reregister_ratio_threshold=0.98)
    ok = "second_ratio=0.991" in s and "재등록 권장(모호 키)" in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_recommends_when_ambiguous: {s!r}")
    return ok


def test_summary_no_recommend_when_distinct() -> bool:
    """second_ratio<=임계 → 값은 보이되 권고 줄 없음."""
    s = build_outcome_summary(_outcome("corrected", second_ratio=0.50),
                              reregister_ratio_threshold=0.98)
    ok = "second_ratio=0.500" in s and "재등록 권장" not in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_no_recommend_when_distinct: {s!r}")
    return ok


def test_summary_omits_when_none() -> bool:
    """second_ratio None(후보 0개/구버전) → 모호도 줄 자체 없음(오늘과 동일)."""
    s = build_outcome_summary(_outcome("corrected", second_ratio=None),
                              reregister_ratio_threshold=0.98)
    ok = "second_ratio=" not in s and "재등록 권장" not in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_omits_when_none: {s!r}")
    return ok


def test_summary_threshold_none_skips_recommend() -> bool:
    """구 호출부(threshold 미전달): second_ratio 값은 보여도 권고는 안 한다."""
    s = build_outcome_summary(_outcome("corrected", second_ratio=0.991))
    ok = "second_ratio=0.991" in s and "재등록 권장" not in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_threshold_none_skips_recommend: {s!r}")
    return ok


def test_notify_corrected_ambiguous_audits_no_cube() -> bool:
    """corrected+모호 → warning audit 1건·cube 0건(성공이라 spam 없음)."""
    events = _patch_events()
    cube = _patch_cube_recorder()
    notify_correction_outcome(
        "EQP1", "CLS/RCP", _outcome("corrected", second_ratio=0.991),
        reregister_ratio_threshold=0.98,
    )
    messages = [e.get("message") for e in events]
    ev = next((e for e in events if e.get("message") == "corrected_but_ambiguous"), None)
    ok = (
        "corrected_but_ambiguous" in messages
        and "corrected_no_notify" not in messages
        and len(cube) == 0
        and ev is not None
        and ev.get("level") == "warning"
        and ev.get("eqp_id") == "EQP1"
        and ev.get("recipe_id") == "CLS/RCP"
    )
    print(f"[{'PASS' if ok else 'FAIL'}] corrected_ambiguous_audits_no_cube: "
          f"messages={messages} cube_calls={len(cube)}")
    return ok


def test_notify_corrected_distinct_no_notify() -> bool:
    """corrected+유일 → 파일 audit 없음, cube 0건."""
    events = _patch_events()
    cube = _patch_cube_recorder()
    notify_correction_outcome(
        "EQP1", "CLS/RCP", _outcome("corrected", second_ratio=0.50),
        reregister_ratio_threshold=0.98,
    )
    messages = [e.get("message") for e in events]
    ok = (
        messages == []
        and len(cube) == 0
    )
    print(f"[{'PASS' if ok else 'FAIL'}] corrected_distinct_no_notify: "
          f"messages={messages} cube_calls={len(cube)}")
    return ok


def test_notify_failure_carries_recommendation() -> bool:
    """실패+모호 → outcome_notify audit 의 summary 에 재등록 권고 포함(=cube 에 실릴 문자열)."""
    events = _patch_events()
    # cube 비활성으로 동기 종료(daemon thread race 회피).
    notify._SEND_CUBE_FN = None
    notify.RICH_NOTIFY_AVAILABLE = False
    notify_correction_outcome(
        "EQP1", "CLS/RCP", _outcome("escalated_no_ok", second_ratio=0.991),
        reregister_ratio_threshold=0.98,
    )
    ev = next((e for e in events if e.get("message") == "outcome_notify"), None)
    ok = (
        ev is not None
        and ev.get("status") == "escalated_no_ok"
        and "재등록 권장(모호 키)" in ev.get("summary", "")
    )
    print(f"[{'PASS' if ok else 'FAIL'}] failure_carries_recommendation: "
          f"summary={ev.get('summary') if ev else None!r}")
    return ok


def test_summary_names_failed_stage() -> bool:
    """보정 미수행 + 실패 step → 어느 단계에서 멈췄는지 한국어 라벨로 나온다.

    엔지니어의 다음 행동이 갈리는 지점이라 필요하다: 접속 실패면 tool 을 직접 열어야
    하고, 보정 실패면 이미 열린 창에서 align point 만 잡으면 된다.
    """
    s = build_outcome_summary(
        None, failed_step="wait_tool_window", failure_class="rcs_occupied_select",
    )
    ok = "접속" in s and "wait_tool_window" in s and "rcs_occupied_select" in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_names_failed_stage: {s!r}")
    return ok


def test_summary_names_correction_stage() -> bool:
    """보정 단계에서 실패 → '접속' 이 아니라 보정 단계로 읽혀야 한다."""
    s = build_outcome_summary(None, failed_step="run_correction", failure_class="")
    ok = "보정" in s and "run_correction" in s and "접속" not in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_names_correction_stage: {s!r}")
    return ok


def test_summary_unknown_step_still_reported() -> bool:
    """라벨 없는 step 이름도 원문 그대로 실려야 한다(모르는 단계를 숨기지 않는다)."""
    s = build_outcome_summary(None, failed_step="brand_new_step")
    ok = "brand_new_step" in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_unknown_step_still_reported: {s!r}")
    return ok


def test_summary_omits_stage_when_absent() -> bool:
    """실패 step 이 없으면(정상 완주/구 호출부) 단계 줄 자체가 없다."""
    s = build_outcome_summary(_outcome("awaiting_engineer_ok"))
    ok = "실패단계" not in s
    print(f"[{'PASS' if ok else 'FAIL'}] summary_omits_stage_when_absent: {s!r}")
    return ok


# ------------------------------------------------------------------
# 자동 보정 불가 경로: 원인 + 요구 행동 + 매칭점수 (2026-09-01).
# ------------------------------------------------------------------


def _outcome_with_history(status, score, scale, **kw):
    """paused_match history 를 단 CorrectionOutcome."""
    out = _outcome(status, **kw)
    out.history = [{
        "stage": "paused_match", "decision": "low",
        "score": score, "best_scale": scale,
    }]
    return out


def test_uncorrected_summary_carries_reason_and_action():
    """no_assets 요약에 원인/요구 행동이 status 앞에 실린다."""
    summary = build_outcome_summary(_outcome("no_assets"))
    assert "자동 보정 불가" in summary, summary
    assert "직접 align point" in summary, summary
    assert summary.index("자동 보정 불가") < summary.index("status="), summary
    print("[PASS] no_assets 요약 = 원인 + 요구 행동")
    return True


def test_uncorrected_summary_carries_match_index():
    """매칭까지 간 실패는 점수를 임계와 나란히 싣는다."""
    summary = build_outcome_summary(
        _outcome_with_history("escalated_key_not_visible", 0.31, 0.55)
    )
    assert "매칭점수=0.310" in summary, summary
    assert "match임계" in summary and "scale=0.55" in summary, summary
    print("[PASS] escalated_key_not_visible 요약 = 매칭점수 지수")
    return True


def test_corrected_paths_unchanged():
    """성공/기존 status 는 원인 줄이 붙지 않는다(요약 모양 보존)."""
    for status in ("corrected", "awaiting_engineer_ok"):
        summary = build_outcome_summary(_outcome(status))
        assert "자동 보정 불가" not in summary, summary
    print("[PASS] corrected/awaiting_engineer_ok 요약 불변")
    return True


def main() -> int:
    print("[INFO] notify self-test 시작")
    results = [
        test_summary_names_failed_stage(),
        test_summary_names_correction_stage(),
        test_summary_unknown_step_still_reported(),
        test_summary_omits_stage_when_absent(),
        test_summary_recommends_when_ambiguous(),
        test_summary_no_recommend_when_distinct(),
        test_summary_omits_when_none(),
        test_summary_threshold_none_skips_recommend(),
        test_notify_corrected_ambiguous_audits_no_cube(),
        test_notify_corrected_distinct_no_notify(),
        test_notify_failure_carries_recommendation(),
        test_uncorrected_summary_carries_reason_and_action(),
        test_uncorrected_summary_carries_match_index(),
        test_corrected_paths_unchanged(),
    ]
    passed = sum(1 for r in results if r)
    print(f"[INFO] {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


