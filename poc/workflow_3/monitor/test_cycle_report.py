"""cycle_report 단위 테스트 - RCS/VLM 없이 Mac 에서 실행된다.

    uv run python poc/workflow_3/monitor/test_cycle_report.py
    uv run pytest poc/workflow_3/monitor/test_cycle_report.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.workflow_3.monitor.cycle_report import (  # noqa: E402
    build_cycle_report,
    print_cycle_report,
)


def _result(**kwargs):
    base = dict(
        eqp_id="MCD916", recipe_id="CLASS/ARDL", tag="260820_113045",
        run_status="completed", failed_step="", failure_class="",
        frame_count=1832, recording_dir="/a/b/260820_113045/recording",
        run_dir="/logs/workflow_runs/260820_113045_align_fail_cycle_MCD916",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _controller(model_id="vlm_live_box", mode_hint="SEM"):
    panel = SimpleNamespace(
        model_id=model_id, panel_roi=(1024, 318, 618, 618), confidence=0.92
    )
    return SimpleNamespace(panel=panel, mode_hint=mode_hint, mode_default="SEM")


def _outcome(status="corrected", **kwargs):
    base = dict(
        status=status, path="primary", key_decision="match",
        best_xy=(812, 455), ok_screen_xy=(1450, 980), error=None,
        second_ratio=0.631, score_gap=0.081, distinctive=True,
        history=[{"stage": "paused_match", "mode": "SEM", "decision": "match",
                  "score": 0.712, "chamfer": 0.684, "best_scale": 1.0}],
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _text(result, context, **kw):
    return "\n".join(build_cycle_report(result, context, **kw))


# --- 판정 한 줄 --------------------------------------------------------------


def test_corrected_is_ok():
    out = _text(_result(), {"controller": _controller(), "outcome": _outcome()})
    assert "[OK] 보정 완료" in out


def test_awaiting_engineer_ok_is_success_not_failure():
    """반자동 모드의 정상 종료다. 실패로 읽히면 엔지니어가 헛되이 원인을 찾는다."""
    out = _text(_result(), {"outcome": _outcome("awaiting_engineer_ok")})
    assert "[OK]" in out and "엔지니어" in out


def test_corrected_unverified_is_warning():
    out = _text(_result(), {"outcome": _outcome("corrected_unverified")})
    assert "[!!]" in out and "미확인" in out


def test_view_only_is_warning_not_ok():
    out = _text(_result(), {"outcome": _outcome("view_only_observation")})
    assert "[!!]" in out and "[OK]" not in out


def test_fallback_status_falls_through_to_prefix_branch():
    out = _text(_result(), {"outcome": _outcome("fallback_escalated_low_score")})
    assert "[!!]" in out and "live search" in out


def test_substituted_status_never_hits_fallback_branch():
    """치환 status 는 정확 비교로 먼저 잡혀야 한다(접두사 분기로 새면 판정이 뒤집힌다)."""
    for status in ("view_only_observation", "corrected_unverified"):
        out = _text(_result(), {"outcome": _outcome(status)})
        assert "live search" not in out, status


def test_run_failure_beats_outcome():
    """step 실패는 outcome 보다 우선한다 - 낡은 outcome 이 성공으로 보이면 안 된다."""
    result = _result(run_status="failed", failed_step="locate_sem_panel",
                     failure_class="panel_not_found")
    out = _text(result, {"outcome": _outcome("corrected")})
    assert "[XX]" in out and "panel_not_found" in out and "[OK]" not in out


def test_no_outcome_reports_no_correction():
    out = _text(_result(), {"controller": _controller()})
    assert "[!!]" in out and "보정 결과 없음" in out


# --- 신호 표시 ---------------------------------------------------------------


def test_sem_box_detected_shows_roi_and_source():
    out = _text(_result(), {"controller": _controller()})
    assert "검출 O" in out and "(1024,318) 618x618px" in out and "VLM live box" in out


def test_landmark_source_is_distinguished():
    out = _text(_result(), {"controller": _controller(model_id="landmark_a")})
    assert "src=landmark" in out


def test_sem_box_missing_is_explicit():
    out = _text(_result(), {})
    assert "검출 X" in out


def test_mode_hint_absent_names_the_default_used():
    out = _text(_result(), {"controller": _controller(mode_hint=None)})
    assert "판독 실패" in out and "SEM" in out


def test_match_scores_are_shown():
    out = _text(_result(), {"outcome": _outcome()})
    assert "score=0.712" in out and "chamfer=0.684" in out and "2nd비=0.631" in out


def test_align_position_is_shown():
    out = _text(_result(), {"outcome": _outcome()})
    assert "align=(812,455)" in out and "OK버튼=(1450,980)" in out


def test_artifact_paths_are_listed():
    context = {"correction_debug_dir": "/d/debug_images/align_fail_cycle/260820_113045"}
    out = _text(_result(), context)
    assert "1832 frames" in out
    assert "align_fail_cycle/260820_113045" in out
    assert "workflow_runs" in out


def test_occupancy_is_translated():
    out = _text(_result(), {"occupancy": "occupied_by_other"})
    assert "다른 엔지니어 점유" in out


def test_elapsed_formats_minutes():
    assert "4m 12s" in _text(_result(), {}, elapsed_sec=252.4)
    assert "42s" in _text(_result(), {}, elapsed_sec=42)


# --- 견고성 ------------------------------------------------------------------


def test_missing_fields_never_raise():
    """보고서는 teardown 뒤에 돈다 - 여기서 뜬 예외는 끝난 사이클 결과를 날린다."""
    broken = SimpleNamespace(eqp_id="X")
    assert build_cycle_report(broken, {"controller": object(), "outcome": object()})


def test_none_context_is_tolerated():
    assert build_cycle_report(_result(), None)


def test_print_swallows_exceptions(capsys=None):
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    print_cycle_report(Exploding(), {})  # 예외가 새어 나오면 이 줄에서 실패한다


def test_report_is_cp949_encodable():
    """오피스 콘솔은 cp949 다. 이모지/em-dash 가 섞이면 print 가 죽는다."""
    context = {
        "controller": _controller(), "outcome": _outcome("corrected_unverified"),
        "occupancy": "unknown", "correction_debug_dir": "/d/x",
    }
    for status in ("corrected", "awaiting_engineer_ok", "view_only_observation",
                   "escalated_ambiguous_key", "no_assets", "fallback_x"):
        context["outcome"] = _outcome(status)
        _text(_result(), context).encode("cp949")
    _text(_result(run_status="failed"), {}).encode("cp949")
    _text(_result(), {"controller": _controller(mode_hint=None)}).encode("cp949")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[OK]   {name}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
