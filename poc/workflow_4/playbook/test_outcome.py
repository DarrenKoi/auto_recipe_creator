"""Recovery Outcome 파생 - 순수 함수라 실장비/workflow_3 없이 돈다.

`uv run pytest poc/workflow_4/`
"""

from poc.workflow_4.playbook.outcome import (
    ABORTED,
    ESCALATED,
    PATH_FALLBACK,
    PATH_PRIMARY,
    PATH_UNKNOWN,
    RECOVERED,
    UNKNOWN,
    derive_outcome,
    evaluate_attempt,
    format_episode_digest,
)


def _reads(*decisions):
    return [{"decision": d, "value": i} for i, d in enumerate(decisions)]


def _attempt(seq=1, measurement=None, reads=(), **extra):
    attempt = {
        "attempt_seq": seq,
        "measurement": {"value": measurement} if measurement else None,
        "numerator_reads": list(reads),
        "guards": [
            {"kind": "screen_observability", "value": True},
            {"kind": "occupancy_control", "value": True},
            {"kind": "align_key_visibility", "value": None},
        ],
    }
    attempt.update(extra)
    return attempt


def _episode(*attempts, **extra):
    evidence = {
        "episode_id": "cef99184-28d1-4ab9-b9e9-513e46a50fb0",
        "eqp_id": "MCD916", "recipe_id": "CLS/RCP",
        "attempts": list(attempts), "complete": True, "incomplete_reasons": [],
    }
    evidence.update(extra)
    return evidence


# ------------------------------------------------------------------
# Verification 우선순위.
# ------------------------------------------------------------------


def test_measurement_success_is_the_only_primary_route_to_recovered():
    verdict = evaluate_attempt(_attempt(measurement="success"))
    assert verdict.recovered is True
    assert verdict.verification_path == PATH_PRIMARY


def test_measurement_failure_is_not_recovered_and_does_not_open_the_fallback():
    """관측된 실패를 카운터 증가로 뒤집으면 화면이 깨진 채로 성공이 된다."""
    verdict = evaluate_attempt(_attempt(
        measurement="failure",
        reads=_reads("first_sample", "strictly_increasing", "strictly_increasing"),
    ))
    assert verdict.recovered is False
    assert verdict.verification_path == PATH_PRIMARY


def test_fallback_is_consulted_only_when_measurement_is_unknown():
    verdict = evaluate_attempt(_attempt(
        measurement="unknown",
        reads=_reads("first_sample", "strictly_increasing", "strictly_increasing"),
    ))
    assert verdict.recovered is True
    assert verdict.verification_path == PATH_FALLBACK


def test_fallback_needs_a_strictly_increasing_run_and_breaks_are_fatal():
    """OCR miss / 같음·감소 / reground reset 은 연속을 끊어 회복을 만들지 못한다."""
    for broken in (
        ("first_sample", "ocr_miss", "strictly_increasing", "strictly_increasing"),
        ("first_sample", "strictly_increasing", "equal_or_decrease", "first_sample"),
        ("first_sample", "strictly_increasing", "reground_reset", "first_sample"),
        ("first_sample", "first_sample", "first_sample"),  # 증가가 한 번도 없다.
        ("first_sample", "strictly_increasing"),           # 표본이 모자란다.
    ):
        verdict = evaluate_attempt(_attempt(measurement="unknown", reads=_reads(*broken)))
        assert verdict.recovered is False, broken
        assert verdict.verification_path == PATH_UNKNOWN, broken


def test_not_sampled_polls_neither_count_nor_break_the_run():
    """읽기를 시도조차 안 한 회차는 연속의 일부가 아니다(끊지도 않는다)."""
    verdict = evaluate_attempt(_attempt(measurement="unknown", reads=_reads(
        "first_sample", "not_sampled", "strictly_increasing", "not_sampled",
        "strictly_increasing",
    )))
    assert verdict.recovered is True
    assert verdict.verification_path == PATH_FALLBACK


# ------------------------------------------------------------------
# Episode 판정.
# ------------------------------------------------------------------


def test_a_later_qualified_recovery_wins_and_history_is_preserved():
    evidence = _episode(
        _attempt(1, measurement="unknown"),
        _attempt(2, measurement="failure"),
        _attempt(3, measurement="success"),
    )
    result = derive_outcome(evidence)
    assert result.outcome == RECOVERED
    assert result.deciding_attempt == 3
    # attempt 이력은 지워지지 않는다 - 앞선 판정도 그대로 보인다.
    assert [v.attempt_seq for v in result.attempts] == [1, 2, 3]
    assert [v.recovered for v in result.attempts] == [False, False, True]


def test_provenance_only_signals_cannot_produce_recovered():
    """알람 해제 / OK 클릭 / corrected / runner 완료 / 커서 정지 / 창 닫힘 / 닫기 정황."""
    evidence = _episode(
        _attempt(
            1, measurement="unknown",
            ok_clicked=True, correction_status="corrected", run_status="completed",
            cursor_idle=True, window_closed=True, probable_close=True,
        ),
        alarm_cleared=True,
    )
    result = derive_outcome(evidence)
    assert result.outcome == UNKNOWN
    assert result.verification_path == PATH_UNKNOWN


def test_explicit_abort_and_handoff_records_decide_the_rest():
    aborted = derive_outcome(_episode(_attempt(
        1, measurement="unknown", abort={"aborted": True, "reason": "hotkey"})))
    assert aborted.outcome == ABORTED and aborted.reason == "hotkey"

    escalated = derive_outcome(_episode(_attempt(
        1, measurement="unknown",
        handoff={"explicit": True, "reason": "awaiting_engineer_ok"})))
    assert escalated.outcome == ESCALATED

    # 긴급 해제가 더 종결적인 사실이라 handoff 보다 앞선다.
    both = derive_outcome(_episode(_attempt(
        1, measurement="unknown",
        abort={"aborted": True, "reason": "hotkey"},
        handoff={"explicit": True, "reason": "engineer"})))
    assert both.outcome == ABORTED


def test_a_node_or_status_named_handoff_alone_does_not_escalate():
    """이름만으로는 escalated 가 되지 않는다 - 명시 기록이 있어야 한다."""
    evidence = _episode(_attempt(
        1, measurement="unknown",
        node="handoff", run_status="handoff", failed_step="handoff",
        outcome_status="escalated_ambiguous_key",
        handoff=None,
    ))
    assert derive_outcome(evidence).outcome == UNKNOWN

    # 기록이 있어도 explicit 이 아니면 escalated 가 아니다.
    soft = _episode(_attempt(1, measurement="unknown", handoff={"explicit": False}))
    assert derive_outcome(soft).outcome == UNKNOWN


def test_episode_without_attempts_is_unknown():
    assert derive_outcome(_episode()).outcome == UNKNOWN
    assert derive_outcome(None).outcome == UNKNOWN


# ------------------------------------------------------------------
# digest.
# ------------------------------------------------------------------


def test_digest_is_one_line_carrying_the_required_fields():
    evidence = _episode(
        _attempt(1, measurement="unknown"),
        _attempt(2, measurement="success"),
    )
    line = format_episode_digest(evidence, derive_outcome(evidence))

    assert "\n" not in line
    assert line.startswith("[DIGEST] episode ")
    assert "id=cef99184 " in line
    assert "eqp=MCD916" in line
    assert "recipe=CLS/RCP" in line
    assert "attempts=2" in line
    assert f"outcome={RECOVERED}" in line
    assert "verify=primary" in line
    assert "complete=yes" in line
    # Guard 3값이 모두 실린다.
    assert "screen:true" in line and "occupancy:true" in line and "align:unknown" in line


def test_digest_reports_incomplete_with_its_reasons():
    evidence = _episode(
        _attempt(1, measurement="unknown"),
        complete=False, incomplete_reasons=["attempt_1:run_status:error"],
    )
    line = format_episode_digest(evidence, derive_outcome(evidence))
    assert "complete=no(attempt_1:run_status:error)" in line
    assert "outcome=unknown" in line


def test_playbook_layer_does_not_import_workflow_3():
    """도메인 계층은 workflow_3 를 모른다 - import 문 자체가 없어야 한다.

    문자열 검색이 아니라 AST 로 본다. 문서에 'workflow_3' 를 **언급**하는 것은
    정상이고(어느 계층이 무엇을 넘기는지 적어야 한다), 금지되는 것은 의존이다.
    """
    import ast
    from pathlib import Path

    import poc.workflow_4.playbook as pkg

    for path in Path(pkg.__file__).parent.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "workflow_3" not in name, (path, name)
