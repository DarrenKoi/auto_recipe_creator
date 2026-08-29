"""Episode-level Recovery Guard 판독 테스트 - attempt 폴더에 남는 record 를 본다.

Guard 는 정확히 세 종류이고, 관측하지 못한 것은 절대 `false` 가 되지 않는다. 이 두
가지가 이 모듈의 존재 이유이므로 테스트도 그 둘에 집중한다.

`uv run pytest poc/workflow_3/monitor/test_guard_readings.py`
"""

import json

from poc.workflow_3.monitor.guard_readings import (
    GUARD_ALIGN_KEY,
    GUARD_KINDS,
    GUARD_OCCUPANCY,
    GUARD_SCREEN,
    align_key_guard,
    ok_control_precondition,
    occupancy_guard,
    screen_observability_guard,
    write_guard_records,
)
from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER, UNKNOWN

_RECT = {"left": 0, "top": 0, "right": 100, "bottom": 100}


def _sidecar(occlusion="none", rect=_RECT):
    return {
        "frame": "seq_0001", "t_sec": 1.0,
        "window_rect": rect,
        "foreground_title": "Remote Monitoring System - EQP1",
        "occlusion": occlusion, "cursor_screen_xy": None, "cursor_in_window": False,
    }


def _match_kwargs(**overrides):
    base = dict(
        mode="SEM", key_decision="match", distinctive=True, second_ratio=0.5,
        matcher_error=None, correction_status="corrected",
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------------
# 스키마 / 직렬화.
# ------------------------------------------------------------------


def test_exactly_three_guard_kinds_serialize_with_the_required_fields(tmp_path):
    """Guard 는 셋뿐이고 각 reading 은 값/사유/관측시각/Episode-relative evidence 를 갖는다."""
    guards = [
        screen_observability_guard(_sidecar(), age_sec=0.5,
                                   evidence="attempt_1/recording/frame_meta.jsonl"),
        occupancy_guard(FREE, share_status="", evidence=""),
        align_key_guard(evidence="attempt_1/recording", **_match_kwargs()),
    ]
    path = write_guard_records(tmp_path, attempt_seq=1, guards=guards,
                               preconditions=[ok_control_precondition(
                                   ok_screen_xy=(10, 20), correction_status="corrected")])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "guards.json"
    assert data["schema_version"] == "recovery_guards.v1"
    assert data["attempt_seq"] == 1
    assert [g["kind"] for g in data["guards"]] == list(GUARD_KINDS)
    assert len(GUARD_KINDS) == 3
    for guard in data["guards"]:
        assert set(guard) == {"kind", "value", "reason", "observed_at", "evidence", "detail"}
        assert guard["value"] in (True, False, None)
        assert guard["reason"]
        assert guard["observed_at"]
        ref = guard["evidence"]
        assert not ref.startswith("/") and ".." not in ref.split("/"), ref

    # OK 컨트롤 가용성은 Guard 가 아니라 precondition 이다.
    assert [p["kind"] for p in data["preconditions"]] == ["ok_control_available"]
    assert "ok_control_available" not in [g["kind"] for g in data["guards"]]


def test_guard_kinds_are_a_closed_set():
    """Guard 종류를 늘리는 경로가 없다 - 새 Guard 는 새 observation contract 다."""
    assert GUARD_KINDS == (GUARD_SCREEN, GUARD_OCCUPANCY, GUARD_ALIGN_KEY)


# ------------------------------------------------------------------
# 관측 실패는 전부 unknown.
# ------------------------------------------------------------------


def test_screen_guard_is_unknown_when_the_sidecar_is_missing_or_stale():
    """사이드카 부재/미판정/stale/rect 없음은 unknown 이다 - false 로 새면 안 된다."""
    assert screen_observability_guard(None, age_sec=0.0)["value"] is None
    assert screen_observability_guard(_sidecar(occlusion="unknown"), age_sec=0.0)["value"] is None
    assert screen_observability_guard(_sidecar(rect=None), age_sec=0.0)["value"] is None
    stale = screen_observability_guard(_sidecar(), age_sec=999.0, max_age_sec=10.0)
    assert stale["value"] is None and "stale" in stale["reason"]


def test_screen_guard_reads_occlusion_when_observed():
    """가림이 실제로 판정된 프레임만 true/false 를 낸다."""
    assert screen_observability_guard(_sidecar("none"), age_sec=0.1)["value"] is True
    assert screen_observability_guard(_sidecar("partial"), age_sec=0.1)["value"] is False
    assert screen_observability_guard(_sidecar("full"), age_sec=0.1)["value"] is False


def test_occupancy_guard_maps_the_three_states_without_collapsing_unknown():
    """점유 3상태가 그대로 3상태로 간다 - unknown 을 free 로 접지 않는다."""
    assert occupancy_guard(FREE, share_status="")["value"] is True
    assert occupancy_guard(OCCUPIED_BY_OTHER, share_status="")["value"] is False
    assert occupancy_guard(UNKNOWN, share_status="")["value"] is None
    # 화면 공유 요청 결과는 provenance 로만 남는다(제어를 얻은 것이 아니다).
    shared = occupancy_guard(OCCUPIED_BY_OTHER, share_status="granted")
    assert shared["value"] is False
    assert shared["detail"]["share_status"] == "granted"


def test_align_key_guard_is_unknown_for_every_unreadable_input():
    """mode 미판독 / asset 없음 / matcher 예외 / 미실행 / 유일성 미판독은 전부 unknown."""
    for overrides, hint in (
        ({"mode": ""}, "mode"),
        ({"correction_status": "no_assets"}, "assets"),
        ({"matcher_error": "boom"}, "matcher"),
        ({"key_decision": ""}, "matcher"),
        ({"second_ratio": None}, "uniqueness"),
    ):
        guard = align_key_guard(**_match_kwargs(**overrides))
        assert guard["value"] is None, (overrides, guard)
        assert hint in guard["reason"], (overrides, guard["reason"])


def test_align_key_guard_is_true_only_when_matched_and_unique():
    """매칭됐고 유일할 때만 true - ambiguous/candidate 는 true 가 아니다."""
    assert align_key_guard(**_match_kwargs())["value"] is True
    # 만성 모호(engineer_review 로 보류된 경로).
    ambiguous = align_key_guard(**_match_kwargs(
        correction_status="escalated_ambiguous_key"))
    assert ambiguous["value"] is False and "ambiguous" in ambiguous["reason"]
    # 약한 후보(adjust)인데 구조 유일성이 없다 = candidate, true 아님.
    candidate = align_key_guard(**_match_kwargs(key_decision="adjust", distinctive=False))
    assert candidate["value"] is False
    # 키가 아예 안 보인다 - 관측된 부정이므로 false 다.
    assert align_key_guard(**_match_kwargs(key_decision="low"))["value"] is False


def test_read_mode_stays_in_detail_and_never_becomes_the_guard_value():
    """읽은 OM/SEM 은 detail/provenance 에만 있다 - v1 signature 밖이다."""
    for mode in ("OM", "SEM"):
        guard = align_key_guard(**_match_kwargs(mode=mode))
        assert guard["detail"]["mode"] == mode
        assert guard["value"] is True  # mode 가 무엇이든 Guard 값은 같다.
        assert mode not in str(guard["kind"])


def test_ok_control_precondition_is_three_state_and_not_a_guard():
    """OK 컨트롤 가용성도 3상태이고, Guard 목록에 들어가지 않는다."""
    assert ok_control_precondition(
        ok_screen_xy=(1, 2), correction_status="corrected")["value"] is True
    assert ok_control_precondition(
        ok_screen_xy=None, correction_status="escalated_no_ok")["value"] is False
    unknown = ok_control_precondition(ok_screen_xy=None, correction_status="ok_detect_error")
    assert unknown["value"] is None
    assert ok_control_precondition(
        ok_screen_xy=None, correction_status="corrected")["kind"] not in GUARD_KINDS


# ------------------------------------------------------------------
# 사이클 배선 - attempt 폴더에 실제로 남는가.
# ------------------------------------------------------------------


def test_cycle_writes_guards_into_the_attempt_folder(tmp_path, monkeypatch):
    """수집 on 이면 attempt 폴더에 guards.json 이 남고 evidence 가 Episode-relative 다."""
    import dataclasses

    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    settings = dataclasses.replace(load_workflow3_settings(), episode_collect_enabled=True)

    class _Outcome:
        status = "corrected"
        key_decision = "match"
        distinctive = True
        second_ratio = 0.4
        ok_screen_xy = (5, 6)

    class _Controller:
        mode_hint = "SEM"

    class _Meta:
        last_record = _sidecar()
        last_at = 0.0

    _Meta.last_at = __import__("time").time()
    context = {
        "eqp_id": "EQP1", "recipe_id": "CLS/RCP", "tag": "T1", "attempt_seq": 2,
        "occupancy": FREE, "share_status": "", "outcome": _Outcome(),
        "controller": _Controller(), "frame_meta": _Meta(),
    }
    result = cycle.CycleResult(eqp_id="EQP1", recipe_id="CLS/RCP", tag="T1")
    cycle.write_attempt_guards(context, result, settings)

    path = (tmp_path / "EQP1" / "CLS" / "RCP" / "captured_img_from_rcs" / "T1"
            / "attempt_2" / "guards.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [g["kind"] for g in data["guards"]] == list(GUARD_KINDS)
    assert [g["value"] for g in data["guards"]] == [True, True, True]
    assert data["guards"][0]["evidence"] == "attempt_2/recording/frame_meta.jsonl"
    assert data["preconditions"][0]["value"] is True


def test_broken_cycle_still_records_three_unknown_guards(tmp_path, monkeypatch):
    """관측이 하나도 없던 attempt 도 Guard 셋을 남긴다 - 전부 unknown 이 정직한 기록이다."""
    import dataclasses

    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    settings = dataclasses.replace(load_workflow3_settings(), episode_collect_enabled=True)
    context = {"eqp_id": "EQP1", "recipe_id": "", "tag": "T1", "attempt_seq": 1}
    result = cycle.CycleResult(eqp_id="EQP1", recipe_id="", tag="T1")
    cycle.write_attempt_guards(context, result, settings)

    path = tmp_path / "EQP1" / "_unregistered" / "T1" / "attempt_1" / "guards.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [g["value"] for g in data["guards"]] == [None, None, None]


def test_collection_off_writes_no_guard_file(tmp_path, monkeypatch):
    """수집 off 면 Guard 파일도 없다(플래그 뒤의 추가 동작)."""
    import dataclasses

    from poc.workflow_3.config import load_workflow3_settings
    from poc.workflow_3.monitor import cycle

    monkeypatch.setattr(cycle, "ALIGN_IMAGES_DIR", tmp_path)
    settings = dataclasses.replace(load_workflow3_settings(), episode_collect_enabled=False)
    context = {"eqp_id": "EQP1", "recipe_id": "", "tag": "T1", "attempt_seq": 1}
    cycle.write_attempt_guards(
        context, cycle.CycleResult(eqp_id="EQP1", recipe_id="", tag="T1"), settings
    )
    assert list(tmp_path.rglob("guards.json")) == []
