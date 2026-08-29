"""Episode-level Recovery Guard 판독 - 정확히 세 종류, 3상태.

Recovery Guard 는 "이 사전 상태에서 그 Action 을 골라도 되는가" 를 가르는 관측값이다.
값은 `True`/`False`/`None`(=unknown)이며, **관측하지 못한 것은 `False` 가 아니라 unknown**
이다. 이 구분이 없으면 "못 봤다" 가 "아니다" 로 둔갑해 rule 선택이 근거 없이 넓어진다.

`None` 을 쓰는 이유는 JSON `null` 이 unknown 을 명시적으로 나르고, 파이썬에서 실수로
`if value:` 를 써도 unknown 이 활성화 쪽으로 새지 않기 때문이다(문자열 `"false"` 는
truthy 라 정반대로 샌다).

Guard 는 셋뿐이고 늘리는 경로가 없다 - 새 Guard 종류는 새 observation contract 버전이다.

  1. `screen_observability` - 그 순간 우리 창이 실제로 보이고 있었는가(사이드카 가림 + rect).
  2. `occupancy_control`    - 그 tool 을 다른 엔지니어가 쥐고 있는가(List 행 점유 3상태).
  3. `align_key_visibility` - mode 를 읽었고, 그 mode 의 템플릿이 매칭됐고, key 가 유일한가.

읽은 OM/SEM 값은 **Guard 값이 아니라** detail 에만 남는다(v1 rule signature 밖). OK 컨트롤
가용성도 Guard 가 아니라 `confirm_align` 의 precondition 이다 - Episode 상태와 Action
전제조건을 섞으면 "컨트롤이 없었다" 가 "복구에 실패했다" 로 읽힌다.

판정 함수는 전부 순수 함수라 실장비 없이 시험된다. 화면/OCR/matcher 호출은 하지 않고
호출부가 이미 얻은 값을 받아 분류만 한다.
"""

import json
import os
import time
from pathlib import Path

GUARD_SCREEN = "screen_observability"
GUARD_OCCUPANCY = "occupancy_control"
GUARD_ALIGN_KEY = "align_key_visibility"
# 닫힌 집합. 여기 없는 kind 를 만드는 경로는 존재하지 않는다.
GUARD_KINDS = (GUARD_SCREEN, GUARD_OCCUPANCY, GUARD_ALIGN_KEY)

GUARDS_FILENAME = "guards.json"
GUARDS_SCHEMA_VERSION = "recovery_guards.v1"

# 사이드카 레코드가 이보다 오래되면 그 프레임으로 지금을 말할 수 없다 -> unknown.
DEFAULT_SIDECAR_MAX_AGE_SEC = 30.0

# 키가 '보인다' 고 볼 matcher decision. `adjust` 는 구조 유일성이 있을 때만이다
# (align/correction.py 의 key_visibility_gate 와 같은 기준).
_PRESENT_DECISIONS = ("match", "adjust")
# 만성 모호로 자동 보정을 보류한 status - 키는 보이지만 유일하지 않다.
_AMBIGUOUS_STATUS = "escalated_ambiguous_key"
# 자산이 없어 매칭 자체를 못 한 status.
_NO_ASSETS_STATUS = "no_assets"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _reading(kind: str, value, reason: str, evidence: str = "", **detail) -> dict:
    """Guard/precondition reading 1건. `value` 는 True/False/None 뿐이다."""
    return {
        "kind": kind,
        "value": value,
        "reason": reason,
        "observed_at": _now_iso(),
        "evidence": str(evidence or ""),
        "detail": detail,
    }


def screen_observability_guard(
    sidecar_record,
    *,
    age_sec: float,
    max_age_sec: float = DEFAULT_SIDECAR_MAX_AGE_SEC,
    evidence: str = "",
) -> dict:
    """① 화면 관측 가능성 - 프레임 사이드카의 가림 판정 + 창 rect 존재.

    `capture_window` 는 창 rect 의 스크린 그랩이라 다른 앱이 위에 뜨면 그 앱이 찍힌다.
    그래서 "그때 우리 창이 보이고 있었는가" 는 프레임과 함께 기록된 사이드카만이 답할 수
    있고, 사이드카가 없거나 낡았거나 판정 자체가 `unknown` 이면 여기도 unknown 이다.

    부분 가림(`partial`)도 `False` 다 - 화면 일부가 남의 창이었다면 그 프레임을 근거로
    "관측 가능했다" 고 말할 수 없다(fail-closed).
    """
    if not sidecar_record:
        return _reading(GUARD_SCREEN, None, "sidecar_missing", evidence)
    if age_sec is None or float(age_sec) > float(max_age_sec):
        return _reading(
            GUARD_SCREEN, None, "sidecar_stale", evidence, age_sec=age_sec,
        )
    rect = sidecar_record.get("window_rect")
    if not rect:
        return _reading(GUARD_SCREEN, None, "window_rect_missing", evidence)
    occlusion = str(sidecar_record.get("occlusion") or "unknown")
    if occlusion == "unknown":
        return _reading(GUARD_SCREEN, None, "occlusion_unreadable", evidence)
    if occlusion == "none":
        return _reading(GUARD_SCREEN, True, "window_fully_visible", evidence,
                        occlusion=occlusion)
    return _reading(GUARD_SCREEN, False, f"occluded:{occlusion}", evidence,
                    occlusion=occlusion)


def occupancy_guard(occupancy: str, *, share_status: str = "", evidence: str = "") -> dict:
    """② 점유/제어 - List 행 점유 3상태를 그대로 3상태로 옮긴다.

    `free` -> True, `occupied_by_other` -> False, `unknown` -> unknown. 판독 실패를
    '비어 있음' 으로 접으면 view-only 세션이 보정 가능 세션으로 오인된다
    (`rcs/row_occupant.py` 가 3상태인 이유와 같다).

    화면 공유 요청 결과는 provenance 로만 남는다 - 공유를 받아도 그것은 **관전**이지
    제어가 아니므로 Guard 값을 True 로 올리지 않는다.
    """
    from poc.workflow_3.rcs.row_occupant import FREE, OCCUPIED_BY_OTHER

    state = str(occupancy or "")
    detail = {"occupancy": state, "share_status": str(share_status or "")}
    if state == FREE:
        return _reading(GUARD_OCCUPANCY, True, "row_free", evidence, **detail)
    if state == OCCUPIED_BY_OTHER:
        return _reading(GUARD_OCCUPANCY, False, "occupied_by_other", evidence, **detail)
    return _reading(GUARD_OCCUPANCY, None, "occupancy_unreadable", evidence, **detail)


def align_key_guard(
    *,
    mode: str,
    key_decision: str,
    distinctive: bool,
    second_ratio,
    matcher_error=None,
    correction_status: str = "",
    evidence: str = "",
) -> dict:
    """③ SEM mode + align key 가시성/유일성 - 셋이 **모두** 성립할 때만 True.

    unknown 으로 가는 경우: mode 미판독(어느 template 을 봤는지 모른다), 자산 없음,
    matcher 예외, matcher 미실행, 유일성 미판독(`second_ratio` 없음). 마지막이 중요한데,
    matcher 의 `distinctive` 는 데이터 결손 시 false-flag 를 피하려고 True 로 기본값을
    갖기 때문이다 - Guard 에서는 그 True 를 '유일함이 확인됐다' 로 읽으면 안 된다.

    False 로 가는 경우: 키가 안 보임(관측된 부정), 만성 모호(`escalated_ambiguous_key`),
    구조 유일성 없는 약한 후보(candidate). 셋 다 "봤는데 조건을 만족하지 않았다" 이다.

    읽은 OM/SEM 은 detail 에만 남는다 - v1 에서 mode 는 rule signature 의 일부가 아니다.
    """
    detail = {
        "mode": str(mode or ""),
        "key_decision": str(key_decision or ""),
        "distinctive": bool(distinctive),
        "second_ratio": second_ratio,
        "correction_status": str(correction_status or ""),
    }
    if not mode:
        return _reading(GUARD_ALIGN_KEY, None, "mode_unread", evidence, **detail)
    if correction_status == _NO_ASSETS_STATUS:
        return _reading(GUARD_ALIGN_KEY, None, "no_assets", evidence, **detail)
    if matcher_error:
        return _reading(GUARD_ALIGN_KEY, None, f"matcher_error:{matcher_error}",
                        evidence, **detail)
    if not key_decision:
        return _reading(GUARD_ALIGN_KEY, None, "matcher_not_run", evidence, **detail)

    present = key_decision in _PRESENT_DECISIONS and (
        key_decision != "adjust" or bool(distinctive)
    )
    if not present:
        return _reading(GUARD_ALIGN_KEY, False, "key_not_visible", evidence, **detail)
    if correction_status == _AMBIGUOUS_STATUS or not distinctive:
        return _reading(GUARD_ALIGN_KEY, False, "key_ambiguous", evidence, **detail)
    if second_ratio is None:
        return _reading(GUARD_ALIGN_KEY, None, "uniqueness_unreadable", evidence, **detail)
    return _reading(GUARD_ALIGN_KEY, True, "key_visible_and_unique", evidence, **detail)


def ok_control_precondition(*, ok_screen_xy, correction_status: str = "",
                            evidence: str = "") -> dict:
    """OK 컨트롤 가용성 - **Guard 가 아니라** `confirm_align` 의 전제조건 기록.

    Episode 상태와 Action 전제조건을 섞으면 "컨트롤을 못 찾았다" 가 "복구에 실패했다" 로
    읽힌다. 그래서 같은 파일에 담기되 목록을 나눈다.
    """
    detail = {"correction_status": str(correction_status or "")}
    if ok_screen_xy:
        return _reading("ok_control_available", True, "ok_button_located", evidence,
                        **detail)
    if correction_status == "escalated_no_ok":
        return _reading("ok_control_available", False, "ok_button_absent", evidence,
                        **detail)
    return _reading("ok_control_available", None, "ok_button_unreadable", evidence,
                    **detail)


def write_guard_records(attempt_dir, *, attempt_seq: int, guards, preconditions=()) -> Path:
    """attempt 폴더에 `guards.json` 을 원자적으로 쓰고 그 경로를 돌려준다."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GUARDS_SCHEMA_VERSION,
        "attempt_seq": int(attempt_seq),
        "guards": list(guards),
        "preconditions": list(preconditions),
    }
    path = attempt_dir / GUARDS_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


__all__ = [
    "GUARDS_FILENAME",
    "GUARDS_SCHEMA_VERSION",
    "GUARD_ALIGN_KEY",
    "GUARD_KINDS",
    "GUARD_OCCUPANCY",
    "GUARD_SCREEN",
    "align_key_guard",
    "occupancy_guard",
    "ok_control_precondition",
    "screen_observability_guard",
    "write_guard_records",
]
