"""조작 미확인 화면 변화 - 클릭/타이핑으로 설명되지 않은 변화를 버리지 않고 남긴다.

엔지니어는 녹화 PC 가 아닌 곳에서 조작하므로 조작의 증거는 프레임뿐이고, 클릭은
"프레임 안 커서 + 그 주변 변화"로만 판정된다(Stage 2a). 커서를 못 찾았거나 변화가
커서에서 먼 곳에 나타나면 그 변화는 클릭이 되지 못하는데, 종전에는 타임라인에서
아예 사라졌다 - 놓친 클릭이 절차서에 흔적조차 남기지 않았다.

여기서 만드는 `screen_change` 이벤트는 **관측이지 동작이 아니다**: 좌표도 라벨도
없고 재생할 수 없다. 장비가 스스로 바꾸는 화면(카운터, 상태 문구)도 같이 올라온다 -
사람이 일으킨 변화인지는 이 단계에서 알 수 없으므로 가르지 않고 사유만 적는다.
"""

from pathlib import Path

from poc.workflow_3.debug_artifacts import save_marked_bboxes
from poc.workflow_3.recording_filter.type_detect import _union_box


REASON_CURSOR_NOT_FOUND = "cursor_not_found"   # 커서를 못 찾아 클릭 여부를 판정하지 못했다.
REASON_CURSOR_ELSEWHERE = "cursor_elsewhere"   # 커서는 찾았지만 변화가 그 주변이 아니었다.


def _reason(judged) -> str:
    """Stage 2a 판정에서 '왜 클릭이 되지 못했는지'를 읽는다.

    좌표가 있어도 `no_click` 이 아니면 커서를 찾은 것이 아니다 - 고정 그래픽 오탐
    (`cursor_static_decoy`)은 그 좌표가 무효라는 뜻이다.
    """
    if judged.status == "no_click" and judged.cursor_xy:
        return REASON_CURSOR_ELSEWHERE
    return REASON_CURSOR_NOT_FOUND


def _event(change, gate, reason) -> dict:
    """변화 1건을 타임라인 스키마의 screen_change 이벤트로 만든다."""
    return {
        "t_sec": change.timestamp_sec,
        "t_sec_end": change.timestamp_sec,
        "seq": 0,
        "action": "screen_change",
        "coords": None,
        "element": None,
        "element_source": "none",
        "target_kind": "unknown",
        "region": str(gate.get("region") or "unknown"),
        "generation": int(gate.get("generation") or 0),
        "occlusion": str(gate.get("occlusion") or "unknown"),
        "cursor_source": "none",
        "text": None,
        "confidence": 0.0,
        "frame": Path(change.frame_path).name,
        "source_frames": {
            "prev": Path(change.prev_frame_path).name,
            "curr": Path(change.frame_path).name,
        },
        "reasons": [reason],
        "change_bbox": dict(change.change_bbox),
        "change_ranks": [change.rank],
    }


def _extend(event, change, reason) -> None:
    """이어진 변화를 같은 이벤트에 합친다(끝 시각, 영역 합집합, 사유, 마지막 프레임)."""
    event["t_sec_end"] = change.timestamp_sec
    event["change_bbox"] = _union_box(event["change_bbox"], change.change_bbox)
    event["change_ranks"].append(change.rank)
    event["reasons"] = sorted(set(event["reasons"]) | {reason})
    event["source_frames"]["curr"] = Path(change.frame_path).name


def find_unattributed_changes(change_events, click_events, claimed_ranks, gate_info, settings) -> list:
    """클릭/타이핑이 가져가지 않은 변화를 screen_change 이벤트 목록으로 돌려준다.

    claimed_ranks 는 클릭이 아닌 다른 해석이 이미 가져간 변화의 rank 다(타이핑 구간,
    닫기 정황) - 같은 변화를 두 번 보고하지 않는다.

    가까운 변화는 하나로 묶기만 하고, **앞선 동작이 있다는 이유로 버리지 않는다** -
    드롭다운을 연 클릭 직후의 변화가 그 클릭의 결과 화면인지, 놓친 항목 선택인지는
    시간 간격만으로 가릴 수 없다(2026-10-02 Codex 설계 리뷰).
    """
    judged_by_rank = {ce.rank: ce for ce in click_events}
    events = []
    current = None
    for change in change_events:
        judged = judged_by_rank.get(change.rank)
        if judged is None or judged.is_click or change.rank in claimed_ranks:
            current = None      # 남의 변화를 건너 묶으면 타임라인 순서가 뒤집힌다.
            continue
        reason = _reason(judged)
        gate = gate_info.get(change.rank) or {}
        if (
            current is not None
            # 음수 간격(시간 순서가 뒤집힌 입력)은 가까운 것이 아니다.
            and 0 <= change.timestamp_sec - current["t_sec_end"] <= settings.unattributed_merge_gap_sec
            and int(gate.get("generation") or 0) == current["generation"]
        ):
            _extend(current, change, reason)
            continue
        current = _event(change, gate, reason)
        events.append(current)
    return events


def write_unattributed_overlays(events, change_events, out_dir: Path) -> None:
    """관측마다 마지막 프레임에 변화 영역을 그려 저장한다(엔지니어 대조용).

    관측이 없으면 폴더도 만들지 않는다 - 빈 폴더는 '돌았는데 없었다'와 '안 돌았다'를
    구분하지 못하게 한다.
    """
    from PIL import Image

    frame_by_rank = {change.rank: change.frame_path for change in change_events}
    for event in events:
        ranks = event["change_ranks"]
        try:
            image = Image.open(frame_by_rank[ranks[-1]]).convert("RGB")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            save_marked_bboxes(
                image, {"change": {"bbox": event["change_bbox"]}}, {"change": "orange"},
                Path(out_dir) / f"{ranks[0]:03d}_{event['source_frames']['curr']}",
            )
        except Exception as exc:
            # 오버레이는 부가 산출물이다 - 프레임 하나가 나쁘다고 타임라인을 잃으면 안 된다.
            print(f"[WARNING] 화면 변화 오버레이 저장 실패(건너뜀, rank={ranks[0]}): {exc}")
