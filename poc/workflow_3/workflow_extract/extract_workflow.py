"""workflow_extract 엔트리포인트 - 타임라인을 workflow.json + workflow.md 로.

CLI 인자 없음(프로젝트 규칙). 입력은 env/자동탐색으로, 산출은 입력 폴더에 쓴다.
VLM 을 부르지 않으므로 임계값을 바꿔가며 몇 번이든 재실행할 수 있다.

실행:
    uv run python poc/workflow_3/workflow_extract/extract_workflow.py
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.debug_artifacts import save_debug_json
from poc.workflow_3.workflow_extract.grouping import GroupingContext, group_events
from poc.workflow_3.workflow_extract.render import render_markdown
from poc.workflow_3.workflow_extract.settings import load_workflow_extract_settings


def _read_json(path):
    """JSON 을 읽는다. 없거나 깨졌으면 None (호출부가 degrade 를 결정한다)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_input_dir():
    """분석할 recording_filter/ 폴더를 결정한다(env -> 자동탐색)."""
    env_path = os.getenv("WORKFLOW_EXTRACT_INPUT_DIR", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_dir():
            return path.resolve()
        print(f"[ERROR] WORKFLOW_EXTRACT_INPUT_DIR 디렉터리를 찾지 못했습니다: {path}")
        return None
    candidates = sorted(
        ALIGN_IMAGES_DIR.glob("*/_manual/*/recording_filter"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        latest = candidates[0].resolve()
        print(f"[INFO] 최신 recording_filter/ 자동 선택: {latest}")
        return latest
    print(f"[ERROR] 분석할 recording_filter/ 폴더를 찾지 못했습니다(루트: {ALIGN_IMAGES_DIR}).")
    return None


def _load_live_boxes(out_dir):
    """region_map.json 에서 {generation: live_box} 를 만든다. 없으면 빈 dict.

    entry 의 live_box 는 그 세대의 detect_sem_box 가 실패했으면 None 이다 - "탐지
    실패로 없음"이지 "박스가 존재하되 falsy 인 값"이 아니므로 `is not None` 으로만
    걸러낸다(단순 truthy 체크도 결과는 같지만, 의도가 "None=결측"임을 코드에서
    분명히 하기 위해 명시적으로 쓴다). generation 은 0 이 가장 흔한 정상 값이라
    `int(... or 0)` 폴백은 "없음"과 "명시적 0" 을 같은 결과(0)로 수렴시키기 위한
    것이지, 0 을 걸러내려는 게 아니다.
    """
    payload = _read_json(Path(out_dir) / "region_map.json")
    if not payload:
        print("[WARNING] region_map.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return {}
    boxes = {}
    for entry in payload.get("maps") or []:
        if entry.get("live_box") is not None:
            boxes[int(entry.get("generation") or 0)] = entry["live_box"]
    return boxes


def _load_changes(out_dir):
    """change_events.json 의 events 를 돌려준다. 없으면 빈 목록."""
    payload = _read_json(Path(out_dir) / "change_events.json")
    if not payload:
        print("[WARNING] change_events.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return []
    return payload.get("events") or []


def _frame_size(capture_dir):
    """프레임 하나를 열어 (w, h) 를 얻는다. 실패하면 None (R2 degrade)."""
    from poc.workflow_3.recording_filter.region_gate import read_frame_size

    try:
        for frame in sorted(Path(capture_dir).glob("*.jpg")):
            size = read_frame_size(frame)
            if size:
                return size
    except Exception:
        pass
    print("[WARNING] 프레임 크기를 얻지 못했습니다 - R2(드롭다운)가 비활성화됩니다.")
    return None


def _resolve_frame_wh(capture_dir):
    """R2 가 쓸 프레임 크기를 구한다. capture_dir 자체가 없으면 경고 후 None.

    capture_dir 가 빈 문자열이면 `Path("").glob(...)` 이 현재 작업 디렉터리를
    훑어 버려 엉뚱한 폴더의 jpg 를 프레임으로 오인할 수 있다 - 그래서 _frame_size
    호출 자체를 건너뛰지만, "왜 R2 가 꺼졌는지"는 여전히 알려야 하므로 이 갈래에서도
    경고를 남긴다(호출을 건너뛰면 _frame_size 내부 경고도 함께 사라지기 때문).
    """
    if not capture_dir:
        print("[WARNING] capture_dir 가 없습니다 - R2(드롭다운)가 비활성화됩니다.")
        return None
    return _frame_size(capture_dir)


def _eqp_id_from_capture_dir(capture_dir):
    """capture_dir 경로에서 장비 id 를 뽑는다.

    수동 녹화 레이아웃은 `<root>/<eqp>/_manual/<tag>/recording` 이라 `_manual`
    바로 앞 컴포넌트가 장비 id 다. 고정 음수 인덱스(parts[-4])는 이 레이아웃에서만
    우연히 맞고, 다른 캡처 레이아웃(`<root>/<eqp>/<class>/<recipe>/captured_img_from_rcs/
    <tag>/recording`)에서는 조용히 틀린 라벨을 낸다. 엔지니어가 확인해야 하는 문서에
    틀린 장비 id 를 박아 넣느니 `"?"` 로 정직하게 모른다고 하는 편이 낫다.
    """
    if not capture_dir:
        return "?"
    parts = Path(capture_dir).parts
    for i, part in enumerate(parts):
        if part == "_manual" and i > 0:
            return parts[i - 1]
    return "?"


def run_extract(*, input_dir=None, settings=None) -> str:
    """타임라인을 workflow.json + workflow.md 로 만든다. 상태 문자열 반환."""
    settings = settings or load_workflow_extract_settings()
    out_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if out_dir is None:
        return "input_not_found"

    timeline_payload = _read_json(out_dir / "interaction_timeline.json")
    if not timeline_payload:
        print(
            f"[ERROR] interaction_timeline.json 이 없습니다: {out_dir}\n"
            "        먼저 filter_recording.py 를 실행하세요."
        )
        return "timeline_not_found"

    events = timeline_payload.get("events") or []
    if not events:
        print("[ERROR] 타임라인에 이벤트가 0건입니다 - 추출할 절차가 없습니다.")
        return "no_events"

    capture_dir = timeline_payload.get("capture_dir") or ""
    ctx = GroupingContext(
        settings=settings,
        live_boxes=_load_live_boxes(out_dir),
        changes=_load_changes(out_dir),
        frame_wh=_resolve_frame_wh(capture_dir),
    )
    steps = group_events(events, ctx)

    duration = max(float(e["t_sec"]) for e in events)
    session = {
        "eqp_id": _eqp_id_from_capture_dir(capture_dir),
        "tag": Path(capture_dir).parent.name if capture_dir else "?",
        "capture_dir": capture_dir,
        "total_events": len(events),
        "duration_sec": duration,
    }

    save_debug_json(
        out_dir / "workflow.json",
        {"session": session, "settings": asdict(settings), "steps": steps},
    )
    (out_dir / "workflow.md").write_text(
        render_markdown(steps, session), encoding="utf-8"
    )
    print(
        f"[INFO] 완료: 이벤트 {len(events)} 건 -> step {len(steps)} 건, out={out_dir}"
    )
    return "success"


if __name__ == "__main__":
    result = run_extract()
    raise SystemExit(0 if result == "success" else 1)
