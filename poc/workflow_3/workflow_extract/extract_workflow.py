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


def _diagnose_read_failure(path):
    """_read_json 이 None 을 돌려준 이유를 "없음"과 "있는데 못 읽음"으로 가른다.

    _read_json 자체는 두 실패를 하나로 뭉뚱그린다 - 호출부가 결정해야 하는 것은
    어차피 "degrade 할지 여부" 하나뿐이라 그걸로 충분하다. 하지만 사람에게 보여줄
    진단 문구는 다르다. 파일이 있는데 깨졌을 때 "없습니다"라고 알리면, 사람은 있지도
    않은 파일을 찾아 헤매고 실제 원인(손상된 JSON)은 못 보고 지나간다 - 실제로
    일어난 일을 정직하게 보고한다는 이 프로젝트 컨벤션 위반이다.

    반환값: 파일이 정말 없으면 None(호출부가 기존 "없습니다" 문구를 그대로 쓴다),
    있는데 읽기/파싱에 실패했으면 그 예외 메시지 문자열.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return str(exc)
    # 파싱은 성공했는데 호출부가 None/falsy 를 받은 경우 - 내용이 {} / [] / null 이라
    # `if not payload` 에 걸린 것이다. 이때 "손상되었을 수 있습니다" 는 거짓말이고,
    # 정직한 진단 함수가 유일하게 거짓말하는 자리였다(2026-08-11 리뷰 낮은 우선순위).
    if not payload:
        return f"JSON 은 정상인데 내용이 비어 있습니다({payload!r}) - 파일 손상이 아닙니다"
    return "원인을 다시 파악하지 못했습니다(일시적 문제일 수 있음)"


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

    최상위 키는 생산자와 공유하는 상수(`region_gate.REGION_MAP_KEY`)로 읽는다 -
    예전에는 이쪽만 "maps" 라고 적어 두어, 파일이 정상 파싱되는데도 박스가 0개라
    R1 이 모든 실행에서 죽어 있었는데 경고조차 없었다(2026-08-11 리뷰 C1).

    entry 의 live_box 는 그 세대의 detect_sem_box 가 실패했으면 None 이다 - "탐지
    실패로 없음"이지 "박스가 존재하되 falsy 인 값"이 아니므로 `is not None` 으로만
    걸러낸다(단순 truthy 체크도 결과는 같지만, 의도가 "None=결측"임을 코드에서
    분명히 하기 위해 명시적으로 쓴다). generation 은 0 이 가장 흔한 정상 값이라
    `int(... or 0)` 폴백은 "없음"과 "명시적 0" 을 같은 결과(0)로 수렴시키기 위한
    것이지, 0 을 걸러내려는 게 아니다.

    파싱은 됐는데 박스가 0개인 경우에도 경고한다 - "박스 없음"을 조용히 빈 dict 로
    돌려주는 것이 C1 을 13개 태스크 동안 안 보이게 만든 원인이다.
    """
    from poc.workflow_3.recording_filter.region_gate import REGION_MAP_KEY

    path = Path(out_dir) / "region_map.json"
    payload = _read_json(path)
    if not payload:
        parse_error = _diagnose_read_failure(path)
        if parse_error:
            print(
                f"[WARNING] region_map.json 이 있지만 읽지 못했습니다({parse_error}) - "
                "R1(FOV 더블클릭)이 비활성화됩니다."
            )
        else:
            print("[WARNING] region_map.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return {}
    entries = payload.get(REGION_MAP_KEY) or []
    boxes = {}
    for entry in entries:
        if entry.get("live_box") is not None:
            boxes[int(entry.get("generation") or 0)] = entry["live_box"]
    if not boxes:
        print(
            f"[WARNING] region_map.json 을 읽었지만 live_box 가 0개입니다"
            f"(키 '{REGION_MAP_KEY}' 항목 {len(entries)} 건) - R1(FOV 더블클릭)이 "
            "비활성화되고 coords_in_live_box 가 전부 null 이 됩니다. "
            "region_map_gen0.jpg 와 detect_sem_box 결과를 먼저 확인하세요."
        )
    return boxes


def _load_changes(out_dir):
    """change_events.json 에서 R1 이 근거로 쓸 변화 이벤트만 돌려준다.

    (2026-08-11 리뷰 I1) 이 파일은 Stage 1 전체를 담는다 - Stage 1.5 가 걷어낸
    `ambient`(라이브 박스 안에서만 변했고 커서는 밖 = 라이브 SEM 영상의 자율 갱신)
    까지 들어 있고, README 기준 그쪽이 90% 이상이다. R1 의 recenter 시그니처는
    "라이브 박스 대부분이 다시 그려졌다"인데 ambient 전면 리페인트는 그 비율이 ~1.0
    이라 구분이 불가능하다 - 걸러내지 않으면 라이브 박스 근처 클릭이 평범한 영상
    갱신 때문에 `double_click(intent=fov_move, inferred=true)` 로 승격된다. 비율
    임계(recenter_min_ratio)를 올려서는 막을 수 없다.

    그래서 Stage 1.5 가 이미 계산해 둔 판정을 그대로 쓴다(filter_recording 이
    change_events.json 의 각 이벤트에 verdict/occlusion 을 함께 적는다). 판정
    필드가 아예 없는 예전 형식은 "전부 통과"로 degrade 하지 않고 R1 을 끈다 -
    ambient 를 근거로 삼는 것보다 규칙이 안 도는 편이 낫고, 대신 경고를 남긴다.
    """
    path = Path(out_dir) / "change_events.json"
    payload = _read_json(path)
    if not payload:
        parse_error = _diagnose_read_failure(path)
        if parse_error:
            print(
                f"[WARNING] change_events.json 이 있지만 읽지 못했습니다({parse_error}) - "
                "R1(FOV 더블클릭)이 비활성화됩니다."
            )
        else:
            print("[WARNING] change_events.json 이 없습니다 - R1(FOV 더블클릭)이 비활성화됩니다.")
        return []
    events = payload.get("events") or []
    if not events:
        return []
    with_verdict = [ev for ev in events if ev.get("verdict")]
    if not with_verdict:
        print(
            f"[WARNING] change_events.json 의 {len(events)} 건에 Stage 1.5 판정(verdict)이 "
            "없습니다(예전 형식이거나 영역 게이트를 끄고 돌린 산출물) - ambient 를 "
            "recenter 근거로 오인하지 않도록 R1(FOV 더블클릭)을 비활성화합니다. "
            "filter_recording.py 를 다시 실행하세요."
        )
        return []
    candidates = [
        ev for ev in with_verdict
        if ev["verdict"] == "candidate" and ev.get("occlusion") != "full"
    ]
    print(
        f"[INFO] R1 근거 변화 이벤트: {len(candidates)} / {len(events)} 건"
        f"(ambient/가림 {len(events) - len(candidates)} 건 제외)"
    )
    return candidates


def _frame_size(capture_dir):
    """프레임 하나를 열어 (w, h) 를 얻는다. 실패하면 None (R2 degrade).

    프레임은 녹화 루트에 바로 있거나 `frames/` 하위에 있다 - filter_recording
    (`_resolve_frames_dir`)이 두 레이아웃을 모두 받으므로 이쪽도 같이 봐야 한다.
    루트만 보면 frames/ 레이아웃에서 R2 가 조용히 꺼진다(경고는 나오지만 원인이
    "프레임 없음" 으로만 보여, 실제 원인인 레이아웃 차이를 짚기 어렵다).
    """
    from poc.workflow_3.recording_filter.region_gate import read_frame_size

    try:
        roots = [Path(capture_dir), Path(capture_dir) / "frames"]
        for root in roots:
            for frame in sorted(root.glob("*.jpg")):
                size = read_frame_size(frame)
                if size:
                    return size
    except Exception:
        pass
    print(
        f"[WARNING] 프레임 크기를 얻지 못했습니다(찾은 곳: {capture_dir} 및 그 안의 "
        "frames/) - R2(드롭다운)가 비활성화됩니다."
    )
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


def _event_end_sec(event) -> float:
    """이벤트가 끝난 시각. 타이핑 구간은 t_sec_end 를 쓴다.

    타이핑 이벤트는 구간이라 시작 시각 하나로는 길이를 잃는다 - 마지막 조작이 긴
    타이핑이면 세션 길이가 그만큼 짧게 보고된다. `is not None` 으로 판정하는 이유는
    `steps.make_step` 과 같다(정상적인 t_sec_end == 0.0 을 "없음"으로 버리지 않는다).
    """
    end = event.get("t_sec_end")
    start = float(event["t_sec"])
    if end is None:
        return start
    return max(start, float(end))


def run_extract(*, input_dir=None, settings=None) -> str:
    """타임라인을 workflow.json + workflow.md 로 만든다. 상태 문자열 반환."""
    settings = settings or load_workflow_extract_settings()
    out_dir = Path(input_dir).resolve() if input_dir else _resolve_input_dir()
    if out_dir is None:
        return "input_not_found"

    timeline_path = out_dir / "interaction_timeline.json"
    timeline_payload = _read_json(timeline_path)
    if not timeline_payload:
        parse_error = _diagnose_read_failure(timeline_path)
        if parse_error:
            # 원인 문구는 _diagnose_read_failure 가 실제로 관찰한 것만 말한다 -
            # 예전에는 여기서 무조건 "손상되었을 수 있습니다" 라고 덧붙여, 내용이
            # 비었을 뿐인 정상 JSON 에도 손상을 의심하게 만들었다.
            print(
                f"[ERROR] interaction_timeline.json 이 있지만 읽지 못했습니다: {timeline_path}\n"
                f"        원인: {parse_error}\n"
                "        filter_recording.py 를 다시 실행하거나 파일을 직접 확인하세요."
            )
        else:
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

    duration = max(_event_end_sec(e) for e in events)
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


def _exit_code(status: str) -> int:
    """상태 문자열을 프로세스 종료 코드로 매핑한다("success" 만 0, 그 외 전부 1).

    이 매핑을 `__main__` 안에 인라인으로 두면 아무도 직접 테스트할 수 없다 - 나중에
    누군가 실수로 뒤집거나(`SystemExit(0)` 로 고정) "success" 집합에 상태를 잘못
    추가해도 상태 문자열만 검사하는 테스트는 전부 통과한 채로 파이프라인이 빈
    절차서를 만들어 놓고 종료 코드 0(성공)을 돌려주는, degrade 계약이 막으려는
    바로 그 실패 모드가 조용히 통과한다. 이름 붙은 함수로 빼서 subprocess 없이
    직접 고정할 수 있게 한다.
    """
    return 0 if status == "success" else 1


if __name__ == "__main__":
    result = run_extract()
    raise SystemExit(_exit_code(result))
