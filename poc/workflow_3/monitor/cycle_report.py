"""사이클 판정 요약 - tool 창을 닫고 빠져나온 직후 콘솔에 한 장으로 찍는다.

오피스에서 실시간으로 보는 것은 콘솔뿐이라, "이번 테이크가 성공인가" 를 판단하는 데
필요한 신호(SEM box 검출, OM/SEM 판독, align 매칭 점수와 좌표, 보정 결과)가 사이클
로그 수백 줄에 흩어져 있으면 사실상 안 보인다. 이 모듈은 그 신호만 모아 한 블록으로
낸다.

두 가지가 계약이다.

  1) **보고서는 절대 사이클을 죽이지 않는다.** teardown 뒤에 도는 코드라, 여기서 뜬
     예외는 이미 끝난 사이클의 결과를 통째로 날린다. 모든 값은 방어적으로 읽는다.
  2) **ASCII 마커만 쓴다.** 오피스 콘솔은 cp949 라 이모지와 em-dash 를 인코딩하지
     못한다(`print` 가 UnicodeEncodeError 로 죽는다).
"""

_WIDTH = 66

# 정상 종료 status -> (마커, 사람이 읽는 판정). 접두사가 아니라 **정확 비교**로 먼저
# 찾는다: monitor 계층이 치환하는 status(view_only_observation/corrected_unverified)가
# fallback_ 접두사 분기로 새면 판정이 조용히 뒤집힌다(correction.py 의 경고 참조).
_VERDICTS = {
    "corrected": ("OK", "보정 완료 (reposition + OK 클릭)"),
    "awaiting_engineer_ok": ("OK", "reposition 완료 - OK 클릭은 엔지니어(반자동 모드)"),
    "corrected_unverified": ("!!", "보정했으나 화면 반영 미확인(점유/미상) - 직접 확인 필요"),
    "view_only_observation": ("!!", "다른 엔지니어 점유 - 관전/녹화만 (보정 안 함)"),
    "escalated_ambiguous_key": ("!!", "align key 가 만성 모호 - 자동 보정 보류"),
    "escalated_no_ok": ("!!", "OK 버튼을 찾지 못해 보류"),
    "ok_detect_error": ("XX", "OK 버튼 탐지 중 오류"),
    "no_assets": ("XX", "rcp align 이미지 없음 - ALIGN_IMAGES_DIR 확인"),
}

_OCCUPANCY_TEXT = {
    "free": "비어 있음",
    "occupied_by_other": "다른 엔지니어 점유",
    "unknown": "판독 실패(미상)",
}


def _fmt_elapsed(elapsed_sec) -> str:
    """초를 'Xm YYs' 로 만든다. 값이 없으면 '-'."""
    try:
        total = int(float(elapsed_sec))
    except (TypeError, ValueError):
        return "-"
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60:02d}s"


def _fmt_xy(xy) -> str:
    """(x, y) 를 '(x,y)' 로. None/형식 불량이면 '-'."""
    try:
        return f"({int(xy[0])},{int(xy[1])})"
    except (TypeError, ValueError, IndexError):
        return "-"


def _fmt_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _tail_path(path, keep: int = 3) -> str:
    """경로의 뒤 몇 조각만 남긴다 - 콘솔 폭 안에서 어느 폴더인지 알아보게."""
    text = str(path or "").replace("\\", "/").rstrip("/")
    if not text:
        return "-"
    parts = [p for p in text.split("/") if p]
    if len(parts) <= keep:
        return text
    return ".../" + "/".join(parts[-keep:])


def _paused_match(outcome) -> dict:
    """outcome.history 에서 마지막 paused_match 항목을 꺼낸다(없으면 빈 dict)."""
    history = getattr(outcome, "history", None) or []
    for entry in reversed(history):
        if isinstance(entry, dict) and entry.get("stage") == "paused_match":
            return entry
    return {}


def _verdict(result, outcome) -> tuple[str, str]:
    """마커와 한 줄 판정을 정한다. run 실패가 outcome 보다 우선한다."""
    run_status = getattr(result, "run_status", "") or ""
    if run_status not in ("completed", "success"):
        failed = getattr(result, "failed_step", "") or "-"
        klass = getattr(result, "failure_class", "") or "-"
        return "XX", f"사이클 미완주 (run={run_status}, step={failed}, class={klass})"

    status = getattr(outcome, "status", "") if outcome is not None else ""
    if not status:
        return "!!", "보정 결과 없음 (RECIPE_ID 없음 / 보정 비활성 / 접속만 함)"
    if status in _VERDICTS:
        return _VERDICTS[status]
    if status.startswith("fallback_"):
        return "!!", f"1차 매칭 실패 -> live search 폴백 ({status})"
    return "!!", f"알 수 없는 status: {status}"


def build_cycle_report(result, context, *, elapsed_sec=None) -> list[str]:
    """판정 카드를 줄 목록으로 만든다(순수 함수 - RCS/VLM 없이 테스트 가능)."""
    context = context or {}
    controller = context.get("controller")
    outcome = context.get("outcome")
    match = _paused_match(outcome)

    recipe = getattr(result, "recipe_id", "") or "(미등록)"
    lines = [
        "=" * _WIDTH,
        f"[판정] {getattr(result, 'eqp_id', '-')} / {recipe}"
        f"  tag={getattr(result, 'tag', '-')}  경과={_fmt_elapsed(elapsed_sec)}",
        "-" * _WIDTH,
    ]

    # --- SEM box + 모달리티 (보정이 옳은 화면/템플릿을 봤는지의 근거) ---
    panel = getattr(controller, "panel", None)
    if panel is not None:
        source = "VLM live box" if getattr(panel, "model_id", "") == "vlm_live_box" else "landmark"
        roi = getattr(panel, "panel_roi", None)
        roi_text = "-"
        try:
            roi_text = f"({roi[0]},{roi[1]}) {roi[2]}x{roi[3]}px"
        except (TypeError, IndexError):
            pass
        lines.append(
            f" SEM box  검출 O  roi={roi_text}"
            f"  conf={_fmt_float(getattr(panel, 'confidence', None))}  src={source}"
        )
        mode = getattr(controller, "mode_hint", None)
        lines.append(
            f" 모달리티 {mode} (화면 PM 판독)" if mode
            else f" 모달리티 판독 실패 -> 기본값 {getattr(controller, 'mode_default', '-')} 사용"
        )
    else:
        lines.append(" SEM box  검출 X (panel 미확보 - 보정 진행 안 됨)")

    # --- align 매칭: agent 가 key 를 어디서, 얼마나 확신하고 찾았나 ---
    if match:
        lines.append(
            f" 매칭     decision={match.get('decision', '-')}"
            f"  score={_fmt_float(match.get('score'))}"
            f"  chamfer={_fmt_float(match.get('chamfer'))}"
            f"  scale={_fmt_float(match.get('best_scale'), 2)}"
        )
    if outcome is not None:
        distinctive = getattr(outcome, "distinctive", None)
        lines.append(
            f"          2nd비={_fmt_float(getattr(outcome, 'second_ratio', None))}"
            f"  gap={_fmt_float(getattr(outcome, 'score_gap', None))}"
            f"  구별={'O' if distinctive else 'X'}"
        )
        lines.append(
            f" 위치     align={_fmt_xy(getattr(outcome, 'best_xy', None))} (FOV)"
            f"  OK버튼={_fmt_xy(getattr(outcome, 'ok_screen_xy', None))} (screen)"
        )
        lines.append(
            f" 보정     {getattr(outcome, 'status', '-')}"
            f"  (path={getattr(outcome, 'path', '-')})"
        )
        error = getattr(outcome, "error", None)
        if error:
            lines.append(f"          error={error}")

    occupancy = context.get("occupancy")
    if occupancy:
        lines.append(f" 점유     {_OCCUPANCY_TEXT.get(occupancy, occupancy)}")

    # --- 산출물 경로: 흩어진 폴더를 찾아다니지 않도록 여기 모아 찍는다 ---
    lines.append(
        f" 녹화     {getattr(result, 'frame_count', 0)} frames"
        f" -> {_tail_path(getattr(result, 'recording_dir', ''))}"
    )
    lines.append(f" 보정산출 -> {_tail_path(context.get('correction_debug_dir'))}")
    lines.append(f" step저널 -> {_tail_path(getattr(result, 'run_dir', ''))}")

    marker, text = _verdict(result, outcome)
    lines.append("-" * _WIDTH)
    lines.append(f" [{marker}] {text}")
    lines.append("=" * _WIDTH)
    return lines


def print_cycle_report(result, context, *, elapsed_sec=None) -> None:
    """판정 카드를 콘솔에 찍는다. 여기서 나는 예외는 사이클 결과를 삼키면 안 된다."""
    try:
        for line in build_cycle_report(result, context, elapsed_sec=elapsed_sec):
            print(line)
    except Exception as exc:
        print(f"[WARNING] 사이클 판정 요약 생성 실패(결과에는 영향 없음): {exc}")


__all__ = ["build_cycle_report", "print_cycle_report"]
