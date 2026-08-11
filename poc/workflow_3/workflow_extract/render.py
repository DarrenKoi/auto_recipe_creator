"""step 목록을 엔지니어가 읽을 한국어 절차서(markdown)로 만든다.

이 문서의 목적은 자동화가 아니라 판단 근거다 - 엔지니어에게 "이게 당신이 한
절차가 맞습니까"를 물을 수 있어야 한다. 그래서 한계를 푸터에 명시한다.
"""

_ACTION_LABEL = {
    "click": "클릭",
    "double_click": "더블클릭",
    "select_from_dropdown": "드롭다운 선택",
    "type_text": "값 입력",
    "click_repeat": "반복 클릭",
}

_LIMITATIONS = [
    "키 입력은 기록하지 않는다. 화면에 렌더된 값만 OCR 로 복원했다.",
    "Enter/Tab/단축키는 관측할 수 없다.",
    "드래그는 관측할 수 없다(버튼 눌림 상태를 폴링하지 않는다).",
    "스크롤/휠은 관측할 수 없다.",
    "더블클릭은 관측이 아니라 화면 변화로부터의 추론이다.",
    "라이브 영상 위 조작은 좌표가 아니라 내용에 의존하므로 재생하려면 CV 재해석이 필요하다.",
    "드롭다운 선택은 기하 추론이다(드롭다운이 실제로 열렸다는 증거는 없다). 세로로 쌓인 "
    "컨트롤 두 개를 순서대로 누른 것도 하나의 선택으로 잘못 보고될 수 있다.",
]


def _describe(step) -> str:
    """step 하나를 한 줄 한국어로 서술한다.

    `target`/`value` 는 "없음"과 "빈 문자열"을 구분해야 한다 - 빈 문자열은 값이
    입력됐다가 지워진 상태를 OCR 로 그대로 복원한 결과이지, 값이 없다는 뜻이
    아니다. 그래서 falsy 체크(`if value:`) 대신 `is not None` 을 쓴다.
    """
    action = _ACTION_LABEL.get(step["action"], step["action"])
    target = step.get("target")
    value = step.get("value")
    parts = []
    if target is not None:
        parts.append(f"**{target}**")
    elif step.get("target_kind") == "live_image":
        parts.append("**라이브 SEM 영상**")
    else:
        parts.append("**(라벨 없음)**")
    parts.append(action)
    if value is not None:
        parts.append(f"-> `{value}`")
    if step.get("count"):
        parts.append(f"({step['count']}회)")
    if step.get("inferred"):
        parts.append("_(추론)_")
    return " ".join(parts)


def _coverage_rows(steps) -> list:
    """규칙별 step 수를 센다 - 오작동 규칙을 지목할 수 있어야 한다."""
    counts = {}
    for step in steps:
        rule = step.get("grouping_rule") or "?"
        counts[rule] = counts.get(rule, 0) + 1
    return sorted(counts.items())


def render_markdown(steps, session) -> str:
    """step 목록 + 세션 정보로 절차서 markdown 문자열을 만든다."""
    duration_sec = session.get("duration_sec")
    duration_sec = float(duration_sec) if duration_sec is not None else 0.0
    lines = [
        f"# 수동 조작 절차 - {session.get('eqp_id', '?')} ({session.get('tag', '?')})",
        "",
        f"- 녹화 경로: `{session.get('capture_dir', '?')}`",
        f"- 세션 길이: {duration_sec:.1f}s",
        f"- 원본 이벤트 {session.get('total_events', 0)} 건 -> step {len(steps)} 건",
        "",
        "## 절차",
        "",
    ]
    if not steps:
        lines.append("_추출된 step 이 없습니다._")
    for step in steps:
        start, end = step["t_sec"]
        lines.append(f"{step['seq'] + 1}. [{start:.1f}s] {_describe(step)}")
    lines.extend(["", "## 규칙별 분포", "", "| 규칙 | step 수 |", "|------|---------|"])
    for rule, count in _coverage_rows(steps):
        lines.append(f"| {rule} | {count} |")
    lines.extend(["", "## 한계", ""])
    for item in _LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
