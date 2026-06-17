"""점유 'select' 팝업 확인용 VLM 프롬프트 빌더.

다른 엔지니어가 tool 을 점유 중이면 RCS 가 작은 'select' 팝업을 띄운다 — 세 옵션
(제어 공유 요청 / 화면 공유 요청 / 기존 사용자 강제 종료 요청). 제목('select')으로 1차
검출한 창이 정말 이 점유 팝업인지(세 옵션이 보이는지) VLM 으로 확인해 오검출을 줄인다.

좌표를 요구하지 않는다 — 단순 yes/no 확인이다(역할 분담: VLM 은 식별만).
"""


def build_select_popup_prompt() -> tuple:
    """점유 'select' 팝업 확인 프롬프트 (system_message, user_message) 반환."""
    system_message = (
        "You analyse a small dialog window captured from a Windows CD-SEM Remote "
        "Monitoring application. Return strict JSON only. "
        "Decide whether this is the 'occupied tool' selection popup that appears when "
        "another engineer is already using the tool. That popup offers THREE choices, "
        "typically: request to SHARE CONTROL of the tool, request to SHARE the SCREEN, "
        "and request TERMINATION of the existing user's session. "
        "It is a small modal dialog, NOT the live SEM monitor window and NOT the main "
        "RCS list view. If you do not clearly see those share/terminate choices, it is "
        "NOT this popup."
    )
    user_message = (
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "is_select_popup": true,\n'
        '  "options_seen": ["share control", "share screen", "terminate existing user"],\n'
        '  "evidence": "short string naming the option texts you actually read"\n'
        "}\n"
        "Set is_select_popup=true ONLY if you can read option(s) about sharing control / "
        "sharing screen / terminating the existing user. "
        "options_seen lists the choice labels you actually read (use [] if none). "
        "If this is any other window, set is_select_popup=false and options_seen=[]."
    )
    return system_message, user_message


__all__ = ["build_select_popup_prompt"]
