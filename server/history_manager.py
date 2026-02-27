"""
대화 이력 관리자

LLM에 전달할 메시지 리스트를 구성: 시스템 프롬프트 + 사용자 프로필 요약 + 최근 N개 메시지.
"""

from server.config import AppConfig
from server.db_handler import ChatDBHandler
from server.token_counter import trim_messages_to_budget


def build_messages(channel_id: str, user: str, db: ChatDBHandler, config: AppConfig) -> list[dict]:
    """LLM에 전달할 OpenAI 형식 메시지 리스트 구성.

    구성 순서:
    1. 시스템 프롬프트
    2. 사용자 프로필 요약 (있으면 시스템 메시지로 주입)
    3. 최근 N개 대화 메시지
    4. 토큰 예산에 맞게 트리밍

    Args:
        channel_id: 채널 ID
        user: 사용자 이름
        db: DB 핸들러
        config: 앱 설정

    Returns:
        OpenAI 형식 메시지 리스트
    """
    openai_messages = []

    # 1. 시스템 프롬프트
    if config.llm.system_prompt:
        openai_messages.append({"role": "system", "content": config.llm.system_prompt})

    # 2. 사용자 프로필 요약 주입
    profile = db.get_user_profile(user, channel_id)
    if profile:
        profile_context = (
            f"[사용자 프로필 요약] {user}님의 이전 대화 요약 "
            f"(메시지 {profile.message_count}개 기반):\n{profile.summary}"
        )
        openai_messages.append({"role": "system", "content": profile_context})
        print(f"[INFO] 사용자 프로필 주입: user={user}, message_count={profile.message_count}")

    # 3. 최근 N개 메시지 로드
    recent = db.get_recent_history(channel_id, limit=config.history.max_history)
    for msg in recent:
        openai_messages.append({"role": msg.role, "content": msg.message})

    print(f"[INFO] 메시지 구성: system={len([m for m in openai_messages if m['role'] == 'system'])}개, "
          f"대화={len(recent)}개 (최대 {config.history.max_history}개)")

    # 4. 토큰 예산 트리밍
    openai_messages = trim_messages_to_budget(openai_messages, config.history.max_tokens_per_request)

    return openai_messages
