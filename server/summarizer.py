"""
대화 요약기

오래된 대화를 요약하여 사용자 프로필로 저장하는 스케줄러.
단독 실행: uv run python -m server.summarizer
"""

import time
from datetime import datetime, timezone, timedelta

from server.config import AppConfig, LLMConfig, HistoryConfig
from server.db_handler import ChatDBHandler
from server.llm_client import send_chat
from server.models import ChatMessage, UserProfile
from server.token_counter import chunk_messages


def summarize_old_conversations(db: ChatDBHandler, llm_config: LLMConfig, history_config: HistoryConfig) -> int:
    """오래된 대화를 요약하여 사용자 프로필 저장.

    Args:
        db: DB 핸들러
        llm_config: LLM 설정
        history_config: 이력 관리 설정

    Returns:
        요약된 총 메시지 수
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=history_config.summary_age_days)
    total_summarized = 0

    # 요약 대상 채널 조회
    channels = db.get_channels_with_old_messages(cutoff)
    if not channels:
        print("[INFO] 요약 대상 대화 없음")
        return 0

    print(f"[INFO] 요약 대상 채널 {len(channels)}개 발견 (기준: {history_config.summary_age_days}일 이상)")

    for channel_id in channels:
        old_messages = db.get_old_messages(channel_id, before=cutoff, summarized=False)
        if not old_messages:
            continue

        # 사용자별 그룹핑
        users_messages: dict[str, list[ChatMessage]] = {}
        for msg in old_messages:
            users_messages.setdefault(msg.user, []).append(msg)
            # assistant 메시지도 해당 채널의 모든 사용자에 포함
            if msg.role == "assistant":
                for u in users_messages:
                    if u != msg.user and msg not in users_messages[u]:
                        users_messages[u].append(msg)

        for user, msgs in users_messages.items():
            if user == "assistant":
                continue

            chunks = chunk_messages(msgs, history_config.summary_chunk_size)

            for chunk in chunks:
                summary = _summarize_chunk(chunk, llm_config)
                if not summary:
                    continue

                profile = UserProfile(
                    user=user,
                    channel_id=channel_id,
                    summary=summary,
                    message_count=len(chunk),
                    last_summarized=datetime.now(timezone.utc),
                    oldest_message=chunk[0].timestamp,
                    newest_message=chunk[-1].timestamp,
                )
                db.upsert_user_profile(profile)

                message_ids = [m.message_id for m in chunk]
                db.mark_summarized(message_ids)
                total_summarized += len(chunk)

                print(f"[INFO] 요약 완료: channel={channel_id}, user={user}, messages={len(chunk)}개")

    print(f"[INFO] 전체 요약 완료: 총 {total_summarized}개 메시지 처리")
    return total_summarized


def _summarize_chunk(messages: list[ChatMessage], llm_config: LLMConfig) -> str:
    """메시지 청크를 요약하여 텍스트 반환."""
    prompt_messages = _build_summary_prompt(messages)

    try:
        summary = send_chat(prompt_messages, llm_config)
        return summary.strip()
    except Exception as e:
        print(f"[ERROR] 요약 LLM 호출 실패: {e}")
        return ""


def _build_summary_prompt(messages: list[ChatMessage]) -> list[dict]:
    """요약용 프롬프트 구성."""
    system = (
        "다음 대화를 요약하여 사용자의 주요 관심사, 선호도, 핵심 정보를 추출하세요. "
        "간결하고 구조적인 요약을 작성하되, 향후 대화에서 참고할 수 있는 정보 위주로 정리하세요. "
        "한국어로 작성하세요."
    )

    conversation_text = []
    for msg in messages:
        timestamp_str = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        conversation_text.append(f"[{timestamp_str}] {msg.user} ({msg.role}): {msg.message}")

    user_content = "다음 대화 내용을 요약해 주세요:\n\n" + "\n".join(conversation_text)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def run_scheduler(db: ChatDBHandler, llm_config: LLMConfig, history_config: HistoryConfig,
                  interval_hours: int = 24) -> None:
    """주기적으로 요약 실행하는 스케줄러.

    Args:
        db: DB 핸들러
        llm_config: LLM 설정
        history_config: 이력 관리 설정
        interval_hours: 실행 간격 (시간)
    """
    interval_sec = interval_hours * 3600
    print(f"[INFO] 요약 스케줄러 시작 (간격={interval_hours}시간)")

    while True:
        try:
            summarize_old_conversations(db, llm_config, history_config)
        except Exception as e:
            print(f"[ERROR] 요약 스케줄러 오류: {e}")

        print(f"[INFO] 다음 요약까지 {interval_hours}시간 대기")
        time.sleep(interval_sec)


if __name__ == "__main__":
    config = AppConfig.load()
    db = ChatDBHandler(config.mongo)
    db.initialize()

    try:
        count = summarize_old_conversations(db, config.llm, config.history)
        print(f"[INFO] 수동 요약 완료: {count}개 메시지 처리")
    finally:
        db.close()
