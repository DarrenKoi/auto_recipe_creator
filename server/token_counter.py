"""
토큰 카운팅 유틸리티

토큰 추정, 메시지 청킹, 토큰 예산 트리밍 기능 제공.
"""

import re

from server.models import ChatMessage


def estimate_tokens(text: str) -> int:
    """텍스트의 대략적인 토큰 수 추정.

    영어: 약 4자당 1토큰, 한국어: 약 2자당 1토큰.
    한국어 비율을 감지하여 가중 평균 적용.
    """
    if not text:
        return 0

    korean_chars = len(re.findall(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]", text))
    total_chars = len(text)

    if total_chars == 0:
        return 0

    korean_ratio = korean_chars / total_chars
    chars_per_token = 4.0 * (1 - korean_ratio) + 2.0 * korean_ratio

    return max(1, int(total_chars / chars_per_token))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """OpenAI 형식 메시지 리스트의 총 토큰 수 추정.

    각 메시지에 role 오버헤드(4토큰) 추가.
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", "")) + 4
    return total


def chunk_messages(messages: list[ChatMessage], chunk_size: int) -> list[list[ChatMessage]]:
    """메시지를 지정 크기의 배치로 분할.

    Args:
        messages: 분할할 메시지 리스트
        chunk_size: 배치당 최대 메시지 수

    Returns:
        메시지 배치 리스트
    """
    if chunk_size <= 0:
        return [messages] if messages else []

    chunks = []
    for i in range(0, len(messages), chunk_size):
        chunks.append(messages[i:i + chunk_size])
    return chunks


def trim_messages_to_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    """토큰 예산에 맞게 메시지 트리밍.

    시스템 메시지(첫 번째)와 최근 메시지를 유지하면서
    중간의 오래된 메시지를 제거.

    Args:
        messages: OpenAI 형식 메시지 리스트
        max_tokens: 최대 토큰 예산

    Returns:
        트리밍된 메시지 리스트
    """
    if not messages:
        return messages

    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages

    # 시스템 메시지 분리
    system_msgs = []
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            non_system.append(msg)

    system_tokens = estimate_messages_tokens(system_msgs)
    remaining_budget = max_tokens - system_tokens

    if remaining_budget <= 0:
        return system_msgs

    # 최근 메시지부터 역순으로 추가
    trimmed = []
    current_tokens = 0
    for msg in reversed(non_system):
        msg_tokens = estimate_tokens(msg.get("content", "")) + 4
        if current_tokens + msg_tokens > remaining_budget:
            break
        trimmed.append(msg)
        current_tokens += msg_tokens

    trimmed.reverse()

    removed_count = len(non_system) - len(trimmed)
    if removed_count > 0:
        print(f"[INFO] 토큰 예산 초과로 {removed_count}개 메시지 트리밍 (예산={max_tokens}, 사용={system_tokens + current_tokens})")

    return system_msgs + trimmed
