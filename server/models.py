"""
채팅 메시지 모델
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChatMessage:
    """채팅 메시지."""
    message_id: str          # 고유 키 (외부 시스템 제공)
    channel_id: str          # 대화 채널 ID
    user: str                # 사용자 이름 또는 "assistant"
    role: str                # "user" | "assistant" | "system"
    message: str             # 메시지 내용
    timestamp: datetime      # 생성 시각 (UTC)

    def to_dict(self) -> dict:
        """MongoDB 저장용 딕셔너리 변환."""
        return {
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "user": self.user,
            "role": self.role,
            "message": self.message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "ChatMessage":
        """MongoDB 문서에서 복원."""
        return cls(
            message_id=doc["message_id"],
            channel_id=doc["channel_id"],
            user=doc["user"],
            role=doc["role"],
            message=doc["message"],
            timestamp=doc["timestamp"],
        )
