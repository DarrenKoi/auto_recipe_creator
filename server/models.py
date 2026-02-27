"""
채팅 메시지 및 사용자 프로필 모델
"""

from dataclasses import dataclass
from datetime import datetime, timezone


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


@dataclass
class UserProfile:
    """사용자 대화 요약 프로필."""
    user: str                    # 사용자 이름
    channel_id: str              # 채널 ID
    summary: str                 # 요약된 대화 프로필
    message_count: int           # 요약에 포함된 메시지 수
    last_summarized: datetime    # 마지막 요약 시각
    oldest_message: datetime     # 요약된 가장 오래된 메시지 시각
    newest_message: datetime     # 요약된 가장 최근 메시지 시각

    def to_dict(self) -> dict:
        """MongoDB 저장용 딕셔너리 변환."""
        return {
            "user": self.user,
            "channel_id": self.channel_id,
            "summary": self.summary,
            "message_count": self.message_count,
            "last_summarized": self.last_summarized,
            "oldest_message": self.oldest_message,
            "newest_message": self.newest_message,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "UserProfile":
        """MongoDB 문서에서 복원."""
        return cls(
            user=doc["user"],
            channel_id=doc["channel_id"],
            summary=doc["summary"],
            message_count=doc["message_count"],
            last_summarized=doc["last_summarized"],
            oldest_message=doc["oldest_message"],
            newest_message=doc["newest_message"],
        )
