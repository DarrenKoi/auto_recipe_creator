"""
채팅 메시지 MongoDB 핸들러
"""

from datetime import datetime, timezone

try:
    from pymongo import MongoClient, ASCENDING
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

from server.config import MongoConfig
from server.models import ChatMessage


class ChatDBHandler:
    """채팅 메시지 MongoDB 핸들러."""

    def __init__(self, config: MongoConfig):
        if not MONGO_AVAILABLE:
            raise ImportError("pymongo가 설치되어 있지 않습니다. pip install pymongo")
        self.config = config
        self._client = None
        self._db = None
        self._collection = None

    def initialize(self) -> None:
        """MongoDB 연결 및 인덱스 생성."""
        self._client = MongoClient(self.config.uri)
        self._db = self._client[self.config.database]
        self._collection = self._db[self.config.collection]

        self._collection.create_index("message_id", unique=True)
        self._collection.create_index([("channel_id", ASCENDING), ("timestamp", ASCENDING)])
        print(f"[INFO] MongoDB 연결 완료: {self.config.database}.{self.config.collection}")

    def save_message(self, message_id: str, channel_id: str, user: str, role: str, message: str) -> ChatMessage:
        """메시지 저장 후 ChatMessage 반환."""
        msg = ChatMessage(
            message_id=message_id,
            channel_id=channel_id,
            user=user,
            role=role,
            message=message,
            timestamp=datetime.now(timezone.utc),
        )
        self._collection.insert_one(msg.to_dict())
        print(f"[INFO] 메시지 저장: message_id={message_id}, channel_id={channel_id}, role={role}")
        return msg

    def get_channel_history(self, channel_id: str) -> list[ChatMessage]:
        """채널의 전체 대화 이력 조회 (timestamp 오름차순)."""
        cursor = self._collection.find(
            {"channel_id": channel_id}
        ).sort("timestamp", ASCENDING)
        return [ChatMessage.from_dict(doc) for doc in cursor]

    def get_message(self, message_id: str) -> ChatMessage | None:
        """단일 메시지 조회."""
        doc = self._collection.find_one({"message_id": message_id})
        if doc:
            return ChatMessage.from_dict(doc)
        return None

    def close(self) -> None:
        """연결 종료."""
        if self._client:
            self._client.close()
            print("[INFO] MongoDB 연결 종료")
