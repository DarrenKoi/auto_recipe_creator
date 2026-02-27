"""
채팅 메시지 MongoDB 핸들러
"""

from datetime import datetime, timezone

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

from server.config import MongoConfig, HistoryConfig
from server.models import ChatMessage, UserProfile


class ChatDBHandler:
    """채팅 메시지 MongoDB 핸들러."""

    def __init__(self, config: MongoConfig, history_config: HistoryConfig = None):
        if not MONGO_AVAILABLE:
            raise ImportError("pymongo가 설치되어 있지 않습니다. pip install pymongo")
        self.config = config
        self._history_config = history_config or HistoryConfig()
        self._client = None
        self._db = None
        self._collection = None
        self._profiles = None

    def initialize(self) -> None:
        """MongoDB 연결 및 인덱스 생성."""
        self._client = MongoClient(self.config.uri)
        self._db = self._client[self.config.database]
        self._collection = self._db[self.config.collection]
        self._profiles = self._db[self._history_config.profiles_collection]

        self._collection.create_index("message_id", unique=True)
        self._collection.create_index([("channel_id", ASCENDING), ("timestamp", ASCENDING)])
        self._profiles.create_index([("user", ASCENDING), ("channel_id", ASCENDING)], unique=True)
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

    def get_recent_history(self, channel_id: str, limit: int = 20) -> list[ChatMessage]:
        """최근 N개 메시지 조회 (timestamp 내림차순 → 오름차순 반환)."""
        cursor = self._collection.find(
            {"channel_id": channel_id}
        ).sort("timestamp", DESCENDING).limit(limit)
        messages = [ChatMessage.from_dict(doc) for doc in cursor]
        messages.reverse()
        return messages

    def get_old_messages(self, channel_id: str, before: datetime, summarized: bool = False) -> list[ChatMessage]:
        """특정 시점 이전의 요약되지 않은 메시지 조회."""
        query = {
            "channel_id": channel_id,
            "timestamp": {"$lt": before},
        }
        if not summarized:
            query["summarized"] = {"$ne": True}

        cursor = self._collection.find(query).sort("timestamp", ASCENDING)
        return [ChatMessage.from_dict(doc) for doc in cursor]

    def get_channels_with_old_messages(self, before: datetime) -> list[str]:
        """지정 시점 이전의 요약되지 않은 메시지가 있는 채널 목록."""
        return self._collection.distinct("channel_id", {
            "timestamp": {"$lt": before},
            "summarized": {"$ne": True},
        })

    def mark_summarized(self, message_ids: list[str]) -> int:
        """메시지들을 요약 완료로 마킹."""
        if not message_ids:
            return 0
        result = self._collection.update_many(
            {"message_id": {"$in": message_ids}},
            {"$set": {"summarized": True}},
        )
        return result.modified_count

    def upsert_user_profile(self, profile: UserProfile) -> None:
        """사용자 프로필 저장/갱신.

        기존 프로필이 있으면 요약을 병합하고 메시지 수를 누적.
        """
        existing = self.get_user_profile(profile.user, profile.channel_id)
        if existing:
            merged_summary = f"{existing.summary}\n\n---\n\n{profile.summary}"
            merged_count = existing.message_count + profile.message_count
            oldest = min(existing.oldest_message, profile.oldest_message)
            newest = max(existing.newest_message, profile.newest_message)
        else:
            merged_summary = profile.summary
            merged_count = profile.message_count
            oldest = profile.oldest_message
            newest = profile.newest_message

        doc = {
            "user": profile.user,
            "channel_id": profile.channel_id,
            "summary": merged_summary,
            "message_count": merged_count,
            "last_summarized": profile.last_summarized,
            "oldest_message": oldest,
            "newest_message": newest,
        }
        self._profiles.update_one(
            {"user": profile.user, "channel_id": profile.channel_id},
            {"$set": doc},
            upsert=True,
        )

    def get_user_profile(self, user: str, channel_id: str) -> UserProfile | None:
        """사용자 프로필 조회."""
        doc = self._profiles.find_one({"user": user, "channel_id": channel_id})
        if doc:
            return UserProfile.from_dict(doc)
        return None

    def close(self) -> None:
        """연결 종료."""
        if self._client:
            self._client.close()
            print("[INFO] MongoDB 연결 종료")
