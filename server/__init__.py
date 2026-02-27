"""
LLM 대화 서버 모듈

Flask 기반 LLM 대화 중개 서버 — MongoDB 대화 이력 관리, LangGraph 기반 LLM 호출.
"""

from .models import ChatMessage, UserProfile
from .config import AppConfig, MongoConfig, LLMConfig, ServerConfig, HistoryConfig
from .db_handler import ChatDBHandler
from .llm_client import send_chat
from .history_manager import build_messages
from .token_counter import estimate_tokens, estimate_messages_tokens, chunk_messages, trim_messages_to_budget
from .summarizer import summarize_old_conversations

try:
    from .graph import create_chat_graph, ChatState
    _GRAPH_EXPORTS = ["create_chat_graph", "ChatState"]
except ImportError:
    _GRAPH_EXPORTS = []

__all__ = [
    "ChatMessage",
    "UserProfile",
    "AppConfig",
    "MongoConfig",
    "LLMConfig",
    "ServerConfig",
    "HistoryConfig",
    "ChatDBHandler",
    "send_chat",
    "build_messages",
    "estimate_tokens",
    "estimate_messages_tokens",
    "chunk_messages",
    "trim_messages_to_budget",
    "summarize_old_conversations",
    *_GRAPH_EXPORTS,
]
