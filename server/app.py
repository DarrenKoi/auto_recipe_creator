"""
Flask 대화 서버

클라이언트 요청 → MongoDB 저장 → LangGraph/LLM 호출 (대화 이력 포함) → 응답 반환.
"""

from flask import Flask, request, jsonify

from server.config import AppConfig
from server.db_handler import ChatDBHandler
from server.history_manager import build_messages
from server.llm_client import send_chat

try:
    from server.graph import create_chat_graph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


def create_app(config: AppConfig = None) -> Flask:
    """Flask 앱 팩토리."""
    if config is None:
        config = AppConfig.load()

    app = Flask(__name__)

    db = ChatDBHandler(config.mongo, config.history)
    db.initialize()

    # LangGraph 그래프 생성 (실패 시 None → requests 폴백)
    graph = None
    if LANGGRAPH_AVAILABLE:
        try:
            graph = create_chat_graph(config.llm)
        except Exception as e:
            print(f"[WARNING] LangGraph 초기화 실패, requests 폴백 사용: {e}")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "langgraph": graph is not None})

    @app.route("/chat", methods=["POST"])
    def chat():
        body = request.get_json()
        if not body:
            return jsonify({"error": "요청 본문이 비어 있습니다"}), 400

        message_id = body.get("message_id")
        channel_id = body.get("channel_id")
        user = body.get("user")
        message = body.get("message")
        response_message_id = body.get("response_message_id")

        if not all([message_id, channel_id, user, message, response_message_id]):
            return jsonify({"error": "message_id, channel_id, user, message, response_message_id 필수"}), 400

        # 1. 사용자 메시지 저장
        db.save_message(message_id, channel_id, user, "user", message)

        # 2. 대화 이력 + 사용자 프로필 기반 메시지 구성 (윈도우 + 토큰 트리밍)
        openai_messages = build_messages(channel_id, user, db, config)

        # 3. LLM 호출 (LangGraph 또는 requests 폴백)
        try:
            llm_response = send_chat(openai_messages, config.llm, graph=graph, channel_id=channel_id)
        except Exception as e:
            print(f"[ERROR] LLM 호출 실패: {e}")
            return jsonify({"error": f"LLM 호출 실패: {e}"}), 502

        # 4. 어시스턴트 응답 저장
        assistant_msg = db.save_message(response_message_id, channel_id, "assistant", "assistant", llm_response)

        return jsonify(assistant_msg.to_dict(), default=str)

    @app.route("/history/<channel_id>", methods=["GET"])
    def history(channel_id):
        messages = db.get_channel_history(channel_id)
        return jsonify({
            "channel_id": channel_id,
            "messages": [m.to_dict() for m in messages],
        }, default=str)

    return app
