"""
Flask 대화 서버

클라이언트 요청 → MongoDB 저장 → LLM 호출 (대화 이력 포함) → 응답 반환.
"""

from flask import Flask, request, jsonify

from server.config import AppConfig
from server.db_handler import ChatDBHandler
from server.llm_client import send_chat


def create_app(config: AppConfig = None) -> Flask:
    """Flask 앱 팩토리."""
    if config is None:
        config = AppConfig.load()

    app = Flask(__name__)

    db = ChatDBHandler(config.mongo)
    db.initialize()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

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

        # 2. 채널 대화 이력 조회
        history = db.get_channel_history(channel_id)

        # 3. OpenAI 형식 메시지 리스트 구성
        openai_messages = []
        if config.llm.system_prompt:
            openai_messages.append({"role": "system", "content": config.llm.system_prompt})
        for msg in history:
            openai_messages.append({"role": msg.role, "content": msg.message})

        # 4. LLM 호출
        try:
            llm_response = send_chat(openai_messages, config.llm)
        except Exception as e:
            print(f"[ERROR] LLM 호출 실패: {e}")
            return jsonify({"error": f"LLM 호출 실패: {e}"}), 502

        # 5. 어시스턴트 응답 저장
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
