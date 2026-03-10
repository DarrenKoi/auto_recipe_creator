"""Flask blueprint 라우트 정의."""

from flask import Blueprint, jsonify


api_blueprint = Blueprint("flask_api", __name__)


@api_blueprint.route("/", methods=["GET"])
def home():
    """기본 상태 확인 엔드포인트."""
    return jsonify(
        {
            "message": "Flask API server is running",
            "status": "ok",
        }
    )


@api_blueprint.route("/health", methods=["GET"])
def health():
    """헬스 체크 엔드포인트."""
    return jsonify({"status": "ok"})
