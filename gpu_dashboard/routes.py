"""GPU dashboard blueprint 템플릿 라우트 정의."""

from flask import Blueprint, jsonify


gpu_dashboard_dp = Blueprint("gpu_dashboard", __name__)


@gpu_dashboard_dp.route("/", methods=["GET"])
def home():
    """GPU dashboard 템플릿 기본 엔드포인트."""
    return jsonify(
        {
            "service": "gpu_dashboard",
            "message": "GPU dashboard template blueprint is running",
            "status": "ok",
            "hint": "Register this blueprint with url_prefix='/gpu-dashboard'.",
        }
    )


@gpu_dashboard_dp.route("/health", methods=["GET"])
def health():
    """GPU dashboard 헬스 체크 엔드포인트."""
    return jsonify(
        {
            "service": "gpu_dashboard",
            "status": "ok",
        }
    )
