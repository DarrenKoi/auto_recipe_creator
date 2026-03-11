"""Flask API package router definition."""

from flask import Blueprint, jsonify

from .vlm_serve import build_vlm_health_payload, register_vlm_serve_routes

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/health", methods=["GET"])
def health():
    """API root health endpoint."""
    return jsonify(
        {
            "service": "api",
            "status": "ok",
            "base_path": "/api",
            "vlm_serve": build_vlm_health_payload(),
        }
    )


register_vlm_serve_routes(api_blueprint)

__all__ = ["api_blueprint"]
