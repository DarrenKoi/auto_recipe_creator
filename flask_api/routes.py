"""Flask blueprint 라우트 정의."""

from flask import Blueprint

from .vlm_serve import register_vlm_serve_routes

api_blueprint = Blueprint("api", __name__)
register_vlm_serve_routes(api_blueprint)
