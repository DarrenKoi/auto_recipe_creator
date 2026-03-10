"""Flask blueprint 라우트 정의."""

from flask import Blueprint

from .topics import register_topic_routes

api_blueprint = Blueprint("api", __name__)
register_topic_routes(api_blueprint)
