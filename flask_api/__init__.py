"""Flask API 패키지."""

from flask import Flask

from .router import api_blueprint

DEFAULT_URL_PREFIX = "/api"


def register_flask_api(app: Flask, url_prefix: str = DEFAULT_URL_PREFIX) -> None:
    """앱에 api blueprint를 등록한다."""
    app.register_blueprint(api_blueprint, url_prefix=url_prefix)


__all__ = ["api_blueprint", "register_flask_api", "DEFAULT_URL_PREFIX"]
