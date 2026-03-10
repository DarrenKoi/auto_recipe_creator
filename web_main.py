"""Flask WSGI 앱 생성 엔트리포인트."""

from flask import Flask

from flask_api import register_flask_api
from gpu_dashboard import gpu_dashboard_dp


def create_app() -> Flask:
    """Blueprint들을 등록한 Flask 앱을 생성한다."""
    app = Flask(__name__)
    app.register_blueprint(gpu_dashboard_dp, url_prefix="/gpu-dashboard")
    register_flask_api(app)
    return app


app = create_app()
