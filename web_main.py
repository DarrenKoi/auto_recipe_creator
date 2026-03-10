"""Flask WSGI 앱 생성 엔트리포인트."""

from flask import Flask

from flask_api import api_blueprint


def create_app() -> Flask:
    """Blueprint를 등록한 Flask 앱을 생성한다."""
    app = Flask(__name__)
    app.register_blueprint(api_blueprint)
    return app


app = create_app()

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    print("[INFO] Flask 서버 시작: 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
