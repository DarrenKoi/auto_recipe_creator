"""WSGI 서버용 엔트리포인트."""

from web_main import app as application


if __name__ == "__main__":
    application.run(debug=True, host="0.0.0.0")
