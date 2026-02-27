"""
채팅 서버 실행

Usage:
    uv run python server/run.py
"""

from server.config import AppConfig
from server.app import create_app


config = AppConfig.load()
app = create_app(config)

if __name__ == "__main__":
    print(f"[INFO] 서버 시작: {config.server.host}:{config.server.port}")
    app.run(host=config.server.host, port=config.server.port, debug=config.server.debug)
