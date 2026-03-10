"""system topic 상태 확인 라우트."""

from flask import Blueprint, jsonify


def register_routes(api_blueprint: Blueprint) -> None:
    """system topic 기본 라우트를 등록한다."""

    @api_blueprint.route("/", methods=["GET"])
    def home():
        """기본 상태 확인 엔드포인트."""
        return jsonify(
            {
                "service": "api",
                "message": "API blueprint is running",
                "status": "ok",
                "hint": "Register this blueprint with url_prefix='/api'.",
            }
        )

    @api_blueprint.route("/health", methods=["GET"])
    def health():
        """헬스 체크 엔드포인트."""
        return jsonify(
            {
                "service": "api",
                "status": "ok",
            }
        )

    @api_blueprint.route("/example", methods=["GET"])
    def example():
        """추가 API를 붙일 때 사용할 예제 엔드포인트."""
        return jsonify(
            {
                "service": "api",
                "status": "ok",
                "message": "Add your own endpoints under the api blueprint.",
            }
        )
