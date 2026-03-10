"""VLM 서비스 router."""

from flask import Blueprint, jsonify

from .mai_ui import SERVICE_CONFIG as MAI_UI_CONFIG
from .mai_ui import service_blueprint as mai_ui_blueprint
from .ui_tars import SERVICE_CONFIG as UI_TARS_CONFIG
from .ui_tars import service_blueprint as ui_tars_blueprint
from .ui_venus import SERVICE_CONFIG as UI_VENUS_CONFIG
from .ui_venus import service_blueprint as ui_venus_blueprint

VLM_SERVICE_BLUEPRINTS = [
    (UI_VENUS_CONFIG, ui_venus_blueprint),
    (MAI_UI_CONFIG, mai_ui_blueprint),
    (UI_TARS_CONFIG, ui_tars_blueprint),
]

vlm_serve_blueprint = Blueprint("vlm_serve", __name__)


@vlm_serve_blueprint.route("/", methods=["GET"])
def home():
    """API root 안내 엔드포인트."""
    return jsonify(
        {
            "service": "vlm_serve",
            "status": "ok",
            "mode": "proxy",
            "message": "VLM service proxy routes are registered.",
            "base_path": "/api/vlm_serve",
            "vlm_services": [
                service_config.to_dict()
                for service_config, _ in VLM_SERVICE_BLUEPRINTS
            ],
        }
    )


@vlm_serve_blueprint.route("/health", methods=["GET"])
def health():
    """API root 헬스 체크 엔드포인트."""
    return jsonify(
        {
            "service": "vlm_serve",
            "status": "ok",
            "mode": "proxy",
            "base_path": "/api/vlm_serve",
            "registered_vlms": [
                service_config.route_slug
                for service_config, _ in VLM_SERVICE_BLUEPRINTS
            ],
        }
    )


def register_vlm_serve_routes(api_blueprint: Blueprint) -> None:
    """API blueprint 에 VLM 서비스 router 를 등록한다."""
    api_blueprint.register_blueprint(vlm_serve_blueprint, url_prefix="/vlm_serve")


for service_config, service_blueprint in VLM_SERVICE_BLUEPRINTS:
    vlm_serve_blueprint.register_blueprint(
        service_blueprint,
        url_prefix=f"/{service_config.route_slug}",
    )


__all__ = ["register_vlm_serve_routes", "vlm_serve_blueprint", "VLM_SERVICE_BLUEPRINTS"]
